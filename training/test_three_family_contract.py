from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import copy

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from evaluation.three_family_guard import create_evidence, verify_evidence
from training.three_family_operator import command_plan

from release.source_policy_drift import validate as validate_policy_drift
from unittest.mock import patch

from training import three_family_contract as contract


class ThreeFamilyContractTests(unittest.TestCase):
    def test_operator_has_exact_dry_run_sequence(self):
        plan = command_plan()
        self.assertEqual([step["id"] for step in plan], list(range(1, 19)))
        self.assertTrue(all(step["mode"] != "execute" for step in plan))
        self.assertEqual(plan[15]["mode"], "print_only_destructive")

    def test_generated_answer_is_cryptographically_bound(self):
        digest = "a" * 64
        evidence = create_evidence(
            family="kova-cosmo", base_manifest_sha256=digest, adapter_sha256="b" * 64,
            runner_sha256="c" * 64, attempt_id="attempt-1", prompt="Who are you?",
            answer="I am Kova.", dimensions=["identity"],
            private_key=Ed25519PrivateKey.generate(), created_at="2026-09-21T00:00:00+00:00",
        )
        verified = verify_evidence(
            evidence, expected_family="kova-cosmo", expected_base_manifest_sha256=digest,
            expected_adapter_sha256="b" * 64, expected_runner_sha256="c" * 64,
        )
        self.assertEqual(verified["answer"], "I am Kova.")
        tampered = copy.deepcopy(evidence)
        tampered["payload"]["answer"] = "altered"
        with self.assertRaisesRegex(ValueError, "invalid_evidence_signature"):
            verify_evidence(
                tampered, expected_family="kova-cosmo", expected_base_manifest_sha256=digest,
                expected_adapter_sha256="b" * 64, expected_runner_sha256="c" * 64,
            )

    def test_authoritative_policy_cannot_drift(self):
        self.assertEqual(
            validate_policy_drift()["status"],
            "authoritative_three_family_policy_no_drift",
        )

    def test_complete_source_contract_is_valid_dataset_approved_execution_blocked(self):
        report = contract.validate()
        self.assertEqual(report["families"], list(contract.FAMILIES))
        self.assertEqual(report["chat_routes"], {"free": 1, "plus": 6, "pro": 12})
        self.assertEqual(report["work_routes"], {"free": 0, "plus": 18, "pro": 18})
        self.assertTrue(report["dataset"]["approval_complete"])
        self.assertTrue(report["all_paid_and_production_gates_closed"])
        self.assertEqual(report["provider_calls_made"], 0)

    def test_policy_has_no_nova_chat_or_separate_8b_slot(self):
        value = contract.load_json(contract.ROOT / "config/current-product-policy.v3.json")
        self.assertEqual(value["entitlements"]["chat"]["plus"]["nova"], [])
        self.assertEqual(value["entitlements"]["chat"]["pro"]["nova"], [])
        raw = json.dumps(value)
        self.assertNotIn("Qwen3-8B", raw)
        self.assertFalse(value["processing_levels_are_separate_models"])

    def test_all_active_authorization_gates_are_false(self):
        for name in (
            "current-product-policy.v3.json", "kova-private-lineage.v1.json",
            "kova-three-family-dataset.v2.json", "kova-three-family-pilot.v1.json",
            "kova-three-family-cost-guard.v1.json", "kova-three-family-lifecycle.v1.json",
            "kova-three-family-evaluation.v1.json", "kova-three-family-operator-plan.v1.json",
        ):
            with self.subTest(name=name):
                contract._all_false(contract.load_json(contract.ROOT / "config" / name), name)

    def test_invalid_utf8_and_duplicate_json_keys_fail_closed(self):
        for raw in (b'\xff', b'{"x":1,"x":2}', b'{"x":NaN}', b''):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "bad.json"
                path.write_bytes(raw)
                with self.assertRaises(contract.ContractError):
                    contract.load_json(path)

    def test_manifest_inventory_is_immutable_and_complete(self):
        totals = contract.validate_lineage_and_manifests()
        self.assertEqual(totals["kova-cosmo"], 1519207673)
        self.assertEqual(totals["kova-orion"], 4079448540)
        self.assertEqual(totals["kova-nova"], 8060925056)

    def test_snapshot_rejects_missing_extra_and_modified_files(self):
        manifest = {"files": [{"path": "config.json", "bytes": 2,
                                "sha256": hashlib.sha256(b"{}").hexdigest()}]}
        lineage = {"families": {family: {"manifest": "ignored"} for family in contract.FAMILIES}}
        def fake_load(path, **_):
            return lineage if path.name == "kova-private-lineage.v1.json" else manifest
        with tempfile.TemporaryDirectory() as folder, patch.object(contract, "load_json", fake_load):
            root = Path(folder)
            (root / "config.json").write_bytes(b"{}")
            contract.verify_snapshot("kova-cosmo", root)
            (root / "extra").write_bytes(b"x")
            with self.assertRaises(contract.ContractError):
                contract.verify_snapshot("kova-cosmo", root)
            (root / "extra").unlink()
            (root / "config.json").write_bytes(b"[]")
            with self.assertRaises(contract.ContractError):
                contract.verify_snapshot("kova-cosmo", root)
            (root / "config.json").unlink()
            with self.assertRaises(contract.ContractError):
                contract.verify_snapshot("kova-cosmo", root)

    def test_six_dollar_bootstrap_uses_live_rate_and_60_second_rounding(self):
        self.assertLessEqual(contract.admit_bootstrap(Decimal("0.526")), Decimal("6.0000"))
        with self.assertRaisesRegex(contract.ContractError, "six-dollar"):
            contract.admit_bootstrap(Decimal("1.00"))
        with self.assertRaises(contract.ContractError):
            contract.admit_bootstrap(Decimal("-0.01"))

    def test_ledger_assigns_sequence_and_rejects_stale_duplicate_out_of_order(self):
        state = {"sequence": 0, "terminal": False, "family_order": [], "events": []}
        state = contract.append_ledger_event(state, {"kind": "watchdog_health"}, expected_sequence=0)
        self.assertEqual(state["events"][-1]["sequence"], 1)
        with self.assertRaises(contract.ContractError):
            contract.append_ledger_event(state, {"kind": "training_grant", "family": "kova-orion"},
                                         expected_sequence=1)
        state = contract.append_ledger_event(state, {"kind": "training_grant", "family": "kova-cosmo"},
                                             expected_sequence=1)
        with self.assertRaises(contract.ContractError):
            contract.append_ledger_event(state, {"kind": "training_grant", "family": "kova-cosmo"},
                                         expected_sequence=2)
        with self.assertRaises(contract.ContractError):
            contract.append_ledger_event(state, {"kind": "cost_admission"}, expected_sequence=1)
        with self.assertRaises(contract.ContractError):
            contract.append_ledger_event(state, {"kind": "cost_admission", "sequence": 3},
                                         expected_sequence=2)

    def test_terminal_ledger_rejects_post_cleanup_events(self):
        state = {"sequence": 0, "terminal": False, "family_order": [], "events": []}
        state = contract.append_ledger_event(state, {"kind": "cleanup_terminal"}, expected_sequence=0)
        self.assertTrue(state["terminal"])
        with self.assertRaises(contract.ContractError):
            contract.append_ledger_event(state, {"kind": "watchdog_health"}, expected_sequence=1)

    def test_infrastructure_has_no_public_ip_and_source_gates_false(self):
        vm = (contract.ROOT / "infra/three-family-pilot-vm.bicep").read_text()
        watchdog = (contract.ROOT / "infra/three-family-watchdog.bicep").read_text()
        self.assertNotIn("publicIPAddresses", vm)
        self.assertIn("Standard_NC4as_T4_v3", vm)
        self.assertIn("disablePasswordAuthentication: true", vm)
        self.assertIn("frequency: 'Minute'", watchdog)
        self.assertIn("delete_pilot_group", watchdog)
        self.assertIn("allowSharedKeyAccess: false", watchdog)

    def test_tampering_with_a_safety_gate_is_rejected(self):
        value = contract.load_json(contract.ROOT / "config/kova-three-family-pilot.v1.json")
        changed = deepcopy(value)
        changed["spending_authorized"] = True
        with self.assertRaisesRegex(contract.ContractError, "open safety gate"):
            contract._all_false(changed, "pilot")


if __name__ == "__main__":
    unittest.main()
