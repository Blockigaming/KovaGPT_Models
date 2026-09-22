import base64
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import copy

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from evaluation import three_family_guard as guard
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
        self.assertIn("--retail-price-evidence", plan[0]["verify_argv"])
        self.assertIn("provisionPilot=true", plan[2]["argv"])
        self.assertIn("provisionWatchdog=true", plan[3]["argv"])
        self.assertIn("provisionPilot=true", plan[4]["argv"])
        self.assertIn("--all-families", plan[6]["argv"])
        cleanup = json.dumps(plan[15:17])
        self.assertIn("${PILOT_RESOURCE_GROUP}", cleanup)
        self.assertIn("${WATCHDOG_RESOURCE_GROUP}", cleanup)

    def test_generated_answer_is_bound_to_pinned_key_and_runtime_context(self):
        digest = "a" * 64
        private_key = Ed25519PrivateKey.generate()
        trusted_public = base64.b64encode(private_key.public_key().public_bytes_raw()).decode("ascii")
        bindings = {
            "expected_source_commit": "d" * 40,
            "expected_family": "kova-cosmo",
            "expected_base_revision": "e" * 40,
            "expected_base_manifest_sha256": digest,
            "expected_adapter_sha256": "b" * 64,
            "expected_runner_sha256": "c" * 64,
            "expected_case_id": "case-1",
            "expected_runtime_profile": "high",
            "expected_conversation_id": "conversation-1",
            "expected_session_id": "session-1",
        }
        evidence = guard.create_evidence(
            source_commit=bindings["expected_source_commit"], family="kova-cosmo",
            base_revision=bindings["expected_base_revision"], base_manifest_sha256=digest,
            adapter_sha256="b" * 64, runner_sha256="c" * 64, case_id="case-1",
            prompt="Who are you?", answer="I am Kova.", runtime_profile="high",
            conversation_id="conversation-1", session_id="session-1",
            dimensions=["identity"], private_key=private_key,
            created_at="2026-09-21T00:00:00+00:00",
        )
        self.assertNotIn("public_key_ed25519_b64", evidence)
        with patch.object(guard, "PINNED_RUNNER_PUBLIC_KEY_B64", trusted_public):
            verified = guard.verify_evidence(evidence, **bindings)
            self.assertEqual(verified["answer"], "I am Kova.")
            with self.assertRaisesRegex(ValueError, "evidence_binding_mismatch:session_id"):
                guard.verify_evidence(evidence, **{**bindings, "expected_session_id": "session-2"})
            tampered = copy.deepcopy(evidence)
            tampered["payload"]["answer"] = "altered"
            tampered["payload"]["answer_sha256"] = hashlib.sha256(b"altered").hexdigest()
            with self.assertRaisesRegex(ValueError, "invalid_evidence_signature"):
                guard.verify_evidence(tampered, **bindings)

    def test_unconfigured_runner_key_fails_closed(self):
        private_key = Ed25519PrivateKey.generate()
        evidence = guard.create_evidence(
            source_commit="d" * 40, family="kova-cosmo", base_revision="e" * 40,
            base_manifest_sha256="a" * 64, adapter_sha256="b" * 64,
            runner_sha256="c" * 64, case_id="case-1", prompt="p", answer="a",
            runtime_profile="light", conversation_id="conversation-1", session_id="session-1",
            dimensions=["identity"], private_key=private_key,
        )
        with patch.object(guard, "PINNED_RUNNER_PUBLIC_KEY_B64", None), self.assertRaisesRegex(
                ValueError, "trusted_runner_key_not_configured"):
            guard.verify_evidence(
                evidence, expected_source_commit="d" * 40, expected_family="kova-cosmo",
                expected_base_revision="e" * 40, expected_base_manifest_sha256="a" * 64,
                expected_adapter_sha256="b" * 64, expected_runner_sha256="c" * 64,
                expected_case_id="case-1", expected_runtime_profile="light",
                expected_conversation_id="conversation-1", expected_session_id="session-1",
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

    def test_captured_live_price_is_parsed_and_admitted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "price.json"
            path.write_text(json.dumps({"Items": [{
                "armSkuName": "Standard_NC4as_T4_v3", "armRegionName": "eastus",
                "currencyCode": "USD", "unitOfMeasure": "1 Hour", "retailPrice": 0.526,
            }]}))
            self.assertEqual(contract.validate_live_price_evidence(path)["status"], "live_price_admitted")
            value = json.loads(path.read_text())
            value["Items"][0]["retailPrice"] = 1.0
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(contract.ContractError, "six-dollar"):
                contract.validate_live_price_evidence(path)

    def test_t4_probe_evidence_is_actually_consumed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "probe.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "device_name": "NVIDIA T4",
                "compute_capability": "7.5",
                "cuda_version": "12.8",
                "bitsandbytes_four_bit_available": True,
                "available_vram_bytes": 16000000000,
                "free_disk_bytes": 40000000000,
                "family_probes": {
                    "kova-cosmo": {"peak_vram_bytes": 6000000000, "maximum_sequence_length": 1024, "probe_passed": True},
                    "kova-orion": {"peak_vram_bytes": 9000000000, "maximum_sequence_length": 1024, "probe_passed": True},
                    "kova-nova": {"peak_vram_bytes": 15000000000, "maximum_sequence_length": 768, "probe_passed": True},
                },
            }))
            self.assertEqual(contract.validate_probe_evidence(path)["status"], "t4_probe_evidence_valid")
            value = json.loads(path.read_text())
            value["device_name"] = "not-a-t4"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(contract.ContractError, "unexpected GPU"):
                contract.validate_probe_evidence(path)

    def test_ledger_assigns_sequence_and_rejects_stale_duplicate_out_of_order(self):
        state = {"sequence": 0, "terminal": False, "family_order": [], "events": []}
        state = contract.append_ledger_event(state, {"kind": "watchdog_health"}, expected_sequence=0)
        self.assertEqual(state["events"][-1]["sequence"], 1)
        with self.assertRaises(contract.ContractError):
            contract.append_ledger_event(state, {"kind": "training_grant", "family": "kova-orion"},
                                         expected_sequence=1)
        before_grant = deepcopy(state)
        state = contract.append_ledger_event(state, {"kind": "training_grant", "family": "kova-cosmo"},
                                             expected_sequence=1)
        self.assertEqual(before_grant["family_order"], [])
        self.assertEqual(state["family_order"], ["kova-cosmo"])
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
        self.assertIn("Microsoft.HpcCompute", vm)
        self.assertIn("NvidiaGpuDriverLinux", vm)
        self.assertIn("enableAutomaticUpgrade: false", vm)
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
