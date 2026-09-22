"""Adversarial tests for signed Cosmo runtime and cleanup evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from training import cosmo_runtime_guard as guard
from training.cosmo_generation_attestation import public_key_hex
from training.cosmo_lifecycle_authority import canonical


NOW = datetime(2026, 9, 19, 19, 0, tzinfo=timezone.utc)


class CosmoRuntimeGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "repo"
        (self.root / "config").mkdir(parents=True)

        policy = json.loads(guard.POLICY_PATH.read_text(encoding="utf-8"))
        policy["status"] = "operator_preflight_required"
        policy["owner_approvals"]["resource_creation_release"] = True
        policy["owner_approvals"]["spending_release"] = True
        (self.root / "config/kova-cosmo-runtime-guard.v1.json").write_text(
            json.dumps(policy, indent=2) + "\n", encoding="utf-8"
        )

        self.authority_key = Ed25519PrivateKey.from_private_bytes(b"a" * 32)
        public = public_key_hex(self.authority_key)
        lifecycle_trust = {
            "schema_version": 1,
            "status": "authority_pinned",
            "issuer": "kova-cosmo-lifecycle-authority-v1",
            "algorithm": "ed25519",
            "endpoint": "https://authority.example/v1/pilot/grants",
            "public_key_hex": public,
            "public_key_sha256": hashlib.sha256(bytes.fromhex(public)).hexdigest(),
            "bearer_token_file_environment_variable":
                "KOVA_COSMO_LIFECYCLE_TOKEN_FILE",
            "azure_managed_identity_token_audience":
                "api://kova-cosmo-lifecycle-authority",
            "append_only_remote_ledger_required": True,
            "independent_azure_reader_required": True,
            "runner_ledger_mutation_allowed": False,
            "checked_in_private_key_allowed": False,
        }
        (self.root / "config/kova-cosmo-lifecycle-trust.v1.json").write_text(
            json.dumps(lifecycle_trust, indent=2) + "\n", encoding="utf-8"
        )

    def preflight_payload(self) -> dict:
        subscription = "11111111-2222-3333-4444-555555555555"
        group = "kova-cosmo-pilot"
        return {
            "schema_version": 1,
            "kind": "kova_cosmo_runtime_preflight",
            "issuer": "kova-cosmo-lifecycle-authority-v1",
            "pilot_id": guard.PILOT_ID,
            "lifecycle_id": "lifecycle-001",
            "ledger_sequence": 1,
            "provider_observation_id": "azure-observation-preflight-001",
            "azure_query_source": "independent_azure_control_plane_reader",
            "captured_at_utc": "2026-09-19T18:55:00Z",
            "subscription_id": subscription,
            "resource_group": group,
            "cleanup_scope_resource_group_id":
                f"/subscriptions/{subscription}/resourceGroups/{group}",
            "resource_group_exclusive_to_pilot": True,
            "vm_name": "kova-cosmo-t4",
            "vm_resource_id": (
                f"/subscriptions/{subscription}/resourceGroups/{group}/"
                "providers/Microsoft.Compute/virtualMachines/kova-cosmo-t4"
            ),
            "vm_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "vm_system_assigned_identity_principal_id":
                "99999999-8888-7777-6666-555555555555",
            "region": "eastus",
            "vm_size": "Standard_NC4as_T4_v3",
            "family_quota_limit_vcpus": 4,
            "capacity_confirmed": True,
            "compute_usd_per_hour": "0.5260",
            "ancillary_cost_bound_usd": "1.4740",
            "preflight_power_state": "deallocated",
            "public_ip_attached": False,
            "control_plane_deallocation_deadline_utc":
                "2026-09-19T19:55:00Z",
            "control_plane_deallocation_rule_id": "deallocate-rule-001",
            "control_plane_cleanup_rule_id": "cleanup-rule-001",
            "watchdog_principal": "independent-watchdog-principal",
            "watchdog_permission_tested_at_utc": "2026-09-19T18:50:00Z",
            "watchdog_test_result": "passed",
            "cleanup_permission_tested_at_utc": "2026-09-19T18:50:00Z",
            "cleanup_test_result": "passed",
        }

    def post_run_payload(self) -> dict:
        preflight = self.preflight_payload()
        return {
            "schema_version": 1,
            "kind": "kova_cosmo_post_run_lifecycle",
            "issuer": "kova-cosmo-lifecycle-authority-v1",
            "pilot_id": guard.PILOT_ID,
            "lifecycle_id": preflight["lifecycle_id"],
            "ledger_sequence": 5,
            "ledger_commit_id": "append-only-ledger-terminal-005",
            "ledger_append_only": True,
            "ledger_status": "terminal_cleanup_committed_no_future_grants",
            "last_paid_grant_ledger_sequence": 4,
            "future_grants_allowed": False,
            "phase_grants_committed": {
                "runtime_probe": 1,
                "training": 1,
                "evaluation": 1,
            },
            "training_runs_consumed": 1,
            "aggregate_reserved_seconds": 3600,
            "aggregate_reserved_cost_usd": "0.5260",
            "ledger_closed_at_utc": "2026-09-19T19:33:00Z",
            "provider_observation_id": "azure-observation-cleanup-001",
            "azure_query_source": "independent_azure_control_plane_reader",
            "observed_at_utc": "2026-09-19T19:35:00Z",
            "allocation_started_at_utc": "2026-09-19T19:00:00Z",
            "deallocated_at_utc": "2026-09-19T19:30:00Z",
            "subscription_id": preflight["subscription_id"],
            "resource_group": preflight["resource_group"],
            "vm_name": preflight["vm_name"],
            "vm_resource_id": preflight["vm_resource_id"],
            "vm_id": preflight["vm_id"],
            "deallocation_deadline_utc":
                preflight["control_plane_deallocation_deadline_utc"],
            "power_state": "deallocated",
            "public_ip_attached": False,
            "allocated_seconds": 1800,
            "compute_cost_upper_bound_usd": "0.2630",
            "ancillary_cost_observed_or_bound_usd": "1.4740",
            "all_in_cost_upper_bound_usd": "1.7370",
            "residual_resources": [],
            "automatic_deallocation_execution": {
                "mechanism": "azure_control_plane",
                "rule_id": preflight["control_plane_deallocation_rule_id"],
                "execution_id": "deallocate-execution-001",
                "principal": preflight["watchdog_principal"],
                "trigger": "deadline_rule",
                "status": "succeeded",
                "completed_at_utc": "2026-09-19T19:30:00Z",
                "evidence": "azure-activity-log-deallocation-record",
            },
            "automatic_cleanup_execution": {
                "mechanism": "azure_control_plane",
                "rule_id": preflight["control_plane_cleanup_rule_id"],
                "execution_id": "cleanup-execution-001",
                "principal": preflight["watchdog_principal"],
                "trigger": "post_run_automatic_cleanup",
                "status": "succeeded",
                "scope_resource_group_id":
                    preflight["cleanup_scope_resource_group_id"],
                "completed_at_utc": "2026-09-19T19:32:00Z",
                "evidence": "azure-activity-log-resource-group-delete-record",
            },
            "scoped_inventory": {
                "scope_resource_group_id":
                    preflight["cleanup_scope_resource_group_id"],
                "query_id": "azure-resource-graph-query-001",
                "query_succeeded": True,
                "queried_at_utc": "2026-09-19T19:34:00Z",
                "resource_group_state": "deleted",
                "remaining_resource_ids": [],
                "evidence": "azure-resource-graph-empty-inventory-record",
            },
        }

    def signed_path(self, payload: dict, name: str,
                    key: Ed25519PrivateKey | None = None) -> Path:
        signer = key or self.authority_key
        envelope = {
            "payload": payload,
            "signature": signer.sign(canonical(payload)).hex(),
        }
        path = self.directory / name
        path.write_text(json.dumps(envelope), encoding="utf-8")
        return path

    def test_checked_in_policy_is_bounded_and_default_source_stays_blocked(self):
        report = guard.dry_run()
        self.assertEqual(report["approved_all_in_budget_usd"], "2.0000")
        self.assertEqual(report["maximum_allocated_minutes"], 60)
        self.assertEqual(report["maximum_paid_phase_grants"], 3)
        self.assertEqual(report["maximum_training_runs"], 1)
        self.assertIn("lifecycle_authority_unprovisioned", report["blockers"])
        self.assertFalse(report["resource_created"])
        self.assertFalse(report["spending_started"])
        self.assertFalse(report["deployment_authorized"])

    def test_fresh_authority_signed_preflight_is_accepted(self):
        path = self.signed_path(self.preflight_payload(), "preflight.json")
        report = guard.require_ready(
            root=self.root, evidence_path=path, now=NOW
        )
        self.assertEqual(report["status"], "ready_for_single_bounded_pilot")
        self.assertEqual(report["maximum_training_runs"], 1)
        self.assertTrue(report["remote_paid_phase_grant_required"])
        self.assertEqual(
            report["runtime_evidence_sha256"],
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def test_preflight_tampering_after_signature_is_rejected(self):
        path = self.signed_path(
            self.preflight_payload(), "tampered-preflight.json"
        )
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["payload"]["capacity_confirmed"] = False
        path.write_text(json.dumps(envelope), encoding="utf-8")
        with self.assertRaises(guard.RuntimeGuardError):
            guard.require_ready(root=self.root, evidence_path=path, now=NOW)

    def test_preflight_from_fake_signer_is_rejected(self):
        fake = Ed25519PrivateKey.from_private_bytes(b"f" * 32)
        path = self.signed_path(
            self.preflight_payload(), "fake-preflight.json", fake
        )
        with self.assertRaises(guard.RuntimeGuardError):
            guard.require_ready(root=self.root, evidence_path=path, now=NOW)

    def test_unsigned_self_asserted_preflight_is_rejected(self):
        path = self.directory / "unsigned-preflight.json"
        path.write_text(json.dumps(self.preflight_payload()), encoding="utf-8")
        with self.assertRaises(guard.RuntimeGuardError):
            guard.require_ready(root=self.root, evidence_path=path, now=NOW)

    def test_preflight_must_be_external_to_repository(self):
        path = self.signed_path(self.preflight_payload(), "external.json")
        checked_in = self.root / "evidence.json"
        checked_in.write_bytes(path.read_bytes())
        with self.assertRaises(guard.RuntimeGuardError):
            guard.require_ready(
                root=self.root, evidence_path=checked_in, now=NOW
            )

    def test_preflight_rejects_stale_or_unbounded_provider_claims(self):
        mutations = [
            lambda value: value.update(captured_at_utc="2026-09-19T18:00:00Z"),
            lambda value: value.update(public_ip_attached=True),
            lambda value: value.update(vm_size="Standard_NC4as_T4_v3-fake"),
            lambda value: value.update(capacity_confirmed=False),
            lambda value: value.update(compute_usd_per_hour="0.5261"),
            lambda value: value.update(azure_query_source="runner_self_report"),
            lambda value: value.update(vm_id=
                                       "00000000-1111-2222-3333-bad"),
            lambda value: value.update(vm_resource_id=
                                       value["vm_resource_id"] + "-other"),
        ]
        for index, mutate in enumerate(mutations):
            payload = self.preflight_payload()
            mutate(payload)
            path = self.signed_path(payload, f"invalid-preflight-{index}.json")
            with self.subTest(index=index), self.assertRaises(
                guard.RuntimeGuardError
            ):
                guard.require_ready(
                    root=self.root, evidence_path=path, now=NOW
                )

    def test_authority_signed_cleanup_and_empty_inventory_are_accepted(self):
        preflight = self.preflight_payload()
        path = self.signed_path(self.post_run_payload(), "post-run.json")
        report = guard.verify_post_run(preflight, path, root=self.root)
        self.assertTrue(report["automatic_deallocation_verified"])
        self.assertTrue(report["automatic_cleanup_verified"])
        self.assertTrue(report["resource_group_deleted"])
        self.assertTrue(report["terminal_ledger_verified"])
        self.assertFalse(report["future_grants_allowed"])
        self.assertEqual(report["residual_resource_count"], 0)
        self.assertEqual(
            report["post_run_evidence_sha256"],
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def test_cleanup_tampering_fake_signer_and_unsigned_claims_are_rejected(self):
        preflight = self.preflight_payload()
        valid = self.signed_path(self.post_run_payload(), "valid-post.json")
        tampered = self.directory / "tampered-post.json"
        envelope = json.loads(valid.read_text(encoding="utf-8"))
        envelope["payload"]["scoped_inventory"]["remaining_resource_ids"] = [
            "/subscriptions/fake/resourceGroups/kova/providers/fake/resource"
        ]
        tampered.write_text(json.dumps(envelope), encoding="utf-8")
        fake = self.signed_path(
            self.post_run_payload(), "fake-post.json",
            Ed25519PrivateKey.from_private_bytes(b"f" * 32),
        )
        unsigned = self.directory / "unsigned-post.json"
        unsigned.write_text(json.dumps(self.post_run_payload()), encoding="utf-8")
        for path in (tampered, fake, unsigned):
            with self.subTest(path=path.name), self.assertRaises(
                guard.RuntimeGuardError
            ):
                guard.verify_post_run(preflight, path, root=self.root)

    def test_cleanup_accepts_each_committed_phase_prefix(self):
        cases = [
            ({"runtime_probe": 1, "training": 0, "evaluation": 0},
             0, 600, "0.0877"),
            ({"runtime_probe": 1, "training": 1, "evaluation": 0},
             1, 3000, "0.4384"),
        ]
        for index, (counts, runs, seconds, cost) in enumerate(cases):
            payload = self.post_run_payload()
            payload["phase_grants_committed"] = counts
            payload["training_runs_consumed"] = runs
            payload["aggregate_reserved_seconds"] = seconds
            payload["aggregate_reserved_cost_usd"] = cost
            path = self.signed_path(payload, f"partial-{index}.json")
            with self.subTest(counts=counts):
                report = guard.verify_post_run(
                    self.preflight_payload(), path, root=self.root
                )
                self.assertTrue(report["resource_group_deleted"])

    def test_cleanup_rejects_skipped_phase_and_mismatched_aggregates(self):
        mutations = [
            lambda value: value["phase_grants_committed"].update(
                runtime_probe=1, training=0, evaluation=1
            ),
            lambda value: value.update(training_runs_consumed=0),
            lambda value: value.update(aggregate_reserved_seconds=3000),
            lambda value: value.update(aggregate_reserved_cost_usd="0.4384"),
        ]
        for index, mutate in enumerate(mutations):
            payload = self.post_run_payload()
            mutate(payload)
            path = self.signed_path(payload, f"bad-progression-{index}.json")
            with self.subTest(index=index), self.assertRaises(
                guard.RuntimeGuardError
            ):
                guard.verify_post_run(
                    self.preflight_payload(), path, root=self.root
                )

    def test_cleanup_rejects_wrong_lifecycle_or_nonmonotonic_ledger(self):
        for field, wrong in (
            ("lifecycle_id", "different-lifecycle"),
            ("ledger_sequence", 1),
            ("azure_query_source", "runner_self_report"),
        ):
            payload = self.post_run_payload()
            payload[field] = wrong
            path = self.signed_path(payload, f"wrong-{field}.json")
            with self.subTest(field=field), self.assertRaises(
                guard.RuntimeGuardError
            ):
                guard.verify_post_run(
                    self.preflight_payload(), path, root=self.root
                )

    def test_cleanup_rejects_stale_or_nonterminal_ledger_closure(self):
        mutations = [
            lambda value: value.update(
                ledger_status="grants_open",
                future_grants_allowed=True,
            ),
            lambda value: value.update(last_paid_grant_ledger_sequence=5),
            lambda value: value["phase_grants_committed"].update(
                evaluation=0
            ),
            lambda value: value["phase_grants_committed"].update(
                evaluation=True
            ),
            lambda value: value.update(training_runs_consumed=True),
            lambda value: value.update(ledger_closed_at_utc=
                                       "2026-09-19T19:31:00Z"),
        ]
        for index, mutate in enumerate(mutations):
            payload = self.post_run_payload()
            mutate(payload)
            path = self.signed_path(payload, f"bad-terminal-{index}.json")
            with self.subTest(index=index), self.assertRaises(
                guard.RuntimeGuardError
            ):
                guard.verify_post_run(
                    self.preflight_payload(), path, root=self.root
                )

    def test_cleanup_rejects_false_automatic_provenance_or_residuals(self):
        mutations = [
            lambda value: value["automatic_deallocation_execution"].update(
                status="manual"
            ),
            lambda value: value["automatic_cleanup_execution"].update(
                trigger="operator_command"
            ),
            lambda value: value["scoped_inventory"].update(
                resource_group_state="exists"
            ),
            lambda value: value.update(residual_resources=[{
                "resource_id": "fake", "billable": True
            }]),
        ]
        for index, mutate in enumerate(mutations):
            payload = self.post_run_payload()
            mutate(payload)
            path = self.signed_path(payload, f"bad-cleanup-{index}.json")
            with self.subTest(index=index), self.assertRaises(
                guard.RuntimeGuardError
            ):
                guard.verify_post_run(
                    self.preflight_payload(), path, root=self.root
                )

    def test_environment_path_is_supported_but_cannot_override_signature(self):
        path = self.signed_path(self.preflight_payload(), "from-env.json")
        with patch.dict("os.environ", {guard.EVIDENCE_ENV: str(path)}, clear=True):
            report = guard.require_ready(root=self.root, now=NOW)
        self.assertEqual(report["pilot_id"], guard.PILOT_ID)


if __name__ == "__main__":
    unittest.main()
