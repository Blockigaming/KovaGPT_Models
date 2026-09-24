"""Signed Cosmo launch admission regressions; all fixtures are synthetic."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from training import cosmo_lifecycle_authority as authority
from training import cosmo_qlora_launch as launch
from training import cosmo_qlora_grant as grant
from training import three_family_contract as contract


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "config").mkdir()
        self.key = Ed25519PrivateKey.generate()
        public = self.key.public_key().public_bytes_raw()
        trust = json.loads((launch.ROOT / authority.TRUST_PATH).read_text())
        trust.update({
            "status": "authority_pinned", "endpoint": "https://example.test/v1/pilot/grants",
            "public_key_hex": public.hex(),
            "public_key_sha256": hashlib.sha256(public).hexdigest(),
            "azure_managed_identity_token_audience": "api://cosmo-authority-test",
        })
        (self.root / authority.TRUST_PATH).write_text(json.dumps(trust))
        self.subscription = "12345678-1234-1234-1234-123456789abc"
        self.commit = "d" * 40
        self.now = datetime(2026, 9, 24, 12, 1, tzinfo=timezone.utc)
        self.payload = {
            "schema_version": 1, "kind": launch.KIND, "issuer": authority.ISSUER,
            "subscription_id": self.subscription, "source_commit": self.commit,
            "observed_at_utc": "2026-09-24T12:00:00Z",
            "expires_at_utc": "2026-09-24T12:05:00Z",
            "allocation_deadline_utc": "2026-09-24T13:59:00Z",
            "region": "eastus", "vm_sku": launch.SKU, "image_urn": launch.IMAGE,
            "model_revision": launch.MODEL_REVISION,
            "model_manifest_sha256": contract.MANIFEST_SHA256["kova-cosmo"],
            "dataset_sha256": contract.APPROVED_DATASET_SHA256,
            "train_records": 27, "validation_records": 15, "gpu_name": "NVIDIA T4",
            "quota": {"family_limit_vcpus": 4, "family_used_vcpus": 0,
                      "regional_limit_vcpus": 14, "regional_used_vcpus": 0},
            "sku_restrictions": [], "account_compute_hourly_usd": "0.5260",
            "account_meter_source": "subscription_specific_billing_price_sheet",
            "category_upper_bounds_usd": dict(contract.COST_CATEGORY_BOUNDS),
            "all_category_rates_checked": True, "watchdog_health_tested": True,
            "watchdog_can_deallocate_and_delete": True,
            "exclusive_pilot_group_empty": True, "no_public_ip": True,
        }

    def signed(self, payload=None):
        payload = deepcopy(payload or self.payload)
        signature = self.key.sign(authority.canonical(payload)).hex()
        path = self.root / "quote.json"
        path.write_text(json.dumps({"payload": payload, "signature": signature}))
        return path

    def check(self, payload=None):
        return launch.assess_signed_quote(self.signed(payload), root=self.root,
                                          source_commit=self.commit,
                                          subscription_id=self.subscription,
                                          now=self.now)

    def test_approved_source_and_signed_account_quote_reconcile(self):
        planned = launch.proposal()
        self.assertEqual((planned["train_records"], planned["validation_records"]), (27, 15))
        self.assertFalse(planned["paid_actions_enabled"])
        self.assertEqual(planned["all_in_ceiling_usd"], "3.3000")
        assessed = self.check()
        self.assertEqual(assessed["worst_case_all_in_usd"], "3.2020")
        self.assertFalse(assessed["paid_actions_enabled"])

    def test_unsigned_tampered_and_unpinned_authorities_fail(self):
        path = self.signed()
        contents = json.loads(path.read_text())
        contents["payload"]["account_compute_hourly_usd"] = "0.1000"
        path.write_text(json.dumps(contents))
        with self.assertRaises(launch.LaunchRejected):
            launch.assess_signed_quote(path, root=self.root, source_commit=self.commit,
                                       subscription_id=self.subscription, now=self.now)
        with self.assertRaises(launch.LaunchRejected):
            launch.assess_signed_quote(self.signed(), root=launch.ROOT,
                                       source_commit=self.commit,
                                       subscription_id=self.subscription, now=self.now)

    def test_retailability_budget_quotas_identity_and_watchdog_fail_closed(self):
        edits = (
            ("account_meter_source", "azure_retail_price_api"),
            ("account_compute_hourly_usd", "0.5751"),
            ("source_commit", "e" * 40),
            ("dataset_sha256", "a" * 64),
            ("model_revision", "a" * 40),
            ("image_urn", "Canonical:ubuntu-24_04-lts:server:latest"),
            ("gpu_name", "Generic 7.5 GPU"),
            ("subscription_id", "12345678-1234-1234-1234-123456789abd"),
            ("sku_restrictions", ["NotAvailableForSubscription"]),
            ("watchdog_health_tested", False),
            ("watchdog_can_deallocate_and_delete", False),
            ("all_category_rates_checked", False),
            ("no_public_ip", False),
            ("expires_at_utc", "2026-09-24T12:00:00Z"),
            ("allocation_deadline_utc", "2026-09-24T14:01:00Z"),
        )
        for field, value in edits:
            with self.subTest(field=field):
                payload = deepcopy(self.payload)
                payload[field] = value
                with self.assertRaises((launch.LaunchRejected, contract.ContractError)):
                    self.check(payload)
        payload = deepcopy(self.payload)
        payload["quota"]["family_used_vcpus"] = 1
        with self.assertRaises(launch.LaunchRejected):
            self.check(payload)
        payload = deepcopy(self.payload)
        del payload["category_upper_bounds_usd"]["managed_disks"]
        with self.assertRaises(launch.LaunchRejected):
            self.check(payload)
        payload = deepcopy(self.payload)
        payload["category_upper_bounds_usd"]["managed_disks"] = "0.4000"
        with self.assertRaises(launch.LaunchRejected):
            self.check(payload)

    def test_command_line_cannot_start_a_paid_action(self):
        with redirect_stderr(StringIO()), patch("subprocess.run") as run:
            with self.assertRaises(SystemExit):
                launch.main(["--execute"])
            run.assert_not_called()
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(launch.main([]), 0)
        self.assertFalse(json.loads(output.getvalue())["paid_actions_enabled"])

    def test_runtime_preflight_and_one_run_grant_bind_instance_and_quote(self):
        quote = self.signed()
        admission = self.check()
        vm = {"resource_id": "/subscriptions/" + self.subscription +
              "/resourceGroups/pilot/providers/Microsoft.Compute/virtualMachines/cosmo",
              "vm_id": "12345678-1234-1234-1234-123456789abd",
              "system_assigned_identity_principal_id":
              "12345678-1234-1234-1234-123456789abe"}
        runtime = {"schema_version": 1, "kind": "kova_cosmo_qlora_runtime_preflight",
                   "issuer": authority.ISSUER, "quote_sha256": admission["quote_sha256"],
                   "source_commit": self.commit, "subscription_id": self.subscription,
                   "lifecycle_id": "one-pilot", "preflight_ledger_sequence": 3,
                   "azure_instance": vm, "allocation_deadline_utc":
                   admission["allocation_deadline_utc"],
                   "observed_at_utc": "2026-09-24T12:00:00Z",
                   "watchdog_healthy": True, "cleanup_scope_verified": True}
        outside = TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        runtime_path = Path(outside.name) / "runtime.json"

        def save_runtime(value):
            runtime_path.write_text(json.dumps({"payload": value,
                "signature": self.key.sign(authority.canonical(value)).hex()}))

        save_runtime(runtime)
        received = grant.read_runtime_preflight(runtime_path,
            quote_sha256=admission["quote_sha256"], source_commit=self.commit,
            subscription_id=self.subscription,
            deadline_utc=admission["allocation_deadline_utc"],
            now=self.now, root=self.root)
        self.assertEqual(received["azure_instance"], vm)
        save_runtime({**runtime, "quote_sha256": "a" * 64})
        with self.assertRaises(grant.GrantRejected):
            grant.read_runtime_preflight(runtime_path,
                quote_sha256=admission["quote_sha256"], source_commit=self.commit,
                subscription_id=self.subscription,
                deadline_utc=admission["allocation_deadline_utc"],
                now=self.now, root=self.root)

        compute = {"resourceId": vm["resource_id"], "vmId": vm["vm_id"],
                   "location": "eastus", "vmSize": launch.SKU,
                   "storageProfile": {"imageReference": {
                       "publisher": "Canonical", "offer": "ubuntu-24_04-lts",
                       "sku": "server", "exactVersion": "24.04.202609040"}}}
        token_sha = "b" * 64

        def committed(_endpoint, _token, request, *, runs=1):
            payload = {"schema_version": 1,
                       "kind": "kova_cosmo_qlora_training_grant",
                       "issuer": authority.ISSUER,
                       **{key: request[key] for key in (
                           "source_commit", "subscription_id", "quote_sha256",
                           "lifecycle_id", "preflight_ledger_sequence",
                           "azure_instance", "request_nonce", "allocation_deadline_utc",
                           "all_in_ceiling_usd")},
                       "ledger_sequence": 4, "ledger_commit_id": "atomic-commit",
                       "ledger_append_only": True,
                       "ledger_status": "grant_committed_before_response",
                       "grant_id": "one-and-only", "azure_identity_token_sha256": token_sha,
                       "issued_at_utc": "2026-09-24T12:01:00Z",
                       "expires_at_utc": "2026-09-24T13:59:00Z",
                       "training_runs_consumed": runs, "all_in_reserved_usd": "3.3000",
                       "watchdog_healthy": True, "cleanup_scope_verified": True,
                       "deployment_authorized": False}
            return {"payload": payload,
                    "signature": self.key.sign(authority.canonical(payload)).hex()}

        def call(transport):
            with patch.object(authority, "_executing_azure_identity", return_value=(
                    "synthetic-token", token_sha, "2026-09-24T14:30:00Z")), \
                 patch.object(authority, "_load_bearer_token", return_value="synthetic"), \
                 patch.object(grant.launch, "assess_signed_quote", return_value=admission):
                return grant.acquire_training_grant(
                    quote=quote, source_commit=self.commit,
                    subscription_id=self.subscription, lifecycle_id="one-pilot",
                    preflight_ledger_sequence=3, azure_instance=vm, now=self.now,
                    root=self.root, transport=transport,
                    instance_transport=lambda _url: compute)

        self.assertEqual(call(committed)["training_runs_consumed"], 1)
        with self.assertRaises(grant.GrantRejected):
            call(lambda e, t, r: committed(e, t, r, runs=2))
        compute["storageProfile"]["imageReference"]["exactVersion"] = "latest"
        with self.assertRaises(grant.GrantRejected):
            call(committed)


if __name__ == "__main__":
    unittest.main()
