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
from unittest.mock import Mock, patch

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
            "watchdog_cleanup_trigger_utc": "2026-09-24T13:15:00Z",
            "region": "eastus", "vm_sku": launch.SKU, "image_urn": launch.IMAGE,
            "model_revision": launch.MODEL_REVISION,
            "model_manifest_sha256": contract.MANIFEST_SHA256["kova-cosmo"],
            "dataset_sha256": contract.APPROVED_DATASET_SHA256,
            "train_records": 27, "validation_records": 15, "gpu_name": "NVIDIA T4",
            "quota": {"family_limit_vcpus": 4, "family_used_vcpus": 0,
                      "regional_limit_vcpus": 14, "regional_used_vcpus": 0},
            "sku_restrictions": [], "account_compute_hourly_usd": "0.4000",
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
        self.assertEqual(planned["minimum_signed_allocation_seconds"], 5400)
        assessed = self.check()
        self.assertEqual(assessed["worst_case_all_in_usd"], "3.1934")
        self.assertEqual(assessed["signed_allocation_seconds"], 7140)
        self.assertFalse(assessed["paid_actions_enabled"])

    def test_shorter_signed_window_fits_only_when_compute_reservation_covers_it(self):
        payload = deepcopy(self.payload)
        payload["account_compute_hourly_usd"] = "0.5260"
        payload["allocation_deadline_utc"] = "2026-09-24T13:30:00Z"
        result = self.check(payload)
        self.assertEqual(result["signed_allocation_seconds"], 5400)
        self.assertEqual(result["worst_case_all_in_usd"], "3.1890")
        payload["allocation_deadline_utc"] = "2026-09-24T13:42:00Z"
        self.assertEqual(self.check(payload)["worst_case_all_in_usd"], "3.2942")
        payload["allocation_deadline_utc"] = "2026-09-24T13:43:00Z"
        with self.assertRaisesRegex(contract.ContractError, "Cosmo worst case"):
            self.check(payload)
        payload["allocation_deadline_utc"] = "2026-09-24T13:29:59Z"
        with self.assertRaisesRegex(launch.LaunchRejected, "90-120 minute"):
            self.check(payload)
        payload["allocation_deadline_utc"] = "2026-09-24T13:30:00Z"
        payload["account_compute_hourly_usd"] = "0.6001"
        with self.assertRaisesRegex(contract.ContractError, "Cosmo worst case"):
            self.check(payload)

    def test_cleanup_trigger_must_precede_signed_cost_deadline(self):
        payload = deepcopy(self.payload)
        payload["allocation_deadline_utc"] = "2026-09-24T13:30:00Z"
        payload["watchdog_cleanup_trigger_utc"] = "2026-09-24T13:15:00Z"
        self.assertEqual(self.check(payload)["watchdog_cleanup_trigger_utc"],
                         "2026-09-24T13:15:00Z")
        for trigger in ("2026-09-24T13:15:01Z", "2026-09-24T12:01:00Z",
                        "2026-09-24T12:00:59Z"):
            with self.subTest(trigger=trigger):
                payload["watchdog_cleanup_trigger_utc"] = trigger
                with self.assertRaisesRegex(launch.LaunchRejected, "insufficient deletion time"):
                    self.check(payload)

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
            ("account_compute_hourly_usd", "0.4600"),
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
              "/resourceGroups/pilot/providers/Microsoft.Compute/virtualMachines/kova-t4-test01",
              "vm_id": "12345678-1234-1234-1234-123456789abd",
              "system_assigned_identity_principal_id":
              "12345678-1234-1234-1234-123456789abe"}
        network_prefix = ("/subscriptions/" + self.subscription +
                          "/resourceGroups/pilot/providers/Microsoft.Network/")
        nic = network_prefix + "networkInterfaces/kova-t4-nic-test01"
        live_network = {
            "vm_nic_ids": [nic], "vm_nic_id": nic, "nic_public_ip_id": None,
            "subnet_id": network_prefix + "virtualNetworks/kova-t4-vnet-test01/subnets/pilot",
            "subnet_default_outbound_access": False,
            "nat_gateway_id": network_prefix + "natGateways/kova-t4-egress-nat-test01",
            "nat_gateway_public_ip_id": network_prefix + "publicIPAddresses/kova-t4-egress-ip-test01",
            "nat_gateway_sku": "Standard", "nat_public_ip_sku": "Standard",
            "network_security_group_id": network_prefix +
            "networkSecurityGroups/kova-t4-egress-nsg-test01",
            "inbound_deny_rule": "deny-all-inbound",
            "allowed_outbound_tcp_ports": [80, 443],
            "other_outbound_denied": True,
            "verified_from_azure_control_plane": True,
            "observed_at_utc": "2026-09-24T12:00:00Z",
        }
        runtime = {"schema_version": 1, "kind": "kova_cosmo_qlora_runtime_preflight",
                   "issuer": authority.ISSUER, "quote_sha256": admission["quote_sha256"],
                   "source_commit": self.commit, "subscription_id": self.subscription,
                   "lifecycle_id": "one-pilot", "preflight_ledger_sequence": 3,
                   "azure_instance": vm, "allocation_deadline_utc":
                   admission["allocation_deadline_utc"],
                   "watchdog_cleanup_trigger_utc":
                   admission["watchdog_cleanup_trigger_utc"],
                   "observed_at_utc": "2026-09-24T12:00:00Z",
                   "watchdog_healthy": True, "cleanup_scope_verified": True,
                   "azure_network": live_network}
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
            cleanup_trigger_utc=admission["watchdog_cleanup_trigger_utc"],
            now=self.now, root=self.root)
        self.assertEqual(received["azure_instance"], vm)
        save_runtime({**runtime, "quote_sha256": "a" * 64})
        with self.assertRaises(grant.GrantRejected):
            grant.read_runtime_preflight(runtime_path,
                quote_sha256=admission["quote_sha256"], source_commit=self.commit,
                subscription_id=self.subscription,
                deadline_utc=admission["allocation_deadline_utc"],
                cleanup_trigger_utc=admission["watchdog_cleanup_trigger_utc"],
                now=self.now, root=self.root)
        save_runtime({**runtime, "watchdog_cleanup_trigger_utc":
                      "2026-09-24T13:14:59Z"})
        with self.assertRaises(grant.GrantRejected):
            grant.read_runtime_preflight(runtime_path,
                quote_sha256=admission["quote_sha256"], source_commit=self.commit,
                subscription_id=self.subscription,
                deadline_utc=admission["allocation_deadline_utc"],
                cleanup_trigger_utc=admission["watchdog_cleanup_trigger_utc"],
                now=self.now, root=self.root)
        for key, value in (("nic_public_ip_id", "https://public.example.test/"),
                           ("nat_gateway_id", "untrusted-nat"),
                           ("network_security_group_id", "untrusted-nsg"),
                           ("subnet_default_outbound_access", True),
                           ("other_outbound_denied", False)):
            changed = deepcopy(runtime)
            changed["azure_network"][key] = value
            save_runtime(changed)
            with self.subTest(key=key), self.assertRaises(grant.GrantRejected):
                grant.read_runtime_preflight(runtime_path,
                    quote_sha256=admission["quote_sha256"], source_commit=self.commit,
                    subscription_id=self.subscription,
                    deadline_utc=admission["allocation_deadline_utc"],
                    cleanup_trigger_utc=admission["watchdog_cleanup_trigger_utc"],
                    now=self.now, root=self.root)
        save_runtime(runtime)

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
                           "network_evidence_sha256",
                           "azure_instance", "request_nonce", "allocation_deadline_utc",
                           "watchdog_cleanup_trigger_utc",
                           "all_in_ceiling_usd")},
                       "ledger_sequence": 4, "ledger_commit_id": "atomic-commit",
                       "ledger_append_only": True,
                       "ledger_status": "grant_committed_before_response",
                       "grant_id": "one-and-only", "azure_identity_token_sha256": token_sha,
                       "issued_at_utc": "2026-09-24T12:01:00Z",
                       "expires_at_utc": "2026-09-24T13:15:00Z",
                       "training_runs_consumed": runs, "all_in_reserved_usd": "3.3000",
                       "watchdog_healthy": True, "cleanup_scope_verified": True,
                       "deployment_authorized": False}
            return {"payload": payload,
                    "signature": self.key.sign(authority.canonical(payload)).hex()}

        def call(transport, *, now=None, response_now=None):
            with patch.object(authority, "_executing_azure_identity", return_value=(
                    "synthetic-token", token_sha, "2026-09-24T14:30:00Z")), \
                 patch.object(authority, "_load_bearer_token", return_value="synthetic"), \
                 patch.object(grant.launch, "assess_signed_quote", return_value=admission):
                return grant.acquire_training_grant(
                    quote=quote, source_commit=self.commit,
                    subscription_id=self.subscription, lifecycle_id="one-pilot",
                    preflight_ledger_sequence=3, azure_instance=vm,
                    now=now or self.now, response_now=response_now,
                    root=self.root, transport=transport, runtime_evidence=runtime_path,
                    instance_transport=lambda _url: compute)

        self.assertEqual(call(committed)["training_runs_consumed"], 1)
        def signed_later(endpoint, token, request):
            envelope = committed(endpoint, token, request)
            envelope["payload"]["issued_at_utc"] = "2026-09-24T12:01:02Z"
            envelope["signature"] = self.key.sign(
                authority.canonical(envelope["payload"])).hex()
            return envelope
        self.assertEqual(call(signed_later, response_now=datetime(
            2026, 9, 24, 12, 1, 2, tzinfo=timezone.utc))["training_runs_consumed"], 1)
        def signed_expired(endpoint, token, request):
            envelope = committed(endpoint, token, request)
            envelope["payload"]["expires_at_utc"] = "2026-09-24T12:01:01Z"
            envelope["signature"] = self.key.sign(
                authority.canonical(envelope["payload"])).hex()
            return envelope
        with self.assertRaises(grant.GrantRejected):
            call(signed_expired, response_now=datetime(
                2026, 9, 24, 12, 1, 2, tzinfo=timezone.utc))
        with self.assertRaises(grant.GrantRejected):
            call(committed, response_now=datetime(
                2026, 9, 24, 12, 2, 1, tzinfo=timezone.utc))
        with self.assertRaises(grant.GrantRejected):
            call(lambda e, t, r: committed(e, t, r, runs=2))
        for key, bad in (("watchdog_cleanup_trigger_utc", "2026-09-24T13:16:00Z"),
                         ("expires_at_utc", "2026-09-24T13:16:00Z")):
            def signed_mismatch(endpoint, token, request):
                envelope = committed(endpoint, token, request)
                envelope["payload"][key] = bad
                envelope["signature"] = self.key.sign(
                    authority.canonical(envelope["payload"])).hex()
                return envelope
            with self.subTest(key=key), self.assertRaises(grant.GrantRejected):
                call(signed_mismatch)
        compute["storageProfile"]["imageReference"]["exactVersion"] = "latest"
        with self.assertRaises(grant.GrantRejected):
            call(committed)
        compute["storageProfile"]["imageReference"]["exactVersion"] = "24.04.202609040"
        updated = deepcopy(runtime)
        updated["observed_at_utc"] = "2026-09-24T12:30:00Z"
        updated["azure_network"]["observed_at_utc"] = "2026-09-24T12:30:00Z"
        save_runtime(updated)
        transport = Mock()
        with self.assertRaises(grant.GrantRejected):
            call(transport, now=datetime(2026, 9, 24, 12, 31, tzinfo=timezone.utc))
        transport.assert_not_called()
        updated["observed_at_utc"] = "2026-09-24T12:29:00Z"
        updated["azure_network"]["observed_at_utc"] = "2026-09-24T12:29:00Z"
        save_runtime(updated)
        with self.assertRaises(grant.GrantRejected):
            call(transport, now=datetime(2026, 9, 24, 12, 29, 30,
                                         tzinfo=timezone.utc))
        transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
