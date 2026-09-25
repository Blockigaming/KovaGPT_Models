"""Exercise the guest-to-issuer protocol using synthetic independent observations."""
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from training import cosmo_lifecycle_authority as authority
from training import cosmo_qlora_grant as client
from training import cosmo_qlora_launch as launch
from training.cosmo_controller_grants import GrantIssuer
from training.cosmo_controller_ledger import LedgerRejected, digest
from training import test_cosmo_controller_ledger as ledger_tests
from training import test_cosmo_qlora_launch as launch_tests


class GrantIssuerTests(unittest.TestCase):
    def setUp(self):
        # Reuse existing synthetic fixtures without inheriting their test cases.
        self.quote_fixture = launch_tests.LaunchTests("runTest")
        self.quote_fixture.setUp()
        self.addCleanup(self.quote_fixture.doCleanups)
        self.f = ledger_tests.ControllerLedgerTests("runTest")
        self.f.setUp()
        self.f.key = self.quote_fixture.key
        self.ledger = self.f.make()
        self.quote = self.quote_fixture.signed()
        self.admission = self.quote_fixture.check()
        self.f.cost["quote_sha256"] = self.admission["quote_sha256"]
        self.ledger.initialize()
        self.ledger.append(self.f.health, expected_sequence=0)
        self.ledger.append(self.f.cost, expected_sequence=1)
        group = self.f.context["lifecycle"]["pilot_resource_group_id"]
        self.vm = {"resource_id": group + "/providers/Microsoft.Compute/virtualMachines/kova-t4-test01",
            "vm_id": "12345678-1234-1234-1234-123456789abd",
            "system_assigned_identity_principal_id": "12345678-1234-1234-1234-123456789abe"}
        prefix = group + "/providers/Microsoft.Network/"
        nic = prefix + "networkInterfaces/kova-t4-nic-test01"
        network = {"vm_nic_ids": [nic], "vm_nic_id": nic, "nic_public_ip_id": None,
            "subnet_id": prefix + "virtualNetworks/kova-t4-vnet-test01/subnets/pilot",
            "subnet_default_outbound_access": False,
            "nat_gateway_id": prefix + "natGateways/kova-t4-egress-nat-test01",
            "nat_gateway_public_ip_id": prefix + "publicIPAddresses/kova-t4-egress-ip-test01",
            "nat_gateway_sku": "Standard", "nat_public_ip_sku": "Standard",
            "network_security_group_id": prefix + "networkSecurityGroups/kova-t4-egress-nsg-test01",
            "inbound_deny_rule": "deny-all-inbound", "allowed_outbound_tcp_ports": [80, 443],
            "other_outbound_denied": True, "verified_from_azure_control_plane": True,
            "observed_at_utc": "2026-09-24T12:01:00Z"}
        runtime = {"schema_version": 1, "kind": "kova_cosmo_qlora_runtime_preflight",
            "issuer": authority.ISSUER, "quote_sha256": self.admission["quote_sha256"],
            "source_commit": self.f.context["source_commit"],
            "subscription_id": self.quote_fixture.subscription,
            "lifecycle_id": self.f.context["lifecycle"]["lifecycle_id"],
            "preflight_ledger_sequence": 2, "azure_instance": self.vm,
            "allocation_deadline_utc": self.admission["allocation_deadline_utc"],
            "watchdog_cleanup_trigger_utc": self.admission["watchdog_cleanup_trigger_utc"],
            "observed_at_utc": "2026-09-24T12:01:00Z", "watchdog_healthy": True,
            "cleanup_scope_verified": True, "azure_network": network}
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.runtime_path = Path(temp.name) / "runtime.json"
        self.runtime_path.write_text(json.dumps({"payload": runtime,
            "signature": self.f.key.sign(authority.canonical(runtime)).hex()}))
        self.token = ".".join(["syntheticmanagedidentitytoken00000"] * 3)
        self.token_digest = hashlib.sha256(self.token.encode()).hexdigest()
        self.observation_override = {}
        self.issuer = GrantIssuer(ledger=self.ledger, quote=self.quote,
            runtime_evidence=self.runtime_path, verify_live_request=self.verify_live,
            root=self.quote_fixture.root)
        self.request = {"schema_version": 1, "kind": "kova_cosmo_qlora_training_grant_request",
            "source_commit": self.f.context["source_commit"],
            "subscription_id": self.quote_fixture.subscription,
            "quote_sha256": self.admission["quote_sha256"],
            "network_evidence_sha256": digest(network),
            "lifecycle_id": runtime["lifecycle_id"], "preflight_ledger_sequence": 2,
            "azure_instance": self.vm, "azure_instance_identity_token": self.token,
            "azure_instance_identity_token_audience": "api://cosmo-authority-test",
            "azure_instance_identity_token_expires_at_utc": "2026-09-24T14:30:00Z",
            "allocation_deadline_utc": runtime["allocation_deadline_utc"],
            "watchdog_cleanup_trigger_utc": runtime["watchdog_cleanup_trigger_utc"],
            "training_runs_limit": 1, "all_in_ceiling_usd": "3.3000",
            "request_nonce": "a" * 64, "requested_at_utc": "2026-09-24T12:01:00Z"}
        clean = patch.object(launch, "clean_source_commit", return_value=self.f.context["source_commit"])
        clean.start()
        self.addCleanup(clean.stop)

    def verify_live(self, *, token, audience, instance, network, cleanup_trigger_utc):
        self.assertEqual(token, self.token)
        return {"token_sha256": hashlib.sha256(token.encode()).hexdigest(), "audience": audience,
            "azure_instance": deepcopy(instance), "token_expires_at_utc": "2026-09-24T14:30:00Z",
            "network_evidence_sha256": digest(network),
            "watchdog_cleanup_trigger_utc": cleanup_trigger_utc,
            "observed_at_utc": self.f.now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "entra_signature_verified": True, "azure_control_plane_verified": True,
            "watchdog_healthy": True, "cleanup_scope_verified": True,
            **self.observation_override}

    def test_committed_response_matches_existing_guest_verifier(self):
        compute = {"resourceId": self.vm["resource_id"], "vmId": self.vm["vm_id"],
            "location": "eastus", "vmSize": launch.SKU, "storageProfile": {
                "imageReference": {"publisher": "Canonical", "offer": "ubuntu-24_04-lts",
                    "sku": "server", "exactVersion": "24.04.202609040"}}}
        with patch.object(authority, "_executing_azure_identity", return_value=(
            self.token, self.token_digest, "2026-09-24T14:30:00Z")), \
             patch.object(authority, "_load_bearer_token", return_value="synthetic"), \
             patch.object(authority, "_https_transport",
                          side_effect=lambda endpoint, token, request, **kwargs:
                          self.issuer.issue(request)) as http:
            received = client.acquire_training_grant(quote=self.quote,
                source_commit=self.f.context["source_commit"], subscription_id=self.quote_fixture.subscription,
                lifecycle_id=self.request["lifecycle_id"], preflight_ledger_sequence=2,
                azure_instance=self.vm, runtime_evidence=self.runtime_path, now=self.f.now,
                root=self.quote_fixture.root, instance_transport=lambda _: compute,
                transport=None)
        self.assertEqual(http.call_args.kwargs["timeout"],
                         client.MAX_GRANT_RESPONSE_DELAY.total_seconds())
        self.assertEqual(received["training_runs_consumed"], 1)
        self.assertNotIn(self.token.encode(), self.f.io.body)
        event = self.ledger.replay(self.f.io.body)[0]["events"][-1]
        self.assertEqual(digest(event["response_envelope"]), received["grant_sha256"])

    def test_wrong_identity_or_unhealthy_watchdog_never_consumes_grant(self):
        before = self.f.io.body
        for override in ({"entra_signature_verified": False}, {"watchdog_healthy": False},
                         {"azure_control_plane_verified": False}, {"token_sha256": "e" * 64}):
            self.observation_override = override
            with self.subTest(override=override), self.assertRaises(LedgerRejected):
                self.issuer.issue(self.request)
        self.assertEqual(self.f.io.body, before)

    def test_guest_cannot_change_source_quote_vm_or_scope(self):
        before = self.f.io.body
        for key, value in (("source_commit", "e" * 40), ("quote_sha256", "e" * 64),
                           ("lifecycle_id", "another-run"), ("all_in_ceiling_usd", "6.0000"),
                           ("azure_instance_identity_token_audience", "api://other-service")):
            with self.subTest(key=key), self.assertRaises(LedgerRejected):
                self.issuer.issue({**self.request, key: value})
        self.assertEqual(self.f.io.body, before)

    def test_duplicate_nonce_or_new_nonce_cannot_issue_second_grant(self):
        self.issuer.issue(self.request)
        for nonce in (self.request["request_nonce"], "b" * 64):
            with self.assertRaises(LedgerRejected):
                self.issuer.issue({**self.request, "request_nonce": nonce})

    def test_lost_commit_response_leaves_slot_consumed(self):
        self.f.io.lose_append_response = True
        with self.assertRaises(LedgerRejected):
            self.issuer.issue(self.request)
        self.assertEqual(self.ledger.replay(self.f.io.body)[0]["sequence"], 3)
        with self.assertRaises(LedgerRejected):
            self.issuer.issue(self.request)

    def test_slow_commit_returns_no_grant_and_still_consumes_slot(self):
        original = self.ledger.transport
        def delayed(method, url, headers, body=b""):
            response = original(method, url, headers, body)
            if "comp=appendblock" in url:
                self.f.now += timedelta(seconds=61)
            return response
        self.ledger.transport = delayed
        with self.assertRaises(LedgerRejected):
            self.issuer.issue(self.request)
        self.assertEqual(self.ledger.replay(self.f.io.body)[0]["sequence"], 3)

    def test_quote_changed_during_observation_is_rejected_before_commit(self):
        original = self.issuer.verify_live_request
        def changed(**kwargs):
            observation = original(**kwargs)
            payload = deepcopy(self.quote_fixture.payload)
            payload["account_compute_hourly_usd"] = "0.4100"
            self.quote_fixture.signed(payload)
            return observation
        self.issuer.verify_live_request = changed
        with self.assertRaises(LedgerRejected):
            self.issuer.issue(self.request)
        self.assertEqual(self.ledger.replay(self.f.io.body)[0]["sequence"], 2)

    def test_json_key_order_is_irrelevant_but_extra_vm_fields_reject(self):
        reordered = json.loads(authority.canonical(self.vm))
        self.assertEqual(authority.validate_azure_instance(reordered), self.vm)
        for invalid in ({**reordered, "guest_claim": True},
                        {key: value for key, value in reordered.items() if key != "vm_id"}):
            with self.assertRaises(authority.AuthorityError):
                authority.validate_azure_instance(invalid)


if __name__ == "__main__":
    unittest.main()
