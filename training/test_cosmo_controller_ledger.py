"""Durability/concurrency regressions against an in-process Blob REST model.

No Azure credentials, requests, resources, model files, or training are used.
The model enforces lease IDs, ETags and append offsets rather than simply
returning success to the implementation.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from io import StringIO
import threading
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from training import cosmo_controller_ledger as ledger
from training import cosmo_lifecycle_authority as authority
from training import three_family_contract as contract


class BlobModel:
    def __init__(self):
        self.lock = threading.Lock()
        self.body = None
        self.lease = None
        self.version = 0
        self.calls = []
        self.policy = {"state": "Locked", "immutabilityPeriodSinceCreationInDays": 1,
                       "allowProtectedAppendWrites": True, "allowProtectedAppendWritesAll": False}
        self.props = {"publicAccess": "None", "hasImmutabilityPolicy": True,
                      "immutableStorageWithVersioning": {"enabled": False}}
        self.lose_append_response = False
        self.corrupt_readback = False

    @property
    def etag(self):
        return '"' + str(self.version) + '"'

    def __call__(self, method, url, headers, body=b""):
        with self.lock:
            self.calls.append((method, url, deepcopy(headers)))
            if "management.azure.com" in url:
                props = self.policy if "immutabilityPolicies" in url else self.props
                return ledger.Response(200, {}, json.dumps({"properties": props}).encode())
            if "comp=lease" in url:
                if self.body is None:
                    return ledger.Response(404, {})
                if headers["x-ms-lease-action"] == "acquire":
                    if self.lease:
                        return ledger.Response(409, {})
                    self.lease = headers["x-ms-proposed-lease-id"]
                    return ledger.Response(201, {"x-ms-lease-id": self.lease})
                if headers["x-ms-lease-id"] != self.lease:
                    return ledger.Response(412, {})
                self.lease = None
                return ledger.Response(200, {})
            if method == "GET":
                if self.body is None:
                    return ledger.Response(404, {})
                if self.lease != headers.get("x-ms-lease-id"):
                    return ledger.Response(412, {})
                raw = self.body
                if self.corrupt_readback and self.version > 1:
                    raw += b"corrupt"
                return ledger.Response(200, {"etag": self.etag,
                    "x-ms-blob-type": "AppendBlob"}, raw)
            if "comp=appendblock" in url:
                if (not self.lease or self.lease != headers.get("x-ms-lease-id") or
                    headers.get("If-Match") != self.etag or
                    headers.get("x-ms-blob-condition-appendpos") != str(len(self.body)) or
                    len(self.body) + len(body) > int(headers["x-ms-blob-condition-maxsize"])):
                    return ledger.Response(412, {})
                offset = len(self.body)
                self.body += body
                self.version += 1
                if self.lose_append_response:
                    self.lose_append_response = False
                    raise ledger.LedgerRejected("synthetic lost response after commit")
                return ledger.Response(201, {"x-ms-blob-append-offset": str(offset)})
            if headers.get("If-None-Match") != "*" or self.body is not None:
                return ledger.Response(412, {})
            self.body = b""
            return ledger.Response(201, {})


class ControllerLedgerTests(unittest.TestCase):
    def setUp(self):
        prefix = "/subscriptions/12345678-1234-1234-1234-123456789abc/resourceGroups/"
        self.context = {"source_commit": "d" * 40,
            "lifecycle": {"lifecycle_id": "cosmo42-once", "ledger_id": "cosmo42-ledger",
                "admission_scope": "cosmo-only", "pilot_resource_group_id": prefix + "pilot",
                "watchdog_resource_group_id": prefix + "watchdog"},
            "storage_resource_group_id": prefix + "evidence", "storage_account": "kovatestledger",
            "container": "cosmo42", "blob": "pilot.jsonl", "retention_days": 1,
            "artifact_container": "cosmo-adapters",
            "external_archive": {"uri": "https://evidence.example.test/archived-ledger",
                "retention_days": 30, "maximum_bytes": 1048576},
            "not_before_utc": "2026-09-24T12:00:00Z", "grant_deadline_utc": "2026-09-24T13:15:00Z"}
        self.now = datetime(2026, 9, 24, 12, 1, tzinfo=timezone.utc)
        self.key = Ed25519PrivateKey.generate()
        self.verifier = Ed25519PrivateKey.generate()
        self.cleanup_verifier = Ed25519PrivateKey.generate()
        self.io = BlobModel()
        self.subject = self.make()
        self.health = {"kind": "watchdog_health", "family": "kova-cosmo", "healthy": True,
            "rule_id": "independently-tested-rule", "observed_at_utc": "2026-09-24T12:00:00Z",
            "expires_at_utc": "2026-09-24T12:05:00Z"}
        self.cost = {"kind": "cost_admission", "family": "kova-cosmo",
            "account_price_verified": True, "quote_sha256": "a" * 64,
            "remaining_budget_usd": "3.3000", "observed_at_utc": "2026-09-24T12:00:00Z",
            "expires_at_utc": "2026-09-24T12:05:00Z"}
        payload = {"schema_version": 1, "kind": "kova_cosmo_qlora_training_grant",
            "issuer": authority.ISSUER, "source_commit": self.context["source_commit"],
            "subscription_id": "12345678-1234-1234-1234-123456789abc",
            "quote_sha256": "a" * 64, "lifecycle_id": "cosmo42-once",
            "preflight_ledger_sequence": 2, "ledger_sequence": 3,
            "network_evidence_sha256": "b" * 64, "azure_identity_token_sha256": "c" * 64,
            "request_nonce": "e" * 64, "ledger_commit_id": "12345678-1234-1234-1234-123456789abc",
            "grant_id": "12345678-1234-1234-1234-123456789abd", "ledger_append_only": True,
            "ledger_status": "grant_committed_before_response", "azure_instance": {
                "resource_id": prefix + "pilot/providers/Microsoft.Compute/virtualMachines/kova-t4-test01",
                "vm_id": "12345678-1234-1234-1234-123456789abd",
                "system_assigned_identity_principal_id": "12345678-1234-1234-1234-123456789abe"},
            "issued_at_utc": "2026-09-24T12:01:00Z", "expires_at_utc": "2026-09-24T13:15:00Z",
            "allocation_deadline_utc": "2026-09-24T13:30:00Z",
            "watchdog_cleanup_trigger_utc": "2026-09-24T13:15:00Z",
            "training_runs_consumed": 1, "all_in_reserved_usd": "3.1890",
            "all_in_ceiling_usd": "3.3000", "watchdog_healthy": True,
            "cleanup_scope_verified": True, "deployment_authorized": False}
        self.grant = {"kind": "training_grant", "family": "kova-cosmo", "quote_sha256": "a" * 64,
            "request_sha256": "f" * 64, "response_envelope": {"payload": payload,
                "signature": self.key.sign(authority.canonical(payload)).hex()}}

    def make(self, context=None):
        return ledger.ControllerLedger(context=context or self.context, signing_key=self.key,
            transport=self.io, preservation_public_key=self.verifier.public_key().public_bytes_raw(),
            cleanup_public_key=self.cleanup_verifier.public_key().public_bytes_raw(), clock=lambda: self.now)

    def test_cleanup_and_preservation_verifiers_must_be_distinct(self):
        with self.assertRaisesRegex(ledger.LedgerRejected, "distinct verifiers"):
            ledger.ControllerLedger(context=self.context, signing_key=self.key,
                transport=self.io,
                preservation_public_key=self.verifier.public_key().public_bytes_raw(),
                cleanup_public_key=self.verifier.public_key().public_bytes_raw(),
                clock=lambda: self.now)

    def test_artifact_container_must_be_a_valid_distinct_blob_container(self):
        for name in ("cosmo--adapters", "Invalid", "cosmo42"):
            with self.subTest(name=name), self.assertRaisesRegex(
                    ledger.LedgerRejected, "invalid artifact container"):
                self.make({**self.context, "artifact_container": name})

    def prepared(self):
        self.subject.initialize()
        self.subject.append(self.health, expected_sequence=0)
        self.subject.append(self.cost, expected_sequence=1)

    def test_bare_or_tampered_grant_cannot_consume_the_slot(self):
        self.prepared()
        before = self.io.body
        bare = {"kind": "training_grant", "family": "kova-cosmo"}
        tampered = deepcopy(self.grant)
        tampered["response_envelope"]["payload"]["request_nonce"] = "0" * 64
        for bad in (bare, {**self.grant, "extra": True}, tampered):
            with self.subTest(bad=bad), self.assertRaises(ledger.LedgerRejected):
                self.subject.append(bad, expected_sequence=2)
            self.assertEqual(before, self.io.body)
        self.subject.append(self.grant, expected_sequence=2)

    def test_restart_replays_history_and_cannot_grant_twice(self):
        self.prepared()
        record = self.subject.append(self.grant, expected_sequence=2)
        self.assertEqual(record["payload"]["sequence"], 3)
        self.subject.public_key.verify(bytes.fromhex(record["signature"]),
                                       authority.canonical(record["payload"]))
        restarted = self.make()
        self.assertEqual(restarted.replay(self.io.body)[0]["family_order"], ["kova-cosmo"])
        with self.assertRaises((ledger.LedgerRejected, contract.ContractError)):
            restarted.append(self.grant, expected_sequence=3)
        self.assertIsNone(self.io.lease)

    def test_competing_controllers_commit_one_grant(self):
        self.prepared()
        barrier = threading.Barrier(2)
        def attempt(_):
            other = self.make()
            barrier.wait(timeout=5)
            try:
                other.append(self.grant, expected_sequence=2)
                return "committed"
            except (ledger.LedgerRejected, contract.ContractError):
                return "rejected"
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(attempt, range(2)))
        self.assertCountEqual(outcomes, ["committed", "rejected"])
        self.assertEqual(self.subject.replay(self.io.body)[0]["sequence"], 3)

    def test_lost_response_preserves_consumed_grant_and_blocks_retry(self):
        self.prepared()
        self.io.lose_append_response = True
        with self.assertRaises(ledger.LedgerRejected):
            self.subject.append(self.grant, expected_sequence=2)
        self.assertEqual(self.make().replay(self.io.body)[0]["sequence"], 3)
        with self.assertRaises(ledger.LedgerRejected):
            self.make().append(self.grant, expected_sequence=2)
        self.assertIsNone(self.io.lease)

    def test_no_automatic_initialization_of_missing_or_empty_ledger(self):
        with self.assertRaises(ledger.LedgerRejected):
            self.subject.append(self.health, expected_sequence=0)
        self.assertIsNone(self.io.body)
        self.io.body = b""
        with self.assertRaises(ledger.LedgerRejected):
            self.subject.initialize()
        with self.assertRaises(ledger.LedgerRejected):
            self.subject.append(self.health, expected_sequence=0)
        self.assertEqual(self.io.body, b"")

    def test_unsigned_modified_truncated_and_reordered_history_fail(self):
        self.prepared()
        original = self.io.body
        lines = original.splitlines(keepends=True)
        changed = json.loads(lines[-1])
        changed["payload"]["event"]["remaining_budget_usd"] = "6.0000"
        for bad in (original[:-1], b"".join(lines[::-1]),
                    b"".join(lines[:-1]) + authority.canonical(changed) + b"\n",
                    b"".join(lines[1:])):
            with self.subTest(raw=bad[:20]), self.assertRaises(ledger.LedgerRejected):
                self.subject.replay(bad)

    def test_wrong_context_or_storage_protection_blocks_all_writes(self):
        for change in ({"state": "Unlocked"}, {"allowProtectedAppendWrites": False},
                       {"allowProtectedAppendWritesAll": True},
                       {"immutabilityPeriodSinceCreationInDays": 2}):
            original = deepcopy(self.io.policy)
            self.io.policy.update(change)
            with self.subTest(change=change), self.assertRaises(ledger.LedgerRejected):
                self.subject.initialize()
            self.assertFalse(any(method == "PUT" for method, *_ in self.io.calls))
            self.io.policy = original
        self.io.props["immutableStorageWithVersioning"]["enabled"] = True
        with self.assertRaises(ledger.LedgerRejected):
            self.subject.initialize()
        self.io.props["immutableStorageWithVersioning"]["enabled"] = False
        self.prepared()
        other_context = deepcopy(self.context)
        other_context["source_commit"] = "e" * 40
        with self.assertRaises(ledger.LedgerRejected):
            self.make(other_context).append(self.grant, expected_sequence=2)

    def test_expired_admission_and_other_family_never_append(self):
        self.prepared()
        before = self.io.body
        self.now += timedelta(minutes=5)
        with self.assertRaises((ledger.LedgerRejected, contract.ContractError)):
            self.subject.append(self.grant, expected_sequence=2)
        self.now = authority.timestamp(self.context["grant_deadline_utc"])
        with self.assertRaises(ledger.LedgerRejected):
            self.subject.append(self.grant, expected_sequence=2)
        with self.assertRaises(ledger.LedgerRejected):
            self.subject.append({**self.grant, "family": "kova-orion"}, expected_sequence=2)
        self.assertEqual(self.io.body, before)

    def test_readback_failure_never_returns_success(self):
        self.subject.initialize()
        self.io.corrupt_readback = True
        with self.assertRaises(ledger.LedgerRejected):
            self.subject.append(self.health, expected_sequence=0)
        self.io.corrupt_readback = False
        self.assertEqual(self.subject.replay(self.io.body)[0]["sequence"], 1)

    def test_cleanup_after_deadline_and_terminal_refuses_new_events(self):
        self.subject.initialize()
        self.now += timedelta(days=1, seconds=3)
        stamp = self.now.strftime("%Y-%m-%dT%H:%M:%SZ")
        export_stamp = (self.now - timedelta(seconds=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        deleted_stamp = (self.now - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        archive = self.io.body
        _, head, last_time = self.subject.replay(archive)
        export_payload = {"kind": "kova_cosmo_ledger_archive_v1",
            "context_sha256": self.subject.context_sha256, "blob_url": self.subject.blob_url,
            "archive_sha256": ledger.hashlib.sha256(archive).hexdigest(),
            "archive_bytes": len(archive), "head_sha256": head, "ledger_sequence": 0,
            "blob_etag": self.io.etag,
            "observed_at_utc": export_stamp,
            "immutable_archive_uri": "https://evidence.example.test/archived-ledger",
            "immutable_archive_version": "verified-v1", "archive_retention_days": 30}
        def signed(value, signing_key=None):
            return {"payload": value, "signature_ed25519_hex":
                    (signing_key or self.cleanup_verifier).sign(authority.canonical(value)).hex()}
        export = signed(export_payload)
        payload = {"schema_version": 1, "ledger_sequence": 0, **self.context["lifecycle"],
            "pilot_remaining_resources": [], "watchdog_remaining_resources": [],
            "subscription_scoped_residual_resources": [], "pilot_deleted_at_utc": deleted_stamp,
            "watchdog_deleted_at_utc": deleted_stamp, "verified_at_utc": stamp,
            "cost_posting_complete": True, "final_cost_usd": "0.0000",
            "cost_evidence_sha256": "b" * 64, "evidence_uri": "https://evidence.example.test/terminal",
            "immutable_evidence_version": "version-one", "outside_both_groups": True,
            "verification_succeeded": True, "unpreserved_grants": []}
        receipt = signed(payload, self.cleanup_verifier)
        event = {"kind": "cleanup_terminal", "cleanup_receipt": receipt}
        with self.assertRaisesRegex(ledger.LedgerRejected, 'after deleting'):
            self.subject.append(event, expected_sequence=0)
        self.assertEqual(self.io.body, archive)
        # The independently attested copy exists before Azure storage deletion;
        # final cost and zero-resource evidence are signed only afterwards.
        self.io.body = None
        final_payload = {"kind": "kova_cosmo_external_terminal_v1",
            "export_sha256": ledger.digest(export),
            "context_sha256": self.subject.context_sha256,
            "storage_resource_group_id": self.context["storage_resource_group_id"],
            "ledger_deleted_at_utc": deleted_stamp, "conditional_delete_etag": self.io.etag,
            "ledger_remaining_resources": [],
            "cleanup_event": event}
        final = signed(final_payload)
        verifier = ledger.ControllerLedger(context=self.context, transport=None,
            signing_public_key=self.key.public_key().public_bytes_raw(),
            preservation_public_key=self.verifier.public_key().public_bytes_raw(),
            cleanup_public_key=self.cleanup_verifier.public_key().public_bytes_raw(),
            clock=lambda: self.now)
        self.assertTrue(verifier.verify_external_terminal(archive, export, final)["terminal"])
        with self.assertRaisesRegex(ledger.LedgerRejected, "untrusted external cleanup"):
            verifier.verify_external_terminal(archive, signed(export_payload, self.verifier), final)
        premature_export = signed({**export_payload,
            "observed_at_utc": last_time.strftime("%Y-%m-%dT%H:%M:%SZ")})
        premature_final = signed({**final_payload,
            "export_sha256": ledger.digest(premature_export),
            "ledger_deleted_at_utc": (last_time + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")})
        with self.assertRaisesRegex(ledger.LedgerRejected, "retention has not expired"):
            verifier.verify_external_terminal(archive, premature_export, premature_final)
        premature_cost_receipt = signed({**payload, "verified_at_utc": deleted_stamp}, self.cleanup_verifier)
        premature_cost_final = signed({**final_payload,
            "ledger_deleted_at_utc": stamp,
            "cleanup_event": {"kind": "cleanup_terminal", "cleanup_receipt": premature_cost_receipt}})
        with self.assertRaisesRegex(ledger.LedgerRejected, "reconciliation must follow ledger deletion"):
            verifier.verify_external_terminal(archive, export, premature_cost_final)
        self.assertIsNone(self.io.body)
        with self.assertRaisesRegex(ledger.LedgerRejected, 'archive verifier cannot append'):
            verifier.append(self.health, expected_sequence=0)
        with self.assertRaisesRegex(ledger.LedgerRejected, 'archive verifier cannot initialize'):
            verifier.initialize()
        bad_export = signed({**export_payload, 'archive_sha256': '0' * 64})
        wrong_archive = signed({**export_payload,
            'immutable_archive_uri': 'https://other.example.test/archived-ledger'})
        short_archive = signed({**export_payload, 'archive_retention_days': 1})
        forged_export = deepcopy(export)
        forged_export['signature_ed25519_hex'] = '0' * 128
        for archived, attestation, termination in (
            (archive[:-1], export, final),
            (archive, bad_export, final),
            (archive, wrong_archive, final),
            (archive, short_archive, final),
            (archive, forged_export, final),
            (archive, export, signed({**final_payload, 'ledger_remaining_resources': ['storage']})),
            (archive, export, signed({**final_payload, 'ledger_deleted_at_utc': '2026-09-24T12:00:00Z'})),
            (archive, export, signed({**final_payload, 'export_sha256': '0' * 64})),
            (archive, export, signed({**final_payload, 'conditional_delete_etag': '"changed"'})),
            (archive, export, signed({**final_payload, 'cleanup_event': {
                'kind': 'cleanup_terminal', 'cleanup_receipt': signed({**payload, 'cost_posting_complete': False}, self.cleanup_verifier)}})),
        ):
            with (self.subTest(archived=archived[:8], attestation=attestation, termination=termination),
                  self.assertRaises((ledger.LedgerRejected, contract.ContractError))):
                verifier.verify_external_terminal(archived, attestation, termination)

    def test_production_cli_remains_closed_without_network(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
            ledger.main(["--execute"])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(self.io.calls, [])

    def test_transport_rejects_untrusted_host_before_accessing_credentials(self):
        calls = []
        transport = ledger.AzureBlobIO(account="kovatestledger", token_for=lambda resource: calls.append(resource))
        for url in ("https://untrusted.example/", "http://kovatestledger.blob.core.windows.net/",
                    "https://management.azure.com@untrusted.example/",
                    "https://kovatestledger.blob.core.windows.net:444/"):
            with self.subTest(url=url), self.assertRaises(ledger.LedgerRejected):
                transport("GET", url, {})
        with self.assertRaises(ledger.LedgerRejected):
            transport("PUT", "https://management.azure.com/", {})
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
