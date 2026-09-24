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
            "not_before_utc": "2026-09-24T12:00:00Z", "grant_deadline_utc": "2026-09-24T13:15:00Z"}
        self.now = datetime(2026, 9, 24, 12, 1, tzinfo=timezone.utc)
        self.key = Ed25519PrivateKey.generate()
        self.verifier = Ed25519PrivateKey.generate()
        self.io = BlobModel()
        self.subject = self.make()
        self.health = {"kind": "watchdog_health", "family": "kova-cosmo", "healthy": True,
            "rule_id": "independently-tested-rule", "observed_at_utc": "2026-09-24T12:00:00Z",
            "expires_at_utc": "2026-09-24T12:05:00Z"}
        self.cost = {"kind": "cost_admission", "family": "kova-cosmo",
            "account_price_verified": True, "quote_sha256": "a" * 64,
            "remaining_budget_usd": "3.3000", "observed_at_utc": "2026-09-24T12:00:00Z",
            "expires_at_utc": "2026-09-24T12:05:00Z"}
        self.grant = {"kind": "training_grant", "family": "kova-cosmo"}

    def make(self, context=None):
        return ledger.ControllerLedger(context=context or self.context, signing_key=self.key,
            transport=self.io, preservation_public_key=self.verifier.public_key().public_bytes_raw(),
            cleanup_public_key=self.verifier.public_key().public_bytes_raw(), clock=lambda: self.now)

    def prepared(self):
        self.subject.initialize()
        self.subject.append(self.health, expected_sequence=0)
        self.subject.append(self.cost, expected_sequence=1)

    def test_restart_replays_history_and_cannot_grant_twice(self):
        self.prepared()
        record = self.subject.append(self.grant, expected_sequence=2)
        self.assertEqual(record["payload"]["sequence"], 3)
        self.subject.public_key.verify(bytes.fromhex(record["signature"]),
                                       authority.canonical(record["payload"]))
        restarted = self.make()
        self.assertEqual(restarted.replay(self.io.body)[0]["family_order"], ["kova-cosmo"])
        with self.assertRaises(contract.ContractError):
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
        with self.assertRaises(contract.ContractError):
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
        self.now += timedelta(days=1)
        stamp = self.now.strftime("%Y-%m-%dT%H:%M:%SZ")
        payload = {"schema_version": 1, "ledger_sequence": 0, **self.context["lifecycle"],
            "pilot_remaining_resources": [], "watchdog_remaining_resources": [],
            "subscription_scoped_residual_resources": [], "pilot_deleted_at_utc": stamp,
            "watchdog_deleted_at_utc": stamp, "verified_at_utc": stamp,
            "cost_posting_complete": True, "final_cost_usd": "0.0000",
            "cost_evidence_sha256": "b" * 64, "evidence_uri": "https://evidence.example.test/terminal",
            "immutable_evidence_version": "version-one", "outside_both_groups": True,
            "verification_succeeded": True, "unpreserved_grants": []}
        receipt = {"payload": payload, "signature_ed25519_hex":
                   self.verifier.sign(authority.canonical(payload)).hex()}
        self.subject.append({"kind": "cleanup_terminal", "cleanup_receipt": receipt}, expected_sequence=0)
        self.assertIs(self.subject.replay(self.io.body)[0]["terminal"], True)
        with self.assertRaises((ledger.LedgerRejected, contract.ContractError)):
            self.subject.append(self.health, expected_sequence=1)

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
