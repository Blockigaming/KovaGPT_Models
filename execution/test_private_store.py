"""Encrypted journal integration using ephemeral keys and synthetic tasks only."""

import base64
from dataclasses import replace
import json
from pathlib import Path
import secrets
import socket
import tempfile
from time import time_ns
import unittest
from unittest.mock import patch

from execution.contracts import ALL_ROUTES, ExecutionError, ExecutionGrant
from execution.private_store import PrivateJobStore, RetentionExpired
from execution.record_cipher import KeyMaterial, RecordCipher, RecordProtectionError
from execution.runner import LocalRunner
from execution.store import LocalJobStore
from execution.test_support import OWNER, ModelFixture, grant_for, make_spec


class PrivateStoreTests(unittest.TestCase):
    def setUp(self):
        self.now = time_ns() // 1000000
        self.expiry = self.now + 120000
        self.key = KeyMaterial("key-1", secrets.token_bytes(64))
        self.active = self.key
        self.keys = {"key-1": self.key}
        self.allowed = True
        self.cipher = RecordCipher(lambda owner: self.active, lambda owner, key_id: self.keys[key_id])
        self.store = self.new_store()
        self.addCleanup(self.store.close)

    def new_store(self, path=":memory:"):
        return PrivateJobStore(path, cipher=self.cipher,
            retention_for=lambda *_: self.expiry, deletion_authorized=lambda *_: self.allowed,
            enabled=True, clock_ms=lambda: self.now)

    def run_job(self, route="high", store=None, name="private-test"):
        store = self.store if store is None else store
        spec = make_spec(route)
        grant = grant_for(spec)
        model = ModelFixture()
        job = store.create(grant, name, spec)
        status = LocalRunner(store, lambda: grant, model.worker()).run(job)
        return job, spec, grant, model, status

    def test_every_explicit_route_runs_with_encrypted_prompts_and_stage_artifacts(self):
        for route in sorted(ALL_ROUTES):
            with self.subTest(route=route):
                job, spec, grant, model, status = self.run_job(route, name=route)
                self.assertEqual(status["state"], "succeeded")
                self.assertEqual(self.store.server_spec(OWNER, job).encoded, spec.encoded)
                self.assertEqual(self.store.result(OWNER, job)["content"], "Kova final response")
                raw = self.store._db.execute("SELECT spec FROM jobs WHERE id=?", (job,)).fetchone()[0]
                self.assertNotIn(b"PRIVATE input text", raw)
                self.assertNotIn(b"You are Kova", raw)
                self.assertEqual(json.loads(raw)["algorithm"], "AES-256-SIV")
                for row in self.store._db.execute("SELECT result FROM stages WHERE job=? AND result IS NOT NULL", (job,)):
                    self.assertNotIn(b"PRIVATE stage result", row[0])
                    self.assertNotIn(b"Kova final response", row[0])

    def test_cipher_authenticates_owner_job_slot_and_retention_deadline(self):
        data = b"synthetic-private-message"
        value = self.cipher.seal(OWNER, "job-1", "stage-1", self.expiry, data)
        self.assertEqual(self.cipher.open(OWNER, "job-1", "stage-1", self.expiry, value), data)
        for args in (("other", "job-1", "stage-1", self.expiry),
                     (OWNER, "job-2", "stage-1", self.expiry), (OWNER, "job-1", "stage-2", self.expiry),
                     (OWNER, "job-1", "stage-1", self.expiry + 1)):
            with self.assertRaises(RecordProtectionError):
                self.cipher.open(*args, value)

    def test_repeated_seals_use_independent_nonces_and_hide_plaintext(self):
        outputs = [self.cipher.seal(OWNER, "job-1", "slot", self.expiry, b"private") for _ in range(50)]
        self.assertEqual(len(set(outputs)), 50)
        self.assertEqual(len({json.loads(value)["nonce"] for value in outputs}), 50)
        self.assertTrue(all(b'"private"' not in value for value in outputs))

    def test_cipher_rejects_modified_bytes_algorithm_and_unknown_key(self):
        good = self.cipher.seal(OWNER, "job", "slot", self.expiry, b"secret")
        for changed in ({"algorithm": "none"}, {"version": True}, {"key_id": "other"},
                        {"nonce": base64.b64encode(b"x" * 16).decode()}, {"ciphertext": base64.b64encode(b"x" * 40).decode()}):
            from execution.contracts import canonical
            data = canonical(json.loads(good) | changed)
            with self.assertRaises(RecordProtectionError):
                self.cipher.open(OWNER, "job", "slot", self.expiry, data)

    def test_key_service_failure_does_not_expose_private_data(self):
        def unavailable(*_):
            raise RuntimeError("KEY SECRET VALUE")
        cipher = RecordCipher(unavailable, unavailable)
        with self.assertRaises(RecordProtectionError) as caught:
            cipher.seal(OWNER, "job", "slot", self.expiry, b"PRIVATE")
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertNotIn(self.key.key.hex(), repr(self.key))

    def test_cannot_use_invalid_key_sizes_or_guess_key_ids(self):
        for data in (b"", b"x" * 32, bytearray(64), "x" * 64):
            with self.assertRaises(RecordProtectionError):
                KeyMaterial("key", data)
        with self.assertRaises(ExecutionError):
            KeyMaterial("../secret-key", secrets.token_bytes(64))

    def test_disabled_private_store_does_not_create_a_file_or_read_a_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "disabled.sqlite3"
            with self.assertRaises(RecordProtectionError):
                PrivateJobStore(path, cipher=self.cipher, retention_for=lambda *_: self.expiry,
                                deletion_authorized=lambda *_: True)
            self.assertFalse(path.exists())

    def test_unencrypted_existing_jobs_require_explicit_migration_instead_of_silent_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.sqlite3"
            plain = LocalJobStore(path)
            spec = make_spec()
            plain.create(grant_for(spec), "plain-job", spec)
            plain.close()
            with self.assertRaises(RecordProtectionError):
                self.new_store(path)

    def test_private_disk_journal_reopens_and_resumes_without_plaintext_payloads(self):
        spec = make_spec("max")
        grant = grant_for(spec)
        model = ModelFixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encrypted.sqlite3"
            store = self.new_store(path)
            job = store.create(grant, "resume", spec)
            try:
                self.assertEqual(LocalRunner(store, lambda: grant, model.worker()).run(job, max_stages=2)["state"], "paused")
            finally:
                store.close()
            self.assertNotIn(b"PRIVATE input text", path.read_bytes())
            self.assertNotIn(b"PRIVATE stage result", path.read_bytes())
            reopened = self.new_store(path)
            try:
                self.assertEqual(LocalRunner(reopened, lambda: grant, model.worker()).run(job)["state"], "succeeded")
                self.assertEqual(len(model.calls), len(set(model.calls)))
                self.assertEqual(reopened.result(OWNER, job)["content"], "Kova final response")
            finally:
                reopened.close()
            self.assertNotIn(b"Kova final response", path.read_bytes())

    def test_current_key_rotation_is_atomic_and_preserves_spec_and_results(self):
        job, spec, grant, _, _ = self.run_job()
        before = self.store._db.execute("SELECT spec FROM jobs WHERE id=?", (job,)).fetchone()[0]
        new = KeyMaterial("key-2", secrets.token_bytes(64))
        self.keys[new.key_id] = new
        self.active = new
        with self.assertRaises(RecordProtectionError):
            self.store.rotate_job_key(grant, job)
        self.store.rotate_job_key(grant, job, administration_authorized=True)
        self.keys.pop("key-1")
        after = self.store._db.execute("SELECT spec FROM jobs WHERE id=?", (job,)).fetchone()[0]
        self.assertNotEqual(before, after)
        self.assertEqual(self.store.server_spec(OWNER, job).encoded, spec.encoded)
        self.assertEqual(self.store.result(OWNER, job)["content"], "Kova final response")
        self.assertEqual(self.store._expiry(job), self.expiry)

    def test_rotation_failure_rolls_back_all_ciphertext_changes(self):
        job, _, grant, _, _ = self.run_job()
        before = self.store._db.execute("SELECT spec FROM jobs WHERE id=?", (job,)).fetchone()[0]
        original = self.cipher.seal
        calls = []
        def fail_later(*args):
            calls.append(1)
            if len(calls) == 2:
                raise RecordProtectionError("fixture failure")
            return original(*args)
        with patch.object(self.cipher, "seal", side_effect=fail_later):
            with self.assertRaises(RecordProtectionError):
                self.store.rotate_job_key(grant, job, administration_authorized=True)
        self.assertEqual(self.store._db.execute("SELECT spec FROM jobs WHERE id=?", (job,)).fetchone()[0], before)
        self.assertEqual(self.store.result(OWNER, job)["content"], "Kova final response")

    def test_revoked_keys_prevent_decryption_without_plaintext_fallback(self):
        job, _, _, _, _ = self.run_job()
        self.keys.clear()
        with self.assertRaises(RecordProtectionError):
            self.store.server_spec(OWNER, job)
        with self.assertRaises(RecordProtectionError):
            self.store.result(OWNER, job)

    def test_retention_expiry_blocks_private_reads_and_does_not_extend_on_retry(self):
        job, spec, grant, _, _ = self.run_job()
        self.now = self.expiry
        with self.assertRaises(RetentionExpired):
            self.store.server_spec(OWNER, job)
        with self.assertRaises(RetentionExpired):
            self.store.result(OWNER, job)
        self.expiry += 100000
        with self.assertRaises(RetentionExpired):
            self.store.create(grant, "private-test", spec)

    def test_retention_metadata_cannot_be_edited_to_extend_read_access(self):
        job, _, _, _, _ = self.run_job()
        self.store._db.execute("UPDATE private_job_policy SET retain_until_ms=retain_until_ms+1000 WHERE job=?", (job,))
        with self.assertRaises(RecordProtectionError):
            self.store.server_spec(OWNER, job)

    def test_named_owner_deletion_removes_artifacts_and_preserves_tombstone(self):
        job, spec, grant, _, _ = self.run_job()
        other, _, _, _, _ = self.run_job(name="other-job")
        result = self.store.delete_job(grant, job, reason="owner_request")
        self.assertTrue(result["deleted"])
        self.assertFalse(result["physical_backup_erasure_claimed"])
        self.assertFalse(result["billing_refund_claimed"])
        self.assertEqual(self.store._db.execute("SELECT COUNT(*) FROM stages WHERE job=?", (job,)).fetchone()[0], 0)
        self.assertEqual(self.store._db.execute("SELECT COUNT(*) FROM events WHERE job=?", (job,)).fetchone()[0], 0)
        with self.assertRaises(RecordProtectionError):
            self.store.create(grant, "private-test", spec)
        self.assertEqual(self.store.result(OWNER, other)["content"], "Kova final response")

    def test_cross_owner_deletion_and_read_are_rejected(self):
        job, _, grant, _, _ = self.run_job()
        other = replace(grant, owner_id="other-owner")
        with self.assertRaises(ExecutionError):
            self.store.delete_job(other, job, reason="owner_request")
        with self.assertRaises(ExecutionError):
            self.store.server_spec(other.owner_id, job)
        self.assertEqual(self.store.result(OWNER, job)["content"], "Kova final response")

    def test_truthy_delete_permission_does_not_authorize_erasure(self):
        job, _, grant, _, _ = self.run_job()
        for permission in ("true", 1, None, {"authorized": True}):
            self.allowed = permission
            with self.assertRaises(RecordProtectionError):
                self.store.delete_job(grant, job, reason="owner_request")

    def test_retention_deletion_requires_expiry_but_not_access_to_plaintext_or_old_key(self):
        job, _, grant, _, _ = self.run_job()
        with self.assertRaises(RecordProtectionError):
            self.store.delete_job(grant, job, reason="retention_expired")
        self.now = self.expiry
        self.keys.clear()
        self.assertTrue(self.store.delete_job(grant, job, reason="retention_expired")["deleted"])

    def test_deleting_a_running_or_paused_job_is_forbidden(self):
        spec = make_spec()
        grant = grant_for(spec)
        job = self.store.create(grant, "not-terminal", spec)
        for state in ("queued", "running", "paused"):
            self.store._db.execute("UPDATE jobs SET state=? WHERE id=?", (state, job))
            with self.assertRaises(RecordProtectionError):
                self.store.delete_job(grant, job, reason="owner_request")

    def test_uncertain_started_work_requires_a_verified_quiescence_receipt(self):
        spec = make_spec()
        grant = grant_for(spec)
        job = self.store.create(grant, "uncertain", spec)
        fence = self.store.begin(grant, job)
        self.store.claim(grant, job, *fence, spec.stages[0].id)
        self.store.recover_abandoned(OWNER, job, runner=fence[0], epoch=fence[1], supervisor_confirmed_stopped=True)
        with self.assertRaises(ExecutionError):
            self.store.delete_job(grant, job, reason="owner_request")
        self.assertTrue(self.store.delete_job(grant, job, reason="owner_request", quiescence_receipt="verified-stop-1")["deleted"])

    def test_deletion_remains_available_to_authenticated_owner_after_plan_revocation(self):
        job, _, grant, _, _ = self.run_job()
        self.assertTrue(self.store.delete_job(replace(grant, execution_authorized=False), job, reason="owner_request")["deleted"])

    def test_ciphertext_cannot_be_moved_between_two_jobs_with_the_same_spec(self):
        first, spec, grant, _, _ = self.run_job()
        second = self.store.create(grant, "same-spec-different-job", spec)
        raw = self.store._db.execute("SELECT spec FROM jobs WHERE id=?", (first,)).fetchone()[0]
        self.store._db.execute("UPDATE jobs SET spec=? WHERE id=?", (raw, second))
        with self.assertRaises(RecordProtectionError):
            self.store.server_spec(OWNER, second)

    def test_private_stage_ciphertext_cannot_be_substituted_into_another_stage(self):
        job, _, _, _, _ = self.run_job()
        rows = self.store._db.execute("SELECT * FROM stages WHERE job=? ORDER BY ordinal", (job,)).fetchall()
        self.store._db.execute("UPDATE stages SET result=?,checksum=? WHERE job=? AND id=?",
                               (rows[0]["result"], rows[0]["checksum"], job, rows[-1]["id"]))
        with self.assertRaises(RecordProtectionError):
            self.store.result(OWNER, job)

    def test_invalid_retention_does_not_write_plaintext_or_partial_job(self):
        spec = make_spec()
        grant = grant_for(spec)
        self.expiry = spec.limits.deadline_unix_ms - 1
        with self.assertRaises(RecordProtectionError):
            self.store.create(grant, "invalid-retention", spec)
        self.assertEqual(self.store._db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)

    def test_private_payloads_never_enter_sql_parameters(self):
        spec = make_spec()
        grant = grant_for(spec)
        statements = []
        self.store._db.set_trace_callback(statements.append)
        job = self.store.create(grant, "sql-trace", spec)
        LocalRunner(self.store, lambda: grant, ModelFixture().worker()).run(job)
        self.assertNotIn("PRIVATE input text", "\n".join(statements))
        self.assertNotIn("PRIVATE stage result", "\n".join(statements))
        self.assertNotIn("Kova final response", "\n".join(statements))

    def test_privacy_integration_starts_no_network_or_key_service_by_itself(self):
        with patch.object(socket, "socket", side_effect=AssertionError("network called")):
            _, _, _, _, status = self.run_job("instant")
        self.assertEqual(status["state"], "succeeded")
