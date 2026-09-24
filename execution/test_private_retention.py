"""Expired-record cleanup must not leak expired private data or strand a runner."""

import secrets
from time import time_ns
import unittest

from execution.contracts import ExecutionError
from execution.private_store import PrivateJobStore, RetentionExpired
from execution.record_cipher import KeyMaterial, RecordCipher
from execution.runner import LocalRunner
from execution.test_support import OWNER, grant_for, make_spec


class ExpiredPrivateRecordTests(unittest.TestCase):
    def setUp(self):
        self.now = time_ns() // 1000000
        self.expiry = self.now + 120000
        self.key = KeyMaterial("test-key", secrets.token_bytes(64))
        cipher = RecordCipher(lambda _: self.key, lambda *_: self.key)
        self.store = PrivateJobStore(cipher=cipher, retention_for=lambda *_: self.expiry,
            deletion_authorized=lambda *_: True, enabled=True, clock_ms=lambda: self.now)
        self.addCleanup(self.store.close)
        self.spec = make_spec("instant")
        self.grant = grant_for(self.spec)
        self.job = self.store.create(self.grant, "expired-cleanup", self.spec)

    def test_expired_failed_attempt_releases_fence_but_keeps_budget(self):
        fence = self.store.begin(self.grant, self.job)
        attempt, _ = self.store.claim(self.grant, self.job, *fence, "answer-1")
        self.now = self.expiry
        self.store.fail_stage(OWNER, self.job, *fence, "answer-1", attempt)
        self.assertEqual(self.store.finish(OWNER, self.job, *fence, "succeeded"), "expired")
        self.assertIsNone(self.store._db.execute("SELECT runner FROM jobs WHERE id=?", (self.job,)).fetchone()[0])
        status = self.store.status(OWNER, self.job)
        self.assertFalse(status["private_data_available"])
        self.assertEqual(status["cost_reserved_microusd"], 100)
        with self.assertRaises(ExecutionError):
            self.store.result(OWNER, self.job)

    def test_expiry_during_worker_returns_failed_read_not_late_success(self):
        def worker(*_args):
            self.now = self.expiry
            return {"content": "must never publish", "tool_calls": [], "input_tokens": 10,
                    "output_tokens": 3, "debate_required": None}
        status = LocalRunner(self.store, lambda: self.grant, worker, clock_ms=lambda: self.now).run(self.job)
        self.assertEqual(status["state"], "expired")
        self.assertIsNone(status["display_name"])
        with self.assertRaises(ExecutionError):
            self.store.result(OWNER, self.job)

    def test_live_workers_are_not_abandoned_merely_because_retention_expired(self):
        fence = self.store.begin(self.grant, self.job)
        self.store.claim(self.grant, self.job, *fence, "answer-1")
        self.now = self.expiry
        with self.assertRaises(ExecutionError):
            self.store.finish(OWNER, self.job, *fence, "failed")
        self.assertEqual(self.store._db.execute("SELECT runner FROM jobs WHERE id=?", (self.job,)).fetchone()[0], fence[0])

    def test_expired_record_read_remains_owner_scoped(self):
        self.now = self.expiry
        with self.assertRaises(ExecutionError):
            self.store.status("other-owner", self.job)
        with self.assertRaises(RetentionExpired):
            self.store.server_spec(OWNER, self.job)
        self.assertTrue(self.store.status(OWNER, self.job)["retention_expired"])

    def test_cancellation_still_wins_during_expired_cleanup(self):
        fence = self.store.begin(self.grant, self.job)
        attempt, _ = self.store.claim(self.grant, self.job, *fence, "answer-1")
        self.store.request_cancel(OWNER, self.job)
        self.now = self.expiry
        self.store.fail_stage(OWNER, self.job, *fence, "answer-1", attempt)
        self.assertEqual(self.store.finish(OWNER, self.job, *fence, "succeeded"), "cancelled")

    def test_expired_replay_does_not_decrypt_private_records(self):
        self.now = self.expiry
        with self.assertRaises(RetentionExpired):
            self.store.replay(OWNER, self.job)
