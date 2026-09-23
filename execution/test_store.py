from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from execution.contracts import ExecutionBusy, ExecutionError, ExecutionIntegrityError, canonical
from execution.store import LocalJobStore
from execution.test_support import OWNER, SyntheticAdapterTestCase, grant_for, make_spec


def result(content="PRIVATE draft", *, debate=None):
    return {"content": content, "tool_calls": [], "input_tokens": 10,
            "output_tokens": 3, "debate_required": debate}


class JobStoreTests(SyntheticAdapterTestCase):
    def setUp(self):
        super().setUp()
        self.spec = make_spec("high")
        self.grant = grant_for(self.spec)
        self.store = LocalJobStore()
        self.job = self.store.create(self.grant, "idempotency-1", self.spec)

    def tearDown(self):
        self.store.close()

    def start(self):
        return self.store.begin(self.grant, self.job)

    def claim(self, stage=None, fence=None):
        runner, epoch = fence or self.start()
        stage = stage or self.spec.stages[0].id
        return (runner, epoch), self.store.claim(self.grant, self.job, runner, epoch, stage)

    def test_idempotent_create_preserves_job_and_budget(self):
        again = self.store.create(self.grant, "idempotency-1", self.spec)
        self.assertEqual(again, self.job)
        self.assertEqual(self.store.status(OWNER, self.job)["sequence"], 1)
        changed = self.spec.snapshot()
        changed["limits"]["deadline_unix_ms"] += 10000
        from execution.contracts import ExecutionSpec
        with self.assertRaisesRegex(ExecutionError, "idempotency"):
            self.store.create(self.grant, "idempotency-1", ExecutionSpec(canonical(changed)))
        other = self.store.create(replace(self.grant, owner_id="other"), "idempotency-1", self.spec)
        self.assertNotEqual(other, self.job)

    def test_owner_is_required_for_status_result_replay_cancel_and_spec(self):
        actions = (self.store.status, self.store.result, self.store.replay,
                   self.store.request_cancel, self.store.server_spec)
        for action in actions:
            with self.subTest(action=action.__name__), self.assertRaisesRegex(ExecutionError, "job not found"):
                action("other", self.job)

    def test_result_is_unavailable_until_all_required_stages_finish(self):
        with self.assertRaises(ExecutionError):
            self.store.result(OWNER, self.job)
        fence, claimed = self.claim()
        attempt, _ = claimed
        self.store.complete(OWNER, self.job, *fence, self.spec.stages[0].id, attempt, result())
        with self.assertRaises(ExecutionError):
            self.store.finish(OWNER, self.job, *fence, "succeeded")
        with self.assertRaises(ExecutionError):
            self.store.result(OWNER, self.job)

    def test_atomic_claim_has_exactly_one_winner(self):
        fence = self.start()
        def try_claim():
            try:
                return self.store.claim(self.grant, self.job, *fence, self.spec.stages[0].id)
            except ExecutionError:
                return None
        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(lambda _: try_claim(), range(10)))
        self.assertEqual(sum(r is not None for r in results), 1)
        status = self.store.status(OWNER, self.job)
        self.assertEqual(status["tokens_reserved"], self.spec.stages[0].token_reservation)
        self.assertEqual(status["cost_reserved_microusd"], 100)

    def test_second_runner_and_out_of_order_stage_are_rejected(self):
        fence = self.start()
        with self.assertRaises(ExecutionBusy):
            self.store.begin(self.grant, self.job)
        with self.assertRaisesRegex(ExecutionError, "dependencies"):
            self.store.claim(self.grant, self.job, *fence, self.spec.stages[-1].id)
        self.assertEqual(self.store.status(OWNER, self.job)["tokens_reserved"], 0)

    def test_duplicate_completion_wrong_attempt_and_stale_fence_are_rejected(self):
        fence, claimed = self.claim()
        attempt, _ = claimed
        stage = self.spec.stages[0].id
        for bad_fence, bad_attempt in (((fence[0], fence[1] + 1), attempt), (fence, "wrong")):
            with self.assertRaises(ExecutionError):
                self.store.complete(OWNER, self.job, *bad_fence, stage, bad_attempt, result())
        self.store.complete(OWNER, self.job, *fence, stage, attempt, result())
        with self.assertRaises(ExecutionError):
            self.store.complete(OWNER, self.job, *fence, stage, attempt, result())

    def test_missing_usage_oversize_and_hidden_reasoning_are_not_committed(self):
        fence, claimed = self.claim()
        stage = self.spec.stages[0]
        attempt, _ = claimed
        cases = [result(content="<think>private chain</think>"), result(content="x" * 250001),
                 {**result(), "input_tokens": True}, {**result(), "output_tokens": stage.maximum_output_tokens + 1},
                 {**result(), "input_tokens": stage.maximum_input_tokens + 1}, {**result(), "extra": "secret"}]
        for invalid in cases:
            with self.subTest(invalid=str(invalid)[:70]), self.assertRaises(ValueError):
                self.store.complete(OWNER, self.job, *fence, stage.id, attempt, invalid)
        self.assertEqual(self.store.frontier(OWNER, self.job, *fence)["stages"][stage.id], "running")
        self.store.fail_stage(OWNER, self.job, *fence, stage.id, attempt)

    def test_cancel_is_idempotent_and_cannot_publish_a_late_answer(self):
        spec = make_spec("instant")
        grant = grant_for(spec)
        job = self.store.create(grant, "instant", spec)
        fence = self.store.begin(grant, job)
        attempt, _ = self.store.claim(grant, job, *fence, spec.stages[0].id)
        self.assertTrue(self.store.request_cancel(OWNER, job))
        self.assertFalse(self.store.request_cancel(OWNER, job))
        self.assertEqual(self.store.complete(OWNER, job, *fence, spec.stages[0].id, attempt, result()), "cancelled")
        self.assertEqual(self.store.finish(OWNER, job, *fence, "succeeded"), "cancelled")
        with self.assertRaises(ExecutionError):
            self.store.result(OWNER, job)
        self.assertEqual(self.store.status(OWNER, job)["cost_reserved_microusd"], 100)

    def test_cancel_before_start_never_has_a_runner(self):
        self.store.request_cancel(OWNER, self.job)
        self.assertEqual(self.store.status(OWNER, self.job)["state"], "cancelled")
        with self.assertRaises(ExecutionError):
            self.start()

    def test_progress_contains_no_prompt_draft_or_credentials(self):
        fence, claimed = self.claim()
        self.store.complete(OWNER, self.job, *fence, self.spec.stages[0].id, claimed[0], result())
        replay = self.store.replay(OWNER, self.job)
        serialized = json.dumps(replay) + json.dumps(self.store.status(OWNER, self.job))
        for secret in ("PRIVATE", "Bearer", "api_key", "IDENTITY_HEADER", "draft"):
            self.assertNotIn(secret, serialized)
        self.assertEqual([e["sequence"] for e in replay["events"]], [1, 2, 3, 4])
        page1 = self.store.replay(OWNER, self.job, limit=2)
        page2 = self.store.replay(OWNER, self.job, after=page1["next_sequence"], limit=2)
        self.assertEqual(page1["events"] + page2["events"], replay["events"])

    def test_instant_progress_omits_stage_events_but_advances_cursor(self):
        spec = make_spec("instant")
        grant = grant_for(spec)
        job = self.store.create(grant, "instant", spec)
        fence = self.store.begin(grant, job)
        attempt, _ = self.store.claim(grant, job, *fence, spec.stages[0].id)
        self.store.complete(OWNER, job, *fence, spec.stages[0].id, attempt, result("Kova answer"))
        self.store.finish(OWNER, job, *fence, "succeeded")
        replay = self.store.replay(OWNER, job)
        self.assertFalse(any(e["type"].startswith("stage_") for e in replay["events"]))
        self.assertEqual(replay["next_sequence"], 5)
        self.assertEqual(self.store.result(OWNER, job), {"content": "Kova answer", "tool_calls": []})

    def test_invalid_event_cursors_and_page_sizes_are_rejected(self):
        for value in (-1, True, "1", float("nan")):
            with self.assertRaises(ExecutionError):
                self.store.replay(OWNER, self.job, after=value)
        for value in (0, 501, True, "10"):
            with self.assertRaises(ExecutionError):
                self.store.replay(OWNER, self.job, limit=value)

    def test_snapshot_checksum_corruption_is_detected_before_execution(self):
        self.store._db.execute("UPDATE jobs SET spec=? WHERE id=?", (b"{}", self.job))
        with self.assertRaises(ExecutionIntegrityError):
            self.start()
        with self.assertRaises(ExecutionIntegrityError):
            self.store.replay(OWNER, self.job)

    def test_artifact_corruption_blocks_the_next_stage(self):
        fence, claimed = self.claim()
        first = self.spec.stages[0].id
        self.store.complete(OWNER, self.job, *fence, first, claimed[0], result())
        self.store._db.execute("UPDATE stages SET result=? WHERE job=? AND id=?", (b"tampered", self.job, first))
        with self.assertRaises(ExecutionIntegrityError):
            self.store.claim(self.grant, self.job, *fence, self.spec.stages[1].id)

    def test_interrupted_inflight_work_is_not_replayed_or_refunded(self):
        fence, claimed = self.claim()
        with self.assertRaises(ExecutionError):
            self.store.recover_abandoned(OWNER, self.job, runner=fence[0], epoch=fence[1])
        state = self.store.recover_abandoned(OWNER, self.job, runner=fence[0], epoch=fence[1],
                                             supervisor_confirmed_stopped=True)
        self.assertEqual(state, "interrupted")
        with self.assertRaises(ExecutionError):
            self.start()
        with self.assertRaises(ExecutionError):
            self.store.complete(OWNER, self.job, *fence, self.spec.stages[0].id, claimed[0], result())
        self.assertEqual(self.store.status(OWNER, self.job)["cost_reserved_microusd"], 100)

    def test_clean_frontier_crash_can_resume_but_old_runner_is_fenced(self):
        fence, claimed = self.claim()
        self.store.complete(OWNER, self.job, *fence, self.spec.stages[0].id, claimed[0], result())
        self.assertEqual(self.store.recover_abandoned(OWNER, self.job, runner=fence[0], epoch=fence[1],
                                                      supervisor_confirmed_stopped=True), "paused")
        newer = self.start()
        self.assertGreater(newer[1], fence[1])
        with self.assertRaises(ExecutionError):
            self.store.claim(self.grant, self.job, *fence, self.spec.stages[1].id)
        self.store.claim(self.grant, self.job, *newer, self.spec.stages[1].id)

    def test_resume_deadline_never_resets(self):
        now = self.spec.limits.deadline_unix_ms - 1000
        self.store._clock = lambda: now
        fence = self.start()
        self.store.finish(OWNER, self.job, *fence, "paused")
        now += 1001
        self.assertIsNone(self.start())
        self.assertEqual(self.store.status(OWNER, self.job)["state"], "expired")
        self.assertEqual(self.store.status(OWNER, self.job)["tokens_reserved"], 0)

    def test_crash_after_failed_stage_does_not_create_a_fake_resumable_job(self):
        fence, claimed = self.claim()
        self.store.fail_stage(OWNER, self.job, *fence, self.spec.stages[0].id, claimed[0])
        state = self.store.recover_abandoned(OWNER, self.job, runner=fence[0], epoch=fence[1],
                                             supervisor_confirmed_stopped=True)
        self.assertEqual(state, "failed")
        with self.assertRaises(ExecutionError):
            self.start()

    def test_finish_refuses_live_workers(self):
        fence, _ = self.claim()
        for state in ("succeeded", "paused", "failed", "cancelled"):
            with self.subTest(state=state), self.assertRaises(ExecutionError):
                self.store.finish(OWNER, self.job, *fence, state)

    def test_private_local_journal_survives_close_and_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.sqlite3"
            store = LocalJobStore(path)
            job = store.create(self.grant, "resume", self.spec)
            fence = store.begin(self.grant, job)
            attempt, _ = store.claim(self.grant, job, *fence, self.spec.stages[0].id)
            store.complete(OWNER, job, *fence, self.spec.stages[0].id, attempt, result())
            store.finish(OWNER, job, *fence, "paused")
            replay = store.replay(OWNER, job)
            store.close()
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            store = LocalJobStore(path)
            self.assertEqual(store.status(OWNER, job)["state"], "paused")
            self.assertEqual(store.status(OWNER, job)["completed_stages"], 1)
            self.assertEqual(store.replay(OWNER, job), replay)
            self.assertEqual(store.server_spec(OWNER, job).fingerprint, self.spec.fingerprint)
            store.close()

    def test_parallel_claim_limit_is_atomic_across_separate_store_connections(self):
        spec = make_spec("ultra", parallel=1)
        grant = grant_for(spec)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.sqlite3"
            stores = [LocalJobStore(path) for _ in range(2)]
            try:
                job = stores[0].create(grant, "two-connections", spec)
                fence = stores[0].begin(grant, job)
                stores[0].claim(grant, job, *fence, "specialist-1")
                with self.assertRaisesRegex(ExecutionError, "parallel"):
                    stores[1].claim(grant, job, *fence, "specialist-2")
                self.assertEqual(stores[1].status(OWNER, job)["cost_reserved_microusd"], 100)
            finally:
                for store in stores:
                    store.close()


if __name__ == "__main__":
    unittest.main()
