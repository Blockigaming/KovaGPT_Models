"""Actual process/crash and queue tests on a disposable Unix-socket PostgreSQL.

Child processes use synthetic workers only. Killing a local fixture process does
not establish remote model/GPU cancellation or permission to release its budget.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import multiprocessing
import os
import secrets
from threading import Barrier
from time import sleep, time_ns
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import psycopg

from execution.contracts import ALL_ROUTES, ExecutionError, ExecutionGrant
from execution.postgres_admission import AccountLimit, AccountPostgresStore, install_account_tables
from execution.postgres_store import install_postgres_schema
from execution.postgres_supervisor import DispatchClaim, DispatchRejected, PostgresSupervisor, _DispatchStore, install_dispatch_table
from execution.record_cipher import KeyMaterial, RecordCipher
from execution.runner import LocalRunner
from execution.test_postgres_store import LocalPostgres
from execution.test_support import OWNER, ModelFixture, SyntheticAdapterTestCase, grant_for, make_spec


def crashed_worker(socket_path, schema, key_bytes, expiry, pipe):
    """Disposable fixture launched by spawn; no inherited live DB connections."""
    key = KeyMaterial("test-key", key_bytes)
    cipher = RecordCipher(lambda _: key, lambda *_: key)
    def connect():
        return psycopg.connect(host=socket_path, port=55432, dbname="postgres", connect_timeout=3,
                               autocommit=True, prepare_threshold=None)
    store = AccountPostgresStore(connect, schema=schema, cipher=cipher, retention_for=lambda *_: expiry,
        deletion_authorized=lambda *_: False, quiescence_authorized=lambda *_: False,
        statement_timeout_ms=5000, lock_timeout_ms=1000, enabled=True)
    grant = ExecutionGrant(OWNER, "pro", ALL_ROUTES, True)
    instance = "fixture-process-" + str(os.getpid())
    queue = PostgresSupervisor(store, lambda name: name == instance, lambda _: grant, lambda *_: False,
        maximum_stages_per_dispatch=2, enabled=True)
    claim = queue.claim_one(instance)
    pipe.send(claim)
    def synthetic_stage(*_):
        # The stage and dispatch runner fences are already durably committed.
        pipe.send("stage_started")
        while True:
            sleep(0.02)
    try:
        queue.dispatch(claim, synthetic_stage)
    finally:
        store.close()
        pipe.close()


class PostgresSupervisorTests(SyntheticAdapterTestCase):
    @classmethod
    def setUpClass(cls):
        cls.cluster = LocalPostgres()

    @classmethod
    def tearDownClass(cls):
        cls.cluster.close()

    def setUp(self):
        super().setUp()
        self.schema = "kova_queue_" + uuid4().hex
        self.expiry = time_ns() // 1000000 + 600000
        with self.cluster.connect() as connection:
            install_postgres_schema(connection, self.schema, administration_authorized=True)
            install_account_tables(connection, self.schema, administration_authorized=True)
            install_dispatch_table(connection, self.schema, administration_authorized=True)
        self.key = KeyMaterial("test-key", secrets.token_bytes(64))
        self.cipher = RecordCipher(lambda _: self.key, lambda *_: self.key)
        self.grant = ExecutionGrant(OWNER, "pro", ALL_ROUTES, True)
        self.authorized_workers = {"worker-1", "worker-2"}
        self.stop_confirmed = False
        self.store = self.open_store()
        self.addCleanup(self.store.close)
        policy = AccountLimit(OWNER, "period", time_ns() // 1000000 - 1000, self.expiry,
                              10000000, 1000000, 50, True)
        self.store.configure_account(policy, expected_revision=0, administration_authorized=True)
        self.queue = self.open_queue(self.store)

    def open_store(self):
        return AccountPostgresStore(self.cluster.connect, schema=self.schema, cipher=self.cipher,
            retention_for=lambda *_: self.expiry + 60000, deletion_authorized=lambda *_: True,
            quiescence_authorized=lambda *_: False, statement_timeout_ms=5000, lock_timeout_ms=1000, enabled=True)

    def open_queue(self, store, **overrides):
        return PostgresSupervisor(store, lambda name: name in self.authorized_workers,
            lambda owner: self.grant if owner == self.grant.owner_id else None,
            overrides.get("stopped_observation", lambda *_: self.stop_confirmed),
            maximum_stages_per_dispatch=2, enabled=True)

    def enqueue(self, route="high", key=None):
        spec = make_spec(route)
        job = self.queue.admit_and_enqueue(self.grant, key or uuid4().hex, spec)
        return job, spec

    def test_all_routes_execute_in_bounded_slices_without_duplicate_work_or_reservations(self):
        for route in sorted(ALL_ROUTES):
            with self.subTest(route=route):
                job, spec = self.enqueue(route)
                model = ModelFixture()
                before = self.store.account_status(self.grant)["reserved_cost_microusd"]
                for _ in range(16):
                    status = self.queue.run_once("worker-1", model.worker())
                    self.assertIsNotNone(status)
                    if status["state"] == "succeeded":
                        break
                    self.assertEqual(status["state"], "paused")
                self.assertEqual(status["state"], "succeeded")
                self.assertEqual(len(model.calls), len(set(model.calls)))
                self.assertEqual(self.store.result(OWNER, job)["content"], "Kova final response")
                self.assertEqual(self.queue.status(self.grant, job)["dispatch_state"], "finished")
                self.assertEqual(self.store.account_status(self.grant)["reserved_cost_microusd"], before)
                self.assertEqual(self.store.account_status(self.grant)["outstanding_jobs"], 0)

    def test_enqueue_and_account_admission_are_one_atomic_transaction(self):
        spec = make_spec()
        with patch.object(self.queue, "enqueue", side_effect=RuntimeError("fixture enqueue failure")):
            with self.assertRaises(ExecutionError):
                self.queue.admit_and_enqueue(self.grant, "failed-enqueue", spec)
        with self.store._transaction() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)
        self.assertEqual(self.store.account_status(self.grant)["reserved_tokens"], 0)

    def test_duplicate_submission_and_enqueue_do_not_duplicate_queue_items(self):
        job, spec = self.enqueue(key="idempotent")
        self.assertEqual(self.queue.admit_and_enqueue(self.grant, "idempotent", spec), job)
        self.assertFalse(self.queue.enqueue(self.grant, job))
        claim = self.queue.claim_one("worker-1")
        self.assertEqual(claim.job_id, job)
        self.assertIsNone(self.queue.claim_one("worker-2"))

    def test_two_connections_cannot_claim_the_same_queue_item(self):
        job, _ = self.enqueue()
        second = self.open_store()
        self.addCleanup(second.close)
        queue2 = self.open_queue(second)
        barrier = Barrier(2)
        def take(item):
            queue, worker = item
            barrier.wait(timeout=5)
            return queue.claim_one(worker)
        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(take, ((self.queue, "worker-1"), (queue2, "worker-2"))))
        self.assertEqual(sum(c is not None for c in claims), 1)
        self.assertEqual(next(c for c in claims if c is not None).job_id, job)

    def test_claimed_work_is_not_taken_over_just_because_it_is_old(self):
        job, _ = self.enqueue()
        claim = self.queue.claim_one("worker-1")
        with self.store._transaction():
            self.queue._sql("UPDATE {} SET enqueued_ms=1 WHERE job=%s", (job,))
        self.assertIsNone(self.queue.claim_one("worker-2"))
        with self.assertRaises(DispatchRejected):
            self.queue.recover_stopped_worker(claim, "unverified-age")

    def test_disabled_dispatcher_does_not_touch_database_or_worker_callbacks(self):
        called = Mock(side_effect=AssertionError("disabled callback invoked"))
        with patch.object(self.store, "_transaction", side_effect=AssertionError("database opened")):
            queue = PostgresSupervisor(self.store, called, called, called, maximum_stages_per_dispatch=1)
            with self.assertRaises(DispatchRejected):
                queue.claim_one("worker-1")
        called.assert_not_called()

    def test_unknown_or_truthy_worker_permission_is_rejected(self):
        with self.assertRaises(DispatchRejected):
            self.queue.claim_one("unknown-worker")
        queue = PostgresSupervisor(self.store, lambda _: "yes", lambda _: self.grant, lambda *_: False,
            maximum_stages_per_dispatch=1, enabled=True)
        with self.assertRaises(DispatchRejected):
            queue.claim_one("worker-1")

    def test_stale_forged_claim_and_owner_cannot_dispatch(self):
        self.enqueue()
        claim = self.queue.claim_one("worker-1")
        called = Mock(side_effect=AssertionError("worker called"))
        for altered in (replace(claim, generation=claim.generation+1), replace(claim, token="wrong"),
                        replace(claim, owner_id="other"), replace(claim, worker_instance="worker-2")):
            with self.assertRaises(DispatchRejected):
                self.queue.dispatch(altered, called)
        called.assert_not_called()

    def test_pending_cancelled_job_never_dispatches_a_model(self):
        job, _ = self.enqueue()
        self.store.request_cancel(OWNER, job)
        model = ModelFixture()
        self.assertIsNone(self.queue.run_once("worker-1", model.worker()))
        self.assertEqual(model.calls, [])
        self.assertEqual(self.queue.status(self.grant, job)["dispatch_state"], "finished")

    def test_revoked_grant_holds_queue_until_explicit_reauthorization(self):
        job, _ = self.enqueue()
        self.grant = replace(self.grant, execution_authorized=False)
        self.assertIsNone(self.queue.claim_one("worker-1"))
        self.assertEqual(self.queue.status(self.grant, job)["dispatch_state"], "held")
        self.grant = replace(self.grant, execution_authorized=True)
        self.assertFalse(self.queue.enqueue(self.grant, job))
        self.assertIsNotNone(self.queue.claim_one("worker-2"))

    def test_clean_pause_reopens_on_another_connection_and_reuses_completed_stages(self):
        job, spec = self.enqueue("max")
        model = ModelFixture()
        self.assertEqual(self.queue.run_once("worker-1", model.worker())["state"], "paused")
        self.assertEqual(len(model.calls), 2)
        reopened = self.open_store()
        self.addCleanup(reopened.close)
        queue = self.open_queue(reopened)
        for _ in range(8):
            status = queue.run_once("worker-2", model.worker())
            if status["state"] == "succeeded":
                break
        self.assertEqual(status["state"], "succeeded")
        self.assertEqual(model.calls, [stage.id for stage in spec.stages])

    def test_crash_before_runner_start_can_be_confirmed_and_safely_requeued(self):
        job, _ = self.enqueue("instant")
        old = self.queue.claim_one("worker-1")
        self.stop_confirmed = True
        self.assertEqual(self.queue.recover_stopped_worker(old, "confirmed-prestart-stop"), "pending")
        with self.assertRaises(DispatchRejected):
            self.queue.dispatch(old, ModelFixture().worker())
        model = ModelFixture()
        self.assertEqual(self.queue.run_once("worker-2", model.worker())["state"], "succeeded")
        self.assertEqual(model.calls, ["answer-1"])

    def test_success_before_queue_ack_is_not_reexecuted_after_recovery(self):
        job, _ = self.enqueue("instant")
        claim = self.queue.claim_one("worker-1")
        model = ModelFixture()
        bound = _DispatchStore(self.queue, claim)
        self.assertEqual(LocalRunner(bound, lambda: self.grant, model.worker()).run(job)["state"], "succeeded")
        self.stop_confirmed = True
        self.assertEqual(self.queue.recover_stopped_worker(claim, "confirmed-postcommit-stop"), "finished")
        self.assertIsNone(self.queue.run_once("worker-2", model.worker()))
        self.assertEqual(model.calls, ["answer-1"])

    def test_clean_completed_frontier_crash_resumes_without_repeating_artifact(self):
        job, spec = self.enqueue()
        claim = self.queue.claim_one("worker-1")
        fence = _DispatchStore(self.queue, claim).begin(self.grant, job)
        attempt, _ = self.store.claim(self.grant, job, *fence, spec.stages[0].id)
        self.store.complete(OWNER, job, *fence, spec.stages[0].id, attempt,
            {"content":"completed fixture stage", "tool_calls":[], "input_tokens":10,
             "output_tokens":3, "debate_required":None})
        self.stop_confirmed = True
        self.assertEqual(self.queue.recover_stopped_worker(claim, "confirmed-frontier-stop"), "pending")
        model = ModelFixture()
        for _ in range(6):
            status = self.queue.run_once("worker-2", model.worker())
            if status["state"] == "succeeded":
                break
        self.assertEqual(status["state"], "succeeded")
        self.assertNotIn(spec.stages[0].id, model.calls)

    def test_actual_killed_worker_process_leaves_uncertain_attempt_unreplayed(self):
        job, _ = self.enqueue("instant")
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe(duplex=False)
        process = ctx.Process(target=crashed_worker,
            args=(str(self.cluster.socket), self.schema, self.key.key, self.expiry, child))
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(10), "child did not claim job")
            claim = parent.recv()
            self.assertEqual(claim.job_id, job)
            self.assertTrue(parent.poll(10), "child did not start its stage")
            self.assertEqual(parent.recv(), "stage_started")
            self.assertTrue(process.is_alive())
            process.terminate()
            process.join(5)
            self.assertFalse(process.is_alive())
            self.assertIsNotNone(process.exitcode)
            verifier = lambda observed, receipt: (
                observed == claim and receipt == "joined-fixture-process" and not process.is_alive()
                and observed.worker_instance == "fixture-process-" + str(process.pid))
            queue = self.open_queue(self.store, stopped_observation=verifier)
            self.assertEqual(queue.recover_stopped_worker(claim, "joined-fixture-process"), "interrupted")
            model = ModelFixture()
            self.assertIsNone(queue.run_once("worker-2", model.worker()))
            self.assertEqual(model.calls, [])
            self.assertEqual(self.store.status(OWNER, job)["state"], "interrupted")
            self.assertEqual(self.store.account_status(self.grant)["outstanding_jobs"], 1)
            with self.store._transaction() as db:
                self.assertEqual(db.execute("SELECT state FROM stages WHERE job=?", (job,)).fetchone()[0], "uncertain")
        finally:
            if process.is_alive():
                process.kill()
                process.join(5)
            parent.close()
            process.close()

    def test_tool_suspension_is_held_without_executing_or_retrying_tools(self):
        job, _ = self.enqueue("instant")
        calls = []
        def worker(*_):
            calls.append(1)
            return {"content":"", "tool_calls":[{"id":"call-1", "type":"function",
                "function":{"name":"lookup", "arguments":"{}"}}], "input_tokens":10,
                "output_tokens":3, "debate_required":None}
        self.assertEqual(self.queue.run_once("worker-1", worker)["state"], "waiting_tools")
        self.assertEqual(self.queue.status(self.grant, job)["dispatch_state"], "held")
        self.assertIsNone(self.queue.run_once("worker-2", worker))
        self.assertEqual(calls, [1])

    def test_other_runner_fence_cannot_be_recovered_using_an_unrelated_dispatch_claim(self):
        job, _ = self.enqueue()
        claim = self.queue.claim_one("worker-1")
        self.store.begin(self.grant, job)  # Not registered by this dispatch wrapper.
        self.stop_confirmed = True
        with self.assertRaises(DispatchRejected):
            self.queue.recover_stopped_worker(claim, "wrong-runner-receipt")

    def test_dispatch_metadata_never_exposes_prompts_tokens_or_private_artifacts(self):
        job, _ = self.enqueue()
        value = self.queue.status(self.grant, job)
        self.assertEqual(set(value), {"job_id", "dispatch_state", "generation"})
        self.assertNotIn("PRIVATE", json.dumps(value))
        with self.assertRaises(ExecutionError):
            self.queue.status(replace(self.grant, owner_id="other"), job)

    def test_live_runner_cannot_be_acknowledged_as_finished(self):
        job, _ = self.enqueue()
        claim = self.queue.claim_one("worker-1")
        _DispatchStore(self.queue, claim).begin(self.grant, job)
        with self.assertRaises(DispatchRejected):
            self.queue.acknowledge(claim)

    def test_bad_dispatch_slice_and_unapproved_installation_are_rejected(self):
        for count in (0, True, 17, "2"):
            with self.assertRaises(ExecutionError):
                PostgresSupervisor(self.store, lambda _:True, lambda _:self.grant, lambda *_:False,
                    maximum_stages_per_dispatch=count)
        with self.cluster.connect() as connection:
            with self.assertRaises(DispatchRejected):
                install_dispatch_table(connection, "kova_unapproved")
