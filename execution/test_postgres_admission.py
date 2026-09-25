"""Account reservation tests against a private disposable PostgreSQL cluster."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import secrets
from threading import Barrier
from time import time_ns
import unittest
from unittest.mock import patch
from uuid import uuid4

from execution.contracts import ALL_ROUTES, ExecutionError
from execution.postgres_admission import AccountLimit, AccountPostgresStore, AccountAdmissionError, install_account_tables
from execution.postgres_store import install_postgres_schema
from execution.record_cipher import KeyMaterial, RecordCipher
from execution.runner import LocalRunner
from execution.test_postgres_store import LocalPostgres
from execution.test_support import OWNER, ModelFixture, SyntheticAdapterTestCase, grant_for, make_spec


class AccountAdmissionTests(SyntheticAdapterTestCase):
    @classmethod
    def setUpClass(cls):
        cls.cluster = LocalPostgres()

    @classmethod
    def tearDownClass(cls):
        cls.cluster.close()

    def setUp(self):
        super().setUp()
        self.now = time_ns() // 1000000
        self.ends = self.now + 600000
        self.schema = "kova_account_" + uuid4().hex
        with self.cluster.connect() as connection:
            install_postgres_schema(connection, self.schema, administration_authorized=True)
            install_account_tables(connection, self.schema, administration_authorized=True)
        self.key = KeyMaterial("fixture", secrets.token_bytes(64))
        self.cipher = RecordCipher(lambda _: self.key, lambda *_: self.key)
        self.quiescent = False
        self.store = self.open()
        self.addCleanup(self.store.close)
        self.policy = AccountLimit(OWNER, "period-1", self.now - 1000, self.ends, 10000000, 1000000, 50, True)
        self.store.configure_account(self.policy, expected_revision=0, administration_authorized=True)

    def open(self):
        return AccountPostgresStore(self.cluster.connect, schema=self.schema, cipher=self.cipher,
            retention_for=lambda *_: self.ends + 60000, deletion_authorized=lambda *_: True,
            quiescence_authorized=lambda *_: self.quiescent,
            statement_timeout_ms=5000, lock_timeout_ms=1000, enabled=True, clock_ms=lambda: self.now)

    def configure(self, **changes):
        self.policy = replace(self.policy, **changes)
        revision = self.store.account_status(grant_for(make_spec()))["revision"]
        return self.store.configure_account(self.policy, expected_revision=revision, administration_authorized=True)

    def test_all_explicit_routes_reserve_once_and_execute_without_extra_account_cost(self):
        total_tokens = total_cost = 0
        for route in sorted(ALL_ROUTES):
            with self.subTest(route=route):
                spec = make_spec(route)
                grant = grant_for(spec)
                job = self.store.create(grant, route, spec)
                self.assertEqual(self.store.create(grant, route, spec), job)
                self.assertEqual(LocalRunner(self.store, lambda: grant, ModelFixture().worker()).run(job)["state"], "succeeded")
                total_tokens += spec.limits.token_limit
                total_cost += spec.limits.cost_limit_microusd
                account = self.store.account_status(grant)
                self.assertEqual(account["reserved_tokens"], total_tokens)
                self.assertEqual(account["reserved_cost_microusd"], total_cost)
                self.assertEqual(account["outstanding_jobs"], 0)
                self.assertFalse(account["measured_billing"])

    def test_two_connections_cannot_spend_the_same_last_reservation(self):
        spec = make_spec()
        grant = grant_for(spec)
        self.configure(token_cap=spec.limits.token_limit, cost_cap_microusd=spec.limits.cost_limit_microusd)
        second = self.open()
        self.addCleanup(second.close)
        gate = Barrier(2)
        def create(item):
            store, key = item
            gate.wait(timeout=5)
            try:
                return store.create(grant, key, spec)
            except AccountAdmissionError:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = list(pool.map(create, ((self.store, "one"), (second, "two"))))
        self.assertEqual(sum(value is not None for value in result), 1)
        self.assertEqual(self.store.account_status(grant)["remaining_tokens"], 0)
        with self.store._transaction() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)

    def test_outstanding_job_cap_is_atomic_across_connections(self):
        self.configure(maximum_outstanding_jobs=1)
        spec = make_spec()
        grant = grant_for(spec)
        self.store.create(grant, "one", spec)
        second = self.open()
        self.addCleanup(second.close)
        with self.assertRaises(AccountAdmissionError):
            second.create(grant, "two", spec)
        self.assertEqual(self.store.account_status(grant)["outstanding_jobs"], 1)

    def test_policy_absence_disabled_flag_and_window_prevent_admission(self):
        spec = make_spec()
        grant = grant_for(spec)
        with self.assertRaises(AccountAdmissionError):
            self.store.create(replace(grant, owner_id="unconfigured"), "no-policy", spec)
        self.configure(enabled=False)
        with self.assertRaises(AccountAdmissionError):
            self.store.create(grant, "disabled", spec)
        self.configure(enabled=True)
        self.now = self.ends
        with self.assertRaises(AccountAdmissionError):
            self.store.create(grant, "expired", spec)

    def test_job_deadline_cannot_run_beyond_its_explicit_account_period(self):
        spec = make_spec()
        grant = grant_for(spec)
        self.now = self.policy.starts_ms + 1
        short = AccountLimit("short-owner", "short", self.now - 1, self.now + 20, 1000000, 1000000, 10, True)
        self.store.configure_account(short, expected_revision=0, administration_authorized=True)
        with self.assertRaises(AccountAdmissionError):
            self.store.create(replace(grant, owner_id=short.owner_id), "too-long", spec)

    def test_policy_revisions_do_not_allow_lost_updates_or_unapproved_administration(self):
        with self.assertRaises(AccountAdmissionError):
            self.store.configure_account(self.policy, expected_revision=1)
        self.configure(maximum_outstanding_jobs=2)
        with self.assertRaises(AccountAdmissionError):
            self.store.configure_account(self.policy, expected_revision=1, administration_authorized=True)

    def test_window_cannot_be_extended_or_replaced_before_it_ends(self):
        for changes in ({"ends_ms": self.ends + 1}, {"starts_ms": self.policy.starts_ms - 1},
                        {"period_id": "early-new-period"}):
            with self.subTest(changes=changes), self.assertRaises(AccountAdmissionError):
                self.store.configure_account(replace(self.policy, **changes), expected_revision=1, administration_authorized=True)

    def test_reserved_cost_cannot_be_erased_by_lowering_caps_or_deleting_completed_job(self):
        spec = make_spec()
        grant = grant_for(spec)
        job = self.store.create(grant, "delete-test", spec)
        with self.assertRaises(AccountAdmissionError):
            self.store.configure_account(replace(self.policy, token_cap=1), expected_revision=1, administration_authorized=True)
        LocalRunner(self.store, lambda: grant, ModelFixture().worker()).run(job)
        before = self.store.account_status(grant)
        self.store.delete_job(grant, job, reason="owner_request")
        self.assertEqual(self.store.account_status(grant), before)
        with self.assertRaises(ExecutionError):
            self.store.create(grant, "delete-test", spec)

    def test_failure_and_unknown_started_work_keep_the_concurrency_slot(self):
        spec = make_spec("instant")
        grant = grant_for(spec)
        job = self.store.create(grant, "failure", spec)
        fixture = ModelFixture()
        fixture.override["answer-1"] = "<think>reject this fixture</think>"
        self.assertEqual(LocalRunner(self.store, lambda: grant, fixture.worker()).run(job)["state"], "failed")
        account = self.store.account_status(grant)
        self.assertEqual(account["outstanding_jobs"], 1)
        self.assertEqual(account["reserved_cost_microusd"], spec.limits.cost_limit_microusd)
        with self.assertRaises(AccountAdmissionError):
            self.store.confirm_quiescence(grant, job, "unverified-receipt")
        self.quiescent = True
        self.store.confirm_quiescence(grant, job, "verified-receipt")
        self.assertEqual(self.store.account_status(grant)["outstanding_jobs"], 0)
        self.assertEqual(self.store.account_status(grant)["reserved_cost_microusd"], account["reserved_cost_microusd"])

    def test_cancellation_before_any_attempt_releases_only_slot_not_financial_reservation(self):
        spec = make_spec()
        grant = grant_for(spec)
        job = self.store.create(grant, "cancel-before-start", spec)
        self.store.request_cancel(OWNER, job)
        account = self.store.account_status(grant)
        self.assertEqual(account["outstanding_jobs"], 0)
        self.assertEqual(account["reserved_cost_microusd"], spec.limits.cost_limit_microusd)

    def test_revocation_during_model_work_withholds_output_and_releases_runner(self):
        spec = make_spec("max")
        grant = grant_for(spec)
        job = self.store.create(grant, "revoke", spec)
        def revoke(_stage, _control):
            self.configure(enabled=False)
        fixture = ModelFixture(hook=revoke)
        status = LocalRunner(self.store, lambda: grant, fixture.worker()).run(job)
        self.assertEqual(status["state"], "failed")
        self.assertEqual(len(fixture.calls), 1)
        self.assertEqual(self.store.account_status(grant)["outstanding_jobs"], 1)
        with self.store._transaction() as db:
            self.assertIsNone(db.execute("SELECT runner FROM jobs WHERE id=?", (job,)).fetchone()[0])
        with self.assertRaises(ExecutionError):
            self.store.result(OWNER, job)

    def test_revoked_accounts_can_still_cancel_without_a_new_execution_grant(self):
        spec = make_spec()
        grant = grant_for(spec)
        job = self.store.create(grant, "cancel-revoked", spec)
        self.configure(enabled=False)
        self.assertTrue(self.store.request_cancel(OWNER, job))
        self.assertEqual(self.store.status(OWNER, job)["state"], "cancelled")

    def test_clean_pause_and_restart_do_not_reserve_the_same_job_twice(self):
        spec = make_spec("max")
        grant = grant_for(spec)
        job = self.store.create(grant, "resume-account", spec)
        fixture = ModelFixture()
        self.assertEqual(LocalRunner(self.store, lambda: grant, fixture.worker()).run(job, max_stages=2)["state"], "paused")
        before = self.store.account_status(grant)
        reopened = self.open()
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.create(grant, "resume-account", spec), job)
        self.assertEqual(reopened.account_status(grant), before)
        self.assertEqual(LocalRunner(reopened, lambda: grant, fixture.worker()).run(job)["state"], "succeeded")
        after = reopened.account_status(grant)
        self.assertEqual(after["reserved_tokens"], before["reserved_tokens"])
        self.assertEqual(after["reserved_cost_microusd"], before["reserved_cost_microusd"])
        self.assertEqual(len(fixture.calls), len(set(fixture.calls)))

    def test_cross_owner_reservations_and_readout_are_separate(self):
        spec = make_spec()
        grant = grant_for(spec)
        second = replace(grant, owner_id="second-owner")
        self.store.configure_account(replace(self.policy, owner_id=second.owner_id), expected_revision=0, administration_authorized=True)
        self.store.create(grant, "same-key", spec)
        self.assertEqual(self.store.account_status(second)["reserved_tokens"], 0)
        self.store.create(second, "same-key", spec)
        self.assertEqual(self.store.account_status(grant)["reserved_tokens"], spec.limits.token_limit)
        self.assertEqual(self.store.account_status(second)["reserved_tokens"], spec.limits.token_limit)

    def test_outstanding_uncertain_holds_survive_an_explicit_period_change(self):
        spec = make_spec("instant")
        grant = grant_for(spec)
        job = self.store.create(grant, "unknown", spec)
        fence = self.store.begin(grant, job)
        self.store.claim(grant, job, *fence, "answer-1")
        self.store.recover_abandoned(OWNER, job, runner=fence[0], epoch=fence[1], supervisor_confirmed_stopped=True)
        self.now = self.ends
        next_period = replace(self.policy, period_id="period-2", starts_ms=self.ends, ends_ms=self.ends + 600000, maximum_outstanding_jobs=1)
        self.store.configure_account(next_period, expected_revision=1, administration_authorized=True)
        account = self.store.account_status(grant)
        self.assertEqual(account["outstanding_jobs"], 1)
        self.assertEqual(account["reserved_cost_microusd"], 0)
        self.assertFalse(account["measured_billing"])

    def test_failed_account_reservation_rolls_back_the_entire_new_job(self):
        spec = make_spec()
        grant = grant_for(spec)
        self.configure(token_cap=1)
        with self.assertRaises(AccountAdmissionError):
            self.store.create(grant, "insufficient", spec)
        with self.store._transaction() as db:
            for table in ("jobs", "stages", "events", "private_job_policy"):
                self.assertEqual(db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0], 0)
        self.assertEqual(self.store.account_status(grant)["outstanding_jobs"], 0)

    def test_missing_reservation_prevents_execution_without_silent_repair(self):
        spec = make_spec()
        grant = grant_for(spec)
        job = self.store.create(grant, "missing", spec)
        with self.store._transaction():
            self.store._account_sql("DELETE FROM {} WHERE job=%s", ("execution_reservations",), (job,))
        with self.assertRaises(AccountAdmissionError):
            self.store.begin(grant, job)
        with self.assertRaises(AccountAdmissionError):
            self.open()

    def test_live_runner_cannot_be_declared_quiescent_or_deleted(self):
        spec = make_spec()
        grant = grant_for(spec)
        job = self.store.create(grant, "live", spec)
        self.store.begin(grant, job)
        self.quiescent = True
        with self.assertRaises(AccountAdmissionError):
            self.store.confirm_quiescence(grant, job, "premature")
        with self.assertRaises(AccountAdmissionError):
            self.store.delete_job(grant, job, reason="owner_request")

    def test_invalid_numeric_policies_never_reach_database_configuration(self):
        for field in ("starts_ms", "ends_ms", "token_cap", "cost_cap_microusd", "maximum_outstanding_jobs"):
            for value in (True, 0, -1, "100", float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ExecutionError):
                    replace(self.policy, **{field:value})
        with self.assertRaises(AccountAdmissionError):
            replace(self.policy, enabled="true")

    def test_nested_savepoint_error_cannot_partially_commit_the_outer_admission(self):
        spec = make_spec()
        grant = grant_for(spec)
        original = self.store._account_sql
        def fail_insert(query, tables, parameters=()):
            if query.startswith("INSERT INTO") and "execution_reservations" in tables:
                raise RuntimeError("synthetic ledger failure")
            return original(query, tables, parameters)
        with patch.object(self.store, "_account_sql", side_effect=fail_insert):
            with self.assertRaises(ExecutionError):
                self.store.create(grant, "savepoint-fail", spec)
        with self.store._transaction() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)
