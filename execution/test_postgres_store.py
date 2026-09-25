"""Real isolated PostgreSQL tests using only a private local Unix socket.

No external DSN/environment credentials, Docker image, production service, or GPU.
The installed PostgreSQL binaries create a disposable cluster for this test class.
Missing required binaries is a failure, not a skipped durability test.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
from threading import Barrier
from time import monotonic, time_ns
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import psycopg

from execution.contracts import ALL_ROUTES, ExecutionBusy, ExecutionError
from execution.postgres_store import (
    PostgresConnectionConfig, PostgresPrivateJobStore, PostgresStoreError,
    connect_postgres, install_postgres_schema,
)
from execution.record_cipher import KeyMaterial, RecordCipher
from execution.runner import LocalRunner
from execution.test_support import OWNER, ModelFixture, SyntheticAdapterTestCase, grant_for, make_spec


class LocalPostgres:
    def __init__(self):
        candidates = list(Path("/usr/lib/postgresql").glob("*/bin/pg_ctl"))
        if not candidates:
            raise RuntimeError("PostgreSQL test binaries are required")
        self.bin = max(candidates, key=lambda p: int(p.parents[1].name)).parent
        self.directory = tempfile.TemporaryDirectory(prefix="kova-pg-test-")
        self.root = Path(self.directory.name)
        self.data = self.root / "data"
        self.socket = self.root / "socket"
        self.socket.mkdir(mode=0o700)
        self.started = False
        self.env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG") if key in os.environ}
        self.env["LC_ALL"] = "C"
        try:
            self.command("initdb", "-D", str(self.data), "--no-locale", "--encoding=UTF8",
                         "--auth-local=trust", "--auth-host=reject")
            self.start()
        except BaseException:
            self.close()
            raise

    def command(self, command, *args):
        return subprocess.run([str(self.bin / command), *args], capture_output=True, text=True,
                              check=True, timeout=20, env=self.env)

    def start(self):
        self.command("pg_ctl", "-D", str(self.data), "-l", str(self.root / "postgres.log"),
                     "-o", f"-k {self.socket} -h '' -p 55432", "-w", "-t", "10", "start")
        self.started = True

    def restart(self):
        self.command("pg_ctl", "-D", str(self.data), "-w", "-t", "10", "stop", "-m", "fast")
        self.started = False
        self.start()

    def connect(self):
        return psycopg.connect(host=str(self.socket), port=55432, dbname="postgres",
                               connect_timeout=3, autocommit=True, prepare_threshold=None)

    def close(self):
        if getattr(self, "started", False):
            try:
                self.command("pg_ctl", "-D", str(self.data), "-w", "-t", "10", "stop", "-m", "fast")
            finally:
                self.started = False
        if hasattr(self, "directory"):
            self.directory.cleanup()


class PostgresStoreTests(SyntheticAdapterTestCase):
    @classmethod
    def setUpClass(cls):
        cls.cluster = LocalPostgres()
        print("Isolated PostgreSQL:", cls.cluster.command("postgres", "--version").stdout.strip(), flush=True)
        with cls.cluster.connect() as connection:
            assert connection.execute("SHOW listen_addresses").fetchone()[0] == ""
            assert connection.execute("SHOW fsync").fetchone()[0] == "on"
            assert connection.execute("SHOW full_page_writes").fetchone()[0] == "on"
            assert connection.execute("SHOW synchronous_commit").fetchone()[0] == "on"

    @classmethod
    def tearDownClass(cls):
        cls.cluster.close()

    def setUp(self):
        super().setUp()
        self.schema = "kova_test_" + uuid4().hex
        with self.cluster.connect() as connection:
            install_postgres_schema(connection, self.schema, administration_authorized=True)
        self.key = KeyMaterial("ephemeral", secrets.token_bytes(64))
        self.cipher = RecordCipher(lambda _: self.key, lambda *_: self.key)
        self.expiry = time_ns() // 1000000 + 120000
        self.store = self.open()
        self.addCleanup(self.store.close)

    def open(self, **changes):
        args = {"schema": self.schema, "cipher": self.cipher,
                "retention_for": lambda *_: self.expiry, "deletion_authorized": lambda *_: True,
                "statement_timeout_ms": 5000, "lock_timeout_ms": 1000, "enabled": True}
        args.update(changes)
        return PostgresPrivateJobStore(self.cluster.connect, **args)

    def job(self, route="high", key=None, store=None):
        spec = make_spec(route)
        grant = grant_for(spec)
        active = self.store if store is None else store
        job = active.create(grant, key or uuid4().hex, spec)
        return job, spec, grant

    def test_all_24_routes_execute_on_real_postgres_with_encrypted_artifacts(self):
        stages = 0
        for route in sorted(ALL_ROUTES):
            with self.subTest(route=route):
                job, spec, grant = self.job(route)
                model = ModelFixture()
                status = LocalRunner(self.store, lambda: grant, model.worker()).run(job)
                self.assertEqual(status["state"], "succeeded")
                self.assertEqual(self.store.result(OWNER, job)["content"], "Kova final response")
                self.assertEqual(self.store.server_spec(OWNER, job).encoded, spec.encoded)
                with self.store._transaction() as db:
                    row = db.execute("SELECT spec FROM jobs WHERE id=?", (job,)).fetchone()
                    self.assertNotIn(b"PRIVATE input text", bytes(row[0]))
                    for row in db.execute("SELECT result FROM stages WHERE job=? AND result IS NOT NULL", (job,)):
                        self.assertNotIn(b"PRIVATE stage result", bytes(row[0]))
                stages += status["completed_stages"]
        self.assertGreater(stages, 113)

    def test_separate_connections_create_idempotent_job_exactly_once(self):
        stores = [self.open() for _ in range(4)]
        spec = make_spec()
        grant = grant_for(spec)
        barrier = Barrier(4)
        def create(store):
            barrier.wait(timeout=5)
            return store.create(grant, "shared-submission", spec)
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs = list(pool.map(create, stores))
            self.assertEqual(len(set(jobs)), 1)
            self.assertEqual(self.store.status(OWNER, jobs[0])["sequence"], 1)
        finally:
            for store in stores:
                store.close()

    def test_separate_connections_cannot_claim_one_stage_twice(self):
        job, spec, grant = self.job()
        fence = self.store.begin(grant, job)
        second = self.open()
        self.addCleanup(second.close)
        barrier = Barrier(2)
        def claim(store):
            barrier.wait(timeout=5)
            try:
                return store.claim(grant, job, *fence, spec.stages[0].id)
            except ExecutionError:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(claim, (self.store, second)))
        self.assertEqual(sum(value is not None for value in results), 1)
        self.assertEqual(self.store.status(OWNER, job)["cost_reserved_microusd"], 100)

    def test_concurrent_runner_ownership_and_out_of_order_stages_are_rejected(self):
        job, spec, grant = self.job()
        fence = self.store.begin(grant, job)
        second = self.open()
        self.addCleanup(second.close)
        with self.assertRaises(ExecutionBusy):
            second.begin(grant, job)
        with self.assertRaises(ExecutionError):
            second.claim(grant, job, *fence, spec.stages[-1].id)

    def test_transaction_failure_rolls_back_encrypted_job_and_events(self):
        spec = make_spec()
        grant = grant_for(spec)
        with patch.object(self.store, "_event", side_effect=RuntimeError("synthetic write failure")):
            with self.assertRaises(PostgresStoreError):
                self.store.create(grant, "rollback", spec)
        with self.store._transaction() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM private_job_policy").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM events").fetchone()[0], 0)

    def test_disconnect_before_commit_rolls_back_without_partial_records(self):
        connection = self.cluster.connect()
        from psycopg import sql
        connection.execute("BEGIN")
        connection.execute(sql.SQL("INSERT INTO {} VALUES (%s,%s,%s)").format(
            sql.Identifier(self.schema, "private_job_tombstones")), (OWNER, "uncommitted", 1900000000000))
        connection.close()
        with self.store._transaction() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM private_job_tombstones").fetchone()[0], 0)

    def test_committed_clean_frontier_survives_database_process_restart(self):
        job, spec, grant = self.job("max")
        model = ModelFixture()
        first = LocalRunner(self.store, lambda: grant, model.worker()).run(job, max_stages=2)
        self.assertEqual(first["state"], "paused")
        self.store.close()
        self.cluster.restart()
        self.store = self.open()
        self.addCleanup(self.store.close)
        self.assertEqual(self.store.server_spec(OWNER, job).encoded, spec.encoded)
        self.assertEqual(LocalRunner(self.store, lambda: grant, model.worker()).run(job)["state"], "succeeded")
        self.assertEqual(len(model.calls), len(set(model.calls)))

    def test_unknown_inflight_restart_is_fenced_not_replayed(self):
        job, spec, grant = self.job()
        fence = self.store.begin(grant, job)
        self.store.claim(grant, job, *fence, spec.stages[0].id)
        second = self.open()
        self.addCleanup(second.close)
        self.assertEqual(second.recover_abandoned(OWNER, job, runner=fence[0], epoch=fence[1],
            supervisor_confirmed_stopped=True), "interrupted")
        with self.assertRaises(ExecutionError):
            self.store.claim(grant, job, *fence, spec.stages[0].id)
        model = ModelFixture()
        self.assertEqual(LocalRunner(second, lambda: grant, model.worker()).run(job)["state"], "interrupted")
        self.assertEqual(model.calls, [])
        self.assertEqual(second.status(OWNER, job)["cost_reserved_microusd"], 100)

    def test_owner_isolation_applies_to_postgres_reads_replay_and_cancel(self):
        job, _, _ = self.job()
        for action in (self.store.status, self.store.server_spec, self.store.replay, self.store.request_cancel):
            with self.subTest(action=action.__name__), self.assertRaises(ExecutionError):
                action("other-owner", job)

    def test_retention_and_deletion_tombstones_survive_connection_reopen(self):
        job, spec, grant = self.job("instant", key="delete-once")
        LocalRunner(self.store, lambda: grant, ModelFixture().worker()).run(job)
        self.store.delete_job(grant, job, reason="owner_request")
        reopened = self.open()
        self.addCleanup(reopened.close)
        with self.assertRaises(ExecutionError):
            reopened.create(grant, "delete-once", spec)
        with self.assertRaises(ExecutionError):
            reopened.result(OWNER, job)

    def test_statement_timeout_is_bounded_and_does_not_leave_open_transaction(self):
        store = self.open(statement_timeout_ms=100, lock_timeout_ms=50)
        self.addCleanup(store.close)
        started = monotonic()
        with self.assertRaises(PostgresStoreError):
            with store._transaction() as db:
                db.execute("SELECT pg_sleep(2)")
        self.assertLess(monotonic() - started, 2)
        self.assertEqual(store._db.connection.info.transaction_status, psycopg.pq.TransactionStatus.IDLE)
        with store._transaction() as db:
            self.assertEqual(db.execute("SELECT 1").fetchone()[0], 1)

    def test_lock_timeout_does_not_steal_or_retry_an_owned_transaction(self):
        store = self.open(statement_timeout_ms=200, lock_timeout_ms=50)
        self.addCleanup(store.close)
        with self.store._transaction():
            with self.assertRaises(PostgresStoreError):
                with store._transaction():
                    self.fail("lock was not enforced")
        self.assertEqual(store._db.connection.info.transaction_status, psycopg.pq.TransactionStatus.IDLE)

    def test_schema_install_is_separate_and_missing_schema_is_not_created_at_runtime(self):
        new_schema = "kova_missing_" + uuid4().hex
        with self.cluster.connect() as connection:
            with self.assertRaises(ExecutionError):
                install_postgres_schema(connection, new_schema)
        with self.assertRaises(PostgresStoreError):
            self.open(schema=new_schema)
        with self.cluster.connect() as connection:
            self.assertIsNone(connection.execute("SELECT to_regnamespace(%s)", (new_schema,)).fetchone()[0])

    def test_schema_names_and_parameters_cannot_inject_sql(self):
        for schema in ("public", "pg_catalog", "kova_x;DROP TABLE jobs", "kova_x.y", "kova_x\"", ""):
            with self.subTest(schema=schema), self.assertRaises(ExecutionError):
                self.open(schema=schema)
        spec = make_spec()
        grant = grant_for(spec)
        with self.assertRaises(ExecutionError):
            self.store.create(grant, "bad'; DROP TABLE jobs;--", spec)
        with self.store._transaction() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)

    def test_schema_version_mismatch_fails_closed(self):
        with self.store._transaction() as db:
            db.execute("DELETE FROM model_store_version")
        with self.assertRaises(ExecutionError):
            self.open()

    def test_no_public_schema_access_is_granted_by_installation(self):
        with self.cluster.connect() as connection:
            rows = connection.execute("SELECT a.privilege_type FROM pg_namespace n, LATERAL aclexplode(n.nspacl) a WHERE n.nspname=%s AND a.grantee=0", (self.schema,)).fetchall()
        self.assertEqual(rows, [])

    def test_disabled_store_does_not_invoke_connection_factory(self):
        factory = Mock(side_effect=AssertionError("database opened"))
        with self.assertRaises(ExecutionError):
            PostgresPrivateJobStore(factory, schema=self.schema, cipher=self.cipher,
                retention_for=lambda *_: self.expiry, deletion_authorized=lambda *_: True,
                statement_timeout_ms=1000, lock_timeout_ms=100)
        factory.assert_not_called()

    def test_network_factory_requires_explicit_authorization_before_credentials(self):
        config = PostgresConnectionConfig("fixture.postgres.database.azure.com", "192.0.2.1", 5432,
            "model_execution", "model_worker", "/trusted/root.pem", 3)
        credential = Mock(side_effect=AssertionError("credential read"))
        with patch("psycopg.connect", side_effect=AssertionError("network called")):
            with self.assertRaises(PostgresStoreError):
                connect_postgres(config, credential)
        credential.assert_not_called()

    def test_network_factory_enforces_explicit_tls_destination_and_timeout(self):
        config = PostgresConnectionConfig("fixture.postgres.database.azure.com", "192.0.2.1", 5432,
            "model_execution", "model_worker", "/trusted/root.pem", 3, True)
        with patch("psycopg.connect", return_value="synthetic-connection") as connect:
            self.assertEqual(connect_postgres(config, lambda: "synthetic-token"), "synthetic-connection")
        kwargs = connect.call_args.kwargs
        self.assertEqual(kwargs["sslmode"], "verify-full")
        self.assertEqual(kwargs["hostaddr"], "192.0.2.1")
        self.assertEqual(kwargs["host"], config.host)
        self.assertEqual(kwargs["connect_timeout"], 3)
        self.assertTrue(kwargs["autocommit"])
