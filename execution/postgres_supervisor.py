"""Durable dispatch and confirmed-crash recovery for the existing model runner.

No background daemon, external tool, public endpoint or production queue is
installed. A service process explicitly calls run_once. The queue references
already admitted encrypted jobs and never stores model prompts or credentials.
"""

from dataclasses import dataclass
from uuid import uuid4

from execution.contracts import ExecutionError, ExecutionGrant, identifier, positive_integer
from execution.postgres_admission import AccountPostgresStore
from execution.postgres_store import _schema
from execution.runner import LocalRunner
from execution.store import TERMINAL


class DispatchRejected(ExecutionError):
    """The dispatcher, queue fence or recovery observation is not authorized."""


def need(condition):
    if not condition:
        raise DispatchRejected("model dispatch rejected")


def install_dispatch_table(connection, schema, *, administration_authorized=False):
    need(administration_authorized is True)
    from psycopg import sql
    import psycopg
    need(type(connection) is psycopg.Connection and connection.autocommit)
    schema = _schema(schema)
    with connection.transaction():
        connection.execute(sql.SQL('''CREATE TABLE {} (
          job text PRIMARY KEY REFERENCES {}(id) ON DELETE CASCADE, owner text NOT NULL,
          state text NOT NULL CHECK(state IN ('pending','claimed','held','finished','interrupted')),
          enqueued_ms bigint NOT NULL, generation bigint NOT NULL DEFAULT 0,
          worker_instance text, claim_token text, runner_token text, runner_epoch bigint,
          recovery_receipt text)''').format(sql.Identifier(schema, "model_dispatch"), sql.Identifier(schema, "jobs")))
        connection.execute(sql.SQL("REVOKE ALL ON {} FROM PUBLIC").format(sql.Identifier(schema, "model_dispatch")))


@dataclass(frozen=True)
class DispatchClaim:
    job_id: str
    owner_id: str
    worker_instance: str
    generation: int
    token: str


class _DispatchStore:
    """Bind the kernel's independent runner fence without shadowing store methods."""
    def __init__(self, queue, claim):
        self._queue, self._dispatch_claim = queue, claim

    def __getattr__(self, name):
        return getattr(self._queue.store, name)

    def begin(self, grant, job_id):
        claim = self._dispatch_claim
        need(job_id == claim.job_id and grant.owner_id == claim.owner_id)
        with self._queue.store._transaction():
            self._queue._check_claim(claim)
            fence = self._queue.store.begin(grant, job_id)
            if fence is not None:
                self._queue._sql("UPDATE {} SET runner_token=%s,runner_epoch=%s WHERE job=%s",
                                 (*fence, job_id))
            return fence

    def control_state(self, owner, job_id, runner, epoch):
        with self._queue.store._transaction():
            self._queue._check_claim(self._dispatch_claim)
            return self._queue.store.control_state(owner, job_id, runner, epoch)


class PostgresSupervisor:
    """Current worker authorization and owner grants are trusted server callbacks.

    worker_authorized(instance) returns literal True only for an authenticated
    current worker instance. grant_for_owner(owner) returns the current user/plan
    grant. stopped_observation(claim,receipt) verifies the exact worker process is
    stopped, not just that its lease is old. No automatic takeover or retry of an
    unknown provider outcome is allowed. Remote quiescence for financial holds is
    still a separate AccountPostgresStore confirmation.
    """
    def __init__(self, store, worker_authorized, grant_for_owner, stopped_observation,
                 *, maximum_stages_per_dispatch, enabled=False):
        need(type(store) is AccountPostgresStore and type(enabled) is bool)
        need(all(callable(fn) for fn in (worker_authorized, grant_for_owner, stopped_observation)))
        positive_integer(maximum_stages_per_dispatch, "dispatch stage slice", 16)
        self.store, self._enabled = store, enabled
        self._worker_authorized, self._grant, self._stopped = worker_authorized, grant_for_owner, stopped_observation
        self._maximum_stages = maximum_stages_per_dispatch
        if enabled:
            with store._transaction():
                self._sql("SELECT count(*) FROM {}")

    def _sql(self, query, parameters=()):
        from psycopg import sql
        return self.store._db.connection.execute(sql.SQL(query).format(
            sql.Identifier(self.store._db.schema, "model_dispatch")), parameters)

    def _worker(self, instance):
        need(self._enabled)
        identifier(instance, "worker instance")
        try:
            allowed = self._worker_authorized(instance)
        except Exception:
            raise DispatchRejected("worker authorization unavailable") from None
        need(allowed is True)

    def enqueue(self, grant, job_id):
        need(self._enabled and type(grant) is ExecutionGrant)
        with self.store._transaction():
            job = self.store._owned(self.store._db, grant.owner_id, job_id)
            spec = self.store._spec(job)
            grant.authorize(grant.owner_id, spec.plan["route_id"])
            need(job["runner"] is None and job["state"] in ("queued", "paused"))
            self.store._check_job_admission(grant.owner_id, job_id)
            row = self._sql("SELECT * FROM {} WHERE job=%s", (job_id,)).fetchone()
            if row is not None:
                need(row["owner"] == grant.owner_id and row["state"] in ("pending", "held"))
                self._sql("UPDATE {} SET state='pending' WHERE job=%s", (job_id,))
                return False
            self._sql("INSERT INTO {}(job,owner,state,enqueued_ms) VALUES (%s,%s,'pending',%s)",
                      (job_id, grant.owner_id, self.store._time()))
            return True

    def admit_and_enqueue(self, grant, idempotency_key, spec):
        """Internal server transaction, not a public submission/authentication API."""
        need(self._enabled)
        with self.store._transaction():
            job = self.store.create(grant, idempotency_key, spec)
            state = self.store.status(grant.owner_id, job)["state"]
            if state in ("queued", "paused"):
                self.enqueue(grant, job)
            return job

    def claim_one(self, worker_instance):
        self._worker(worker_instance)
        with self.store._transaction():
            rows = self._sql("SELECT * FROM {} WHERE state='pending' ORDER BY enqueued_ms,job LIMIT 32").fetchall()
            for row in rows:
                job = self.store._owned(self.store._db, row["owner"], row["job"])
                if job["state"] in TERMINAL:
                    self._sql("UPDATE {} SET state='finished' WHERE job=%s", (row["job"],))
                    continue
                if job["runner"] is not None:
                    self._sql("UPDATE {} SET state='held' WHERE job=%s", (row["job"],))
                    continue
                try:
                    grant = self._grant(row["owner"])
                    need(type(grant) is ExecutionGrant and grant.owner_id == row["owner"])
                    grant.authorize(row["owner"], self.store._spec(job).plan["route_id"])
                    self.store._check_job_admission(row["owner"], row["job"])
                except Exception:
                    self._sql("UPDATE {} SET state='held' WHERE job=%s", (row["job"],))
                    continue
                token, generation = str(uuid4()), row["generation"] + 1
                self._sql("UPDATE {} SET state='claimed',worker_instance=%s,claim_token=%s,generation=%s,runner_token=NULL,runner_epoch=NULL WHERE job=%s",
                          (worker_instance, token, generation, row["job"]))
                return DispatchClaim(row["job"], row["owner"], worker_instance, generation, token)
            return None

    def _check_claim(self, claim):
        need(type(claim) is DispatchClaim)
        self._worker(claim.worker_instance)
        row = self._sql("SELECT * FROM {} WHERE job=%s AND owner=%s", (claim.job_id, claim.owner_id)).fetchone()
        need(row is not None and row["state"] == "claimed"
             and (row["claim_token"], row["worker_instance"], row["generation"]) ==
                 (claim.token, claim.worker_instance, claim.generation))
        return row

    def dispatch(self, claim, worker):
        """Execute one bounded stage slice, synchronously in the calling worker process."""
        need(callable(worker))
        with self.store._transaction():
            self._check_claim(claim)
        bound = _DispatchStore(self, claim)
        status = LocalRunner(bound, lambda: self._grant(claim.owner_id), worker).run(
            claim.job_id, max_stages=self._maximum_stages)
        self.acknowledge(claim)
        return status

    def run_once(self, worker_instance, worker):
        claim = self.claim_one(worker_instance)
        return None if claim is None else self.dispatch(claim, worker)

    def acknowledge(self, claim):
        with self.store._transaction():
            self._check_claim(claim)
            job = self.store._owned(self.store._db, claim.owner_id, claim.job_id)
            need(job["runner"] is None)
            need(job["state"] in TERMINAL | {"paused"})
            state = "pending" if job["state"] == "paused" else "held" if job["state"] == "waiting_tools" else "finished"
            self._sql("UPDATE {} SET state=%s,claim_token=NULL,worker_instance=NULL,runner_token=NULL,runner_epoch=NULL WHERE job=%s",
                      (state, claim.job_id))

    def recover_stopped_worker(self, claim, receipt_id):
        """Explicit confirmed recovery. Queue age/timeout alone never authorizes it."""
        need(self._enabled and type(claim) is DispatchClaim)
        identifier(receipt_id, "worker stop receipt")
        try:
            stopped = self._stopped(claim, receipt_id)
        except Exception:
            raise DispatchRejected("worker stop verification unavailable") from None
        need(stopped is True)
        with self.store._transaction():
            row = self._sql("SELECT * FROM {} WHERE job=%s AND owner=%s", (claim.job_id, claim.owner_id)).fetchone()
            need(row is not None and row["state"] == "claimed"
                 and (row["claim_token"], row["worker_instance"], row["generation"]) ==
                     (claim.token, claim.worker_instance, claim.generation))
            job = self.store._owned(self.store._db, claim.owner_id, claim.job_id)
            if job["runner"] is not None:
                need((row["runner_token"], row["runner_epoch"]) == (job["runner"], job["epoch"]))
                self.store.recover_abandoned(claim.owner_id, claim.job_id,
                    runner=job["runner"], epoch=job["epoch"], supervisor_confirmed_stopped=True)
                job = self.store._owned(self.store._db, claim.owner_id, claim.job_id)
            state = ("pending" if job["state"] in ("queued", "paused") else
                     "interrupted" if job["state"] == "interrupted" else
                     "held" if job["state"] == "waiting_tools" else "finished")
            self._sql("UPDATE {} SET state=%s,generation=generation+1,claim_token=NULL,worker_instance=NULL,runner_token=NULL,runner_epoch=NULL,recovery_receipt=%s WHERE job=%s",
                      (state, receipt_id, claim.job_id))
            return state

    def status(self, grant, job_id):
        need(type(grant) is ExecutionGrant)
        with self.store._transaction():
            self.store._owned(self.store._db, grant.owner_id, job_id)
            row = self._sql("SELECT state,generation FROM {} WHERE job=%s AND owner=%s", (job_id, grant.owner_id)).fetchone()
            need(row is not None)
            return {"job_id": job_id, "dispatch_state": row["state"], "generation": row["generation"]}
