"""Cross-job account reservations integrated into the encrypted PostgreSQL store.

No public/job API, model call or financial charge is implemented here. This is a
separate database admission component, not republication of the blocked HTTP API.
Policies and quiescence decisions come from trusted server administration only.
"""

from contextlib import contextmanager
from dataclasses import dataclass

from execution.contracts import ExecutionError, ExecutionGrant, ExecutionSpec, identifier, positive_integer, require
from execution.postgres_store import PostgresPrivateJobStore, PostgresStoreError, _schema
from execution.store import TERMINAL


ACCOUNT_TABLES = frozenset(("execution_accounts", "execution_reservations"))


class AccountAdmissionError(ExecutionError):
    """Current account permission or conservative reservation does not admit work."""


def need(condition):
    if not condition:
        raise AccountAdmissionError("account execution admission rejected")


@dataclass(frozen=True)
class AccountLimit:
    owner_id: str
    period_id: str
    starts_ms: int
    ends_ms: int
    token_cap: int
    cost_cap_microusd: int
    maximum_outstanding_jobs: int
    enabled: bool = False

    def __post_init__(self):
        identifier(self.owner_id, "account owner")
        identifier(self.period_id, "account period")
        for name in ("starts_ms", "ends_ms", "token_cap", "cost_cap_microusd", "maximum_outstanding_jobs"):
            positive_integer(getattr(self, name), name)
        need(self.starts_ms < self.ends_ms and type(self.enabled) is bool)


def install_account_tables(connection, schema, *, administration_authorized=False):
    """Explicit additive installation on an EMPTY execution schema only.

    Existing unreserved work needs an independently reviewed migration; do not
    silently assume its requests have consumed no account budget.
    """
    need(administration_authorized is True)
    from psycopg import sql
    import psycopg
    need(type(connection) is psycopg.Connection and connection.autocommit)
    schema = _schema(schema)
    table = lambda name: sql.Identifier(schema, name)
    with connection.transaction():
        need(connection.execute(sql.SQL("SELECT count(*) FROM {}").format(table("jobs"))).fetchone()[0] == 0)
        connection.execute(sql.SQL('''CREATE TABLE {} (
          owner text PRIMARY KEY, period text NOT NULL, starts_ms bigint NOT NULL, ends_ms bigint NOT NULL,
          token_cap bigint NOT NULL CHECK(token_cap>0), cost_cap bigint NOT NULL CHECK(cost_cap>0),
          maximum_jobs bigint NOT NULL CHECK(maximum_jobs>0), enabled integer NOT NULL CHECK(enabled IN (0,1)),
          revision bigint NOT NULL CHECK(revision>0), CHECK(starts_ms>0 AND ends_ms>starts_ms))''').format(table("execution_accounts")))
        connection.execute(sql.SQL('''CREATE TABLE {} (
          job text PRIMARY KEY, owner text NOT NULL REFERENCES {}(owner), period text NOT NULL,
          fingerprint text NOT NULL, reserved_tokens bigint NOT NULL CHECK(reserved_tokens>0),
          reserved_cost bigint NOT NULL CHECK(reserved_cost>0), slot_held integer NOT NULL CHECK(slot_held IN (0,1)),
          quiescence_receipt text)''').format(table("execution_reservations"), table("execution_accounts")))
        connection.execute(sql.SQL("REVOKE ALL ON {}, {} FROM PUBLIC").format(table("execution_accounts"), table("execution_reservations")))


class AccountPostgresStore(PostgresPrivateJobStore):
    """Atomic admission and current-policy checks across all jobs for one account.

    Full route reservations are conserved after success/failure/deletion. They are
    not measured bills. Uncertain started work holds its concurrency slot until a
    separately verified server quiescence receipt is supplied; no automatic retry,
    deadline reset, refund, or assumption of remote GPU cancellation is allowed.
    """
    def __init__(self, *args, quiescence_authorized, **kwargs):
        need(callable(quiescence_authorized))
        self._quiescence_authorized = quiescence_authorized
        self._admission_depth = 0
        super().__init__(*args, **kwargs)
        try:
            with self._transaction() as db:
                self._account_sql("SELECT count(*) FROM {}", ("execution_accounts",))
                orphaned = self._db.connection.execute(__import__('psycopg').sql.SQL(
                    "SELECT count(*) FROM {} j LEFT JOIN {} r ON r.job=j.id WHERE r.job IS NULL"
                ).format(__import__('psycopg').sql.Identifier(self._db.schema, "jobs"),
                         __import__('psycopg').sql.Identifier(self._db.schema, "execution_reservations"))).fetchone()[0]
                need(orphaned == 0)
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _transaction(self):
        # The outer transaction keeps the parent's cross-process advisory lock;
        # nested inherited store operations use database savepoints, not a second
        # isolation change or an independently committed metadata transaction.
        with self._lock:
            if self._admission_depth:
                with self._db.connection.transaction():
                    yield self._db
                return
            self._admission_depth += 1
            try:
                with super()._transaction() as db:
                    yield db
            finally:
                self._admission_depth -= 1

    def _account_sql(self, query, tables, parameters=()):
        from psycopg import sql
        need(all(name in ACCOUNT_TABLES for name in tables))
        return self._db.connection.execute(sql.SQL(query).format(
            *(sql.Identifier(self._db.schema, name) for name in tables)), parameters)

    def _policy(self, owner):
        row = self._account_sql("SELECT * FROM {} WHERE owner=%s", ("execution_accounts",), (owner,)).fetchone()
        need(row is not None)
        policy = AccountLimit(row["owner"], row["period"], row["starts_ms"], row["ends_ms"],
                              row["token_cap"], row["cost_cap"], row["maximum_jobs"], bool(row["enabled"]))
        return policy, row["revision"]

    def _totals(self, owner, period):
        row = self._account_sql('''SELECT
          coalesce(sum(CASE WHEN period=%s THEN reserved_tokens ELSE 0 END),0),
          coalesce(sum(CASE WHEN period=%s THEN reserved_cost ELSE 0 END),0),
          coalesce(sum(slot_held),0) FROM {} WHERE owner=%s''',
          ("execution_reservations",), (period, period, owner)).fetchone()
        return tuple(int(value) for value in row)

    def configure_account(self, policy, *, expected_revision, administration_authorized=False):
        need(administration_authorized is True and type(policy) is AccountLimit
             and type(expected_revision) is int and 0 <= expected_revision < 2**53)
        with self._transaction():
            existing = self._account_sql("SELECT * FROM {} WHERE owner=%s", ("execution_accounts",), (policy.owner_id,)).fetchone()
            need((existing["revision"] if existing else 0) == expected_revision)
            if existing:
                if existing["period"] == policy.period_id:
                    need((existing["starts_ms"], existing["ends_ms"]) == (policy.starts_ms, policy.ends_ms))
                else:
                    need(self._time() >= existing["ends_ms"] and policy.starts_ms >= existing["ends_ms"])
                    need(self._account_sql("SELECT count(*) FROM {} WHERE owner=%s AND period=%s",
                        ("execution_reservations",), (policy.owner_id, policy.period_id)).fetchone()[0] == 0)
            need(self._time() < policy.ends_ms)
            tokens, cost, _ = self._totals(policy.owner_id, policy.period_id)
            need(policy.token_cap >= tokens and policy.cost_cap_microusd >= cost)
            self._account_sql('''INSERT INTO {} VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
              ON CONFLICT(owner) DO UPDATE SET period=EXCLUDED.period, starts_ms=EXCLUDED.starts_ms,
              ends_ms=EXCLUDED.ends_ms, token_cap=EXCLUDED.token_cap, cost_cap=EXCLUDED.cost_cap,
              maximum_jobs=EXCLUDED.maximum_jobs, enabled=EXCLUDED.enabled, revision=EXCLUDED.revision''',
              ("execution_accounts",), (policy.owner_id, policy.period_id, policy.starts_ms, policy.ends_ms,
               policy.token_cap, policy.cost_cap_microusd, policy.maximum_outstanding_jobs,
               int(policy.enabled), expected_revision + 1))
            return expected_revision + 1

    def create(self, grant, idempotency_key, spec):
        need(type(grant) is ExecutionGrant and type(spec) is ExecutionSpec)
        with self._transaction():
            policy, _ = self._policy(grant.owner_id)
            need(policy.enabled and policy.starts_ms <= self._time() < policy.ends_ms
                 and spec.limits.deadline_unix_ms <= policy.ends_ms)
            job = super().create(grant, idempotency_key, spec)
            existing = self._account_sql("SELECT * FROM {} WHERE job=%s", ("execution_reservations",), (job,)).fetchone()
            if existing:
                need(existing["owner"] == grant.owner_id and existing["fingerprint"] == spec.fingerprint)
                return job
            tokens, cost, slots = self._totals(grant.owner_id, policy.period_id)
            need(tokens + spec.limits.token_limit <= policy.token_cap
                 and cost + spec.limits.cost_limit_microusd <= policy.cost_cap_microusd
                 and slots < policy.maximum_outstanding_jobs)
            self._account_sql("INSERT INTO {} VALUES (%s,%s,%s,%s,%s,%s,1,NULL)", ("execution_reservations",),
                (job, grant.owner_id, policy.period_id, spec.fingerprint,
                 spec.limits.token_limit, spec.limits.cost_limit_microusd))
            return job

    def _check_job_admission(self, owner, job_id):
        job = self._owned(self._db, owner, job_id)
        policy, _ = self._policy(owner)
        reservation = self._account_sql("SELECT * FROM {} WHERE job=%s AND owner=%s",
                                       ("execution_reservations",), (job_id, owner)).fetchone()
        need(reservation is not None and reservation["fingerprint"] == job["fingerprint"]
             and reservation["period"] == policy.period_id and reservation["slot_held"] == 1)
        need(policy.enabled and policy.starts_ms <= self._time() < policy.ends_ms)
        tokens, cost, _ = self._totals(owner, policy.period_id)
        need(tokens <= policy.token_cap and cost <= policy.cost_cap_microusd)

    def begin(self, grant, job_id):
        with self._transaction():
            self._check_job_admission(grant.owner_id, job_id)
            result = super().begin(grant, job_id)
            if result is None:
                self._release_if_safe(grant.owner_id, job_id)
            return result

    def claim(self, grant, job_id, runner, epoch, stage_id):
        with self._transaction():
            self._check_job_admission(grant.owner_id, job_id)
            return super().claim(grant, job_id, runner, epoch, stage_id)

    def control_state(self, owner, job_id, runner, epoch):
        with self._transaction():
            self._check_job_admission(owner, job_id)
            return super().control_state(owner, job_id, runner, epoch)

    def complete(self, owner, job_id, runner, epoch, stage_id, attempt, result):
        with self._transaction():
            self._check_job_admission(owner, job_id)
            return super().complete(owner, job_id, runner, epoch, stage_id, attempt, result)

    def _release_if_safe(self, owner, job_id):
        job = self._owned(self._db, owner, job_id)
        if job["runner"] is None and job["state"] in TERMINAL and (job["state"] == "succeeded" or job["tokens_reserved"] == 0):
            self._account_sql("UPDATE {} SET slot_held=0 WHERE job=%s AND owner=%s",
                              ("execution_reservations",), (job_id, owner))

    def finish(self, owner, job_id, runner, epoch, state):
        with self._transaction():
            if state == "succeeded":
                try:
                    self._check_job_admission(owner, job_id)
                except AccountAdmissionError:
                    state = "failed"
            final = super().finish(owner, job_id, runner, epoch, state)
            self._release_if_safe(owner, job_id)
            return final

    def request_cancel(self, owner, job_id):
        with self._transaction():
            result = super().request_cancel(owner, job_id)
            self._release_if_safe(owner, job_id)
            return result

    def confirm_quiescence(self, grant, job_id, receipt_id):
        need(type(grant) is ExecutionGrant)
        identifier(receipt_id, "quiescence receipt")
        with self._transaction():
            job = self._owned(self._db, grant.owner_id, job_id)
            need(job["state"] in TERMINAL and job["runner"] is None)
            try:
                approved = self._quiescence_authorized(grant.owner_id, job_id, receipt_id)
            except Exception:
                raise AccountAdmissionError("quiescence verification unavailable") from None
            need(approved is True)
            row = self._account_sql("SELECT * FROM {} WHERE job=%s AND owner=%s",
                ("execution_reservations",), (job_id, grant.owner_id)).fetchone()
            need(row is not None and row["quiescence_receipt"] in (None, receipt_id))
            self._account_sql("UPDATE {} SET slot_held=0,quiescence_receipt=%s WHERE job=%s AND owner=%s",
                ("execution_reservations",), (receipt_id, job_id, grant.owner_id))

    def delete_job(self, grant, job_id, **kwargs):
        with self._transaction():
            row = self._account_sql("SELECT * FROM {} WHERE job=%s AND owner=%s",
                ("execution_reservations",), (job_id, grant.owner_id)).fetchone()
            need(row is not None and row["slot_held"] == 0)
            # The reservation intentionally has no cascading job foreign key:
            # deleting private payloads must not silently refund admitted costs.
            return super().delete_job(grant, job_id, **kwargs)

    def account_status(self, grant):
        need(type(grant) is ExecutionGrant)
        with self._transaction():
            policy, revision = self._policy(grant.owner_id)
            tokens, cost, slots = self._totals(grant.owner_id, policy.period_id)
            return {"period_id": policy.period_id, "revision": revision, "enabled": policy.enabled,
                    "reserved_tokens": tokens, "reserved_cost_microusd": cost, "outstanding_jobs": slots,
                    "remaining_tokens": policy.token_cap - tokens,
                    "remaining_cost_microusd": policy.cost_cap_microusd - cost,
                    "measured_billing": False}
