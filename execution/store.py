"""Local SQLite reference journal for offline execution/restart tests.

NOT a production Azure storage choice: use an approved external durable store
before deployment. This local database can contain prompts/private artifacts;
use synthetic data only until encryption, retention and deployment are reviewed.
No model calls, networking, background threads or database opens occur on import.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
from threading import RLock
from time import time_ns
from uuid import uuid4

from execution.contracts import (
    ExecutionBusy, ExecutionError, ExecutionGrant, ExecutionIntegrityError,
    ExecutionInterrupted, ExecutionSpec, canonical, identifier, positive_integer, require,
)
from worker.handler import sanitize_engine_response


TERMINAL = frozenset(("succeeded", "failed", "cancelled", "expired", "interrupted", "waiting_tools"))
PUBLIC_EVENT_TYPES = frozenset((
    "job_created", "job_resumed", "job_paused", "job_succeeded", "job_failed",
    "job_cancel_requested", "job_cancelled", "job_expired", "job_interrupted", "job_waiting_tools",
    "stage_started", "stage_completed", "stage_skipped", "stage_failed",
))
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, idem TEXT NOT NULL,
 fingerprint TEXT NOT NULL, spec BLOB NOT NULL, state TEXT NOT NULL,
 runner TEXT, epoch INTEGER NOT NULL DEFAULT 0, cancel_requested INTEGER NOT NULL DEFAULT 0,
 tokens_reserved INTEGER NOT NULL DEFAULT 0, cost_reserved INTEGER NOT NULL DEFAULT 0,
 sequence INTEGER NOT NULL DEFAULT 0, UNIQUE(owner, idem)
);
CREATE TABLE IF NOT EXISTS stages (
 job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE, id TEXT NOT NULL,
 ordinal INTEGER NOT NULL, state TEXT NOT NULL, attempt TEXT, epoch INTEGER,
 result BLOB, checksum TEXT, PRIMARY KEY(job,id), UNIQUE(job,ordinal)
);
CREATE TABLE IF NOT EXISTS events (
 job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE, sequence INTEGER NOT NULL,
 type TEXT NOT NULL, stage TEXT, occurred_ms INTEGER NOT NULL,
 PRIMARY KEY(job,sequence)
);
"""


def now_ms():
    return time_ns() // 1_000_000


def _safe_result(value, stage):
    require(isinstance(value, dict) and set(value) == {
        "content", "tool_calls", "input_tokens", "output_tokens", "debate_required",
    }, "invalid stage result")
    decision = value["debate_required"]
    require(decision is None or (stage.id == "judge" and type(decision) is bool),
            "only the judge may return a debate decision")
    require(stage.id != "judge" or type(decision) is bool, "judge decision is missing")
    for field in ("input_tokens", "output_tokens"):
        positive_integer(value[field], field)
    require(value["input_tokens"] <= stage.maximum_input_tokens
            and value["output_tokens"] <= stage.maximum_output_tokens, "stage token budget exceeded")
    require(value["content"] is None or (isinstance(value["content"], str)
            and len(value["content"]) <= 250_000), "stage artifact is too large")
    # Reuse the existing user-visible content/tool/reasoning checks before storing.
    sanitized = sanitize_engine_response("journal", {
        "choices": [{"message": {"content": value["content"], "tool_calls": value["tool_calls"]},
                     "finish_reason": "tool_calls" if value["tool_calls"] else "stop"}],
        "usage": {"prompt_tokens": value["input_tokens"], "completion_tokens": value["output_tokens"]},
    })
    result = {**value, "content": sanitized["content"], "tool_calls": sanitized["tool_calls"]}
    encoded = canonical(result)
    require(len(encoded) <= 2 * 1024 * 1024, "stage result too large")
    return encoded


class LocalJobStore:
    """Serialized reference implementation with explicit atomic write transactions.

    Multiple instances/processes may open the same private local file. The runner
    token and epoch are fencing controls, not expiring leases. A live owner is
    never stolen automatically; crash recovery is an explicit supervisor action.
    """

    def __init__(self, path=":memory:", *, clock_ms=now_ms):
        require(callable(clock_ms), "trusted journal clock required")
        self._clock = clock_ms
        self._lock = RLock()
        if path != ":memory:":
            p = Path(path)
            require(p.is_absolute() and p.parent.exists(), "journal requires an absolute private local path")
            try:
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            except FileExistsError:
                info = p.lstat()
                require(stat.S_ISREG(info.st_mode) and not (info.st_mode & 0o077),
                        "journal must be a private regular file")
            else:
                os.close(fd)
        self._db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False, timeout=1)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA synchronous=FULL")
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        require(version in (0, 1), "unsupported execution journal version")
        self._db.executescript(SCHEMA)
        self._db.execute("PRAGMA user_version=1")

    def close(self):
        with self._lock:
            self._db.close()

    @contextmanager
    def _transaction(self):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
                self._db.execute("COMMIT")
            except BaseException:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise

    def _time(self):
        value = self._clock()
        positive_integer(value, "journal time")
        return value

    @staticmethod
    def _owned(db, owner, job_id):
        identifier(owner, "owner")
        identifier(job_id, "job ID")
        row = db.execute("SELECT * FROM jobs WHERE id=? AND owner=?", (job_id, owner)).fetchone()
        require(row is not None, "job not found")
        return row

    @staticmethod
    def _spec(job):
        encoded = bytes(job["spec"])
        if hashlib.sha256(encoded).hexdigest() != job["fingerprint"]:
            raise ExecutionIntegrityError("execution snapshot checksum mismatch")
        return ExecutionSpec(encoded)

    @staticmethod
    def _runner(job, runner, epoch):
        require(isinstance(runner, str) and runner and type(epoch) is int
                and job["runner"] == runner and job["epoch"] == epoch,
                "stale or invalid runner fence")

    def _event(self, db, job_id, kind, stage=None):
        require(kind in PUBLIC_EVENT_TYPES, "invalid journal event")
        db.execute("UPDATE jobs SET sequence=sequence+1 WHERE id=?", (job_id,))
        seq = db.execute("SELECT sequence FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
        db.execute("INSERT INTO events VALUES (?,?,?,?,?)", (job_id, seq, kind, stage, self._time()))

    def create(self, grant, idempotency_key, spec):
        require(type(grant) is ExecutionGrant and type(spec) is ExecutionSpec, "trusted grant and spec required")
        grant.authorize(grant.owner_id, spec.plan["route_id"])
        identifier(idempotency_key, "idempotency key")
        with self._transaction() as db:
            existing = db.execute("SELECT * FROM jobs WHERE owner=? AND idem=?",
                                  (grant.owner_id, idempotency_key)).fetchone()
            if existing is not None:
                self._spec(existing)
                require(existing["fingerprint"] == spec.fingerprint,
                        "idempotency key conflicts with another request or budget")
                return existing["id"]
            require(self._time() < spec.limits.deadline_unix_ms, "execution deadline already elapsed")
            job_id = f"kova-exec-{uuid4()}"
            db.execute("INSERT INTO jobs(id,owner,idem,fingerprint,spec,state) VALUES (?,?,?,?,?,?)",
                       (job_id, grant.owner_id, idempotency_key, spec.fingerprint, spec.encoded, "queued"))
            for index, stage in enumerate(spec.stages):
                db.execute("INSERT INTO stages(job,id,ordinal,state) VALUES (?,?,?,?)",
                           (job_id, stage.id, index, "pending"))
            self._event(db, job_id, "job_created")
            return job_id

    def server_spec(self, owner, job_id):
        """Private server API; do not expose stored plans/prompts through a web route."""
        with self._transaction() as db:
            return self._spec(self._owned(db, owner, job_id))

    def status(self, owner, job_id):
        with self._transaction() as db:
            job = self._owned(db, owner, job_id)
            spec = self._spec(job)
            states = db.execute("SELECT id,state FROM stages WHERE job=? ORDER BY ordinal", (job_id,)).fetchall()
            return {
                "job_id": job_id, "state": job["state"], "route_id": spec.plan["route_id"],
                "display_name": spec.plan["display_name"], "sequence": job["sequence"],
                "completed_stages": sum(r["state"] == "completed" for r in states),
                "skipped_stages": sum(r["state"] == "skipped" for r in states),
                "total_stages": len(states), "cancel_requested": bool(job["cancel_requested"]),
                "tokens_reserved": job["tokens_reserved"], "cost_reserved_microusd": job["cost_reserved"],
                "cost_is_measured": False, "deadline_unix_ms": spec.limits.deadline_unix_ms,
            }

    def replay(self, owner, job_id, *, after=0, limit=100):
        """Owner-scoped ordered lifecycle events; never include model text/credentials.

        Stage events are omitted for profiles that disallow activity updates.
        The returned cursor advances over inspected events, including filtered ones.
        """
        require(type(after) is int and 0 <= after <= 2**53 - 1, "invalid event cursor")
        positive_integer(limit, "event limit", 500)
        with self._transaction() as db:
            job = self._owned(db, owner, job_id)
            spec = self._spec(job)
            activity = {s.id: s.activity for s in spec.stages}
            rows = db.execute("SELECT sequence,type,stage,occurred_ms FROM events "
                              "WHERE job=? AND sequence>? ORDER BY sequence LIMIT ?", (job_id, after, limit)).fetchall()
            return {
                "events": [{"job_id": job_id, "sequence": r["sequence"], "type": r["type"],
                            "stage_id": r["stage"], "occurred_ms": r["occurred_ms"]}
                           for r in rows if r["stage"] is None or activity[r["stage"]]],
                "next_sequence": rows[-1]["sequence"] if rows else after,
            }

    def result(self, owner, job_id):
        with self._transaction() as db:
            job = self._owned(db, owner, job_id)
            require(job["state"] == "succeeded", "final answer is not available")
            spec = self._spec(job)
            row = db.execute("SELECT * FROM stages WHERE job=? AND id=?", (job_id, spec.stages[-1].id)).fetchone()
            value = self._result(row)
            # No intermediate drafts, judge notes, provider metadata or credentials.
            return {"content": value["content"], "tool_calls": value["tool_calls"]}

    @staticmethod
    def _result(row):
        if row["result"] is None or hashlib.sha256(bytes(row["result"])).hexdigest() != row["checksum"]:
            raise ExecutionIntegrityError("stage artifact checksum mismatch")
        return json.loads(bytes(row["result"]))

    def begin(self, grant, job_id):
        require(type(grant) is ExecutionGrant, "current server grant required")
        with self._transaction() as db:
            job = self._owned(db, grant.owner_id, job_id)
            spec = self._spec(job)
            grant.authorize(grant.owner_id, spec.plan["route_id"])
            if job["runner"] is not None:
                raise ExecutionBusy("job already has an active runner")
            require(job["state"] in ("queued", "paused"), "job is not resumable")
            if self._time() >= spec.limits.deadline_unix_ms:
                db.execute("UPDATE jobs SET state='expired' WHERE id=?", (job_id,))
                self._event(db, job_id, "job_expired")
                return None
            runner, epoch = str(uuid4()), job["epoch"] + 1
            db.execute("UPDATE jobs SET runner=?,epoch=?,state='running' WHERE id=?", (runner, epoch, job_id))
            self._event(db, job_id, "job_resumed")
            return runner, epoch

    def frontier(self, owner, job_id, runner, epoch):
        """Server-only state/dependency view for one fenced runner."""
        with self._transaction() as db:
            job = self._owned(db, owner, job_id)
            self._runner(job, runner, epoch)
            self._spec(job)
            rows = db.execute("SELECT * FROM stages WHERE job=? ORDER BY ordinal", (job_id,)).fetchall()
            return {
                "state": job["state"], "cancel_requested": bool(job["cancel_requested"]),
                "stages": {r["id"]: r["state"] for r in rows},
            }

    def claim(self, grant, job_id, runner, epoch, stage_id):
        require(type(grant) is ExecutionGrant, "current grant required")
        with self._transaction() as db:
            job = self._owned(db, grant.owner_id, job_id)
            self._runner(job, runner, epoch)
            spec = self._spec(job)
            grant.authorize(grant.owner_id, spec.plan["route_id"])
            require(job["state"] == "running" and not job["cancel_requested"], "job is not running")
            require(self._time() < spec.limits.deadline_unix_ms, "execution deadline elapsed")
            stage = next((s for s in spec.stages if s.id == stage_id), None)
            require(stage is not None, "unknown stage")
            rows = {r["id"]: r for r in db.execute("SELECT * FROM stages WHERE job=?", (job_id,))}
            require(rows[stage_id]["state"] == "pending", "stage cannot be retried or duplicated")
            require(sum(r["state"] == "running" for r in rows.values()) < spec.limits.max_parallel,
                    "parallel stage limit reached")
            artifacts = {}
            for dependency in stage.dependencies:
                dep = rows[dependency]
                if dep["state"] == "skipped":
                    require(stage.id == "synthesis" and dependency == "debate-round-1", "invalid skipped dependency")
                    artifacts[dependency] = None
                else:
                    require(dep["state"] == "completed", "stage dependencies are incomplete")
                    artifacts[dependency] = self._result(dep)
            if stage.condition is not None:
                decision = artifacts["judge"]["debate_required"]
                require(type(decision) is bool, "judge decision must be boolean")
                if not decision:
                    db.execute("UPDATE stages SET state='skipped' WHERE job=? AND id=?", (job_id, stage_id))
                    self._event(db, job_id, "stage_skipped", stage_id)
                    return None
            tokens = job["tokens_reserved"] + stage.token_reservation
            cost = job["cost_reserved"] + stage.cost_cap_microusd
            require(tokens <= spec.limits.token_limit and cost <= spec.limits.cost_limit_microusd,
                    "execution reservation exceeds admission")
            attempt = str(uuid4())
            db.execute("UPDATE jobs SET tokens_reserved=?,cost_reserved=? WHERE id=?", (tokens, cost, job_id))
            db.execute("UPDATE stages SET state='running',attempt=?,epoch=? WHERE job=? AND id=?",
                       (attempt, epoch, job_id, stage_id))
            self._event(db, job_id, "stage_started", stage_id)
            return attempt, artifacts

    def complete(self, owner, job_id, runner, epoch, stage_id, attempt, result):
        with self._transaction() as db:
            job = self._owned(db, owner, job_id)
            self._runner(job, runner, epoch)
            spec = self._spec(job)
            stage = next((s for s in spec.stages if s.id == stage_id), None)
            require(stage is not None, "unknown stage")
            row = db.execute("SELECT * FROM stages WHERE job=? AND id=?", (job_id, stage_id)).fetchone()
            require(row["state"] == "running" and row["attempt"] == attempt and row["epoch"] == epoch,
                    "stage completion has a stale fence")
            encoded = _safe_result(result, stage)
            # Cancellation/expiry beats late success. Reservations remain charged to
            # admission even when discarded; no claim of refunded provider billing.
            state = "completed"
            if job["cancel_requested"] or job["state"] != "running":
                state = "cancelled"
            elif self._time() >= spec.limits.deadline_unix_ms:
                state = "expired"
            elif result["tool_calls"]:
                state = "waiting_tools"
            keep = state in ("completed", "waiting_tools")
            db.execute("UPDATE stages SET state=?,result=?,checksum=? WHERE job=? AND id=?", (
                state, encoded if keep else None, hashlib.sha256(encoded).hexdigest() if keep else None, job_id, stage_id,
            ))
            self._event(db, job_id, "stage_completed" if state == "completed" else "stage_failed", stage_id)
            return state

    def fail_stage(self, owner, job_id, runner, epoch, stage_id, attempt):
        with self._transaction() as db:
            job = self._owned(db, owner, job_id)
            self._runner(job, runner, epoch)
            row = db.execute("SELECT * FROM stages WHERE job=? AND id=?", (job_id, stage_id)).fetchone()
            require(row and row["state"] == "running" and row["attempt"] == attempt and row["epoch"] == epoch,
                    "stage failure has a stale fence")
            db.execute("UPDATE stages SET state='failed' WHERE job=? AND id=?", (job_id, stage_id))
            self._event(db, job_id, "stage_failed", stage_id)

    def request_cancel(self, owner, job_id):
        with self._transaction() as db:
            job = self._owned(db, owner, job_id)
            if job["state"] in TERMINAL or job["cancel_requested"]:
                return False
            state = "cancelling" if job["runner"] is not None else "cancelled"
            db.execute("UPDATE jobs SET cancel_requested=1,state=? WHERE id=?", (state, job_id))
            self._event(db, job_id, "job_cancel_requested")
            if state == "cancelled":
                self._event(db, job_id, "job_cancelled")
            return True

    def finish(self, owner, job_id, runner, epoch, state):
        require(state in TERMINAL | {"paused"}, "invalid final job state")
        with self._transaction() as db:
            job = self._owned(db, owner, job_id)
            self._runner(job, runner, epoch)
            spec = self._spec(job)
            rows = db.execute("SELECT * FROM stages WHERE job=? ORDER BY ordinal", (job_id,)).fetchall()
            require(all(r["state"] != "running" for r in rows), "cannot finish while stages still run")
            if job["cancel_requested"]:
                state = "cancelled"
            elif self._time() >= spec.limits.deadline_unix_ms:
                state = "expired"
            if state == "succeeded":
                require(all(r["state"] in ("completed", "skipped") for r in rows)
                        and rows[-1]["state"] == "completed", "all required stages must complete")
                for row in rows:
                    if row["state"] == "completed":
                        self._result(row)
            if state == "waiting_tools":
                require(any(r["state"] == "waiting_tools" for r in rows), "no stage requested tools")
            if state == "paused":
                require(all(r["state"] in ("completed", "skipped", "pending") for r in rows),
                        "only a clean stage frontier can pause")
            db.execute("UPDATE jobs SET state=?,runner=NULL WHERE id=?", (state, job_id))
            self._event(db, job_id, "job_" + state)
            return state

    def recover_abandoned(self, owner, job_id, *, runner, epoch, supervisor_confirmed_stopped=False):
        """Administrative offline recovery; never call merely because a lease is old.

        Unknown in-flight attempts are quarantined without replay or budget release.
        Completed-frontier-only crashes can be resumed with a new runner fence.
        A deployment supervisor and external durable store are still required live.
        """
        require(supervisor_confirmed_stopped is True, "supervisor stop confirmation required")
        with self._transaction() as db:
            job = self._owned(db, owner, job_id)
            self._runner(job, runner, epoch)
            self._spec(job)
            states = {row[0] for row in db.execute("SELECT state FROM stages WHERE job=?", (job_id,))}
            if "running" in states or "uncertain" in states:
                state = "interrupted"
            elif job["cancel_requested"] or "cancelled" in states:
                state = "cancelled"
            elif "expired" in states:
                state = "expired"
            elif "failed" in states:
                state = "failed"
            elif "waiting_tools" in states:
                state = "waiting_tools"
            else:
                state = "paused"
            db.execute("UPDATE stages SET state='uncertain' WHERE job=? AND state='running'", (job_id,))
            db.execute("UPDATE jobs SET state=?,runner=NULL,epoch=epoch+1 WHERE id=?", (state, job_id))
            self._event(db, job_id, "job_" + state)
            return state

    def control_state(self, owner, job_id, runner, epoch):
        """Small fenced check for active workers; no prompt/result reconstruction."""
        with self._transaction() as db:
            job = self._owned(db, owner, job_id)
            self._runner(job, runner, epoch)
            return job["state"], bool(job["cancel_requested"])
