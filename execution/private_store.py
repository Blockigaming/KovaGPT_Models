"""Encrypt prompts and stage artifacts before any local journal SQL write.

A source-tested privacy integration for the existing kernel, not an Azure durable
store deployment. Only metadata (IDs/state/counters/fingerprints) is plaintext.
No plaintext migration, network, KMS client, key persistence or automatic purge is
installed. Tests use ephemeral keys and synthetic local databases.
"""

import hashlib
import json
from uuid import uuid4

from execution.contracts import ExecutionGrant, ExecutionSpec, canonical, identifier, require
from execution.record_cipher import RecordCipher, RecordProtectionError, need
from execution.store import LocalJobStore, TERMINAL, _safe_result


class RetentionExpired(RecordProtectionError):
    """The explicitly configured record retention deadline has elapsed."""


class PrivateJobStore(LocalJobStore):
    def __init__(self, path=":memory:", *, cipher, retention_for, deletion_authorized,
                 enabled=False, **kwargs):
        need(type(enabled) is bool and enabled and type(cipher) is RecordCipher
             and callable(retention_for) and callable(deletion_authorized))
        self._cipher, self._retention_for, self._deletion_authorized = cipher, retention_for, deletion_authorized
        super().__init__(path, **kwargs)
        try:
            with self._transaction() as db:
                installed = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='private_job_policy'").fetchone()
                need(installed is not None or db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0)
                db.execute("CREATE TABLE IF NOT EXISTS private_job_policy (job TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE, retain_until_ms INTEGER NOT NULL)")
                db.execute("CREATE TABLE IF NOT EXISTS private_job_tombstones (owner TEXT NOT NULL, idem TEXT NOT NULL, deleted_ms INTEGER NOT NULL, PRIMARY KEY(owner,idem))")
                need(db.execute("SELECT COUNT(*) FROM jobs j LEFT JOIN private_job_policy p ON p.job=j.id WHERE p.job IS NULL").fetchone()[0] == 0)
        except BaseException:
            self.close()
            raise

    def _expiry(self, job_id, *, allow_expired=False):
        row = self._db.execute("SELECT retain_until_ms FROM private_job_policy WHERE job=?", (job_id,)).fetchone()
        need(row is not None and type(row[0]) is int and 0 < row[0] < 2**53)
        if not allow_expired and self._time() >= row[0]:
            raise RetentionExpired("private record retention expired")
        return row[0]

    def create(self, grant, idempotency_key, spec):
        require(type(grant) is ExecutionGrant and type(spec) is ExecutionSpec, "trusted grant and spec required")
        grant.authorize(grant.owner_id, spec.plan["route_id"])
        identifier(idempotency_key, "idempotency key")
        with self._transaction() as db:
            need(db.execute("SELECT 1 FROM private_job_tombstones WHERE owner=? AND idem=?",
                            (grant.owner_id, idempotency_key)).fetchone() is None)
            existing = db.execute("SELECT * FROM jobs WHERE owner=? AND idem=?", (grant.owner_id, idempotency_key)).fetchone()
            if existing is not None:
                self._spec(existing)
                require(existing["fingerprint"] == spec.fingerprint, "idempotency key conflicts with another request or budget")
                return existing["id"]
            require(self._time() < spec.limits.deadline_unix_ms, "execution deadline already elapsed")
            try:
                expiry = self._retention_for(grant.owner_id, spec)
            except Exception:
                raise RecordProtectionError("private retention policy unavailable") from None
            need(type(expiry) is int and spec.limits.deadline_unix_ms <= expiry < 2**53)
            job_id = f"kova-exec-{uuid4()}"
            encrypted = self._cipher.seal(grant.owner_id, job_id, "spec", expiry, spec.encoded)
            db.execute("INSERT INTO jobs(id,owner,idem,fingerprint,spec,state) VALUES (?,?,?,?,?,?)",
                       (job_id, grant.owner_id, idempotency_key, spec.fingerprint, encrypted, "queued"))
            db.execute("INSERT INTO private_job_policy VALUES (?,?)", (job_id, expiry))
            for index, stage in enumerate(spec.stages):
                db.execute("INSERT INTO stages(job,id,ordinal,state) VALUES (?,?,?,?)", (job_id, stage.id, index, "pending"))
            self._event(db, job_id, "job_created")
            return job_id

    def _spec(self, job):
        encoded = self._cipher.open(job["owner"], job["id"], "spec", self._expiry(job["id"]), bytes(job["spec"]))
        if hashlib.sha256(encoded).hexdigest() != job["fingerprint"]:
            raise RecordProtectionError("private execution snapshot integrity failed")
        return ExecutionSpec(encoded)

    def _result(self, row):
        need(row is not None and row["result"] is not None)
        encrypted = bytes(row["result"])
        need(hashlib.sha256(encrypted).hexdigest() == row["checksum"])
        owner = self._db.execute("SELECT owner FROM jobs WHERE id=?", (row["job"],)).fetchone()
        need(owner is not None)
        decoded = self._cipher.open(owner[0], row["job"], row["id"], self._expiry(row["job"]), encrypted)
        return json.loads(decoded)

    def status(self, owner, job_id):
        try:
            return super().status(owner, job_id)
        except RetentionExpired:
            # Return only the operational fields needed to cancel/clean up expired
            # work. Do not decrypt an expired plan just to display its route name.
            with self._transaction() as db:
                job = self._owned(db, owner, job_id)
                expiry = self._expiry(job_id, allow_expired=True)
                need(self._time() >= expiry)
                rows = db.execute("SELECT state FROM stages WHERE job=?", (job_id,)).fetchall()
                return {"job_id": job_id, "state": job["state"], "route_id": None, "display_name": None,
                        "sequence": job["sequence"], "completed_stages": sum(r[0] == "completed" for r in rows),
                        "skipped_stages": sum(r[0] == "skipped" for r in rows), "total_stages": len(rows),
                        "cancel_requested": bool(job["cancel_requested"]),
                        "tokens_reserved": job["tokens_reserved"], "cost_reserved_microusd": job["cost_reserved"],
                        "cost_is_measured": False, "deadline_unix_ms": None,
                        "private_data_available": False, "retention_expired": True}

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
            state = "completed"
            if job["cancel_requested"] or job["state"] != "running":
                state = "cancelled"
            elif self._time() >= spec.limits.deadline_unix_ms:
                state = "expired"
            elif result["tool_calls"]:
                state = "waiting_tools"
            keep = state in ("completed", "waiting_tools")
            encrypted = self._cipher.seal(owner, job_id, stage_id, self._expiry(job_id), encoded) if keep else None
            db.execute("UPDATE stages SET state=?,result=?,checksum=? WHERE job=? AND id=?", (
                state, encrypted, hashlib.sha256(encrypted).hexdigest() if keep else None, job_id, stage_id))
            self._event(db, job_id, "stage_completed" if state == "completed" else "stage_failed", stage_id)
            return state

    def finish(self, owner, job_id, runner, epoch, state):
        try:
            return super().finish(owner, job_id, runner, epoch, state)
        except RetentionExpired:
            with self._transaction() as db:
                job = self._owned(db, owner, job_id)
                self._runner(job, runner, epoch)
                need(self._time() >= self._expiry(job_id, allow_expired=True))
                need(db.execute("SELECT COUNT(*) FROM stages WHERE job=? AND state='running'", (job_id,)).fetchone()[0] == 0)
                # Expiration must not strand runner ownership or admit late success.
                final = "cancelled" if job["cancel_requested"] else "expired"
                db.execute("UPDATE jobs SET state=?,runner=NULL WHERE id=?", (final, job_id))
                self._event(db, job_id, "job_" + final)
                return final

    def rotate_job_key(self, grant, job_id, *, administration_authorized=False):
        """Explicit maintenance; preserves plan bytes, retention deadline and job state.

        This rewrites ciphertext under the current key-service selection. It does
        not delete old keys, prove backup erasure or restore a revoked key.
        """
        need(administration_authorized is True and type(grant) is ExecutionGrant)
        with self._transaction() as db:
            job = self._owned(db, grant.owner_id, job_id)
            need(job["runner"] is None)
            expiry = self._expiry(job_id)
            spec = self._spec(job)
            encoded = self._cipher.seal(grant.owner_id, job_id, "spec", expiry, spec.encoded)
            db.execute("UPDATE jobs SET spec=? WHERE id=?", (encoded, job_id))
            for row in db.execute("SELECT * FROM stages WHERE job=? AND result IS NOT NULL", (job_id,)).fetchall():
                result = self._result(row)
                value = self._cipher.seal(grant.owner_id, job_id, row["id"], expiry, canonical(result))
                db.execute("UPDATE stages SET result=?,checksum=? WHERE job=? AND id=?",
                           (value, hashlib.sha256(value).hexdigest(), job_id, row["id"]))

    def delete_job(self, grant, job_id, *, reason, quiescence_receipt=None):
        """Authenticated deletion requires terminal, unowned work and a server policy.

        The callback must verify real owner authorization and, for uncertain/failed
        started work, supervisor/provider quiescence. This API never accepts a
        model's request as permission to delete. Only that job is affected.
        """
        need(type(grant) is ExecutionGrant and reason in ("owner_request", "retention_expired"))
        with self._transaction() as db:
            job = self._owned(db, grant.owner_id, job_id)
            need(job["state"] in TERMINAL and job["runner"] is None)
            expiry = self._expiry(job_id, allow_expired=True)
            if reason == "retention_expired":
                need(self._time() >= expiry)
            uncertain = job["tokens_reserved"] > 0 and job["state"] != "succeeded"
            if uncertain:
                identifier(quiescence_receipt, "verified quiescence receipt")
            try:
                allowed = self._deletion_authorized(grant.owner_id, job_id, reason, quiescence_receipt)
            except Exception:
                raise RecordProtectionError("private deletion authorization unavailable") from None
            need(allowed is True)
            # Retain a minimal idempotency tombstone; deletion does not authorize
            # the same request to be admitted/executed again or refund its cost.
            db.execute("INSERT INTO private_job_tombstones VALUES (?,?,?)", (grant.owner_id, job["idem"], self._time()))
            db.execute("DELETE FROM jobs WHERE id=? AND owner=?", (job_id, grant.owner_id))
            return {"deleted": True, "idempotency_tombstone_retained": True,
                    "physical_backup_erasure_claimed": False, "billing_refund_claimed": False}
