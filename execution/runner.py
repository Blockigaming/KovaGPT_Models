"""Synchronous, bounded-width scheduler over the local reference journal.

Only specialist stages are parallel-ready in the existing Ultra DAG. This starts
real Python worker calls, not fake agent labels. It has no autonomous background
service, network defaults or production deployment. Workers MUST honor cooperative
cancellation/deadlines (the Azure transport does); Python cannot kill arbitrary
running threads. The runner drains started workers before releasing ownership.
"""

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
import math
from threading import Event
from time import monotonic

from execution.contracts import (
    ExecutionBlocked, ExecutionCancelled, ExecutionError, ExecutionExpired,
    ExecutionGrant, ExecutionIntegrityError, positive_integer, require,
)
from execution.store import TERMINAL, now_ms
from worker.azure_container_apps import AzureCancelled, AzureDeadlineExceeded


@dataclass(frozen=True)
class StageControl:
    """Checked before/after work and passed into every nested transport operation."""

    stage_id: str
    attempt_id: str
    check: object
    remaining_seconds: object
    cancelled: object


class LocalRunner:
    def __init__(self, store, authorization, worker, *, clock=monotonic, clock_ms=now_ms):
        require(all(callable(x) for x in (authorization, worker, clock, clock_ms)),
                "trusted execution dependencies required")
        self.store = store
        self.authorization = authorization
        self.worker = worker
        self.clock = clock
        self.clock_ms = clock_ms

    def _grant(self, owner=None, route=None):
        value = self.authorization()
        require(type(value) is ExecutionGrant, "current execution grant required")
        if owner is not None:
            value.authorize(owner, route)
        elif not value.execution_authorized:
            raise ExecutionBlocked("execution remains disabled")
        return value

    def run(self, job_id, *, max_stages=None):
        """Run to completion or a clean pause boundary; resume never repeats a stage.

        max_stages is a caller-controlled scheduling slice, not a model budget.
        Queue/pause time remains inside the original persisted job deadline.
        """
        require(max_stages is None or (type(max_stages) is int and max_stages > 0), "invalid stage slice")
        grant = self._grant()
        owner = grant.owner_id
        spec = self.store.server_spec(owner, job_id)
        route = spec.plan["route_id"]
        self._grant(owner, route)
        initial = self.store.status(owner, job_id)
        if initial["state"] in TERMINAL:
            return initial
        started_clock = self.clock()
        require(type(started_clock) in (int, float) and math.isfinite(started_clock), "invalid execution clock")
        started_wall_ms = self.clock_ms()
        positive_integer(started_wall_ms, "execution wall clock")
        # Validate both clocks before taking ownership; bad clock data must not
        # leave a queued job fenced to a runner that never entered its cleanup.
        deadline = started_clock + max(0, (spec.limits.deadline_unix_ms - started_wall_ms) / 1000)
        require(math.isfinite(deadline), "invalid execution deadline")
        lease = self.store.begin(grant, job_id)
        if lease is None:
            return self.store.status(owner, job_id)
        runner, epoch = lease
        stopped = Event()
        started_count = 0
        final_state = "failed"
        # Job expiry is not renewed by a new process or a resumed scheduler call.

        def global_check():
            if stopped.is_set():
                raise ExecutionCancelled("sibling execution stopped")
            self._grant(owner, route)  # Recheck downgrades/revocation while work is running.
            state, cancelled = self.store.control_state(owner, job_id, runner, epoch)
            if cancelled or state == "cancelling":
                raise ExecutionCancelled("job cancellation requested")
            require(state == "running", "execution is no longer running")
            now = self.clock()
            require(type(now) in (int, float) and math.isfinite(now) and now >= started_clock,
                    "invalid execution clock")
            wall_ms = self.clock_ms()
            positive_integer(wall_ms, "execution wall clock")
            if now >= deadline or wall_ms >= spec.limits.deadline_unix_ms:
                raise ExecutionExpired("original execution deadline exceeded")
            return now

        def perform(stage):
            attempt = None
            try:
                global_check()
                claimed = self.store.claim(self._grant(owner, route), job_id, runner, epoch, stage.id)
                if claimed is None:
                    return "skipped"
                attempt, artifacts = claimed
                stage_deadline = min(deadline, self.clock() + spec.limits.stage_timeout_seconds)
                def check():
                    now = global_check()
                    if now >= stage_deadline:
                        raise ExecutionExpired("stage deadline exceeded")
                    return now
                def remaining():
                    return stage_deadline - check()
                def cancelled():
                    try:
                        check()
                        return False
                    except ExecutionError:
                        return True
                control = StageControl(stage.id, attempt, check, remaining, cancelled)
                check()
                result = self.worker(spec, stage, artifacts, control, job_id, attempt)
                check()
                return self.store.complete(owner, job_id, runner, epoch, stage.id, attempt, result)
            except BaseException:
                if attempt is not None:
                    try:
                        self.store.fail_stage(owner, job_id, runner, epoch, stage.id, attempt)
                    except ExecutionError:
                        # An administrative recovery fence can reject late completions.
                        # Do not overwrite its interrupted state or replay this attempt.
                        pass
                raise

        try:
            with ThreadPoolExecutor(max_workers=spec.limits.max_parallel, thread_name_prefix="kova-stage") as pool:
                while True:
                    global_check()
                    frontier = self.store.frontier(owner, job_id, runner, epoch)["stages"]
                    if all(state in ("completed", "skipped") for state in frontier.values()):
                        final_state = "succeeded"
                        break
                    if max_stages is not None and started_count >= max_stages:
                        final_state = "paused"
                        break
                    ready = [s for s in spec.stages if frontier[s.id] == "pending"
                             and all(frontier[dep] in ("completed", "skipped") for dep in s.dependencies)]
                    require(bool(ready), "execution has no valid ready stage")
                    width = spec.limits.max_parallel
                    if max_stages is not None:
                        width = min(width, max_stages - started_count)
                    batch = ready[:width]
                    futures = {pool.submit(perform, s) for s in batch}
                    started_count += len(batch)
                    failure = None
                    while futures:
                        done, futures = wait(futures, timeout=0.05, return_when=FIRST_COMPLETED)
                        if failure is None:
                            try:
                                global_check()
                            except ExecutionError as error:
                                failure = error
                                stopped.set()
                        for future in done:
                            try:
                                outcome = future.result()
                                if outcome not in ("completed", "skipped") and failure is None:
                                    final_state = outcome if outcome in ("waiting_tools", "cancelled", "expired") else "failed"
                                    failure = ExecutionError("stage did not complete")
                                    stopped.set()
                            except BaseException as error:
                                if failure is None:
                                    failure = error
                                    stopped.set()
                    if failure is not None:
                        if isinstance(failure, (ExecutionExpired, AzureDeadlineExceeded)):
                            final_state = "expired"
                        elif isinstance(failure, (ExecutionCancelled, AzureCancelled)):
                            final_state = "cancelled"
                        elif final_state not in ("waiting_tools", "cancelled", "expired"):
                            final_state = "failed"
                        break
        except ExecutionExpired:
            final_state = "expired"
            stopped.set()
        except ExecutionCancelled:
            final_state = "cancelled"
            stopped.set()
        except BaseException:
            final_state = "failed"
            stopped.set()
            # Worker failures are persisted without potentially secret-bearing text.
            # This is a synchronous kernel, not a mechanism for background retries.
        try:
            self.store.finish(owner, job_id, runner, epoch, final_state)
        except ExecutionError:
            # A confirmed supervisor recovery may have already fenced this runner.
            status = self.store.status(owner, job_id)
            if status["state"] not in ("interrupted", "paused", "cancelled"):
                raise
        return self.store.status(owner, job_id)
