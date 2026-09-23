"""Core and Ultra model-stage execution over explicitly injected guarded clients.

No model/provider is selected here. The same existing pinned-candidate allowlist
is usable for offline fixtures; an Ultra production worker is still unselected.
Neither tools nor arbitrary model-supplied commands are executed by this module.
"""

from time import perf_counter_ns

from core.adapter import build_core_plan
from release.model_revisions import source_reference_for_route
from execution.contracts import ExecutionError, ExecutionIntegrityError, canonical, require
from ultra.orchestrator import build_ultra_plan
from ultra.binding import bind_ultra_operation, judge_requires_debate, validate_disagreements
from worker.handler import (
    CORE_SERVING, PINNED_CORE_CANDIDATES, RUNTIME_IDENTITY_FIELDS, consume_engine_response,
    handle_job, sanitize_engine_response, validate_runtime_probe,
)


def _verify_current_plan(plan, identity, token_counter):
    """Reject stale snapshots before constructing a provider client.

    Core's handler rebuilds operations from current source. A restarted job must
    not silently mix completed old-policy artifacts with newly generated policy.
    Reconstructing admission for the pure Ultra planner starts no execution and
    does not replace the runner's independently checked current server grant.
    """
    route = plan["route_id"]
    selection = {"route_id": route}
    if route.startswith("work:"):
        _, family, effort = route.split(":")
        selection = {"surface": "work", "family": family,
                     "effort": effort.replace("-", " ").title()}
    if plan["engine"] == "kova-core":
        rebuilt = build_core_plan(
            {"request_id": plan["request_id"],
             "messages": plan["operations"][0]["request_template"]["messages"][3:],
             **selection},
            candidate_model=identity["model"], token_counter=token_counter,
        )
    else:
        # New snapshots reconstruct from the full history, never only the latest
        # task. Old task-only snapshots keep their exact original schema.
        content = ({"messages": plan["conversation_messages"]} if "conversation_messages" in plan
                   else {"task": plan["task"]})
        rebuilt = build_ultra_plan(
            {"request_id": plan["request_id"], **content, **selection},
            admission={
                "entitlement": "pro", "ultra_authorized": True,
                "remaining_usd": plan["estimated_max_usd"],
                "estimated_max_usd": plan["estimated_max_usd"],
                "max_agents": sum(op["id"].startswith("specialist-") for op in plan["operations"]),
                "max_total_tokens": plan["maximum_total_tokens"],
            },
            token_counter=lambda messages: token_counter(identity["model"], messages),
        )
    if canonical(plan) != canonical(rebuilt):
        raise ExecutionIntegrityError("saved execution plan differs from the current planning contract")


class ModelStageWorker:
    """Trusted dependencies must be thread-safe or scoped per stage.

    client_factory(control, identity) must apply control.remaining_seconds() and
    control.cancelled to every network request; arbitrary noncooperative callbacks
    cannot be forcefully stopped by the Python scheduler. runtime_probe receives
    (stage_id, phase) and must observe the real worker, not client/model assertions.
    """

    def __init__(self, client_factory, runtime_probe, token_counter, telemetry_sink,
                 *, clock_ns=perf_counter_ns):
        require(all(callable(x) for x in (client_factory, runtime_probe, token_counter, telemetry_sink, clock_ns)),
                "trusted worker dependencies required")
        self.client_factory = client_factory
        self.runtime_probe = runtime_probe
        self.token_counter = token_counter
        self.telemetry_sink = telemetry_sink
        self.clock_ns = clock_ns

    def __call__(self, spec, stage, artifacts, control, logical_id, attempt):
        control.check()
        require(stage in spec.stages, "worker stage differs from its execution snapshot")
        plan = spec.plan
        identity = spec.snapshot()["runtime_identity"]
        source = source_reference_for_route(plan["route_id"])
        require(identity["model"] == source.slot and identity["model_revision"] == source.revision,
                "worker identity differs from selected route family")
        candidates = [c for c in PINNED_CORE_CANDIDATES.values()
                      if c["model"] == identity["model"] and c["model_revision"] == identity["model_revision"]]
        require(len(candidates) == 1, "worker model/revision not in the pinned candidate allowlist")
        candidate = candidates[0]
        serving = next(c for c in CORE_SERVING["candidates"] if c["id"] == candidate["id"])
        require(identity["context_tokens"] <= serving["context_tokens"], "served context exceeds candidate context")
        _verify_current_plan(plan, identity, self.token_counter)
        client = self.client_factory(control, dict(identity))
        require(callable(client), "guarded inference client required")
        control.check()
        records = []
        try:
            if plan["engine"] == "kova-core":
                operation = next(op for op in plan["operations"] if op["stage_id"] == stage.id)
                # The first Core stage has no artifact messages; its first three
                # system messages are regenerated by handle_job, not accepted as
                # caller-controlled instructions.
                messages = plan["operations"][0]["request_template"]["messages"][3:]
                result = handle_job(
                    {"input": {"request_id": plan["request_id"], "messages": messages,
                               "reasoning_effort": operation["request_template"]["reasoning_effort"],
                               "max_output_tokens": stage.maximum_output_tokens}},
                    client, lambda phase: self.runtime_probe(stage.id, phase), records.append,
                    execution_context={
                        "logical_request_id": logical_id, "benchmark_candidate_id": candidate["id"],
                        "route_id": plan["route_id"], "stage_id": stage.id,
                        "public_response": stage.public,
                        "prior_stage_outputs": {key: value["content"] for key, value in artifacts.items()},
                    }, token_counter=self.token_counter, clock_ns=self.clock_ns, attempt_id_factory=lambda: attempt,
                )
            else:
                result = self._ultra(plan, stage, artifacts, identity, candidate, client,
                                     records, logical_id, attempt)
            control.check()
            usage = result["usage"]
            require(0 < usage["prompt_tokens"] <= stage.maximum_input_tokens
                    and 0 < usage["completion_tokens"] <= stage.maximum_output_tokens,
                    "provider usage exceeds the reserved stage budget")
            decision = None
            if plan["engine"] == "kova-ultra":
                specialists = {s.id for s in spec.stages if s.id.startswith("specialist-")}
                if stage.id == "disagreement-check":
                    require(not result["tool_calls"], "disagreement stage must produce structured evidence")
                    validate_disagreements(result["content"], specialists)
                elif stage.id == "judge":
                    require(not result["tool_calls"], "judge must produce a structured decision")
                    decision = judge_requires_debate(result["content"],
                                                     artifacts["disagreement-check"]["content"], specialists)
            return {
                "content": result["content"], "tool_calls": result["tool_calls"],
                "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
                "debate_required": decision,
            }
        except BaseException:
            for record in records:
                if record["outcome"] == "success":
                    record["outcome"] = "quarantined"
            # A nested transport may deliberately redact arbitrary exceptions.
            # Recheck server controls so a real expiry/revocation/cancellation
            # remains distinguishable instead of becoming a generic model failure.
            control.check()
            raise
        finally:
            for record in records:
                self.telemetry_sink(record)

    def _ultra(self, plan, stage, artifacts, identity, candidate, client, records, logical_id, attempt):
        request = bind_ultra_operation(
            plan, stage.id, {key: None if value is None else value["content"] for key, value in artifacts.items()},
            runtime_identity=identity, token_counter=self.token_counter,
        )
        before = validate_runtime_probe(lambda phase: self.runtime_probe(stage.id, phase), "before", candidate)
        started_ns = self.clock_ns()
        timing = {"time_to_first_token_ms": None}
        normalized = None
        result = None
        failure = None
        finished_ns = None
        try:
            response = client(request)
            normalized, finished_ns = consume_engine_response(
                response, expect_stream=stage.public, clock_ns=self.clock_ns,
                started_ns=started_ns, timing_state=timing,
            )
            result = sanitize_engine_response(plan["request_id"], normalized)
        except BaseException as error:
            failure = error
        if finished_ns is None:
            finished_ns = self.clock_ns()
        outcome = "success" if failure is None else "failed"
        try:
            after = validate_runtime_probe(lambda phase: self.runtime_probe(stage.id, phase), "after", candidate)
            require(all(before[key] == after[key] for key in RUNTIME_IDENTITY_FIELDS), "Ultra runtime identity changed")
            require(type(finished_ns) is int and type(started_ns) is int and finished_ns >= started_ns,
                    "invalid Ultra runtime clock")
            elapsed = (finished_ns - started_ns) / 1_000_000
            first = timing["time_to_first_token_ms"]
            require(first is None or 0 <= first <= elapsed, "invalid Ultra first-token timing")
            require(stage.public or first is None, "private Ultra stage cannot report first-token timing")
            require(failure is not None or not stage.public or first is not None,
                    "public Ultra stage missing first-visible output")
        except BaseException:
            outcome = "quarantined"
            if failure is None:
                failure = ExecutionIntegrityError("Ultra runtime integrity verification failed")
        usage = (normalized or {}).get("usage") or {}
        records.append({
            "record_type": "attempt", "request_id": logical_id, "correlation_id": plan["request_id"],
            "attempt_id": attempt, "outcome": outcome, "route_id": plan["route_id"], "stage_id": stage.id,
            "public_response": stage.public, "engine": "kova-ultra",
            "model": before["loaded_model"], "model_revision": before["loaded_model_revision"],
            "worker_lifecycle_id": before["worker_lifecycle_id"], "measurement_source": before["source"],
            "container_image_digest": before["container_image_digest"],
            "inference_ms": max(0, finished_ns - started_ns) / 1_000_000,
            "time_to_first_token_ms": timing["time_to_first_token_ms"],
            "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
            "cost_is_measured": False,
        })
        if failure is not None:
            raise failure
        return result
