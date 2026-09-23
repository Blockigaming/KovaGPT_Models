"""Synthetic fixtures for the source-only execution tests; no provider calls."""

import json
from threading import Lock
from time import time_ns

from core.adapter import build_core_plan
from execution.contracts import ExecutionGrant, ExecutionLimits, ExecutionSpec
from execution.workers import ModelStageWorker
from ultra.orchestrator import build_ultra_plan
from worker.azure_container_apps import AzureResponse, AzureSettings, make_azure_inference_client
from worker.handler import CORE_SERVING, PINNED_CORE_CANDIDATES
from release.model_revisions import source_reference_for_route


SYNTHETIC_ADAPTER_SHA256 = "f" * 64
# This fixture is imported only by tests. No trained adapter exists in the
# source registry; bind synthetic identities solely for in-memory execution.
for _candidate in CORE_SERVING["candidates"]:
    _candidate["adapter_sha256"] = SYNTHETIC_ADAPTER_SHA256
    PINNED_CORE_CANDIDATES[_candidate["id"]]["adapter_sha256"] = SYNTHETIC_ADAPTER_SHA256
CANDIDATE = CORE_SERVING["candidates"][0]
IDENTITY = {"model": CANDIDATE["model"], "model_revision": CANDIDATE["revision"],
            "context_tokens": CANDIDATE["context_tokens"], "adapter_sha256": SYNTHETIC_ADAPTER_SHA256}
OWNER = "fixture-owner"


def tokens(_model, messages):
    # Approximate deterministic fixture counter only, NOT a real model tokenizer.
    return max(1, sum((len(m["content"]) + 3) // 4 for m in messages))


def make_plan(route="high", *, agents=3, task="Prove this equation."):
    if route == "ultra" or route.endswith(":ultra"):
        request = {"request_id": "fixture-correlation", "task": task}
        if route == "ultra":
            request["route_id"] = route
        else:
            surface, family, _ = route.split(":")
            request.update(surface=surface, family=family, effort="Ultra")
        return build_ultra_plan(request, admission={
            "entitlement": "pro", "ultra_authorized": True, "remaining_usd": 1,
            "estimated_max_usd": 0.5, "max_agents": agents, "max_total_tokens": 65536,
        }, token_counter=lambda msgs: tokens(IDENTITY["model"], msgs))
    request = {"request_id": "fixture-correlation", "messages": [{"role": "user", "content": "PRIVATE input text"}]}
    if route.startswith(("work:", "chat:")):
        surface, family, effort = route.split(":")
        request.update(surface=surface, family=family, effort=effort.replace("-", " ").title())
    else:
        request["route_id"] = route
    return build_core_plan(request, candidate_model=source_reference_for_route(route).slot,
                           token_counter=tokens)


def make_spec(route="high", *, deadline=None, parallel=3, agents=3, task="Prove this equation."):
    plan = make_plan(route, agents=agents, task=task)
    ids = [op.get("stage_id", op.get("id")) for op in plan["operations"]]
    reserved = sum(op["maximum_input_tokens"] + op["maximum_output_tokens"] for op in plan["operations"])
    runtime_identity = ({**IDENTITY, "model": plan["candidate_model"],
                         "model_revision": plan["candidate_revision"]}
                        if plan["engine"] == "kova-core" else {**IDENTITY, "model": source_reference_for_route(route).slot,
                                                         "model_revision": source_reference_for_route(route).revision})
    return ExecutionSpec.from_plan(plan, limits=ExecutionLimits(
        deadline_unix_ms=deadline or time_ns() // 1_000_000 + 60_000,
        token_limit=reserved, cost_limit_microusd=len(ids) * 100,
        max_parallel=parallel, stage_timeout_seconds=5,
    ), runtime_identity=runtime_identity, stage_cost_caps={stage: 100 for stage in ids})


def grant_for(spec, *, owner=OWNER, tier="pro", enabled=True):
    return ExecutionGrant(owner, tier, frozenset((spec.plan["route_id"],)), enabled)


class ModelFixture:
    def __init__(self, *, debate=False, hook=None):
        self.debate = debate
        self.hook = hook
        self.calls = []
        self.requests = []
        self.records = []
        self.closed = []
        self.lock = Lock()
        self.override = {}
        self.active_identity = IDENTITY

    def probe(self, stage, phase):
        return {
            "source": "server_provider_runtime", "worker_lifecycle_id": "fixture-lifecycle",
            "loaded_model": self.active_identity["model"],
            "loaded_model_revision": self.active_identity["model_revision"],
            "loaded_adapter_sha256": self.active_identity["adapter_sha256"],
            "cold_start": False, "worker_start_ms": 0, "model_load_ms": 0, "queue_ms": 0,
            "gpu_rate_per_second_usd": 0.001, "gpu_type_id": "fixture-no-real-gpu", "gpu_count": 1,
            "serving_engine": "vllm", "endpoint_type": "load_balancing",
            "container_image_digest": "sha256:" + "a" * 64,
        }

    def content(self, stage):
        if stage == "disagreement-check":
            return json.dumps({"disagreements": [{"id": "d1", "stage_ids": ["specialist-1", "specialist-2"],
                                                  "summary": "Different fixture conclusions"}] if self.debate else []})
        if stage == "judge":
            return json.dumps({"material_disagreement": self.debate, "disagreement_ids": ["d1"] if self.debate else [],
                               "summary": "Fixture evidence judgment"})
        if stage == "synthesis":
            return "Kova final response"
        return "PRIVATE stage result " + stage

    def client_factory(self, control, identity):
        self.active_identity = dict(identity)
        stage = control.stage_id
        def transport(request, _headers):
            with self.lock:
                self.calls.append(stage)
                self.requests.append((stage, json.loads(request.body)))
            if self.hook:
                self.hook(stage, control)
            content = self.override.get(stage, "Kova final response" if request.stream else self.content(stage))
            if isinstance(content, BaseException):
                raise content
            if request.stream:
                values = [
                    {"model": identity["model"], "choices": [{"delta": {"content": content}}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
                ]
                body = ("".join("data: " + json.dumps(v) + "\n\n" for v in values) + "data: [DONE]\n\n").encode()
            else:
                body = json.dumps({"model": identity["model"],
                                   "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 10, "completion_tokens": 3}}).encode()
            return AzureResponse(200, "text/event-stream" if request.stream else "application/json", request.url,
                                 body, lambda: self.closed.append(stage))
        return make_azure_inference_client(AzureSettings(
            "https://fixture.internal.test.azurecontainerapps.io", identity["model"],
            min(5, control.remaining_seconds()), True, True, True,
        ), transport, lambda: "synthetic.fixture.token", cancelled=control.cancelled)

    def worker(self):
        return ModelStageWorker(self.client_factory, self.probe, tokens, self.records.append)
