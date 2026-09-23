"""Kova handler/plan integration with Azure-shaped in-memory HTTP responses."""

from dataclasses import replace
import json
import unittest

from core.adapter import build_core_plan
from router.policy import CHAT_POLICIES, WORK_FAMILIES, WORK_EFFORTS
from worker.azure_container_apps import AzureResponse, AzureSettings, make_azure_inference_client
from worker.handler import PINNED_CORE_CANDIDATES, handle_job
from release.model_revisions import source_reference_for_route


class AzureKovaIntegrationTests(unittest.TestCase):
    candidate = PINNED_CORE_CANDIDATES["kova-cosmo"]

    def setUp(self):
        self.closed = 0
        self.captured = []
        self.records = []
        self.config = AzureSettings(
            "https://kova-core.internal.fixture.eastus.azurecontainerapps.io",
            self.candidate["model"], 30, True, True, True,
        )
        self.json_body = {
            "model": self.candidate["model"],
            "choices": [{"message": {"content": "Kova fixture answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }
        self.stream_values = [
            {"choices": [{"delta": {"content": "Kova fixture answer"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
        ]

    def runtime(self, phase):
        self.assertIn(phase, ("before", "after"))
        return {
            "source": "server_provider_runtime", "worker_lifecycle_id": "azure-fixture-lifecycle",
            "loaded_model": self.candidate["model"], "loaded_model_revision": self.candidate["model_revision"],
            "cold_start": False, "worker_start_ms": 0, "model_load_ms": 0, "queue_ms": 0,
            "gpu_rate_per_second_usd": 0.001, "gpu_type_id": "fixture-not-real-gpu", "gpu_count": 1,
            "serving_engine": "vllm", "endpoint_type": "load_balancing",
            "container_image_digest": "sha256:" + "a" * 64,
        }

    def transport(self, plan, headers):
        self.captured.append((plan, headers))
        if plan.stream:
            body = "".join("data: " + json.dumps(value) + "\n\n" for value in self.stream_values)
            body += "data: [DONE]\n\n"
        else:
            body = json.dumps(self.json_body)
        def close():
            self.closed += 1
        return AzureResponse(200, "text/event-stream" if plan.stream else "application/json",
                             plan.url, body.encode(), close)

    def client(self):
        return make_azure_inference_client(self.config, self.transport, lambda: "fixture.token", clock=lambda: 0)

    def execute(self, plan, operation, prior, *, client=None):
        value = {
            "request_id": "azure-fixture-request",
            "messages": [{"role": "user", "content": "Help with this fixture"}],
            "reasoning_effort": operation["request_template"]["reasoning_effort"],
            "max_output_tokens": operation["maximum_output_tokens"],
        }
        times = iter((0, 10_000_000, 20_000_000))
        return handle_job(
            {"input": value}, self.client() if client is None else client, self.runtime, self.records.append,
            execution_context={
                "logical_request_id": "kova-exec-00000000-0000-4000-8000-000000000001",
                "benchmark_candidate_id": self.candidate["id"], "route_id": plan["route_id"],
                "stage_id": operation["stage_id"], "public_response": operation["public_response"],
                "prior_stage_outputs": prior,
            },
            token_counter=lambda *_: 10, clock_ns=lambda: next(times),
            attempt_id_factory=lambda: "azure-fixture-attempt",
        )

    def plan(self, selection):
        route = selection.get("route_id") or (
            f"{selection['surface']}:{selection['family']}:{selection['effort'].lower().replace(' ', '-')}")
        self.candidate = PINNED_CORE_CANDIDATES[source_reference_for_route(route).slot]
        self.config = replace(self.config, served_model=self.candidate["model"])
        self.json_body["model"] = self.candidate["model"]
        return build_core_plan(
            {"request_id": "azure-fixture-request", "messages": [{"role": "user", "content": "Help with this fixture"}], **selection},
            candidate_model=self.candidate["model"], token_counter=lambda *_: 10,
        )

    def test_all_twenty_core_profiles_pass_every_stage_through_azure_boundary(self):
        selections = [{"route_id": route} for route in CHAT_POLICIES if route != "ultra"]
        selections.extend({"surface": "work", "family": family, "effort": effort}
                          for family in sorted(WORK_FAMILIES) for effort in WORK_EFFORTS if effort != "Ultra")
        self.assertEqual(len(selections), 20)
        stages_checked = 0
        for selection in selections:
            with self.subTest(selection=selection):
                plan = self.plan(selection)
                outputs = {}
                for operation in plan["operations"]:
                    prior = {stage: outputs[stage] for stage in operation["depends_on_stage_ids"]}
                    result = self.execute(plan, operation, prior)
                    outputs[operation["stage_id"]] = result["content"]
                    body = json.loads(self.captured[-1][0].body)
                    self.assertNotIn("input", body)
                    self.assertTrue(body["messages"][0]["content"].startswith("You are Kova"))
                    self.assertEqual(body["max_tokens"], operation["maximum_output_tokens"])
                    self.assertEqual(body["chat_template_kwargs"], operation["request_template"]["chat_template_kwargs"])
                    self.assertEqual(self.records[-1]["outcome"], "success")
                    if operation["public_response"]:
                        self.assertEqual(result["benchmark"]["time_to_first_token_ms"], 10)
                    else:
                        self.assertIsNone(result["benchmark"]["time_to_first_token_ms"])
                    stages_checked += 1
        self.assertEqual(self.closed, stages_checked)
        self.assertEqual(len(self.captured), stages_checked)

    def test_hidden_reasoning_rejection_closes_even_a_retained_stream(self):
        self.stream_values[0]["choices"][0]["delta"]["reasoning_content"] = "private"
        plan = self.plan({"route_id": "instant"})
        retained = []
        base_client = self.client()
        def retaining_client(request):
            response = base_client(request)
            retained.append(response)
            return response
        try:
            with self.assertRaisesRegex(ValueError, "hidden reasoning"):
                self.execute(plan, plan["operations"][0], {}, client=retaining_client)
            self.assertEqual(self.closed, 1)
            self.assertEqual(self.records[-1]["outcome"], "failed")
        finally:
            for response in retained:
                response.close()

    def test_public_tool_call_fragments_are_assembled_but_not_executed(self):
        self.stream_values = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                "function": {"name": "look", "arguments": '{"q":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0,
                "function": {"name": "up", "arguments": '"Kova"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
        ]
        plan = self.plan({"route_id": "instant"})
        result = self.execute(plan, plan["operations"][0], {})
        self.assertEqual(result["tool_calls"][0]["function"], {"name": "lookup", "arguments": '{"q":"Kova"}'})
        self.assertEqual(result["content"], "")
        self.assertEqual(self.closed, 1)

    def test_runtime_revision_changes_are_quarantined(self):
        plan = self.plan({"route_id": "instant"})
        original_runtime = self.runtime
        def changed(phase):
            probe = original_runtime(phase)
            if phase == "after":
                probe["loaded_model_revision"] = "0" * 40
            return probe
        self.runtime = changed
        with self.assertRaisesRegex(ValueError, "revision"):
            self.execute(plan, plan["operations"][0], {})
        self.assertEqual(self.records[-1]["outcome"], "quarantined")
        self.assertEqual(self.closed, 1)

    def test_unknown_free_thinking_and_ultra_are_not_silently_mapped_to_core(self):
        for route in ("thinking", "ultra", "kova-auto"):
            with self.subTest(route=route), self.assertRaises(ValueError):
                self.plan({"route_id": route})


if __name__ == "__main__":
    unittest.main()
