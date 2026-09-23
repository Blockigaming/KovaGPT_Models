import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from core.adapter import CANDIDATES, IDENTITY, bind_core_operation, build_core_plan
from core.identity import APPROVED_PROMPT_SHA256, PROMPT_PATH
from release.model_revisions import source_reference_for_route


class CoreAdapterTests(unittest.TestCase):
    def request(self, **overrides):
        value = {"request_id": "request-1", "route_id": "instant", "messages": [{"role": "user", "content": "hello"}]}
        value.update(overrides)
        return value

    def model(self):
        return "kova-cosmo"

    def build(self, request=None, model=None, token_count=100):
        request = request or self.request()
        route_id = request.get("route_id") or (
            f"{request['surface']}:{request['family']}:{request['effort'].lower().replace(' ', '-')}")
        return build_core_plan(
            request, candidate_model=model or source_reference_for_route(route_id).slot,
            token_counter=lambda _model, _messages: token_count,
        )

    def test_verified_candidate_registry_is_loaded_from_architecture(self):
        self.assertIn(self.model(), CANDIDATES)
        self.assertEqual(CANDIDATES[self.model()]["context_tokens"], 32768)

    def test_canonical_route_cannot_run_a_different_family(self):
        with self.assertRaisesRegex(ValueError, "candidate differs from authoritative family route"):
            self.build(self.request(route_id="medium"), model="kova-cosmo")
        with self.assertRaisesRegex(ValueError, "candidate differs from authoritative family route"):
            self.build({"request_id": "r", "messages": [{"role": "user", "content": "hi"}],
                        "surface": "work", "family": "nova", "effort": "High"},
                       model="kova-orion")

    def test_instant_is_one_streaming_operation(self):
        plan = self.build()
        provenance = plan["operations"][0]["request_template"]["messages"][2]["content"]
        self.assertIn("Qwen/Qwen3-0.6B", provenance)
        self.assertIn("when asked", provenance)
        self.assertEqual(len(plan["operations"]), 1)
        operation = plan["operations"][0]
        self.assertEqual(operation["stage_id"], "answer-1")
        self.assertEqual(plan["display_name"], "Kova Cosmo")
        self.assertTrue(operation["public_response"])
        self.assertTrue(operation["request_template"]["stream"])
        self.assertFalse(operation["activity_event_allowed_after_start"])

    def test_max_has_real_multi_pass_operations_and_only_final_streams(self):
        plan = self.build(self.request(route_id="max"))
        self.assertEqual(len(plan["operations"]), 10)
        self.assertTrue(all(not item["public_response"] for item in plan["operations"][:-1]))
        self.assertTrue(plan["operations"][-1]["public_response"])
        self.assertEqual(plan["operations"][-1]["phase"], "verification")
        self.assertEqual(plan["operations"][0]["depends_on_stage_ids"], [])
        self.assertEqual(plan["operations"][1]["depends_on_stage_ids"], [plan["operations"][0]["stage_id"]])
        self.assertEqual(plan["operations"][1]["input_context"], "conversation_plus_prior_private_artifacts")
        self.assertGreater(plan["operations"][1]["reserved_prior_artifact_tokens"], 0)
        self.assertEqual(
            plan["operations"][1]["request_template"]["artifact_bindings"][0]["source_stage_id"],
            plan["operations"][0]["stage_id"],
        )
        binding = plan["operations"][1]["request_template"]["artifact_bindings"][0]
        messages = plan["operations"][1]["request_template"]["messages"]
        self.assertEqual(messages[binding["target_message_index"]]["role"], "assistant")
        self.assertIn("UNTRUSTED PRIOR MODEL OUTPUT", messages[binding["target_message_index"]]["content"])
        self.assertIn(binding["placeholder"], messages[binding["target_message_index"]]["content"])
        self.assertTrue(binding["replace_exact_target_only"])
        self.assertTrue(plan["operations"][1]["request_template"]["reject_placeholder_outside_binding_targets"])

    def test_identity_model_effort_and_limits_are_server_controlled(self):
        plan = self.build(self.request(route_id="medium"))
        provider_request = plan["operations"][-1]["request_template"]
        self.assertEqual(provider_request["messages"][0]["content"], IDENTITY)
        self.assertEqual(IDENTITY, PROMPT_PATH.read_text(encoding="utf-8"))
        self.assertIn("Cosmo, Orion, and Nova are Kova model families", IDENTITY)
        self.assertEqual(len(APPROVED_PROMPT_SHA256), 64)
        self.assertEqual(provider_request["reasoning_effort"], "medium")
        self.assertEqual(provider_request["max_completion_tokens"], 4096)
        with self.assertRaisesRegex(ValueError, "server-controlled"):
            self.build(self.request(model="attacker/model"))

    def test_identity_prompt_change_after_import_rejects_core_plan(self):
        with TemporaryDirectory() as temporary:
            changed = Path(temporary) / "prompt.txt"
            changed.write_bytes(PROMPT_PATH.read_bytes() + b"\nUNTRUSTED CHANGE")
            with patch("core.identity.PROMPT_PATH", changed), self.assertRaisesRegex(ValueError, "identity prompt mismatch"):
                self.build()

    def test_ultra_and_unknown_models_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "not eligible"):
            self.build(self.request(route_id="ultra"))
        with self.assertRaisesRegex(ValueError, "unverified"):
            self.build(model="attacker/model")

    def test_no_operation_is_production_ready(self):
        plan = self.build(self.request(route_id="high"))
        self.assertFalse(plan["production_ready"])
        self.assertTrue(all(operation["request_template"]["store"] is False for operation in plan["operations"]))
        self.assertTrue(plan["executor_contract"]["trusted_token_recount_after_binding"])

    def test_trusted_token_count_is_required_and_context_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "token counter"):
            build_core_plan(self.request(), candidate_model=self.model(), token_counter=None)
        with self.assertRaisesRegex(ValueError, "exceeds candidate context"):
            self.build(token_count=262144)

    def test_worst_case_prior_artifacts_are_in_context_budget(self):
        plan = self.build(self.request(route_id="max"), token_count=100)
        candidate_context = CANDIDATES[self.model()]["context_tokens"]
        for operation in plan["operations"]:
            self.assertLessEqual(
                operation["maximum_input_tokens"] + operation["maximum_output_tokens"],
                candidate_context,
            )
        self.assertEqual(plan["provider"], "runpod_serverless")
        self.assertEqual(plan["endpoint_name"], "kova-core")
        self.assertFalse(plan["endpoint_deployed"])

    def test_qwen_thinking_mode_tracks_kova_route(self):
        instant = self.build()
        high = self.build(self.request(route_id="high"))
        self.assertFalse(instant["operations"][0]["request_template"]["chat_template_kwargs"]["enable_thinking"])
        self.assertTrue(high["operations"][0]["request_template"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(high["operations"][0]["request_template"]["reasoning_effort"], "xhigh")
        self.assertFalse(high["operations"][0]["request_template"]["chat_template_kwargs"]["preserve_thinking"])

    def test_executor_binds_only_exact_declared_artifact_targets(self):
        plan = self.build(self.request(route_id="medium"))
        request = bind_core_operation(
            plan, "answer-1", {"planning-1": "server-recorded plan"},
            token_counter=lambda _model, _messages: 110,
        )
        joined = "\n".join(message["content"] for message in request["messages"])
        self.assertIn("UNTRUSTED PRIOR MODEL OUTPUT (planning-1)", joined)
        self.assertIn("server-recorded plan", joined)
        self.assertNotIn("{{server_stage_output:", joined)
        self.assertFalse(request["stream"])
        self.assertNotIn("stream_options", request)

    def test_executor_rejects_missing_artifacts_and_bound_token_overflow(self):
        plan = self.build(self.request(route_id="medium"))
        with self.assertRaisesRegex(ValueError, "do not match Core DAG"):
            bind_core_operation(plan, "answer-1", {}, token_counter=lambda _model, _messages: 110)
        with self.assertRaisesRegex(ValueError, "exceeds reserved maximum"):
            bind_core_operation(
                plan, "answer-1", {"planning-1": "plan"},
                token_counter=lambda _model, _messages: 999_999,
            )

    def test_work_routes_bind_distinct_family_behavior(self):
        plans = {}
        for family in ("cosmo", "orion", "nova"):
            request = {
                "request_id": f"request-{family}", "messages": [{"role": "user", "content": "draft report"}],
                "surface": "work", "family": family, "effort": "Medium",
            }
            plans[family] = self.build(request)
        self.assertEqual(len({plan["behavior_contract_id"] for plan in plans.values()}), 3)
        family_prompts = {
            plan["operations"][0]["request_template"]["messages"][1]["content"]
            for plan in plans.values()
        }
        self.assertEqual(len(family_prompts), 3)


if __name__ == "__main__":
    unittest.main()
