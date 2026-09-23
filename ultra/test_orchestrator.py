import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from core.current_candidates import NATIVE_CONTEXT_TOKENS
from core.identity import PROMPT_PATH
from ultra.orchestrator import IDENTITY, build_ultra_plan


class UltraPlannerTests(unittest.TestCase):
    def admission(self, **overrides):
        value = {
            "entitlement": "pro", "ultra_authorized": True, "remaining_usd": 2.0,
            "estimated_max_usd": 0.5, "max_agents": 5, "max_total_tokens": 32768,
        }
        value.update(overrides)
        return value

    def request(self, task="Research competitors, pricing, security, and produce a launch report."):
        return {"request_id": "request-1", "task": task, "route_id": "ultra"}

    def token_counter(self, messages):
        return max(1, sum((len(message["content"]) + 3) // 4 for message in messages))

    def build(self, request=None, **admission_overrides):
        return build_ultra_plan(
            request or self.request(), admission=self.admission(**admission_overrides),
            token_counter=self.token_counter,
        )

    def test_plan_uses_dynamic_bounded_specialists(self):
        plan = self.build()
        self.assertEqual(plan["display_name"], "Kova Orion — Ultra")
        specialists = [operation for operation in plan["operations"] if operation.get("parallel_group") == "specialists"]
        self.assertGreaterEqual(len(specialists), 2)
        self.assertLessEqual(len(specialists), 5)
        self.assertIn("research", plan["domains"])
        self.assertIn("business", plan["domains"])
        self.assertTrue(set(plan["domains"]).issubset(plan["covered_domains"]))

    def test_ultra_uses_approved_prompt_on_every_operation(self):
        self.assertEqual(IDENTITY, PROMPT_PATH.read_text(encoding="utf-8"))
        plan = self.build()
        self.assertTrue(all("Qwen/Qwen3-1.7B" in operation["input_template"]["messages"][2]["content"]
                            for operation in plan["operations"]))
        self.assertTrue(all(operation["input_template"]["messages"][0] ==
                            {"role": "system", "content": IDENTITY} for operation in plan["operations"]))

    def test_identity_prompt_change_after_import_rejects_ultra_plan(self):
        with TemporaryDirectory() as temporary:
            changed = Path(temporary) / "prompt.txt"
            changed.write_bytes(PROMPT_PATH.read_bytes() + b"\nUNTRUSTED CHANGE")
            with patch("core.identity.PROMPT_PATH", changed), self.assertRaisesRegex(ValueError, "identity prompt mismatch"):
                self.build()

    def test_judge_debate_and_synthesis_have_real_dependencies(self):
        plan = self.build(self.request("Research competitors and compare market pricing."), max_agents=3)
        by_id = {operation["id"]: operation for operation in plan["operations"]}
        self.assertIn("disagreement-check", by_id)
        self.assertEqual(len(by_id["disagreement-check"]["depends_on"]), 3)
        self.assertEqual(len(by_id["judge"]["depends_on"]), 4)
        self.assertIn("disagreement-check", by_id["judge"]["depends_on"])
        self.assertEqual(by_id["debate-round-1"]["condition"], "judge_detected_material_disagreement")
        self.assertIn("debate-round-1", by_id["synthesis"]["depends_on"])
        self.assertNotIn("debate-round-1_if_executed", by_id["synthesis"]["depends_on"])
        operation_ids = set(by_id)
        self.assertTrue(all(set(operation["depends_on"]).issubset(operation_ids) for operation in plan["operations"]))
        self.assertTrue(by_id["synthesis"]["public_output"])
        self.assertTrue(all(not operation["public_output"] for operation in plan["operations"][:-1]))

    def test_plan_fails_if_agent_cap_cannot_cover_detected_domains(self):
        with self.assertRaisesRegex(ValueError, "domain coverage"):
            self.build(max_agents=3)

    def test_direct_ultra_requires_pro_authorization_and_budget(self):
        with self.assertRaisesRegex(ValueError, "Pro"):
            self.build(entitlement="plus")
        with self.assertRaisesRegex(ValueError, "not authorized"):
            self.build(ultra_authorized=False)
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            self.build(remaining_usd=0.1)

    def test_caller_cannot_supply_model_or_agent_count(self):
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            self.build({**self.request(), "model": "attacker/model"})
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            self.build({**self.request(), "max_agents": 100})

    def test_source_plan_never_claims_deployment_or_selected_model(self):
        plan = self.build(self.request("Prove this probability equation."))
        self.assertFalse(plan["production_ready"])
        self.assertTrue(plan["model_selection_required"])
        self.assertEqual(plan["provider"], "runpod_serverless")
        self.assertEqual(plan["endpoint_name"], "kova-ultra")
        self.assertFalse(plan["endpoint_deployed"])

    def test_task_and_private_artifact_bindings_are_in_every_required_input(self):
        plan = self.build()
        self.assertEqual(plan["task"], self.request()["task"])
        for operation in plan["operations"]:
            messages = operation["input_template"]["messages"]
            self.assertTrue(any(message["role"] == "user" and message["content"] == plan["task"] for message in messages))
            self.assertEqual(
                [binding["source_stage_id"] for binding in operation["input_template"]["artifact_bindings"]],
                operation["depends_on"],
            )
            for binding in operation["input_template"]["artifact_bindings"]:
                self.assertIn(binding["placeholder"], messages[binding["target_message_index"]]["content"])
                self.assertEqual(messages[binding["target_message_index"]]["role"], "assistant")
                self.assertIn("UNTRUSTED PRIOR MODEL OUTPUT", messages[binding["target_message_index"]]["content"])
                self.assertTrue(binding["replace_exact_target_only"])
            self.assertTrue(operation["input_template"]["reject_placeholder_outside_binding_targets"])

    def test_total_token_cap_includes_inputs_outputs_and_propagated_artifacts(self):
        plan = self.build(max_total_tokens=32768)
        reserved = sum(
            operation["maximum_input_tokens"] + operation["maximum_output_tokens"]
            for operation in plan["operations"]
        )
        self.assertEqual(plan["reserved_maximum_total_tokens"], reserved)
        self.assertLessEqual(reserved, plan["maximum_total_tokens"])
        with self.assertRaisesRegex(ValueError, "after input and artifact reservation"):
            self.build(self.request("x" * 100_000), max_total_tokens=4096)

    def test_large_valid_admission_still_fits_each_native_context_window(self):
        plan = self.build(self.request("Write a short general report."), max_total_tokens=131072)
        for operation in plan["operations"]:
            self.assertLessEqual(operation["maximum_input_tokens"] +
                                 operation["maximum_output_tokens"], NATIVE_CONTEXT_TOKENS)

    def test_plus_work_ultra_is_admitted_and_preserves_distinct_family_behavior(self):
        plans = []
        for family in ("cosmo", "orion", "nova"):
            plans.append(self.build({
                "request_id": family, "task": "Create a project report.",
                "surface": "work", "family": family, "effort": "Ultra",
            }, entitlement="plus"))
        self.assertEqual(len({plan["behavior_contract_id"] for plan in plans}), 3)

    def test_trusted_token_counter_is_required(self):
        with self.assertRaisesRegex(ValueError, "token counter"):
            build_ultra_plan(self.request(), admission=self.admission(), token_counter=None)


if __name__ == "__main__":
    unittest.main()
