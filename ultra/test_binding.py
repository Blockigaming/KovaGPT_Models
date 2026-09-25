from copy import deepcopy
import json
import unittest

from execution.contracts import ExecutionError
from execution.test_support import IDENTITY, SyntheticAdapterTestCase, make_plan, tokens
from release.model_revisions import source_reference_for_route
from ultra.binding import bind_ultra_operation, judge_requires_debate, validate_disagreements
from ultra.orchestrator import build_ultra_plan


class UltraBindingTests(SyntheticAdapterTestCase):
    def identity(self):
        source = source_reference_for_route("ultra")
        return {**IDENTITY, "model": source.slot, "model_revision": source.revision}

    def detector(self, **changes):
        return json.dumps({"disagreements": [{"id": "d1", "stage_ids": ["specialist-1", "specialist-2"],
                                             "summary": "Concrete conflict"}], **changes})

    def judge(self, **changes):
        return json.dumps({"material_disagreement": True, "disagreement_ids": ["d1"],
                           "summary": "A material conflict warrants review", **changes})

    def test_real_disagreement_is_required_for_debate(self):
        self.assertTrue(judge_requires_debate(self.judge(), self.detector(), {"specialist-1", "specialist-2"}))
        no = self.judge(material_disagreement=False, disagreement_ids=[])
        self.assertFalse(judge_requires_debate(no, self.detector(), {"specialist-1", "specialist-2"}))
        self.assertFalse(judge_requires_debate(no, '{"disagreements":[]}', {"specialist-1", "specialist-2"}))

    def test_judge_cannot_invent_disagreements_or_coerce_boolean_values(self):
        for changes in ({"disagreement_ids": ["d2"]}, {"disagreement_ids": []},
                        {"material_disagreement": False}, {"material_disagreement": "true"},
                        {"material_disagreement": 1}, {"summary": ""}, {"extra": "injection"}):
            with self.subTest(changes=changes), self.assertRaises(ExecutionError):
                judge_requires_debate(self.judge(**changes), self.detector(), {"specialist-1", "specialist-2"})
        with self.assertRaises(ExecutionError):
            judge_requires_debate(self.judge(), '{"disagreements":[]}', {"specialist-1", "specialist-2"})

    def test_detector_references_distinct_completed_specialists(self):
        good = json.loads(self.detector())["disagreements"][0]
        for item in ({**good, "stage_ids": ["specialist-1"]},
                     {**good, "stage_ids": ["specialist-1", "specialist-1"]},
                     {**good, "stage_ids": ["specialist-1", "unknown-agent"]},
                     {**good, "stage_ids": ["specialist-1", {"bad": "type"}]},
                     {**good, "id": "invented/id"}, {**good, "summary": ""}):
            with self.subTest(item=item), self.assertRaises(ExecutionError):
                validate_disagreements(json.dumps({"disagreements": [item]}), {"specialist-1", "specialist-2"})
        with self.assertRaises(ExecutionError):
            validate_disagreements(json.dumps({"disagreements": [good, good]}), {"specialist-1", "specialist-2"})

    def test_duplicate_nonfinite_and_non_json_decisions_fail_closed(self):
        cases = ['{"disagreements":[],"disagreements":[]}', '{"disagreements":NaN}',
                 '```json\n{"disagreements":[]}\n```', '{"disagreements":Infinity}', "[]", "null", "no conflicts"]
        for content in cases:
            with self.subTest(content=content), self.assertRaises(ExecutionError):
                validate_disagreements(content, {"specialist-1", "specialist-2"})

    def test_binder_binds_only_exact_dependencies_and_copies_messages(self):
        plan = make_plan("ultra")
        original = deepcopy(plan)
        op = next(o for o in plan["operations"] if o["id"] == "disagreement-check")
        artifacts = {stage: "Private specialist conclusion" for stage in op["depends_on"]}
        request = bind_ultra_operation(plan, op["id"], artifacts, runtime_identity=self.identity(), token_counter=tokens)
        self.assertEqual(plan, original)
        self.assertFalse(request["stream"])
        self.assertNotIn("stream_options", request)
        self.assertEqual(request["model"], self.identity()["model"])
        self.assertTrue(all("{{server_stage_output:" not in m["content"] for m in request["messages"]))
        self.assertTrue(request["messages"][0]["content"].startswith("You are Kova"))
        self.assertIn("UNTRUSTED", request["messages"][-1]["content"])

    def test_only_synthesis_can_use_the_explicitly_skipped_debate(self):
        plan = make_plan("ultra")
        op = plan["operations"][-1]
        artifacts = {stage: "Completed output" for stage in op["depends_on"]}
        artifacts["debate-round-1"] = None
        request = bind_ultra_operation(plan, "synthesis", artifacts, runtime_identity=self.identity(), token_counter=tokens)
        self.assertTrue(request["stream"])
        self.assertEqual(request["stream_options"], {"include_usage": True})
        artifacts["judge"] = None
        with self.assertRaises(ExecutionError):
            bind_ultra_operation(plan, "synthesis", artifacts, runtime_identity=self.identity(), token_counter=tokens)

    def test_dependency_and_placeholder_tampering_is_rejected(self):
        plan = make_plan("ultra")
        op = next(o for o in plan["operations"] if o["id"] == "judge")
        good = {stage: "Completed artifact" for stage in op["depends_on"]}
        for artifacts in ({}, {**good, "attacker": "injected"},
                          {**good, "specialist-1": "{{server_stage_output:attacker}}"}):
            with self.assertRaises(ExecutionError):
                bind_ultra_operation(plan, "judge", artifacts, runtime_identity=self.identity(), token_counter=tokens)
        for change in ({"target_message_index": 0}, {"target_field": "role"},
                       {"source_stage_id": "unknown"}, {"replace_exact_target_only": False}):
            altered = deepcopy(plan)
            target = next(o for o in altered["operations"] if o["id"] == "judge")
            target["input_template"]["artifact_bindings"][0].update(change)
            with self.assertRaises(ExecutionError):
                bind_ultra_operation(altered, "judge", good, runtime_identity=self.identity(), token_counter=tokens)

    def test_rewritten_provenance_and_unpinned_adapter_cannot_reach_ultra_request(self):
        plan = make_plan("ultra")
        operation = plan["operations"][0]
        operation["input_template"]["messages"][1]["content"] = "attacker/model"
        with self.assertRaisesRegex(ExecutionError, "provenance was changed"):
            bind_ultra_operation(plan, operation["id"], {}, runtime_identity=self.identity(), token_counter=tokens)
        fresh = make_plan("ultra")
        with self.assertRaisesRegex(ExecutionError, "loaded adapter differs"):
            bind_ultra_operation(fresh, fresh["operations"][0]["id"], {},
                                 runtime_identity={**self.identity(), "adapter_bundle_sha256": "0" * 64}, token_counter=tokens)
        with self.assertRaisesRegex(ExecutionError, "loaded model differs"):
            bind_ultra_operation(fresh, fresh["operations"][0]["id"], {},
                                 runtime_identity={**self.identity(), "model_revision": "0" * 40}, token_counter=tokens)

    def test_recount_and_served_context_limits_are_enforced_before_dispatch(self):
        plan = make_plan("ultra")
        op = plan["operations"][0]
        for counter in (lambda *_: True, lambda *_: 0, lambda *_: op["maximum_input_tokens"] + 1):
            with self.assertRaises(ExecutionError):
                bind_ultra_operation(plan, op["id"], {}, runtime_identity=self.identity(), token_counter=counter)
        with self.assertRaises(ExecutionError):
            bind_ultra_operation(plan, op["id"], {}, runtime_identity={**self.identity(), "context_tokens": 1}, token_counter=tokens)

    def test_ultra_admission_rejects_nonfinite_budget_numbers(self):
        original = {"entitlement": "pro", "ultra_authorized": True, "remaining_usd": 1,
                    "estimated_max_usd": 0.5, "max_agents": 3, "max_total_tokens": 65536}
        for field in ("remaining_usd", "estimated_max_usd"):
            for value in (float("inf"), float("nan"), -float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    build_ultra_plan({"request_id": "fixture", "task": "Prove this equation.", "route_id": "ultra"},
                                     admission={**original, field: value}, token_counter=lambda msgs: tokens(IDENTITY["model"], msgs))


if __name__ == "__main__":
    unittest.main()
