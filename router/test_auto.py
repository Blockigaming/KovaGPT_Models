import json
from pathlib import Path
import unittest

from router.auto import classify_auto


class AutoClassifierTests(unittest.TestCase):
    def budget(self, **overrides):
        value = {"ultra_authorized": True, "remaining_usd": 2.0, "estimated_ultra_usd": 0.5}
        value.update(overrides)
        return value

    def classify(self, prompt, entitlement="pro", **budget_overrides):
        return classify_auto(prompt, entitlement=entitlement, budget=self.budget(**budget_overrides))

    def test_simple_requests_route_to_instant(self):
        self.assertEqual(self.classify("What's 8 × 7?")["route_id"], "instant")
        self.assertEqual(self.classify("Rewrite this sentence to sound nicer.")["route_id"], "instant")

    def test_tool_or_debug_request_routes_to_orion_medium(self):
        self.assertEqual(self.classify("What is the latest weather today?")["route_id"], "medium")
        self.assertEqual(self.classify("Debug this React hydration error.")["route_id"], "medium")

    def test_serious_analysis_routes_to_nova(self):
        route = self.classify("Analyze this architecture and identify race conditions and security risks.")
        self.assertEqual(route["route_id"], "high")
        self.assertEqual(route["engine"], "kova-core")

    def test_classifier_terms_match_boundaries_not_substrings(self):
        self.assertEqual(self.classify("What is a bracelet?")["route_id"], "instant")

    def test_multi_domain_project_can_route_to_ultra(self):
        prompt = "Research 30 competitors, create a comprehensive full report, compare pricing, and produce a launch strategy."
        route = self.classify(prompt)
        self.assertEqual(route["route_id"], "ultra")
        self.assertEqual(route["engine"], "kova-ultra")

    def test_ultra_fallback_respects_entitlement_and_budget(self):
        prompt = "Research competitors and create a comprehensive cross-functional launch strategy and full report."
        self.assertEqual(self.classify(prompt, entitlement="plus")["route_id"], "high")
        self.assertEqual(self.classify(prompt, remaining_usd=0.1)["route_id"], "max")
        self.assertEqual(self.classify(prompt, ultra_authorized=False)["route_id"], "max")

    def test_free_plan_is_server_capped_to_instant(self):
        route = self.classify("Create a comprehensive multi-domain competitor launch strategy.", entitlement="free")
        self.assertEqual(route["route_id"], "instant")
        self.assertEqual(route["feature_ids"], ["free_plan_instant_cap"])

    def test_invalid_budget_context_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "budget context"):
            classify_auto("hello", entitlement="pro", budget={})

    def test_plus_caps_deep_max_and_ultra_branches_at_high(self):
        cases = (
            ("Analyze architecture " + "context " * 65, "extra-high"),
            ("Analyze architecture " + "context " * 145, "max"),
            ("Research competitors and create a comprehensive full report.", "ultra"),
        )
        for prompt, pro_route in cases:
            with self.subTest(pro_route=pro_route):
                self.assertEqual(self.classify(prompt)["route_id"], pro_route)
                route = self.classify(prompt, entitlement="plus")
                self.assertEqual(route["route_id"], "high")
                self.assertEqual(route["engine"], "kova-core")
                self.assertIn("plus_plan_high_cap", route["feature_ids"])

    def test_plus_keeps_allowed_routes_unchanged(self):
        for prompt, expected in (
            ("What is 8 + 7?", "instant"),
            ("Debug this React error", "medium"),
            ("Analyze security", "high"),
        ):
            with self.subTest(expected=expected):
                route = self.classify(prompt, entitlement="plus")
                self.assertEqual(route["route_id"], expected)
                self.assertNotIn("plus_plan_high_cap", route["feature_ids"])

    def test_every_branch_obeys_plan_caps_across_budget_states(self):
        prompts = (
            "What is 8 + 7?", "Debug this React error", "Analyze security",
            "Analyze architecture " + "context " * 65,
            "Analyze architecture " + "context " * 145,
            "Research competitors and create a comprehensive full report.",
        )
        allowed = {
            "free": {"instant"},
            "plus": {"instant", "medium", "high"},
            "pro": {"instant", "medium", "high", "extra-high", "max", "ultra"},
        }
        for tier, routes in allowed.items():
            for authorized in (False, True):
                for remaining in (0, 0.49, 0.5, 1):
                    for prompt in prompts:
                        with self.subTest(tier=tier, authorized=authorized, remaining=remaining, prompt=prompt[:25]):
                            route = self.classify(prompt, entitlement=tier,
                                                  ultra_authorized=authorized, remaining_usd=remaining)
                            self.assertIn(route["route_id"], routes)
                            if route["route_id"] == "ultra":
                                self.assertEqual(tier, "pro")
                                self.assertTrue(authorized)
                                self.assertGreaterEqual(remaining, 0.5)

    def test_nonfinite_or_invalid_budget_numbers_fail_closed(self):
        for field in ("remaining_usd", "estimated_ultra_usd"):
            for value in (float("nan"), float("inf"), -float("inf"), True, "1", None, -1):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.classify("Research competitors and create a comprehensive full report.", **{field: value})

    def test_source_contract_records_the_same_plus_cap(self):
        policy = json.loads((Path(__file__).resolve().parents[1] / "config/route-policy.v1.json").read_text(encoding="utf-8"))
        self.assertEqual(policy["auto"]["free_plan_route_cap"], "instant")
        self.assertEqual(policy["auto"]["plus_plan_route_cap"], "high")
        self.assertEqual(policy["auto"]["ultra_entitlement"], "pro")
        self.assertFalse(policy["auto"]["production_ready"])

    def test_invalid_server_entitlement_fails_closed(self):
        for entitlement in ("unknown", "PLUS", "", None, True, [], {}):
            with self.subTest(entitlement=entitlement), self.assertRaises(ValueError):
                self.classify("hello", entitlement=entitlement)


if __name__ == "__main__":
    unittest.main()
