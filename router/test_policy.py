import unittest
from unittest.mock import patch

from router.policy import resolve_route, RUNTIME_PROFILES


class RoutePolicyTests(unittest.TestCase):
    def test_declared_profile_caps_are_checked_at_routing(self):
        for effort in ("Light", "Medium", "High", "Extra High", "Max"):
            route = resolve_route({"surface": "work", "family": "nova", "effort": effort})
            profile = RUNTIME_PROFILES[effort.lower().replace(" ", "-")]
            self.assertLessEqual(route["maximum_output_tokens"], profile["maximum_output_tokens"])
            self.assertLessEqual(sum(route["passes"]), profile["maximum_passes"])
        with patch.dict(RUNTIME_PROFILES["max"], {"maximum_passes": 1}):
            with self.assertRaisesRegex(ValueError, "profile pass bound"):
                resolve_route({"surface": "work", "family": "nova", "effort": "Max"})
        with patch.dict(RUNTIME_PROFILES["light"], {"maximum_output_tokens": 1}):
            with self.assertRaisesRegex(ValueError, "profile output bound"):
                resolve_route({"surface": "work", "family": "nova", "effort": "Light"})
    def test_instant_is_one_pass_core_without_activity(self):
        route = resolve_route({"surface": "chat", "route_id": "instant"})
        self.assertEqual(route["engine"], "kova-core")
        self.assertEqual(route["display_name"], "Kova Cosmo")
        self.assertEqual(route["passes"], (0, 1, 0, 0))
        self.assertFalse(route["activity_updates"])

    def test_max_is_still_single_core_engine(self):
        route = resolve_route({"surface": "chat", "route_id": "max"})
        self.assertEqual(route["engine"], "kova-core")
        self.assertEqual(route["profile"], "orion")

    def test_canonical_chat_has_twelve_cosmo_orion_combinations(self):
        routes = {
            resolve_route({"surface": "chat", "family": family, "effort": effort})["route_id"]
            for family in ("cosmo", "orion")
            for effort in ("Light", "Medium", "High", "Extra High", "Max", "Ultra")
        }
        self.assertEqual(len(routes), 12)
        with self.assertRaisesRegex(ValueError, "invalid chat family"):
            resolve_route({"surface": "chat", "family": "nova", "effort": "High"})

    def test_ultra_changes_engine_and_requires_judged_synthesis(self):
        route = resolve_route({"surface": "chat", "route_id": "ultra"})
        self.assertEqual(route["engine"], "kova-ultra")
        self.assertTrue(route["judge"])
        self.assertTrue(route["synthesis"])

    def test_work_has_eighteen_valid_combinations(self):
        routes = {
            resolve_route({"surface": "work", "family": family, "effort": effort})["route_id"]
            for family in ("cosmo", "orion", "nova")
            for effort in ("Light", "Medium", "High", "Extra High", "Max", "Ultra")
        }
        self.assertEqual(len(routes), 18)

    def test_work_ultra_uses_ultra_engine(self):
        route = resolve_route({"surface": "work", "family": "orion", "effort": "Ultra"})
        self.assertEqual(route["engine"], "kova-ultra")

    def test_work_families_have_distinct_server_behavior_contracts(self):
        routes = [
            resolve_route({"surface": "work", "family": family, "effort": "Medium"})
            for family in ("cosmo", "orion", "nova")
        ]
        self.assertEqual(len({route["behavior_contract_id"] for route in routes}), 3)
        self.assertEqual(len({route["behavior_instruction"] for route in routes}), 3)
        self.assertEqual(len({route["answer_style"] for route in routes}), 3)

    def test_caller_cannot_override_server_policy(self):
        with self.assertRaisesRegex(ValueError, "server-controlled"):
            resolve_route({"surface": "chat", "route_id": "instant", "model": "attacker/model"})
        with self.assertRaisesRegex(ValueError, "server-controlled"):
            resolve_route({"surface": "chat", "route_id": "instant", "engine": "kova-ultra"})
        with self.assertRaisesRegex(ValueError, "server-controlled"):
            resolve_route({"surface": "work", "family": "cosmo", "effort": "Light", "behavior_contract_id": "fake"})
        with self.assertRaisesRegex(ValueError, "unsupported chat route fields"):
            resolve_route({"surface": "chat", "route_id": "instant", "ignored": "payload"})
        with self.assertRaisesRegex(ValueError, "unsupported work route fields"):
            resolve_route({"surface": "work", "family": "cosmo", "effort": "Light", "ignored": "payload"})

    def test_auto_cannot_bypass_server_classifier_context(self):
        with self.assertRaisesRegex(ValueError, "server classifier context"):
            resolve_route({"surface": "chat", "route_id": "kova-auto"})


if __name__ == "__main__":
    unittest.main()
