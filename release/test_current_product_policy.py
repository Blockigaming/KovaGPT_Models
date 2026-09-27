import unittest
from release.current_product_policy import validate


class CurrentProductPolicyTests(unittest.TestCase):
    def test_exact_three_family_policy(self):
        report = validate()
        self.assertEqual(report["schema_version"], 3)
        self.assertEqual(report["family_slots"], ["kova-cosmo", "kova-orion", "kova-nova"])
        self.assertEqual(report["chat_route_counts_by_tier"], {"free": 1, "plus": 6, "pro": 12})
        self.assertEqual(report["work_route_counts_by_tier"], {"free": 0, "plus": 18, "pro": 18})
        self.assertFalse(report["nova_chat_selectable"])
        self.assertFalse(report["separate_chat_8b_slot"])
        self.assertFalse(report["normal_surface_lineage_exposure"])
        self.assertFalse(report["phase_b_ready"])


if __name__ == "__main__":
    unittest.main()
