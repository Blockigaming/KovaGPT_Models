import unittest
from release.product_decisions import validate


class ProductDecisionTests(unittest.TestCase):
    def test_all_architecture_choices_are_resolved_without_execution(self):
        report = validate()
        self.assertEqual(len(report["resolved_product_decision_ids"]), 5)
        self.assertEqual(report["chat_route_counts_by_tier"], {"free":1,"plus":6,"pro":12})
        self.assertEqual(report["work_route_counts_by_tier"], {"free":0,"plus":18,"pro":18})
        self.assertFalse(report["spending_authorized"])
        self.assertFalse(report["deployment_authorized"])


if __name__ == "__main__":
    unittest.main()
