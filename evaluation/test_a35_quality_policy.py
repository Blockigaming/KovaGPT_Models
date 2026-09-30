"""Reject quality relaxation while operational authorizations remain separate."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from evaluation import a35_quality_policy as policy
from evaluation.quality_evidence import EvidenceRejected


class A35QualityPolicyTests(unittest.TestCase):
    def test_owner_quality_matches_every_pinned_case_and_manual_criterion(self):
        value = policy.load_policy()
        self.assertEqual(value["strict_per_repetition"] * value["required_repetitions"],
                         value["strict_aggregate"])
        result = policy.summarize_policy()
        self.assertEqual(result["operational_acceptance"], "pending")
        self.assertFalse(result["phase_a_item_closed"])

    def test_relaxing_any_floor_or_enabling_spending_is_rejected(self):
        original = policy.load_policy()
        for name, value in (("strict_per_repetition", 35), ("strict_aggregate", 107),
                            ("manual_per_repetition", 13), ("required_repetitions", 2),
                            ("pooling_or_averaging_failures_allowed", True),
                            ("paid_execution_authorized", True), ("latency_targets", {}),
                            ("cost_ceiling", 0), ("identity_safety_grounding_failures_allowed", 1)):
            changed = deepcopy(original); changed[name] = value
            with patch.object(type(policy.POLICY_PATH), "read_text", return_value=json.dumps(changed)):
                with self.assertRaises(EvidenceRejected):
                    policy.load_policy()


if __name__ == "__main__":
    unittest.main()
