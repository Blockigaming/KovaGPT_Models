"""The published diagnostic must not score coerced JSON or disclose reasoning."""

import json
import unittest

from evaluation.cpu_candidate_quality import score_case
from evaluation.historical_suite_bridge import load_archived_suite
from evaluation.completion_evidence import generation_evidence


class CandidateQualityScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_archived_suite()["cases"]

    def test_exact_scoring_requires_valid_typed_json(self):
        case = next(c for c in self.cases if c["evaluation"]["kind"] == "exact_json")
        expected = json.dumps(case["evaluation"]["expected"])
        complete = generation_evidence([7, 2], max_new_tokens=8, eos_token_id=2)
        self.assertEqual(score_case(case, expected, complete)[0], "exact_json_pass")
        self.assertEqual(score_case(case, expected)[0], "completion_unverified")
        self.assertEqual(score_case(case, '{"answer": true}')[0], "exact_json_fail")
        self.assertEqual(score_case(case, '{"answer": 200, "answer": 200}')[0],
                         "exact_json_fail")
        self.assertEqual(score_case(case, '```json\n' + expected + '\n```')[0],
                         "exact_json_fail")

    def test_hidden_reasoning_never_enters_review_artifact(self):
        case = next(c for c in self.cases if c["evaluation"]["kind"] == "review_required")
        complete = generation_evidence([7, 2], max_new_tokens=8, eos_token_id=2)
        self.assertEqual(score_case(case, "Needs a human review.", complete),
                         ("pending_human_review", "Needs a human review."))
        self.assertEqual(score_case(case, "<THINK>private</THINK>public"),
                         ("private_output_blocked", None))


if __name__ == "__main__":
    unittest.main()
