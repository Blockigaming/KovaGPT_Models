"""Completion must be established by tokens, independently of answer content."""

from copy import deepcopy
import unittest

from evaluation.completion_evidence import generation_evidence, completion_status, output_budgets
from evaluation.cpu_candidate_quality import score_case


class CompletionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.case = {"evaluation": {"kind": "exact_json", "expected": {"counter": 17}}}
        self.manual = {"evaluation": {"kind": "review_required"}}
        self.complete = generation_evidence([4, 7, 2], max_new_tokens=8, eos_token_id=2)

    def test_only_real_terminal_eos_earns_strict_credit(self):
        self.assertEqual(score_case(self.case, '{"counter":17}', self.complete)[0],
                         "exact_json_pass")
        for evidence in (None, {}, {"complete": True},
                         generation_evidence([4, 7], max_new_tokens=8, eos_token_id=2)):
            self.assertEqual(score_case(self.case, '{"counter":17}', evidence)[0],
                             "completion_unverified")
        self.assertEqual(score_case(self.case, '{"counter":18}', None)[0], "exact_json_fail")

    def test_length_limit_cannot_be_success_even_if_answer_looks_finished(self):
        cap = generation_evidence([4, 7], max_new_tokens=2, eos_token_id=2)
        self.assertEqual(completion_status(cap), "length_limit")
        for case, answer in ((self.case, '{"counter":17}'), (self.manual, "Finished.")):
            self.assertEqual(score_case(case, answer, cap)[0], "incomplete_output")
        final_eos = generation_evidence([4, 2], max_new_tokens=2, eos_token_id=[2, 3])
        self.assertEqual(completion_status(final_eos), "verified_complete")
        self.assertEqual(final_eos["generated_token_count"], 2)

    def test_completion_gate_never_hides_privacy_or_contract_failure(self):
        for evidence in (None, self.complete,
                         generation_evidence([4], max_new_tokens=1, eos_token_id=2)):
            self.assertEqual(score_case(self.manual, "<THINK>private</THINK>", evidence),
                             ("private_output_blocked", None))
            self.assertEqual(score_case(self.manual, "reasoning_content", evidence)[0],
                             "output_contract_failed")

    def test_malformed_metadata_is_unverified_and_token_lists_are_rejected(self):
        for key, value in (("generated_token_count", True), ("max_new_tokens", 0),
                           ("eos_token_ids", [[2]]), ("terminal_token_id", 9),
                           ("generated_token_ids_sha256", "x" * 64),
                           ("finish_reason", "stop")):
            evidence = deepcopy(self.complete); evidence[key] = value
            self.assertEqual(completion_status(evidence), "unverified")
        for tokens in ([], [True], [4, 2, 7], [4] * 9):
            with self.assertRaises(ValueError):
                generation_evidence(tokens, max_new_tokens=8, eos_token_id=2)

    def test_manual_budget_is_explicit_bounded_and_cannot_change_strict_budget(self):
        self.assertEqual(output_budgets([self.case, self.manual], 512), (128, 512))
        for budget in (None, True, 0, 4097, "512"):
            with self.assertRaises(ValueError):
                output_budgets([self.case, self.manual], budget)


if __name__ == "__main__":
    unittest.main()
