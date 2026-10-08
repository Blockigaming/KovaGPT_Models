"""Synthetic auditor regressions, not private outputs or model quality.

The private 36-case report is replayed separately with --preserved-report and
retained outside this public repository. Fixtures below derive only from the
already-public suite and historical fixture, never from that private report.
"""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from evaluation import a35_latest_screen_audit as audit
from evaluation.cpu_candidate_quality import score_case
from evaluation.completion_evidence import generation_evidence
from collections import Counter
from evaluation.historical_suite_bridge import load_archived_suite
from training import a35_nova_screen as screen


class LatestNovaScreenAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reconstruct a clearly synthetic report from the public contract.
        cls.report = json.loads(audit.previous.REPORT.read_bytes())
        cls.report.update(source_commit=audit.SOURCE, run_id=audit.RUN,
            prepared_pack_sha256=audit.PACK, plan_sha256=audit.PLAN,
            suite_sha256=audit.SUITE, base_revision=audit.BASE)
        cls.report["generation_profile"]["system_prompt_sha256"] = audit.PROMPT
        cls.report["training_receipt"].update(source_commit=audit.SOURCE, run_id=audit.RUN,
            prepared_pack_sha256=audit.PACK, base_revision=audit.BASE,
            adapter_sha256={"adapter_model.safetensors": audit.ADAPTER,
                            "adapter_config.json": audit.ADAPTER_CONFIG})
        cls.cases = {case["id"]: case for case in load_archived_suite()["cases"]}
        for index, row in enumerate(cls.report["cases"]):
            case = cls.cases[row["case_id"]]
            expected = case["evaluation"]["expected"]
            answer = json.dumps(expected)
            if row["case_id"] in audit.FAILURES:
                if audit.FAILURES[row["case_id"]][0] == "format_only":
                    answer = (repr(set(expected)) if row["case_id"] == "data-05" else
                              json.dumps({"synthetic_unrequested_wrapper": expected}))
                else:
                    answer = "null"  # deliberately wrong public-contract fixture
            row["answer"] = answer
            row["answer_sha256"] = hashlib.sha256(answer.encode()).hexdigest()
            row["latency_seconds"] = 0.0
            row["completion_evidence"] = generation_evidence([7] * (20 if index == 0 else 1) + [9],
                max_new_tokens=128, eos_token_id=9)
            row["result"] = score_case(case, answer, row["completion_evidence"])[0]
        cls.report["result_counts"] = dict(Counter(r["result"] for r in cls.report["cases"]))
        cls.report["category_counts"] = {cat: dict(Counter(r["result"] for r in cls.report["cases"]
            if r["category"] == cat)) for cat in audit.COUNTS}
        cls.rows = {row["case_id"]: row for row in cls.report["cases"]}
        cls.temporary = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temporary.name) / "synthetic-screen.json"
        cls.path.write_text(json.dumps(cls.report))
        cls.digest_patch = patch.object(audit, "REPORT_SHA256", hashlib.sha256(cls.path.read_bytes()).hexdigest())
        cls.digest_patch.start()
        cls.addClassCleanup(cls.digest_patch.stop)
        cls.addClassCleanup(cls.temporary.cleanup)

    def test_all_15_failures_classified_without_quality_or_causal_claims(self):
        result = audit.audit(self.path)
        self.assertEqual(len(result["failures"]), 15)
        self.assertEqual(result["strict"], "21/36")
        self.assertEqual(result["previous_strict"], "23/36")
        self.assertFalse(result["quality_gate_closed"])
        self.assertEqual(result["model_calls_made"], 0)
        self.assertTrue(all(row["model_internal_root_cause"] == "not isolated by these two screens"
                            for row in result["failures"]))
        self.assertEqual(sum(row["defect_kind"] == "content" for row in result["failures"]), 11)
        self.assertEqual(sum(row["defect_kind"] == "format_only" for row in result["failures"]), 4)

    def test_original_failures_remain_and_two_new_regressions_are_not_averaged_away(self):
        result = audit.audit(self.path)
        self.assertEqual(result["new_regressions"], ["code-06", "data-05"])
        self.assertEqual(result["new_passes"], [])
        self.assertEqual(result["previous_failures_still_failing"], sorted(audit.previous.FAILURES))
        self.assertEqual(result["category_pass_counts"], {"math": "6/10", "code_reading": "2/8",
            "reasoning": "3/6", "data_analysis": "5/6", "instruction_following": "5/6"})

    def test_expected_source_adapter_pack_and_prompt_bindings(self):
        result = audit.audit(self.path)
        self.assertEqual(result["source_head"], audit.SOURCE)
        self.assertEqual(result["candidate_sha256"], audit.ADAPTER)
        self.assertEqual(result["prepared_pack_sha256"], audit.PACK)
        self.assertEqual(result["system_prompt_sha256"], audit.PROMPT)
        self.assertEqual(result["execution_plan_sha256"], audit.PLAN)
        self.assertNotEqual(audit.ADAPTER, audit.previous.ADAPTER)
        self.assertNotEqual(audit.PACK, audit.previous.PACK)
        self.assertFalse(self.report["generation_profile"]["decoder"]["do_sample"])

    def test_synthetic_answer_and_prompt_hash_validation(self):
        for case_id, row in self.rows.items():
            with self.subTest(case=case_id):
                self.assertEqual(hashlib.sha256(row["answer"].encode()).hexdigest(), row["answer_sha256"])
                self.assertEqual(hashlib.sha256(self.cases[case_id]["prompt"].encode()).hexdigest(), row["prompt_sha256"])
                self.assertEqual(score_case(self.cases[case_id], row["answer"], row["completion_evidence"])[0],
                                 row["result"])

    def test_all_outputs_reach_eos_well_before_the_token_budget(self):
        result = audit.audit(self.path)
        self.assertTrue(result["all_outputs_complete"])
        self.assertEqual(result["generated_token_range"], [2, 21])
        for row in self.rows.values():
            self.assertEqual(row["completion_status"], "verified_complete")
            self.assertEqual(row["completion_evidence"]["finish_reason"], "eos")
            self.assertLess(row["completion_evidence"]["generated_token_count"], 128)

    def test_preserved_report_tampering_fails_closed(self):
        altered = deepcopy(self.report)
        altered["cases"][0]["answer"] = "tampered"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.json"
            path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                audit.audit(path)

    def test_synthetic_21_of_36_never_generates_writes_or_uploads_manual(self):
        by_prompt = {self.cases[case_id]["prompt"]: row for case_id, row in self.rows.items()}
        calls, checkpoints = [], []
        def replay(prompt, budget):
            self.assertIn(prompt, by_prompt, "manual generation forbidden")
            self.assertEqual(budget, 128)
            row = by_prompt[prompt]
            calls.append(row["case_id"])
            return row["answer"], row["completion_evidence"], row["latency_seconds"]
        results, status = screen.run_screen(list(self.cases.values()), replay, checkpoints.append)
        self.assertEqual(calls, list(self.rows))
        self.assertEqual(results, self.report["cases"])
        self.assertEqual(checkpoints, results)
        self.assertEqual(status, "strict_threshold_failed_manual_skipped")
        self.assertFalse(screen.strict_passed(list(self.cases.values()), results))
        with tempfile.TemporaryDirectory() as tmp, patch.object(screen, "manual_packet") as packet:
            storage = Mock()
            screen.preserve_screen(Path(tmp), storage, {"status": status, "cases": results})
            packet.assert_not_called()
            self.assertFalse((Path(tmp) / "manual-review.md").exists())
            self.assertEqual([path.name for path in Path(tmp).iterdir()], ["screen.json"])
            self.assertEqual([call.args[0] for call in storage.put.call_args_list], ["screen.json"])
            self.assertEqual(json.loads((Path(tmp) / "screen.json").read_bytes())["cases"], results)

    def test_report_does_not_claim_manual_authenticated_route_or_gate_completion(self):
        self.assertFalse(self.report["human_quality_review_complete"])
        self.assertFalse(self.report["phase_a_item_closed"])
        self.assertFalse(self.report["completion_evidence_authenticated"])
        self.assertEqual(self.report["live_routes_verified"], 0)
        self.assertEqual(audit.audit(self.path)["manual"], "NOT RUN")
        self.assertTrue(all(not row["review_criteria"] for row in self.rows.values()))

    def test_audit_oracles_are_not_imported_by_model_execution(self):
        for path in [Path(screen.__file__), audit.ROOT / "evaluation/a35_correction_data.py"]:
            source = path.read_text()
            self.assertNotIn("a35_latest_screen_audit", source)
            self.assertNotIn("a35-nova-screen-bb328ecf60.json", source)
            self.assertNotIn("independent_expected", source)


def failure_regression(case_id):
    def test(self):
        case, row = self.cases[case_id], self.rows[case_id]
        expected = audit.independent_expected(case_id)
        self.assertEqual(expected, case["evaluation"]["expected"])
        self.assertEqual(score_case(case, row["answer"], row["completion_evidence"])[0], "exact_json_fail")
        self.assertEqual(row["completion_status"], "verified_complete")
        self.assertEqual(row["completion_evidence"]["finish_reason"], "eos")
        kind, _ = audit.FAILURES[case_id]
        if kind == "format_only":
            if case_id == "data-05":
                with self.assertRaises(json.JSONDecodeError):
                    json.loads(row["answer"])
                # Establish content classification only; never repair the graded answer.
                literal = ast.literal_eval(row["answer"])
                self.assertIsInstance(literal, set)
                self.assertEqual(sorted(literal), expected)
            else:
                wrapped = json.loads(row["answer"])
                self.assertIsInstance(wrapped, dict)
                self.assertEqual(len(wrapped), 1)
                value = next(iter(wrapped.values()))
                self.assertIs(type(value), type(expected))
                self.assertEqual(value, expected)
        # Independent oracle validates the fixed benchmark, not candidate quality.
        self.assertEqual(score_case(case, json.dumps(expected), row["completion_evidence"])[0], "exact_json_pass")
    return test


for identifier in audit.FAILURES:
    setattr(LatestNovaScreenAuditTests, "test_synthetic_failure_" + identifier.replace("-", "_"),
            failure_regression(identifier))


if __name__ == "__main__":
    unittest.main()
