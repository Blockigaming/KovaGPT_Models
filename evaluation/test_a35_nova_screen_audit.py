"""Preserved-output regressions and next-input integrity, never model quality."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from evaluation import a35_nova_screen_audit as audit
from evaluation.a35_correction_data import prepared_revision_rows, validate_prompt
from evaluation.cpu_candidate_quality import score_case
from evaluation.historical_suite_bridge import load_archived_suite
from training import a35_nova_screen as screen


class NovaCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = screen.load_plan()
        cls.report = json.loads(audit.REPORT.read_bytes())
        cls.cases = {c["id"]: c for c in load_archived_suite()["cases"]}
        cls.rows = {r["case_id"]: r for r in cls.report["cases"]}

    def test_all_13_failures_classified_without_claiming_causal_or_quality_proof(self):
        result = audit.audit()
        self.assertEqual(len(result["failures"]), 13)
        self.assertEqual(result["strict"], "23/36")
        self.assertFalse(result["quality_gate_closed"])
        self.assertEqual(result["model_calls_made"], 0)
        self.assertTrue(all(r["model_internal_root_cause"] == "not isolated by this single screen"
                            for r in result["failures"]))

    def test_replay_entire_preserved_strict_sweep_never_generates_or_uploads_manual(self):
        by_prompt = {self.cases[cid]["prompt"]: row for cid, row in self.rows.items()}
        calls = []
        def replay(prompt, budget):
            self.assertIn(prompt, by_prompt, "manual generation forbidden")
            self.assertEqual(budget, 128)
            row = by_prompt[prompt]
            calls.append(row["case_id"])
            return row["answer"], row["completion_evidence"], row["latency_seconds"]
        results, status = screen.run_screen(list(self.cases.values()), replay, lambda _: None)
        self.assertEqual(calls, list(self.rows))
        self.assertEqual(results, self.report["cases"])
        self.assertEqual(status, "strict_threshold_failed_manual_skipped")
        with tempfile.TemporaryDirectory() as tmp, patch.object(screen, "manual_packet") as packet:
            storage = Mock()
            screen.preserve_screen(Path(tmp), storage, {"status": status, "cases": results})
            packet.assert_not_called()
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["screen.json"])
            self.assertEqual([c.args[0] for c in storage.put.call_args_list], ["screen.json"])

    def test_historical_110_rows_still_differ_only_in_nova_system_instruction(self):
        before = prepared_revision_rows()
        after = screen.historical_nova_rows(self.plan)
        self.assertEqual(screen.sha(screen.encoded(before)), audit.PACK)
        self.assertEqual(screen.sha(screen.encoded(after)), "d670792ddf8f38896ee9adcac338268a7677cd4b2c60079a33d80a018cdccdf8")
        self.assertNotEqual(self.plan["prepared_pack_sha256"], audit.PACK)
        self.assertEqual(self.plan["previous_screen_adapter_sha256"], audit.ADAPTER)
        self.assertEqual(self.plan["preserved_adapter_sha256"],
                         "f6d09354b5db910288be1e4ac9e22467bfe0c4bf8b554ef576162a245955e352")
        changed = 0
        old_system = (screen.ROOT / "prompts/kova-identity.v4.draft.txt").read_text()
        new_system = old_system + "\n" + (screen.ROOT / "prompts/kova-nova-task-checks.v1.txt").read_text()
        self.assertTrue(new_system.startswith(old_system + "\n"))
        self.assertEqual(validate_prompt(new_system), 34)
        for (old_id, old_split, old), (new_id, new_split, new) in zip(before, after[:110], strict=True):
            self.assertEqual((old_id, old_split), (new_id, new_split))
            restored = deepcopy(new)
            restored["prompt"][0]["content"] = old_system + new["prompt"][0]["content"][len(new_system):]
            self.assertEqual(restored, old)  # all targets, users, splits and trusted fixtures
            changed += new["prompt"][0]["content"] != old["prompt"][0]["content"]
        self.assertEqual(changed, 110)

    def test_data_six_of_six_and_all_original_dataset_targets_remain_unchanged(self):
        data = [r for r in self.rows.values() if r["category"] == "data_analysis"]
        self.assertEqual(len(data), 6)
        for row in data:
            self.assertEqual(score_case(self.cases[row["case_id"]], row["answer"],
                                        row["completion_evidence"])[0], "exact_json_pass")
        # This preserves evidence/inputs; it cannot promise future data quality.
        old = prepared_revision_rows()
        new = screen.historical_nova_rows(self.plan)
        self.assertEqual([r[2]["completion"] for r in old], [r[2]["completion"] for r in new[:110]])

    def test_no_golden_case_or_answer_lookup_enters_the_new_prompt(self):
        text = (screen.ROOT / self.plan["task_checks_path"]).read_text()
        for cid in self.cases:
            self.assertNotIn(cid, text)
        for case in self.cases.values():
            self.assertNotIn(case["prompt"], text)
        self.assertNotIn("a35_nova_screen_audit", Path(screen.__file__).read_text())
        self.assertNotIn("independent_expected", Path(screen.__file__).read_text())

    def test_changed_prompt_and_unknown_instruction_source_fail_closed(self):
        original = self.plan["task_checks_path"]
        plan = deepcopy(self.plan)
        plan["task_checks_path"] = "prompts/unknown.txt"
        with self.assertRaisesRegex(ValueError, "task-check"):
            screen.prepared_inputs(plan)
        read = Path.read_bytes
        with patch.object(Path, "read_bytes", lambda p: b"altered" if p == screen.ROOT / original else read(p)):
            with self.assertRaisesRegex(ValueError, "input pin drift"):
                screen.load_plan()

    def test_every_targeted_behavior_already_has_independent_training_coverage(self):
        review = json.loads((screen.ROOT / "data/a35-correction-supplement-review.v1.json").read_text())
        rows = [json.loads(line) for line in (screen.ROOT / "data/a35-correction-supplement.v1.draft.jsonl").read_text().splitlines()]
        split = {r["id"]: r["split"] for r in rows}
        for _, _, _, group in audit.FAILURES.values():
            self.assertEqual({split[r["id"]] for r in review["records"] if r["group"] == group},
                             {"train", "validation"})

    def test_previous_screen_artifact_cannot_be_relabelled_as_new_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            (adapter / "adapter_model.safetensors").write_bytes(b"synthetic previous candidate")
            digest = screen.sha((adapter / "adapter_model.safetensors").read_bytes())
            plan = self.plan | {"rejected_adapter_sha256": [digest]}
            receipt = deepcopy(self.report["training_receipt"])
            receipt.update(experiment_id=plan["experiment_id"], prepared_pack_sha256=plan["prepared_pack_sha256"],
                           training_records=plan["train_records"], optimizer_steps=plan["training"]["expected_optimizer_steps"],
                           completed_epochs=float(plan["training"]["epochs"]))
            receipt["adapter_sha256"]["adapter_model.safetensors"] = digest
            with self.assertRaisesRegex(ValueError, "unchanged/corrupt candidate"):
                screen.new_candidate(receipt, plan, audit.SOURCE, audit.RUN, adapter)


def failure_regression(case_id):
    def test(self):
        case, row = self.cases[case_id], self.rows[case_id]
        self.assertEqual(audit.independent_expected(case_id), case["evaluation"]["expected"])
        self.assertEqual(score_case(case, row["answer"], row["completion_evidence"])[0], "exact_json_fail")
        self.assertEqual(row["completion_status"], "verified_complete")
        self.assertEqual(row["completion_evidence"]["finish_reason"], "eos")
        self.assertLess(row["completion_evidence"]["generated_token_count"], 128)
        kind, _, instruction, _ = audit.FAILURES[case_id]
        # The historical audit records the instruction proposed at that time.
        self.assertIn(instruction, (screen.ROOT / "prompts/kova-nova-task-checks.v1.txt").read_text())
        current_instruction = ("never replace an object with a list or wrap a scalar"
                               if kind == "format_only" else instruction)
        self.assertIn(current_instruction, screen.system_prompt(self.plan))
        parsed = json.loads(row["answer"])
        if kind == "format_only":
            self.assertIsInstance(parsed, dict)
            self.assertEqual(len(parsed), 1)
            value = next(iter(parsed.values()))
            expected = case["evaluation"]["expected"]
            self.assertIs(type(value), type(expected))
            self.assertEqual(value, expected)
        # The reference is a test oracle, never substituted into candidate output.
        self.assertEqual(score_case(case, json.dumps(audit.independent_expected(case_id)),
                                   row["completion_evidence"])[0], "exact_json_pass")
    return test


for identifier in audit.FAILURES:
    setattr(NovaCorrectionTests, "test_preserved_failure_" + identifier.replace("-", "_"),
            failure_regression(identifier))


if __name__ == "__main__":
    unittest.main()
