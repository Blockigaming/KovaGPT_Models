"""New supplemental fixtures; NOT the recovered historical 50-case benchmark."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation.quality_evidence import (ROOT, TIMINGS, EvidenceRejected, analyze, answer_hash,
                                        build_route_manifest, digest, strict_json, validate_suite)

COMMIT = "a" * 40


def suite():
    return {"schema_version": 1, "suite_id": "supplemental-ingestion-fixtures-v1",
            "provenance": "source_regression_fixture", "cases": [
        {"id": "arithmetic", "prompt": 'Return 23 + 19 as JSON {"answer": number}.',
         "scorer": "exact_json", "expected_json": {"answer": 42}, "rubric": [], "source_urls": []},
        {"id": "explanation", "prompt": "Explain one tradeoff and identify uncertainty.",
         "scorer": "human_rubric", "expected_json": None, "rubric": ["accuracy", "uncertainty"], "source_urls": []},
    ]}


def bundle():
    return {"schema_version": 1, "kind": "synthetic", "suite_sha256": digest(suite()),
            "source_commit": COMMIT, "attempts": [], "reviews": []}


def attempt(case="arithmetic", route="instant", condition="warm", **changes):
    entry = next(c for c in suite()["cases"] if c["id"] == case)
    manifest = {r["route_id"]: r for r in build_route_manifest()}
    actual = "instant" if route == "kova-auto" else route
    answer = '{"answer":42}' if case == "arithmetic" else "A concise explanation acknowledges uncertainty."
    row = {"id": f"{case}-{route}-{condition}", "case_id": case, "case_sha256": digest(entry),
        "route_id": route, "actual_route_id": actual, "condition": condition, "ordinal": 1, "outcome": "success",
        "answer": answer, "answer_sha256": answer_hash(answer),
        "runtime": {"provider": "synthetic-fixture", "model": "fixture-model", "model_revision": "b" * 40,
                    "image_sha256": "c" * 64, "manifest_sha256": "d" * 64, "engine": manifest[actual]["engine"],
                    "serving_version": "fixture-only", "lifecycle_id": f"synthetic-{case}-{route}-{condition}", "context_tokens": 4096,
                    "concurrency": 1, "gpu_type": "no-gpu-used", "gpu_count": 1, "quantization": "fixture-only"},
        "timings": dict(zip(TIMINGS, (10, 20, 100))),
        "accounting": {"attributable_cost_microusd": 100, "receipt_sha256": "e" * 64}}
    row.update(changes)
    if "accounting" not in changes:
        row["accounting"]["receipt_sha256"] = digest({"synthetic_attempt": row["id"], "cost": 100})
    return row


def review(row, **changes):
    return {"id": "review-" + row["id"], "attempt_id": row["id"], "suite_sha256": digest(suite()),
            "answer_sha256": row["answer_sha256"], "reviewer": "synthetic-reviewer-not-a-person",
            "kind": "synthetic", "verdicts": {"accuracy": "pass", "uncertainty": "pass"}, **changes}


class QualityEvidenceTests(unittest.TestCase):
    def run_bundle(self, value, source=None):
        return analyze(source or suite(), value, expected_suite_sha256=digest(source or suite()), expected_source_commit=COMMIT)

    def test_no_records_keeps_all_37_routes_both_conditions_and_every_case_missing(self):
        result = self.run_bundle(bundle())
        self.assertEqual(result["expected_units"], 148)
        self.assertEqual(result["unit_counts"], {"missing": 148})
        self.assertIsNone(result["reported_total_cost_microusd"])
        self.assertEqual(result["timings_by_configuration"], {})

    def test_all_route_synthetic_evidence_never_claims_live_quality_or_historical_recovery(self):
        value = bundle()
        for route in build_route_manifest():
            for condition in ("cold", "warm"):
                for case in ("arithmetic", "explanation"):
                    row = attempt(case, route["route_id"], condition)
                    value["attempts"].append(row)
                    if case == "explanation":
                        value["reviews"].append(review(row))
        result = self.run_bundle(value)
        self.assertEqual(result["unit_counts"], {"pass": 148})
        self.assertEqual(result["provided_reviews"], 74)
        for field in ("phase_b_ready", "provenance_authenticated", "human_reviewer_identity_verified", "historical_suite_reconciled"):
            self.assertFalse(result[field])
        self.assertEqual(result["live_routes_verified"], [])
        self.assertEqual(result["model_calls_made"], 0)

    def test_wrong_exact_json_duplicate_keys_nonfinite_and_python_code_fail_without_execution(self):
        for output in ('{"answer":43}', '{"answer":41,"answer":42}', '{"answer":NaN}',
                       '{"answer":1e999}', '__import__("os").system("echo MUST_NOT_RUN")', '```json\n{"answer":42}\n```'):
            value = bundle()
            value["attempts"] = [attempt(answer=output, answer_sha256=answer_hash(output))]
            with patch("subprocess.Popen", side_effect=AssertionError("generated code executed")):
                result = self.run_bundle(value)
            self.assertEqual(result["unit_counts"].get("fail"), 1)

    def test_boolean_is_not_accepted_as_numeric_golden_answer(self):
        source = suite()
        source["cases"][0]["expected_json"] = {"answer": 1}
        value = bundle()
        value["suite_sha256"] = digest(source)
        value["attempts"] = [attempt(answer='{"answer":true}', answer_sha256=answer_hash('{"answer":true}'),
                                    case_sha256=digest(source["cases"][0]))]
        self.assertEqual(self.run_bundle(value, source)["unit_counts"]["fail"], 1)

    def test_failed_attempt_costs_remain_in_total_when_a_later_attempt_succeeds(self):
        value = bundle()
        first = attempt(outcome="failed", answer=None, answer_sha256=None)
        first["timings"] = dict.fromkeys(TIMINGS)
        second = attempt(id="second", ordinal=2)
        value["attempts"] = [first, second]
        result = self.run_bundle(value)
        self.assertEqual(result["attempt_outcomes"], {"failed": 1, "success": 1})
        self.assertEqual(result["reported_total_cost_microusd"], 200)
        self.assertEqual(result["unit_counts"]["pass"], 1)

    def test_reusing_one_cost_allocation_receipt_for_two_attempts_is_rejected(self):
        first, second = attempt(), attempt(condition="cold")
        second["accounting"]["receipt_sha256"] = first["accounting"]["receipt_sha256"]
        value = bundle()
        value["attempts"] = [first, second]
        with self.assertRaises(EvidenceRejected):
            self.run_bundle(value)

    def test_one_worker_lifecycle_cannot_claim_conflicting_runtime_identities(self):
        first, second = attempt(), attempt(condition="cold")
        second["runtime"]["lifecycle_id"] = first["runtime"]["lifecycle_id"]
        second["runtime"]["gpu_count"] = 2
        value = bundle()
        value["attempts"] = [first, second]
        with self.assertRaises(EvidenceRejected):
            self.run_bundle(value)

    def test_missing_cost_is_unknown_not_zero_or_a_complete_total(self):
        value = bundle()
        value["attempts"] = [attempt(accounting={"attributable_cost_microusd": None, "receipt_sha256": None})]
        result = self.run_bundle(value)
        self.assertIsNone(result["reported_total_cost_microusd"])
        self.assertEqual(result["missing_cost_attempts"], 1)

    def test_uncertain_cancelled_and_successful_attempts_cannot_be_retried_or_cherry_picked(self):
        for status in ("success", "uncertain", "cancelled"):
            value = bundle()
            value["attempts"] = [attempt(outcome=status), attempt(id="later", ordinal=2)]
            with self.subTest(status=status), self.assertRaises(EvidenceRejected):
                self.run_bundle(value)

    def test_missing_human_reviews_failures_and_pending_dimensions_remain_explicit(self):
        value = bundle()
        row = attempt("explanation")
        value["attempts"] = [row]
        self.assertEqual(self.run_bundle(value)["unit_counts"]["pending_human_review"], 1)
        for verdict, result in (("pending", "pending_human_review"), ("fail", "fail"), ("pass", "pass")):
            value["reviews"] = [review(row, verdicts={"accuracy": "pass", "uncertainty": verdict})]
            self.assertEqual(self.run_bundle(value)["unit_counts"][result], 1)

    def test_reviews_are_bound_to_exact_case_output_suite_and_provenance(self):
        for changes in ({"suite_sha256": "f" * 64}, {"answer_sha256": "f" * 64}, {"kind": "recorded_unverified"},
                        {"attempt_id": "missing"}, {"verdicts": {"accuracy": "pass"}}, {"reviewer": ""}):
            value = bundle()
            row = attempt("explanation")
            value["attempts"] = [row]
            value["reviews"] = [review(row, **changes)]
            with self.subTest(changes=changes), self.assertRaises(EvidenceRejected):
                self.run_bundle(value)

    def test_duplicate_attempt_review_ids_and_ordinal_gaps_are_rejected(self):
        value = bundle()
        value["attempts"] = [attempt(), attempt()]
        with self.assertRaises(EvidenceRejected):
            self.run_bundle(value)
        value["attempts"] = [attempt(ordinal=2)]
        with self.assertRaises(EvidenceRejected):
            self.run_bundle(value)
        row = attempt("explanation")
        value["attempts"] = [row]
        value["reviews"] = [review(row), review(row)]
        with self.assertRaises(EvidenceRejected):
            self.run_bundle(value)

    def test_source_case_answer_and_suite_mutations_are_rejected(self):
        for changes in ({"case_sha256": "f" * 64}, {"answer": "changed"}, {"actual_route_id": "ultra"},
                        {"reasoning_content": "private-channel"}, {"case_id": []}):
            value = bundle()
            value["attempts"] = [attempt(**changes)]
            with self.subTest(changes=changes), self.assertRaises(EvidenceRejected):
                self.run_bundle(value)
        for changes in ({"source_commit": "f" * 40}, {"suite_sha256": "f" * 64}, {"kind": "verified_live"}):
            with self.assertRaises(EvidenceRejected):
                self.run_bundle({**bundle(), **changes})

    def test_auto_records_actual_selected_route_without_becoming_a_seventh_engine(self):
        value = bundle()
        value["attempts"] = [attempt(route="kova-auto")]
        unit = next(u for u in self.run_bundle(value)["units"] if u["result"] == "pass")
        self.assertEqual(unit["actual_route_id"], "instant")
        value["attempts"][0]["actual_route_id"] = "kova-auto"
        with self.assertRaises(EvidenceRejected):
            self.run_bundle(value)

    def test_hardware_context_quantization_and_provider_are_not_mixed_across_retries(self):
        for field, changed in (("gpu_type", "other"), ("context_tokens", 8192), ("quantization", "other"),
                               ("provider", "other"), ("gpu_count", 2), ("concurrency", 2)):
            first = attempt(outcome="failed")
            second = attempt(id="second", ordinal=2)
            second["runtime"][field] = changed
            value = bundle()
            value["attempts"] = [first, second]
            with self.subTest(field=field), self.assertRaises(EvidenceRejected):
                self.run_bundle(value)

    def test_cold_warm_and_runtime_configurations_have_separate_latency_groups(self):
        value = bundle()
        warm = attempt()
        cold = attempt(condition="cold")
        cold["runtime"]["gpu_type"] = "other-fixture"
        value["attempts"] = [warm, cold]
        result = self.run_bundle(value)
        self.assertEqual(len(result["timings_by_configuration"]), 2)
        self.assertTrue(any("|cold|" in k for k in result["timings_by_configuration"]))
        self.assertTrue(any("|warm|" in k for k in result["timings_by_configuration"]))

    def test_missing_first_answer_token_is_not_replaced_by_acknowledgement(self):
        value = bundle()
        row = attempt()
        row["timings"]["first_answer_token_ms"] = None
        value["attempts"] = [row]
        timing = next(iter(self.run_bundle(value)["timings_by_configuration"].values()))
        self.assertEqual(timing["acknowledgement_ms"]["mean"], 10)
        self.assertIsNone(timing["first_answer_token_ms"]["mean"])
        self.assertEqual(timing["first_answer_token_ms"]["missing"], 1)

    def test_invalid_numeric_timings_costs_and_receipts_are_not_coerced(self):
        for bad in (True, -1, float("inf"), "100"):
            value = bundle()
            row = attempt()
            row["timings"]["first_answer_token_ms"] = bad
            value["attempts"] = [row]
            with self.assertRaises(EvidenceRejected):
                self.run_bundle(value)
            row = attempt(accounting={"attributable_cost_microusd": bad, "receipt_sha256": "e" * 64})
            value["attempts"] = [row]
            with self.assertRaises(EvidenceRejected):
                self.run_bundle(value)

    def test_response_contract_violations_fail_without_echoing_private_output(self):
        output = "<think>synthetic-private-marker</think>"
        value = bundle()
        value["attempts"] = [attempt("explanation", answer=output, answer_sha256=answer_hash(output))]
        report = self.run_bundle(value)
        self.assertEqual(report["unit_counts"]["output_contract_failed"], 1)
        self.assertNotIn("synthetic-private-marker", json.dumps(report))
        self.assertNotIn(output, json.dumps(report))

    def test_recorded_label_does_not_authenticate_reviews_or_live_routes(self):
        result = self.run_bundle({**bundle(), "kind": "recorded_unverified"})
        self.assertFalse(result["provenance_authenticated"])
        self.assertFalse(result["phase_b_ready"])
        self.assertEqual(result["live_routes_verified"], [])

    def test_no_network_model_or_generated_program_execution_is_needed(self):
        value = bundle()
        value["attempts"] = [attempt()]
        with patch("socket.socket", side_effect=AssertionError("network")), \
             patch("subprocess.Popen", side_effect=AssertionError("execute")):
            self.assertEqual(self.run_bundle(value)["model_calls_made"], 0)

    def test_cli_accepts_only_explicit_pins_and_does_not_mutate_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "suite.json").write_text(json.dumps(suite()))
            (root / "bundle.json").write_text(json.dumps(bundle()))
            before = (root / "bundle.json").read_bytes()
            command = [sys.executable, "-m", "evaluation.quality_evidence", str(root / "suite.json"),
                       str(root / "bundle.json"), digest(suite()), COMMIT]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(json.loads(result.stdout)["expected_units"], 148)
            self.assertEqual((root / "bundle.json").read_bytes(), before)
            command[-1] = "f" * 40
            self.assertNotEqual(subprocess.run(command, cwd=ROOT, capture_output=True, timeout=10).returncode, 0)

    def test_invalid_utf8_is_rejected_with_sanitized_cli_errors(self):
        for raw in ('"\ud800"', '"\\ud800"'):
            with self.assertRaises(EvidenceRejected):
                strict_json(raw)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            private_name = root / "private-input-name.json"
            private_name.write_bytes(b"\xffSECRET-MARKER")
            command = [sys.executable, "-m", "evaluation.quality_evidence", str(private_name),
                       str(private_name), digest(suite()), COMMIT]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, b"")
            self.assertEqual(result.stderr, b"evaluation evidence rejected\n")

    def test_strict_json_and_suite_golden_binding_reject_malformed_documents(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '[' * 1100):
            with self.assertRaises(EvidenceRejected):
                strict_json(raw)
        with self.assertRaises(EvidenceRejected):
            validate_suite(suite(), "f" * 64)


if __name__ == "__main__":
    unittest.main()
