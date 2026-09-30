"""Verify the answer-bound diagnostic ledger, without changing historical scores."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from evaluation.historical_suite_bridge import load_archived_suite, HISTORICAL_CONTENT_SHA256
from evaluation.quality_evidence import canonical, need, strict_json

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "evaluations/a35-nova-preserved-classification.v1.json"
REPORT_SHA256 = "eca10a9f06af46c3b9deea90f9d6820ea231f496cbd2cb0a17d561e934867bb1"


def audit(report_path: Path | None = None) -> dict:
    ledger = strict_json(LEDGER.read_text())
    cases = {case["id"]: case for case in load_archived_suite()["cases"]}
    need(ledger["suite_sha256"] == HISTORICAL_CONTENT_SHA256)
    need(ledger["raw_report_sha256"] == REPORT_SHA256)
    need(ledger["preserved_strict_score"] == {"pass": 23, "total": 36})
    for name in ("phase_a_item_closed", "human_quality_review_complete", "report_bytes_modified"):
        need(ledger[name] is False)
    need(ledger["model_calls_made"] == 0)
    failures = ledger["strict_failures"]
    manual = ledger["manual_cases"]
    need(len(failures) == len({row["case_id"] for row in failures}) == 13)
    need(Counter(row["category"] for row in failures) == {
        "math": 4, "code_reading": 5, "reasoning": 3, "data_analysis": 1})
    need(len(manual) == len({row["case_id"] for row in manual}) == 14)
    need({row["case_id"] for row in manual} == {case["id"] for case in cases.values()
                                               if case["evaluation"]["kind"] == "review_required"})
    criteria = 0
    for row in manual:
        need(set(row["criteria_observations"]) == set(cases[row["case_id"]]["evaluation"]["criteria"]))
        need(all(value in ("observed_pass", "observed_fail", "pending")
                 for value in row["criteria_observations"].values()))
        need(row["formal_review_verdict"] == "pending")
        need(row["human_reviewer_identity_verified"] is False and row["quality_credit_awarded"] is False)
        need(row["historical_completion_evidence"] == "absent")
        criteria += len(row["criteria_observations"])
    need(criteria == 48)
    for row in failures:
        need(cases[row["case_id"]]["category"] == row["category"])
        need(cases[row["case_id"]]["evaluation"]["kind"] == "exact_json")
        need(row["recorded_result"] == "exact_json_fail" and bool(row["observed_failure_classes"]))
        need(row["dataset_training_adapter_decoder_cause"] == "not_isolated")
    if report_path is not None:
        raw = report_path.read_bytes()
        need(hashlib.sha256(raw).hexdigest() == REPORT_SHA256)
        report = strict_json(raw.decode())
        need(report["family"] == "kova-nova" and report["suite_sha256"] == HISTORICAL_CONTENT_SHA256)
        need(len(report["cases"]) == 50 and report["case_count"] == 50)
        need([r["case_id"] for r in report["cases"]] == list(cases))
        need(report["result_counts"] == {"exact_json_pass": 23, "exact_json_fail": 13,
                                          "pending_human_review": 14})
        rows = {row["case_id"]: row for row in report["cases"]}
        need({row["case_id"] for row in failures} ==
             {row["case_id"] for row in rows.values() if row["result"] == "exact_json_fail"})
        for row in failures + manual:
            answer = rows[row["case_id"]]["answer"]
            need(hashlib.sha256(answer.encode()).hexdigest() == row["answer_sha256"])
        # Check consistency with the historical parser, not the new completion
        # gate: preserve the original measurement and its absent finish metadata.
        for row in report["cases"]:
            case = cases[row["case_id"]]
            if case["evaluation"]["kind"] == "exact_json":
                matches = canonical(strict_json(row["answer"])) == canonical(case["evaluation"]["expected"])
                need(matches == (row["result"] == "exact_json_pass"))
    return {"strict_failures_classified": 13, "manual_cases_classified": 14,
            "manual_criteria_annotated": criteria, "historical_strict_score": "23/36",
            "raw_report_hash_checked": report_path is not None,
            "human_quality_review_complete": False, "phase_a_item_closed": False,
            "model_calls_made": 0}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.report), sort_keys=True))
