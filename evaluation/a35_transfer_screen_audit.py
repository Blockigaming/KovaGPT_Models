"""Offline audit of the completed transfer screen; private reports stay external.

No model calls, output repair, new targets, manual packet, or quality claim.
The two recovered cases do not isolate why the other thirteen still fail.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from evaluation import a35_latest_screen_audit as latest
from evaluation.cpu_candidate_quality import score_case
from evaluation.historical_suite_bridge import load_archived_suite
from evaluation.quality_evidence import need

REPORT_SHA256 = "f9b1bf4f84cfcc128d5a5e7fae3db85c9169eda98b26bb8b944db2c6b304fa54"
SOURCE = "c7d4ee23684110caa30b52b4469af3e7ddecb8b2"
RUN = "a1d10484ddab453cac5f848306e7622c"
ADAPTER = "95fd3383589ef1bfe942345a9df09e3a3afaa8cf8d24ea886a273eea2a5089ee"
ADAPTER_CONFIG = "0a5a621a0ae182a379db70d87ace1faffcbec0ca980060e1f0ea7fc1ab841840"
PACK = "d0a7bbd518601ef95f9f04736aaad02fa5a20f92677042ae6d4ae3f8ce447d52"
PLAN = "b9fb23abf47430014aac237ff88a7c2739655c2d9cb785e77ac4d4cc6b15a2b9"
COUNTS = {"math": (6, 10), "code_reading": (2, 8), "reasoning": (3, 6),
          "data_analysis": (6, 6), "instruction_following": (6, 6)}
RECOVERED = ("data-05", "instructions-05")
FAILURES = {key: value for key, value in latest.FAILURES.items() if key not in RECOVERED}
FAILURES["logic-05"] = ("format_only", "The correct string is an object value instead of the JSON root.")


def audit(report_path, latest_path):
    latest.audit(latest_path)  # Separately prove the private 21/36 baseline.
    raw = report_path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == REPORT_SHA256)
    report = json.loads(raw)
    need((report["source_commit"], report["run_id"], report["prepared_pack_sha256"],
          report["plan_sha256"], report["suite_sha256"], report["base_revision"])
         == (SOURCE, RUN, PACK, PLAN, latest.SUITE, latest.BASE))
    need(report["generation_profile"]["system_prompt_sha256"] == latest.PROMPT)
    need(report["generation_profile"]["decoder"]["do_sample"] is False)
    need(report["generation_profile"]["chat_template_kwargs"] == {"enable_thinking": False})
    receipt = report["training_receipt"]
    need((receipt["source_commit"], receipt["run_id"], receipt["prepared_pack_sha256"],
          receipt["base_revision"]) == (SOURCE, RUN, PACK, latest.BASE))
    need(receipt["adapter_sha256"] == {"adapter_model.safetensors": ADAPTER,
                                       "adapter_config.json": ADAPTER_CONFIG})
    need((receipt["completed_epochs"], receipt["optimizer_steps"], receipt["training_records"],
          receipt["validation_records"]) == (1.0, 9, 72, 60))
    need(report["status"] == "strict_threshold_failed_manual_skipped")
    need(report["phase_a_item_closed"] is False and report["human_quality_review_complete"] is False)
    need(report["completion_evidence_authenticated"] is False and report["live_routes_verified"] == 0)
    previous = {row["case_id"]: row for row in json.loads(latest_path.read_bytes())["cases"]}
    historical = {row["case_id"]: row for row in json.loads(latest.previous.REPORT.read_bytes())["cases"]}
    strict = [case for case in load_archived_suite()["cases"] if case["evaluation"]["kind"] == "exact_json"]
    need(len(report["cases"]) == len(strict) == 36)
    need([row["case_id"] for row in report["cases"]] == [case["id"] for case in strict])
    ledger, all_cases = [], []
    def transition(before, after):
        return {(True, True): "retained_pass", (False, True): "recovered",
                (True, False): "newly_regressed", (False, False): "persistent_failure"}[before, after]
    for case, row in zip(strict, report["cases"], strict=True):
        case_id = case["id"]
        need(row["prompt_sha256"] == hashlib.sha256(case["prompt"].encode()).hexdigest())
        need(row["answer_sha256"] == hashlib.sha256(row["answer"].encode()).hexdigest())
        need(score_case(case, row["answer"], row["completion_evidence"])[0] == row["result"])
        evidence = row["completion_evidence"]
        need(row["completion_status"] == "verified_complete" and evidence["finish_reason"] == "eos")
        need(evidence["terminal_token_id"] in evidence["eos_token_ids"])
        need(0 < evidence["generated_token_count"] < evidence["max_new_tokens"] == 128)
        passed = row["result"] == "exact_json_pass"
        comparison = {"case_id": case_id, "category": case["category"], "result": row["result"],
            "against_latest_21": transition(previous[case_id]["result"] == "exact_json_pass", passed),
            "against_historical_23": transition(historical[case_id]["result"] == "exact_json_pass", passed),
            "output_changed_from_latest": row["answer"] != previous[case_id]["answer"]}
        all_cases.append(comparison)
        if passed:
            need(case_id not in FAILURES)
            continue
        kind, defect = FAILURES[case_id]
        need(latest.independent_expected(case_id) == case["evaluation"]["expected"])
        ledger.append({**comparison, "expected_behavior": case["prompt"],
            "expected": case["evaluation"]["expected"], "actual": row["answer"],
            "answer_sha256": row["answer_sha256"], "defect_kind": kind,
            "deterministic_output_defect": defect,
            "failure_class": latest.FAILURE_CLASSES[case_id],
            "observed_layer": "output formatting/JSON" if kind == "format_only" else "content/generalization behavior",
            "root_cause_category": "output formatting/JSON" if kind == "format_only" else "training/adapter behavior",
            "category_causality": "behavioral classification; a unique internal cause is not established",
            "internal_root_cause": "not isolated by these screens",
            "target_reference_defect": False, "truncation": False,
            "prompt_template_omission": False, "decoder_setting_drift": False,
            "dataset_or_training_defect_demonstrated": False,
            "next_hypothesis": "insufficient training exposure; controlled two-epoch probe, not a proven fix",
            "smallest_proposed_change": "same correct 72/60 examples and prompt; two complete epochs instead of one",
            "completion_evidence": evidence})
    need({row["case_id"] for row in ledger} == set(FAILURES))
    need(Counter(row["defect_kind"] for row in ledger) == {"content": 11, "format_only": 2})
    need([r["case_id"] for r in all_cases if r["against_latest_21"] == "recovered"] == list(RECOVERED))
    need(not [r for r in all_cases if r["against_latest_21"] == "newly_regressed"])
    need([r["case_id"] for r in all_cases if r["against_historical_23"] == "recovered"] == ["instructions-05"])
    need([r["case_id"] for r in all_cases if r["against_historical_23"] == "newly_regressed"] == ["code-06"])
    need(report["result_counts"] == {"exact_json_pass": 23, "exact_json_fail": 13})
    for category, (passed, total) in COUNTS.items():
        rows = [r for r in report["cases"] if r["category"] == category]
        counts = dict(Counter(r["result"] for r in rows))
        need(len(rows) == total and counts.get("exact_json_pass", 0) == passed)
        need(report["category_counts"][category] == counts)
    return {"attempt_id": RUN, "source_head": SOURCE, "candidate_sha256": ADAPTER,
        "report_sha256": REPORT_SHA256, "strict": "23/36", "latest_baseline": "21/36",
        "historical_baseline": "23/36", "category_pass_counts": {k: f"{p}/{n}" for k, (p, n) in COUNTS.items()},
        "cases": all_cases, "failures": ledger, "manual": "NOT RUN", "quality_gate_closed": False,
        "model_calls_made": 0, "all_outputs_complete": True,
        "generated_token_range": [min(r["completion_evidence"]["generated_token_count"] for r in report["cases"]),
                                  max(r["completion_evidence"]["generated_token_count"] for r in report["cases"])],
        "causal_limits": "The screen after adding correct examples recovered two format cases. All eleven content failures remain. Dataset weighting, optimization exposure and base-model capability remain unisolated. No further answer-specific data or prompt change is justified by this screen alone.",
        "authenticated_route_evidence": "Separate serving-route gate remains unverified; offline replay cannot establish it."}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preserved-report", required=True, type=Path)
    parser.add_argument("--latest-report", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.preserved_report, args.latest_report), indent=2, sort_keys=True))
