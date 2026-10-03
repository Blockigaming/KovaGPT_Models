"""Replay preserved Nova outputs without generation, repair or quality claims.

The 21/36 screen measured the task-check prompt intervention. It fixed none of
the previous 13 failures and introduced two regressions. Observable output
defects are reproducible; one screen does not isolate model-internal causes.
This module and its benchmark oracles are audit-only, never training inputs.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path

from evaluation import a35_nova_screen_audit as previous
from evaluation.cpu_candidate_quality import score_case
from evaluation.historical_suite_bridge import load_archived_suite
from evaluation.quality_evidence import need

ROOT = Path(__file__).resolve().parents[1]
REPORT_SHA256 = "d7d5fd2e5e4b9b25b50d55316a883ae290f742c7cfcff1676cb293a79ff9bfe2"
SOURCE = "cfb0ddb3e9db6ddb7812233a1529589024318c3a"
RUN = "bb328ecf604e4a03b99a1eef98c7575e"
ADAPTER = "9e9c991332e979d3b3b73452e643d0272f859965fc054329f32308491a0a7e0b"
ADAPTER_CONFIG = "65437f3e906442b3da8d91cb161123ed02d769bd156b563fbcb2e9aca8e1236e"
PACK = "272d2a6e98fa57fec743618591d054e31298f6d36f01e5eacdc61ece61e1e591"
PROMPT = "258b48ccdc670cfd32312983de0ca4e71b136de1e71aa4ef97fb88d026e554d8"
PLAN = "bff6646fae9d67a4f8e51664aeaeef0256e8341ff20bd772eb114ea8f07ecc82"
SUITE = "85d1883bab76f1d94da6a9ec63e9ddeee72e83736b9e84f13297b9d0dfe9a70a"
BASE = "1cfa9a7208912126459214e8b04321603b3df60c"
COUNTS = {"math": (6, 10), "code_reading": (2, 8), "reasoning": (3, 6),
          "data_analysis": (5, 6), "instruction_following": (5, 6)}

# Benchmark IDs and goldens stay exclusively in the offline audit layer.
# Content includes errors that also have an additional formatting defect.
FAILURES = {
    "math-03": ("content", "The add-then-multiply answer is wrong; its internal calculation is not recoverable."),
    "math-05": ("content", "The add-then-multiply answer is wrong; its internal calculation is not recoverable."),
    "math-07": ("content", "The without-replacement probability has a wrong denominator and an unrequested result wrapper."),
    "math-10": ("content", "The area equals perimeter minus length instead of length times the derived width."),
    "code-01": ("content", "Inputs rejected by the even predicate survive; the first transformed value is also wrong."),
    "code-02": ("content", "All inputs are doubled, including values rejected by the multiple-of-three predicate."),
    "code-03": ("content", "All inputs are doubled, including the value rejected by the even predicate."),
    "code-04": ("content", "All inputs are doubled, including values rejected by the multiple-of-three predicate."),
    "code-06": ("content", "A mutation of a separate copied list is incorrectly reflected in the original list."),
    "code-07": ("content", "The destination is replaced with the source rather than incremented by it."),
    "logic-01": ("content", "Task identifiers are replaced by numeric positions with an unrequested order wrapper."),
    "logic-05": ("format_only", "The correct string is wrapped in next_ready_stage instead of being the JSON root."),
    "logic-06": ("format_only", "The correct integer is wrapped in value instead of being the JSON root."),
    "data-05": ("format_only", "The correct repeated identifiers are emitted as a Python set literal, not a JSON array."),
    "instructions-05": ("format_only", "The correct boolean is wrapped in false instead of being the JSON root."),
}

FAILURE_CLASSES = {
    "math-03": "arithmetic", "math-05": "arithmetic", "math-07": "probability",
    "math-10": "geometry", "code-01": "predicate", "code-02": "predicate",
    "code-03": "predicate", "code-04": "predicate", "code-06": "copy",
    "code-07": "increment", "logic-01": "task_ids", "logic-05": "string_root",
    "logic-06": "integer_root", "data-05": "array_root", "instructions-05": "boolean_root",
}


def independent_expected(case_id):
    """Independent unchanged-prompt oracles, never substituted into output."""
    if case_id == "code-06":
        original = [1, 2]
        copied = original.copy()
        copied.append(3)
        need(copied is not original)
        return original
    if case_id == "data-05":
        counts = Counter(["a", "b", "a", "c", "b"])
        return sorted(identifier for identifier, count in counts.items() if count > 1)
    return previous.independent_expected(case_id)


def audit(report_path):
    raw = report_path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == REPORT_SHA256)
    report = json.loads(raw)
    need((report["source_commit"], report["run_id"], report["prepared_pack_sha256"],
          report["plan_sha256"], report["suite_sha256"], report["base_revision"])
         == (SOURCE, RUN, PACK, PLAN, SUITE, BASE))
    need(report["generation_profile"]["system_prompt_sha256"] == PROMPT)
    need(report["generation_profile"]["decoder"]["do_sample"] is False)
    need(report["generation_profile"]["chat_template_kwargs"] == {"enable_thinking": False})
    receipt = report["training_receipt"]
    need((receipt["source_commit"], receipt["run_id"], receipt["prepared_pack_sha256"],
          receipt["base_revision"]) == (SOURCE, RUN, PACK, BASE))
    need(receipt["adapter_sha256"] == {"adapter_model.safetensors": ADAPTER,
                                       "adapter_config.json": ADAPTER_CONFIG})
    need(receipt["completed_epochs"] == 1.0 and receipt["optimizer_steps"] == 8)
    need((receipt["training_records"], receipt["validation_records"]) == (61, 49))
    need(report["status"] == "strict_threshold_failed_manual_skipped")
    need(report["phase_a_item_closed"] is False and report["human_quality_review_complete"] is False)
    need(report["completion_evidence_authenticated"] is False and report["live_routes_verified"] == 0)
    old_raw = previous.REPORT.read_bytes()
    need(hashlib.sha256(old_raw).hexdigest() == previous.REPORT_SHA256)
    old_rows = {row["case_id"]: row for row in json.loads(old_raw)["cases"]}
    strict = [case for case in load_archived_suite()["cases"] if case["evaluation"]["kind"] == "exact_json"]
    need(len(report["cases"]) == len(strict) == 36)
    need([row["case_id"] for row in report["cases"]] == [case["id"] for case in strict])
    ledger, regressions, improvements = [], [], []
    for case, row in zip(strict, report["cases"], strict=True):
        case_id = case["id"]
        need(row["prompt_sha256"] == hashlib.sha256(case["prompt"].encode()).hexdigest())
        need(row["answer_sha256"] == hashlib.sha256(row["answer"].encode()).hexdigest())
        need(score_case(case, row["answer"], row["completion_evidence"])[0] == row["result"])
        need(row["completion_status"] == "verified_complete")
        evidence = row["completion_evidence"]
        need(evidence["finish_reason"] == "eos" and evidence["terminal_token_id"] in evidence["eos_token_ids"])
        need(0 < evidence["generated_token_count"] < evidence["max_new_tokens"] == 128)
        passed = row["result"] == "exact_json_pass"
        old_passed = old_rows[case_id]["result"] == "exact_json_pass"
        if old_passed and not passed:
            regressions.append(case_id)
        if passed and not old_passed:
            improvements.append(case_id)
        if passed:
            need(case_id not in FAILURES)
            continue
        kind, defect = FAILURES[case_id]
        need(independent_expected(case_id) == case["evaluation"]["expected"])
        ledger.append({"case_id": case_id, "category": case["category"], "prompt": case["prompt"],
            "expected": case["evaluation"]["expected"], "actual": row["answer"],
            "answer_sha256": row["answer_sha256"], "defect_kind": kind,
            "deterministic_output_defect": defect, "previous_actual": old_rows[case_id]["answer"],
            "new_regression": old_passed, "target_reference_defect": False, "truncation": False,
            "model_internal_root_cause": "not isolated by these two screens",
            "failure_class": FAILURE_CLASSES[case_id],
            "observed_layer": "format/JSON behavior" if kind == "format_only" else "content/generalization behavior",
            "proposed_correction_layer": "dataset coverage / insufficient examples hypothesis",
            "smallest_supported_intervention": "one distinct train/validation scenario pair for " + FAILURE_CLASSES[case_id],
            "correction_is_model_quality_proof": False,
            "prompt_template_omission": False, "decoder_setting_drift": False,
            "incorrect_or_conflicting_reference_found": False,
            "regression_attribution": "changed prompt and adapter are confounded" if old_passed else "not a new regression",
            "completion_evidence": evidence})
    need({row["case_id"] for row in ledger} == set(FAILURES))
    need(Counter(row["defect_kind"] for row in ledger) == {"content": 11, "format_only": 4})
    need(regressions == ["code-06", "data-05"] and improvements == [])
    need(report["result_counts"] == {"exact_json_pass": 21, "exact_json_fail": 15})
    for category, (passed, total) in COUNTS.items():
        rows = [row for row in report["cases"] if row["category"] == category]
        need(len(rows) == total and sum(row["result"] == "exact_json_pass" for row in rows) == passed)
        need(report["category_counts"][category] == {"exact_json_pass": passed, "exact_json_fail": total - passed})
    tokens = [row["completion_evidence"]["generated_token_count"] for row in report["cases"]]
    return {"attempt_id": RUN, "source_head": SOURCE, "candidate_sha256": ADAPTER,
        "report_sha256": REPORT_SHA256, "prepared_pack_sha256": PACK, "system_prompt_sha256": PROMPT,
        "execution_plan_sha256": PLAN, "strict": "21/36", "previous_strict": "23/36",
        "category_pass_counts": {key: f"{passed}/{total}" for key, (passed, total) in COUNTS.items()},
        "manual": "NOT RUN", "all_outputs_complete": True,
        "generated_token_range": [min(tokens), max(tokens)],
        "new_regressions": regressions, "new_passes": improvements,
        "previous_failures_still_failing": sorted(set(FAILURES) & set(previous.FAILURES)),
        "previous_intervention_findings": "The prior correction changed only the shared system instruction for 110 rows and evaluation. All failing skill families already had correct examples. The added instructions were present live, but no original failure passed. This demonstrates lack of measured transfer, not an isolated internal cause.",
        "model_calls_made": 0, "quality_gate_closed": False,
        "authenticated_route_evidence": "NOT VERIFIED by this offline replay; preserved separately",
        "causal_limits": "The measured task-check prompt intervention fixed no original failure. No target defect, truncation or omitted task-check prompt explains this result. Dataset weighting, adapter/training generalization and model capability remain unisolated.",
        "failures": ledger}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preserved-report", type=Path, required=True,
                        help="Private preserved report; never copied into the public repository")
    args = parser.parse_args()
    print(json.dumps(audit(args.preserved_report), indent=2, sort_keys=True))
