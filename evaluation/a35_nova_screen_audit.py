"""Offline audit of one preserved Nova screen; no generation or answer repair.

The deterministic output defects are known. Their statistical cause inside the
model is not isolated by one run. The next prompt is a bounded intervention,
not a claim that software tests have made the candidate pass.
"""

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from evaluation.a35_correction_data import reference_answer
from evaluation.cpu_candidate_quality import score_case
from evaluation.historical_suite_bridge import load_archived_suite
from evaluation.quality_evidence import need

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "evaluations/a35-nova-screen-9ce963e3b2.json"
REPORT_SHA256 = "9afd9826d43d2c620bca75e9181552c0598411e48fb938af91669a7c5013c95b"
SOURCE = "1783367753b38ed662c611f7e2f45dfcda531a7e"
RUN = "9ce963e3b2f34b578810ee505d008965"
ADAPTER = "95470d46db4a430ce6f5e5d53f5dc322dcb4a04aec824b69b922dbe9b1d71213"
PACK = "fcbe8556c9587d960a531bd0f91d5b7de13fa77fd69d3aa9d62a893dcab91726"

# Case IDs, goldens and preserved answers remain audit-only, never model inputs.
# Each tuple: defect kind, observable defect, next generic instruction, coverage.
FAILURES = {
    "math-03": ("content", "Returns 204*3, omitting the parenthesized +19 term.",
                "check arithmetic precedence and carries", "arithmetic"),
    "math-05": ("content", "The add-then-multiply result is wrong; no unique internal calculation can be inferred.",
                "check arithmetic precedence and carries", "arithmetic"),
    "math-07": ("content", "Wrong without-replacement denominator plus an unrequested result wrapper.",
                "update both eligible and total counts after each draw and reduce the resulting fraction", "probability"),
    "math-10": ("content", "Area is 21 instead of 9*(30/2-9); it coincides with perimeter minus length.",
                "Derive rectangle sides from perimeter before computing area", "rectangle"),
    "code-01": ("content", "Odd inputs survive the predicate; the first transformed value is also wrong.",
                "test the predicate on the original value, transform only survivors, and retain duplicates", "even-filter"),
    "code-02": ("content", "Maps every element; nonmultiples of three survive the predicate.",
                "test the predicate on the original value, transform only survivors, and retain duplicates", "modulo-filter"),
    "code-03": ("content", "The odd input survives the predicate; remaining order and duplicates are preserved.",
                "test the predicate on the original value, transform only survivors, and retain duplicates", "even-filter"),
    "code-04": ("content", "Maps every element; nonmultiples of three survive the predicate.",
                "test the predicate on the original value, transform only survivors, and retain duplicates", "modulo-filter"),
    "code-07": ("content", "The destination becomes the source value instead of adding to its current value.",
                "For +=, add to the current value", "increment"),
    "logic-01": ("content", "Substitutes numeric positions for task IDs and adds an unrequested object wrapper.",
                 "Schedule only ready tasks and preserve their identifiers, never substitute positions", "task-order"),
    "logic-05": ("format_only", "The correct string is wrapped in next_stage instead of being the JSON root.",
                 "a requested string, integer or boolean stands alone", "next-stage"),
    "logic-06": ("format_only", "The correct integer is wrapped in value instead of being the JSON root.",
                 "a requested string, integer or boolean stands alone", "integer-root"),
    "instructions-05": ("format_only", "The correct boolean is wrapped in boolean instead of being the JSON root.",
                        "a requested string, integer or boolean stands alone", "boolean-root"),
}


def independent_expected(case_id):
    """Test-only calculations from the unchanged prompts, not model generation."""
    if case_id == "math-03":
        return {"answer": (204 + 19) * 3}
    if case_id == "math-05":
        return {"answer": (125 + 8) * 9}
    if case_id == "math-07":
        probability = Fraction(3, 8) * Fraction(2, 7)
        return {"numerator": probability.numerator, "denominator": probability.denominator}
    if case_id == "math-10":
        return {"area": 9 * (30 // 2 - 9)}
    if case_id.startswith("code-") and case_id != "code-07":
        data = {"code-01": ([1, 4, 7, 2, 9, 0], 2),
                "code-02": ([3, 3, 2, 6, 5], 3),
                "code-03": ([8, 1, 8, 4, 2], 2),
                "code-04": ([6, 9, 12, 2, 5], 3)}
        values, divisor = data[case_id]
        return [value * 2 for value in values if value % divisor == 0]
    if case_id == "code-07":
        values = {"a": 2, "b": 3}
        values["a"] += values["b"]
        return values
    if case_id == "logic-01":
        return reference_answer("dependency_schedule", {
            "tasks": {"A": [], "B": [], "C": ["A", "B"], "D": ["C"]}, "partition": False})
    if case_id == "logic-05":
        return reference_answer("ready_stage", {"tasks": {
            "p1": [], "a1": ["p1"], "c1": ["p1", "a1"], "v1": ["p1", "a1", "c1"]},
            "available": ["p1", "a1"]})
    if case_id == "logic-06":
        matches = [n for n in range(12, 14) if n % 2]
        need(len(matches) == 1)
        return matches[0]
    need(case_id == "instructions-05")
    return False


def audit():
    raw = REPORT.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == REPORT_SHA256)
    report = json.loads(raw)
    need((report["source_commit"], report["run_id"], report["prepared_pack_sha256"]) == (SOURCE, RUN, PACK))
    receipt = report["training_receipt"]
    need(receipt["adapter_sha256"]["adapter_model.safetensors"] == ADAPTER)
    need(receipt["completed_epochs"] == 1.0 and receipt["optimizer_steps"] == 8)
    need((receipt["training_records"], receipt["validation_records"]) == (61, 49))
    strict = [c for c in load_archived_suite()["cases"] if c["evaluation"]["kind"] == "exact_json"]
    need(len(report["cases"]) == 36)
    need([r["case_id"] for r in report["cases"]] == [c["id"] for c in strict])
    ledger = []
    for case, row in zip(strict, report["cases"], strict=True):
        need(row["prompt_sha256"] == hashlib.sha256(case["prompt"].encode()).hexdigest())
        need(row["answer_sha256"] == hashlib.sha256(row["answer"].encode()).hexdigest())
        need(score_case(case, row["answer"], row["completion_evidence"])[0] == row["result"])
        need(row["completion_status"] == "verified_complete")
        need(row["completion_evidence"]["finish_reason"] == "eos")
        if row["result"] == "exact_json_pass":
            need(case["id"] not in FAILURES)
            continue
        kind, defect, correction, coverage = FAILURES[case["id"]]
        need(independent_expected(case["id"]) == case["evaluation"]["expected"])
        ledger.append({"case_id": case["id"], "category": case["category"],
            "prompt": case["prompt"], "expected": case["evaluation"]["expected"],
            "actual": row["answer"], "answer_sha256": row["answer_sha256"],
            "defect_kind": kind, "deterministic_defect": defect,
            "correction_layer": "output formatting/JSON" if kind == "format_only" else "prompt/template",
            "smallest_correction": correction, "existing_dataset_group": coverage,
            "model_internal_root_cause": "not isolated by this single screen",
            "target_reference_defect": False, "truncation": False,
            "completion_evidence": row["completion_evidence"]})
    need({r["case_id"] for r in ledger} == set(FAILURES))
    need(Counter(r["defect_kind"] for r in ledger) == {"content": 10, "format_only": 3})
    return {"attempt_id": RUN, "source_head": SOURCE, "candidate_sha256": ADAPTER,
        "report_sha256": REPORT_SHA256, "strict": "23/36", "data": "6/6",
        "manual": "NOT RUN", "model_calls_made": 0, "quality_gate_closed": False,
        "causal_limits": "Coverage exists; no reference error, truncation or decoder-setting drift found. Dataset weighting, adapter/training generalization and model capability remain unisolated. The prompt intervention is unmeasured.",
        "failures": ledger}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
