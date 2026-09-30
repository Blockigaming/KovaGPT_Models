"""Measure the saved CPU candidates against the pinned 50-case quality suite.

This is one offline candidate diagnostic, not a 37-route release benchmark or
an authenticated human review. No generated answer is executed or judged by a
model. The only published answers come from public-safe suite prompts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.historical_suite_bridge import (
    HISTORICAL_CONTENT_SHA256, load_archived_suite,
)
from evaluation.quality_evidence import (
    EvidenceRejected, canonical, strict_json, validate_response_artifact,
)
from training.candidate_inference_smoke import generate
from evaluation.completion_evidence import completion_status, output_budgets


def score_case(case: dict, answer: str,
               completion_evidence: dict | None = None) -> tuple[str, str | None]:
    """Keep hidden reasoning out of the review file; leave manual cases pending."""
    if any(marker in answer.lower() for marker in ("<think", "</think>")):
        return "private_output_blocked", None
    if validate_response_artifact({"text": answer, "user_source_urls": []}):
        return "output_contract_failed", answer
    status = completion_status(completion_evidence)
    if case["evaluation"]["kind"] == "review_required":
        return ("pending_human_review" if status == "verified_complete" else
                "incomplete_output" if status == "length_limit" else
                "completion_unverified"), answer
    try:
        actual = strict_json(answer)
    except EvidenceRejected:
        return "exact_json_fail", answer
    expected = case["evaluation"]["expected"]
    if canonical(actual) != canonical(expected):
        return "exact_json_fail", answer
    return ("exact_json_pass" if status == "verified_complete" else
            "incomplete_output" if status == "length_limit" else
            "completion_unverified"), answer


def run(family: str, snapshot: Path, candidate: Path, output: Path,
        *, manual_max_new_tokens: int) -> dict:
    if output.exists() or not output.parent.is_dir():
        raise ValueError("new report path required")
    suite = load_archived_suite()
    cases = suite["cases"]
    budgets = output_budgets(cases, manual_max_new_tokens)
    with TemporaryDirectory(prefix="kova-cpu-quality-") as temporary:
        sample_path = Path(temporary) / "samples.json"
        sample_report = generate(
            family, snapshot, candidate, sample_path,
            prompts=tuple(case["prompt"] for case in cases), token_budgets=budgets)
        results = []
        counts = Counter()
        categories = defaultdict(Counter)
        samples = sample_report["samples"]
        if len(samples) != len(cases):
            raise ValueError("incomplete candidate generation")
        for case, sample in zip(cases, samples):
            if sample["prompt"] != case["prompt"]:
                raise ValueError("candidate prompt mismatch")
            evidence = sample.get("completion_evidence")
            result, safe_answer = score_case(case, sample["completion"], evidence)
            counts[result] += 1
            categories[case["category"]][result] += 1
            results.append({"case_id": case["id"], "category": case["category"],
                            "result": result, "answer": safe_answer,
                            "completion_evidence": evidence,
                            "completion_status": completion_status(evidence)})
    report = {
        "kind": "experimental_cpu_candidate_quality_diagnostic",
        "family": family, "suite_sha256": HISTORICAL_CONTENT_SHA256,
        "training_commit": sample_report["training_commit"],
        "base_revision": sample_report["base_revision"],
        "adapter_sha256": sample_report["adapter_sha256"],
        "generation": {"do_sample": False, "enable_thinking": False,
                       "strict_max_new_tokens": 128,
                       "manual_max_new_tokens": manual_max_new_tokens},
        "generation_profile": sample_report["generation_profile"],
        "completion_evidence_authenticated": False,
        "case_count": len(cases), "result_counts": dict(counts),
        "category_counts": {key: dict(value) for key, value in categories.items()},
        "cases": results, "human_quality_review_complete": False,
        "release_thresholds_approved": False, "live_routes_verified": 0,
        "phase_a_item_closed": False,
    }
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("kova-cosmo", "kova-orion"), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manual-max-new-tokens", type=int, required=True)
    args = parser.parse_args()
    measured = run(args.family, args.snapshot, args.candidate, args.output,
                   manual_max_new_tokens=args.manual_max_new_tokens)
    print(json.dumps({"family": args.family, "case_count": measured["case_count"],
                      "result_counts": measured["result_counts"]}, sort_keys=True))
