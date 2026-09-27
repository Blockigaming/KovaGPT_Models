"""Verify and map the archived 50-case source suite without running models.

This bridge carries only archived case definitions into the supplemental evidence
schema. It does not load the historical evaluator, collect outcomes, authenticate
reviewers, approve an evaluation, or change release/training gates.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from evaluation.quality_evidence import (
    MAX_BYTES, ROOT, EvidenceRejected, analyze, build_route_manifest, digest, keys,
    need, strict_json, text, validate_suite,
)

SUITE_PATH = ROOT / "evaluations" / "model-quality-suite.v1.json"
HISTORICAL_FILE_SHA256 = "8c91f70c8e4d0522c48aa7b1588b9c427e49a0f5a2a9b0bfc0c5091320c3150a"
HISTORICAL_CONTENT_SHA256 = "85d1883bab76f1d94da6a9ec63e9ddeee72e83736b9e84f13297b9d0dfe9a70a"
MAPPED_SUITE_SHA256 = "b6fdd7be8734a79d1cec45a5a07672eeac6add1618781aae33c5e5e17d5aae28"
EXPECTED_CATEGORIES = {
    "math": 10, "code_reading": 8, "reasoning": 6, "data_analysis": 6,
    "instruction_following": 6, "coding_implementation": 4,
    "writing": 2, "research_grounding": 4, "identity_and_safety": 4,
}


def load_archived_suite():
    """Require the exact archived bytes and original fail-closed policy fields."""
    with SUITE_PATH.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    need(len(raw) <= MAX_BYTES and hashlib.sha256(raw).hexdigest() == HISTORICAL_FILE_SHA256)
    historical = strict_json(raw.decode("utf-8"))
    need(digest(historical) == HISTORICAL_CONTENT_SHA256)
    keys(historical, ("schema_version", "id", "source", "purpose", "cases",
                      "release_approval_enabled", "paid_execution_enabled",
                      "required_repetitions_per_condition", "per_route_acceptance_thresholds",
                      "latency_targets", "review_policy"))
    need(type(historical["schema_version"]) is int and historical["schema_version"] == 1)
    need(historical["id"] == "kova-model-quality-v1")
    need(historical["source"] == "authored_public_safe_synthetic_prompts")
    need(historical["purpose"] == "candidate_and_route_evaluation_preparation_not_benchmark_results")
    need(historical["release_approval_enabled"] is False and historical["paid_execution_enabled"] is False)
    need(historical["required_repetitions_per_condition"] is None)
    need(historical["per_route_acceptance_thresholds"] is None and historical["latency_targets"] is None)
    need(historical["review_policy"] == "manual_review_required_for_rubric_cases_no_automatic_model_judge")
    need(type(historical["cases"]) is list and len(historical["cases"]) == 50)
    return historical


def map_cases(historical):
    """Keep exact prompts/goldens/rubrics; preserve policy metadata in the source file."""
    cases = []
    categories, kinds = Counter(), Counter()
    for item in historical["cases"]:
        keys(item, ("id", "category", "prompt", "evaluation"))
        text(item["category"])
        categories[item["category"]] += 1
        evaluation = item["evaluation"]
        need(type(evaluation) is dict)
        kind = evaluation.get("kind")
        need(type(kind) is str)
        kinds[kind] += 1
        if kind == "exact_json":
            keys(evaluation, ("kind", "expected"))
        else:
            need(kind == "review_required")
            keys(evaluation, ("kind", "criteria"))
        cases.append({
            "id": item["id"], "prompt": item["prompt"],
            "scorer": "exact_json" if kind == "exact_json" else "human_rubric",
            "expected_json": evaluation["expected"] if kind == "exact_json" else None,
            "rubric": [] if kind == "exact_json" else evaluation["criteria"],
            "source_urls": [],
        })
    need(categories == EXPECTED_CATEGORIES and kinds == {"exact_json": 36, "review_required": 14})
    mapped = {
        "schema_version": 1, "suite_id": historical["id"],
        "provenance": "operator_supplied_unverified", "cases": cases,
    }
    validate_suite(mapped, MAPPED_SUITE_SHA256)
    return mapped


def load_mapped_suite():
    return map_cases(load_archived_suite())


def source_report():
    mapped = load_mapped_suite()
    routes = build_route_manifest()
    return {
        "case_count": len(mapped["cases"]),
        "exact_json_cases": sum(c["scorer"] == "exact_json" for c in mapped["cases"]),
        "manual_review_cases": sum(c["scorer"] == "human_rubric" for c in mapped["cases"]),
        "route_contracts": len(routes),
        "expected_cold_warm_units": len(mapped["cases"]) * len(routes) * 2,
        "historical_file_sha256": HISTORICAL_FILE_SHA256,
        "historical_suite_sha256": HISTORICAL_CONTENT_SHA256,
        "mapped_suite_sha256": MAPPED_SUITE_SHA256,
        "model_calls_made": 0,
        "historical_suite_reconciled": False,
        "phase_b_ready": False,
    }


def summarize_evidence(bundle, *, expected_source_commit):
    """Group unverified results by original category without losing missing units.

    The archived case bytes and the mapped schema are checked anew; this only
    summarizes the current analyzer's result. It sets no acceptance thresholds,
    authenticates no run/reviewer, and calls no model.
    """
    historical = load_archived_suite()
    mapped = map_cases(historical)
    report = analyze(mapped, bundle, expected_suite_sha256=MAPPED_SUITE_SHA256,
                     expected_source_commit=expected_source_commit)
    categories = {case["id"]: case["category"] for case in historical["cases"]}
    need(len(categories) == len(mapped["cases"]))
    counts = {name: Counter() for name in EXPECTED_CATEGORIES}
    route_count = len(build_route_manifest())
    for unit in report["units"]:
        counts[categories[unit["case_id"]]][unit["result"]] += 1
    coverage = {
        name: {"case_count": case_count, "expected_units": case_count * route_count * 2,
               "unit_counts": dict(counts[name])}
        for name, case_count in EXPECTED_CATEGORIES.items()
    }
    need(all(sum(item["unit_counts"].values()) == item["expected_units"]
             for item in coverage.values()))
    need(sum((counter for counter in counts.values()), Counter()) == Counter(report["unit_counts"]))
    return {
        "schema_version": 1, "kind": report["kind"],
        "source_commit": report["source_commit"],
        "historical_file_sha256": HISTORICAL_FILE_SHA256,
        "historical_suite_sha256": HISTORICAL_CONTENT_SHA256,
        "mapped_suite_sha256": MAPPED_SUITE_SHA256,
        "bundle_sha256": report["bundle_sha256"],
        "expected_units": report["expected_units"], "unit_counts": report["unit_counts"],
        "category_coverage": coverage, "attempts": report["attempts"],
        "provided_reviews": report["provided_reviews"],
        "reported_total_cost_microusd": report["reported_total_cost_microusd"],
        "provenance_authenticated": report["provenance_authenticated"],
        "human_reviewer_identity_verified": report["human_reviewer_identity_verified"],
        "historical_suite_reconciled": report["historical_suite_reconciled"],
        "phase_b_ready": report["phase_b_ready"],
        "live_routes_verified": report["live_routes_verified"],
        "model_calls_made": report["model_calls_made"],
    }


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--evidence", type=Path)
        parser.add_argument("--source-commit")
        args = parser.parse_args()
        need((args.evidence is None) == (args.source_commit is None))
        if args.evidence is None:
            report = source_report()
        else:
            with args.evidence.open("rb") as stream:
                raw = stream.read(MAX_BYTES + 1)
            need(len(raw) <= MAX_BYTES)
            report = summarize_evidence(strict_json(raw.decode("utf-8")),
                                        expected_source_commit=args.source_commit)
        print(json.dumps(report, sort_keys=True))
    except (EvidenceRejected, OSError, UnicodeError, ValueError):
        raise SystemExit("historical suite source rejected") from None
