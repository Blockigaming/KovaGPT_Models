"""Provider-free scoring/accounting of supplied, unverified evaluation evidence.

This is a new ingestion schema, not reconstruction of the inaccessible historical
50-case suite. No generation, retry, tool/code execution or human review is run.
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from evaluation.offline import build_route_manifest, validate_response_artifact

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 16 * 1024 * 1024
TIMINGS = ("acknowledgement_ms", "first_answer_token_ms", "completed_answer_ms")
KINDS = ("synthetic", "recorded_unverified")
HISTORICAL_SUITE_SHA256 = "85d1883bab76f1d94da6a9ec63e9ddeee72e83736b9e84f13297b9d0dfe9a70a"


class EvidenceRejected(ValueError):
    pass


def need(condition):
    if not condition:
        raise EvidenceRejected("evaluation evidence rejected")


def keys(value, names):
    need(type(value) is dict and set(value) == set(names))


def bounded_int(value, minimum=0):
    need(type(value) is int and minimum <= value <= 2**53 - 1)


def text(value, limit=256):
    need(type(value) is str and 0 < len(value) <= limit)
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise EvidenceRejected("evaluation evidence rejected") from None


def pin(value, length=64):
    need(type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None)


def canonical(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        need(len(raw) <= MAX_BYTES)
        return raw.encode("ascii")
    except (ValueError, TypeError, RecursionError):
        raise EvidenceRejected("evaluation evidence rejected") from None


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def answer_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def strict_json(raw):
    need(type(raw) is str)
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result)
            result[key] = value
        return result
    def constant(_):
        raise EvidenceRejected("evaluation evidence rejected")
    try:
        need(len(raw.encode("utf-8")) <= MAX_BYTES)
        result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        canonical(result)
        # Reject invalid Unicode even when introduced through JSON escapes.
        json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return result
    except (ValueError, TypeError, RecursionError, OverflowError, UnicodeError):
        raise EvidenceRejected("evaluation evidence rejected") from None


def validate_suite(suite, expected_sha256):
    pin(expected_sha256)
    need(digest(suite) == expected_sha256)
    keys(suite, ("schema_version", "suite_id", "provenance", "cases"))
    need(type(suite["schema_version"]) is int and suite["schema_version"] == 1)
    text(suite["suite_id"])
    need(suite["provenance"] in ("source_regression_fixture", "operator_supplied_unverified"))
    need(type(suite["cases"]) is list and 1 <= len(suite["cases"]) <= 1000)
    cases = {}
    for case in suite["cases"]:
        keys(case, ("id", "prompt", "scorer", "expected_json", "rubric", "source_urls"))
        text(case["id"])
        text(case["prompt"], 250000)
        need(case["id"] not in cases and case["scorer"] in ("exact_json", "human_rubric"))
        need(type(case["source_urls"]) is list and len(case["source_urls"]) <= 100)
        need(all(type(url) is str and 0 < len(url) <= 2048 for url in case["source_urls"]))
        if case["scorer"] == "exact_json":
            need(case["rubric"] == [])
            canonical(case["expected_json"])
        else:
            need(case["expected_json"] is None and type(case["rubric"]) is list and 1 <= len(case["rubric"]) <= 32)
            for dimension in case["rubric"]:
                text(dimension)
            need(len(set(case["rubric"])) == len(case["rubric"]))
        cases[case["id"]] = case
    return cases


def analyze(suite, bundle, *, expected_suite_sha256, expected_source_commit):
    cases = validate_suite(suite, expected_suite_sha256)
    canonical(bundle)
    keys(bundle, ("schema_version", "kind", "suite_sha256", "source_commit", "attempts", "reviews"))
    need(type(bundle["schema_version"]) is int and bundle["schema_version"] == 1)
    need(bundle["kind"] in KINDS and bundle["suite_sha256"] == expected_suite_sha256)
    pin(expected_source_commit, 40)
    need(bundle["source_commit"] == expected_source_commit)
    need(type(bundle["attempts"]) is list and len(bundle["attempts"]) <= 10000)
    need(type(bundle["reviews"]) is list and len(bundle["reviews"]) <= 10000)
    manifest = {item["route_id"]: item for item in build_route_manifest()}
    actual_routes = set(manifest) - {"kova-auto"}
    attempts, groups = {}, defaultdict(list)
    lifecycles, seen_receipts = {}, set()
    for row in bundle["attempts"]:
        keys(row, ("id", "case_id", "case_sha256", "route_id", "actual_route_id", "condition", "ordinal",
                   "outcome", "answer", "answer_sha256", "runtime", "timings", "accounting"))
        for field in ("id", "case_id", "route_id", "actual_route_id", "condition"):
            text(row[field])
        need(row["id"] not in attempts and row["case_id"] in cases)
        need(row["case_sha256"] == digest(cases[row["case_id"]]))
        need(row["route_id"] in manifest and row["actual_route_id"] in actual_routes)
        need(row["route_id"] == "kova-auto" or row["actual_route_id"] == row["route_id"])
        need(row["condition"] in ("cold", "warm"))
        bounded_int(row["ordinal"], 1)
        need(row["outcome"] in ("success", "failed", "cancelled", "uncertain"))
        if row["answer"] is None:
            need(row["answer_sha256"] is None and row["outcome"] != "success")
        else:
            text(row["answer"], 750000)
            need(row["answer_sha256"] == answer_hash(row["answer"]))
        identity = row["runtime"]
        keys(identity, ("provider", "model", "model_revision", "image_sha256", "manifest_sha256",
                        "engine", "serving_version", "lifecycle_id", "context_tokens", "concurrency",
                        "gpu_type", "gpu_count", "quantization"))
        for key in ("provider", "model", "serving_version", "lifecycle_id", "gpu_type", "quantization"):
            text(identity[key])
        pin(identity["model_revision"], 40)
        pin(identity["image_sha256"])
        pin(identity["manifest_sha256"])
        need(identity["engine"] == manifest[row["actual_route_id"]]["engine"])
        bounded_int(identity["context_tokens"], 1)
        bounded_int(identity["concurrency"], 1)
        bounded_int(identity["gpu_count"], 1)
        identity_pin = digest({k: v for k, v in identity.items() if k != "lifecycle_id"})
        previous = lifecycles.setdefault(identity["lifecycle_id"], identity_pin)
        need(previous == identity_pin)
        keys(row["timings"], TIMINGS)
        for value in row["timings"].values():
            if value is not None:
                bounded_int(value)
        first, completed = (row["timings"][k] for k in TIMINGS[1:])
        need(first is None or completed is None or first <= completed)
        # A status/acknowledgement is never substituted for an answer token.
        need(row["answer"] is not None or first is None)
        account = row["accounting"]
        keys(account, ("attributable_cost_microusd", "receipt_sha256"))
        if account["attributable_cost_microusd"] is not None:
            bounded_int(account["attributable_cost_microusd"])
            pin(account["receipt_sha256"])
            need(account["receipt_sha256"] not in seen_receipts)
            seen_receipts.add(account["receipt_sha256"])
        else:
            need(account["receipt_sha256"] is None)
        attempts[row["id"]] = row
        groups[(row["case_id"], row["route_id"], row["condition"])].append(row)
    reviews, review_ids = {}, set()
    for review in bundle["reviews"]:
        keys(review, ("id", "attempt_id", "suite_sha256", "answer_sha256", "reviewer", "kind", "verdicts"))
        text(review["id"])
        text(review["reviewer"])
        text(review["attempt_id"])
        need(review["id"] not in review_ids and review["attempt_id"] in attempts and review["attempt_id"] not in reviews)
        row = attempts[review["attempt_id"]]
        case = cases[row["case_id"]]
        need(case["scorer"] == "human_rubric" and row["outcome"] == "success")
        need(review["kind"] == bundle["kind"] and review["suite_sha256"] == expected_suite_sha256)
        need(review["answer_sha256"] == row["answer_sha256"])
        keys(review["verdicts"], case["rubric"])
        need(all(v in ("pass", "fail", "pending") for v in review["verdicts"].values()))
        review_ids.add(review["id"])
        reviews[review["attempt_id"]] = review
    units = []
    for case_id, case in cases.items():
        for route in manifest:
            for condition in ("cold", "warm"):
                rows = sorted(groups[(case_id, route, condition)], key=lambda r: r["ordinal"])
                need([r["ordinal"] for r in rows] == list(range(1, len(rows) + 1)))
                # Do not select the best of multiple successes or permit a retry
                # after uncertain/cancelled work without a separately reviewed reconciliation.
                need(all(r["outcome"] == "failed" for r in rows[:-1]))
                configs = {digest({k: v for k, v in r["runtime"].items() if k != "lifecycle_id"}) for r in rows}
                need(len(configs) <= 1 and len({r["actual_route_id"] for r in rows}) <= 1)
                result, output_digest = "missing", None
                if rows:
                    row = rows[-1]
                    result = row["outcome"]
                    output_digest = row["answer_sha256"]
                    if row["outcome"] == "success":
                        violations = validate_response_artifact({"text": row["answer"], "user_source_urls": case["source_urls"]})
                        if violations:
                            result = "output_contract_failed"
                        elif case["scorer"] == "exact_json":
                            try:
                                result = "pass" if canonical(strict_json(row["answer"])) == canonical(case["expected_json"]) else "fail"
                            except EvidenceRejected:
                                result = "fail"
                        else:
                            review = reviews.get(row["id"])
                            verdicts = list(review["verdicts"].values()) if review else ["pending"]
                            result = "fail" if "fail" in verdicts else "pending_human_review" if "pending" in verdicts else "pass"
                units.append({"case_id": case_id, "route_id": route, "condition": condition,
                              "result": result, "attempts": len(rows), "answer_sha256": output_digest,
                              "configuration_sha256": next(iter(configs), None),
                              "actual_route_id": rows[-1]["actual_route_id"] if rows else None})
    costs = [r["accounting"]["attributable_cost_microusd"] for r in attempts.values()]
    known_cost = sum(c for c in costs if c is not None)
    bounded_int(known_cost)
    timing_groups = {}
    for condition in ("cold", "warm"):
        for route in manifest:
            rows = [r for r in attempts.values() if r["route_id"] == route and r["condition"] == condition and r["outcome"] == "success"]
            # Never aggregate incompatible runtime configurations into a purported comparison.
            by_config = defaultdict(list)
            for row in rows:
                by_config[digest({k: v for k, v in row["runtime"].items() if k != "lifecycle_id"})].append(row)
            for config, samples in by_config.items():
                measured = {}
                for metric in TIMINGS:
                    values = sorted(r["timings"][metric] for r in samples if r["timings"][metric] is not None)
                    measured[metric] = {"samples": len(values), "missing": len(samples) - len(values),
                        "mean": statistics.mean(values) if values else None,
                        "median": statistics.median(values) if values else None,
                        "p95_nearest_rank": values[math.ceil(len(values) * 0.95) - 1] if values else None}
                timing_groups[f"{route}|{condition}|{config}"] = measured
    counts = dict(Counter(u["result"] for u in units))
    return {"schema_version": 1, "kind": bundle["kind"], "source_commit": expected_source_commit,
            "suite_sha256": expected_suite_sha256, "bundle_sha256": digest(bundle),
            "expected_units": len(cases) * len(manifest) * 2, "unit_counts": counts, "units": units,
            "attempts": len(attempts), "attempt_outcomes": dict(Counter(r["outcome"] for r in attempts.values())),
            "reported_known_cost_microusd": known_cost, "missing_cost_attempts": costs.count(None),
            "reported_total_cost_microusd": known_cost if costs and None not in costs else None,
            "timings_by_configuration": timing_groups, "provided_reviews": len(reviews),
            "provenance_authenticated": False, "human_reviewer_identity_verified": False,
            "historical_suite_reconciled": False, "historical_suite_sha256": HISTORICAL_SUITE_SHA256,
            "phase_b_ready": False, "live_routes_verified": [], "model_calls_made": 0}


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print(json.dumps({"status": "quality_evidence_ingestion_source_only", "model_calls_made": 0,
                          "historical_suite_reconciled": False, "phase_b_ready": False}))
    else:
        try:
            need(len(sys.argv) == 5)
            def load(path):
                with open(path, "rb") as stream:
                    raw = stream.read(MAX_BYTES + 1)
                need(len(raw) <= MAX_BYTES)
                return strict_json(raw.decode("utf-8"))
            report = analyze(load(sys.argv[1]), load(sys.argv[2]),
                             expected_suite_sha256=sys.argv[3], expected_source_commit=sys.argv[4])
        except (EvidenceRejected, OSError, UnicodeError):
            print("evaluation evidence rejected", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(report, sort_keys=True))
