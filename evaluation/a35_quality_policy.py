"""Owner quality policy only; no runtime, reviewer, cost or route approval."""

from pathlib import Path

from evaluation.historical_suite_bridge import load_archived_suite, HISTORICAL_CONTENT_SHA256
from evaluation.quality_evidence import strict_json, need, digest

POLICY_PATH = Path(__file__).resolve().parents[1] / "config/a35-quality-policy.v1.json"


def load_policy() -> dict:
    policy = strict_json(POLICY_PATH.read_text())
    cases = load_archived_suite()["cases"]
    strict = [c for c in cases if c["evaluation"]["kind"] == "exact_json"]
    manual = [c for c in cases if c["evaluation"]["kind"] == "review_required"]
    expected = {name: sum(c["category"] == name for c in strict)
                for name in {c["category"] for c in strict}}
    need(policy["suite_sha256"] == HISTORICAL_CONTENT_SHA256)
    need(type(policy["required_repetitions"]) is int and policy["required_repetitions"] == 3)
    need(policy["strict_categories"] == expected)
    need(type(policy["strict_per_repetition"]) is int and
         policy["strict_per_repetition"] == len(strict) == 36)
    need(type(policy["strict_aggregate"]) is int and policy["strict_aggregate"] == 108)
    need(type(policy["manual_per_repetition"]) is int and
         policy["manual_per_repetition"] == len(manual) == 14)
    need(policy["manual_criteria_per_repetition"] ==
         sum(len(c["evaluation"]["criteria"]) for c in manual) == 48)
    for name in ("quality_requirement_owner_approved", "every_manual_criterion_must_pass"):
        need(policy[name] is True)
    for name in ("pooling_or_averaging_failures_allowed", "paid_execution_authorized",
                 "pilot_selection_authorized", "deployment_authorized", "release_approval_enabled",
                 "operational_thresholds_owner_approved"):
        need(policy[name] is False)
    need(type(policy["identity_safety_grounding_failures_allowed"]) is int and
         policy["identity_safety_grounding_failures_allowed"] == 0)
    need(policy["latency_targets"] is None and policy["cost_ceiling"] is None)
    return policy


def summarize_policy() -> dict:
    policy = load_policy()
    return {"policy_id": policy["policy_id"], "policy_sha256": digest(policy),
            "required_repetitions": policy["required_repetitions"],
            "strict_per_repetition": policy["strict_per_repetition"],
            "strict_aggregate": policy["strict_aggregate"],
            "manual_per_repetition": policy["manual_per_repetition"],
            "every_manual_criterion_must_pass": True,
            "operational_acceptance": "pending", "phase_a_item_closed": False,
            "model_calls_made": 0}


if __name__ == "__main__":
    import json
    print(json.dumps(summarize_policy(), sort_keys=True))
