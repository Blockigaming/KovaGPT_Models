"""Validate the owner-approved three-family product policy without enabling it."""
import json
from pathlib import Path
import sys

from training.three_family_contract import ContractError, load_json

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config/current-product-policy.v3.json"
MAX_POLICY_BYTES = 64 * 1024
CurrentPolicyError = ContractError
CANONICAL_POLICY = load_json(POLICY_PATH, maximum_bytes=MAX_POLICY_BYTES)


def _exact(actual, expected):
    if type(actual) is not type(expected):
        raise CurrentPolicyError("current product policy rejected")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise CurrentPolicyError("current product policy rejected")
        for key in expected:
            _exact(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise CurrentPolicyError("current product policy rejected")
        for left, right in zip(actual, expected, strict=True):
            _exact(left, right)
    elif actual != expected:
        raise CurrentPolicyError("current product policy rejected")


def load_policy():
    try:
        value = load_json(POLICY_PATH, maximum_bytes=MAX_POLICY_BYTES)
        _exact(value, CANONICAL_POLICY)
        return value
    except ContractError:
        raise CurrentPolicyError("current product policy rejected") from None


def validate():
    try:
        value = load_policy()
    except ContractError:
        raise CurrentPolicyError("current product policy rejected") from None
    counts = {}
    for surface in ("chat", "work"):
        counts[surface] = {tier: sum(len(levels) for levels in value["entitlements"][surface][tier].values())
                           for tier in ("free", "plus", "pro")}
    return {
        "schema_version": 3,
        "status": "three_family_policy_valid_runtime_integration_pending",
        "family_slots": [item["model_slot"] for item in value["families"].values()],
        "chat_route_counts_by_tier": counts["chat"],
        "work_route_counts_by_tier": counts["work"],
        "nova_chat_selectable": False,
        "separate_chat_8b_slot": False,
        "normal_surface_lineage_exposure": False,
        "runtime_integration_required": True,
        "phase_b_ready": False,
        "all_execution_and_spending_gates_closed": True,
    }


def main():
    try:
        print(json.dumps(validate(), sort_keys=True))
        return 0
    except CurrentPolicyError:
        print("current product policy rejected", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
