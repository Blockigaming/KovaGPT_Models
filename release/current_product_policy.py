"""Validate the owner-approved three-family product policy without enabling it."""
import hashlib
import json
from pathlib import Path
import sys

from training.three_family_contract import ContractError, load_json

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config/current-product-policy.v3.json"
MAX_POLICY_BYTES = 64 * 1024
CurrentPolicyError = ContractError
# Canonical SHA-256 of the separately reviewed v3 policy. The expected value is
# intentionally independent of POLICY_PATH so changing that file alone cannot
# redefine the accepted policy at process start.
CANONICAL_POLICY_SHA256 = "39bc4ac719eb8939ad3680314e4e55f891771693c7973732196f683059ed098a"


def _canonical_digest(value) -> str:
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise CurrentPolicyError("current product policy rejected") from None
    return hashlib.sha256(encoded).hexdigest()


def load_policy():
    try:
        value = load_json(POLICY_PATH, maximum_bytes=MAX_POLICY_BYTES)
        if _canonical_digest(value) != CANONICAL_POLICY_SHA256:
            raise CurrentPolicyError("current product policy rejected")
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
