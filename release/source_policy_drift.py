"""Fail-closed cross-file checks for the authoritative three-family policy."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED = (
    "model-catalog.v1.json", "product-surface.v1.json", "route-policy.v1.json",
    "application-bridge.v1.json", "core-serving.v1.json",
)
PUBLIC_SOURCE = (
    "router/policy.py", "router/entitlements.py", "router/application.py",
)
FORBIDDEN_PUBLIC = ("Qwen/", "Qwen3-", "qwen3-", "Kova 5.6", "chat-shared")


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate_key:{key}")
        value[key] = item
    return value


def _load(path: Path) -> dict:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"bom_forbidden:{path.name}")
    return json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_unique)


def validate(root: Path = ROOT) -> dict:
    policy = _load(root / "config/current-product-policy.v3.json")
    expected = {"free": (1, 0), "plus": (6, 18), "pro": (12, 18)}
    if any(policy["entitlements"]["chat"][tier]["nova"] for tier in expected):
        raise ValueError("nova_must_be_work_only")
    for tier, (chat, work) in expected.items():
        actual_chat = sum(len(levels) for levels in policy["entitlements"]["chat"][tier].values())
        actual_work = sum(len(levels) for levels in policy["entitlements"]["work"][tier].values())
        if actual_chat != chat or actual_work != work:
            raise ValueError(f"entitlement_drift:{tier}")
    if policy["processing_levels_are_separate_models"] is not False:
        raise ValueError("runtime_profiles_must_not_be_models")
    for name in ARCHIVED:
        value = _load(root / "config" / name)
        if value.get("status") != "superseded_non_authoritative_history":
            raise ValueError(f"legacy_file_not_superseded:{name}")
        if value.get("must_not_drive_current_routing") is not True:
            raise ValueError(f"legacy_file_can_drive_routing:{name}")
    for relative in PUBLIC_SOURCE:
        text = (root / relative).read_text("utf-8", errors="strict")
        for token in FORBIDDEN_PUBLIC:
            if token in text:
                raise ValueError(f"private_upstream_leak:{Path(relative).name}:{token}")
    gate_names = (
        "resource_creation_authorized", "spending_authorized", "model_download_authorized",
        "training_authorized", "deployment_authorized", "production_routing_enabled",
    )
    if any(policy[name] is not False for name in gate_names):
        raise ValueError("current_policy_gate_promoted")
    return {
        "status": "authoritative_three_family_policy_no_drift",
        "chat_routes": {tier: counts[0] for tier, counts in expected.items()},
        "work_routes": {tier: counts[1] for tier, counts in expected.items()},
        "provider_calls_made": 0,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
