"""Deterministic, server-authoritative baseline for Kova Auto."""

import math
import re
from copy import deepcopy

from router.policy import CHAT_POLICIES


SIMPLE_PATTERNS = (
    re.compile(r"^\s*(what(?:'s| is)|who|when|where)\b", re.IGNORECASE),
    re.compile(r"^\s*(rewrite|rephrase|summarize|list|give me)\b", re.IGNORECASE),
    re.compile(r"^\s*\d+(?:\s*[+\-*/×÷]\s*\d+)+\s*\??\s*$"),
)
CURRENT_OR_TOOL_TERMS = frozenset((
    "latest", "current", "today", "news", "weather", "price", "schedule", "search",
    "browse", "email", "calendar", "drive", "document", "website",
))
ORION_TERMS = frozenset((
    "debug", "error", "explain", "review", "fix", "compare", "code", "react",
))
NOVA_TERMS = frozenset((
    "analyze", "architecture", "race", "security", "audit", "research", "strategy",
    "proof", "optimize", "migration", "financial", "business", "evaluate",
))
ULTRA_TERMS = frozenset((
    "competitors", "multi-domain", "comprehensive", "launch strategy", "full report",
    "multiple specialists", "parallel agents", "cross-functional",
))
ENTITLEMENTS = frozenset(("free", "plus", "pro"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tokens(text):
    return re.findall(r"[a-z0-9'-]+", text.lower())


def _contains(text, phrases):
    lowered = text.lower()
    return {
        phrase for phrase in phrases
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", lowered)
    }


def _validate_budget(budget):
    _require(isinstance(budget, dict), "trusted budget context missing")
    _require(set(budget) == {"ultra_authorized", "remaining_usd", "estimated_ultra_usd"}, "invalid budget context")
    _require(isinstance(budget["ultra_authorized"], bool), "invalid ultra_authorized")
    for field in ("remaining_usd", "estimated_ultra_usd"):
        value = budget[field]
        _require(isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 and
                 (not isinstance(value, float) or math.isfinite(value)), f"invalid {field}")
    _require(budget["estimated_ultra_usd"] > 0, "invalid estimated_ultra_usd")


def classify_auto(prompt, *, entitlement, budget):
    """Return a route plus auditable feature IDs, never private reasoning."""
    _require(isinstance(prompt, str) and prompt.strip(), "prompt must be nonempty text")
    _require(len(prompt) <= 250_000, "prompt too large")
    _require(isinstance(entitlement, str) and entitlement in ENTITLEMENTS, "invalid server entitlement")
    _validate_budget(budget)
    words = _tokens(prompt)
    current_hits = _contains(prompt, CURRENT_OR_TOOL_TERMS)
    orion_hits = _contains(prompt, ORION_TERMS)
    nova_hits = _contains(prompt, NOVA_TERMS)
    ultra_hits = _contains(prompt, ULTRA_TERMS)
    feature_ids = []

    if entitlement == "free":
        route_id = "instant"
        feature_ids.append("free_plan_instant_cap")
    else:
        complex_score = min(len(nova_hits), 2) + (1 if len(words) >= 60 else 0) + (1 if len(words) >= 140 else 0)
        ultra_candidate = len(ultra_hits) >= 2 or (len(ultra_hits) >= 1 and complex_score >= 3)
        if ultra_candidate:
            feature_ids.append("multi_domain_ultra_candidate")
            ultra_allowed = (
                entitlement == "pro" and budget["ultra_authorized"] and
                budget["remaining_usd"] >= budget["estimated_ultra_usd"]
            )
            route_id = "ultra" if ultra_allowed else "max"
            feature_ids.append("ultra_budget_admitted" if ultra_allowed else "ultra_budget_or_entitlement_fallback")
        elif complex_score >= 4:
            route_id = "max"
            feature_ids.append("maximum_single_engine_complexity")
        elif complex_score >= 3:
            route_id = "extra-high"
            feature_ids.append("deep_single_engine_complexity")
        elif complex_score >= 1:
            route_id = "high"
            feature_ids.append("reasoning_complexity")
        elif current_hits or orion_hits:
            route_id = "medium"
            feature_ids.append("tool_or_balanced_reasoning")
        elif len(words) <= 30 and any(pattern.search(prompt) for pattern in SIMPLE_PATTERNS):
            route_id = "instant"
            feature_ids.append("simple_direct_request")
        else:
            route_id = "medium"
            feature_ids.append("balanced_default")

    # Apply plan limits after every complexity/budget branch, including Ultra fallback.
    # The calling application must supply the authenticated server entitlement.
    if entitlement == "plus" and route_id not in ("instant", "medium", "high"):
        route_id = "high"
        feature_ids.append("plus_plan_high_cap")

    policy = deepcopy(CHAT_POLICIES[route_id])
    return {
        "route_id": route_id,
        "engine": policy["engine"],
        "profile": policy["profile"],
        "feature_ids": feature_ids,
        "activity_updates": policy["activity_updates"],
    }
