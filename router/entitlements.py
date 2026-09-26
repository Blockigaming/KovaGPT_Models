"""Server-authoritative entitlements from the pinned three-family policy."""
from pathlib import Path

from release.current_product_policy import load_policy


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config/current-product-policy.v3.json"
TIERS = ("free", "plus", "pro")
FAMILIES = ("cosmo", "orion", "nova")
EFFORT_IDS = ("light", "medium", "high", "extra-high", "max", "ultra")
PRODUCT_POLICY = load_policy(POLICY_PATH)


def _routes(surface, tier):
    return frozenset(
        f"{surface}:{family}:{effort}"
        for family, levels in PRODUCT_POLICY["entitlements"][surface][tier].items()
        for effort in levels
    )


ALL_WORK_ROUTES = frozenset(
    f"work:{family}:{effort}" for family in FAMILIES for effort in EFFORT_IDS
)
WORK_ALLOWED_BY_TIER = {tier: _routes("work", tier) for tier in TIERS}
CHAT_ALLOWED_BY_TIER = {tier: _routes("chat", tier) for tier in TIERS}
PLUS_WORK_ROUTES = WORK_ALLOWED_BY_TIER["plus"]
# Internal migration aliases only; they are not current customer model choices.
COMPAT_CHAT_ALLOWED_BY_TIER = {
    "free": frozenset(("instant",)),
    "plus": frozenset(("instant", "medium", "high")),
    "pro": frozenset(("instant", "medium", "high", "extra-high", "max", "ultra")),
}
FREE_THINKING_MODE_ID = "__superseded_free_thinking__"
FREE_THINKING_ROUTE = "instant"
