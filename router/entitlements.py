"""Server-authoritative entitlements for the three shared Kova families."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config/current-product-policy.v3.json"
TIERS = ("free", "plus", "pro")
FAMILIES = ("cosmo", "orion", "nova")
EFFORT_IDS = ("light", "medium", "high", "extra-high", "max", "ultra")
ALL_WORK_ROUTES = frozenset(
    f"work:{family}:{effort}" for family in FAMILIES for effort in EFFORT_IDS
)
PLUS_WORK_ROUTES = ALL_WORK_ROUTES
WORK_ALLOWED_BY_TIER = {
    "free": frozenset(), "plus": ALL_WORK_ROUTES, "pro": ALL_WORK_ROUTES,
}
CHAT_ALLOWED_BY_TIER = {
    "free": frozenset(("chat:cosmo:light",)),
    "plus": frozenset(f"chat:{family}:{effort}" for family in ("cosmo", "orion")
                      for effort in ("light", "medium", "high")),
    "pro": frozenset(f"chat:{family}:{effort}" for family in ("cosmo", "orion")
                     for effort in EFFORT_IDS),
}
# Internal migration aliases only; they are not current customer model choices.
COMPAT_CHAT_ALLOWED_BY_TIER = {
    "free": frozenset(("instant",)),
    "plus": frozenset(("instant", "medium", "high")),
    "pro": frozenset(("instant", "medium", "high", "extra-high", "max", "ultra")),
}
FREE_THINKING_MODE_ID = "__superseded_free_thinking__"
FREE_THINKING_ROUTE = "instant"


def _load():
    try:
        value = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise RuntimeError("invalid approved product policy") from None
    if value.get("schema_version") != 3 or value.get("status") != "owner_approved_three_family_source_policy":
        raise RuntimeError("invalid approved product policy")
    return value


PRODUCT_POLICY = _load()
