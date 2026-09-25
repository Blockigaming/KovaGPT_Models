"""Versioned KovaGPT selection bridge, never a legacy-cache rewrite.

Only server-authenticated callers may supply ExecutionGrant, Auto eligibility and
budget. The payload carries a selection, not execution permissions or provider
settings. Existing unversioned app/history aliases remain owned by the legacy app.
Importing or resolving a selection invokes no model, tool, database or network.
"""

from dataclasses import dataclass

from execution.contracts import ExecutionBlocked, ExecutionError, ExecutionGrant, require
from router.auto import classify_auto
from router.policy import CHAT_FAMILIES, CHAT_POLICIES, WORK_EFFORTS, WORK_FAMILIES, resolve_route


SELECTION_SCHEMA = "kova-models.v2"
AUTO_ALIASES = frozenset(("auto", "kova-auto"))
WORK_EFFORT_ALIASES = {effort: effort for effort in WORK_EFFORTS}
WORK_EFFORT_ALIASES.update({effort.lower().replace(" ", "-"): effort for effort in WORK_EFFORTS})
WORK_EFFORT_ALIASES["extra_high"] = "Extra High"


class LegacySelectionRequired(ExecutionError):
    """A legacy request must remain on its existing application path."""


@dataclass(frozen=True)
class ResolvedSelection:
    """Safe metadata only; obtain fresh server policy via policy() when planning."""

    route_id: str
    surface: str
    application_mode_id: str | None
    selected_by_auto: bool
    feature_ids: tuple[str, ...]

    def policy(self):
        if self.surface == "chat":
            if self.route_id.startswith("chat:"):
                _, family, effort = self.route_id.split(":")
                return resolve_route({"surface": "chat", "family": family,
                                      "effort": WORK_EFFORT_ALIASES[effort]})
            return resolve_route({"surface": "chat", "route_id": self.route_id})
        _, family, effort = self.route_id.split(":")
        return resolve_route({"surface": "work", "family": family,
                              "effort": WORK_EFFORT_ALIASES[effort]})

    def metadata(self):
        policy = self.policy()
        return {"schema_version": SELECTION_SCHEMA, "route_id": self.route_id,
                "surface": self.surface, "application_mode_id": self.application_mode_id,
                "display_name": policy["display_name"], "engine": policy["engine"],
                "selected_by_auto": self.selected_by_auto, "feature_ids": list(self.feature_ids),
                "production_routing_enabled": False}


def resolve_application_selection(value, *, grant, prompt=None, auto_budget=None, auto_enabled=False):
    """Resolve a new versioned selection under current explicit server permissions.

    Auto is a router, and its output must pass
    both the tier cap and exact route allowlist. An allowed route does NOT enable
    execution, GPU capacity or production routing. Existing execution and Azure
    runtime guards still apply independently.
    """
    require(type(grant) is ExecutionGrant, "current authenticated server grant required")
    if not grant.execution_authorized:
        raise ExecutionBlocked("model selection remains disabled")
    require(type(auto_enabled) is bool, "server Auto eligibility must be boolean")
    require(isinstance(value, dict), "selection must be an object")
    if "schema_version" not in value:
        raise LegacySelectionRequired("unversioned selections must use the existing legacy handler")
    require(value["schema_version"] == SELECTION_SCHEMA, "unsupported selection schema")
    surface = value.get("surface")
    require(isinstance(surface, str) and surface in ("chat", "work"), "invalid selection surface")
    if surface == "chat":
        if set(value) == {"schema_version", "surface", "family", "effort"}:
            family, raw_effort = value["family"], value["effort"]
            require(isinstance(family, str) and family in CHAT_FAMILIES, "invalid Chat family")
            require(isinstance(raw_effort, str) and raw_effort in WORK_EFFORT_ALIASES,
                    "invalid Chat effort")
            effort = WORK_EFFORT_ALIASES[raw_effort]
            policy = resolve_route({"surface": "chat", "family": family, "effort": effort})
            grant.authorize(grant.owner_id, policy["route_id"])
            return ResolvedSelection(policy["route_id"], "chat", None, False, ())
        require(set(value) == {"schema_version", "surface", "mode_id"},
                "unsupported selection fields")
        mode = value["mode_id"]
        require(isinstance(mode, str), "invalid Chat mode")
        require(mode in AUTO_ALIASES, "migration Chat aliases are not v2 selections")
        if not auto_enabled:
            raise ExecutionBlocked("Auto is not enabled by the server for this request")
        route = classify_auto(prompt, entitlement=grant.tier, budget=auto_budget)
        alias = route["route_id"]
        family = CHAT_POLICIES[alias]["profile"]
        effort = "light" if alias == "instant" else alias
        route_id = f"chat:{family}:{effort}"
        grant.authorize(grant.owner_id, route_id)
        return ResolvedSelection(route_id, "chat", None, True, tuple(route["feature_ids"]))
    require(set(value) == {"schema_version", "surface", "family", "effort"}, "unsupported selection fields")
    family, raw_effort = value["family"], value["effort"]
    require(isinstance(family, str) and family in WORK_FAMILIES, "invalid Work family")
    require(isinstance(raw_effort, str) and raw_effort in WORK_EFFORT_ALIASES, "invalid Work effort")
    effort = WORK_EFFORT_ALIASES[raw_effort]
    policy = resolve_route({"surface": "work", "family": family, "effort": effort})
    grant.authorize(grant.owner_id, policy["route_id"])
    return ResolvedSelection(policy["route_id"], "work", None, False, ())
