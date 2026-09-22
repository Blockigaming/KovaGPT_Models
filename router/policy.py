"""Resolve public Kova modes to immutable server-controlled compute policies."""

from copy import deepcopy


FORBIDDEN_FIELDS = frozenset((
    "model", "provider", "engine", "reasoning_effort", "planning_passes",
    "answer_passes", "critic_passes", "verification_passes", "dynamic_agent_maximum",
    "behavior_contract_id", "behavior_instruction", "display_name", "family_display_name",
))

CHAT_POLICIES = {
    "instant": {
        "engine": "kova-core", "profile": "cosmo", "reasoning_effort": "low",
        "passes": (0, 1, 0, 0), "activity_updates": False, "thinking_enabled": False,
        "display_name": "Kova Cosmo",
        "behavior_contract_id": "chat_instant_direct",
        "behavior_instruction": "Answer directly and concisely. Do not add process narration or activity updates.",
    },
    "medium": {
        "engine": "kova-core", "profile": "orion", "reasoning_effort": "medium",
        "passes": (1, 1, 0, 1), "activity_updates": False, "thinking_enabled": True,
        "display_name": "Kova Orion",
        "behavior_contract_id": "chat_medium_balanced",
        "behavior_instruction": "Give a balanced answer, check material assumptions, and verify the result once.",
    },
    "high": {
        "engine": "kova-core", "profile": "orion", "reasoning_effort": "xhigh",
        "passes": (1, 1, 1, 1), "activity_updates": True, "thinking_enabled": True,
        "display_name": "Kova Orion — High",
        "behavior_contract_id": "chat_high_rigorous",
        "behavior_instruction": "Decompose difficult work, test the draft for concrete defects, and return a rigorous answer.",
    },
    "extra-high": {
        "engine": "kova-core", "profile": "orion", "reasoning_effort": "xhigh",
        "passes": (2, 2, 2, 2), "activity_updates": True, "thinking_enabled": True,
        "display_name": "Kova Orion — Extra High",
        "behavior_contract_id": "chat_extra_high_deep",
        "behavior_instruction": "Explore alternatives and edge cases across multiple private passes before the final answer.",
    },
    "max": {
        "engine": "kova-core", "profile": "orion", "reasoning_effort": "xhigh",
        "passes": (2, 3, 2, 3), "activity_updates": True, "thinking_enabled": True,
        "display_name": "Kova Orion — Max",
        "behavior_contract_id": "chat_max_single_engine",
        "behavior_instruction": "Use the maximum bounded single-engine analysis, critique, and verification policy.",
    },
    "ultra": {
        "engine": "kova-ultra", "profile": "orion", "reasoning_effort": "xhigh",
        "dynamic_agents": (2, 5), "judge": True, "synthesis": True, "activity_updates": True,
        "display_name": "Kova Orion — Ultra",
        "behavior_contract_id": "chat_ultra_multi_specialist",
        "behavior_instruction": "Use bounded specialists, evidence comparison, a judge, optional debate, and final Kova synthesis.",
    },
}

WORK_FAMILY_POLICIES = {
    "cosmo": {
        "family_display_name": "Kova Cosmo",
        "behavior_contract_id": "work_cosmo_action_first",
        "behavior_instruction": "Prioritize fast, concise, action-ready output and include only essential detail.",
        "answer_style": "concise_action_first",
        "tool_posture": "minimum_necessary",
    },
    "orion": {
        "family_display_name": "Kova Orion",
        "behavior_contract_id": "work_orion_balanced",
        "behavior_instruction": "Balance speed and depth, state material assumptions, and verify important details.",
        "answer_style": "balanced_explanatory",
        "tool_posture": "verify_when_material",
    },
    "nova": {
        "family_display_name": "Kova Nova",
        "behavior_contract_id": "work_nova_rigorous",
        "behavior_instruction": "Prioritize rigor: decompose the work, compare alternatives, test edge cases, and surface material risks.",
        "answer_style": "rigorous_comprehensive",
        "tool_posture": "evidence_first",
    },
}

WORK_FAMILIES = frozenset(WORK_FAMILY_POLICIES)
CHAT_FAMILIES = frozenset(("cosmo", "orion"))
WORK_EFFORTS = {
    "Light": {"reasoning_effort": "low", "passes": (0, 1, 0, 0), "maximum_output_tokens": 2048, "thinking_enabled": False},
    "Medium": {"reasoning_effort": "medium", "passes": (1, 1, 0, 1), "maximum_output_tokens": 4096, "thinking_enabled": True},
    "High": {"reasoning_effort": "xhigh", "passes": (1, 2, 1, 1), "maximum_output_tokens": 8192, "thinking_enabled": True},
    "Extra High": {"reasoning_effort": "xhigh", "passes": (2, 3, 2, 2), "maximum_output_tokens": 16384, "thinking_enabled": True},
    "Max": {"reasoning_effort": "xhigh", "passes": (2, 4, 2, 3), "maximum_output_tokens": 24576, "thinking_enabled": True},
    "Ultra": {"reasoning_effort": "xhigh", "dynamic_agents": (2, 5), "maximum_output_tokens": 32768, "thinking_enabled": True},
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _reject_policy_overrides(request):
    _require(isinstance(request, dict), "route request must be an object")
    overridden = sorted(FORBIDDEN_FIELDS.intersection(request))
    _require(not overridden, f"server-controlled routing fields: {','.join(overridden)}")


def resolve_route(request):
    """Resolve a Chat or Work route without accepting provider/model overrides."""
    _reject_policy_overrides(request)
    surface = request.get("surface")
    _require(surface in ("chat", "work"), "invalid surface")

    if surface == "chat":
        if set(request) == {"surface", "route_id"}:
            route_id = request.get("route_id")
            _require(route_id != "kova-auto", "auto requires server classifier context")
            _require(route_id in CHAT_POLICIES, "invalid chat compatibility route")
            return {"surface": "chat", "route_id": route_id,
                    "compatibility_alias": True, **deepcopy(CHAT_POLICIES[route_id])}
        _require(set(request) == {"surface", "family", "effort"},
                 "unsupported chat route fields")
        family, effort = request.get("family"), request.get("effort")
        _require(family in CHAT_FAMILIES, "invalid chat family")
        _require(effort in WORK_EFFORTS, "invalid chat effort")
        policy = deepcopy(WORK_EFFORTS[effort])
        policy.update(deepcopy(WORK_FAMILY_POLICIES[family]))
        policy.update({
            "surface": "chat",
            "route_id": f"chat:{family}:{effort.lower().replace(' ', '-')}",
            "profile": family,
            "engine": "kova-ultra" if effort == "Ultra" else "kova-core",
            "activity_updates": effort not in ("Light", "Medium"),
            "display_name": f"{policy['family_display_name']} — {effort}",
            "compatibility_alias": False,
        })
        if effort == "Ultra":
            policy.update({"judge": True, "synthesis": True})
        return policy

    _require(set(request) == {"surface", "family", "effort"}, "unsupported work route fields")
    family = request.get("family")
    effort = request.get("effort")
    _require(family in WORK_FAMILIES, "invalid work family")
    _require(effort in WORK_EFFORTS, "invalid work effort")
    policy = deepcopy(WORK_EFFORTS[effort])
    policy.update(deepcopy(WORK_FAMILY_POLICIES[family]))
    policy.update({
        "surface": "work",
        "route_id": f"work:{family}:{effort.lower().replace(' ', '-')}",
        "profile": family,
        "engine": "kova-ultra" if effort == "Ultra" else "kova-core",
        "activity_updates": True,
        "display_name": f"{policy['family_display_name']} — {effort}",
    })
    if effort == "Ultra":
        policy.update({"judge": True, "synthesis": True})
    return policy
