"""Build bounded, owner-scoped Kova personalization context without inference.

This source contract does not read databases, invoke tools, select a model, widen
an entitlement, or persist a correction. Callers must supply already-authorized
records from trusted application services.
"""

from copy import deepcopy
import json
from pathlib import Path

from execution.contracts import ExecutionError, require


ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / "config/personalization-context.v1.json").read_text(encoding="utf-8"))
SECTIONS = tuple(POLICY["context_order"])
LIMITS = POLICY["limits"]
FORBIDDEN = frozenset((
    "tier", "plan", "model", "model_slot", "provider", "engine", "effort",
    "reasoning_effort", "allowed_routes", "execution_authorized", "system_prompt",
))


def _text(value, label):
    require(type(value) is str and value.strip() == value and 0 < len(value) <= LIMITS["maximum_item_characters"],
            f"invalid {label}")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ExecutionError(f"invalid {label}") from error
    return value


def _digest(value, label):
    _text(value, label)
    require(len(value) == 64 and value == value.lower()
            and all(character in "0123456789abcdef" for character in value),
            f"invalid {label}")
    return value


def _all_string_characters(value):
    if type(value) is str:
        _text(value, "personalization string")
        return len(value)
    if type(value) is list:
        return sum(_all_string_characters(item) for item in value)
    if type(value) is dict:
        return sum(_all_string_characters(item) for item in value.values())
    return 0


def _items(items, *, owner_id, active_conversation_id, section):
    require(type(items) is list and len(items) <= LIMITS["maximum_items_per_section"],
            f"invalid {section}")
    result = []
    for item in items:
        require(type(item) is dict and not FORBIDDEN.intersection(item),
                f"invalid {section} item")
        expected = {"id", "owner_id", "text", "authorized"}
        if section == "saved_memory":
            expected.add("approved")
        if section == "task_context":
            expected.update({"project_id", "source_type", "source_id",
                             "source_revision", "source_digest_sha256"})
            if item.get("source_type") == "tool_result":
                expected.update({"tool_name", "tool_call_id"})
        if section in {"current_conversation", "session_corrections"}:
            expected.add("conversation_id")
        require(set(item) == expected, f"invalid {section} item")
        require(item["owner_id"] == owner_id, f"cross-owner {section} rejected")
        require(item["authorized"] is True, f"unauthorized {section} rejected")
        _text(item["id"], f"{section} id")
        _text(item["text"], f"{section} text")
        if section == "saved_memory":
            require(item["approved"] is True, "unapproved memory rejected")
        if section == "task_context":
            _text(item["project_id"], "project id")
            require(item["source_type"] in {"project_file", "tool_result"},
                    "invalid task context source type")
            _text(item["source_id"], "task context source id")
            _text(item["source_revision"], "task context source revision")
            _digest(item["source_digest_sha256"], "task context source digest")
            if item["source_type"] == "tool_result":
                _text(item["tool_name"], "task context tool name")
                _text(item["tool_call_id"], "task context tool call id")
        if section in {"current_conversation", "session_corrections"}:
            require(item["conversation_id"] == active_conversation_id,
                    f"cross-conversation {section} rejected")
            _text(item["conversation_id"], f"{section} conversation id")
        result.append(deepcopy({key: item[key] for key in expected if key != "owner_id"}))
    return result


def build_personalization_context(value, *, authenticated_owner_id, active_conversation_id):
    """Validate trusted context and return public-safe routing-independent input."""
    require(type(authenticated_owner_id) is str and authenticated_owner_id,
            "authenticated owner required")
    _text(authenticated_owner_id, "authenticated owner")
    _text(active_conversation_id, "active conversation")
    require(type(value) is dict and not FORBIDDEN.intersection(value),
            "invalid personalization context")
    require(set(value) == {"schema_version", "owner_id", "conversation_id", *SECTIONS},
            "invalid personalization context fields")
    require(value["schema_version"] == "kova-personalization.v1",
            "unsupported personalization schema")
    require(value["owner_id"] == authenticated_owner_id, "owner mismatch")
    require(value["conversation_id"] == active_conversation_id,
            "conversation mismatch")
    _text(value["owner_id"], "owner id")
    _text(value["conversation_id"], "conversation id")
    require(_all_string_characters(value) <= LIMITS["maximum_total_characters"],
            "personalization context too large")

    result = {"schema_version": value["schema_version"], "owner_id": authenticated_owner_id,
              "conversation_id": active_conversation_id}
    for section in SECTIONS:
        result[section] = _items(value[section], owner_id=authenticated_owner_id,
                                 active_conversation_id=active_conversation_id,
                                 section=section)
    result.update({
        "changes_model_weights": False,
        "may_expand_entitlements": False,
        "runtime_integrated": False,
        "production_ready": False,
    })
    return result


def render_personalization_instructions(context, *, authenticated_owner_id,
                                         active_conversation_id):
    """Render bounded Kova context; never include routing or provider controls."""
    validated = build_personalization_context(
        context,
        authenticated_owner_id=authenticated_owner_id,
        active_conversation_id=active_conversation_id,
    )
    lines = ["Use the following authorized user context only when relevant. Later session corrections override earlier style preferences, but never safety or server policy."]
    labels = {
        "current_conversation": "Current conversation",
        "saved_memory": "Approved memory",
        "custom_instructions": "Custom instructions",
        "task_context": "Authorized task context",
        "session_corrections": "Session corrections",
    }
    for section in SECTIONS:
        if validated[section]:
            lines.append(f"{labels[section]}:")
            for item in validated[section]:
                if section != "task_context":
                    lines.append(f"- {item['text']}")
                    continue
                source = (f"{item['source_type']}:{item['source_id']}"
                          f"@{item['source_revision']}"
                          f" sha256:{item['source_digest_sha256']}")
                if item["source_type"] == "tool_result":
                    source += f" tool:{item['tool_name']} call:{item['tool_call_id']}"
                lines.append(f"- [{source}] {item['text']}")
    return "\n".join(lines)
