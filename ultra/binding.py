"""Bind Ultra artifacts and validate a grounded, machine-readable debate gate.

These checks validate references and schema, not the factual correctness of model
judgments. Only the already completed detector/judge artifacts may trigger debate.
"""

from copy import deepcopy
import json
import re

from core.current_candidates import CORE_SERVING
from core.identity import TRUSTED_SYSTEM_MESSAGE_COUNT, load_runtime_identity, selected_candidate_provenance
from execution.contracts import ExecutionError, require
from release.model_revisions import source_reference_for_route
from ultra.conversation import validated_conversation


DISAGREEMENT_INSTRUCTION = (
    ' Return only JSON: {"disagreements":[{"id":"d1","stage_ids":'
    '["specialist-1","specialist-2"],"summary":"concrete conflict"}]}. '
    'Use an empty disagreements array when none exist. Each entry must name '
    'at least two distinct completed specialist stage IDs from the supplied artifacts.'
)
JUDGE_INSTRUCTION = (
    ' Return only JSON: {"material_disagreement":true,"disagreement_ids":["d1"],'
    '"summary":"brief evidence judgment"}. Refer only to disagreement IDs actually '
    'recorded by disagreement-check. When no material disagreement warrants debate, '
    'return false with an empty disagreement_ids array. Do not invent evidence or IDs.'
)


def _load(value):
    require(isinstance(value, str) and len(value) <= 250_000, "invalid decision artifact")
    def unique(pairs):
        result = {}
        for key, item in pairs:
            require(key not in result, "duplicate decision field")
            result[key] = item
        return result
    def reject(_):
        raise ExecutionError("nonfinite decision value")
    try:
        return json.loads(value, object_pairs_hook=unique, parse_constant=reject)
    except (ValueError, RecursionError):
        raise ExecutionError("invalid structured Ultra decision") from None


def _summary(value):
    require(isinstance(value, str) and value.strip() and len(value) <= 8192,
            "invalid decision summary")


def validate_disagreements(content, specialist_ids):
    value = _load(content)
    require(isinstance(value, dict) and set(value) == {"disagreements"}, "invalid disagreement artifact")
    entries = value["disagreements"]
    require(isinstance(entries, list) and len(entries) <= 32, "invalid disagreement list")
    seen = set()
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == {"id", "stage_ids", "summary"},
                "invalid disagreement entry")
        key = entry["id"]
        require(isinstance(key, str) and re.fullmatch(r"d[1-9][0-9]{0,2}", key)
                and key not in seen, "invalid disagreement ID")
        refs = entry["stage_ids"]
        require(isinstance(refs, list) and all(isinstance(ref, str) for ref in refs)
                and 2 <= len(refs) <= 5 and len(set(refs)) == len(refs)
                and set(refs) <= set(specialist_ids), "disagreement cites unavailable specialists")
        _summary(entry["summary"])
        seen.add(key)
    return seen


def judge_requires_debate(content, detector_content, specialist_ids):
    recorded = validate_disagreements(detector_content, specialist_ids)
    value = _load(content)
    require(isinstance(value, dict)
            and set(value) == {"material_disagreement", "disagreement_ids", "summary"},
            "invalid judge artifact")
    require(type(value["material_disagreement"]) is bool, "judge decision must be boolean")
    refs = value["disagreement_ids"]
    require(isinstance(refs, list) and all(isinstance(ref, str) for ref in refs)
            and len(set(refs)) == len(refs) and set(refs) <= recorded,
            "judge cites unrecorded disagreement")
    require(bool(refs) == value["material_disagreement"], "judge decision and evidence disagree")
    _summary(value["summary"])
    return value["material_disagreement"]


def bind_ultra_operation(plan, stage_id, artifacts, *, runtime_identity, token_counter):
    """Bind exact dependency targets and recount the complete conversation."""
    require(isinstance(plan, dict) and plan.get("engine") == "kova-ultra", "Ultra plan required")
    matches = [op for op in plan["operations"] if op["id"] == stage_id]
    require(len(matches) == 1, "Ultra stage must exist exactly once")
    operation = matches[0]
    require(isinstance(artifacts, dict) and set(artifacts) == set(operation["depends_on"]),
            "Ultra artifacts differ from declared dependencies")
    template = deepcopy(operation["input_template"])
    messages = template["messages"]
    bindings = template["artifact_bindings"]
    require(len(bindings) == len(artifacts), "Ultra artifact binding count mismatch")
    source = source_reference_for_route(plan["route_id"])
    candidates = [candidate for candidate in CORE_SERVING["candidates"] if candidate["id"] == source.slot]
    require(len(candidates) == 1 and candidates[0]["model"] == source.slot and
            candidates[0]["revision"] == source.revision and
            isinstance(runtime_identity, dict) and
            runtime_identity.get("model") == source.slot and
            runtime_identity.get("model_revision") == source.revision,
            "Ultra loaded model differs from selected candidate provenance")
    for field in ("adapter_sha256", "adapter_bundle_sha256"):
        pinned = candidates[0][field]
        require(isinstance(pinned, str) and re.fullmatch(r"[0-9a-f]{64}", pinned) is not None and
                runtime_identity.get(field) == pinned,
                "Ultra loaded adapter differs from selected candidate provenance")
    require(isinstance(messages, list) and len(messages) >= TRUSTED_SYSTEM_MESSAGE_COUNT and
            messages[0] == {"role": "system", "content": load_runtime_identity()} and
            messages[1] == selected_candidate_provenance(plan["route_id"], source.slot),
            "Ultra trusted identity or provenance was changed")
    conversation = validated_conversation(plan["conversation_messages"]) if "conversation_messages" in plan else [
        {"role": "user", "content": plan["task"]},
    ]
    first_artifact = TRUSTED_SYSTEM_MESSAGE_COUNT + len(conversation)
    require(isinstance(messages, list) and len(messages) == first_artifact + len(bindings)
            and messages[TRUSTED_SYSTEM_MESSAGE_COUNT:first_artifact] == conversation,
            "Ultra conversation differs from its saved snapshot")
    sources, targets = set(), set()
    total_chars = 0
    for offset, binding in enumerate(bindings):
        source = binding["source_stage_id"]
        index = binding["target_message_index"]
        require(source in artifacts and source not in sources
                and source == operation["depends_on"][offset], "duplicate or undeclared artifact source")
        require(type(index) is int and index == first_artifact + offset
                and index < len(messages) and index not in targets,
                "invalid Ultra artifact target")
        require(binding["target_field"] == "content" and binding["replace_exact_target_only"] is True,
                "Ultra binding must target exact message content")
        target = messages[index]
        placeholder = binding["placeholder"]
        require(target["role"] == "assistant" and isinstance(target["content"], str)
                and placeholder == f"{{{{server_stage_output:{source}}}}}"
                and target["content"].count(placeholder) == 1, "invalid Ultra binding placeholder")
        output = artifacts[source]
        if output is None:
            require(stage_id == "synthesis" and source == "debate-round-1"
                    and binding["when_source_skipped"] == "bind_empty", "unexpected skipped artifact")
            output = ""
        else:
            require(isinstance(output, str) and output.strip() and len(output) <= 250_000,
                    "invalid Ultra stage artifact")
        total_chars += len(output)
        target["content"] = target["content"].replace(placeholder, output, 1)
        sources.add(source)
        targets.add(index)
    require(total_chars <= 750_000, "aggregate Ultra artifacts too large")
    require(all("{{server_stage_output:" not in msg["content"] for msg in messages),
            "unbound Ultra artifact placeholder")
    require(callable(token_counter), "trusted Ultra token counter missing")
    counted = token_counter(runtime_identity["model"], messages)
    require(type(counted) is int and counted > 0 and counted <= operation["maximum_input_tokens"],
            "bound Ultra input exceeds reservation")
    require(counted + operation["maximum_output_tokens"] <= runtime_identity["context_tokens"],
            "bound Ultra operation exceeds served context")
    request = {
        "model": runtime_identity["model"], "messages": messages, "reasoning_effort": "xhigh",
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
        "max_tokens": operation["maximum_output_tokens"], "stream": operation["public_output"],
    }
    if request["stream"]:
        request["stream_options"] = {"include_usage": True}
    return request
