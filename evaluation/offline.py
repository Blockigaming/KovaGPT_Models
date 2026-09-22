"""Run reproducible, provider-free checks for the 37 Kova route contracts."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from core.adapter import CANDIDATES, build_core_plan
from router.auto import classify_auto
from router.policy import resolve_route
from ultra.orchestrator import build_ultra_plan


ROOT = Path(__file__).resolve().parents[1]
SUITE = json.loads((ROOT / "evaluations" / "offline-suite.v1.json").read_text(encoding="utf-8"))
ACTIVITY_CONTRACT = json.loads((ROOT / "config" / "activity-event.v1.json").read_text(encoding="utf-8"))
CORE_CANDIDATE = "Qwen/Qwen3.8-27B"
URL_PATTERN = re.compile(r"https?://[^\s<>\]\)]+", re.IGNORECASE)
FORBIDDEN_RESPONSE_PATTERNS = (
    "foundation model trained from scratch",
    "cosmo, orion, and nova are separate foundation models",
    "<think",
    "</think>",
    "reasoning_content",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _token_count(messages):
    _require(isinstance(messages, list) and messages, "token counter requires messages")
    total = sum((len(message["content"]) + 3) // 4 for message in messages)
    return max(1, total)


def _core_token_count(_model, messages):
    return _token_count(messages)


def _route_contract(route_id):
    matches = [item for item in SUITE["route_contracts"] if item["route_id"] == route_id]
    _require(len(matches) == 1, f"missing or duplicate route contract:{route_id}")
    return matches[0]


def build_route_manifest():
    """Resolve the checked-in expectations through the actual server router."""
    contracts = SUITE["route_contracts"]
    _require(SUITE["target_routes"] == 37 and len(contracts) == 37, "offline suite must define exactly 37 routes")
    _require(len({item["route_id"] for item in contracts}) == 37, "offline route IDs must be unique")
    manifest = []
    for contract in contracts:
        selector = contract["selector"]
        if selector == {"surface": "auto"}:
            actual = {
                "route_id": "kova-auto",
                "display_name": "Kova Auto",
                "engine": "server-selected",
                "profile": "server-selected",
                "behavior_contract_id": "auto_server_classifier",
                "activity_updates": "route-dependent",
            }
        else:
            policy = resolve_route(selector)
            actual = {
                "route_id": policy["route_id"],
                "display_name": policy["display_name"],
                "engine": policy["engine"],
                "profile": policy["profile"],
                "behavior_contract_id": policy["behavior_contract_id"],
                "activity_updates": policy["activity_updates"],
            }
        _require(actual["route_id"] == contract["route_id"], f"route ID mismatch:{contract['route_id']}")
        for field, expected in contract["expected"].items():
            _require(actual[field] == expected, f"route contract mismatch:{contract['route_id']}:{field}")
        manifest.append({**actual, "selector": selector})
    return manifest


def _core_request(contract):
    selector = contract["selector"]
    base = {
        "request_id": f"offline-{contract['route_id']}",
        "messages": [{"role": "user", "content": "Produce a concise, accurate project assessment."}],
    }
    if selector["surface"] == "chat" and "route_id" in selector:
        return {**base, "route_id": selector["route_id"]}
    return {**base, **selector}


def _ultra_request(contract):
    selector = contract["selector"]
    base = {
        "request_id": f"offline-{contract['route_id']}",
        "task": "Research competitors, compare pricing and security, and create a project report.",
    }
    if selector["surface"] == "chat" and "route_id" in selector:
        return {**base, "route_id": selector["route_id"]}
    return {**base, **selector}


def _validate_core_plan(contract):
    plan = build_core_plan(
        _core_request(contract),
        candidate_model=CORE_CANDIDATE,
        token_counter=_core_token_count,
    )
    _require(plan["route_id"] == contract["route_id"], f"Core plan route mismatch:{contract['route_id']}")
    _require(plan["display_name"] == contract["expected"]["display_name"], "Core display name mismatch")
    _require(plan["behavior_contract_id"] == contract["expected"]["behavior_contract_id"], "Core behavior mismatch")
    _require(plan["production_ready"] is False, "offline Core plan claimed production readiness")
    operations = plan["operations"]
    _require(operations and sum(operation["public_response"] for operation in operations) == 1, "Core public stage invalid")
    _require(operations[-1]["public_response"] is True, "Core final stage must be public")
    prior_ids = []
    for operation in operations:
        _require(operation["depends_on_stage_ids"] == prior_ids, "Core stage dependencies are not ordered")
        bindings = operation["request_template"]["artifact_bindings"]
        _require([binding["source_stage_id"] for binding in bindings] == prior_ids, "Core artifact bindings mismatch")
        for binding in bindings:
            target = operation["request_template"]["messages"][binding["target_message_index"]][binding["target_field"]]
            _require(binding["replace_exact_target_only"] is True and binding["placeholder"] in target, "Core exact binding target invalid")
        _require(operation["request_template"]["reject_placeholder_outside_binding_targets"] is True, "Core binding scope missing")
        _require(operation["request_template"]["recount_bound_messages_with_trusted_tokenizer"] is True, "Core recount missing")
        _require(
            operation["maximum_input_tokens"] + operation["maximum_output_tokens"]
            <= CANDIDATES[CORE_CANDIDATE]["context_tokens"],
            "Core context reservation exceeded",
        )
        prior_ids.append(operation["stage_id"])
    return len(operations)


def _validate_ultra_plan(contract):
    admission = {
        "entitlement": "pro",
        "ultra_authorized": True,
        "remaining_usd": 2.0,
        "estimated_max_usd": 0.5,
        "max_agents": 5,
        "max_total_tokens": 32768,
    }
    plan = build_ultra_plan(
        _ultra_request(contract),
        admission=admission,
        token_counter=_token_count,
    )
    _require(plan["route_id"] == contract["route_id"], f"Ultra plan route mismatch:{contract['route_id']}")
    _require(plan["display_name"] == contract["expected"]["display_name"], "Ultra display name mismatch")
    _require(plan["behavior_contract_id"] == contract["expected"]["behavior_contract_id"], "Ultra behavior mismatch")
    _require(plan["production_ready"] is False and plan["model_selection_required"] is True, "Ultra must stay blocked")
    operation_ids = {operation["id"] for operation in plan["operations"]}
    _require(len(operation_ids) == len(plan["operations"]), "Ultra operation IDs must be unique")
    _require("disagreement-check" in operation_ids, "Ultra disagreement-check stage missing")
    by_id = {operation["id"]: operation for operation in plan["operations"]}
    _require("disagreement-check" in by_id["judge"]["depends_on"], "Ultra judge must follow disagreement check")
    for operation in plan["operations"]:
        _require(set(operation["depends_on"]).issubset(operation_ids), "Ultra dependency references missing operation")
        bindings = operation["input_template"]["artifact_bindings"]
        _require([binding["source_stage_id"] for binding in bindings] == operation["depends_on"], "Ultra bindings mismatch")
        for binding in bindings:
            target = operation["input_template"]["messages"][binding["target_message_index"]][binding["target_field"]]
            _require(binding["replace_exact_target_only"] is True and binding["placeholder"] in target, "Ultra exact binding target invalid")
        _require(operation["input_template"]["reject_placeholder_outside_binding_targets"] is True, "Ultra binding scope missing")
        _require(any(message["role"] == "user" and message["content"] == plan["task"] for message in operation["input_template"]["messages"]), "Ultra task missing")
    _require(plan["reserved_maximum_total_tokens"] <= plan["maximum_total_tokens"], "Ultra total token cap exceeded")
    _require(sum(operation["public_output"] for operation in plan["operations"]) == 1, "Ultra public output invalid")
    _require(plan["operations"][-1]["id"] == "synthesis" and plan["operations"][-1]["public_output"], "Ultra synthesis invalid")
    return len(plan["operations"])


# Evidence limits bound local validation work, not model compute or product timing.
MAX_EVIDENCE_ITEMS = 4096
STARTED_OPERATION_STATES = frozenset((
    "started", "running", "success", "failed", "cancelled", "expired", "interrupted", "uncertain",
))


def _evidence_text(value, maximum=8192):
    if type(value) is not str or not value.strip() or len(value) > maximum:
        return False
    try:
        value.encode("utf-8")
    except UnicodeError:
        return False
    return True


def _evidence_urls(value, label, violations):
    if type(value) is not list or len(value) > MAX_EVIDENCE_ITEMS:
        violations.append(f"invalid_{label}")
        return set()
    result = set()
    for url in value:
        if not _evidence_text(url):
            violations.append(f"invalid_{label}_entry")
        else:
            result.add(url)
    return result


def _evidence_index(rows, label, violations, *, require_tool=False):
    """Index unambiguous supplied records; never turn dropped failures into pass."""
    if type(rows) is not list or len(rows) > MAX_EVIDENCE_ITEMS:
        violations.append(f"invalid_{label}_collection")
        return {}
    result, seen = {}, set()
    for index, row in enumerate(rows, 1):
        if type(row) is not dict or not _evidence_text(row.get("operation_id")):
            violations.append(f"invalid_{label}_{index}")
            continue
        operation_id = row["operation_id"]
        if operation_id in seen:
            violations.append(f"duplicate_{label}_operation_id")
            result.pop(operation_id, None)
            continue
        seen.add(operation_id)
        if require_tool and not _evidence_text(row.get("tool")):
            violations.append(f"invalid_{label}_{index}_tool")
            continue
        result[operation_id] = row
    return result


def validate_response_artifact(artifact):
    """Check supplied output/receipt consistency, not factuality or provenance."""
    _require(type(artifact) is dict, "response artifact must be an object")
    text = artifact.get("text")
    _require(type(text) is str, "response text missing")
    violations = []
    try:
        text.encode("utf-8")
    except UnicodeError:
        return ["invalid_response_text"]
    if len(text) > 750000:
        return ["response_text_limit_exceeded"]
    lowered = text.lower()
    for flag in ("identity_requested", "provider_disclosure_requested"):
        if flag in artifact and type(artifact[flag]) is not bool:
            violations.append("invalid_" + flag)
    if artifact.get("identity_requested") is True and re.search(r"\bkova\b", text, re.IGNORECASE) is None:
        violations.append("kova_identity_missing")
    if artifact.get("provider_disclosure_requested") is True:
        provider = artifact.get("selected_provider")
        model = artifact.get("selected_upstream_model")
        if not _evidence_text(provider) or provider.lower() not in lowered:
            violations.append("selected_provider_missing")
        if not _evidence_text(model) or model.lower() not in lowered:
            violations.append("selected_upstream_model_missing")
    for pattern in FORBIDDEN_RESPONSE_PATTERNS:
        if pattern in lowered:
            violations.append(f"forbidden_response_pattern:{pattern}")
    for field in ("reasoning", "reasoning_content", "reasoning_details"):
        if artifact.get(field) not in (None, "", [], {}):
            violations.append(f"private_response_field:{field}")

    runtime_results = _evidence_index(artifact.get("runtime_tool_results", []), "runtime_tool",
                                      violations, require_tool=True)
    result_urls = {}
    successful_urls = _evidence_urls(artifact.get("user_source_urls", []), "user_source_urls", violations)
    for operation_id, result in runtime_results.items():
        urls = _evidence_urls(result.get("source_urls", []), "runtime_source_urls", violations)
        result_urls[operation_id] = urls
        if not _evidence_text(result.get("status")):
            violations.append("invalid_runtime_tool_status")
        elif result["status"] == "success":
            successful_urls.update(urls)
    text_urls = {match.group(0).rstrip(".,;:!?") for match in URL_PATTERN.finditer(text)}
    for url in sorted(text_urls - successful_urls):
        violations.append(f"ungrounded_source_url:{url}")

    claims = _evidence_index(artifact.get("tool_claims", []), "tool_claim", violations, require_tool=True)
    for operation_id, claim in claims.items():
        result = runtime_results.get(operation_id)
        if not result or result.get("status") != "success":
            violations.append(f"unverified_tool_claim:{operation_id}")
            continue
        if claim["tool"] != result["tool"]:
            violations.append(f"tool_claim_mismatch:{operation_id}")
        if "source_url" in claim:
            source_url = claim["source_url"]
            if not _evidence_text(source_url) or source_url not in result_urls[operation_id]:
                violations.append(f"tool_source_mismatch:{operation_id}")
    return violations


def _timestamp(value):
    _require(isinstance(value, str), "timestamp must be text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        _require(parsed.tzinfo is not None and parsed.utcoffset() is not None, "activity timestamp must include timezone")
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid activity timestamp") from error


def validate_activity_events(route_id, events, runtime_operations):
    """Require unambiguous, already-started supplied operations for each event.

    Receipt consistency does not authenticate the caller, actual tool execution,
    icon-map provenance or the truth of arbitrary summary prose.
    """
    contract = _route_contract(route_id)
    activity_allowed = contract["expected"]["activity_updates"]
    _require(type(events) is list and type(runtime_operations) is list, "activity inputs must be arrays")
    violations = []
    if len(events) > MAX_EVIDENCE_ITEMS or len(runtime_operations) > MAX_EVIDENCE_ITEMS:
        return ["activity_evidence_limit_exceeded"]
    if activity_allowed is False and events:
        violations.append("route_forbids_activity")
    runtime = _evidence_index(runtime_operations, "runtime_activity", violations)
    urls = {key: _evidence_urls(op.get("source_urls", []), "runtime_source_urls", violations)
            for key, op in runtime.items()}
    required = set(ACTIVITY_CONTRACT["required_fields"])
    allowed = required | set(ACTIVITY_CONTRACT["optional_grounding_fields"])
    seen_event_ids = set()
    previous_time = None
    request_id = None
    for index, event in enumerate(events, start=1):
        if type(event) is not dict:
            violations.append(f"event_{index}_not_object")
            continue
        missing = required - set(event)
        extra = set(event) - allowed
        if missing:
            violations.append(f"event_{index}_missing:{','.join(sorted(missing))}")
        if extra:
            violations.append(f"event_{index}_unsupported_fields")
        if missing:
            continue
        event_id = event["event_id"]
        if not _evidence_text(event_id) or event_id in seen_event_ids:
            violations.append(f"event_{index}_invalid_event_id")
        else:
            seen_event_ids.add(event_id)
        if type(event["sequence"]) is not int or event["sequence"] != index:
            violations.append(f"event_{index}_invalid_sequence")
        if not _evidence_text(event["phase"]) or event["phase"] not in ACTIVITY_CONTRACT["allowed_phases"]:
            violations.append(f"event_{index}_invalid_phase")
        valid_fields = True
        for field in ("request_id", "title", "summary", "grounding_operation_id"):
            if not _evidence_text(event[field]):
                violations.append(f"event_{index}_invalid_{field}")
                valid_fields = False
        if not valid_fields:
            continue  # In particular, never use malformed grounding IDs as dict keys.
        if request_id is None:
            request_id = event["request_id"]
        elif event["request_id"] != request_id:
            violations.append(f"event_{index}_request_stream_mismatch")
        try:
            occurred_at = _timestamp(event["occurred_at"])
        except ValueError:
            violations.append(f"event_{index}_invalid_occurred_at")
            continue
        if previous_time and occurred_at < previous_time:
            violations.append(f"event_{index}_time_regression")
        previous_time = occurred_at
        operation = runtime.get(event["grounding_operation_id"])
        if not operation:
            violations.append(f"event_{index}_missing_runtime_operation")
            continue
        if operation.get("request_id") != event["request_id"]:
            violations.append(f"event_{index}_request_mismatch")
        if operation.get("phase") != event["phase"]:
            violations.append(f"event_{index}_phase_mismatch")
        status = operation.get("status")
        if type(status) is not str or status not in STARTED_OPERATION_STATES:
            violations.append(f"event_{index}_runtime_not_started")
        try:
            started_at = _timestamp(operation.get("started_at"))
        except ValueError:
            violations.append(f"event_{index}_invalid_runtime_start")
            continue
        if occurred_at < started_at:
            violations.append(f"event_{index}_precedes_runtime_start")
        for field in ("tool", "domain", "content_type", "icon_key"):
            if field in event and (not _evidence_text(event[field]) or event[field] != operation.get(field)):
                violations.append(f"event_{index}_{field}_mismatch")
        if "result_count" in event:
            count, observed = event["result_count"], operation.get("result_count")
            if (type(count) is not int or not 0 <= count <= 2**53 - 1
                    or type(observed) is not int or count != observed):
                violations.append(f"event_{index}_result_count_mismatch")
        if "source_url" in event:
            source_url = event["source_url"]
            if (not _evidence_text(source_url) or status != "success"
                    or source_url not in urls[event["grounding_operation_id"]]):
                violations.append(f"event_{index}_ungrounded_source")
    return violations


def run_offline_suite():
    manifest = build_route_manifest()
    engine_counts = {
        "auto": sum(item["engine"] == "server-selected" for item in manifest),
        "core": sum(item["engine"] == "kova-core" for item in manifest),
        "ultra": sum(item["engine"] == "kova-ultra" for item in manifest),
    }
    _require(engine_counts == {"auto": 1, "core": 30, "ultra": 6}, "unexpected 37-route engine split")

    operation_counts = {}
    for contract in SUITE["route_contracts"]:
        engine = contract["expected"]["engine"]
        if engine == "kova-core":
            operation_counts[contract["route_id"]] = _validate_core_plan(contract)
        elif engine == "kova-ultra":
            operation_counts[contract["route_id"]] = _validate_ultra_plan(contract)

    family_behavior_ids = {
        item["profile"]: item["behavior_contract_id"]
        for item in manifest
        if item["route_id"].startswith("work:")
    }
    _require(len(family_behavior_ids) == 3 and len(set(family_behavior_ids.values())) == 3, "Work families are name-only")

    auto_results = []
    for case in SUITE["auto_cases"]:
        result = classify_auto(case["prompt"], entitlement=case["entitlement"], budget=case["budget"])
        passed = result["route_id"] == case["expected_route_id"]
        _require(passed, f"Auto case failed:{case['id']}")
        auto_results.append({"id": case["id"], "route_id": result["route_id"], "passed": True})

    response_results = []
    for case in SUITE["response_cases"]:
        violations = validate_response_artifact(case["artifact"])
        actual_pass = not violations
        _require(actual_pass == case["expect_pass"], f"response evaluator expectation mismatch:{case['id']}")
        response_results.append({
            "id": case["id"],
            "expected_pass": case["expect_pass"],
            "evaluator_matched_expectation": True,
            "violations": violations,
        })

    activity_results = []
    for case in SUITE["activity_cases"]:
        violations = validate_activity_events(case["route_id"], case["events"], case["runtime_operations"])
        actual_pass = not violations
        _require(actual_pass == case["expect_pass"], f"activity evaluator expectation mismatch:{case['id']}")
        activity_results.append({
            "id": case["id"],
            "expected_pass": case["expect_pass"],
            "evaluator_matched_expectation": True,
            "violations": violations,
        })

    evidence = SUITE["release_evidence"]
    _require(evidence["actual_model_outputs_evaluated"] is False, "offline suite cannot claim model evaluation")
    _require(evidence["quality_or_factuality_claimed"] is False, "offline suite cannot claim quality")
    _require(evidence["paid_provider_calls"] == 0, "offline suite must make no paid calls")
    _require(evidence["passing_release_routes"] == [], "offline suite cannot release routes")
    return {
        "schema_version": 1,
        "status": "offline_contracts_passed_release_still_blocked",
        "route_contracts_checked": len(manifest),
        "engine_counts": engine_counts,
        "planned_route_operation_counts": operation_counts,
        "auto_cases_checked": len(auto_results),
        "response_evaluator_cases_checked": len(response_results),
        "activity_evaluator_cases_checked": len(activity_results),
        "actual_model_outputs_evaluated": False,
        "quality_or_factuality_claimed": False,
        "paid_provider_calls": 0,
        "passing_release_routes": [],
    }


if __name__ == "__main__":
    print(json.dumps(run_offline_suite(), indent=2, sort_keys=True))
