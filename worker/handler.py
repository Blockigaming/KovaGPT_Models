"""Fail-closed Kova Core benchmark worker. Importing it starts no paid work."""

import json
import math
from pathlib import Path
from time import perf_counter_ns
from uuid import UUID, uuid4

from core.adapter import bind_core_operation, build_core_plan
from core.current_candidates import CORE_SERVING
from router.policy import CHAT_POLICIES, WORK_EFFORTS, WORK_FAMILY_POLICIES, resolve_route


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_EFFORTS = frozenset(("low", "medium", "xhigh"))
ALLOWED_SERVING_ENGINES = frozenset(("vllm", "sglang"))
ALLOWED_ENDPOINT_TYPES = frozenset(("queue_based", "load_balancing"))
MAX_OUTPUT_TOKENS = 32768
MAX_MESSAGES = 256
MAX_MESSAGE_TEXT_CHARS = 250_000
MAX_TOTAL_TEXT_CHARS = 750_000
MAX_TOOL_CALLS = 32
MAX_TOOL_ARGUMENT_CHARS = 250_000
MAX_TOOL_JSON_DEPTH = 64
MAX_TOOL_JSON_NODES = 50_000
TRUSTED_SYSTEM_IDENTITY = json.loads(
    (ROOT / "config" / "identity.v1.json").read_text(encoding="utf-8")
)["system_identity"]
PINNED_CORE_CANDIDATES = {
    candidate["id"]: {
        "id": candidate["id"],
        "model": candidate["model"],
        "model_revision": candidate["revision"],
        "adapter_sha256": candidate["adapter_sha256"],
        "adapter_bundle_sha256": candidate["adapter_bundle_sha256"],
    }
    for candidate in CORE_SERVING["candidates"]
}
RUNTIME_NUMERIC_FIELDS = (
    "worker_start_ms", "model_load_ms", "queue_ms", "gpu_rate_per_second_usd",
)
RUNTIME_IDENTITY_FIELDS = (
    "source", "worker_lifecycle_id", "loaded_model", "loaded_model_revision", "loaded_adapter_sha256",
    "loaded_adapter_bundle_sha256", "cold_start",
    "worker_start_ms", "model_load_ms", "queue_ms", "gpu_rate_per_second_usd", "gpu_type_id",
    "gpu_count", "serving_engine", "endpoint_type", "container_image_digest",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _selected_candidate(candidate_id):
    _require(isinstance(candidate_id, str) and candidate_id, "trusted benchmark_candidate_id missing")
    _require(
        len(PINNED_CORE_CANDIDATES) == len(CORE_SERVING["candidates"]),
        "pinned Core candidate IDs must be unique",
    )
    candidate = PINNED_CORE_CANDIDATES.get(candidate_id)
    _require(candidate is not None, "benchmark candidate is not in the pinned Core allowlist")
    return dict(candidate)


def _stage_ids(policy):
    stages = []
    for phase, count in zip(("planning", "answer", "critic", "verification"), policy["passes"]):
        stages.extend(f"{phase}-{index}" for index in range(1, count + 1))
    return stages


CORE_ROUTE_STAGES = {}
CORE_ROUTE_EFFORTS = {}
WORK_ROUTE_REQUESTS = {}
for route_id, policy in CHAT_POLICIES.items():
    if policy["engine"] == "kova-core":
        CORE_ROUTE_STAGES[route_id] = _stage_ids(resolve_route({"surface": "chat", "route_id": route_id}))
        CORE_ROUTE_EFFORTS[route_id] = policy["reasoning_effort"]
for family in WORK_FAMILY_POLICIES:
    for name, effort in WORK_EFFORTS.items():
        if name != "Ultra":
            route_id = f"work:{family}:{name.lower().replace(' ', '-')}"
            CORE_ROUTE_STAGES[route_id] = _stage_ids(resolve_route({"surface": "work", "family": family, "effort": name}))
            CORE_ROUTE_EFFORTS[route_id] = effort["reasoning_effort"]
            WORK_ROUTE_REQUESTS[route_id] = {
                "surface": "work", "family": family, "effort": name,
            }
            if family in ("cosmo", "orion"):
                chat_route_id = f"chat:{family}:{name.lower().replace(' ', '-')}"
                CORE_ROUTE_STAGES[chat_route_id] = _stage_ids(resolve_route({"surface": "chat", "family": family, "effort": name}))
                CORE_ROUTE_EFFORTS[chat_route_id] = effort["reasoning_effort"]


def _validated_message(message):
    _require(isinstance(message, dict), "message must be an object")
    role = message.get("role")
    _require(role in ("user", "assistant"), "client system and tool messages are forbidden")
    _require(set(message) == {"role", "content"}, "message contains unsupported fields")
    content = message.get("content")
    _require(isinstance(content, str), "only text message content is enabled")
    _require(len(content) <= MAX_MESSAGE_TEXT_CHARS, "message content too large")
    return {"role": role, "content": content}


def validate_input(value):
    _require(isinstance(value, dict), "input must be an object")
    _require("model" not in value, "model is server-controlled")
    _require(set(value).issubset({"request_id", "messages", "reasoning_effort", "max_output_tokens"}), "input contains unsupported fields")
    request_id = value.get("request_id")
    _require(isinstance(request_id, str) and 1 <= len(request_id) <= 128, "invalid request_id")
    messages = value.get("messages")
    _require(isinstance(messages, list) and 1 <= len(messages) <= MAX_MESSAGES, "invalid messages")
    cleaned_messages = [_validated_message(message) for message in messages]
    _require(sum(len(message["content"]) for message in cleaned_messages) <= MAX_TOTAL_TEXT_CHARS, "aggregate message content too large")
    effort = value.get("reasoning_effort")
    _require(effort in ALLOWED_EFFORTS, "invalid reasoning_effort")
    maximum = value.get("max_output_tokens")
    _require(isinstance(maximum, int) and not isinstance(maximum, bool), "invalid max_output_tokens")
    _require(1 <= maximum <= MAX_OUTPUT_TOKENS, "invalid max_output_tokens")
    return {**value, "messages": cleaned_messages}


def validate_execution_context(value):
    _require(isinstance(value, dict), "trusted execution context missing")
    _require(
        set(value) == {
            "logical_request_id", "benchmark_candidate_id", "route_id", "stage_id",
            "public_response", "prior_stage_outputs",
        },
        "invalid trusted execution context",
    )
    logical_request_id = value["logical_request_id"]
    _require(isinstance(logical_request_id, str) and logical_request_id.startswith("kova-exec-"), "invalid logical_request_id")
    try:
        parsed_request_id = UUID(logical_request_id.removeprefix("kova-exec-"))
    except (ValueError, AttributeError) as error:
        raise ValueError("invalid logical_request_id") from error
    _require(
        parsed_request_id.version == 4 and logical_request_id == f"kova-exec-{parsed_request_id}",
        "invalid logical_request_id",
    )
    _selected_candidate(value["benchmark_candidate_id"])
    route_id = value["route_id"]
    stage_id = value["stage_id"]
    _require(route_id in CORE_ROUTE_STAGES, "execution route is not Kova Core")
    _require(stage_id in CORE_ROUTE_STAGES[route_id], "invalid execution stage")
    _require(isinstance(value["public_response"], bool), "invalid public_response")
    _require(value["public_response"] == (stage_id == CORE_ROUTE_STAGES[route_id][-1]), "public_response does not match route DAG")
    prior_outputs = value["prior_stage_outputs"]
    _require(isinstance(prior_outputs, dict), "prior_stage_outputs must be an object")
    _require(all(isinstance(key, str) for key in prior_outputs), "invalid prior stage output key")
    return {**value, "prior_stage_outputs": dict(prior_outputs)}


def _validate_runtime_value(value, selected_candidate):
    _require(isinstance(value, dict), "runtime probe must return an object")
    required = set(RUNTIME_IDENTITY_FIELDS)
    _require(set(value) == required, "runtime probe returned unsupported or missing fields")
    _require(isinstance(value.get("cold_start"), bool), "invalid cold_start")
    _require(value.get("source") == "server_provider_runtime", "untrusted runtime measurement source")
    _require(
        value.get("loaded_model") == selected_candidate["model"],
        "runtime loaded model does not match selected pinned candidate",
    )
    _require(
        value.get("loaded_model_revision") == selected_candidate["model_revision"],
        "runtime loaded model revision does not match selected pinned revision",
    )
    pinned_adapter = selected_candidate["adapter_sha256"]
    _require(isinstance(pinned_adapter, str) and len(pinned_adapter) == 64
             and all(character in "0123456789abcdef" for character in pinned_adapter),
             "trained adapter digest is not pinned for this candidate")
    _require(value.get("loaded_adapter_sha256") == pinned_adapter,
             "runtime loaded adapter does not match pinned trained adapter")
    pinned_bundle = selected_candidate["adapter_bundle_sha256"]
    _require(isinstance(pinned_bundle, str) and len(pinned_bundle) == 64
             and all(character in "0123456789abcdef" for character in pinned_bundle),
             "trained adapter bundle digest is not pinned for this candidate")
    _require(value.get("loaded_adapter_bundle_sha256") == pinned_bundle,
             "runtime loaded adapter bundle does not match pinned manifest")
    for field in RUNTIME_NUMERIC_FIELDS:
        number = value.get(field)
        _require(isinstance(number, (int, float)) and not isinstance(number, bool), f"invalid {field}")
        _require(number >= 0 and (not isinstance(number, float) or math.isfinite(number)), f"invalid {field}")
    _require(value["gpu_rate_per_second_usd"] > 0, "invalid gpu_rate_per_second_usd")
    _require(isinstance(value["worker_lifecycle_id"], str) and value["worker_lifecycle_id"], "invalid worker_lifecycle_id")
    _require(isinstance(value["gpu_type_id"], str) and value["gpu_type_id"].strip(), "invalid gpu_type_id")
    _require(isinstance(value["gpu_count"], int) and not isinstance(value["gpu_count"], bool) and 1 <= value["gpu_count"] <= 8, "invalid gpu_count")
    _require(value["serving_engine"] in ALLOWED_SERVING_ENGINES, "invalid serving_engine")
    _require(value["endpoint_type"] in ALLOWED_ENDPOINT_TYPES, "invalid endpoint_type")
    digest = value["container_image_digest"]
    _require(isinstance(digest, str) and len(digest) == 71 and digest.startswith("sha256:") and all(character in "0123456789abcdef" for character in digest[7:]), "invalid container_image_digest")
    startup_ms = value["worker_start_ms"] + value["model_load_ms"]
    _require(startup_ms > 0 if value["cold_start"] else startup_ms == 0, "cold_start and startup attribution disagree")
    return dict(value)


def validate_runtime_probe(runtime_probe, phase, selected_candidate):
    _require(callable(runtime_probe), "trusted runtime probe missing")
    _require(phase in ("before", "after"), "invalid runtime probe phase")
    return _validate_runtime_value(runtime_probe(phase), selected_candidate)


def _planner_request(value, route_id):
    base = {"request_id": value["request_id"], "messages": value["messages"]}
    if route_id in WORK_ROUTE_REQUESTS:
        return {**base, **WORK_ROUTE_REQUESTS[route_id]}
    return {**base, "route_id": route_id}


def build_engine_request(value, execution_context, *, token_counter):
    value = validate_input(value)
    execution = validate_execution_context(execution_context)
    selected_candidate = _selected_candidate(execution["benchmark_candidate_id"])
    _require(value["reasoning_effort"] == CORE_ROUTE_EFFORTS[execution["route_id"]], "reasoning_effort does not match trusted route")
    plan = build_core_plan(
        _planner_request(value, execution["route_id"]),
        candidate_model=selected_candidate["model"],
        token_counter=token_counter,
    )
    operation = next(
        (item for item in plan["operations"] if item["stage_id"] == execution["stage_id"]),
        None,
    )
    _require(operation is not None, "execution stage missing from Core plan")
    _require(operation["public_response"] == execution["public_response"], "execution visibility does not match Core plan")
    _require(value["max_output_tokens"] == operation["maximum_output_tokens"], "max_output_tokens does not match trusted stage")
    request = bind_core_operation(
        plan,
        execution["stage_id"],
        execution["prior_stage_outputs"],
        token_counter=token_counter,
    )
    _require(request["model"] == selected_candidate["model"], "Core plan changed selected pinned model")
    return request


def _usage_tokens(response):
    if not isinstance(response, dict) or not isinstance(response.get("usage"), dict):
        return 0, 0
    usage = response["usage"]
    values = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        return 0, 0
    return values


def _mapping(value, message):
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        _require(isinstance(dumped, dict), message)
        return dumped
    raise ValueError(message)


def _append_stream_tool_calls(states, fragments):
    _require(isinstance(fragments, list), "stream tool_calls must be an array")
    contributed_visible_data = False
    for raw_fragment in fragments:
        fragment = _mapping(raw_fragment, "stream tool call must be an object")
        _require(
            set(fragment).issubset({"index", "id", "type", "function"}),
            "stream tool call has invalid fields",
        )
        index = fragment.get("index")
        _require(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < MAX_TOOL_CALLS, "invalid stream tool call index")
        _require(index in states or len(states) < MAX_TOOL_CALLS, "too many engine tool_calls")
        state = states.setdefault(index, {"id": "", "type": "", "function": {"name": "", "arguments": ""}})
        for field in ("id", "type"):
            part = fragment.get(field)
            _require(part is None or isinstance(part, str), f"invalid stream tool call {field}")
            if part:
                contributed_visible_data = contributed_visible_data or bool(part.strip())
                state[field] += part
                _require(len(state[field]) <= (256 if field == "id" else 8), "stream tool identity too large")
        function = fragment.get("function")
        if function is not None:
            function = _mapping(function, "stream tool call function must be an object")
            _require(
                set(function).issubset({"name", "arguments"}),
                "stream tool call function has invalid fields",
            )
            for field in ("name", "arguments"):
                part = function.get(field)
                _require(part is None or isinstance(part, str), f"invalid stream tool function {field}")
                if part:
                    contributed_visible_data = contributed_visible_data or bool(part.strip())
                    state["function"][field] += part
                    _require(len(state["function"][field]) <= (64 if field == "name" else MAX_TOOL_ARGUMENT_CHARS),
                             "stream tool function too large")
    return contributed_visible_data


def consume_engine_response(response, *, expect_stream, clock_ns, started_ns, timing_state):
    """Normalize one engine response and measure first visible streamed output."""
    _require(
        isinstance(timing_state, dict) and set(timing_state) == {"time_to_first_token_ms"} and
        timing_state["time_to_first_token_ms"] is None,
        "invalid stream timing state",
    )
    if not expect_stream:
        normalized = _mapping(response, "non-stream engine response must be an object")
        finished_ns = clock_ns()
        return normalized, finished_ns

    _require(not isinstance(response, dict), "streaming engine response must be an iterable of chunks")
    _require(not isinstance(response, (str, bytes)), "streaming engine response must be an iterable of chunks")
    try:
        chunks = iter(response)
    except TypeError as error:
        raise ValueError("streaming engine response must be an iterable of chunks") from error

    content_parts = []
    content_chars = 0
    tool_call_states = {}
    usage = None
    first_token_ns = None
    finish_reason = None
    try:
        for raw_chunk in chunks:
            chunk = _mapping(raw_chunk, "stream chunk must be an object")
            if chunk.get("usage") is not None:
                _require(usage is None, "stream returned duplicate usage evidence")
                usage = _mapping(chunk["usage"], "stream usage must be an object")
            choices = chunk.get("choices", [])
            _require(isinstance(choices, list) and len(choices) <= 1, "stream must return at most one choice")
            for raw_choice in choices:
                _require(finish_reason is None, "stream returned a choice after its terminal finish reason")
                choice = _mapping(raw_choice, "stream choice must be an object")
                _require(type(choice.get("index", 0)) is int and choice.get("index", 0) == 0,
                         "stream returned an unexpected choice index")
                delta = _mapping(choice.get("delta"), "stream choice missing delta")
                reason = choice.get("finish_reason")
                _require(reason is None or isinstance(reason, str), "invalid stream finish_reason")
                if reason is not None:
                    _require(finish_reason is None, "stream returned multiple finish reasons")
                    finish_reason = reason
                _require(delta.get("reasoning_content") in (None, ""), "engine returned hidden reasoning")
                content = delta.get("content")
                _require(content is None or isinstance(content, str), "stream content must be text or null")
                fragments = delta.get("tool_calls", [])
                _require(isinstance(fragments, list), "stream tool_calls must be an array")
                meaningful_tool_fragment = _append_stream_tool_calls(tool_call_states, fragments)
                if first_token_ns is None and ((content and content.strip()) or meaningful_tool_fragment):
                    first_token_ns = clock_ns()
                    timing_state["time_to_first_token_ms"] = max(0, first_token_ns - started_ns) / 1_000_000
                if content:
                    content_chars += len(content)
                    _require(content_chars <= MAX_TOTAL_TEXT_CHARS, "stream content too large")
                    content_parts.append(content)
    finally:
        # The consumer can reject a chunk before exhausting the provider stream.
        # Close explicitly rather than relying on generator garbage collection.
        close = getattr(chunks, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                # Never mask the original rejection with transport cleanup details.
                pass

    finished_ns = clock_ns()
    tool_calls = [tool_call_states[index] for index in sorted(tool_call_states)]
    normalized = {
        "choices": [{
            "message": {"content": "".join(content_parts), "tool_calls": tool_calls},
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }
    return normalized, finished_ns


def _unique_tool_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate tool argument key")
        result[key] = value
    return result


def _finite_tool_float(raw):
    value = float(raw)
    _require(math.isfinite(value), "nonfinite tool argument")
    return value


def _reject_tool_constant(_value):
    raise ValueError("nonfinite tool argument")


def _validate_tool_structure(value):
    # Iterative validation avoids interpreter-dependent recursion limits.
    pending = [(value, 0)]
    visited = 0
    while pending:
        current, depth = pending.pop()
        visited += 1
        _require(depth <= MAX_TOOL_JSON_DEPTH and visited <= MAX_TOOL_JSON_NODES,
                 "tool argument structure too large")
        if isinstance(current, dict):
            pending.extend((item, depth + 1) for pair in current.items() for item in pair)
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)
        elif isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError("tool argument is not valid UTF-8") from None


def _sanitized_tool_calls(value):
    _require(isinstance(value, list), "engine tool_calls must be an array")
    _require(len(value) <= MAX_TOOL_CALLS, "too many engine tool_calls")
    cleaned = []
    seen_ids = set()
    for raw_call in value:
        call = _mapping(raw_call, "engine tool call must be an object")
        _require(set(call) == {"id", "type", "function"}, "engine tool call has invalid fields")
        call_id = call["id"]
        _require(isinstance(call_id, str) and 1 <= len(call_id) <= 256, "invalid engine tool call id")
        _require(call_id not in seen_ids, "duplicate engine tool call id")
        seen_ids.add(call_id)
        _require(call["type"] == "function", "unsupported engine tool call type")
        function = _mapping(call["function"], "engine tool call function must be an object")
        _require(set(function) == {"name", "arguments"}, "engine tool function has invalid fields")
        name = function["name"]
        arguments = function["arguments"]
        _require(
            isinstance(name, str) and 1 <= len(name) <= 64 and name.isascii() and
            all(character.isalnum() or character in "_-" for character in name),
            "invalid engine tool function name",
        )
        _require(
            isinstance(arguments, str) and len(arguments) <= MAX_TOOL_ARGUMENT_CHARS,
            "invalid engine tool function arguments",
        )
        try:
            decoded_arguments = json.loads(arguments, object_pairs_hook=_unique_tool_object,
                                           parse_constant=_reject_tool_constant, parse_float=_finite_tool_float)
        except (ValueError, RecursionError):
            raise ValueError("engine tool function arguments must be valid JSON with finite values and unique keys") from None
        _require(isinstance(decoded_arguments, dict), "engine tool function arguments must be a JSON object")
        _validate_tool_structure(decoded_arguments)
        cleaned.append({
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        })
    return cleaned


def sanitize_engine_response(request_id, response):
    _require(isinstance(response, dict), "engine response must be an object")
    choices = response.get("choices")
    _require(isinstance(choices, list) and len(choices) == 1, "engine response must contain exactly one choice")
    first = choices[0]
    _require(isinstance(first, dict), "engine choice must be an object")
    _require(type(first.get("index", 0)) is int and first.get("index", 0) == 0,
             "engine returned an unexpected choice index")
    message = first.get("message")
    _require(isinstance(message, dict), "engine response missing message")
    finish_reason = first.get("finish_reason")
    _require(finish_reason in ("stop", "tool_calls"), "engine response is incomplete or has invalid finish_reason")
    content = message.get("content")
    tool_calls = _sanitized_tool_calls(message.get("tool_calls", []))
    _require(content is None or isinstance(content, str), "engine response content must be text or null")
    _require(content is None or len(content) <= MAX_TOTAL_TEXT_CHARS, "engine response content too large")
    has_content = bool(content and content.strip())
    _require(has_content or bool(tool_calls), "engine response must contain non-whitespace content or tool_calls")
    _require(finish_reason != "tool_calls" or bool(tool_calls), "tool-call finish_reason missing tool_calls")
    _require(finish_reason != "stop" or not tool_calls, "stop finish_reason cannot contain tool_calls")
    _require(message.get("reasoning_content") in (None, ""), "engine returned hidden reasoning")
    lowered = (content or "").lower()
    _require("<think" not in lowered and "</think>" not in lowered, "engine embedded hidden reasoning in content")
    usage = response.get("usage")
    _require(isinstance(usage, dict), "engine response missing usage")
    input_tokens, output_tokens = _usage_tokens(response)
    _require(input_tokens > 0, "invalid input_tokens")
    _require(output_tokens > 0, "invalid output_tokens")
    return {
        "request_id": request_id,
        "content": content if has_content else "",
        "tool_calls": tool_calls,
        "usage": usage,
    }


def _attempt_record(value, execution, attempt_id, outcome, elapsed_ms, first_token_ms, runtime, response):
    input_tokens, output_tokens = _usage_tokens(response)
    return {
        "record_type": "attempt",
        "request_id": execution["logical_request_id"],
        "correlation_id": value["request_id"],
        "attempt_id": attempt_id,
        "outcome": outcome,
        "model": runtime["loaded_model"],
        "model_revision": runtime["loaded_model_revision"],
        "adapter_sha256": runtime["loaded_adapter_sha256"],
        "adapter_bundle_sha256": runtime["loaded_adapter_bundle_sha256"],
        "route_id": execution["route_id"],
        "stage_id": execution["stage_id"],
        "public_response": execution["public_response"],
        "worker_lifecycle_id": runtime["worker_lifecycle_id"],
        "cold_start": runtime["cold_start"],
        "reasoning_effort": value["reasoning_effort"],
        "worker_start_ms": runtime["worker_start_ms"],
        "model_load_ms": runtime["model_load_ms"],
        "queue_ms": runtime["queue_ms"],
        "inference_ms": elapsed_ms,
        "time_to_first_token_ms": first_token_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "gpu_rate_per_second_usd": runtime["gpu_rate_per_second_usd"],
        "gpu_type_id": runtime["gpu_type_id"],
        "gpu_count": runtime["gpu_count"],
        "serving_engine": runtime["serving_engine"],
        "endpoint_type": runtime["endpoint_type"],
        "container_image_digest": runtime["container_image_digest"],
        "measurement_source": runtime["source"],
    }


def emit_lifecycle_close(
    runtime_close_probe, telemetry_sink, *, benchmark_candidate_id,
    close_event_id_factory=lambda: str(uuid4()),
):
    """Emit the idle-tail event only after the trusted runtime observes worker shutdown."""
    _require(callable(runtime_close_probe), "trusted lifecycle close probe missing")
    _require(callable(telemetry_sink), "telemetry sink missing")
    selected_candidate = _selected_candidate(benchmark_candidate_id)
    value = runtime_close_probe()
    required = {
        "source", "worker_lifecycle_id", "loaded_model", "loaded_model_revision", "loaded_adapter_sha256",
        "loaded_adapter_bundle_sha256", "billed_lifecycle_ms",
        "attributed_idle_timeout_ms", "gpu_rate_per_second_usd", "gpu_type_id", "gpu_count",
        "serving_engine", "endpoint_type", "container_image_digest",
    }
    _require(isinstance(value, dict) and set(value) == required, "invalid lifecycle close probe")
    identity_probe = {
        **{field: value[field] for field in (
            "source", "worker_lifecycle_id", "loaded_model", "loaded_model_revision", "loaded_adapter_sha256",
            "loaded_adapter_bundle_sha256",
            "gpu_rate_per_second_usd", "gpu_type_id", "gpu_count", "serving_engine",
            "endpoint_type", "container_image_digest",
        )},
        "cold_start": False,
        "worker_start_ms": 0,
        "model_load_ms": 0,
        "queue_ms": 0,
    }
    validated = _validate_runtime_value(identity_probe, selected_candidate)
    idle_ms = value["attributed_idle_timeout_ms"]
    _require(isinstance(idle_ms, (int, float)) and not isinstance(idle_ms, bool) and idle_ms > 0
             and (not isinstance(idle_ms, float) or math.isfinite(idle_ms)), "invalid attributed_idle_timeout_ms")
    billed_ms = value["billed_lifecycle_ms"]
    _require(isinstance(billed_ms, (int, float)) and not isinstance(billed_ms, bool) and billed_ms > 0
             and (not isinstance(billed_ms, float) or math.isfinite(billed_ms)), "invalid billed_lifecycle_ms")
    _require(idle_ms <= billed_ms, "idle tail exceeds billed lifecycle")
    close_event_id = close_event_id_factory()
    _require(isinstance(close_event_id, str) and close_event_id, "invalid close_event_id")
    record = {
        "record_type": "lifecycle_close",
        "close_event_id": close_event_id,
        "worker_lifecycle_id": validated["worker_lifecycle_id"],
        "model": validated["loaded_model"],
        "model_revision": validated["loaded_model_revision"],
        "adapter_sha256": validated["loaded_adapter_sha256"],
        "adapter_bundle_sha256": validated["loaded_adapter_bundle_sha256"],
        "billed_lifecycle_ms": billed_ms,
        "attributed_idle_timeout_ms": idle_ms,
        "gpu_rate_per_second_usd": validated["gpu_rate_per_second_usd"],
        "gpu_type_id": validated["gpu_type_id"],
        "gpu_count": validated["gpu_count"],
        "serving_engine": validated["serving_engine"],
        "endpoint_type": validated["endpoint_type"],
        "container_image_digest": validated["container_image_digest"],
        "measurement_source": validated["source"],
    }
    telemetry_sink(record)
    return record


def handle_job(
    job, inference_client, runtime_probe, telemetry_sink, *, execution_context,
    token_counter, clock_ns=perf_counter_ns, attempt_id_factory=lambda: str(uuid4()),
):
    """Execute one Core stage and persist its attempt telemetry before returning or raising."""
    _require(isinstance(job, dict) and set(job) == {"input"} and isinstance(job["input"], dict), "job must contain only input")
    _require(callable(inference_client), "inference client missing")
    _require(callable(telemetry_sink), "telemetry sink missing")
    value = validate_input(job["input"])
    execution = validate_execution_context(execution_context)
    selected_candidate = _selected_candidate(execution["benchmark_candidate_id"])
    engine_request = build_engine_request(value, execution, token_counter=token_counter)
    before = validate_runtime_probe(runtime_probe, "before", selected_candidate)
    attempt_id = attempt_id_factory()
    _require(isinstance(attempt_id, str) and attempt_id, "invalid server attempt_id")
    started_ns = clock_ns()
    response = None
    finished_ns = None
    timing_state = {"time_to_first_token_ms": None}
    result = None
    operation_error = None
    try:
        response = inference_client(engine_request)
        response, finished_ns = consume_engine_response(
            response,
            expect_stream=engine_request["stream"],
            clock_ns=clock_ns,
            started_ns=started_ns,
            timing_state=timing_state,
        )
        result = sanitize_engine_response(value["request_id"], response)
    except Exception as error:
        operation_error = error
        if finished_ns is None:
            finished_ns = clock_ns()
    elapsed_ms = max(0, finished_ns - started_ns) / 1_000_000
    first_token_ms = timing_state["time_to_first_token_ms"]

    try:
        after = validate_runtime_probe(runtime_probe, "after", selected_candidate)
        _require(all(after[field] == before[field] for field in RUNTIME_IDENTITY_FIELDS), "runtime identity changed during attempt")
        _require(first_token_ms is None or first_token_ms <= elapsed_ms, "first token exceeds measured inference")
        _require(execution["public_response"] or first_token_ms is None, "private response cannot claim first-token measurement")
        _require(
            operation_error is not None or not execution["public_response"] or first_token_ms is not None,
            "public response missing first-token measurement",
        )
    except Exception as integrity_error:
        quarantined = _attempt_record(
            value, execution, attempt_id, "quarantined", elapsed_ms, first_token_ms, before, response,
        )
        telemetry_sink(quarantined)
        if operation_error is not None:
            raise operation_error.with_traceback(operation_error.__traceback__) from integrity_error
        raise

    outcome = "failed" if operation_error is not None else "success"
    record = _attempt_record(
        value, execution, attempt_id, outcome, elapsed_ms, first_token_ms, after, response,
    )
    telemetry_sink(record)
    if operation_error is not None:
        raise operation_error.with_traceback(operation_error.__traceback__)
    result["benchmark"] = record
    return result
