"""Provider-neutral OpenAI-compatible request and bounded response protocol.

No provider SDK or network work. Raw output must pass the Kova handler's
visibility, tool and usage checks before anything is sent to the user.
"""

from copy import deepcopy
import codecs
from io import StringIO
import json


OPENAI_CHAT_ROUTE = "/v1/chat/completions"
MAX_SSE_BYTES = 16 * 1024 * 1024
MAX_SSE_EVENT_BYTES = 2 * 1024 * 1024
MAX_SSE_EVENTS = 32 * 1024
FORBIDDEN_TRANSPORT_FIELDS = frozenset(
    (
        "api_key",
        "endpoint_id",
        "input",
        "openai_input",
        "openai_route",
        "route",
        "body",
        "method",
    )
)


class OpenAIProtocolError(ValueError):
    """The pinned worker protocol returned unsafe or malformed output."""


def _require(condition, message):
    if not condition:
        raise OpenAIProtocolError(message)


def _utf8_size(value):
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise OpenAIProtocolError("OpenAI-compatible SSE is not valid UTF-8") from error


def _reject_worker_error(value):
    if not isinstance(value, dict) or value.get("error") is None:
        return value
    error = value["error"]
    if isinstance(error, dict):
        error_type = error.get("type")
        detail = (
            error_type if isinstance(error_type, str) and error_type in {
                "startup_error", "validation_error", "server_error", "engine_error",
                "invalid_request_error", "rate_limit_error", "authentication_error",
            } else "unknown_error"
        )
    else:
        detail = "unknown_error"
    raise OpenAIProtocolError(f"OpenAI-compatible worker error: {detail}")


def validate_chat_request(engine_request):
    """Validate transport-independent fields and return a defensive copy."""
    _require(isinstance(engine_request, dict), "engine request must be an object")
    _require(
        not FORBIDDEN_TRANSPORT_FIELDS.intersection(engine_request),
        "engine request contains transport-controlled fields",
    )
    _require(
        isinstance(engine_request.get("model"), str) and engine_request["model"],
        "engine request model missing",
    )
    _require(isinstance(engine_request.get("messages"), list), "engine request messages missing")
    _require(
        isinstance(engine_request.get("stream"), bool),
        "engine request stream flag missing",
    )
    return deepcopy(engine_request)


def _reject_constant(_value):
    raise OpenAIProtocolError("non-finite OpenAI-compatible JSON")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate OpenAI-compatible JSON key")
        result[key] = value
    return result


def _parse_sse_event(raw_event, event_number):
    _require(
        _utf8_size(raw_event) <= MAX_SSE_EVENT_BYTES,
        "OpenAI-compatible SSE event too large",
    )
    data_lines = []
    for line in raw_event.split("\n"):
        if line.startswith(":"):
            continue
        _require(line.startswith("data:"), "unsupported OpenAI-compatible SSE field")
        data_lines.append(line[5:].removeprefix(" "))
    _require(len(data_lines) == 1 and data_lines[0], "invalid OpenAI-compatible SSE data event")
    payload = data_lines[0]
    if payload == "[DONE]":
        return None
    try:
        value = json.loads(payload, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        raise OpenAIProtocolError(
            f"invalid OpenAI-compatible SSE JSON at event {event_number}"
        ) from None
    _require(isinstance(value, dict), "OpenAI-compatible SSE payload must be an object")
    return _reject_worker_error(value)


def parse_raw_sse(fragments):
    """Yield OpenAI chunk objects from fragmented raw SSE worker output.

    A terminal ``data: [DONE]`` event is mandatory and must be the final event.
    Provider HTTP or queue envelopes must be removed by their own adapters;
    only the strict data-only Chat Completions SSE contract is accepted.
    """
    _require(not isinstance(fragments, (str, bytes, dict)), "SSE output must be an iterable")
    try:
        iterator = iter(fragments)
    except TypeError as error:
        raise OpenAIProtocolError("SSE output must be an iterable") from error

    event_lines = []
    line_buffer = StringIO()
    event_buffer_bytes = 0
    pending_cr = False
    utf8_decoder = codecs.getincrementaldecoder("utf-8")()
    total_bytes = 0
    event_count = 0
    json_event_count = 0
    done_seen = False

    for fragment in iterator:
        _require(isinstance(fragment, (str, bytes)), "SSE fragment must be text or bytes")
        _require(fragment != "" and fragment != b"", "OpenAI-compatible SSE fragment must not be empty")
        if isinstance(fragment, bytes):
            total_bytes += len(fragment)
            _require(total_bytes <= MAX_SSE_BYTES, "OpenAI-compatible SSE response too large")
            try:
                text = utf8_decoder.decode(fragment, final=False)
            except UnicodeDecodeError as error:
                raise OpenAIProtocolError("OpenAI-compatible SSE is not valid UTF-8") from error
        else:
            _require(not utf8_decoder.getstate()[0], "OpenAI-compatible SSE is not valid UTF-8")
            text = fragment
            total_bytes += _utf8_size(text)
            _require(total_bytes <= MAX_SSE_BYTES, "OpenAI-compatible SSE response too large")
        position = 0
        if pending_cr and text:
            _require(text[0] == "\n", "unsupported OpenAI-compatible SSE line ending")
            pending_cr = False
            text = "\n" + text[1:]

        while position < len(text):
            newline = position
            while newline < len(text) and text[newline] not in "\r\n":
                newline += 1
            segment = text[position:newline]
            if segment:
                line_buffer.write(segment)
                event_buffer_bytes += _utf8_size(segment)
            _require(
                event_buffer_bytes <= MAX_SSE_EVENT_BYTES,
                "OpenAI-compatible SSE buffer too large",
            )
            if newline == len(text):
                break

            if text[newline] == "\r":
                if newline + 1 == len(text):
                    pending_cr = True
                    break
                _require(text[newline + 1] == "\n", "unsupported OpenAI-compatible SSE line ending")
                position = newline + 2
            else:
                position = newline + 1
            event_buffer_bytes += 1
            _require(
                event_buffer_bytes <= MAX_SSE_EVENT_BYTES,
                "OpenAI-compatible SSE buffer too large",
            )

            line = line_buffer.getvalue()
            line_buffer = StringIO()
            if line:
                event_lines.append(line)
                continue

            raw_event = "\n".join(event_lines)
            event_lines = []
            event_buffer_bytes = 0
            if not raw_event:
                continue
            _require(not done_seen, "OpenAI-compatible SSE event followed terminal marker")
            event_count += 1
            _require(event_count <= MAX_SSE_EVENTS, "too many OpenAI-compatible SSE events")
            value = _parse_sse_event(raw_event, event_count)
            if value is None:
                _require(json_event_count > 0, "OpenAI-compatible SSE ended before any data")
                done_seen = True
            else:
                json_event_count += 1
                yield value

    try:
        utf8_decoder.decode(b"", final=True)
    except UnicodeDecodeError as error:
        raise OpenAIProtocolError("OpenAI-compatible SSE is not valid UTF-8") from error
    _require(not pending_cr, "unsupported OpenAI-compatible SSE line ending")
    trailing = "\n".join((*event_lines, line_buffer.getvalue()))
    _require(not trailing.strip(), "incomplete OpenAI-compatible SSE event")
    _require(done_seen, "OpenAI-compatible SSE terminal marker missing")


def decode_response(output, *, expect_stream):
    """Decode the pinned worker's yielded values after provider transport handling."""
    _require(isinstance(expect_stream, bool), "expect_stream must be boolean")
    if expect_stream:
        return parse_raw_sse(output)

    if isinstance(output, dict):
        return deepcopy(_reject_worker_error(output))
    _require(not isinstance(output, (str, bytes)), "non-stream worker output must be an object")
    try:
        iterator = iter(output)
    except TypeError as error:
        raise OpenAIProtocolError("non-stream worker output must be an object") from error
    try:
        value = next(iterator)
    except StopIteration as error:
        raise OpenAIProtocolError("non-stream worker must yield one object") from error
    _require(isinstance(value, dict), "non-stream worker must yield one object")
    try:
        next(iterator)
    except StopIteration:
        return deepcopy(_reject_worker_error(value))
    raise OpenAIProtocolError("non-stream worker must yield one object")
