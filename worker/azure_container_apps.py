"""Guarded Azure Container Apps HTTP boundary; imports start no network work.

A trusted transport must implement TLS verification, no redirects, bounded I/O,
and abort-on-close. No HTTP/credential SDK is bound here. CPU fixtures exercise
this boundary; they do not establish Azure authentication or runtime readiness.
Raw provider output belongs to worker.handler, never directly to the UI.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import json
import math
import re
from time import monotonic
from sys import exc_info

from worker.openai_protocol import (
    MAX_SSE_BYTES,
    OPENAI_CHAT_ROUTE,
    OpenAIProtocolError,
    decode_response,
    validate_chat_request,
)
from worker.response_privacy import require_private_reasoning_absent


# A per-HTTP-hop ceiling, NOT a Kova mode's approved active-work budget.
AZURE_HTTP_INGRESS_TIMEOUT_SECONDS = 240
MAX_REQUEST_BYTES = 4 * 1024 * 1024
_ALLOWED_FIELDS = frozenset((
    "model", "messages", "stream", "stream_options", "max_tokens",
    "reasoning_effort", "chat_template_kwargs", "tools", "tool_choice",
    "parallel_tool_calls", "temperature", "top_p", "seed", "response_format",
))
_ORIGIN = re.compile(
    r"https://(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+azurecontainerapps\.io"
    r"(?::443)?\Z", re.ASCII,
)
_BEARER = re.compile(r"[A-Za-z0-9\-._~+/]+=*\Z", re.ASCII)


class AzureProtocolError(OpenAIProtocolError):
    """An Azure request or HTTP response violated the boundary contract."""


class AzureExecutionBlocked(RuntimeError):
    """Paid execution, private authentication, or transport verification is absent."""


class AzureCancelled(RuntimeError):
    """The caller cancelled this execution."""


class AzureDeadlineExceeded(TimeoutError):
    """The caller's bounded per-hop time budget elapsed."""


def _require(condition, message):
    if not condition:
        raise AzureProtocolError(message)


@dataclass(frozen=True)
class AzureSettings:
    """Server-only settings; never construct these from the client request body."""

    origin: str
    served_model: str
    timeout_seconds: float
    paid_execution_authorized: bool = False
    private_auth_verified: bool = False
    live_transport_verified: bool = False

    def __post_init__(self):
        _require(
            isinstance(self.origin, str) and len(self.origin) <= 270
            and _ORIGIN.fullmatch(self.origin) is not None,
            "Azure origin must be one canonical HTTPS Container Apps origin",
        )
        _require(
            isinstance(self.served_model, str) and 0 < len(self.served_model) <= 256
            and self.served_model.isascii()
            and all(32 < ord(char) < 127 for char in self.served_model),
            "trusted served model missing or invalid",
        )
        _require(
            type(self.timeout_seconds) in (int, float)
            and math.isfinite(self.timeout_seconds)
            and 0 < self.timeout_seconds < AZURE_HTTP_INGRESS_TIMEOUT_SECONDS,
            "per-hop timeout must be finite, positive and below Azure ingress limit",
        )
        for flag in (
            self.paid_execution_authorized, self.private_auth_verified,
            self.live_transport_verified,
        ):
            _require(type(flag) is bool, "execution guard must be boolean")


@dataclass(frozen=True)
class AzureRequest:
    """Immutable transport input; request bodies and credentials are not printable."""

    url: str
    body: bytes = field(repr=False)
    stream: bool
    timeout_seconds: float
    method: str = "POST"
    follow_redirects: bool = False
    verify_tls: bool = True


@dataclass(frozen=True)
class AzureResponse:
    """HTTP transport output. close() must abort/release the underlying response."""

    status: int
    content_type: str
    final_url: str
    body: bytes | str | Iterable[bytes | str] = field(repr=False)
    close: Callable[[], None] = field(repr=False)


def prepare_request(engine_request, settings):
    """Serialize a trusted Kova stage directly, without the old queue envelope."""
    _require(type(settings) is AzureSettings, "trusted Azure settings required")
    request = validate_chat_request(engine_request)
    _require(not (set(request) - _ALLOWED_FIELDS), "unsupported Azure request fields")
    _require(request["model"] == settings.served_model, "server-selected model mismatch")
    # Suppress wire reasoning, not model thinking. The selected serving image must
    # support this contract before live_transport_verified may be asserted.
    request["include_reasoning"] = False
    try:
        body = json.dumps(
            request, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise AzureProtocolError("Azure request must be finite UTF-8 JSON") from None
    _require(len(body) <= MAX_REQUEST_BYTES, "Azure request too large")
    return AzureRequest(
        url=settings.origin + OPENAI_CHAT_ROUTE,
        body=body,
        stream=request["stream"],
        timeout_seconds=settings.timeout_seconds,
    )


def _execution_guard(settings):
    if not all((
        settings.paid_execution_authorized,
        settings.private_auth_verified,
        settings.live_transport_verified,
    )):
        raise AzureExecutionBlocked("Azure model execution remains disabled")


def _safe_close(close):
    try:
        close()
        return True
    except Exception:
        # Do not replace a model/transport failure with a potentially secret-bearing
        # cleanup exception. No response details or credential material are logged.
        return False


def _reject_json_constant(_value):
    raise AzureProtocolError("non-finite Azure response JSON")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate Azure response JSON key")
        result[key] = value
    return result


def _raw_fragments(body, check):
    source = (body,) if isinstance(body, (str, bytes)) else body
    _require(not isinstance(source, dict), "Azure body must be raw HTTP bytes or text")
    try:
        iterator = iter(source)
    except TypeError:
        raise AzureProtocolError("Azure body is not iterable") from None
    try:
        while True:
            check()
            try:
                fragment = next(iterator)
            except StopIteration:
                break
            except (AzureCancelled, AzureDeadlineExceeded, AzureExecutionBlocked):
                raise
            except Exception:
                raise AzureProtocolError("Azure response read failed") from None
            check()
            _require(isinstance(fragment, (bytes, str)), "invalid Azure response fragment")
            _require(bool(fragment), "empty Azure response fragment")
            yield fragment
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            _safe_close(close)


def _decode_json(fragments):
    chunks = []
    total = 0
    for fragment in fragments:
        try:
            encoded = fragment.encode("utf-8") if isinstance(fragment, str) else fragment
        except UnicodeError:
            raise AzureProtocolError("Azure response is not valid UTF-8") from None
        total += len(encoded)
        _require(total <= MAX_SSE_BYTES, "Azure response too large")
        chunks.append(encoded)
    try:
        value = json.loads(
            b"".join(chunks).decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeError, ValueError, RecursionError):
        raise AzureProtocolError("invalid Azure response JSON") from None
    _require(isinstance(value, dict), "Azure response JSON must be an object")
    return decode_response(value, expect_stream=False)


def make_azure_inference_client(
    settings, transport, credential_provider, *, clock=monotonic, cancelled=lambda: False,
):
    """Bind an explicitly authorized server transport to worker.handler.

    Transport is called once as transport(request, headers). Its time limit covers
    connect AND reads; it must not follow redirects, retry, or log headers/bodies.
    Cooperative checks here complement (do not replace) transport I/O timeouts.
    CPU tests use in-memory transport and synthetic authorization flags only.
    The repository's real Azure readiness/authorization flags stay false.
    """
    _require(type(settings) is AzureSettings, "trusted Azure settings required")
    for dependency in (transport, credential_provider, clock, cancelled):
        _require(callable(dependency), "trusted Azure dependency must be callable")

    def inference_client(engine_request):
        _execution_guard(settings)  # Before credentials, transport, or iterator creation.
        request = prepare_request(engine_request, settings)
        started = clock()
        _require(type(started) in (int, float) and math.isfinite(started), "invalid clock")
        deadline = started + settings.timeout_seconds

        def check():
            if cancelled():
                raise AzureCancelled("Azure model execution cancelled")
            now = clock()
            _require(type(now) in (int, float) and math.isfinite(now), "invalid clock")
            _require(now >= started, "clock moved backwards")
            if now >= deadline:
                raise AzureDeadlineExceeded("Azure per-hop deadline exceeded")
            return now

        def exchange():
            check()
            try:
                token = credential_provider()
            except (AzureCancelled, AzureDeadlineExceeded, AzureExecutionBlocked):
                raise
            except Exception:
                raise AzureProtocolError("Azure credential acquisition failed") from None
            check()
            _require(
                isinstance(token, str) and 0 < len(token) <= 16384
                and _BEARER.fullmatch(token) is not None,
                "invalid server bearer credential",
            )
            headers = {
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if request.stream else "application/json",
            }
            remaining = deadline - check()
            if remaining <= 0:
                raise AzureDeadlineExceeded("Azure per-hop deadline exceeded")
            bounded_request = AzureRequest(
                request.url, request.body, request.stream, remaining,
            )
            try:
                response = transport(bounded_request, headers)
            except (AzureCancelled, AzureDeadlineExceeded, AzureExecutionBlocked):
                raise
            except Exception:
                raise AzureProtocolError("Azure transport failed") from None
            if type(response) is not AzureResponse:
                cleanup = getattr(response, "close", None)
                if callable(cleanup):
                    _safe_close(cleanup)
                raise AzureProtocolError("invalid Azure transport response")
            _require(callable(response.close), "Azure transport must support response close")
            return response

        def decoded():
            # Delaying this until iteration prevents leaking unopened/unconsumed streams.
            response = exchange()
            fragments = None
            try:
                check()
                _require(
                    type(response.status) is int and response.status == 200,
                    "Azure HTTP response rejected",
                )
                _require(response.final_url == request.url, "Azure redirects are forbidden")
                expected_type = "text/event-stream" if request.stream else "application/json"
                _require(
                    isinstance(response.content_type, str)
                    and response.content_type.split(";", 1)[0].strip().lower() == expected_type,
                    "unexpected Azure response content type",
                )
                fragments = _raw_fragments(response.body, check)
                output = (
                    decode_response(fragments, expect_stream=True)
                    if request.stream else (_decode_json(fragments),)
                )
                for value in output:
                    check()
                    _require(
                        "model" not in value or value["model"] == settings.served_model,
                        "Azure response model mismatch",
                    )
                    require_private_reasoning_absent(value)
                    yield value
                check()
            finally:
                already_failing = exc_info()[0] is not None
                if fragments is not None:
                    fragments.close()
                closed = _safe_close(response.close)
                if not closed and not already_failing:
                    raise AzureProtocolError("Azure response cleanup failed") from None

        if request.stream:
            return decoded()
        iterator = decoded()
        try:
            result = next(iterator)
            try:
                next(iterator)
            except StopIteration:
                return result
            raise AzureProtocolError("Azure response produced multiple objects")
        finally:
            iterator.close()

    return inference_client
