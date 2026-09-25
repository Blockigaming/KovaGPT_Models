"""Legacy RunPod queue adapter over the shared provider-neutral protocol.

The queue envelope remains unchanged. No RunPod SDK or network calls are added;
Azure uses worker.azure_container_apps and never sends this queue envelope.
"""

from worker.openai_protocol import (
    FORBIDDEN_TRANSPORT_FIELDS,
    MAX_SSE_BYTES,
    MAX_SSE_EVENT_BYTES,
    MAX_SSE_EVENTS,
    OPENAI_CHAT_ROUTE,
    OpenAIProtocolError,
    decode_response,
    parse_raw_sse as _parse_raw_sse,
    validate_chat_request,
)


class RunPodVllmError(OpenAIProtocolError):
    """The legacy worker returned unsafe or malformed output."""


def _legacy_error(error):
    return RunPodVllmError(str(error).replace("OpenAI-compatible", "RunPod vLLM"))


def build_queue_job(engine_request):
    """Preserve the pinned worker's exact queue input shape."""
    try:
        body = validate_chat_request(engine_request)
    except OpenAIProtocolError as error:
        raise _legacy_error(error) from None
    return {"input": {"openai_route": OPENAI_CHAT_ROUTE, "openai_input": body}}


def parse_raw_sse(fragments):
    """Preserve the public RunPod error type while sharing the bounded parser."""
    try:
        yield from _parse_raw_sse(fragments)
    except OpenAIProtocolError as error:
        raise _legacy_error(error) from None


def decode_worker_output(output, *, expect_stream):
    if not isinstance(expect_stream, bool):
        raise RunPodVllmError("expect_stream must be boolean")
    if expect_stream:
        return parse_raw_sse(output)
    try:
        return decode_response(output, expect_stream=False)
    except OpenAIProtocolError as error:
        raise _legacy_error(error) from None


def make_queue_inference_client(worker_call):
    """Adapt a trusted worker call without changing the Kova handler contract."""
    if not callable(worker_call):
        raise RunPodVllmError("worker call must be callable")

    def inference_client(engine_request):
        output = worker_call(build_queue_job(engine_request))
        return decode_worker_output(output, expect_stream=engine_request["stream"])

    return inference_client
