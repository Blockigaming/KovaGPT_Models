"""In-memory Azure HTTP fixtures: no credentials, Azure resources or GPU calls."""

from dataclasses import FrozenInstanceError, replace
import json
import unittest
from unittest.mock import patch

from worker import azure_container_apps as azure
from worker.openai_protocol import OpenAIProtocolError


ORIGIN = "https://kova-core.internal.fixture.eastus.azurecontainerapps.io"
MODEL = "fixture-model"
SECRET = "fixture.token.not-a-real-credential"


def settings(**changes):
    # Synthetic flags for exercising the injected in-memory transport, not evidence
    # of Azure readiness. Source-controlled live configuration remains disabled.
    return replace(azure.AzureSettings(
        ORIGIN, MODEL, 30, paid_execution_authorized=True,
        private_auth_verified=True, live_transport_verified=True,
    ), **changes)


def request(stream=False, **changes):
    return {
        "model": MODEL, "messages": [{"role": "user", "content": "hello"}],
        "stream": stream, "max_tokens": 2048, **changes,
    }


def payload(content="Kova answer"):
    return {"model": MODEL, "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def stream_wire():
    values = [
        {"model": MODEL, "choices": [{"delta": {"content": "Kova café 東京"}}]},
        {"model": MODEL, "choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"model": MODEL, "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
    ]
    return ("".join("data: " + json.dumps(value, ensure_ascii=False) + "\n\n" for value in values)
            + "data: [DONE]\n\n").encode()


class FakeTransport:
    def __init__(self, body=None, **changes):
        self.body = json.dumps(payload()).encode() if body is None else body
        self.changes = changes
        self.calls = []
        self.closed = 0

    def close(self):
        self.closed += 1

    def __call__(self, plan, headers):
        self.calls.append((plan, dict(headers)))
        response = azure.AzureResponse(
            200, "text/event-stream" if plan.stream else "application/json",
            plan.url, self.body, self.close,
        )
        return replace(response, **self.changes)


class AzureAdapterTests(unittest.TestCase):
    def client(self, transport, **kwargs):
        return azure.make_azure_inference_client(
            kwargs.pop("config", settings()), transport, kwargs.pop("credentials", lambda: SECRET),
            clock=kwargs.pop("clock", lambda: 0), **kwargs,
        )

    def test_disabled_before_credentials_or_transport(self):
        for flag in ("paid_execution_authorized", "private_auth_verified", "live_transport_verified"):
            transport = FakeTransport()
            credentials = []
            client = self.client(transport, config=settings(**{flag: False}),
                                 credentials=lambda: credentials.append(True))
            for stream in (False, True):
                with self.subTest(flag=flag, stream=stream), self.assertRaises(azure.AzureExecutionBlocked):
                    client(request(stream))
            self.assertEqual(transport.calls, [])
            self.assertEqual(credentials, [])
        defaults = azure.AzureSettings(ORIGIN, MODEL, 30)
        self.assertFalse(defaults.paid_execution_authorized)
        self.assertFalse(defaults.private_auth_verified)
        self.assertFalse(defaults.live_transport_verified)

    def test_origin_allowlist_prevents_url_and_header_injection(self):
        for origin in (
            "http://kova.azurecontainerapps.io", "https://evil.example", "https://127.0.0.1",
            "https://169.254.169.254", "https://user:pass@kova.azurecontainerapps.io",
            ORIGIN + "/", ORIGIN + "/v1", ORIGIN + "?token=bad", ORIGIN + "#bad",
            ORIGIN + ".evil.example", ORIGIN + ":8443", ORIGIN + "\n",
            " https://kova.azurecontainerapps.io", "https://-bad.azurecontainerapps.io",
            "https://foo..azurecontainerapps.io", "https://foo_foo.azurecontainerapps.io",
            "https://FOO.azurecontainerapps.io", "https://foo%2f.azurecontainerapps.io",
        ):
            with self.subTest(origin=origin), self.assertRaises(azure.AzureProtocolError):
                settings(origin=origin)
        self.assertEqual(settings(origin=ORIGIN + ":443").origin, ORIGIN + ":443")

    def test_timeouts_are_explicit_finite_bounded_hop_limits_not_mode_timings(self):
        for timeout in (None, True, False, 0, -1, 240, 600, float("nan"), float("inf"), "30"):
            with self.subTest(timeout=timeout), self.assertRaises(azure.AzureProtocolError):
                settings(timeout_seconds=timeout)
        self.assertEqual(settings(timeout_seconds=239).timeout_seconds, 239)

    def test_guard_types_fail_closed_and_model_names_are_validated(self):
        for value in (1, "true", None):
            with self.assertRaises(azure.AzureProtocolError):
                settings(paid_execution_authorized=value)
        for model in ("", "has space", "x\nheader", "a" * 257, None):
            with self.assertRaises(azure.AzureProtocolError):
                settings(served_model=model)

    def test_direct_json_no_queue_envelope_and_immutable_snapshot(self):
        original = request()
        plan = azure.prepare_request(original, settings())
        original["messages"][0]["content"] = "changed"
        self.assertEqual(plan.url, ORIGIN + "/v1/chat/completions")
        self.assertEqual(json.loads(plan.body)["messages"][0]["content"], "hello")
        self.assertNotIn("input", json.loads(plan.body))
        self.assertFalse(plan.follow_redirects)
        self.assertTrue(plan.verify_tls)
        with self.assertRaises(FrozenInstanceError):
            plan.url = "https://evil.example"
        self.assertNotIn("hello", repr(plan))

    def test_transport_fields_and_wrong_models_fail_before_callbacks(self):
        transport = FakeTransport()
        for field in ("url", "headers", "api_key", "endpoint_id", "input", "openai_route",
                      "base_url", "timeout", "follow_redirects", "verify_tls", "extra_body"):
            with self.subTest(field=field), self.assertRaises(OpenAIProtocolError):
                self.client(transport)(request(**{field: "attacker"}))
        with self.assertRaisesRegex(azure.AzureProtocolError, "model mismatch"):
            self.client(transport)(request(model="different"))
        self.assertEqual(transport.calls, [])

    def test_invalid_json_and_oversized_requests_fail_before_transport(self):
        transport = FakeTransport()
        for value in (float("nan"), object(), "\ud800"):
            with self.assertRaises(azure.AzureProtocolError):
                self.client(transport)(request(temperature=value))
        with patch.object(azure, "MAX_REQUEST_BYTES", 20), self.assertRaisesRegex(azure.AzureProtocolError, "too large"):
            self.client(transport)(request())
        self.assertEqual(transport.calls, [])

    def test_nonstream_response_and_server_credentials(self):
        transport = FakeTransport()
        self.assertEqual(self.client(transport)(request()), payload())
        plan, headers = transport.calls[0]
        self.assertEqual(headers["Authorization"], "Bearer " + SECRET)
        self.assertEqual(headers["Accept"], "application/json")
        self.assertEqual(plan.timeout_seconds, 30)
        self.assertEqual(transport.closed, 1)
        self.assertNotIn(SECRET, repr(plan))

    def test_stream_is_lazy_copied_and_closed_on_completion(self):
        wire = stream_wire()
        transport = FakeTransport([wire[i:i+1] for i in range(len(wire))])
        original = request(True)
        result = self.client(transport)(original)
        self.assertEqual(transport.calls, [])
        original["messages"][0]["content"] = "mutated"
        chunks = list(result)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "Kova café 東京")
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 2)
        self.assertEqual(json.loads(transport.calls[0][0].body)["messages"][0]["content"], "hello")
        self.assertEqual(transport.closed, 1)

    def test_unconsumed_stream_never_obtains_credentials(self):
        calls = []
        transport = FakeTransport(stream_wire())
        result = self.client(transport, credentials=lambda: calls.append(1))(request(True))
        result.close()
        self.assertEqual(calls, [])
        self.assertEqual(transport.calls, [])
        self.assertEqual(transport.closed, 0)

    def test_early_consumer_close_closes_body_and_response(self):
        closed = []
        def source():
            try:
                yield stream_wire()
            finally:
                closed.append(True)
        transport = FakeTransport(source())
        result = self.client(transport)(request(True))
        next(result)
        result.close()
        self.assertEqual(closed, [True])
        self.assertEqual(transport.closed, 1)

    def test_cancellation_before_transport_and_during_stream(self):
        transport = FakeTransport(stream_wire())
        with self.assertRaises(azure.AzureCancelled):
            self.client(transport, cancelled=lambda: True)(request())
        self.assertEqual(transport.calls, [])
        state = [False]
        stream = self.client(transport, cancelled=lambda: state[0])(request(True))
        next(stream)
        state[0] = True
        with self.assertRaises(azure.AzureCancelled):
            next(stream)
        self.assertEqual(transport.closed, 1)

    def test_deadline_includes_credential_time_and_reduces_io_timeout(self):
        clock = [0]
        def credentials():
            clock[0] = 12
            return SECRET
        transport = FakeTransport()
        self.client(transport, clock=lambda: clock[0], credentials=credentials)(request())
        self.assertEqual(transport.calls[0][0].timeout_seconds, 18)
        def expired():
            clock[0] += 31
            return SECRET
        transport = FakeTransport()
        with self.assertRaises(azure.AzureDeadlineExceeded):
            self.client(transport, clock=lambda: clock[0], credentials=expired)(request())
        self.assertEqual(transport.calls, [])

    def test_deadline_after_transport_closes_response(self):
        clock = [0]
        transport = FakeTransport()
        def delayed(plan, headers):
            response = transport(plan, headers)
            clock[0] = 31
            return response
        with self.assertRaises(azure.AzureDeadlineExceeded):
            self.client(delayed, clock=lambda: clock[0])(request())
        self.assertEqual(transport.closed, 1)

    def test_deadline_during_stream_closes_response(self):
        clock = [0]
        transport = FakeTransport(stream_wire())
        result = self.client(transport, clock=lambda: clock[0])(request(True))
        next(result)
        clock[0] = 30
        with self.assertRaises(azure.AzureDeadlineExceeded):
            next(result)
        self.assertEqual(transport.closed, 1)

    def test_invalid_clock_never_reaches_transport(self):
        for value in (float("nan"), float("inf"), True, "time"):
            transport = FakeTransport()
            with self.assertRaises(azure.AzureProtocolError):
                self.client(transport, clock=lambda: value)(request())
            self.assertEqual(transport.calls, [])

    def test_status_errors_redirects_and_content_types_close_without_retry(self):
        for changes in (
            {"status": 301}, {"status": 401}, {"status": 403}, {"status": 429}, {"status": 500},
            {"status": True}, {"status": "200"}, {"final_url": "https://evil.example"},
            {"content_type": "text/html"}, {"content_type": "text/event-stream"},
        ):
            transport = FakeTransport(**changes)
            with self.subTest(changes=changes), self.assertRaises(azure.AzureProtocolError):
                self.client(transport)(request())
            self.assertEqual(len(transport.calls), 1)
            self.assertEqual(transport.closed, 1)
        transport = FakeTransport(content_type="Application/JSON; charset=utf-8")
        self.assertEqual(self.client(transport)(request()), payload())

    def test_bad_credentials_do_not_reach_transport(self):
        for credential in ("", None, "has space", "a\nb", "a\rb", "a" * 16385):
            transport = FakeTransport()
            with self.subTest(credential=str(credential)[:12]), self.assertRaises(azure.AzureProtocolError):
                self.client(transport, credentials=lambda: credential)(request())
            self.assertEqual(transport.calls, [])

    def test_callback_errors_are_sanitized_and_not_retried(self):
        calls = []
        def fail(*_args):
            calls.append(1)
            raise RuntimeError("LEAK: " + SECRET)
        with self.assertRaises(azure.AzureProtocolError) as ctx:
            self.client(fail)(request())
        self.assertEqual(calls, [1])
        self.assertNotIn(SECRET, str(ctx.exception))
        self.assertTrue(ctx.exception.__suppress_context__)
        with self.assertRaises(azure.AzureProtocolError) as ctx:
            self.client(FakeTransport(), credentials=fail)(request())
        self.assertNotIn(SECRET, str(ctx.exception))

    def test_bad_response_json_is_bounded_and_closed(self):
        for body in (b"", b"\xff", b"[]", b'{"x":NaN}', b'{"x":1,"x":2}',
                     b"<html>login</html>", {"already": "decoded"}, [b""]):
            transport = FakeTransport(body)
            with self.subTest(body=body), self.assertRaises(OpenAIProtocolError):
                self.client(transport)(request())
            self.assertEqual(transport.closed, 1)
        transport = FakeTransport(b"x" * 20)
        with patch.object(azure, "MAX_SSE_BYTES", 10), self.assertRaises(OpenAIProtocolError):
            self.client(transport)(request())
        self.assertEqual(transport.closed, 1)

    def test_nonstream_fragmented_unicode_is_preserved(self):
        wire = json.dumps(payload("café 東京"), ensure_ascii=False).encode()
        transport = FakeTransport([wire[i:i+1] for i in range(len(wire))])
        self.assertEqual(self.client(transport)(request()), payload("café 東京"))
        self.assertEqual(transport.closed, 1)

    def test_stream_errors_and_missing_done_close_response(self):
        for wire in (b"data: {}\n\n", b"data: [DONE]\n\n", b"data: bad-json\n\n"):
            transport = FakeTransport(wire)
            with self.subTest(wire=wire), self.assertRaises(OpenAIProtocolError):
                list(self.client(transport)(request(True)))
            self.assertEqual(transport.closed, 1)

    def test_response_model_mismatch_fails_closed(self):
        for stream in (False, True):
            value = {**payload(), "model": "wrong"}
            body = ("data: " + json.dumps(value) + "\n\ndata: [DONE]\n\n").encode() if stream else json.dumps(value).encode()
            transport = FakeTransport(body)
            with self.subTest(stream=stream), self.assertRaisesRegex(azure.AzureProtocolError, "model mismatch"):
                result = self.client(transport)(request(stream))
                if stream:
                    list(result)
            self.assertEqual(transport.closed, 1)

    def test_body_read_errors_are_sanitized_and_closed(self):
        closed = []
        def broken():
            try:
                raise RuntimeError("LEAK " + SECRET)
                yield b"unreachable"
            finally:
                closed.append(True)
        transport = FakeTransport(broken())
        with self.assertRaises(azure.AzureProtocolError) as ctx:
            self.client(transport)(request())
        self.assertNotIn(SECRET, str(ctx.exception))
        self.assertEqual(closed, [True])
        self.assertEqual(transport.closed, 1)

    def test_bad_transport_envelope_is_closed_when_possible(self):
        class Invalid:
            def __init__(self):
                self.closed = False
            def close(self):
                self.closed = True
        response = Invalid()
        with self.assertRaises(azure.AzureProtocolError):
            self.client(lambda *_: response)(request())
        self.assertTrue(response.closed)

    def test_successful_response_with_failed_cleanup_is_not_reported_successful(self):
        def bad_close():
            raise RuntimeError("LEAK " + SECRET)
        transport = FakeTransport(close=bad_close)
        with self.assertRaisesRegex(azure.AzureProtocolError, "cleanup failed") as ctx:
            self.client(transport)(request())
        self.assertNotIn(SECRET, str(ctx.exception))

    def test_cleanup_error_does_not_mask_real_failure(self):
        def bad_close():
            raise RuntimeError("LEAK " + SECRET)
        transport = FakeTransport(status=500, close=bad_close)
        with self.assertRaisesRegex(azure.AzureProtocolError, "HTTP response rejected") as ctx:
            self.client(transport)(request())
        self.assertNotIn(SECRET, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
