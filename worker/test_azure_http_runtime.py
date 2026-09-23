"""Concrete client authorization and managed-identity contract tests, CPU only."""

from dataclasses import replace
import json
import os
import socket
import ssl
from urllib.parse import parse_qs, urlsplit
import unittest
from unittest.mock import patch

from worker import azure_http_runtime as runtime
from worker.azure_container_apps import (
    AzureCancelled, AzureDeadlineExceeded, AzureExecutionBlocked, AzureProtocolError,
    AzureSettings, make_azure_inference_client, prepare_request,
)
from worker import test_bounded_http as fixtures
from worker.test_bounded_http import HOST


CLIENT_ID = "00000000-0000-4000-8000-000000000001"
RESOURCE = "api://00000000-0000-4000-8000-000000000002"
TOKEN = "synthetic.access-token.not-live"
HEADER = "synthetic-rotating-identity-header"
NOW = 1_900_000_000
ORIGIN = "https://" + HOST
ENVIRONMENT = {"IDENTITY_ENDPOINT": "http://127.0.0.1:42356/msi/token", "IDENTITY_HEADER": HEADER}


def config(**changes):
    return replace(runtime.AzureRuntimeConfig(
        AzureSettings(ORIGIN, "fixture-model", 10, True, True, True),
        "127.0.0.1", RESOURCE, CLIENT_ID, True,
    ), **changes)


def token_response(**changes):
    return {"access_token": TOKEN, "token_type": "Bearer", "resource": RESOURCE,
            "client_id": CLIENT_ID, "expires_on": str(NOW + 1000), **changes}


def engine_request(stream=False):
    return {"model": "fixture-model", "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 20, "stream": stream}


class MemoryBody:
    def __init__(self, value, status=200, content_type="application/json"):
        self.value = json.dumps(value).encode() if isinstance(value, dict) else value
        self.status = status
        self.content_type = content_type
        self.closed = False

    def __iter__(self):
        return iter((self.value,))

    def close(self):
        self.closed = True


class AzureHTTPRuntimeTests(unittest.TestCase):
    def credential(self, selected=None, **kwargs):
        return runtime.ManagedIdentityCredential(
            selected or config(), environment=kwargs.pop("environment", lambda: ENVIRONMENT),
            wall_clock=kwargs.pop("wall_clock", lambda: NOW), **kwargs,
        )

    def test_runtime_construction_and_disabled_calls_make_no_environment_or_network_access(self):
        with patch.object(runtime, "open_http_response", side_effect=AssertionError("network called")):
            for flag in ("paid_execution_authorized", "private_auth_verified", "live_transport_verified", "network_execution_authorized"):
                original = config()
                disabled = (replace(original, network_execution_authorized=False) if flag == "network_execution_authorized"
                            else replace(original, settings=replace(original.settings, **{flag: False})))
                touched = []
                client = runtime.make_azure_runtime_client(disabled, environment=lambda: touched.append(True))
                for stream in (False, True):
                    with self.subTest(flag=flag, stream=stream), self.assertRaises(AzureExecutionBlocked):
                        result = client(engine_request(stream))
                        if stream:
                            next(result)
                with self.assertRaises(AzureExecutionBlocked):
                    self.credential(disabled, environment=lambda: touched.append(True))()
                self.assertEqual(touched, [])

    def test_runtime_requires_explicit_numeric_target_resource_and_boolean_grant(self):
        for changes in (
            {"destination_address": "lookup.example"}, {"destination_address": "0.0.0.0"},
            {"network_execution_authorized": "true"}, {"resource": ""},
            {"resource": "https://api.example/.default"}, {"resource": "http://api.example"},
            {"resource": "https://user:password@example.com"}, {"resource": "api://x?y=1"},
            {"resource": "api://x#bad"}, {"resource": "api://x\r\n"},
            {"identity_client_id": "not-a-uuid"}, {"settings": {}},
        ):
            with self.subTest(changes=changes), self.assertRaises(AzureProtocolError):
                config(**changes)

    def test_identity_query_resource_version_user_identity_and_header_are_exact(self):
        body = MemoryBody(token_response())
        with patch.object(runtime, "open_http_response", return_value=body) as transport:
            self.assertEqual(self.credential()(), TOKEN)
        kwargs = transport.call_args.kwargs
        self.assertEqual(kwargs["method"], "GET")
        self.assertFalse(kwargs["tls"])
        self.assertEqual(kwargs["host"], "127.0.0.1")
        self.assertEqual(kwargs["port"], 42356)
        query = parse_qs(urlsplit(kwargs["target"]).query)
        self.assertEqual(query, {"resource": [RESOURCE], "api-version": ["2019-08-01"], "client_id": [CLIENT_ID]})
        self.assertEqual(kwargs["headers"], {"X-IDENTITY-HEADER": HEADER, "Accept": "application/json"})
        self.assertEqual(kwargs["max_response_bytes"], 65536)
        self.assertNotIn("Authorization", kwargs["headers"])
        self.assertTrue(body.closed)

    def test_system_assigned_identity_is_explicit_without_a_client_id_fallback(self):
        body = MemoryBody(token_response())
        with patch.object(runtime, "open_http_response", return_value=body) as transport:
            self.assertEqual(self.credential(config(identity_client_id=None))(), TOKEN)
        self.assertNotIn("client_id", parse_qs(urlsplit(transport.call_args.kwargs["target"]).query))

    def test_rotating_identity_header_is_reread_without_a_token_cache(self):
        current = dict(ENVIRONMENT)
        with patch.object(runtime, "open_http_response", side_effect=lambda **_: MemoryBody(token_response())) as transport:
            provider = self.credential(environment=lambda: current)
            provider()
            current["IDENTITY_HEADER"] = "new-rotated-header"
            provider()
            self.assertEqual(transport.call_count, 2)
            self.assertEqual(transport.call_args_list[0].kwargs["headers"]["X-IDENTITY-HEADER"], HEADER)
            self.assertEqual(transport.call_args_list[1].kwargs["headers"]["X-IDENTITY-HEADER"], "new-rotated-header")

    def test_missing_platform_identity_has_no_cli_key_or_imds_fallback(self):
        for env in ({}, {"AZURE_CLIENT_SECRET": "must-not-be-used"},
                    {"IDENTITY_ENDPOINT": ENVIRONMENT["IDENTITY_ENDPOINT"]},
                    {"IDENTITY_HEADER": HEADER}, None):
            with self.subTest(env=env), patch.object(runtime, "open_http_response") as transport:
                with self.assertRaises(AzureProtocolError):
                    self.credential(environment=lambda: env)()
                transport.assert_not_called()

    def test_identity_endpoint_rejects_external_dns_credentials_query_or_path_injection(self):
        for endpoint in (
            "http://evil.example/msi/token", "http://8.8.8.8/msi/token", "https://127.0.0.1/msi/token",
            "http://user:pass@127.0.0.1/msi/token", "http://127.0.0.1/msi/token?resource=evil",
            "http://127.0.0.1/msi/token#fragment", "http://127.0.0.1//evil", "http://127.0.0.1/a/../msi/token",
            "http://127.0.0.1/msi/%74oken", "http://127.0.0.1:99999/msi/token", "http://127.0.0.1:0/msi/token",
            "http://127.0.0.1/msi/token\r\n", "http://localhost/msi/token", "http://0.0.0.0/msi/token",
        ):
            with self.subTest(endpoint=endpoint), patch.object(runtime, "open_http_response") as transport:
                with self.assertRaises(AzureProtocolError):
                    self.credential(environment=lambda: dict(ENVIRONMENT, IDENTITY_ENDPOINT=endpoint))()
                transport.assert_not_called()

    def test_identity_header_rejects_injection_and_oversize(self):
        for header in (None, "", "\r\nAuthorization: bad", "has space", "x" * 16385):
            with self.subTest(header=str(header)[:15]), patch.object(runtime, "open_http_response") as transport:
                with self.assertRaises(AzureProtocolError):
                    self.credential(environment=lambda: dict(ENVIRONMENT, IDENTITY_HEADER=header))()
                transport.assert_not_called()

    def test_wrong_audience_client_type_expiry_and_token_are_rejected_and_closed(self):
        cases = (
            {"resource": "https://cognitiveservices.azure.com/"}, {"client_id": None},
            {"client_id": "different"}, {"token_type": "Basic"}, {"access_token": ""},
            {"access_token": "has\nnewline"}, {"access_token": "x" * 16385},
            {"expires_on": str(NOW - 1)}, {"expires_on": str(NOW + 40)},
            {"expires_on": True}, {"expires_on": 1.5}, {"expires_on": None},
            {"expires_on": "NaN"}, {"expires_on": "999999999999999999999999"},
        )
        for changes in cases:
            body = MemoryBody(token_response(**changes))
            with self.subTest(changes=changes), patch.object(runtime, "open_http_response", return_value=body):
                with self.assertRaises(AzureProtocolError):
                    self.credential()()
                self.assertTrue(body.closed)

    def test_identity_malformed_and_secret_bearing_failures_are_redacted(self):
        cases = (b"{not-json-secret", b'{"access_token":"SECRET","access_token":"other"}',
                 b'{"expires_on":NaN}', b"\xff", b"[]",
                 {"error": "secret upstream details " + TOKEN}, b"x" * 65537)
        for value in cases:
            body = MemoryBody(value)
            with self.subTest(value=str(value)[:40]), patch.object(runtime, "open_http_response", return_value=body):
                with self.assertRaises(AzureProtocolError) as raised:
                    self.credential()()
                self.assertNotIn(TOKEN, str(raised.exception))
                self.assertNotIn("SECRET", str(raised.exception))
                self.assertTrue(body.closed)

    def test_identity_http_status_and_media_type_rejected(self):
        for body in (MemoryBody(token_response(), status=302), MemoryBody(token_response(), content_type="text/event-stream")):
            with patch.object(runtime, "open_http_response", return_value=body):
                with self.assertRaises(AzureProtocolError):
                    self.credential()()
                self.assertTrue(body.closed)

    def test_inference_transport_sends_only_bearer_to_fixed_https_target(self):
        response = MemoryBody({"choices": []})
        plan = prepare_request(engine_request(), config().settings)
        headers = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json", "Accept": "application/json"}
        with patch.object(runtime, "open_http_response", return_value=response) as exchange:
            result = runtime.AzureHTTPTransport(config())(plan, headers)
            options = exchange.call_args.kwargs
            self.assertEqual(options["host"], HOST)
            self.assertEqual(options["address"], "127.0.0.1")
            self.assertEqual(options["port"], 443)
            self.assertEqual(options["target"], "/v1/chat/completions")
            self.assertNotIn("X-IDENTITY-HEADER", options["headers"])
            self.assertEqual(json.loads(options["body"])["model"], "fixture-model")
            self.assertNotIn("input", json.loads(options["body"]))
            result.close()
            self.assertTrue(response.closed)

    def test_transport_rejects_control_changes_without_contacting_network(self):
        plan = prepare_request(engine_request(), config().settings)
        headers = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json", "Accept": "application/json"}
        for changes in ({"url": "https://evil.example/v1/chat/completions"}, {"follow_redirects": True},
                        {"verify_tls": False}, {"method": "GET"}, {"timeout_seconds": 11},
                        {"timeout_seconds": float("nan")}):
            with self.subTest(changes=changes), patch.object(runtime, "open_http_response") as exchange:
                with self.assertRaises(AzureProtocolError):
                    runtime.AzureHTTPTransport(config())(replace(plan, **changes), headers)
                exchange.assert_not_called()
        for bad_headers in ({**headers, "X-IDENTITY-HEADER": HEADER}, {**headers, "Authorization": "Bearer "},
                            {**headers, "Accept": "text/html"}):
            with patch.object(runtime, "open_http_response") as exchange:
                with self.assertRaises(AzureProtocolError):
                    runtime.AzureHTTPTransport(config())(plan, bad_headers)
                exchange.assert_not_called()

    def test_deadline_and_cancellation_remain_typed_across_credential_transport_and_body(self):
        from worker.azure_container_apps import AzureResponse
        for failure in (AzureCancelled, AzureDeadlineExceeded):
            def fail(*_, **__):
                raise failure("synthetic controlled termination")
            for phase in ("credential", "transport", "body"):
                closed = []
                def body():
                    fail()
                    yield b"unreachable"
                def transport(plan, _headers):
                    return AzureResponse(200, "text/event-stream", plan.url, body(), lambda: closed.append(1))
                client = make_azure_inference_client(
                    config().settings, fail if phase == "transport" else transport,
                    fail if phase == "credential" else lambda: TOKEN,
                )
                with self.subTest(failure=failure.__name__, phase=phase), self.assertRaises(failure):
                    list(client(engine_request(True)))
                self.assertEqual(closed, [1] if phase == "body" else [])

    def test_unconsumed_concrete_stream_never_reads_identity_environment(self):
        calls = []
        with patch.object(runtime, "open_http_response") as exchange:
            client = runtime.make_azure_runtime_client(config(), environment=lambda: calls.append(1))
            stream = client(engine_request(True))
            stream.close()
            self.assertEqual(calls, [])
            exchange.assert_not_called()

    def test_full_client_uses_identity_then_model_and_closes_both(self):
        identity = MemoryBody(token_response(expires_on="9999999999"))
        model = MemoryBody({"model": "fixture-model", "choices": [{"message": {"content": "Kova"}}]})
        with patch.object(runtime, "open_http_response", side_effect=[identity, model]) as exchange:
            client = runtime.make_azure_runtime_client(config(), environment=lambda: ENVIRONMENT)
            self.assertEqual(client(engine_request())["choices"][0]["message"]["content"], "Kova")
            self.assertEqual([item.kwargs["method"] for item in exchange.call_args_list], ["GET", "POST"])
            self.assertTrue(identity.closed)
            self.assertTrue(model.closed)

    def test_network_default_is_disabled_even_when_other_settings_are_verified(self):
        selected = runtime.AzureRuntimeConfig(config().settings, "127.0.0.1", RESOURCE)
        self.assertFalse(selected.network_execution_authorized)
        with patch.object(runtime, "open_http_response") as exchange:
            with self.assertRaises(AzureExecutionBlocked):
                selected.require_authorized()
            exchange.assert_not_called()


class AzureRuntimeLoopbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse temporary-certificate fixture setup, not its test class discovery.
        fixtures.BoundedHTTPTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        fixtures.BoundedHTTPTests.tearDownClass()

    def test_end_to_end_identity_rest_then_verified_tls_chat_stream(self):
        from worker import bounded_http
        from worker.handler import PINNED_CORE_CANDIDATES, handle_job
        candidate = PINNED_CORE_CANDIDATES["kova-cosmo"]
        selected = config(settings=replace(config().settings, served_model=candidate["model"]))
        identity_bytes = json.dumps(token_response(expires_on="9999999999")).encode()
        identity_wire = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                         + str(len(identity_bytes)).encode() + b"\r\n\r\n" + identity_bytes)
        chunks = [
            {"model": candidate["model"], "choices": [{"delta": {"content": "Kova café 東京"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
        ]
        sse = ("".join("data: " + json.dumps(item, ensure_ascii=False) + "\n\n" for item in chunks)
               + "data: [DONE]\n\n").encode()
        model_wire = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                      + f"{len(sse):x}\r\n".encode() + sse + b"\r\n0\r\n\r\n")
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(fixtures.BoundedHTTPTests.cert, fixtures.BoundedHTTPTests.key)
        original_context = ssl.create_default_context
        def client_context():
            context = original_context(cafile=fixtures.BoundedHTTPTests.cert)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            return context
        with fixtures.FixtureServer(fixtures.reply(identity_wire)) as identity_server, \
                fixtures.FixtureServer(fixtures.reply(model_wire), tls) as model_server:
            class FixtureSocket(socket.socket):
                def connect_ex(self, destination):
                    if destination == ("127.0.0.1", 443):
                        destination = ("127.0.0.1", model_server.port)
                    if destination not in (("127.0.0.1", identity_server.port), ("127.0.0.1", model_server.port)):
                        raise AssertionError("test tried non-fixture networking")
                    return super().connect_ex(destination)
            environment = dict(ENVIRONMENT, IDENTITY_ENDPOINT=f"http://127.0.0.1:{identity_server.port}/msi/token")
            with patch.object(bounded_http, "_SOCKET_FACTORY", FixtureSocket), \
                    patch.object(ssl, "create_default_context", client_context), \
                    patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS called")), \
                    patch.dict(os.environ, {"HTTPS_PROXY": "http://must-not-be-used.invalid:1"}):
                client = runtime.make_azure_runtime_client(selected, environment=lambda: environment)
                records = []
                probe = {
                    "source": "server_provider_runtime", "worker_lifecycle_id": "loopback-fixture",
                    "loaded_model": candidate["model"], "loaded_model_revision": candidate["model_revision"],
                    "cold_start": False, "worker_start_ms": 0, "model_load_ms": 0, "queue_ms": 0,
                    "gpu_rate_per_second_usd": 0.001, "gpu_type_id": "synthetic-no-gpu", "gpu_count": 1,
                    "serving_engine": "vllm", "endpoint_type": "load_balancing",
                    "container_image_digest": "sha256:" + "a" * 64,
                }
                result = handle_job(
                    {"input": {"request_id": "loopback-fixture", "messages": [{"role": "user", "content": "hello"}],
                               "reasoning_effort": "low", "max_output_tokens": 2048}},
                    client, lambda _phase: dict(probe), records.append,
                    execution_context={
                        "logical_request_id": "kova-exec-00000000-0000-4000-8000-000000000001",
                        "benchmark_candidate_id": candidate["id"], "route_id": "instant",
                        "stage_id": "answer-1", "public_response": True, "prior_stage_outputs": {},
                    }, token_counter=lambda *_: 10,
                )
                self.assertEqual(result["content"], "Kova café 東京")
                self.assertEqual(records[0]["outcome"], "success")
                self.assertIn(b"X-IDENTITY-HEADER: " + HEADER.encode(), identity_server.request)
                self.assertNotIn(b"Authorization:", identity_server.request)
                self.assertIn(b"Authorization: Bearer " + TOKEN.encode(), model_server.request)
                self.assertNotIn(HEADER.encode(), model_server.request)
                self.assertIn(f"Host: {HOST}\r\n".encode(), model_server.request)


if __name__ == "__main__":
    unittest.main()
