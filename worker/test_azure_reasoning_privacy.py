"""CPU-only Azure reasoning-output suppression regressions; no model calls."""

from copy import deepcopy
from dataclasses import replace
import json
import unittest

from worker.azure_container_apps import (
    AzureExecutionBlocked, AzureResponse, AzureSettings, make_azure_inference_client,
    prepare_request,
)
from worker.openai_protocol import OpenAIProtocolError


SETTINGS = AzureSettings(
    "https://fixture.internal.test.azurecontainerapps.io", "fixture-model", 10,
    paid_execution_authorized=True, private_auth_verified=True, live_transport_verified=True,
)
PRIVATE = "SYNTHETIC_PRIVATE_REASONING_DO_NOT_RETURN"


def request(stream=False, thinking=True):
    return {
        "model": SETTINGS.served_model,
        "messages": [{"role": "user", "content": "Explain the fixture."}],
        "stream": stream, "max_tokens": 128, "reasoning_effort": "medium",
        "chat_template_kwargs": {"enable_thinking": thinking, "preserve_thinking": False},
    }


def response(stream=False):
    return {
        "model": SETTINGS.served_model,
        "choices": [{"index": 0, "delta" if stream else "message": {"content": "Kova answer"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8,
                  "completion_tokens_details": {"reasoning_tokens": 5}},
    }


def fixture(value, *, stream=False):
    closed = []
    def transport(plan, _headers):
        wire = json.dumps(value)
        if stream:
            wire = "data: " + wire + "\n\ndata: [DONE]\n\n"
        return AzureResponse(200, "text/event-stream" if stream else "application/json",
                             plan.url, wire.encode(), lambda: closed.append(True))
    client = make_azure_inference_client(SETTINGS, transport, lambda: "synthetic.fixture.token")
    return client, closed


class AzureReasoningPrivacyTests(unittest.TestCase):
    def assert_rejected(self, value, stream, message=None):
        client, closed = fixture(value, stream=stream)
        with self.assertRaises(OpenAIProtocolError) as caught:
            result = client(request(stream))
            if stream:
                list(result)
        if message:
            self.assertIn(message, str(caught.exception))
        self.assertNotIn(PRIVATE, str(caught.exception))
        self.assertEqual(closed, [True])

    def test_requests_suppress_reasoning_without_disabling_thinking(self):
        for stream in (False, True):
            for thinking in (False, True):
                source = request(stream, thinking)
                before = deepcopy(source)
                with self.subTest(stream=stream, thinking=thinking):
                    body = json.loads(prepare_request(source, SETTINGS).body)
                    self.assertIs(body.get("include_reasoning"), False)
                    self.assertEqual(body["chat_template_kwargs"], before["chat_template_kwargs"])
                    self.assertEqual(body["reasoning_effort"], "medium")
                    self.assertEqual(body["max_tokens"], 128)
                    self.assertEqual(source, before)

    def test_caller_cannot_override_server_reasoning_suppression(self):
        for value in (True, False, None, "false"):
            with self.subTest(value=value), self.assertRaises(OpenAIProtocolError):
                prepare_request({**request(), "include_reasoning": value}, SETTINGS)

    def test_both_reasoning_field_names_are_rejected_before_nonstream_return(self):
        for field in ("reasoning", "reasoning_content"):
            value = response()
            value["choices"][0]["message"][field] = PRIVATE
            with self.subTest(field=field):
                self.assert_rejected(value, False, "hidden reasoning")

    def test_both_reasoning_field_names_are_rejected_before_stream_yield(self):
        for field in ("reasoning", "reasoning_content"):
            value = response(True)
            value["choices"][0]["delta"][field] = PRIVATE
            client, closed = fixture(value, stream=True)
            stream = client(request(True))
            with self.subTest(field=field):
                try:
                    with self.assertRaisesRegex(OpenAIProtocolError, "hidden reasoning") as caught:
                        next(stream)
                    self.assertNotIn(PRIVATE, str(caught.exception))
                    self.assertEqual(closed, [True])
                finally:
                    stream.close()

    def test_nonstring_reasoning_values_do_not_pass_as_empty(self):
        for field in ("reasoning", "reasoning_content"):
            for bad in (False, 0, [], {}):
                value = response(True)
                value["choices"][0]["delta"][field] = bad
                with self.subTest(field=field, bad=bad):
                    self.assert_rejected(value, True)

    def test_empty_reasoning_and_numeric_usage_remain_valid(self):
        for stream in (False, True):
            value = response(stream)
            part = value["choices"][0]["delta" if stream else "message"]
            part.update(reasoning=None, reasoning_content="")
            client, closed = fixture(value, stream=stream)
            result = client(request(stream))
            if stream:
                result = list(result)[0]
            self.assertEqual(result, value)
            self.assertEqual(result["usage"]["completion_tokens_details"]["reasoning_tokens"], 5)
            self.assertEqual(closed, [True])

    def test_reasoning_side_channels_are_rejected_at_protocol_locations(self):
        for stream in (False, True):
            for field in ("logprobs", "prompt_logprobs", "token_ids", "prompt_token_ids"):
                for location in ("root", "choice", "message"):
                    value = response(stream)
                    target = value if location == "root" else value["choices"][0]
                    if location == "message":
                        target = target["delta" if stream else "message"]
                    target[field] = [PRIVATE]
                    with self.subTest(stream=stream, field=field, location=location):
                        self.assert_rejected(value, stream, "private token metadata")

    def test_null_logprob_metadata_is_compatible(self):
        for stream in (False, True):
            value = response(stream)
            value["choices"][0]["logprobs"] = None
            client, closed = fixture(value, stream=stream)
            result = client(request(stream))
            self.assertEqual(list(result)[0] if stream else result, value)
            self.assertEqual(closed, [True])

    def test_visible_explanations_and_tool_argument_words_are_not_reasoning_fields(self):
        value = response()
        value["choices"][0]["message"]["content"] = "My explanation uses the word reasoning."
        value["choices"][0]["message"]["tool_calls"] = [{
            "id": "call_1", "type": "function",
            "function": {"name": "search", "arguments": '{"reasoning":"a search term"}'},
        }]
        value["choices"][0]["finish_reason"] = "tool_calls"
        client, closed = fixture(value)
        self.assertEqual(client(request()), value)
        self.assertEqual(closed, [True])

    def test_late_private_chunk_fails_and_releases_response(self):
        first = response(True)
        first["choices"][0]["finish_reason"] = None
        second = response(True)
        second["choices"][0]["delta"] = {"reasoning": PRIVATE}
        closed = []
        def transport(plan, _headers):
            wire = "".join("data: " + json.dumps(v) + "\n\n" for v in (first, second))
            return AzureResponse(200, "text/event-stream", plan.url,
                                 (wire + "data: [DONE]\n\n").encode(), lambda: closed.append(True))
        stream = make_azure_inference_client(SETTINGS, transport, lambda: "fixture.token")(request(True))
        try:
            self.assertEqual(next(stream), first)
            with self.assertRaisesRegex(OpenAIProtocolError, "hidden reasoning"):
                next(stream)
            self.assertEqual(closed, [True])
        finally:
            stream.close()

    def test_default_execution_guards_still_block_before_callbacks(self):
        for field in ("paid_execution_authorized", "private_auth_verified", "live_transport_verified"):
            def forbidden(*_args):
                self.fail("disabled request called a credential or transport dependency")
            client = make_azure_inference_client(replace(SETTINGS, **{field: False}), forbidden, forbidden)
            with self.subTest(field=field), self.assertRaises(AzureExecutionBlocked):
                client(request(True))


if __name__ == "__main__":
    unittest.main()
