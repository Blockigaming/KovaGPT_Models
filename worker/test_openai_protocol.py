"""Provider-independent protocol regressions. All inputs are CPU fixtures."""

import json
import unittest
from unittest.mock import patch

from worker import openai_protocol as protocol
from worker.runpod_vllm import RunPodVllmError, parse_raw_sse as legacy_parse


def sse(value):
    return "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"


class OpenAIProtocolTests(unittest.TestCase):
    def test_request_copy_and_transport_control(self):
        request = {"model": "fixture", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        copied = protocol.validate_chat_request(request)
        request["messages"][0]["content"] = "mutated"
        self.assertEqual(copied["messages"][0]["content"], "hi")
        for field in protocol.FORBIDDEN_TRANSPORT_FIELDS:
            with self.subTest(field=field), self.assertRaises(protocol.OpenAIProtocolError):
                protocol.validate_chat_request({**request, field: "forged"})

    def test_every_byte_split_and_legacy_equivalence(self):
        value = {"choices": [{"delta": {"content": "Kova café 東京"}}]}
        wire = (sse(value) + "data: [DONE]\n\n").encode()
        for split in range(1, len(wire)):
            with self.subTest(split=split):
                fragments = [wire[:split], wire[split:]]
                self.assertEqual(list(protocol.parse_raw_sse(fragments)), [value])
                self.assertEqual(list(legacy_parse(fragments)), [value])
        self.assertEqual(list(protocol.parse_raw_sse([wire[i:i+1] for i in range(len(wire))])), [value])

    def test_large_byte_fragmented_event(self):
        value = {"choices": [{"delta": {"content": "x" * 65536}}]}
        wire = (sse(value) + "data: [DONE]\n\n").encode()
        self.assertEqual(list(protocol.parse_raw_sse(wire[i:i+1] for i in range(len(wire)))), [value])

    def test_crlf_and_usage_and_tool_deltas_preserved(self):
        values = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]}}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
        ]
        wire = ("".join(sse(value) for value in values) + "data: [DONE]\n\n").replace("\n", "\r\n")
        self.assertEqual(list(protocol.parse_raw_sse(list(wire))), values)

    def test_malformed_and_truncated_streams_fail_in_both_adapters(self):
        invalid = [
            [b"\xff"], [b"\xe2"], ["\ud800"], [""], [b""],
            ["data: []\n\ndata: [DONE]\n\n"],
            ["event: message\ndata: {}\n\ndata: [DONE]\n\n"],
            ["data: {}\n\n"], ["data: [DONE]\n\n"],
            ["data: {}\n\ndata: [DONE]"],
            ["data: {}\n\ndata: [DONE]\n\ndata: {}\n\n"],
            ["data: {}\n\ndata: [DONE]\n\ndata: [DONE]\n\n"],
            ["data: {}\rX\ndata: [DONE]\n\n"],
        ]
        for fragments in invalid:
            with self.subTest(fragments=fragments):
                with self.assertRaises(protocol.OpenAIProtocolError):
                    list(protocol.parse_raw_sse(fragments))
                with self.assertRaises(RunPodVllmError):
                    list(legacy_parse(fragments))

    def test_nonfinite_duplicate_and_deep_json_rejected(self):
        for payload in ('{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}', '[' * 2000):
            with self.subTest(payload=payload[:30]), self.assertRaises(protocol.OpenAIProtocolError):
                list(protocol.parse_raw_sse([f"data: {payload}\n\n", "data: [DONE]\n\n"]))

    def test_error_details_do_not_echo_raw_provider_messages(self):
        for detail in ("Bearer secret goes here", "x" * 1000, "secret\nheader", "alphanumericSecret123", {}, None):
            value = {"error": {"type": detail, "message": "PRIVATE PROMPT OR TOKEN"}}
            with self.subTest(detail=detail), self.assertRaisesRegex(protocol.OpenAIProtocolError, "unknown_error") as ctx:
                protocol.decode_response(value, expect_stream=False)
            self.assertNotIn("PRIVATE", str(ctx.exception))
            self.assertNotIn("secret", str(ctx.exception))
        with self.assertRaisesRegex(protocol.OpenAIProtocolError, "startup_error"):
            protocol.decode_response({"error": {"type": "startup_error"}}, expect_stream=False)

    def test_total_event_and_count_bounds(self):
        with patch.object(protocol, "MAX_SSE_BYTES", 16):
            with self.assertRaisesRegex(protocol.OpenAIProtocolError, "response too large"):
                list(protocol.parse_raw_sse([b"x" * 17]))
        with patch.object(protocol, "MAX_SSE_EVENT_BYTES", 16):
            with self.assertRaisesRegex(protocol.OpenAIProtocolError, "buffer too large"):
                list(protocol.parse_raw_sse(["x" * 17]))
        with patch.object(protocol, "MAX_SSE_EVENTS", 2):
            with self.assertRaisesRegex(protocol.OpenAIProtocolError, "too many"):
                list(protocol.parse_raw_sse(["data: {}\n\n" * 3, "data: [DONE]\n\n"]))

    def test_hidden_reasoning_is_not_silently_laundered_by_protocol(self):
        value = {"choices": [{"delta": {"reasoning_content": "private", "content": "answer"}}]}
        self.assertEqual(list(protocol.parse_raw_sse([sse(value), "data: [DONE]\n\n"])), [value])

    def test_nonstream_exactly_one_object(self):
        value = {"choices": []}
        for output in (value, [value]):
            self.assertEqual(protocol.decode_response(output, expect_stream=False), value)
        for output in (None, [], [value, value], "text", b"bytes"):
            with self.subTest(output=output), self.assertRaises(protocol.OpenAIProtocolError):
                protocol.decode_response(output, expect_stream=False)


if __name__ == "__main__":
    unittest.main()
