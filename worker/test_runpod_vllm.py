import json
from pathlib import Path
import unittest

from worker.handler import PINNED_CORE_CANDIDATES, handle_job
from worker.runpod_vllm import (
    MAX_SSE_EVENT_BYTES,
    MAX_SSE_EVENTS,
    RunPodVllmError,
    build_queue_job,
    decode_worker_output,
    make_queue_inference_client,
    parse_raw_sse,
)


FIXTURES = Path(__file__).with_name("fixtures")


def fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class Clock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class RunPodVllmAdapterTests(unittest.TestCase):
    candidate = PINNED_CORE_CANDIDATES["kova-cosmo"]
    digest = "sha256:" + "b" * 64

    def setUp(self):
        self.original_pin = self.candidate["adapter_sha256"]
        self.original_bundle = self.candidate["adapter_bundle_sha256"]
        self.candidate["adapter_sha256"] = "f" * 64
        self.candidate["adapter_bundle_sha256"] = "d" * 64
        self.addCleanup(self.candidate.update, adapter_sha256=self.original_pin)
        self.addCleanup(self.candidate.update, adapter_bundle_sha256=self.original_bundle)

    def engine_request(self, **overrides):
        value = {
            "model": self.candidate["model"],
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        value.update(overrides)
        return value

    def runtime_probe(self, phase):
        self.assertIn(phase, ("before", "after"))
        return {
            "source": "server_provider_runtime",
            "worker_lifecycle_id": "fixture-lifecycle",
            "loaded_model": self.candidate["model"],
            "loaded_model_revision": self.candidate["model_revision"],
            "loaded_adapter_sha256": self.candidate["adapter_sha256"],
            "loaded_adapter_bundle_sha256": self.candidate["adapter_bundle_sha256"],
            "cold_start": False,
            "worker_start_ms": 0,
            "model_load_ms": 0,
            "queue_ms": 1,
            "gpu_rate_per_second_usd": 0.001,
            "gpu_type_id": "fixture-gpu",
            "gpu_count": 1,
            "serving_engine": "vllm",
            "endpoint_type": "queue_based",
            "container_image_digest": self.digest,
        }

    def test_builds_exact_openai_queue_shape_and_copies_input(self):
        request = self.engine_request()
        job = build_queue_job(request)
        self.assertEqual(set(job), {"input"})
        self.assertEqual(job["input"]["openai_route"], "/v1/chat/completions")
        self.assertEqual(job["input"]["openai_input"], request)
        request["messages"][0]["content"] = "changed"
        self.assertEqual(job["input"]["openai_input"]["messages"][0]["content"], "hello")

    def test_queue_shape_rejects_untrusted_transport_fields_and_missing_contract(self):
        for field in ("api_key", "endpoint_id", "openai_route", "input", "body"):
            with self.subTest(field=field), self.assertRaisesRegex(
                RunPodVllmError, "transport-controlled"
            ):
                build_queue_job(self.engine_request(**{field: "untrusted"}))
        for request in ({}, {"model": "m", "messages": [], "stream": "yes"}):
            with self.assertRaises(RunPodVllmError):
                build_queue_job(request)

    def test_reconstructs_fragmented_stream_and_preserves_usage(self):
        chunks = list(parse_raw_sse(fixture("runpod_vllm_stream_success.json")))
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "Kova")
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 2)
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")

    def test_decodes_utf8_split_across_byte_fragments(self):
        encoded = (
            'data: {"choices":[{"delta":{"content":"café 東京"}}]}\n\n'
            "data: [DONE]\n\n"
        ).encode("utf-8")
        chunks = list(parse_raw_sse([encoded[index : index + 1] for index in range(len(encoded))]))
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "café 東京")

    def test_parses_large_event_from_byte_sized_fragments_without_rescanning(self):
        content = "x" * (64 * 1024)
        encoded = (
            "data: "
            + json.dumps({"choices": [{"delta": {"content": content}}]}, separators=(",", ":"))
            + "\n\ndata: [DONE]\n\n"
        ).encode("utf-8")
        chunks = list(parse_raw_sse(encoded[index : index + 1] for index in range(len(encoded))))
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], content)

    def test_accepts_crlf_split_across_fragments(self):
        chunks = list(
            parse_raw_sse(
                ['data: {"choices": []}\r', "\n\r", "\ndata: [DONE]\r", "\n\r", "\n"]
            )
        )
        self.assertEqual(chunks, [{"choices": []}])

    def test_preserves_tool_call_fragments_without_interpreting_them(self):
        fragments = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_","type":"function","function":{"name":"look","arguments":"{\\"q\\":"}}]}}]}\n\n',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"1","function":{"name":"up","arguments":"\\"Kova\\"}"}}]},"finish_reason":null}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n',
            "data: [DONE]\n\n",
        ]
        chunks = list(parse_raw_sse(fragments))
        self.assertEqual(chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"], "call_")
        self.assertEqual(chunks[1]["choices"][0]["delta"]["tool_calls"][0]["id"], "1")

    def test_hidden_reasoning_is_preserved_for_kova_sanitizer_to_reject(self):
        fragments = [
            'data: {"choices":[{"delta":{"reasoning_content":"secret","content":"answer"}}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n',
            "data: [DONE]\n\n",
        ]
        parsed = list(parse_raw_sse(fragments))
        self.assertEqual(parsed[0]["choices"][0]["delta"]["reasoning_content"], "secret")

    def test_rejects_worker_errors_in_streaming_and_nonstreaming_output(self):
        error = fixture("runpod_vllm_startup_error.json")
        with self.assertRaisesRegex(RunPodVllmError, "startup_error"):
            decode_worker_output(error, expect_stream=False)
        encoded = json.dumps(error, separators=(",", ":"))
        with self.assertRaisesRegex(RunPodVllmError, "startup_error"):
            list(parse_raw_sse([f"data: {encoded}\n\n", "data: [DONE]\n\n"]))

    def test_rejects_invalid_stream_encodings_and_payloads(self):
        cases = (
            [b"\xff"],
            [b"\xe2\x82"],
            ["\ud800"],
            ["event: message\ndata: {}\n\ndata: [DONE]\n\n"],
            ["data: not-json\n\ndata: [DONE]\n\n"],
            ["data: []\n\ndata: [DONE]\n\n"],
            ["data: [DONE]\n\n"],
        )
        for fragments in cases:
            with self.subTest(fragments=fragments), self.assertRaises(RunPodVllmError):
                list(parse_raw_sse(fragments))

    def test_rejects_empty_stream_fragments_without_waiting_for_progress(self):
        def stalled():
            while True:
                yield b""

        with self.assertRaisesRegex(RunPodVllmError, "must not be empty"):
            list(parse_raw_sse(stalled()))

    def test_requires_one_final_terminal_marker_and_complete_events(self):
        cases = (
            ["data: {}\n\n"],
            ["data: {}\n\ndata: [DONE]\n\ndata: [DONE]\n\n"],
            ["data: {}\n\ndata: [DONE]\n\ndata: {}\n\n"],
            ["data: {}\n\ndata: [DONE]"],
        )
        for fragments in cases:
            with self.subTest(fragments=fragments), self.assertRaises(RunPodVllmError):
                list(parse_raw_sse(fragments))

    def test_enforces_event_and_buffer_limits(self):
        self.assertGreaterEqual(MAX_SSE_EVENTS, 24_576 + 3)
        too_many = ["data: {}\n\n" * (MAX_SSE_EVENTS + 1), "data: [DONE]\n\n"]
        with self.assertRaisesRegex(RunPodVllmError, "too many"):
            list(parse_raw_sse(too_many))
        with self.assertRaisesRegex(RunPodVllmError, "buffer too large"):
            list(parse_raw_sse(["x" * (MAX_SSE_EVENT_BYTES + 1)]))

        many_valid_events = "data: {}\n\n" * 220_000
        self.assertGreater(len(many_valid_events), MAX_SSE_EVENT_BYTES)
        with self.assertRaisesRegex(RunPodVllmError, "too many"):
            list(parse_raw_sse([many_valid_events, "data: [DONE]\n\n"]))

    def test_decodes_one_nonstream_object_and_rejects_ambiguous_outputs(self):
        response = fixture("runpod_vllm_nonstream_success.json")
        self.assertEqual(decode_worker_output(response, expect_stream=False), response)
        self.assertEqual(decode_worker_output([response], expect_stream=False), response)
        for invalid in ([], [response, response], "text"):
            with self.subTest(invalid=invalid), self.assertRaises(RunPodVllmError):
                decode_worker_output(invalid, expect_stream=False)

        def unbounded():
            while True:
                yield response

        with self.assertRaisesRegex(RunPodVllmError, "one object"):
            decode_worker_output(unbounded(), expect_stream=False)

    def test_integrates_with_kova_public_stream_contract_without_network(self):
        captured = []

        def worker_call(job):
            captured.append(job)
            return fixture("runpod_vllm_stream_success.json")

        records = []
        result = handle_job(
            {"input": {
                "request_id": "fixture-request",
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": "low",
                "max_output_tokens": 2048,
            }},
            make_queue_inference_client(worker_call),
            self.runtime_probe,
            records.append,
            execution_context={
                "logical_request_id": "kova-exec-00000000-0000-4000-8000-000000000001",
                "benchmark_candidate_id": self.candidate["id"],
                "route_id": "instant",
                "stage_id": "answer-1",
                "public_response": True,
                "prior_stage_outputs": {},
            },
            token_counter=lambda _model, _messages: 10,
            clock_ns=Clock(0, 10_000_000, 20_000_000),
            attempt_id_factory=lambda: "fixture-attempt",
        )
        self.assertEqual(result["content"], "Kova response")
        self.assertEqual(result["benchmark"]["time_to_first_token_ms"], 10)
        self.assertEqual(captured[0]["input"]["openai_route"], "/v1/chat/completions")
        self.assertEqual(records[0]["outcome"], "success")

    def test_integrates_nonstream_private_stage_and_rejects_hidden_reasoning(self):
        response = fixture("runpod_vllm_nonstream_success.json")
        records = []
        result = handle_job(
            {"input": {
                "request_id": "private-fixture",
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": "medium",
                "max_output_tokens": 512,
            }},
            make_queue_inference_client(lambda _job: response),
            self.runtime_probe,
            records.append,
            execution_context={
                "logical_request_id": "kova-exec-00000000-0000-4000-8000-000000000002",
                "benchmark_candidate_id": self.candidate["id"],
                "route_id": "work:cosmo:medium",
                "stage_id": "planning-1",
                "public_response": False,
                "prior_stage_outputs": {},
            },
            token_counter=lambda _model, _messages: 10,
            clock_ns=Clock(0, 20_000_000),
            attempt_id_factory=lambda: "private-fixture-attempt",
        )
        self.assertEqual(result["content"], "Private Kova stage")
        self.assertIsNone(result["benchmark"]["time_to_first_token_ms"])

        hidden = [
            'data: {"choices":[{"delta":{"reasoning_content":"secret","content":"answer"}}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n',
            "data: [DONE]\n\n",
        ]
        with self.assertRaisesRegex(ValueError, "hidden reasoning"):
            handle_job(
                {"input": {
                    "request_id": "hidden-fixture",
                    "messages": [{"role": "user", "content": "hello"}],
                    "reasoning_effort": "low",
                    "max_output_tokens": 2048,
                }},
                make_queue_inference_client(lambda _job: hidden),
                self.runtime_probe,
                lambda _record: None,
                execution_context={
                    "logical_request_id": "kova-exec-00000000-0000-4000-8000-000000000003",
                    "benchmark_candidate_id": self.candidate["id"],
                    "route_id": "instant",
                    "stage_id": "answer-1",
                    "public_response": True,
                    "prior_stage_outputs": {},
                },
                token_counter=lambda _model, _messages: 10,
                clock_ns=Clock(0, 10_000_000, 20_000_000),
                attempt_id_factory=lambda: "hidden-fixture-attempt",
            )


if __name__ == "__main__":
    unittest.main()
