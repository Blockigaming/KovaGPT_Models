"""Adversarial provider-output fixtures, never tool execution or real telemetry."""
from copy import deepcopy
import math
import unittest

from worker.handler import (
    MAX_TOOL_CALLS, MAX_TOOL_ARGUMENT_CHARS, PINNED_CORE_CANDIDATES,
    _validate_runtime_value, consume_engine_response, emit_lifecycle_close,
    sanitize_engine_response,
)
from execution.test_support import ModelFixture


def response():
    return {"choices": [{"index": 0, "message": {"content": "Kova answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}}


def tool(arguments='{"q":"example"}', identifier='call_1'):
    return {"id": identifier, "type": "function", "function": {"name": "lookup", "arguments": arguments}}


def tool_response(calls):
    value = response()
    value["choices"][0].update(message={"content": "", "tool_calls": calls}, finish_reason="tool_calls")
    return value


def stream(chunks):
    return consume_engine_response(iter(chunks), expect_stream=True, clock_ns=lambda: 0,
                                   started_ns=0, timing_state={"time_to_first_token_ms": None})


class ResponseIntegrityTests(unittest.TestCase):
    def test_tool_arguments_reject_duplicate_keys_at_any_depth(self):
        for arguments in ('{"q":1,"q":2}', '{"outer":{"value":1,"value":2}}'):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                sanitize_engine_response('fixture', tool_response([tool(arguments)]))

    def test_tool_arguments_reject_nonfinite_values_including_float_overflow(self):
        for arguments in ('{"q":NaN}', '{"q":Infinity}', '{"q":-Infinity}', '{"q":1e400}'):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                sanitize_engine_response('fixture', tool_response([tool(arguments)]))

    def test_duplicate_tool_ids_cannot_be_presented_as_separate_actions(self):
        with self.assertRaises(ValueError):
            sanitize_engine_response('fixture', tool_response([tool(), tool('{"q":"other"}')]))

    def test_valid_nested_tool_json_and_distinct_ids_are_preserved(self):
        calls = [tool('{"q":{"x":[1,2.5,null,true]},"text":"café"}'), tool(identifier='call_2')]
        self.assertEqual(sanitize_engine_response('fixture', tool_response(calls))['tool_calls'], calls)

    def test_deep_tool_json_fails_with_sanitized_error(self):
        args = '{"x":' + '[' * 2000 + '0' + ']' * 2000 + '}'
        with self.assertRaises(ValueError) as caught:
            sanitize_engine_response('fixture', tool_response([tool(args)]))
        self.assertNotIn(args, str(caught.exception))

    def test_nonstream_requires_one_choice_and_an_actual_integer_zero_index(self):
        for mutation in ('multiple', True, False, 0.0, 1, '0'):
            value = response()
            if mutation == 'multiple':
                value['choices'].append(deepcopy(value['choices'][0]))
            else:
                value['choices'][0]['index'] = mutation
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                sanitize_engine_response('fixture', value)

    def test_stream_rejects_multiple_choices_and_noninteger_index(self):
        delta = {"index": 0, "delta": {"content": "answer"}}
        with self.assertRaises(ValueError):
            stream([{"choices": [delta, deepcopy(delta)]}])
        for index in (True, False, 0.0, '0'):
            with self.subTest(index=index), self.assertRaises(ValueError):
                stream([{"choices": [{**delta, "index": index}]}])

    def test_duplicate_nonnull_usage_chunks_cannot_overwrite_cost_evidence(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 30}
        chunks = [{"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
                  {"choices": [], "usage": usage},
                  {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}]
        with self.assertRaises(ValueError):
            stream(chunks)

    def test_stream_tool_count_and_partial_arguments_are_bounded_while_reading(self):
        too_many = [{"index": i, "id": f"call_{i}"} for i in range(MAX_TOOL_CALLS + 1)]
        too_long = [{"index": 0, "function": {"arguments": 'x' * (MAX_TOOL_ARGUMENT_CHARS + 1)}}]
        for fragments in (too_many, too_long):
            with self.subTest(count=len(fragments)), self.assertRaises(ValueError):
                stream([{"choices": [{"delta": {"tool_calls": fragments}}]}])

    def test_runtime_cost_and_duration_numbers_must_be_finite(self):
        probe = ModelFixture().probe('answer-1', 'before')
        candidate = next(iter(PINNED_CORE_CANDIDATES.values()))
        for field in ('worker_start_ms', 'model_load_ms', 'queue_ms', 'gpu_rate_per_second_usd'):
            for value in (math.inf, math.nan):
                changed = {**probe, field: value}
                if field in ('worker_start_ms', 'model_load_ms'):
                    changed['cold_start'] = True
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    _validate_runtime_value(changed, candidate)

    def test_lifecycle_close_rejects_nonfinite_billing_and_idle(self):
        probe = ModelFixture().probe('answer-1', 'before')
        candidate = next(iter(PINNED_CORE_CANDIDATES.values()))
        base = {k: v for k, v in probe.items() if k not in ('cold_start', 'worker_start_ms', 'model_load_ms', 'queue_ms')}
        base.update(billed_lifecycle_ms=1000, attributed_idle_timeout_ms=10)
        for field in ('billed_lifecycle_ms', 'attributed_idle_timeout_ms'):
            changed = {**base, field: math.inf}
            if field == 'attributed_idle_timeout_ms':
                changed['billed_lifecycle_ms'] = math.inf
            records = []
            with self.subTest(field=field), self.assertRaises(ValueError):
                emit_lifecycle_close(lambda: changed, records.append, benchmark_candidate_id=candidate['id'])
            self.assertEqual(records, [])


if __name__ == '__main__':
    unittest.main()
