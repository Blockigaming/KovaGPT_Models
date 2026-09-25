# Azure reasoning-output privacy contract

This source change separates the request to reason from permission to return a
private reasoning trace. It does not deploy a model, select a serving image, change
Chat/Work pass budgets, or claim any latency or quality measurement.

The Azure boundary adds `include_reasoning: false` after validating the trusted
engine request. Callers cannot supply that field themselves. `reasoning_effort`,
`chat_template_kwargs.enable_thinking`, `preserve_thinking`, token limits, message
content and stage planning remain unchanged. The legacy RunPod queue contract is
not changed.

Every decoded Azure JSON response and SSE chunk is checked before it is returned
or yielded. Both current `reasoning` and historical `reasoning_content` fields are
rejected when populated, as are `reasoning_details` and nonnull token/logprob side
channels. Null or empty textual reasoning fields are allowed. Boolean, numeric,
object and array substitutes are not treated as empty. Errors contain no provider
reasoning text. Cleanup follows the existing completion/failure/cancellation path.

Normal answer explanations, serialized tool-argument terms and numeric usage
fields such as `reasoning_tokens` are preserved. This check does not execute tools,
rewrite answers, convert reasoning into content, or replace the downstream response
sanitizer. The existing content-level hidden-reasoning checks still apply.

## Verification requirements

CPU fixtures exercise suppression in streaming/nonstreaming and thinking/direct
requests, both reasoning field names, malformed values, side channels, late private
chunks, normal explanations, usage preservation and disabled execution controls.
The first test-only commit is an expected-failing negative control against the
published adapter. Final CI must pass before this change is called verified.

## Remaining serving gate

The selected immutable server image must support `include_reasoning=false` and
prove its actual response and usage behavior before live transport approval.
Neither a model-card example nor CPU fixtures establish that deployed compatibility.
No fallback retry that omits suppression is permitted. A server returning private
reasoning despite suppression fails closed. Phase B and production routing remain
blocked by the wider readiness requirements.

Primary documentation checked September 16, 2026:
- https://docs.vllm.ai/en/latest/features/reasoning_outputs/
- https://huggingface.co/Qwen/Qwen3.8-27B

vLLM documents the `reasoning_content` to `reasoning` rename and suppression via
`include_reasoning=false`, including the token/logprob side-channel restriction.
Qwen's official example handles both field names. These are API-contract sources,
not Kova model benchmarks.
