# Verified model artifact to native serving lifecycle

A32 source integration only. The existing verify-only container entrypoint remains
unchanged and blocked. No selected image/model, new HTTP listener, vLLM package,
real weights or GPU are installed by this change or its CPU tests.

## Executable loader handoff

ServingRuntime checks independent explicit loading/GPU flags, current server
approval and supervisor-observed immutable image identity before reading artifact
policy or importing a native loader. It pins the protected startup policy's exact
bytes, requires a nonroot process and OS-enforced read-only artifact mount, calls
the existing real artifact verifier, rereads policy bytes to reject replacement,
and matches the chosen candidate. Offline environment flags and disabled plugins
are mandatory but do not replace the deployment's verified egress policy.

The NativeVllm binding targets the documented vLLM 0.29.0 AsyncEngineArgs and
AsyncLLM.from_engine_args APIs. It loads only the verified local model/tokenizer
path and pinned revision, safetensors format, no remote Python model code, no
request logging or prefix cache, one tensor-parallel worker and explicit context,
concurrency and memory settings. The installed vLLM version must match before
native imports; no version or model download fallback exists. This source-supported
API version is NOT an approved production image or compatibility benchmark.

After initialization, actual engine.model_config values must match the verified
path, revision, served model, context and remote-code policy. check_health must
succeed before readiness. Every identity/engine handoff repeats approval, image,
configuration and health checks; runtime drift or revoked approval fails closed.
A new load always gets a new lifecycle ID, so an old handle cannot silently refer
to a restarted engine. Identity omits private artifact paths, prompts, credentials,
private reasoning and fabricated GPU/cost telemetry. It is a server observation
chain, not independent hardware attestation.

## Failure boundaries

The constructor does no I/O. Flags default false, and the checked-in configuration
selects no path, model, image, memory, context or timeout. The existing source
startup preflight still reports serving blocked. A server application must choose
and authenticate loading approvals independently; arbitrary internal Python code
holding those approvals is trusted, not a public untrusted request boundary.

Native CUDA initialization is synchronous. The explicit elapsed-load budget is
checked around initialization; it cannot forcibly interrupt a stuck native/CUDA
call. A real process supervisor must enforce that hard boundary. Health awaits
have bounded async timeouts; shutdown passes an explicit timeout to the native
engine. Failed shutdown retains the backend reference and failed state for
supervisor cleanup instead of claiming GPU quiescence. Closing a process is not
proof of stopped remote billing.

The engine handoff is internal only. Receiver token validation, actual user/session
and plan authorization, context/tool permissions, byte/token/cost admission,
reasoning-output privacy and cancellation must wrap actual generation in the
application integration (A24 and browser gates). No unauthenticated native HTTP
server or production startup wiring is introduced here.

## Source tests

Tests create tiny intentionally non-loadable artifact fixtures and use the real
startup/hash verifier. A synthetic backend exercises load/health/configuration,
policy tampering, bad artifact hashes, cleanup, restart, lifecycle fencing and
revocation. Another test substitutes the documented native vLLM module interfaces
to inspect every constructor argument, health call and shutdown timeout. This
checks our adapter wiring without loading vLLM, torch or a GPU model; the actual
library/weight/container compatibility still requires separately approved Phase B
execution. Mocked readonly/nonroot environment tests do not claim the temporary
fixture directory is an actual production mount.

Primary API references checked September 16, 2026:
- https://docs.vllm.ai/en/v0.29.0/api/vllm/engine/arg_utils/
- https://docs.vllm.ai/en/v0.29.0/api/vllm/v1/engine/async_llm/
- https://docs.vllm.ai/en/latest/usage/security/

This component neither republishes blocked job/tool APIs nor changes existing model
names, Chat/Work compute policies, output ceilings or production guards.
