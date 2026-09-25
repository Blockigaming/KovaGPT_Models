# Kova 1.0

Kova is the AI system developed for KovaGPT. **Microsoft Azure is the selected
hosting direction** for Kova Core and Kova Ultra. Azure Container Apps Consumption
GPU is the proposed scale-to-zero target; region, quota, hardware, container and
model selections remain blocked. The existing RunPod source is legacy compatibility
material, not the active hosting decision or a live deployment. Cloudflare remains
the DNS, CDN, WAF and DDoS edge, not an inference host.

The guarded [Azure provider boundary](docs/azure-provider-boundary.md) now has
CPU-only HTTP/SSE fixtures, explicit execution gates, cancellation cleanup and
integration coverage for every Core Chat/Work stage. No network/credential SDK is
bound to production, and no model or Azure resource was started. The concrete
[HTTP and managed-identity REST client](docs/azure-http-runtime.md) is implemented
and loopback-tested behind disabled-by-default controls. No live endpoint,
address, audience, identity or timeout is selected. The historical RunPod-specific
sections below describe the retained source until a verified Azure cutover.

The [source-only execution kernel](docs/execution-kernel.md) now runs Core stages
and concurrent Ultra specialist/decision/synthesis workflows over injected clients.
Its local reference journal supports safe-frontier restart, owner-scoped replay,
atomic stage claims, retained unknown-attempt reservations and cancellation. It is
not a production durable store or deployed background service. All model timing,
GPU capacity and quality claims still require separately authorized evidence.

Kova is not a foundation model trained from scratch. Cosmo, Orion, and Nova are behavior and compute profiles, not claims of separately trained foundation weights. The active provider, upstream model, and license must be disclosed truthfully when asked. This repository contains public-safe source only: never commit credentials, private conversations, paid model outputs, private datasets, downloaded weights, adapters, or checkpoints.

## Current state

- Two-engine provider and route contracts defined
- One shared RunPod Core endpoint and one RunPod Ultra endpoint are source-defined; neither exists yet
- BF16 and official FP8 Qwen3.8-27B Core candidates are pinned; no winner is selected
- Official RunPod vLLM worker source is pinned as an unbuilt container candidate; no image digest is selected
- CPU-only compatibility fixtures cover the pinned worker's OpenAI queue input and raw SSE output boundary
- Three self-hosted Qwen candidates remain pinned for evaluation
- Offline planning and dataset validation available
- Paid training disabled
- No trained Kova checkpoint exists yet
- No KovaGPT production routing has changed
- The recorded 22% estimate is historical and has not been recomputed for Azure; zero of 25 target routes are recorded live
- Deterministic Kova Auto baseline implemented with Free-plan and Ultra-budget gates
- Source-only RunPod Core multi-pass request planner and lifecycle-cost summarizer implemented
- Source-only Ultra specialist, disagreement-check, judge, conditional-debate, and synthesis planner implemented
- Reproducible offline contract evaluation covers all 25 target routes without provider calls

## Product-complete target

Kova Auto sits above six Chat modes: Instant, Medium, High, Extra High, Max,
and Ultra. Instant through Max use increasing compute policies over one Kova
Core model. Ultra changes architecture to dynamic specialists, a disagreement
check, a judge, and synthesis. Work exposes Kova Cosmo, Kova Orion,
and Kova Nova, each with Light through Ultra effort. Instant responds directly. Deeper modes
may provide concise, truthful progress updates and ask focused questions when
missing information would materially change the result. Activity text is never
hidden chain-of-thought and may only describe events that actually occurred.

## Free checks

```sh
npm test
npm run preflight
npm run evaluate:offline
npm run dataset:compile
npm run training:command
```

Passing these checks does not authorize GPU spending, training, deployment, or a production model replacement.

`npm run benchmark:candidate:summarize -- benchmark.json` summarizes isolated
model-candidate attempts only. It explicitly cannot claim Core or Ultra route
completion and cannot support customer route pricing. It never calls RunPod.

`npm run benchmark:core:summarize -- core-benchmark.json` prices complete recorded
RunPod worker lifecycles and separates results by pinned model revision, GPU type
and count, serving engine, endpoint type, container digest, and Kova route. A
shutdown observation is recorded separately from request attempts. It requires one
measured cold start, one measured shutdown tail, and provider-reported total billed
wall time per lifecycle. That billed wall time—not the sum of request attempts—is
the authoritative cost basis, so warm gaps and overlapping requests are neither
lost nor double-counted. The conserved startup, active-window, and idle components
are allocated without changing the lifecycle total: startup and idle are shared
equally per logical request, while active-window cost is weighted by observed
attempt inference time. The recorded GPU rate is the total worker rate for its
configured GPU count, and the billed active window must contain the longest
individual inference attempt and the longest sequential per-request inference path.
RunPod-side route TTFT includes cold startup, all
attempt queues and retries, every sequential private DAG stage, and the final
public stage's first visible delta. It reports RunPod compute
only, not a publishable customer price; Azure, tools, storage, payment processing,
taxes, and other attributable costs must still be included.
The request planner also requires a trusted provider tokenizer count, binds only
server-recorded but untrusted prior model artifacts, reserves their worst-case token ceilings,
and rejects any operation whose bound prompt plus output ceiling could exceed the
candidate context. The executor must recount the fully bound request before inference.

The source-only benchmark worker validates request IDs, roles, reasoning effort,
aggregate prompt size, token limits, trusted Kova identity, and measured telemetry.
The caller's request ID is retained only as correlation metadata; a server-generated
`kova-exec-{uuid4}` identifies and groups one logical route execution across stages.
For each benchmark job, trusted server configuration must select exactly one candidate
ID from `config/core-serving.v1.json`; client input cannot select or override it. This
lets the same worker benchmark both pinned BF16 and FP8 candidates without treating
either one as the production winner.
It rebuilds the selected Core stage from the server policy, binds exactly the
server-recorded outputs required by that stage's DAG, recounts the fully bound prompt,
and rejects mismatched stage limits or missing artifacts. Public stages must return
real streaming chunks; the worker assembles content, tool-call fragments, and usage
before sanitizing the result and measures first-visible-delta latency with its
monotonic clock. That measurement is preserved if a later stream chunk fails;
attempts with no visible public delta and all private non-stream stages record TTFT
as unavailable (`null`) instead of inventing a value. Empty or whitespace-only text
does not start TTFT or count as a successful answer; empty tool fragments, truncated
finish reasons, and zero input or completion usage also fail closed. The trusted
runtime probe verifies the actually loaded model and revision against that selected
allowlisted pin before and after every attempt and again at lifecycle close. A
postflight runtime-integrity failure persists the paid attempt as quarantined before
the error propagates. Private stages use non-streaming responses. The worker pins its
candidate model server-side and fails closed if hidden reasoning appears in a separate
field or embedded `<think>` block. Candidate-aware hardware planning starts at 80 GB
VRAM for BF16 and 48 GB for official FP8, without claiming that either checkpoint fits
or performs acceptably. Provider GPU inventory, pricing, and a concrete hardware ID
must be captured fresh at benchmark time. No container image is selected until a
compatible immutable digest and Qwen3.8 serving path are verified.

The CPU-only RunPod adapter wraps trusted engine requests in the pinned worker's
`openai_route`/`openai_input` queue shape and reconstructs fragmented raw OpenAI
SSE. It requires exactly one terminal `[DONE]`, preserves tool-call and usage
fields for Kova's existing sanitizer, and rejects malformed UTF-8, unsupported
SSE fields, provider errors, oversized events, incomplete streams, duplicate
terminal markers, and data after termination. These fixtures exercise only the
pinned worker boundary after provider transport handling. They do not call
RunPod, support a live RunPod HTTP envelope, verify a model response, prove
Qwen3.8 compatibility, or authorize a container build, endpoint, or GPU spend.

## Provider plan

Azure remains the application platform for routing, tools, billing enforcement,
and streaming. New providers must be added behind the existing provider abstraction;
existing Azure deployments are not deleted before a verified canary and cutover.

The final architecture decision rejects Cloudflare Workers AI as the primary
inference backend after user-reported inconsistent interactive latency. That report
is recorded as a product decision, not reproducible benchmark telemetry. Cloudflare
continues to serve only the edge roles.

The historical RunPod source below is retained for compatibility, not the current
Azure hosting decision. Kova Core has two source-verified,
hardware-unbenchmarked candidates representing the same Qwen3.8-27B model family:
the original BF16 checkpoint and the official fine-grained FP8 checkpoint. Model,
quantization, GPU, serving engine, endpoint type, and container digest all remain
unselected until reproducible quality, latency, streaming, memory, and lifecycle-cost
benchmarks pass. Qwen documents a native 262,144-token context, `low`, `medium`, and
`xhigh` reasoning effort, and support for vLLM and SGLang.

The source-only container contract pins the official
[`runpod-workers/worker-vllm` v2.27.0 release](https://github.com/runpod-workers/worker-vllm/releases/tag/v2.27.0)
at source commit `76054c22c79c515f07065f523598d8efb2f9b682`; that release bundles
vLLM 0.29.0. `runpod/worker-v1-vllm:v2.27.0` is only a candidate tag. Its
registry digest is unresolved, the image has not been pulled or built, and compatibility
with Kova's benchmark contract has not been claimed. Runtime model and revision values
are fixed by trusted server configuration for each pinned Core candidate; clients cannot
override them. The maximum served context remains unset until memory and latency tests.

Weight packaging is also intentionally unresolved. Runtime model downloads and network
volumes remain disabled, while neither a model-baked image nor an immutable cached
artifact currently exists. Queue-based versus load-balancing Serverless remains
unselected. Resolving any of those choices requires explicit benchmarks and does not
authorize image pulls, endpoint creation, GPU execution, deployment, or routing.

Kova Auto currently uses deterministic server rules. Free is capped to Instant;
Plus is capped to High on every classification/fallback branch; Ultra requires
Pro entitlement, explicit runtime authorization, and sufficient remaining request
budget. This classifier is tested but not production-routed.

### RunPod Core and Ultra

The checked-in RunPod configuration is planning-only and fail-closed. It reserves
`kova-core` and `kova-ultra`, each with Flex workers, zero active workers, a
one-worker cost cap, five-second idle timeout, required streaming, FlashBoot, and
cached-model loading. Neither endpoint has been created. Startup, request execution,
the post-request idle timeout, storage, retries, payment processing, and applicable
usage taxes must all be measured before customer prices are published.

RunPod's standard account billing is prepaid-credit based even though Serverless
compute is metered per second. This plan explicitly disables auto-pay and automatic
credit reloads: there is no flat-rate compute plan, but a manual prepaid balance is
still required by RunPod. Deposited credits are non-refundable. A true standard
postpaid card charge after usage is not represented as supported.

The target gross margin is 42.6%, using:

```text
customer price = attributable cost / 0.574
```

This formula targets 42.6% before rounding. Realized margin must be measured from
actual usage and recalibrated; it cannot be guaranteed from GPU list prices alone.

The target catalog contains Kova Cosmo, Kova Orion, and Kova Nova. Nova is
Work-only. Lite through Ultra are bounded processing configurations over the
selected family, not separate foundation models. None is live.

The offline evaluator resolves all 25 Auto, Chat, and Work route contracts through
the actual router; builds provider-free Core or Ultra plans; verifies DAG, identity,
disclosure, source-grounding, activity, and worst-case token-budget invariants; and
proves that Cosmo, Orion, and Nova Work profiles have distinct server-controlled
behavior instructions. It evaluates no real model output and therefore leaves all
quality, factuality, latency, cost, and release gates blocked.

The dataset compiler creates immutable, hashed ms-swift train and validation files.
The training command is a dry run by default and its execution path remains blocked
until cost authorization and single-GPU compatibility are separately verified.
