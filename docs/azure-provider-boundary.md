# Azure model provider boundary — source-only verification

Microsoft Azure is the owner-selected model hosting direction. This change keeps
Kova's names, Core/Ultra separation, Chat and Work pass budgets, identity, token
ceilings, tool-result trust and existing production infrastructure unchanged.

## Implemented boundary

`worker/openai_protocol.py` owns the shared defensive request copy and bounded,
fragment-aware UTF-8 SSE decoder. `worker/runpod_vllm.py` retains its exact queue
wrapper and public exception type through this shared module. It is legacy
compatibility source, not an Azure transport.

`worker/azure_container_apps.py` serializes direct JSON to the fixed
`/v1/chat/completions` path and consumes an injected HTTP transport. It validates
canonical HTTPS Azure origins, rejects caller transport overrides, pins the
request model to trusted settings, rejects redirects and unexpected response
content types, enforces byte/JSON limits, redacts credential/error details, and
closes responses on completion, rejection, cooperative timeout or cancellation.
There are no automatic retries and no HTTP or credential SDK is bound to
production. The concrete, standard-library HTTP and managed-identity REST
implementation is now documented in [Azure HTTP runtime](azure-http-runtime.md);
its external configuration and execution remain disabled.

The factory requires all three server-owned execution/authentication/transport
flags before invoking the credential provider or transport. All real repository
readiness and spending flags remain false. Tests use synthetic settings and an
in-memory transport, not credentials or a network connection. These booleans are
not proof of authorization: the future server integration must load them only
from approved, reviewed server configuration, never from a client payload.

Public streams open lazily, so an unconsumed generator cannot start a request.
The Kova handler now explicitly closes its stream when it rejects a chunk;
otherwise a retained iterator could keep a rejected response open. A regression
first reproduced that leak and now verifies closure without weakening hidden
reasoning rejection. Raw decoded provider chunks must go through the handler;
they are not user-visible progress or text.

## What the fixtures prove

Tests exercise all 20 Core profiles (five Chat plus fifteen non-Ultra Work
combinations), covering their 113 private/public stages. They check preserved
identity/prompts, stage dependencies and limits, direct Azure request shape,
private-stage null first-token timing, public first-visible-delta timing,
fragmented tool-call assembly, hidden reasoning rejection and runtime revision
quarantine. Tool-call fixtures do not execute tools. Synthetic timings and costs
are not performance or billing measurements.

The existing offline evaluator still checks all 25 route contracts. These new
Core fixtures do not prove the separate Ultra runtime, Auto's application
integration, real model quality, or live end-to-end streaming to the browser.

## Limits that must remain explicit

Microsoft documents a 240-second HTTP ingress request timeout. The adapter requires
an explicitly supplied finite per-hop timeout below that infrastructure limit;
no production value or model active-work duration has been selected. The 30-second
values in tests are synthetic fixtures, not approved product targets. A status
message is not an answer token. Streaming must not be assumed to bypass the
platform timeout.

Cooperative checks around reads cannot interrupt a blocking transport by
themselves. A future HTTP implementation must enforce connect/read deadlines,
abort on close/cancel, verify TLS, avoid redirects, avoid automatic retries and
redact headers/bodies. The concrete implementation now has loopback-only tests; its live verification
and production binding are still blocked. Longer workflows need separately verified bounded orchestration and
reconnection/cancellation semantics; this PR does not implement a job system.

A valid Azure hostname is not proof of private routing or authentication. Verify
the actual environment/ingress reachability, exclude unintended environment-level
HTTP routes, verify the audience/issuer/caller authorization and fail-closed
responses before any endpoint use. Do not reuse an Azure OpenAI token audience
for a custom Container App without proving the correct auth configuration.

T4/A100 names in the planning contract are platform profile candidates, not proof
that the pinned Core weights fit or perform acceptably on each. Select hardware,
context length, immutable image/weights, region and quotas only after verification.

## Reproduce without paid execution

```sh
npm test
npm run preflight
python3 -m unittest worker.test_openai_protocol worker.test_azure_container_apps worker.test_azure_integration
```

The Azure preflight validates every required readiness/authorization field,
including missing-field negative controls. A passing preflight is not a deployment
permit. CI's one-day source archive contains only `git archive HEAD`, its hash and
exact commit ID; it excludes runtime files/secrets and supports offline reproduction
when direct source downloads are unavailable.

## Remaining gates

Verify the implemented HTTP/auth client's Azure destination, audience, caller
roles and real network behavior only under separate live-use approval;
validate immutable container/weights and Azure quota/region/hardware; benchmark
cold/warm full-route quality, timing, cancellation and attributable cost; implement
long-running work and actual Ultra execution; resolve application entitlements and
legacy mode mappings; then obtain staging/cutover authorization. No merge,
deployment, paid benchmark, training, model download or production routing occurs
as a consequence of these CPU checks.

## Primary documentation checked September 15, 2026

- Azure ingress: https://learn.microsoft.com/en-us/azure/container-apps/ingress-overview
- Azure authentication: https://learn.microsoft.com/en-us/azure/container-apps/authentication
- Azure GPU profiles: https://learn.microsoft.com/en-us/azure/container-apps/workload-profiles-overview
- Azure serverless GPU: https://learn.microsoft.com/en-us/azure/container-apps/gpu-serverless-overview
- vLLM serving contract: https://docs.vllm.ai/en/latest/serving/online_serving/
