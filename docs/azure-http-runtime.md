# Azure HTTP and managed-identity runtime — source-only

This change adds the concrete standard-library HTTP and Container Apps managed-
identity clients behind the existing guarded Azure adapter. It does not deploy
or enable them. No Azure endpoint, model, GPU, token, resource or production route
was contacted or changed during verification. Tests use synthetic tokens and
in-memory or loopback-only fixtures. No external Python dependency is added.

## Implemented

`worker/bounded_http.py` opens one HTTP/1.1 connection to a server-supplied numeric
address and parses responses with Python's `http.client.HTTPResponse`. Sockets are
nonblocking. One monotonic deadline and cancellation signal cover connecting, TLS
handshaking, request writes, header reads and body reads. Polling does not reset the
deadline. There are no retries, redirects, environment proxies, helper threads,
implicit DNS lookups, or compression. Response ownership includes the socket;
completion, rejection, cancellation and explicit close release it.

TLS uses certificate-chain and hostname verification, SNI and TLS 1.2 or newer.
The verified hostname and HTTP Host stay the configured Container App hostname,
not the numeric connection address. Response status, content type, duplicate length
headers, transfer framing, byte ceilings and truncated bodies fail closed.

`worker/azure_http_runtime.py` connects this transport to the existing Azure
Chat Completions boundary. The model request stays direct JSON at the fixed
`/v1/chat/completions` path, not a RunPod queue envelope. Existing Core stage
validation, token accounting and response sanitization still apply.

The concrete factory requires all existing paid/authentication/transport guards
plus a separate `network_execution_authorized` flag. Its default is false. Merely
importing the module or constructing a client does not read platform credentials,
create a socket, or start inference. All real target selections and permissions
in `config/azure-http-runtime.v1.json` remain unset/false. That checked-in contract
is not a deployment configuration and does not automatically construct a client.

## Managed identity, not an API-key fallback

The token client reads `IDENTITY_ENDPOINT` and the rotating `IDENTITY_HEADER` from
trusted server environment data for each acquisition. It calls the local endpoint
with GET, `api-version=2019-08-01`, an explicitly configured resource URI and an
optional explicit user-assigned `client_id`. Omitting that client ID deliberately
requests system-assigned identity; a failed selected identity is not retried as a
different identity. The identity header never goes to the inference endpoint;
only the returned Bearer token does. There is no CLI/local-login credential chain,
secret fallback, token cache, or automatic reuse of an Azure OpenAI audience.

The resource URI must identify the receiving API; it is not an OAuth `/.default`
scope. Response parsing rejects duplicate/nonstandard JSON, missing/wrong resource,
wrong selected identity, malformed tokens and expiration too close to the request
budget. The extra 30-second credential expiry allowance is not a model reasoning
or response-duration target. Errors do not echo tokens, identity headers or remote
response bodies.

A successfully acquired token is **not** proof of API authorization. The receiving
service must verify the intended tenant/issuer/audience and enforce its allowed
caller/app-role policy. Microsoft's Container Apps authentication documentation
explicitly notes that target code must validate expected role claims. Those
configuration and live negative-auth tests are still outstanding.

## Deliberate restrictions and remaining integration gates

**Destination address:** the transport requires a reviewed numeric address. This
avoids an unbounded synchronous DNS call inside the request deadline; TLS still
verifies the original hostname. Automatic DNS resolution, freshness refresh,
failover and ownership/private-reachability verification are not implemented.
Before live use, a trusted integration must supply and refresh the correct Azure
address without accepting it from a client request. A stale address fails rather
than silently falling back. No IP address has been selected in the real contract.

**Identity endpoint:** this implementation accepts numeric loopback/link-local
HTTP endpoints from the platform environment, with canonical paths and no URL
credentials/query/fragment overrides. It does not resolve `localhost` or arbitrary
DNS identity endpoints. The actual deployed environment shape must be verified;
a different platform shape requires an explicitly reviewed extension, not relaxed
SSRF checks. Plain HTTP is never allowed for model inference.

**Timeouts:** the HTTP-hop timeout is explicit and remains unset in the repository's
real configuration. It is below the adapter's existing 240-second ingress ceiling;
this work does not change that infrastructure contract or claim streaming bypasses
it. All model response/work-duration targets remain unchanged/unresolved. Long
work, reconnection, persisted job orchestration and Ultra execution remain separate
workstreams. Cancellation interrupts this client's socket I/O; it does not prove
the remote model scheduler has stopped billable GPU computation. That must be
verified in the deployed serving runtime.

**Runtime:** there is no application startup wiring, selected model container,
real identity token, allowed API role, verified Azure endpoint, paid benchmark or
live browser stream in this change. The client factory's booleans are server-owned
controls, not proof that external verification occurred. Do not source them from
request parameters or turn them on because CPU tests passed.

## Verification

Run from the repository root:

```sh
npm test
npm run preflight
python3 -m unittest worker.test_bounded_http worker.test_azure_http_runtime
```

The socket tests open only loopback listeners. TLS tests generate a temporary,
one-day certificate/key with local OpenSSL and remove them afterward; no private
key or credential is checked into source. Tests exercise successful hostname/SNI
verification, rejection of an untrusted chain and wrong hostname, stalled TLS,
headers and bodies, trickled bytes, bounded chunked/length-framed reads, cancellation,
explicit close, unsafe framing, target/header overrides and no DNS/proxy lookup.

A full fixture round trip uses local identity REST, real loopback TLS, fragmented
Chat Completions SSE and the existing Kova handler. Model output, identity and
runtime/GPU probes in that test are synthetic; no model quality or performance is
measured. Managed-identity tests cover explicit resource/client selection, rotating
headers, expiry, malformed responses and disabled-before-credential guards.

A regression also reproduces the old adapter losing typed cancellation/deadline
exceptions in credential, transport and body phases. The adapter now preserves
those controlled termination types while continuing to sanitize arbitrary failures.
The separate HTTP-runtime preflight rejects missing, enabled or guessed readiness,
authorization, address, audience, role and timeout settings.

## Documentation used

- Microsoft Container Apps managed identity REST contract:
  https://learn.microsoft.com/en-us/azure/container-apps/managed-identity
- Microsoft Container Apps Entra authentication and target app-role enforcement:
  https://learn.microsoft.com/en-us/azure/container-apps/authentication-entra
- Python HTTP client and response parser:
  https://docs.python.org/3/library/http.client.html

Checked September 15, 2026. Provider configuration and live behavior still require
separate authorization and verification. No merge or deployment is implied.
