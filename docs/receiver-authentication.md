# Receiving-service authentication: source-only Entra app-token verifier

This component authenticates a calling service, not a Kova customer or subscription.
It does not install a listener, start a model, bind production application routes,
create an identity or contact Microsoft. Phase B remains blocked.

## Actual verification, not decoding claims as identity

`worker/receiver_auth.py` uses pinned PyJWT/cryptography implementations for RS256
signature verification against a separately approved public-key snapshot. The
single-tenant policy fixes token version, issuer, API audience, allowed pairs of
application client ID AND service-principal object ID, required application roles,
key-snapshot expiry, token lifetime and clock skew. No value comes from a request
body, a claimed owner/plan, a token-supplied key URL or an unsigned JWT decode.

V1 and V2 issuers are separate explicit profiles. V1 uses `appid` and an API URI;
V2 uses `azp` and the API application GUID. The version is selected in trusted
configuration, not by an unverified token. Mixed/ambiguous client claims fail.
Only app-only tokens carrying `idtyp=app` and all required roles are accepted;
delegated `scp`, ID-token nonce and actor-delegation forms are rejected. The target
Entra app configuration must supply the required app-only claims; no missing-claim
fallback is enabled. This is intentionally narrower than a generic Entra verifier.

Token signature, expiry, not-before, issued-at, audience and issuer checks remain
enabled in the library with a fixed `RS256` allowlist. Additional bounded parsing
rejects duplicate keys, numeric coercion, noncanonical base64url, nonfinite JSON,
excess structure, unsupported JOSE headers and private/weak signing keys. A valid
signature by itself does not grant access to an unlisted application/object pair
or a caller lacking one of the required roles.

## Key authority and rotation

A trusted administrator/controller must retrieve the proper tenant/version OpenID
configuration and signing keys over verified TLS, validate their issuer scope,
normalize the public RSA keys into the documented snapshot, then approve its exact
SHA-256 and expiry separately. No actual tenant, audience, caller, role or key pin
is selected in source. Hashing a request-supplied key bundle is NOT authentication.

The normalized snapshot has exactly `schema_version`, `issuer` and `keys`; each
public key has exactly `kty`, `use`, `alg`, `kid`, `n`, and `e`. This is a reviewed
input format, not raw key discovery or a mutable on-disk automatic trust source.
Unknown key IDs fail closed without request-driven network refresh. Replacing the
verifier with a freshly approved snapshot supports rotation; the old object cannot
silently trust the new key. Snapshot freshness is checked before and after signature
verification. Automated trusted metadata refresh and production revocation wiring
remain separate integration work; no live key-rotation availability is claimed.

## Identity and boundary integration

`ReceiverAuthenticator(policy, snapshot)(compact_token)` returns a frozen
`VerifiedServiceIdentity` only after verification. `from_headers(raw_header_pairs)`
rejects missing or duplicate Authorization headers before accepting exactly one
Bearer token. The hosting adapter must preserve raw duplicate headers, enforce TLS,
request/body limits and authenticated admission before dispatching any model work.
Forwarded principal, Cookie, owner and plan headers confer no authority here.

The returned tenant/client/object identity is not an authenticated Kova user ID or
an `ExecutionGrant`. The application still needs its real user session validation,
project/conversation ACL checks, current plan/Work entitlements, cost admission and
independent execution authorization. Tokens are bearer credentials, not single-use
requests: request idempotency and replay-safe work execution remain the job service's
responsibility. Do not expose a model server merely because this module passes tests.

## Dependency and offline checks

The CPU test lock `requirements/receiver-auth-py312-linux.lock` pins PyJWT 2.14.0,
cryptography 50.0.1, cffi 2.1.1 and pycparser 3.0 with wheel SHA-256 hashes from their
maintainers' PyPI metadata. It targets CPython 3.12, Linux x86_64 (glibc 2.34+) only.
Other platforms require separately reviewed wheel hashes, not a source-build fallback.
The Verify workflow installs these into a temporary virtual environment with
`--require-hashes --only-binary=:all:` and checks the dependency graph, then runs the
whole existing suite and receiver tests. Installing public CPU packages is not a
model download, Azure action, GPU execution or production deployment.

The test keys are generated locally in memory and discarded; no private key,
real access token or tenant secret is committed or archived. Tests exercise genuine
RSA signatures, tampering, unknown/mismatched keys, audience/tenant/role/caller
rejection, V1/V2 separation, lifetime/expiry/skew, disabled behavior, key rotation,
malformed input, duplicate headers and service-versus-user identity separation.
The source preflight rejects enabled guards or guessed identity/key selections.

Importing this module or constructing a disabled verifier does not load the optional
crypto backend, read keys, resolve DNS or contact a service. Active verification
is source-tested but not wired to the deployment's request boundary. Application
startup, actual Entra claims/roles, private reachability, metadata refresh, live
negative-auth tests and independent security review remain required.

## Primary references checked September 16, 2026

- Microsoft access-token validation and issuer/key scope:
  https://learn.microsoft.com/en-us/entra/identity-platform/access-tokens
- Microsoft audience, tenant, subject and actor claim authorization:
  https://learn.microsoft.com/en-us/entra/identity-platform/claims-validation
- PyJWT signature/claims API and fixed algorithm guidance:
  https://pyjwt.readthedocs.io/en/stable/api.html
- Maintainer wheel metadata and hashes:
  https://pypi.org/project/PyJWT/2.14.0/
  https://pypi.org/project/cryptography/50.0.1/
  https://pypi.org/project/cffi/2.1.1/
  https://pypi.org/project/pycparser/3.0/

These sources justify the contract and dependency selection, not a claim of verified
live Azure authorization or Phase B readiness.
