# Phase A artifact-verifier startup/container wiring

**Phase B is NOT READY.** This closes the bounded source implementation of the
artifact-verifier startup entrypoint, not the whole infrastructure/loader/release
criterion in issue #10. The parent is PR #9, source commit
`c8a702e314fcc149d905c4eb3fb2e173016c2fda`, tree
`843f0b97396ccb97e91ad59d5c30cd20c48a1d54`. The prior local-only job/API/evaluation
checkpoint is not recovered, republished or counted here.

## Actual entrypoint, deliberately no model-serving path

`worker/model_startup.py` reads a protected server policy, reads its separately
located manifest, and calls the real `verify_model_artifact` with the policy's
approved digest, byte budget and cooperative verification deadline. A restart
rehashes the package; no cached readiness file or bypass token is produced.

The published policy `config/model-startup.v1.json` has no artifact selection,
manifest pin, verification limits or execution permission. Normal invocation
therefore exits **78**, reports `startup_blocked_no_approved_artifact`, and never
opens a model package. Even with an explicitly configured, correctly verified
synthetic package, **normal startup still exits 78**. There is no loader callback,
subprocess, model import, server, socket, download, package installation, execute
switch or automatic fallback in the entrypoint.

Explicit `--verify-only` returns zero only for local byte verification. Every
report still has `ready_for_serving=false`, `phase_b_ready=false`, and all
execution/production authorizations false. Its zero exit must **not** be used as
an application health/readiness check. `--check-source` returns zero only for a
valid disabled policy; it refuses an active artifact configuration.

## Trust and filesystem boundary

Only trusted server/operator code may choose `--config`. This is not an HTTP or
user-message API. Never feed it request parameters, a client manifest digest or
user-selected resource budgets. The separately approved digest must originate
outside the model package and its untrusted manifest. Verifying bytes does not
prove the person/process that approved the digest, upstream provenance, model
compatibility, capacity or release authorization.

Policy/manifest files must be bounded, protected regular files, owned by root or
the current effective user, without hard/symbolic links or group/world write
permission. The immediate containing directory must also be protected. The
canonical ancestor hierarchy remains trusted server configuration, not a defense
against a privileged host racing or rewriting ancestors. Policy and manifest
cannot be embedded in the artifact directory. Reads detect file replacement,
size/content-metadata changes and close descriptors on failure. JSON schemas
reject unknown/missing fields, duplicate keys, nonfinite values and coerced booleans.
An `authorized` field that is true, missing, zero or a string is rejected, not
interpreted as approval. Environment variables cannot select policy or grant execution.

SIGINT/SIGTERM request cooperative cancellation and restore the previous handlers.
The artifact verifier's byte limit and timeout are explicit configuration, not
invented model response/work-duration targets. Checks surround filesystem reads;
they do not claim hard interruption of arbitrary filesystem stalls. Control files
are size-bounded; production filesystem supervision remains a separate requirement.
No file permissions or artifact contents are rewritten. Failure output excludes
control-file contents, filesystem paths and tracebacks.

## Container definition: inspected source, NOT a built image

`container/model-startup.Dockerfile` copies only the two worker modules, package
initializer and two required configuration files. It has no default base image;
an independently reviewed immutable Python base image digest is a prerequisite
for any later approved build. No image digest, registry, GPU, region or runtime
has been selected. The definition itself is not a build authorization mechanism:
any future build controller must enforce the separate approval and digest pin.

The definition selects a nonroot numeric user, root-owned read-only source files,
a trusted `/opt/kova` working directory and an exec-form entrypoint. Python `-E`
ignores `PYTHON*` injection, `-S` skips site customization, and `-B` avoids bytecode
writes. Tests assemble only the declared COPY payload in a temporary directory
and execute that same command with the local Python executable. This tests module
wiring and missing dependencies, **not Docker buildability, image security,
nonroot container execution or Azure behavior**. No Docker/registry call occurs.

A later runtime must preserve a protected read-only model snapshot through actual
loading, provide separate measured model/serving compatibility and trusted runtime
identity, and enforce authentication/admission before readiness. A verified hash
cannot stand in for any of those properties.

## Reproduce locally without model execution

```sh
python3 -m unittest worker.test_model_startup worker.test_model_artifact
npm test
npm run preflight
python3 -E -S -B -m worker.model_startup
# Expected final command exit: 78, readiness false. Do not turn it into a healthcheck.
```

All new artifact evidence uses intentionally non-loadable synthetic weight bytes.
The full test/preflight registration includes the new gate without changing the
route catalog, identity, candidates, policy mappings, pass counts or token ceilings.
Record exact-head CI results on the PR; tests do not replace independent review.

## Still open under issue #10

The broader criterion still needs a trusted real loader/runtime identity handoff,
immutable image/provenance and compatibility evidence, infrastructure definitions,
and disabled release/canary/rollback tooling. Other Phase A gates include local
checkpoint reconciliation, full application/context/tool integration, real browser
answer streaming/recovery, production-storage/queue source tests, evaluation evidence,
unresolved product policies/budgets, an integrated candidate and independent review.
Do not mark that entire checkbox complete or move to Phase B on these results.

No merge, deployment, paid GPU execution, cloud/production mutation or paid review
is authorized. Existing PRs remain unchanged; this is a stacked source-only addition.

Primary references checked September 16, 2026:
- https://docs.python.org/3/using/cmdline.html
- https://docs.python.org/3/library/signal.html
- https://docs.docker.com/reference/dockerfile/
