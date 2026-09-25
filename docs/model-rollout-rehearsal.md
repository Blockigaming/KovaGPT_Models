# Disabled model rollout and rollback source (A34)

This component is **synthetic source verification only**. It installs no Azure
client, public listener, background controller, release workflow, deployment,
model execution or application routing. Existing application providers/fallbacks,
all 25 selections, Core/Ultra policy and every current execution guard are unchanged.

## Source and trust boundaries

`release/rollout.py` composes explicit baseline/candidate revision traffic rules
for each of the Core and Ultra staging apps and persists proposed transitions in
an isolated SQLite rehearsal journal. All plans and observations are labelled
`synthetic_rehearsal`; live/production input is rejected. A successful rehearsal
never sets `phase_b_ready` or claims actual traffic changed. Do not relabel real
observations as synthetic to use this code as a live controller.

`config/model-rollout.v1.json` is disabled, with no release plan, selected canary
percentages or observation-validity duration. CLI invocation validates only that
source contract. There is no `--execute`/`--apply` option or network callback. The
10/50/100 percentages and millisecond values in tests are synthetic fixture
choices, **not approved production policy or recovered model timing requirements**.

Both baseline and candidate carry exact revision names, image SHA-256, approved
model-manifest SHA-256 and upstream model revision. The source commit and digest
of current policy/router/planner/evaluation source bind the plan. Hashes establish
consistency, not vendor provenance or authenticated approval. No real model, image,
account, resource or release candidate is selected in checked-in configuration.

## Rehearsed transition rules

1. Begin only from an observed healthy baseline at 100% for both engines.
2. Before each incremental proposal, require fresh, plan- and state-version-bound
   synthetic gate evidence for all 25 routes and every existing per-route gate.
   Missing, failed, unknown or pending-human-review results cannot be dropped.
3. Record a durable intent before anything can be considered changed. A proposal
   leaves the previous confirmed traffic weight unchanged. It produces named
   revision rules with `latestRevision: false`; weights sum to 100 per engine.
4. Confirm only from a fresh observation bound to that exact intent, with both
   engine operations settled, exact identities intact and both engines at the
   intended weight. Promotion also requires baseline and candidate healthy.
   Two engine changes are **not** represented as a single atomic Azure operation.
5. Missing acknowledgement, process exit or partial application leaves the intent
   pending. No timer, restart or closed connection grants an automatic retry.
6. Rollback restores the captured named baseline at 100% for both engines without
   requiring the failed candidate's quality gates to pass. A prior pending write
   must first have separate settlement evidence; otherwise a late write could
   undo the rollback. Rollback itself remains pending until fresh exact-intent
   observation confirms it. An inactive/unhealthy/drifted baseline blocks it.

The journal uses explicit initialization, FULL-synchronous SQLite transactions,
version checks under `BEGIN IMMEDIATE`, defensive copies and bounded event
history. Tests use separate connections, rollback injection and a child process
that exits without cleanup after committing an intent. This establishes local
rehearsal behavior, **not production distributed-controller durability or Azure
management-plane compare-and-swap semantics**. No customer prompts or secrets
belong in these plan/observation records.

## Platform prerequisite and remaining live work

Azure Container Apps traffic splitting requires multiple-revision mode. The
existing disabled `infra/model-staging.bicep` deliberately still declares **Single**
and is unchanged. The rehearsal rejects Single observations. An owner-approved
live preparation must explicitly establish and observe Multiple, active pinned
revisions, private ingress, authorization and compatible health/identity endpoints
before any real rollout. This library does not switch modes or activate revisions.

Before live use, separately review/authorize the real controller and its durable
store, authenticate approval and observation provenance, serialize/fence actual
Azure operations and verify their settlement, prove baseline availability,
validate source/image/model identity, and supply approved numeric gate/observation
policies. All required Phase A/product/independent-review gates remain controlling.
Actual canary eligibility also requires real application authentication,
entitlements, approved tools, all-route browser checks, live quality/latency/cost
and cancellation evidence. Synthetic `pass` values provide none of that evidence.

A traffic rollback does not terminate running jobs, revoke direct revision access,
stop remote GPU billing, delete data, undo a database migration or roll back the
separate KovaGPT application. Existing admission reservations, queue fencing and
uncertain model attempts remain untouched. No production rollback was performed.

## Reproduction

Run `python3 -m unittest release.test_rollout -v` and `npm run validate:rollout`.
The new methods are appended to the existing combined suite; the disabled CLI is
appended to preflight. Full validation still requires the existing pinned CPU
Python dependencies, isolated PostgreSQL binaries and hash-pinned Bicep compiler.
Unit counts are not additional completed checklist items or independent approval.

Primary platform references, checked September 17, 2026:
- https://learn.microsoft.com/en-us/azure/container-apps/traffic-splitting
- https://learn.microsoft.com/en-us/azure/container-apps/revisions
