# Resumable Core and concurrent Ultra execution — source-only reference runtime

This step implements execution rather than merely describing a DAG. The local
scheduler calls the existing Core worker and a new Ultra worker over explicitly
injected clients. It exercises actual concurrent Python callbacks; it does not
claim concurrent Azure GPUs, a production queue, deployed models or evaluated
model intelligence. All live execution and spending remain disabled.

## Preserved product behavior

All six Chat names and eighteen Work combinations remain unchanged. Core's five
Chat profiles plus fifteen non-Ultra Work profiles retain their exact 113 stages,
identity prompts and output ceilings. The four Ultra routes preserve two to five
dynamic specialists, disagreement checking, a judge, at most one conditional
challenge, then synthesis. Only the final stage is eligible for public output.
Kova Auto remains a router; it is not a directly executable twenty-fifth model.

The only Ultra planner instruction addition is a strict JSON result contract for
the disagreement detector and judge, needed to execute the existing conditional
debate safely. Every disagreement names at least two distinct completed specialist
stages; the judge can select only recorded disagreement IDs. A malformed decision,
a guessed ID, or `"true"` instead of a boolean fails closed. This proves reference
integrity and schema, NOT the factual quality of a model's judgment.

The previous Ultra admission check accepted positive infinity for remaining
budget. A baseline negative control reproduced it; the planner now rejects
non-finite monetary values. No finite budget or model pass policy is widened.

## Internal APIs and authorization

`ExecutionGrant` must be constructed by authenticated server code. It is NOT a
replacement for authentication and must never be deserialized from client JSON.
The application supplies the current owner, tier and allowed route set. The
scheduler checks them before starting, before each stage, during cooperative
I/O checks, and before accepting a result. Explicit Chat caps deny Plus above High.
Ultra additionally requires Pro. Work routes must be explicitly authorized rather
than inferring a still-unrecovered complete Work subscription matrix.

`ExecutionSpec` contains immutable canonical bytes of the server-built plan,
selected fixture model/revision/context, complete stage budgets and explicit job
limits. SHA-256 protects against accidental snapshot/artifact corruption, not a
malicious store administrator capable of rewriting both a record and its hash.
The runtime client factory remains server-owned; no model or destination is
chosen from a user message. Legacy Free Thinking is deliberately not mapped.

## Local journal and safe restart

`LocalJobStore` is a SQLite reference implementation for synthetic local tests.
Atomic write transactions guard owner-scoped idempotency keys, budget reservations,
stage claims and ordered events. Separate connections cannot claim a stage twice
or exceed the configured parallel width. Each runner owns a token and increasing
epoch; stale completions cannot overwrite a new supervisor decision.

Before every model-stage dispatch, the worker rebuilds the pure Core/Ultra plan
under the current server policy and trusted tokenizer and compares the complete
canonical snapshot. A changed instruction, dependency, identity, or token budget
fails before client construction rather than mixing old completed artifacts with
new policy. An existing job is not silently upgraded. Both runner clocks are
validated before taking ownership; invalid clock data cannot strand a queued job.

A clean pause happens only between complete stages. Reopening the journal and
resuming reuses completed artifacts without repeating provider calls or adding a
fresh deadline. Repeating the same owner/idempotency key with changed request,
model, plan or limits is rejected. Repeating a completed job is inert.

A crash with an in-flight attempt is different: provider execution and billing
may already have happened even when no result was committed. Confirmed supervisor
recovery marks that attempt uncertain and the job interrupted. It does not replay
or refund reservations. Automatic lease stealing and automatic retries are absent.
A crash strictly at a completed frontier can return to paused; a failed frontier
stays failed, and stale runner epochs remain fenced out.

The journal's public status and replay methods expose lifecycle metadata only,
not prompts, intermediate artifacts, private judge text, model errors or tokens.
The final result is unavailable until every required stage succeeds. Tool-call
responses become `waiting_tools`, not a successful answer; no tool is executed.
Private server methods are not public web endpoints and must not be exposed as such.

## Deadlines, cancellation and cost

Job expiry, stage timeout, maximum parallelism, token limits and per-stage cost
reservations are explicit server inputs with no production defaults. The job's
absolute deadline includes queue/pause time and survives restart. It is deliberately
NOT presented as the unrecovered per-mode active-work duration or average response
time. A simulated 300-second Max run spans ten separate 30-second stages under a
600-second job deadline; this is a logical-clock test, not a latency measurement or
proof that one Azure HTTP stream lasts that long.

Actual running worker callbacks receive cancellation and remaining-time controls.
The existing Azure client's bounded I/O can use them. The scheduler drains started
workers before releasing runner ownership. Python threads cannot forcibly stop
arbitrary noncooperative callbacks; a production supervisor, verified remote abort,
and a separately approved durable queue are still required. Closing a local socket
is not proof that remote GPU billing stopped.

Reservations include stage inputs, outputs and propagated artifacts. A skipped
debate is not reserved/executed. Failed, cancelled or uncertain work does not
silently release potentially consumed budget. `cost_reserved_microusd` is explicitly
an admission reservation, NOT a measured invoice, a margin calculation, or a hard
guarantee on external charges. The complete account-wide budget/queue admission
service and actual provider lifecycle-cost reconciliation remain unimplemented.

## Truthful activity and parallelism

Only persisted stage-start events can become activity headings and explanations.
The stage callback has actually started when that event is written; it does not
claim an Azure GPU, web browser or external tool has started. No fake timer,
percentage, source URL, tool count or third-party icon is emitted. Stage activity
is omitted for Instant and other profiles whose current policy disables it.

Ultra's specialist callbacks are dispatched concurrently up to the explicit
server limit. Barrier tests prove overlap at widths two, three, four and five.
Downstream stages wait for their dependencies, and either debate branch rejoins
at synthesis. Specialist failures, invalid judge output, ownership/entitlement
revocation and model-revision mismatch prevent successful final publication.

## Verification and reproducibility

```sh
npm test
npm run preflight
python3 -m unittest execution.test_contracts execution.test_store execution.test_runner ultra.test_binding
```

Tests use synthetic model responses and credentials through the Azure-shaped
in-memory adapter. They exercise every Core profile/stage, every Ultra route and
both debate branches, genuine local callback overlap, restart/idempotency,
owner-isolated replay, late cancellation, stage expiry, invalid output/usage,
corrupted artifacts, runtime revision quarantine and unknown in-flight recovery.
A failed pre-fix expiry classification was also reproduced and corrected by
rechecking server controls after a sanitized nested-transport error.

## Not a production storage or deployment decision

The reference journal may contain prompts and private artifacts as plaintext. It
is restricted to synthetic tests until encrypted external storage, access policy,
retention/deletion and production supervision are implemented and reviewed. Do
not deploy this SQLite file on an ephemeral Container Apps filesystem and call
it durable. Microsoft documents that container state must not be expected to
survive restarts. This step does not change Supabase or choose another production
data platform.

Still absent: a deployed queue/supervisor, cross-instance account admission,
production storage integration, real Azure GPU runs, measured Ultra quality/cost,
worker tokenizers and model/image selection, remote abort proof, live browser
streaming/reconnection, and application UI/startup wiring. The source-only runtime
is not approval for paid execution, merge, deployment or production routing.

## Primary references checked September 15, 2026

- Python thread/future cancellation: https://docs.python.org/3.13/library/concurrent.futures.html
- SQLite transaction semantics: https://www.sqlite.org/lang_transaction.html
- Azure container lifecycle and state: https://learn.microsoft.com/en-us/azure/container-apps/application-lifecycle-management
