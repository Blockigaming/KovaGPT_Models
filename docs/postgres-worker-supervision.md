# Durable model dispatch and confirmed worker recovery

A29 source integration. A service process explicitly calls PostgresSupervisor.run_once
to claim and execute a bounded slice of an already admitted model job. This code
installs no background daemon, network listener, public job/tool API or production
queue. Tests run a private Unix-socket PostgreSQL cluster with synthetic workers.

## Atomic queue and independent execution fences

The dispatch table references existing encrypted/account-admitted jobs. Internal
admit-and-enqueue occurs in one database transaction, so a queue failure cannot
leave a charged unqueued job. Repeat owner/idempotency submissions do not duplicate
reservations or queue records. Separate worker connections cannot claim one item
twice. Only explicit current authenticated worker-instance grants can claim work;
current user and account authorization is also checked before dispatch.

Each queue claim has its own worker instance, random token and monotonically
increasing generation. The wrapper binds the LocalRunner's independently generated
runner token/epoch to that exact queue claim in the same transaction. A stale queue
claim cannot recover an unrelated model runner. Model computation runs outside
metadata transactions, using the existing bounded worker and encrypted store.

A clean completed-stage slice returns to pending without changing its plan, original
deadline or admission reservation. Another authorized worker can continue from that
frontier without rerunning committed stages. Completed jobs are acknowledged, while
tool requests remain held; this queue does not execute tools or invent their results.

## Crash semantics

Claim age and elapsed leases never authorize automatic takeover. A trusted process
supervisor must first prove the exact claimed worker instance has stopped and pass
an authenticated observation to recover_stopped_worker. The durable claim and runner
fences are rechecked before any recovery. A stopped pre-start worker can be requeued.
A stopped worker at a completed frontier can resume. A stage that was in flight is
marked uncertain and the job interrupted; it is not automatically replayed.

The tests spawn a real local child process, wait until its stage claim is committed,
terminate and join it, then verify interrupted/uncertain state and no subsequent
model invocation. This proves local worker-process recovery semantics, not that a
remote GPU or external side effect stopped. Account reservations and uncertain
concurrency holds remain until the separate real provider-quiescence verification.

A job that succeeded before a missing queue acknowledgement is finalized without
rerunning it. Revoked user/account permissions hold dispatch; explicit authorized
re-enqueue is required after restoration. A worker that loses its own authority
cannot acknowledge on behalf of a still-live process; confirmed supervisor recovery
handles that case. No generic network/SQL retry converts uncertainty into duplicate
model work.

## Configuration and remaining integration

Worker authentication, user-grant lookup and stopped-process verification are
server callbacks, not owner/role/approval fields from client messages. Production
hosting must provide authenticated process identity and observations, bounded
callback I/O, worker lifecycle management and separately verified remote cancellation.
No actual process service, Azure Container Apps job, cloud identity or timer is
installed in this source pass. The model request admission and Azure execution gates
remain independent. The initial queue-table installation is an explicitly authorized
administrative call; opening the supervisor does not create production schemas.

Tests exercise all 24 explicit routes in bounded slices, separate-connection claims,
atomic admission/queue rollback, idempotency, pause/reopen, actual process death,
clean-frontier recovery, stale claims, owner isolation, cancellation and held tool
requests. All outputs and keys are synthetic. No real credentials, model weights,
GPU, cloud database, paid review, merge or deployment are involved.
