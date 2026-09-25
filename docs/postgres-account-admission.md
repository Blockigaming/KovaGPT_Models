# Account-wide PostgreSQL admission

A25 source/backend integration. This is a separate account-reservation adapter
for the published encrypted PostgreSQL journal, not the earlier blocked HTTP job
API or the blocked tool-execution component. Neither file is republished here.
No real account balance, charge, billing provider or financial transaction is used.

## Atomic reservation rather than independent per-job checks

AccountPostgresStore wraps inherited job creation and its account reservation in
one PostgreSQL transaction under the same cross-connection lock. Nested inherited
operations use savepoints, not separate commits or isolation resets. Failed token,
cost or outstanding-job admission rolls back the new encrypted job and events.
Idempotent retries return the original job without reserving again. Concurrent
connections cannot both consume the same last allowance.

Limits are explicit trusted server AccountLimit records, disabled by default. No
Free/Plus/Pro allowance, per-mode price, margin or period length is invented. Policy
administration requires explicit authorization and optimistic revision checks.
Existing period start/end cannot be extended; a new non-overlapping period can be
installed only after the previous one expires. Outstanding holds survive period
changes, so unresolved external work cannot be hidden by resetting the calendar.

Reservations cover the full source plan token/cost ceiling at admission. They are
conservative admission allocations, not measured bills, refunds or a guarantee of
external provider charges. Caps cannot be lowered below already recorded usage.
All model stage and permission controls remain separately required.

## During execution and cleanup

The account grant/period/reservation is checked during begin, stage claim, active
worker control checks and result completion. Revocation stops later work and blocks
successful final publication. Cancellation remains possible after execution
permission is revoked. Started failed, cancelled or uncertain work retains its
outstanding slot until a separately verified supervisor/provider quiescence receipt
is accepted. Passing a receipt string alone is not proof: the trusted callback
must authenticate the real observation. Active runners cannot be marked quiescent.

Success or cancellation before any attempt releases an outstanding slot, but never
silently refunds the financial reservation. Deleting private job payloads preserves
the separate reservation ledger and existing idempotency tombstone; deletion cannot
re-admit or refund the same request. Production ledger retention and eventual actual
billing reconciliation are distinct administrative policies, not implemented refunds.

## Schema and source verification

Account tables install only through a separately authorized administrative function
on an empty existing private execution schema. Existing jobs without reservations
cause refusal rather than a guessed migration. Runtime opening never creates the
account schema or repairs missing rows. No production schema is selected or changed.

Tests use a real private Unix-socket PostgreSQL cluster, ephemeral keys and synthetic
model results. They cover all explicit model routes, simultaneous admissions,
outstanding limits, whole-transaction rollback, unchanged retry reservations,
revocation, pause/reopen/resume, owner isolation, unknown-work holds, period changes,
policy revisions and deletion conservation. This is backend correctness evidence,
not deployment throughput or actual Azure cost measurement.

The application's authenticated entitlement/approval source and HTTP startup still
need integration; this module is not a public API accepting user-supplied policy.
No live credentials, GPUs, external database, Stripe mutation, merge or deployment.
