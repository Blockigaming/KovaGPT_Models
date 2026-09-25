# Work-only usage reference and current integration boundary

The September 17 owner instruction is recorded in Models issue #10 comment
5721171944. Pro has five times the Kova Plus Work allowance; included paid Chat
has no weekly Work debit. Fast Work consumes 1.5 times customer allowance. An
internal 1.25-times cost target is not actual provider billing, a margin guarantee
or a permitted explanation of customer charges. The base Plus allowance, weekly
reset anchor and verified faster backend are not configured by this source.

`usage/work_week.py` implements an isolated SQLite reference ledger. Its trusted
server callbacks supply the authenticated current grant, administrative week
approval, reconciled receipts and optional verified Fast availability. It does
not accept a request's owner ID as authentication, make a payment, execute a job,
create an API or change any current plan grant. In particular, the published old
Plus Work ceiling remains enforced by the imported grant until the separate new
entitlement implementation is successfully published and verified. This module
does not route around that blocked implementation.

## Implemented behavior

An explicitly configured seven-day window and positive base allowance are
required. Pro capacity is exactly five times that base, not five times the
remaining balance. Upgrades/downgrades preserve consumed and reserved units;
a downgrade below consumption shows zero, not a fresh allowance. A new week
requires its own trusted configuration. The display never assumes a reset merely
because a local calendar advanced. The original per-job deadline is not renewed
or shortened to fit a weekly window. A valid job may cross the reset boundary.
Any outstanding reservation from an earlier period still reduces currently
available capacity until its receipt proves settlement. The snapshot separates
current-period and previous-period holds; settling prior work attributes its
usage to the original period rather than charging it again in the new week.

Reserve an entire trusted estimate before dispatch, pinned to the immutable job
snapshot. Reservations are atomic under `BEGIN IMMEDIATE` and idempotent by owner
and job. Conflicting quotes fail closed. Fast requires explicit consent and a
separate backend-availability callback; it is disabled by default. Charge
`ceil(3 * standard_units / 2)` once per complete Fast job, not separately per
token, stage or streamed fragment. The 1.25-times internal target is not used to
debit customer allowance.

A settled receipt must bind the exact job snapshot, reconciled units and explicit
quiescence. Failure or cancellation alone does not release a reservation.
Uncertain work keeps its full hold; duplicate receipts cannot be charged/refunded
twice. A receipt cannot be allocated to two jobs. Existing execution idempotency,
uncertain-attempt fencing and remote billing verification remain separate.

Paid Chat returns an unmetered result without starting a Work SQL transaction or
reading Work configuration, quota or expiry. Tests exercise actual reference Chat execution after exhausting Work.
This does not remove existing application daily caps or existing infrastructure
admission limits, because this reference is not yet wired into that application.
No UI or commercial claim of live unlimited Chat is justified by these tests.

The owner-scoped snapshot includes used and reserved units, remaining units,
floored remaining percentage, the configured reset timestamp and observation
time. Missing/expired policy is unavailable/null, never an invented 100%.
Callbacks changing owner/grant during a call cannot commit or disclose the old
owner's data. Persisted quote/reservation/receipt fields must agree; incoherent
records fail closed instead of appearing as spare capacity. Storage exceptions
are sanitized, and uncertain commits never authorize dispatch. Reservation
responses explicitly distinguish new, existing-held and settled records and
always state that they do not authorize model dispatch or replay. Read/callback failures are sanitized. No prompt, answer, provider
secret or payment details are stored in this ledger.

## Production work still required

The reference uses a separate database from the existing execution store. A real
integration needs one atomic admission transaction/outbox (or retained ambiguous
reservations and explicit reconciliation), authenticated receipt provenance,
correct cost-to-usage allocation, protected storage, retention and an owner-scoped
Settings endpoint. It must not silently release a hold when a job write fails or
make an uncertain job replayable. Frontend callbacks and consent IDs are not
backend authorization. Neither an HTTP/job service nor that distributed
transaction integration is implemented here.

Run `python3 -m unittest usage.test_work_week usage.test_work_week_integrity -v`. Tests use isolated SQLite,
separate connections and synthetic reference model callbacks. They include
consumption, settlement, races, restart, downgrade, expiry, changed authentication,
Fast consent, missing observations and paid Chat independence. These methods are
added to the existing suite without dropping previous checks. This is additional
source engineering, not another completed checklist item or Phase B approval.

No merge, deployment, cloud/Stripe/production database mutation, real model/GPU,
weights/images or paid review is performed. The blocked authorization, streaming,
tool/job service and Auto timing payloads are absent from this changeset.

## Continuation verification and remaining state

Recovered the three uncommitted files from successful Git tree
`45997b816d5a2d14b5f7caa0696d3dee04d44338`, based on PR #27. The original
27 methods passed. New negative controls reproduced lost cross-week holds,
incorrect cross-week deadline rejection, ambiguous idempotency replies,
corrupt-state acceptance, an authorization change during the final clock read,
and unsanitized storage failures. These were repaired before publication.
The new tests also prove that included Chat does not touch a failed Work SQL
transaction. Existing reset expectations were tightened to retain unresolved
holds, not weakened to pass the new checks.

This remains a reference ledger, not current live subscription enforcement.
The latest requested Free/Plus/Pro presentation and widened Plus Work entitlement
are recorded in issue #10 but are not installed by this changeset. No customer
has been charged and no account balance or subscription has been read or changed.
The absolute Plus weekly allowance and reset anchor remain server inputs. Fast's
1.25-times internal-cost figure remains an unmeasured target, never a fact shown
as provider cost or a promise of profitability.


## Persisted Fast settlement integrity

Persisted Fast debits must be reachable from a whole standard-unit count under
`ceil(3 * standard_units / 2)`, not merely fall between zero and the original
reservation. For example, a stored Fast debit of 1 or 4 cannot be produced by
that rule. Accepting it could expose invented remaining capacity and allow
another reservation from an incoherent record. Standard non-Fast settlements
may legitimately use those values. Zero-use reconciled Fast settlement remains
valid.

The record validator now checks this property using exact integer arithmetic
before snapshots, repeated reservations, settlement replies or aggregate
admission can consume the record. Invalid persisted state is not repaired or
refunded automatically; operations return the existing sanitized unavailable
error and leave it unchanged for authorized reconciliation. Reopening the
reference database does not bypass the check. The check proves internal
numerical consistency only, not authenticity of a stored receipt or protection
against arbitrary database-administrator changes.

Five additional regression methods cover impossible debit residues across those
paths, database reopen, every reachable charge for standard counts zero through
100, unaffected standard settlements and exact arithmetic at the supported
integer ceiling. Together with the existing ledger tests, 43 methods pass
locally. The negative control on the prior implementation produced 22 assertion
failures across related cases with no harness errors; these are not 22 separate
vulnerabilities. No existing test or hosted gate was removed.

This correction does not select a weekly allowance/reset anchor, enable Fast,
change any entitlement or implement the latest application policy. The later
owner refinement in PR #25 comment 5722472946 still requires separate
application/authorization work and supplied product inputs. It adds no fixed
Phase A checklist credit and is not live billing or model-serving evidence.
