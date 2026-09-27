# Evaluation evidence ingestion — supplemental A35 work, not closure

The historical 50-case suite and analyzer could not be recovered as source bytes
in this continuation. Its Library ZIP is listed, but raw-byte materialization is
not authorized by the current file service; the companion report has no readable
body. Historical commit `24e85a246ccdc6fc1a4175326a8eae28f0cf2921` returned 404
from the authenticated Models repository on September 17, 2026. These observations
are not permission to bypass that access boundary or reconstruct blocked payloads.

**A35 and A38 remain open.** This is a new, explicitly supplemental ingestion
schema and regression fixture, not a replacement silently labelled as the old
36-deterministic/14-human suite. The historical suite digest remains recorded as
`85d1883bab76f1d94da6a9ec63e9ddeee72e83736b9e84f13297b9d0dfe9a70a`.
No original cases/golden answers, human reviews or benchmark winner were recovered
or manufactured here. The fixed Phase A denominator remains 40.

## Implemented source

`evaluation/quality_evidence.py` ingests a supplied pinned suite and evidence
bundle without calling models, retrying requests, browsing, executing tools,
running generated programs, training, changing account state or invoking a judge.
It uses the existing `build_route_manifest()` and response-contract evaluator,
so all 25 actual route IDs and the Core/Ultra split remain controlling. Auto
records the actual selected explicit route; it is not a seventh Chat engine.

Every supplied case remains in the expected denominator for all 25 selections and
both cold/warm conditions, including missing outputs. This evaluation denominator
is not the separate fixed 40-item engineering denominator. Uncertain/cancelled
outcomes stay explicit. Multiple successes, ordinal gaps and attempts after an
uncertain/cancelled result are rejected rather than picking the best response.
Existing execution admission, cancellation and no-uncertain-replay semantics are
unchanged. Accepting historical failed-then-success records does not authorize a
retry or prove that a retry was safe when it happened.

Exact-JSON scoring rejects duplicate object keys, nonfinite numbers, malformed
JSON and executable/code-fenced answers. It compares canonical typed JSON, so
`true` does not equal `1`. It never executes a model-generated answer. Human-rubric
results require separately supplied, exact suite/output/attempt-bound records
for every declared rubric dimension. Absent or pending reviews stay pending;
failed dimensions fail. No automated evaluation here impersonates a human reviewer.

Attempts bind case content, source commit, exact answer digest and declared
provider/model/revision/image/manifest/engine/serving version, context, concurrency,
GPU type/count and quantization. Conflicting identities for one lifecycle and
configuration drift across retries are rejected. Different configurations, routes
and cold/warm conditions stay separate in latency summaries. Acknowledgement,
first visible answer token and completed-answer time have separate sample counts,
missing counts, mean, median and nearest-rank p95. Missing values stay null; an
acknowledgement never fills in an unknown first-answer-token measurement.

Accounting retains every supplied attempt, including failures. Optional integer
micro-USD values require unique allocation-receipt digests; duplicate receipts
cannot be charged twice by this summarizer. They must represent already-reconciled,
nonoverlapping per-attempt allocations, not the same shared lifecycle bill copied
into each attempt. Unknown cost is not zero; a complete reported total is null
when any supplied attempt lacks cost. The known subtotal and missing count remain
visible. Totals cover **supplied attempt records**, not unperformed/missing cases.
The existing lifecycle/candidate accounting tools are untouched. Reconciling real
startup/idle/shared GPU, tools, platform/storage/payment costs and failures against
actual billing still requires trusted external accounting evidence; these input
receipt hashes do not prove correctness or completeness of that allocation.

Reports contain hashes, IDs, counts and aggregate timings, not answer bodies,
prompts, private-reasoning text, review prose or credentials. The existing response
checks provide an additional source-regression check, not a replacement for the
real private-output filter, tool receipt authorization, factuality or human review.
Use public-safe fixtures or an approved protected workspace for real input files;
this CLI does not install encryption, retention or an upload service.

## Provenance is deliberately not promoted

Only `synthetic` and `recorded_unverified` bundle kinds are accepted. A file claiming
to contain real output is not authenticated proof of a running model. The report
always sets provenance authentication, verified reviewer identity, historical-suite
reconciliation and Phase B readiness to false; its verified-live route list stays
empty. Even 100% fixture passes do not certify quality, live routes or source
completion. It selects no model, price, entitlement, latency target or active-work
budget. All unresolved product requirements and independent review gates remain.

## Reproduction and future integration

Run `python3 -m unittest evaluation.test_quality_evidence -v` and
`npm run validate:quality:evidence`. A no-argument CLI invocation prints a source-only
status, without claiming to have evaluated a dataset. For a supplied new-schema
suite/bundle, use:

```
python3 -m evaluation.quality_evidence SUITE.json BUNDLE.json SUITE_SHA256 SOURCE_COMMIT
```

Both pins are required, input JSON is bounded/strict, invalid Unicode is rejected,
and CLI validation/read failures do not echo input paths or payloads. No input
file is modified.
The two cases in the test fixture are authored regression inputs only. They are
not a recreated 50-case benchmark and never count as a completed Phase A item.

To close A35, first recover and verify the original archive and historical suite,
reconcile its schema/cases/analyzer and golden answers into the current revision,
then rerun the integrated evaluation checks. Add authenticated runtime/receipt and
review provenance as part of the separately reviewed application/live evidence
path; this ingestion helper cannot provide those attestations itself. A38's
publication-block exclusions remain controlling throughout reconciliation.
