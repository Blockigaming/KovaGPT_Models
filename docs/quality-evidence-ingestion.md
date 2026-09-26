# Evaluation evidence ingestion — supplemental A35 work, not closure

When this supplemental ingestion source was written on September 17, the historical
50-case suite and analyzer were unavailable as readable bytes. On September 26, a
read-only recovery of the original September 16 checkpoint archive verified its
checksums and historical source tree; the original suite is now available for
reconciliation. The historical commit `24e85a246ccdc6fc1a4175326a8eae28f0cf2921`
was not published to the current Models branch. See the recovery evidence below.
This source remains supplemental; archive recovery does not publish or authorize
the historical execution/service implementation.

**A35 and A38 remain open.** This is a new, explicitly supplemental ingestion
schema and regression fixture, not a replacement silently labelled as the old
36-deterministic/14-human suite. The historical suite digest remains recorded as
`85d1883bab76f1d94da6a9ec63e9ddeee72e83736b9e84f13297b9d0dfe9a70a`.
The supplemental source did not include the original cases/golden answers, human
reviews or a benchmark winner. Recovery has since located the historical case bytes,
but no case payload, execution code, review or model result is copied into this
branch. The fixed Phase A denominator remains 40.

## September 26 read-only recovery evidence

The original `KovaGPT_Models_Source_Checkpoint_2026-09-16.zip` has SHA-256
`ab454f7131a74619e0ba36d521ac73942643fdf252b118d685a822058b1fad33`.
All 152 paths listed by its `SHA256SUMS` matched their archive bytes; there were
no duplicate, unlisted or unsafe member paths. Isolated extraction of
`Model_Source/` reconstructed the recorded Git tree
`54483aa5d055e68b822bafb16dc192f2fdc07da0`. This verifies the saved tree's
bytes, not its integration into the current branch or approval of its behavior.

The original `evaluations/model-quality-suite.v1.json` has file SHA-256
`8c91f70c8e4d0522c48aa7b1588b9c427e49a0f5a2a9b0bfc0c5091320c3150a`
and canonical content SHA-256 `85d1883bab76f1d94da6a9ec63e9ddeee72e83736b9e84f13297b9d0dfe9a70a`,
matching the previously recorded historical-suite pin. It defines 50 cases:
36 exact-JSON cases and 14 requiring manual review across nine categories. It
does not approve a release or paid execution, and its repetition counts, route
thresholds and latency targets are unset. In the isolated historical tree,
`python3 -m unittest evaluation.test_quality` passed 22 source tests and
`python3 -m evaluation.quality` reported the same case counts with zero
provider calls and release approval false. These are historical source checks,
not current-integrated evaluation evidence.

An in-memory, read-only case mapping satisfied this supplemental ingester's
schema. An empty, unverified bundle against the current 37-route manifest
accounted for all 3,700 case/route/cold-warm units as missing, with zero
attempts, zero reviews and `phase_b_ready: false`. That mapping drops historical
category and policy metadata, supplies no reviewed source URLs, and does not
reconcile the original analyzer with the current policy. It is neither a
published replacement suite nor an evaluation pass.

Of 119 tracked files in the recovered historical tree, 101 paths are shared
with the current Models checkout: 54 match byte-for-byte, 47 differ, and 18
historical paths do not exist in the current checkout. The missing paths include
the original evaluator and suite as well as the separately restricted job
service and execution source. The historical evaluator uses superseded route
assumptions and cannot be copied as current evidence. Earlier publication
restrictions on the specific job API, approved tool execution, and Auto latency
repair remain in force. A35 and A38 remain open pending a separately reviewed,
safe integration and exact-head validation.

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

The original archive and suite are recovered and verified as noted above. To
close A35, reconcile the schema, cases, analyzer and golden answers safely with
the current revision, then rerun the integrated evaluation checks. Add authenticated
runtime/receipt and review provenance as part of the separately reviewed
application/live evidence path; this ingestion helper cannot provide those
attestations itself. A38's
publication-block exclusions remain controlling throughout reconciliation.
