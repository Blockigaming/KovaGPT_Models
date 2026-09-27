# Unambiguous supplied grounding evidence

The offline evaluator now rejects duplicate tool receipt and runtime operation
IDs instead of letting a later row overwrite an earlier failure. Malformed or
unidentified records remain explicit validation failures; they are not dropped.
Claims require an explicit nonempty tool and unique operation ID. URL collections
must be actual arrays of valid UTF-8 strings, not strings accidentally used for
substring membership. Identity/disclosure switches require literal booleans.
Populated private-reasoning fields fail without echoing their contents.

An activity event requires both a recorded start timestamp and an explicit
started-operation state: started, running, success, failed, cancelled, expired,
interrupted or uncertain. Planned/queued/pending/skipped and unknown states do not
prove execution began. A failed operation may still ground an honest activity
record about that operation; it cannot provide successful source evidence.
One event delivery cannot mix request IDs. Sequence numbers are exact integers,
not bools/floats, and malformed IDs return validation violations instead of
throwing unhandled type errors. Optional tool/domain/content-type/icon fields and
result counts must match their supplied runtime observations, with exact types.

The existing six response fixtures, four activity fixtures, 25-route manifest,
Chat/Work pass policies, activity permissions and Auto contracts are unchanged.
Instant/Medium Chat stays silent; every existing Work family/effort still permits
truthful activity. New local limits bound evidence collections to 4,096 items,
metadata text to 8,192 characters and answer text to the already-used 750,000
character boundary. These are validator resource safeguards, not new product
latencies, Work duration caps or changes to model compute policy.

These routines only assess the consistency of supplied records. They do not
execute a tool, authenticate receipts or owners, prove real browsing occurred,
verify a trusted icon map, judge arbitrary prose truthfulness, or enable any
public/server job API. Actual authentication, approved tool execution, persisted
runtime events and privacy-filtered browser delivery remain separate gates.
No A22/A26 blocked payload is republished or replaced by this change.

Reproduce with `python3 -m unittest evaluation.test_grounding_integrity
 evaluation.test_offline evaluation.test_quality_evidence -v` (on one line).
Eighteen new test methods include all 18 Work combinations, duplicate/failure
orders, malformed rows/URLs/IDs, private fields, typed counters and request
mixing. Existing contract and ingestion tests remain registered. This repairs
source evidence validation; it does not recover the historical 50-case suite,
constitute independent review, close A35/A38 or add a Phase A checklist item.

## Rollback settlement chronology

The same continuation reproduced a temporally impossible settlement acceptance:
a pending change recorded at fixture time 1500 could be superseded using an
observation from fixture time 1000, simply because its settlement ID matched.
The rollback rehearsal now requires that settlement observation to be no earlier
than the original pending intent. A missing/coerced/corrupt pending timestamp is
rejected rather than granting rollback or throwing an unhandled comparison error.
Invalid evidence leaves the original pending intent and journal version intact,
including across reopen. Equality is allowed for millisecond clock granularity;
no sleep, new validity duration or model-work budget is introduced.

Four additional methods in `release.test_rollout_chronology` exercise rejection,
restart preservation, the same-millisecond valid path through confirmed rollback,
and corrupt timestamp types. All pre-existing rollout tests remain unchanged.
This remains a synthetic journal with no Azure apply path, authenticated live
observation or production rollback authorization.
