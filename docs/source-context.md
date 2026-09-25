# Authorized source-context execution

This source component connects scoped file/project text and completed historical
tool receipts to the existing Core/Ultra executor. It is not a public upload/job
API, an application session verifier, a storage-service connection, or permission
to run tools. No production data or credentials are used by the tests.

## Authority and data are separate

`SourceContext` is constructed by trusted application code with current-grant,
ACL/revision-check, and scoped-read callbacks. It is disabled by default. The ACL
callback must authorize the authenticated owner's conversation, any project
membership, and each exact resource ID/revision/digest BEFORE reading bytes. For
a tool result, authorization also checks its persisted completed call/receipt.
A project membership decision can authorize shared content; merely matching an
owner string from a request or accepting a request's `allowed: true` is not valid.

`ContextScope`, `SourceRef`, `SourceRecord`, and `PreparedContext` are internal
server values. They are not trusted merely because client JSON has those fields.
Application wiring and real ACL/storage adapters remain part of A24/A27. This
module provides and tests the source integration seam, not those live adapters.

Reads must return the exact scope and reference, with a canonical payload matching
the selected digest. Missing, changed, unauthorized or oversized sources fail the
whole preparation rather than being silently omitted. Duplicate tool-call or
receipt IDs are rejected. Only completed receipts are evidence; pending, failed,
cancelled and unknown work is not represented as a successful tool result.

## Model input and replay

The complete history is retained. One clearly labelled untrusted-reference JSON
message is inserted immediately before the current user task. No native system,
developer or tool role is granted to source content. The JSON includes stable
source ID/revision/hash and exact text or structured tool arguments/results.
It does not claim that a historical tool was executed again. This is evidence
projection into the text-model contract, not native tool-call replay or an image
attachment pipeline. Treating retrieved text as data is a trust boundary, not a
proof that a real model resists every prompt-injection attempt; model evaluation
is still required.

`bind_worker` uses the existing ModelStageWorker and source-drift checks. It
requires the actual execution plan to contain exactly the prepared messages and
binds to its immutable fingerprint. It rechecks current grant, resource revision
and ACL before model calls, while consuming response chunks, and before returning
stage output. Revocation withholds results and closes the stream. Production
callbacks require bounded I/O, thread safety and appropriate permission-cache
freshness; no remote database polling strategy is selected here.

The protected source-reference metadata and context fingerprint can be used by a
supervisor to rehydrate a paused job. Rehydration checks exact identity/content and
current access, and never reruns a tool. Changed or revoked context is not silently
substituted. The existing journal then resumes completed stages without replay.
Persistence/encryption of that metadata remains a separate store integration.

## Evidence

`execution.test_source_context` exercises all 24 explicit Core/Ultra routes through
the real planning, binding, kernel and response code using synthetic storage/model
fixtures. It checks exact source/history preservation, protected system roles,
recorded tools, owner/project isolation, bad references, revocation, restart,
privacy, limits and cleanup. No network, real files, model weights or paid tool
calls are involved. `npm test` includes these checks; the all-25-route offline
preflight still separately covers Auto selection.

This change does not publish or bypass the previously blocked local-only job API.
Model names, compute policies, identity and output ceilings are unchanged. Phase B
remains blocked by the full issue #10 checklist and independent review gates.
