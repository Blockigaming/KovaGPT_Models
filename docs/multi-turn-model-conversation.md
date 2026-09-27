# Multi-turn text conversation in Core and Ultra execution

Phase A source component A19. This change does not enable application startup,
model inference, a public API, tool execution or Phase B. It is independent of
the previously blocked local-only job API publication and does not republish it.

## Contract

Core already accepts a bounded list of user/assistant text messages. Ultra now
accepts the same `messages` content form as an alternative to its original `task`
string. Exactly one form is required. A new conversation must end in a nonempty
user task, can contain up to 256 messages, and preserves exact text and order.
Per-message and aggregate limits are 250,000 and 750,000 characters. Invalid UTF-8,
reserved internal artifact placeholders, extra metadata and privileged roles
(system, developer, tool, function) are rejected before token counting. No turn
is silently omitted, summarized or truncated. The application must still supply
only the authenticated conversation's authorized messages; this validator is not
an ownership proof or a credential verifier.

The last user turn remains the current task for existing deterministic specialist
selection. Earlier turns remain context for every selected specialist and every
later Ultra stage. This does not add automatic history-aware domain classification,
external retrieval, images/attachments, project files or tool-result replay.

## Execution integration

Every Ultra template starts with the same three trusted Kova/behavior/stage system
messages, then the full conversation, then declared prior-stage artifacts. Artifact
message indices are computed after the complete conversation rather than assuming
one user message. The binder checks the exact conversation segment and exact ordered
dependency targets, including stages with no prior artifacts. It cannot overwrite
an older assistant turn to bind a private intermediate artifact.

New plans save `conversation_messages` in the existing canonical execution snapshot.
The worker's source-drift check reconstructs the plan from that full history before
constructing an inference client. Clean-frontier journal restart preserves the same
history and does not repeat completed stages. Existing owner isolation, phase
visibility, cancellation, finite telemetry and Azure reasoning suppression remain
active. No new database schema or production persistence behavior is introduced.

Legacy task-only requests continue to produce the same plan schema; a single-user-
message request produces the equivalent plan plus its explicit conversation field.
No legacy application cached-chat normalization or alias is modified.

## Budgets and limitations

The trusted tokenizer sees the full history at every planning and bound-request
check. History consumes the existing Ultra admission budget, so actual per-stage
reservations can shrink with longer prompts. The total admission ceiling, 2–5
specialist policy, single conditional debate, Chat/Work names, profiles and output
ceilings are not changed. An oversized request is rejected rather than silently
losing history or granting extra budget. Synthetic tokenizer fixtures are not
actual model tokenizer or GPU performance measurements.

This provides model-side text-context execution, not completed KovaGPT application
wiring, approved tool/project context, browser answer streaming or model intelligence.
Those independent Phase A deliverables stay open. Source tests use in-memory
Azure-shaped responses and a temporary local SQLite reference journal only.

## Checks

`python3 -m unittest execution.test_conversation` exercises all twenty Core profiles
and 113 stages, all four Ultra profiles through both debate branches, restart and
no-replay behavior, exact legacy equivalence, isolation/defensive copies, malformed
and privileged input, size limits, complete token recounting, dynamic artifact
positions and snapshot drift rejection before provider-client construction.

`npm test` registers the suite with all existing tests. `npm run preflight` remains
the complete existing source-only preflight. Exact current-head test results and
review status belong to the PR; this document does not assert a pending run passed.
