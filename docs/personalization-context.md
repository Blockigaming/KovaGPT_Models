# Kova personalization context contract

Kova personalizes responses through bounded request context, not per-user weight
training. The approved layers are the current conversation, explicitly approved
cross-chat memories, owner-scoped custom instructions, authorized project/tool
context and session corrections. Corrections remain session-only unless the user
separately approves saving them as memory.

`router.personalization` accepts only trusted, owner-scoped records. It rejects
cross-owner data, unapproved memories, unauthorized task context, oversized input
and any attempt to supply plan, model, provider, effort or execution fields.
Personalization therefore cannot unlock a model, widen an effort level, authorize
tools or paid compute, or change model weights.

The renderer produces private request instructions for the selected Kova route.
It is not public metadata and must not be logged with prompts, memories, files or
tool results. Production integration still requires authenticated owner storage,
memory consent/revocation, deletion propagation, retrieval relevance, project ACL
checks, retention controls, prompt-injection defenses and end-to-end isolation
tests. This source contract performs no database, network, tool, inference,
training, deployment or paid operation.
