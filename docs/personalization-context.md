# Kova personalization context contract

Kova personalizes responses through bounded request context, not per-user weight
training. The approved layers are the current conversation, explicitly approved
cross-chat memories, owner-scoped custom instructions, authorized project/tool
context and session corrections. Corrections remain session-only unless the user
separately approves saving them as memory.

`router.personalization` accepts only trusted records bound independently to the
authenticated owner and active conversation. It rejects cross-owner or
cross-conversation data, unapproved memories, unauthorized or unattributed task
context, invalid UTF-8, oversized input and any attempt to supply plan, model,
provider, effort or execution fields. The aggregate limit counts every supplied
string value, including identifiers, revisions and digests, rather than only the
rendered text.
Personalization therefore cannot unlock a model, widen an effort level, authorize
tools or paid compute, or change model weights.

Project-file and tool-result context carries a source type, identifier, revision
and SHA-256; tool results additionally carry the tool and call identifiers. The
renderer preserves that attribution in the private request instructions for the
selected Kova route.
It is not public metadata and must not be logged with prompts, memories, files or
tool results. Production integration still requires authenticated owner storage,
memory consent/revocation, deletion propagation, retrieval relevance, project ACL
checks, retention controls, prompt-injection defenses and end-to-end isolation
tests. This source contract performs no database, network, tool, inference,
training, deployment or paid operation.
