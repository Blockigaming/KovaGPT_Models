# Policy decisions are not checklist completion

The owner decision recorded in Models issue #10 comment 5718592571 resolves the
A36/A37 product-policy questions. PR #26 implements that policy overlay. The
`release.product_policy` source check validates policy values and routing/admission
constants; it does not execute a complete authenticated application request,
submit or resume a job, test browser delivery, or obtain independent review.

Previously that check returned `closed_checklist_ids: ["A36", "A37"]` solely from
those static checks. During September 17 verification, Free `thinking` resolved
to Medium/Orion correctly but the subsequent existing job-store submission lost
the alias and rejected Free Medium. Policy validation passed despite this missing
execution connection. The tracker therefore kept A36 open. A37's checklist credit
was supported separately by its approved finite-budget policy, execution tests and
exact-head CI; it was not established merely by this CLI's output.

The checker now reports:

- `resolved_product_decision_ids: ["A36", "A37"]` for the approved policy decisions.
- `closed_checklist_ids: []`, `execution_integration_verified: false` and
  `independent_review_verified: false` for evidence it does not establish.
- The unchanged `product_policy_ready: true` and `phase_b_ready: false`.

These false evidence flags mean **this check did not verify those things**. They
are not a substitute for a fresh repository-wide audit, nor do they revoke a
separately evidenced checklist item. The fixed forty-item rubric on issue #10
remains authoritative. No denominator or product requirement changes.

`npm run validate:product-policy` and `npm run check:product-ready` remain
policy-only commands. Their successful exit confirms the approved source policy,
not application execution, independent review, deployment permission or completion
of Phase A. Existing output fields and exit behavior remain compatible except for
the corrected checklist-closure claim. Consumers must not interpret the new
resolved-decision list as an execution-ready list.

Four new regression methods cover report scope, both CLI modes and the absence of
job/network/external-review calls. The old assertion that incorrectly expected
checklist closure is corrected, not silently retained as proof of implementation.
All policy-value, entitlement, finite-budget, route and drift assertions remain.

This repair changes reporting only. It does not implement or publish the blocked
Free Thinking snapshot propagation, streaming, job API, approved-tool execution or
Auto timing aggregation changes. Runtime, permissions, model compute, infrastructure
and workflow files remain unchanged. No model, account, cloud or production action
is performed.

Source evidence, checked September 17, 2026:
- https://github.com/Blockigaming/Kova-1.0/issues/10#issuecomment-5718592571
- https://github.com/Blockigaming/Kova-1.0/issues/10#issuecomment-5718930290
