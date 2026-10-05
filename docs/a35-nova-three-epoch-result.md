# Nova three-epoch result — October 5, 2026

Phase A: **30/40**. A35: **OPEN**. Latest measured strict: **23/36**, down two
from the preserved **25/36** baseline. The exact authorized attempt is consumed.
The next JSON-shape prompt proposal is **UNMEASURED**, with execution disabled.

## Measured execution

Source `fc83f054df40e9f6f591d7166bfe1524d26a6556`; attempt
`5d4f68aa6d0a4560b50873ca8beef278`. One fresh-base T4 training run completed
three epochs and 30 optimizer steps in **445.738 seconds**, under the 600-second
cap. One strict screen produced 23 passes and 13 failures. All 36 outputs have
verified completion evidence; none reached the output limit. No second model
run or inference sweep was made.

| Category | Strict passes |
|---|---:|
| Math | 5/10 |
| Code reading | 2/8 |
| Reasoning | 5/6 |
| Data analysis | 5/6 |
| Instruction following | 6/6 |

Against 25/36: **23 retained passes, 11 persistent failures, two regressions,
zero recoveries**. The regressions are `math-08` (incorrect equation result)
and `data-04` (correct scalar in a list instead of the required object).
`logic-05` remains a failure because its string lacks JSON quotes. Eleven
failures involve incorrect content, including one that also has invalid JSON;
two are format-only diagnostics. These labels do not alter any raw answer,
reference target, strict grade or pass count. All 36 cases were also compared
with the earlier 24/36, 23/36, 21/36 and historical 23/36 screens privately.

Relative to 25/36, exposure, the identity target pack and the effective system
prompt differ. The comparison cannot isolate internal cause or attribute the
decline to epochs alone. There is no evidence that the next correction fixes
the content failures or improves model quality.

## Evidence, cleanup and billing

The original Cloud Shell session lost its ephemeral operator files during a
UI recovery after the sole guest command had already been accepted. No
operation was replayed. Authenticated resource activity, guest claim,
training receipt and raw output were reconciled against the original source,
grant, attempt and clock. A readback/cleanup-only observer independently
reverified the live watchdog and recovered the result. The missing original
ephemeral ledger is an explicit evidence gap; the replacement readbacks are
not described as the original ledger.

Raw result, claim, training receipt, case outputs and adapter are retained
privately with hashes. The recovered control bundle contains 253 hash-verified
text files. Independent cleanup completed at **21:52:16 UTC** with **8/8 PASS**:
GPU group and inventory, VM, watchdog group and inventory, workflow, Reader
assignment and Writer assignment all returned authenticated absence.
The original 21:25:24 UTC clock remained authoritative throughout.

Billing is **PENDING**: the authenticated post-cleanup query returned no posted
rows. Missing rows do not establish zero cost or prove the final all-in total
is below $5. Final reconciliation must include incremental evidence storage,
transactions, traffic and applicable tax.

No runtime identity/safety/grounding stop occurred in this strict screen.
Because strict failed, the conditional 14-case/48-criterion manual path was
not run and its file was independently verified absent. Those manual gates
remain unmeasured. Serving, repetition, deployment and other A35 gates remain
separate; this result closes none of them.

## Unpaid correction

The new task-check prompt explicitly preserves the requested JSON root type
and keys and requires quoted JSON strings. It retains the v5 identity policy
and all 134 user questions, completion targets and train/validation splits.
The historical v1 task prompt and receipts remain immutable audit evidence.
No held-out answer was added and no grader, target or output was repaired.

The proposal retains three epochs/30 steps, the 600-second training cap,
one T4, one strict screen, the $5 ceiling, 70/75/90-minute bounds and cleanup
reserve. All seven measured adapters are rejected. CPU tokenizer/collator
and source checks confer no model-quality credit. The next package is
inactive; the consumed October 5 grant cannot authorize it. No paid retry,
automatic repetition, other-family run, merge or deployment is authorized.
