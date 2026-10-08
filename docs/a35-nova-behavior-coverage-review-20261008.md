# Nova behavior-v10 coverage review — October 8, 2026

Reviewed PR #63 at `1f053d88c151d75cc4c6132883dd66b5a00313f3`.
The correction covers the reported failure classes with consistent synthetic
labels. It is a plausible input hypothesis, not evidence that Nova can pass
these tasks. Latest measured quality remains **23/36**; behavior-v10 is
**UNMEASURED**, Phase A is **30/40**, and A35 remains **OPEN**.

## Evidence boundary

This review reads the public measured-result summaries, archived case
definitions, all 36 proposed rows and their review references, and the complete
134-row prepared pack. The current 13-failure mapping below is reconstructed
from the published persistent-failure history plus the two October 5
regressions. The private October 5 raw answers and comparison archive were
not available in this checkout; this review does not claim to replay them or
independently authenticate their completion evidence. In particular, older
raw answers must not be presented as the October 5 outputs.

The October 5 summary reports 11 content-involving failures and two format-only
failures, with all 36 outputs complete. That does not support increasing the
generation limit, weakening grading, repairing outputs, or attributing the
decline uniquely to an additional epoch.

## Coverage of the reported failures

Each group below supplies **two training and two validation replacements**.
IDs have the prefix `a35-nova-behavior-`, followed by the group, variant `0` or
`1`, and split `train` or `validation`.

| Reported cases | Group | Supervision actually present | Remaining limitation |
|---|---|---|---|
| `math-03`, `math-05` | `arithmetic` | Group subtotals before multiplication; multiple lots, opening balance and removals; checked and final-only JSON. | Word-problem ledgers differ from compact arithmetic expressions; correct labels do not establish transfer. |
| `math-07` | `probability` | Declining population, repeated-color eligibility, ordered three/four-draw events and reduced exact fractions. Physical-outcome enumeration independently checks labels. | All four use a `fraction` array property; the failed case requests separate `numerator` and `denominator` properties. Generic key-preservation instructions are not measured schema transfer. |
| `math-10` | `geometry` | Fence plus gate/openings, derived sides, area minus cutouts; known-length training and ratio-based validation. | More elaborate geometry than the failed rectangle problem; formulas and boundary checks alone cannot prove simpler-task accuracy. |
| `math-08` (new regression) | `equation` | Distribution, variables on both sides, a negative-root validation example and substitution checks, with final-only variants. | The original 48 synthetic solver checks did not verify the four equations as written in model prompts. Added regression below closes that source-test gap. |
| `code-01`–`code-04` | `predicate` | Select before transforming; quality flags, negatives, zero, duplicates, order, loop and comprehension forms. | Two training examples serve four measured failures. The text mentions empty selections but these four actual results are all nonempty. |
| `code-06` | `copy` | Separate outer containers with shared nested lists, outer append/pop, dictionary field rebinding and alias rebinding. | Both new training examples involve nested sharing. The replaced `a35-nova-copy-isolation-train` had a more direct flat-copy/alias contrast; outer independence is still represented, but this replacement is not demonstrated to improve it. |
| `code-07` | `increment` | Signed repeated updates, current-state cross-key addition, absent-key defaults, and aliased subtraction/multiplication. | Small coverage of sequential state changes; no evidence of transfer to the failed simple compound assignment. |
| `logic-05` | `string_root` | Four quoted root-string targets, including dependency and state-selection tasks; no wrapper. Existing tests reject bare text and distinguish object roots. | The algorithm is provided as Python; the failed task is prose. Correct serialization in these inputs does not establish unaided dependency reasoning or output formatting. |
| `data-04` (new regression) | `median` | Odd/even, frequency-expanded and filtered samples, with object roots; checked variants expose sorted values/count while final-only variants contain only `median`. | Every example supplies the calculation and output construction as Python. The failed task requires deriving both from prose. Only one new training target is a final-only median object. |

The 36 replacements contain only **18 training examples**; validation examples
are not additional training exposure. The pack remains 73 train / 61 validation.
All 98 retained rows, including the 18 identity overrides, are unchanged.
Lexical/structural similarity checks establish zero new detected overlaps, not
semantic independence or generalization. Correct examples already existed in
earlier unsuccessful screens, so this review does not prescribe further
speculative curriculum expansion.

## Smallest correction made

Only `evaluation/test_a35_nova_behavior_data.py` changes executable source.
Two regression tests strengthen checks of the existing examples:

1. Compare the **actual model-visible Python** with the review source used by
   both the label oracle and code-similarity audit, for all 20 Python rows.
   Negative controls reverse a selection predicate and replace a median object
   with an array while leaving metadata and target intact. The prior
   label/reference comparison accepts that prompt-only inconsistency if file
   pins are refreshed; the new assertions reject it.
2. Substitute the target root into **both written sides of all four authored
   equations**, using exact arithmetic independently of the coefficient-based
   root oracle. Verify requested substitution values and reject an off-by-one
   root for every example. This checks distribution and signs in the actual
   questions, not just the synthetic solver's internal consistency.

The negative controls are temporary test fixtures. No held-out case, reference
answer, raw model output, grader, training input, split, prompt, authorization,
launcher, budget, receipt or screen-plan binding is edited. Thus the existing
behavior-v10 tokenizer evidence remains attached to the same input bytes; no
new tokenizer run or model-quality result is claimed.

## Local verification and preserved limits

`python3 scripts/with-synthetic-identity-policy.py python3 -m unittest
evaluation.test_a35_nova_behavior_data training.test_a35_nova_behavior_screen`
passes **17 tests** (15 existing plus two new). This includes the six new
negative-control subcases: two prompt corruptions and four wrong roots.

Whole-pack preparation/similarity validation reproduces
`9d7045a87fce0c2e2748d5c0b658e4f0e5cc598b82a90e02406edf2754f167a7`.
There are zero new similarity issues; the 24 disclosed historical identity
answer duplicate pairs remain unchanged. These are CPU source checks only.
All 44 file references across the correction manifest and screen plan still
match. Publication uses `[skip ci]` to avoid starting hosted execution; the new
commit has local verification, not a new hosted-CI pass claim.

No model inference, training, GPU allocation, paid execution, merge or deployment
is authorized or performed. The consumed October 5 grant remains unusable.
The existing 600-second training cap, $5 all-in bound, 70/75/90-minute bounds,
cleanup reserve, three-family identities and all evaluation standards remain
unchanged. A35 still requires 36/36 in every required repetition (108/108),
strict-gated 14/14 manual cases and all applicable 48 criteria per repetition,
zero applicable identity/safety/grounding failures, and the separate operational
proofs. This review supplies no execution authority and no quality credit.

## Sources

- [PR #63](https://github.com/Blockigaming/KovaGPT_Models/pull/63)
- [October 5 measured summary](a35-nova-three-epoch-result.md) and
  [machine-readable record](../evaluations/a35-nova-three-epoch-result.v1.json)
- [25/36 baseline](a35-nova-grouped-sum-result-20261004.md) and
  [earlier case-by-case failure history](a35-nova-transfer-result-20261003.md)
- [Behavior proposal](a35-nova-behavior-correction-20261006.md),
  [36 source rows](../data/a35-nova-behavior.v1.draft.jsonl), and
  [reference metadata](../data/a35-nova-behavior-review.v1.json)
