# Nova screen correction — October 3, 2026

**Phase A 30/40 verified (75%); A35 OPEN. No new model run or manual evaluation.**

Preserved attempt `9ce963e3b2f34b578810ee505d008965` at `1783367753b38ed662c611f7e2f45dfcda531a7e` trained successfully and scored **23/36**. Adapter SHA-256: `95470d46db4a430ce6f5e5d53f5dc322dcb4a04aec824b69b922dbe9b1d71213`. The exact unmodified raw report is `evaluations/a35-nova-screen-9ce963e3b2.json`, SHA-256 `9afd9826d43d2c620bca75e9181552c0598411e48fb938af91669a7c5013c95b`. Its 36 outputs all reached EOS; 13 failed. Manual generation/upload remained blocked. The retained independent cleanup receipt records eight HTTP 404 checks. Spend remains pending; an empty billing response is not zero.

## Exact failures

| Case/category | Expected JSON | Preserved output | Deterministic defect | Correction layer / smallest intervention |
| --- | --- | --- | --- | --- |
| math-03 / math | `{"answer":669}` | `{"answer": 612}` | Returns 204*3, omitting the parenthesized +19 term. | prompt/template: check arithmetic precedence and carries. |
| math-05 / math | `{"answer":1197}` | `{"answer": 117}` | The add-then-multiply result is wrong; no unique internal calculation can be inferred. | prompt/template: check arithmetic precedence and carries. |
| math-07 / math | `{"numerator":3,"denominator":28}` | `{"result": {"numerator": 3, "denominator": 15}}` | Wrong without-replacement denominator plus an unrequested result wrapper. | prompt/template: update both eligible and total counts after each draw and reduce the resulting fraction. |
| math-10 / math | `{"area":54}` | `{"area": 21}` | Area is 21 instead of 9*(30/2-9); it coincides with perimeter minus length. | prompt/template: Derive rectangle sides from perimeter before computing area. |
| code-01 / code_reading | `[8,4,0]` | `[4, 8, 14, 4, 18, 0]` | Odd inputs survive the predicate; the first transformed value is also wrong. | prompt/template: test the predicate on the original value, transform only survivors, and retain duplicates. |
| code-02 / code_reading | `[6,6,12]` | `[6, 6, 4, 12, 10]` | Maps every element; nonmultiples of three survive the predicate. | prompt/template: test the predicate on the original value, transform only survivors, and retain duplicates. |
| code-03 / code_reading | `[16,16,8,4]` | `[16, 2, 16, 8, 4]` | The odd input survives the predicate; remaining order and duplicates are preserved. | prompt/template: test the predicate on the original value, transform only survivors, and retain duplicates. |
| code-04 / code_reading | `[12,18,24]` | `[12, 18, 24, 4, 10]` | Maps every element; nonmultiples of three survive the predicate. | prompt/template: test the predicate on the original value, transform only survivors, and retain duplicates. |
| code-07 / code_reading | `{"a":5,"b":3}` | `{"a": 3, "b": 3}` | The destination becomes the source value instead of adding to its current value. | prompt/template: For +=, add to the current value. |
| logic-01 / reasoning | `["A","B","C","D"]` | `{"order": [1, 2, 3, 4]}` | Substitutes numeric positions for task IDs and adds an unrequested object wrapper. | prompt/template: Schedule only ready tasks and preserve their identifiers, never substitute positions. |
| logic-05 / reasoning | `"c1"` | `{"next_stage": "c1"}` | The correct string is wrapped in next_stage instead of being the JSON root. | output formatting/JSON: a requested string, integer or boolean stands alone. |
| logic-06 / reasoning | `13` | `{"value": 13}` | The correct integer is wrapped in value instead of being the JSON root. | output formatting/JSON: a requested string, integer or boolean stands alone. |
| instructions-05 / instruction_following | `false` | `{"boolean": false}` | The correct boolean is wrapped in boolean instead of being the JSON root. | output formatting/JSON: a requested string, integer or boolean stands alone. |

**Ten content errors; three format-only errors** (logic-05, logic-06, instructions-05). The probability and task-ID failures also contain format defects, but they still fail after removing their wrappers. No repair, coercion or wrapper removal is applied to model output or scoring.

## What the evidence supports

Each expected value was independently calculated from the unchanged task. No target/reference defect was found. The real training receipt records all 61 training rows, one epoch and eight steps. Decoder evidence shows greedy single-sequence generation, thinking disabled, and matching system/template hashes. All strict outputs completed at EOS below the 128-token cap. The existing input pack contains independently validated train/validation examples for every failed operation. Therefore truncation, missing training rows, a wrong reference label and recorded decoder-setting drift do not explain these failures.

**The internal model cause is not isolated.** A single screen cannot distinguish prompt adherence from dataset weighting/coverage sufficiency, adapter learning, quantization or base capability. Descriptions such as “omits the addition” characterize the output; they do not reveal internal reasoning. No unsupported causal label is presented as proven.

## Minimum applied change

Append one task-general instruction file to the Nova system prompt in both SFT preparation and evaluation. It makes arithmetic checks, without-replacement counts, perimeter-derived dimensions, original-value predicates, additive updates, task identifiers and scalar JSON roots explicit. It contains no case IDs, benchmark answers, task literals, few-shot benchmark substitutes or answer lookup. All 110 records change only in their system instruction; user messages, targets, splits, hypothetical provenance fixtures and all shared identity/safety/grounding policy bytes are unchanged.

No dataset or optimizer change is justified by this evidence alone. The one-epoch/eight-step recipe, immutable Qwen3-4B base, greedy decoding, token budgets, suite bytes, grader, strict/manual threshold, resource/cost/time limits and fail-closed controls remain unchanged. The plan now rejects both the historical adapter and the completed-screen adapter. No execution authority is granted by this change.

The prior data result remains **6/6** as preserved evidence. All data prompts, targets, scorer and training examples are unchanged. A future measured run must establish that data remains 6/6; free tests cannot promise behavior of new weights.

## Free verification and immutable inputs

The focused correction module contains 21 tests, including 13 individually named preserved-failure regressions. It checks independent references, all 36 preserved outputs, exact 23/36 reproduction, and filesystem/upload absence of the manual packet. The existing 107 controller/network/auth/watchdog/manual-gate regressions remain applicable. These are software/input tests, not a new model score.

Real TRL preprocessing/collation verifies 110/110 revised Nova rows: maximum train 726 tokens and validation 728, within 768; completion labels and EOS remain intact. No model construction, weights, training or inference is used by the probe. Verify recomputes the probe and compares it byte-for-byte with the committed receipt. Existing shared tokenizer/mask proofs remain unchanged.

- Prepared pack: `272d2a6e98fa57fec743618591d054e31298f6d36f01e5eacdc61ece61e1e591`
- Effective Nova system prompt: `258b48ccdc670cfd32312983de0ca4e71b136de1e71aa4ef97fb88d026e554d8`
- Task-check file: `465c94b5b127299ea12c00325a25c7869c86b2a568ce221ca1fa004acf5e2de2`
- Pinned execution plan: `bff6646fae9d67a4f8e51664aeaeef0256e8341ff20bd772eb114ea8f07ecc82`
- Tokenizer receipt: `a68b53753be3af5ebaa94dd2f592f2b8a11081609d199ccf9eeea8a394a064fc`

The new pack is distinct from `fcbe8556c9587d960a531bd0f91d5b7de13fa77fd69d3aa9d62a893dcab91726`. No new candidate artifact exists. A future separately authorized Nova screen must still meet 36/36 before manual 14/14 and 48/48; three required repetitions would require 108/108. Route, provenance, latency, cost and other A35 gates remain separate. No GPU, retry, repetition, other family run, deployment or merge is authorized in this correction cycle.
