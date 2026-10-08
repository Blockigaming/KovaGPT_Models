# Nova transfer screen and proposed exposure probe

Phase A remains **30/40 (75%)** and A35 remains **OPEN**.

Attempt `a1d10484ddab453cac5f848306e7622c` executed head
`c7d4ee23684110caa30b52b4469af3e7ddecb8b2` exactly once. Training completed
72 training rows, 60 distinct validation rows, one epoch and nine optimizer
steps in 148.02 seconds. Candidate weights:
`95fd3383589ef1bfe942345a9df09e3a3afaa8cf8d24ea886a273eea2a5089ee`.

The measured result is **23/36**: math 6/10, code reading 2/8, reasoning 3/6,
data 6/6 and instruction following 6/6. All outputs reached EOS in 2–21 tokens.
Manual evaluation was not run and the manual packet was independently absent.
Runtime identity/safety/grounding stop conditions were zero; this does not
establish unperformed manual or authenticated serving-route checks.

Compared with the preceding 21/36 screen, `data-05` and `instructions-05`
recovered, thirteen failures persisted, and no case newly regressed. Compared
with the historical 23/36 screen, `instructions-05` recovered and `code-06`
regressed. Equal totals do not mean equal behavior.

| Cases | Demonstrated failure | Classification |
|---|---|---|
| math-03, math-05 | Incorrect add-then-multiply result | Content/generalization |
| math-07 | Incorrect without-replacement denominator; extra wrapper | Content/generalization |
| math-10 | Incorrect rectangle area | Content/generalization |
| code-01 through code-04 | Predicate filtering or transformed value is incorrect | Content/generalization |
| code-06 | Mutation of a copied list incorrectly changes the original | Content/generalization |
| code-07 | Source value replaces the destination instead of incrementing it | Content/generalization |
| logic-01 | Task identifiers are replaced by numeric positions | Content/generalization |
| logic-05 | Correct string is wrapped in an object | Format-only |
| logic-06 | Correct integer is wrapped in an object | Format-only |

The offline auditor checks all 36 cases against both baselines, unchanged
grader targets, report/input/adapter hashes, token completion evidence and
the exact one-epoch training receipt. Raw outputs and authenticated archives
remain private. The private report SHA-256 is
`f9b1bf4f84cfcc128d5a5e7fae3db85c9169eda98b26bb8b944db2c6b304fa54`.

No incorrect reference, truncation, omitted prompt instruction, decoder drift
or masking defect explains these failures. Correct training examples already
cover every remaining class. The extra examples coincided with two format
recoveries; the screen does not isolate dataset weighting, optimization
exposure or base-model limitations. Adding benchmark-specific answers or
another speculative data supplement is not justified.

The preserved training log reports aggregate loss 3.525 and mean token accuracy
0.5942 during the single epoch. These are in-training metrics, not held-out
quality scores or proof of a unique cause. They support testing additional
exposure while holding the examples and all grading standards fixed.

The proposed next candidate changes **only epoch count, from one to two**,
with the resulting expected optimizer steps changing from nine to eighteen.
This tests insufficient exposure as a hypothesis; it is not a demonstrated
model correction. The data, prompt, split, seed, learning rate, adapter shape,
decoder and grading standards remain fixed. The one-epoch timing suggests
approximately 296 seconds for two epochs, an estimate only; the existing
600-second training cutoff remains mandatory. An incomplete second epoch
must block evaluation, as must an old nine-step receipt.

All four measured adapters are rejected. The next recipe remains unexecuted
and requires a fresh exact-head owner grant. It retains one Nova training run,
one strict screen, one East US T4 allocation, no retry or repetition, the $5
all-in ceiling, and the 70/75/90-minute limits. Credentials, DDoS/network
attestation, pre-allocation watchdog verification, scoped Reader/writer
access, authenticated evidence, strict-before-manual gating, and all eight
independent cleanup checks are unchanged. No resource or model call is
required to prepare or test this proposal.

The completed attempt's evidence and cleanup passed. Fresh and preceding
21/36-attempt billing remain pending. The older failed `63e356…` attempt now
has $0.0552599563376304 posted pre-tax; historical `9ce963…` has
$0.752327121539239 posted pre-tax. Final all-in reconciliation remains pending;
empty rows never establish zero spend.

Passing source tests does not change the measured score. A35 still requires
36/36 on each required repetition, gated human manual review of 14 cases and
48 criteria, and the remaining authenticated-route, provenance, latency and
cost evidence. No failure may be averaged away.
