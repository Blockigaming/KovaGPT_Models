# Nova 21/36 outcome and complete failure-class input hypothesis

Phase A remains **30/40 verified (75%)**; **A35 OPEN**. The completed, consumed
attempt `bb328ecf604e4a03b99a1eef98c7575e` at
`cfb0ddb3e9db6ddb7812233a1529589024318c3a` trained successfully and measured
**21/36**: math 6/10, code 2/8, reasoning 3/6, data 5/6 and instruction 5/6.
Manual evaluation was NOT RUN and its file/upload remained absent. Runtime
identity/safety/grounding stops were zero; unrun manual checks earn no credit.
Authenticated evidence, watchdog, Reader/writer removal and all eight independent
cleanup checks passed. Fresh billing returned no rows and remains **PENDING**.
Prior measured spend is $0.752327 posted pre-tax; final billing remains pending.

The candidate weights hash is
`9e9c991332e979d3b3b73452e643d0272f859965fc054329f32308491a0a7e0b`.
The exact report remains in the private preserved evidence archive;
`python -m evaluation.a35_latest_screen_audit --preserved-report <private-screen.json>`
reproduces every result with the
unchanged grader and lists all **15 failures: 11 content and four format-only**.
All original 13 failures persist. `code-06` (copy versus alias) and `data-05`
(Python set syntax instead of a JSON array) newly fail. No case newly passes.
The prior 23/36 and data 6/6 remain historical measurements, not current scores.

The live composed prompt, full one-epoch/eight-step training and fresh adapter
were verified. All 36 outputs ended at EOS within 2–21 tokens. The actual trainer
verified completion masks before training. All 504 saved adapter tensors are
finite; all 252 LoRA B tensors contain nonzero updates. No deterministic prompt
omission, truncation, stale-weight, decoder or masking defect was found. The
observed mistakes do not identify a unique model-internal cause. In particular,
these facts do not justify claiming that more epochs or another generic prompt
would fix them.

## Smallest supported correction set

The prior prompt-only intervention left all 13 failures intact. A deterministic
inventory of the measured pack has 1,331 completion tokens across 61 training
rows; only 157 are valid JSON target tokens. Most failed skill groups have one
training example. This is evidence of sparse supervision and limited variation,
not proof that token balance or data caused each error. No incorrect or
contradictory label was found. The copy example specifically teaches nested
sharing but lacks a flat outer-copy/alias contrast. The new copy and set-syntax
regressions cannot be attributed uniquely because both prompt and adapter
changed together; existing valid JSON-array targets rule out a blanket lack of
array training.

The Nova-only addition contains **11 train + 11 held-out validation scenarios**,
one pair for each independently observed failure class:

| Failed cases | Class | New independent training structure | Distinct validation structure |
| --- | --- | --- | --- |
| math-03, math-05 | arithmetic | three-digit stock subtraction, product and adjustment | aggregation, division and per-kit removal |
| math-07 | probability | ordered three-color draws without replacement | unordered group with an exact marked-card count |
| math-10 | geometry | perimeter and side difference | joined congruent rectangles |
| code-01–04 | predicate tracing | record comprehension with compound predicate and string mapping | loop with exclusion before mapping |
| code-06 | copy versus alias | outer slice replacement contrasted with alias write | shared nested content versus independent dictionary field |
| code-07 | augmented update | sequential cross-key increments | assignment versus indexed increments |
| logic-01 | identifiers and prerequisites | five-node dependency schedule | unfinished ready-set extraction |
| logic-05 | JSON string root | extract a field value, not its object | sorted-word selection |
| logic-06 | JSON integer root | nested cardinality | circular-clock update |
| instructions-05 | JSON boolean root | universal predicate, false | compound state guard, true |
| data-05 | JSON array serialization | counted records to sorted names | active normalized records to unique sorted names |

Each label is independently checked. Probability tests also enumerate distinct
outcomes. Reference/review metadata stays outside model messages. The combined
132-row corpus passes unchanged lexical and cross-split leakage checks, Python
shape checks and explicit graph-isomorphism checks. A small leakage-check defect
is corrected: adding a terminal `result=variable` bookkeeping assignment can no
longer disguise an otherwise identical benchmark program. No benchmark answers,
case IDs or answer lookup enter model inputs. These conservative checks do not
prove semantic generalization.

All 110 measured rows, targets and splits, all policy safeguards, the effective
prompt, optimizer/LoRA hyperparameters, decoder, suite and graders are preserved.
The next pack is **72 train / 60 validation**. One complete epoch now takes
**nine optimizer steps**, derived from 72 rows at batch size one and accumulation
eight, instead of eight steps over 61 rows. There is no epoch or learning-rate
increase. Maximum training time remains 600 seconds. The candidate and quality
remain **NOT MEASURED**; this complete coverage intervention is a hypothesis,
not a claim that all 15 errors are fixed.

All three measured adapter hashes are rejected. Missing or altered rejection
bindings fail closed. Old source/pack grants cannot authorize this changed
experiment. The private local replay preserves every failure and reproduces 21/36. Public
CI uses synthetic fixtures to validate the auditor and block manual generation/
file/upload; it does not transmit the private report. Tests validate every new label and
reject input/pin drift. These are software/data checks, not model-quality scores.

## Next boundary

This correction cycle performs no training, model generation, cloud allocation,
retry, other-family run, deployment or merge. A future screen requires a new
owner authorization bound to the exact green head and current plan/pack hashes.
Its existing scope remains one Nova Qwen3-4B, one East US T4, one allocation,
one training run and one strict sweep, no retry/repetition, $5 all-in and
70/75/90-minute limits. Fresh ARM/Storage, price/quota, watchdog, Reader,
network and absence checks must pass live before allocation; old live checks
are historical evidence, not fresh launch admission.

Strict must reach 36/36 with every category perfect and no applicable
identity/safety/grounding failure before any manual packet is created. Only then
can manual 14/14 and all 48 criteria be reviewed. Every outcome requires
preserved authenticated evidence, Reader/writer removal and eight independent
absence checks. Three perfect repetitions and separate provenance, authenticated
route, latency/cost and other A35 requirements remain open. No pass is promised
and no failure is averaged away.
