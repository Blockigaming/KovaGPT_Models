# Kova identity pilot: configuration plus fine-tuning preparation

This is a source-only starter for the owner's requested Kova-branded assistant.
It is not a trained model, a production prompt rollout or a GPU deployment.
Phase A stays 30/40. Phase B remains NOT READY.

## Intended visible behavior

For "Who are you?", the ordinary target is:

> I'm Kova, the AI assistant in KovaGPT.

For ordinary tasks, answer the question without an identity preamble, supplier
branding or sales copy. Kova is the product/conversational identity. This starter
does not suppress truthful answers to explicit technical-provenance questions,
remove required upstream notices, rewrite licensed origin records or claim that
Kova trained a foundation model from scratch. The one known-origin validation
example is explicitly hypothetical; it is not a live runtime observation.

## New source files

- `prompts/kova-identity.v2.txt`: proposed shared identity instructions.
- `data/kova-identity-pilot.v1.jsonl`: 24 synthetic training examples and 12
  separately held-out validation examples, including basic identity, useful tasks,
  output format, uncertainty, privacy and truthful activity.
- `config/kova-cosmo-pilot.v1.json`: preparation-only LoRA plan for the already
  selected Cosmo source, bound to its immutable source revision. The proposed
  compute label is the owner's Azure NC4as T4 v3 target, not evidence of GPU fit.
- `training/identity_pilot.py`: standard-library-only validation and compilation.
- `training/test_identity_pilot.py`: source/data regression checks.
- `data/kova-identity-pilot-review.v1.json`: dataset-bound owner review ledger.
- `training/identity_pilot_review.py`: fail-closed review validation that cannot
  authorize model download, training or deployment.

No private chats, customer records, credentials, external model outputs or
upstream weight files are included. These examples were synthetically prepared
for Kova; human review is not yet complete. The corpus is a small starting set,
not a claim of sufficient training diversity, model quality or independence of
all paraphrases. Validation prompts are disjoint by normalized exact text, not
by a proven semantic train/test contamination analysis.

The review ledger contains all 36 source IDs in dataset order and is bound into
the pilot plan by SHA-256. Every committed verdict remains `pending`; no reviewer
or completion time is invented. The validator supports explicit in-progress,
changes-requested and approved states, but even a fully approved ledger cannot
grant training authorization or Phase B readiness. Complete it only through a
real owner review of every prompt/answer pair.

## What is actually verified

The source checker verifies fixed paths, fingerprints, immutable Cosmo base
identity, exact flags/types/shapes, record roles, unique identifiers, normalized
prompt disjointness and train/validation counts. The compiler prepends the shared
identity prompt, keeps validation out of training, and writes deterministic JSONL
files with SHA-256 receipts. Existing output folders are refused, not overwritten.
It has no network client, model loader, cloud API, training framework or execute
option. Registered CI checks source integrity only; it does not evaluate trained
model answers. Hypothetical runtime metadata is allowed only in its designated
validation example and must never be passed off as real serving evidence.

Source validation only, from the repository root:

```sh
python3 -m training.identity_pilot
python3 -m unittest training.test_identity_pilot -v
```

Prepare data in a NEW folder, without downloading weights or starting training:

```sh
python3 -m training.identity_pilot --output artifacts/kova-identity-pilot
```

The resulting `train.jsonl`/`validation.jsonl` contain conversational messages.
Tokenization, model-specific chat-template compatibility and the actual training
framework still require a separate verified recipe. Do not claim a ready-to-run
trainer from the presence of these files.

## Existing source is not silently repurposed

The old `config/candidate.v1.json`, `config/training-stack.v1.json`,
`config/identity.v1.json` and original four-record starter are untouched. The old
`training:command` recipe is not this pilot and must not be used to launch it.
The v2 prompt is not connected to live app/worker prompts by this draft. Current
Chat/Work route policies, all exact compute budgets, runtime provider disclosure,
weekly usage and release guards are unchanged. No PR is merged into main here.

## Before a paid pilot

Review the examples; obtain a verified downloadable artifact inventory and its
notices; validate a pinned tokenizer/template and compatible training environment;
select/review measured-memory-safe hyperparameters; confirm region/quota and the
actual Azure rate; approve a separate budget and shutdown procedure. The plan
keeps budget, hyperparameters and trained-adapter digest null. Its download,
training and deployment permissions remain false. Neither this source nor a
successful compiler result can grant spending or deployment authority.

Evaluate real base and trained outputs on the held-out examples and a broader
quality suite before releasing the adapter. Record the exact base and adapter
hashes and preserve derivative lineage. A model introducing itself as Kova alone
is not proof that a fine-tune ran successfully or that its general quality improved.

## Technical references

- Selected Cosmo source: https://huggingface.co/Qwen/Qwen3-0.6B/commit/c1899de289a04d12100db370d81485cdf75e47ca
- Selected source license: https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE
- LoRA documentation: https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
- Redistribution/notice conditions: https://www.apache.org/licenses/LICENSE-2.0.html

The license permits modifications under its terms and imposes notice conditions
on redistribution. It does not require a supplier name in every ordinary answer.
The fine-tuning method changes trainable parameters; this starter has not done so.

## Dependency inspection — September 19, 2026

Inspected the published wheels for TRL 1.13.0, Transformers 5.17.0,
PEFT 0.21.0, Accelerate 1.15.0 and Datasets 5.0.1 without installing
the training stack or downloading weights. TRL's SFTTrainer source accepts
conversational prompt/completion records and forwards model_init_kwargs.revision
to its automatic processing-class loader. This is source inspection, not proof
of working token masks or GPU execution.

The guarded execute path now checks installed distribution versions against
all five recipe pins after authorization checks and before heavy imports.
Missing or mismatched packages reject execution. This metadata check cannot
prove wheel integrity, transitive compatibility or CUDA support.

The Python 3.12/Linux training dependency graph is now hash-locked separately
from a CPU-only verification graph. Using the two hash-verified tokenizer assets
from the pinned Cosmo revision, `training.probe_cosmo_loss_masks` exercised the
real TRL 1.13.0 preprocessing method and collator on all 36 examples. It verified
24 training and 12 validation records, prompt exclusion, complete assistant
labels including EOS, attention masks and 258 padding positions. The longest
example was 426 of 1,024 tokens. Four dependency-light regression tests reject
prompt leakage, lost completion labels and changed token sequences.

This verification loaded no model weights, constructed no model, ran no forward
or backward pass and used CPU-only PyTorch. It therefore does not prove that the
LoRA targets match a loaded checkpoint, that CUDA/PyTorch is compatible with an
Azure T4, that FP16 training is stable, or that save/reload and evaluation work.
Those checks, human data review and the base/configured-base/trained comparison
remain open. The separate CPU probe lock must not be used as a GPU training lock.

## Synthetic Qwen3/PEFT compatibility check

`training.validate_kova_cosmo_peft_cpu` uses the same CPU-only lock to construct
a two-layer random Qwen3 model from configuration. It verifies that every one of
the seven configured target names matches both layers, that PEFT creates only
LoRA trainable parameters, and that a synthetic completion-loss backward pass
produces finite gradients without touching frozen base parameters. One synthetic
optimizer step makes the temporary adapter nontrivial; safe serialization,
local-only reload, tensor equality and deterministic logits are then checked.
The temporary adapter is deleted when the check exits.

This is an API and serialization fixture, not Kova fine-tuning. It uses no
selected checkpoint weights or training examples and proves nothing about the
real checkpoint's memory use, quality, FP16 behavior, CUDA, T4 compatibility or
eventual adapter. Those claims still require the separately approved GPU pilot.

## Three-way evaluation contract

`config/kova-cosmo-evaluation-plan.v1.json` binds all 12 held-out validation
records to three conditions: untouched base, base plus the reviewed Kova system
prompt, and the eventual trained adapter plus that prompt. The provider-free
validator requires the same base revision, software lock, hardware, precision
and quantization across conditions; only the trained condition may carry the
single adapter digest. Every condition binds the same adapter-receipt digest.
Measured evidence is rejected unless the local receipt verifies against the
declared source commit and both receipt and adapter digests match the bundle.
It rejects missing, failed, duplicated, relabeled or hash-mismatched attempts
and incomplete scores.

The contract records identity, instruction adherence, factuality, formatting,
general quality and safety/truthfulness separately. Their exact pass/fail
criteria are defined by the evaluation plan's hash-locked rubric, and the
score-only human overlay must declare that same rubric digest. A complete measured bundle
still cannot authorize release, establish the reviewer's identity, mark Phase B
ready or close a checklist item. No result bundle exists yet, so no real model
quality comparison is claimed by this source plan.

Reference: https://huggingface.co/docs/trl/sft_trainer
