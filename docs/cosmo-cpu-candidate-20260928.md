# Cosmo CPU trained candidate — verified 2026-09-28

The public [training workflow run 36367369377](https://github.com/Blockigaming/KovaGPT_Models/actions/runs/36367369377)
completed all seven optimizer steps on a free GitHub Actions CPU runner. The
source is commit `19ed0a9ce007f6d90ccb3dbec39daef852dc601c` in the
single integrated Models PR #62. It used Python 3.12, a hash-pinned CPU
PyTorch/SFT stack, the immutable `Qwen/Qwen3-0.6B` base revision
`c1899de289a04d12100db370d81485cdf75e47ca`, completion-only FP32 LoRA,
and the approved 42-record dataset digest
`fa6406b2be2e607f8a40565bf9340db226a6fb8def4d5342d30dc38105696051`.
There were 27 training records and 15 reserved validation records.

The downloaded artifact archive `cosmo-cpu-lora-candidate`, ID
`10948100892`, has SHA-256
`a2bd0e7e0bbb967a44bd4ba37ae5fda27cf6bd861ccdcbd13073c9b1bf7b2695`
and expires on 2026-10-05. Its `receipt.json` reports status `complete`,
7 optimizer steps, and 468.62 training seconds. Independent reads of the
ZIP confirmed its digest and the receipt's hashes:

| Adapter file | SHA-256 |
| --- | --- |
| `adapter/adapter_config.json` | `49b16d39fac90941e7ec4ca890a96ef5d4bfa3874a0f6533a19014b31ef1b580` |
| `adapter/adapter_model.safetensors` | `d7f07111f262cfe7c78cd1d398adf4a9f85ec9e0ec8e96fa0bff2dd9a690d27d` |

The compact archive containing the adapter and receipt has SHA-256
`0ccc774899d87eb6261d0f3e82f6bd2e74577205624e4e54bfa2d4afb5f08f54`
and is saved as `KovaGPT-Cosmo-trained-adapter-2026-09-28.zip` in the KovaGPT
artifact collection. The original workflow ZIP also contains checkpoints;
the compact copy contains only the adapter files, README and receipt. The
original model weights are not redistributed in either adapter archive.

## Held-out loss measurement

[Evaluation run 36368599939](https://github.com/Blockigaming/KovaGPT_Models/actions/runs/36368599939)
loaded the exact pinned base and the hash-matched saved adapter at source
`1a3e2e82fb59e167cf79940a3053d6602210f35b`. It scored all 15 reserved
validation records using the same completion labels for each variant, 318
completion tokens in total. Token-weighted mean loss was **2.76193386** for
the base and **1.48700863** for the adapter, a difference of **−1.27492524**.
The report archive digest is
`db375e9fde5177cc714ad044d4b6693e2e859a4dbff2c7c855e8a1cd524eb58f`;
the extracted `KovaGPT-Cosmo-heldout-evaluation-2026-09-28.json` digest is
`8a53082cccd3bccf25496202aa05016a290c574eee3893bfd8adebc34745d3b4`
and the report is preserved in the KovaGPT artifact collection. It includes
all 15 per-case losses and the adapter file hashes. This is a small, synthetic
held-out loss comparison, not a behavioral benchmark or a human quality pass.

**Scope:** This is one actual trained **experimental CPU candidate**. The
selected Azure T4 four-bit QLoRA run, Orion and Nova training, full behavioral
evaluation, human review, independent spending/cleanup authority,
deployment and live routing remain outstanding. Phase A remains **30/40**;
Phase B remains **NOT READY**. It is not evidence that the adapter improves
quality, and it is not authorized as a production model.
