# Orion CPU trained candidate — verified 2026-09-28

[Training run 36368479166](https://github.com/Blockigaming/KovaGPT_Models/actions/runs/36368479166)
completed all seven FP32 LoRA optimizer steps on a public free CPU runner.
Source commit: `50f8b9a00851eda52e5112f8944882171d0b7dfd`. The immutable
`Qwen/Qwen3-1.7B` base revision was
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; the approved synthetic
42-record dataset digest was
`fa6406b2be2e607f8a40565bf9340db226a6fb8def4d5342d30dc38105696051`
with 27 training and 15 reserved validation records. The receipt reports
901.42 training seconds and status `complete`.

GitHub artifact `orion-cpu-lora-candidate` (ID `10948293907`) has ZIP SHA-256
`0790434e87376ccae207726f901d59dda1b2b97f3593787c0f143bf07551470f`
and expires on 2026-10-05. The downloaded ZIP digest and both receipt file
hashes were independently matched:

| Adapter file | SHA-256 |
| --- | --- |
| `adapter/adapter_config.json` | `ee909c25085c6a81eb18ac9409113164e1bf8bb76171838c45898d3376949825` |
| `adapter/adapter_model.safetensors` | `f3d70456558a325a2e344012376a8a2bdcf749f7058844a5c39ba6c999a01c5b` |

The separate compact adapter archive
`KovaGPT-Orion-trained-adapter-2026-09-28.zip` contains the adapter, README
and receipt; its SHA-256 is
`66cc456e49d6fa137bf406cced50f8c74fe0010d2c755866f7bfca68edb96b17`.
The compact file is preserved in the KovaGPT artifact collection beyond the
seven-day workflow retention. Base weights are not included.

## Held-out loss measurement

[Evaluation run 36369706634](https://github.com/Blockigaming/KovaGPT_Models/actions/runs/36369706634)
loaded the immutable base and saved adapter, then scored all 15 reserved
validation records with identical completion labels: 318 completion tokens
per variant. The base's token-weighted mean loss was **6.76487525**; the
adapter's was **3.01616986**, a difference of **−3.74870538**. The report
archive SHA-256 is
`b788fd83f3f5838c2eac9bfada0fd83a1de44eec5cff22583c2f518b0aea8809`;
the extracted `KovaGPT-Orion-heldout-evaluation-2026-09-28.json` SHA-256 is
`d37cb4d2464db85db6a66b36de558e6d7cdb372e05bca72cd8f9dbab91f1e3d8`.
The report with all 15 per-case values is preserved separately from the
seven-day GitHub artifact. This is a small synthetic loss comparison, not a
human judgment of output quality.

This is a trained **experimental CPU** candidate, separate from the selected
Azure T4 four-bit QLoRA pilot. It does not authorize production routing,
change Phase A's **30/40** checklist count, or make Phase B ready. Human
quality review remains outstanding.
