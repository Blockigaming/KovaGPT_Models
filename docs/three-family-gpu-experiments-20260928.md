# Three trained GPU experiments — 2026-09-28

The approved, private East US `Standard_NC4as_T4_v3` VM ran seven optimizer
steps for **each** of Cosmo, Orion and Nova against the exact approved 27-record
training split. The VM reported a Tesla T4 with CUDA capability 7.5. Each
adapter was uploaded using the VM's system-assigned identity, which held
`Storage Blob Data Contributor` only on the `cosmo-adapters` container in
`kova42c1a27`. An independent GET through that identity reproduced the
complete ZIP bytes and SHA-256 digest for all three artifacts.

| Family | Pinned Qwen3 revision | GPU training commit | Steps | Training seconds | ZIP bytes | ZIP SHA-256 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Cosmo 0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | `240070e7db0cead3ce912780807f3091466bdf97` | 7/7 | 17.71 | 20,245,111 | `452aedd9b19a580799865e7c960e1037931a98cc8b225b4d2283b18ce958f815` |
| Orion 1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | `240070e7db0cead3ce912780807f3091466bdf97` | 7/7 | 25.86 | 34,925,359 | `0d76a2551e30910d28a20707bf6f92d8d5e37db25b98e2a92cd27894b2f05482` |
| Nova 4B | `1cfa9a7208912126459214e8b04321603b3df60c` | `288b7dd08c076bdbeaf7f2114049ae9fa4c3f439` | 7/7 | 82.76 | 33,104,769 | `e87060e984c10e636a59d9e55dfaee1cb52ed47930dfaf2bf5b791435cbdf21b` |

The blobs are under
`cosmo-adapters/gpu-experimental/2026-09-28/<family>/adapter.zip`.
Each adapter includes its configuration, safetensors and a training receipt.
The source and dataset digest
`fa6406b2be2e607f8a40565bf9340db226a6fb8def4d5342d30dc38105696051`
were verified before training.

## Held-out completion loss

The separate GPU evaluator at source commit
`dac61c72a14f4a204bbcff78c9d83f08d26fb59c` checked each adapter hash,
loaded the pinned local base in NF4, and compared base and adapted loss on
the **15 reserved validation records / 318 completion tokens**. It wrote
and independently read back
`cosmo-adapters/gpu-experimental/2026-09-28/<family>/heldout-evaluation.json`
for each family.

| Family | Base loss | Adapted loss | Change |
| --- | ---: | ---: | ---: |
| Cosmo | 3.06558949 | 1.57577603 | −1.48981346 |
| Orion | 5.94963720 | 2.84376730 | −3.10586990 |
| Nova | 5.71162239 | 3.58559673 | −2.12602566 |

Lower held-out loss on this small, fixed split is directional evidence.
It is not an independent human quality verdict, a safety sign-off, or a
production performance test. The Cosmo and Orion CPU experiments previously
completed on GitHub Actions remain separate candidates with different
precision and loss numbers.

## Execution and boundary

The first Nova CPU bounded run ended at its time limit with zero completed
steps and no adapter. The first GPU bootstrap stopped when the NVIDIA
extension's unrelated Ubuntu 22.04 CUDA apt feed lacked a signing key.
`python3.12-venv` was installed from the signed Ubuntu package index;
the corrected bootstrap skips refreshing that feed. The next GPU attempt
reached the optimizer but hit PyTorch's unsupported BF16 gradient unscale
under `fp16=True`. The pinned trainer was corrected to optimize the adapter
without AMP while keeping the model's four-bit NF4 float16 compute. The
subsequent Nova, Cosmo and Orion jobs completed.

The pilot VM had no inbound public IP. Its independent watchdog had only
two Contributor grants: the pilot group and its control group, with a
2026-09-28 12:44:29 UTC cleanup deadline. Azure deallocated the VM after
training and evaluation. The pilot group was deleted and
`az group exists` returned `false`; the watchdog control group was then
deleted and independently returned `false`. The separate evidence storage
account retains the experimental blobs. Azure's final invoice is pending.

These are **3/3 experimental T4-trained adapters**, **0/3 selected
signed-controller pilots**, and **zero live production routes**. The Phase A
checkpoint remains **30/40 verified, Phase B not ready**. A final invoice,
full signed cost admission, production loader, quality review and app route
integration remain outstanding; the owner's $12 total authorization was a
spending ceiling, not a guaranteed Azure charge.
