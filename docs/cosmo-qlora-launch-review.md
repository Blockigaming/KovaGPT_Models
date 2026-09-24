# Cosmo 42-record pilot: launch review

**Status: source-only. Paid execution is disabled.** The first pilot is Cosmo
only; Orion, Nova, deployment, and production routing are outside this release.
This document is an approval worksheet, not an authorization to spend.

## Exact source selection

| Item | Selected value |
| --- | --- |
| Region / VM | `eastus` / `Standard_NC4as_T4_v3`, four vCPUs, NVIDIA T4 |
| Image | `Canonical:ubuntu-24_04-lts:server:24.04.202609040` |
| Model | `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca` |
| Snapshot | `config/qwen3-0.6b-download-manifest.v1.json`, complete SHA-256 `d1dd63b2ee120b0944a58021a21608a46bb03074f87adbafb067b2aedccaf162` |
| Approved data | `data/kova-identity-shared.v2.jsonl`, SHA-256 `fa6406b2be2e607f8a40565bf9340db226a6fb8def4d5342d30dc38105696051`, 27 train + 15 validation |
| Training | 4-bit NF4 QLoRA, FP16 compute, rank 16 / alpha 32 / dropout 0.05, one epoch, at most seven optimizer steps and 1,800 training seconds |
| Allocation | At most 7,200 seconds measured from the authorized deadline, including setup, download, probe, training, preservation, and shutdown |

The new source preflight is `python3 -m training.cosmo_qlora_launch`;
`python3 -m training.cosmo_qlora_training` prepares the approved split. Neither
command invokes Azure or loads model weights. Both reject paid execution with
the checked-in authorization flags. The older 36-record FP16 LoRA path is a
different experiment and must not be used for this pilot.

## Cost arithmetic and what it proves

| Reservation | USD |
| --- | ---: |
| Maximum two-hour VM compute reservation | 1.1500 |
| Managed disk | 0.3000 |
| Storage capacity and transactions | 0.2000 |
| Network, public IP/network, shutdown delay, failed allocation | 0.4000 |
| Snapshots | 0.0000 |
| Emergency cleanup margin | 1.2500 |
| **Conditional ceiling** | **3.3000** |

At the earlier **public retail** `$0.5260` per hour VM rate, two hours of
compute is `$1.0520`, and the listed reservations total `$3.2020`. This is an
estimate, with `$0.0980` remaining. A subscription-specific compute rate
above `$0.5750` per hour fails the `$1.1500` compute reservation. The listed
ancillary figures are **source assumptions**, not verified subscription
quotes. They must be checked against every service used, including the
watchdog, its storage and transactions, GPU driver setup, disks, data transfer,
and resource deletion latency. A two-hour timer or Azure budget alert cannot
guarantee the eventual invoice is at most `$3.30`.

## Read-only evidence to refresh in the intended subscription

Run from an authenticated Azure Cloud Shell or equivalent with the intended
subscription selected. Preserve the output and subscription ID privately;
never paste tokens, SAS links, or billing exports into a public PR.

```sh
az account show --query '{id:id,name:name,state:state}' -o json
az vm list-usage --location eastus -o json
az vm list-skus --location eastus --size Standard_NC4as_T4_v3 --all -o json
az vm image show --location eastus \
  --urn Canonical:ubuntu-24_04-lts:server:24.04.202609040 -o json
```

The earlier owner capture showed family quota `0/4` and regional quota `0/14`,
an unrestricted SKU, and that image version. It does not prove today's
quota, live allocation capacity, account prices, or that the image boots the
pinned CUDA 12.8 / bitsandbytes stack. `what-if` may validate the proposed
resource graph, but it does not reserve a GPU. Capacity and exact hardware
identity can be confirmed only after an approved allocation.

Get the **current billing-account price sheet** for the account's agreement
type (MCA/MPA or EA), locate the exact East US compute meter, and price every
ancillary meter above. The public Retail Prices API is useful for comparison
only. Microsoft documents separate [billing-profile price sheets](https://learn.microsoft.com/en-us/rest/api/cost-management/price-sheet/download-by-billing-profile?view=rest-cost-management-2025-03-01)
for MCA/MPA and [billing-account price sheets](https://learn.microsoft.com/en-us/rest/api/cost-management/price-sheet/download-by-billing-account?view=rest-cost-management-2025-03-01)
for EA. The billing agreement and IDs are not present in this workspace, so
there is no correct account-specific API command to fill in here yet.

## Paid sequence awaiting final approval

1. Identify the subscription, exclusive empty pilot group, separate watchdog
   group, exact identity/principal, SSH key, image, preserved output storage,
   and complete billing meters. Review the account-specific all-in bound.
2. Provision the independent authority, append-only grant ledger, and watchdog
   in a separate approved operation. Test its ability to deallocate and delete
   the **exclusive** pilot group. Test that the watchdog itself and its storage
   stop accruing charges, with evidence preserved outside both groups. If its
   cleanup fails, the controller must stop and the operator must deallocate and
   remove only the named pilot resources using a separate trusted control path.
3. Have the authority issue a short-lived signed, subscription-bound quote.
   `python3 -m training.cosmo_qlora_launch --quote /absolute/quote.json
   --subscription-id <approved-subscription-uuid>` verifies its source,
   account price, quota, SKU/image, watchdog and budget bindings. The trust
   key is unset today, so this command fails closed.
4. **Only after a separate approval of the actual run:** create one VM without
   public IP in the exclusive group, set an independent deallocation/deletion
   deadline no later than two hours, and record its resource ID. Verify the
   real GPU is NVIDIA T4 with capability 7.5, the pinned image and runtime are
   compatible, and the independent watchdog remains healthy. Download and
   hash-check only the pinned nine-file snapshot. Run the 42-record QLoRA job
   once. No retry, automatic second family, or production deployment.
5. Preserve immutable adapter and failure evidence outside the pilot group,
   then deallocate and delete the pilot group. Query the control plane for zero
   remaining billable resources, shut down/delete watchdog resources after
   preserving its terminal ledger, and reconcile actual charges when Azure
   posts them. Guest-process exit alone does not stop disk or network charges.

**Stop immediately** on a missing signature, expired or mismatched quote,
changed source/model/data, insufficient quota, SKU/image mismatch, unverified
account meter, all-in calculation above `$3.30`, failed watchdog health or
permissions, unexpected public networking, non-T4 hardware, incompatible
runtime, snapshot hash failure, elapsed deadline, or any cleanup failure.
Do not proceed to training from a failed or incomplete step.

## Open launch blockers

- This session is not authenticated to the Azure subscription. Account-specific
  pricing and current quota/SKU/image results have not been refreshed here.
- The independent authority endpoint/signing key, atomic remote grants,
  watchdog health and self-cleanup proof, and signed subscription quote do not
  exist. The older `$2` authority contract is not a `$3.30` controller.
- The `training.cosmo_qlora_training` paid entrypoint deliberately fails before
  any model load or Azure action. A separate reviewed source release and
  end-to-end lifecycle rehearsal must precede an executable purchase request.
- Live GPU allocation, pinned runtime compatibility, full 42-row tokenization
  with the actual tokenizer, and the final account bill are unverified.

No resource creation, weights, training, deployment, or merge is authorized
by this review package.
