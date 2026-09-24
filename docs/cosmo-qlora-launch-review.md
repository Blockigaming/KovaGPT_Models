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
| Allocation | Signed window 5,400–7,200 seconds from the authority's observation to its cost deadline, including setup, download, probe, training, preservation, shutdown, and deletion; a full two hours is allowed only if its entire cost fits the owner ceiling |

The new source preflight is `python3 -m training.cosmo_qlora_launch`;
`python3 -m training.cosmo_qlora_training` prepares the approved split. Neither
command invokes Azure or loads model weights. Both reject paid execution with
the checked-in authorization flags. The older 36-record FP16 LoRA path is a
different experiment and must not be used for this pilot.

The paid trainer's proposed command is shown for exact CLI review; it is
**disabled on this head** and its arguments cannot be filled truthfully yet:

```sh
python3 -m training.cosmo_qlora_training --execute \
  --snapshot /absolute/protected/qwen3-0.6b \
  --output /absolute/separate/new/cosmo-adapter \
  --quote /absolute/signed/account-quote.json \
  --runtime-evidence /absolute/signed/vm-preflight.json \
  --subscription-id '<verified-subscription-uuid>'
```

The authority must sign the runtime preflight for the exact quote and source,
including a fresh Azure control-plane read of the executing VM's sole NIC,
private subnet, attached Standard NAT and IP, outbound 80/443 rules, and
inbound denial. The independent authority must also read the deployed watchdog
schedule and sign the same cleanup trigger as the quote and grant. The grant
request binds that signed network proof by digest,
then atomically commits a single-use training grant bound to the quote's
cleanup trigger and cost deadline, the VM resource ID,
immutable VM ID, managed identity token, lifecycle, ledger sequence, and
signed cost deadline of at most two hours. The trainer checks installed package versions and a tiny
NF4 CUDA operation before it opens model weights, and accepts a protected
snapshot with output in a separate tree. The independent server protocol,
signer, cleanup watchdog and verified account prices are not provisioned.
It refuses a grant and stops model work once the signed watchdog cleanup
trigger arrives, leaving the remaining priced window for resource deletion.

The intended paid command order after an owner release is shown below. These
commands are **review text only**: the account-specific parameters, trusted
authority, rate proof, watchdog self-cleanup, SSH key, and approved resource
group names do not exist in an executable launch configuration. Never copy
the deployment lines into an interactive shell on this source head.

```sh
az account show --query '{id:id,state:state}' -o json
az provider show --namespace Microsoft.Network --query registrationState -o tsv
az provider show --namespace Microsoft.Logic --query registrationState -o tsv
az vm list-usage --location eastus -o json
az vm list-skus --location eastus --size Standard_NC4as_T4_v3 --all -o json
az vm image show --location eastus --urn Canonical:ubuntu-24_04-lts:server:24.04.202609040 -o json
# Signed, independent account quote + rate worksheet must pass before the lines below.
az group create --name "$PILOT_RESOURCE_GROUP" --location eastus
az group create --name "$WATCHDOG_RESOURCE_GROUP" --location eastus
az deployment group what-if --resource-group "$PILOT_RESOURCE_GROUP" \
  --template-file infra/three-family-pilot-vm.bicep --parameters \
  provisionPilot=true suffix="$PILOT_SUFFIX" adminUsername="$PILOT_ADMIN_USERNAME" \
  ubuntuImageVersion=24.04.202609040 sshPublicKey="$PILOT_SSH_PUBLIC_KEY"
az deployment group what-if --resource-group "$WATCHDOG_RESOURCE_GROUP" \
  --template-file infra/three-family-watchdog.bicep --parameters \
  provisionWatchdog=true suffix="$WATCHDOG_SUFFIX" subscriptionId="$SUBSCRIPTION_ID" \
  pilotResourceGroupName="$PILOT_RESOURCE_GROUP" pilotSuffix="$PILOT_SUFFIX" \
  controllerPrincipalObjectId="$CONTROLLER_PRINCIPAL_OBJECT_ID" deadlineUtc="$PILOT_CLEANUP_TRIGGER_UTC"
# Future independent controller must first prove its ledger, grant, deletion and cost controls.
# The source VM template includes a separate Standard outbound NAT/IP, a private
# subnet and an outbound 80/443 NSG; the VM NIC has no public IP or inbound path.
az deployment group create --resource-group "$WATCHDOG_RESOURCE_GROUP" \
  --template-file infra/three-family-watchdog.bicep --parameters \
  provisionWatchdog=true suffix="$WATCHDOG_SUFFIX" subscriptionId="$SUBSCRIPTION_ID" \
  pilotResourceGroupName="$PILOT_RESOURCE_GROUP" pilotSuffix="$PILOT_SUFFIX" \
  controllerPrincipalObjectId="$CONTROLLER_PRINCIPAL_OBJECT_ID" deadlineUtc="$PILOT_CLEANUP_TRIGGER_UTC"
az deployment group create --resource-group "$PILOT_RESOURCE_GROUP" \
  --template-file infra/three-family-pilot-vm.bicep --parameters \
  provisionPilot=true suffix="$PILOT_SUFFIX" adminUsername="$PILOT_ADMIN_USERNAME" \
  ubuntuImageVersion=24.04.202609040 sshPublicKey="$PILOT_SSH_PUBLIC_KEY"
# The approved controller then verifies IMDS, T4, immutable snapshot, grant, and training.
```

On any stop condition, an independent control identity must deallocate and
remove only the exclusive pilot group, then verify deletion. Preserve the
terminal ledger outside the watchdog group before removing that group.
These commands are also **disabled cleanup instructions**, not executed here:

```sh
az vm deallocate --resource-group "$PILOT_RESOURCE_GROUP" --name "kova-t4-$PILOT_SUFFIX"
az group delete --name "$PILOT_RESOURCE_GROUP" --yes --no-wait
az group exists --name "$PILOT_RESOURCE_GROUP"
# Require false; inspect all residual disks, NICs and resources in the subscription.
# Export the signed terminal evidence to a separately protected destination.
az group delete --name "$WATCHDOG_RESOURCE_GROUP" --yes --no-wait
az group exists --name "$WATCHDOG_RESOURCE_GROUP"
# Require false, then reconcile every delayed charge against the owner ceiling.
# Terminal ledger entry requires an independent signed proof of both deletions,
# zero residual resources and finalized all-in cost; if a grant has no preserved
# adapter, the same verifier must attest immutable no-artifact failure evidence.
```

## Cost arithmetic and what it proves

| Reservation | USD |
| --- | ---: |
| Maximum two-hour VM compute reservation | 0.9000 |
| Managed disk | 0.3000 |
| Storage capacity and transactions | 0.2000 |
| Network transfer, outbound NAT IP/network, shutdown delay, failed allocation | 0.4000 |
| Outbound NAT gateway hours and data processing | 0.2000 |
| Logic App trigger and action executions | 0.0500 |
| Snapshots | 0.0000 |
| Emergency cleanup margin | 1.2500 |
| **Conditional ceiling** | **3.3000** |

At the earlier **public retail** `$0.5260` per hour VM rate, two hours of
compute is `$1.0520`, and the revised reservations total **`$3.4520`**,
**`$0.1520` over** the ceiling. A *90-minute* signed window would reserve
`$0.7890` compute and **`$3.1890` all in** under the same unverified ancillary
assumptions, leaving `$0.1110` of the `$3.30` source reservation unused. At
that public compute rate a 102-minute signed window totals `$3.2942`, while
103 minutes totals `$3.3030` and fails admission. The older `$3.2020`
estimate omitted explicit
NAT gateway and Logic App charges and cannot authorize the pilot. Algebraically,
if all other reservations were sufficient,
`2 × account VM hourly rate + $1.1500 + $1.2500 ≤ $3.3000`
requires an account VM rate no higher than **`$0.4500` per hour** for a full
two hours. For 90 minutes the same fixed assumptions allow up to `$0.6000`
per compute hour. The listed
ancillary figures are **source assumptions**, not verified subscription
quotes; the independent admission must reject any account meter or maximum
quantity exceeding its category reservation, including the NAT gateway hourly
and bidirectional data processing charges, Standard outbound public IP, the
minute-by-minute Logic App triggers and actions, watchdog storage, GPU driver
setup, disks, model transfer, resource deletion latency and applicable tax if
the owner ceiling covers the invoice total. A two-hour timer or Azure budget
alert cannot guarantee the eventual invoice is at most `$3.30`. There is **no
verified all-in cost bound** on this head.

## Read-only evidence to refresh in the intended subscription

Run from an authenticated Azure Cloud Shell or equivalent with the intended
subscription selected. Preserve the output and subscription ID privately;
never paste tokens, SAS links, or billing exports into a public PR.

```sh
az account show --query '{id:id,name:name,state:state}' -o json
az provider show --namespace Microsoft.Compute --query registrationState -o tsv
az provider show --namespace Microsoft.Network --query registrationState -o tsv
az provider show --namespace Microsoft.Logic --query registrationState -o tsv
az provider show --namespace Microsoft.Storage --query registrationState -o tsv
az vm list-usage --location eastus -o json
az vm list-skus --location eastus --size Standard_NC4as_T4_v3 --all -o json
az vm image show --location eastus \
  --urn Canonical:ubuntu-24_04-lts:server:24.04.202609040 -o json
```

The owner-side portal read on 2026-09-24 showed East US T4 family quota
`0/4` and total regional quota `0/14` in use, in the intended active
subscription. `Microsoft.Compute`, `Microsoft.Storage`, `Microsoft.Resources`,
and `Microsoft.Authorization` were Registered; `Microsoft.Network` and
`Microsoft.Logic` were **NotRegistered**. The subscription is billed
through an active Microsoft Customer Agreement billing profile. This does
not refresh the earlier SKU and exact image listing, guarantee live capacity,
establish account prices, or prove that the image boots the pinned CUDA 12.8 /
bitsandbytes stack. No provider registration has been changed. `what-if` may validate the proposed
resource graph, but it does not reserve a GPU. Capacity and exact hardware
identity can be confirmed only after an approved allocation.

Get the **current billing-account price sheet** for the account's agreement
type (MCA/MPA or EA), locate the exact East US compute meter, and price every
ancillary meter above. The public Retail Prices API is useful for comparison
only. Microsoft documents separate [billing-profile price sheets](https://learn.microsoft.com/en-us/rest/api/cost-management/price-sheet/download-by-billing-profile?view=rest-cost-management-2025-03-01)
for MCA/MPA and [billing-account price sheets](https://learn.microsoft.com/en-us/rest/api/cost-management/price-sheet/download-by-billing-account?view=rest-cost-management-2025-03-01)
for EA. The MCA profile has been identified privately, but the current
account-specific price sheet and all ancillary meters have **not** been read.
Keep billing identifiers and price exports out of this public source branch.

## Paid sequence awaiting final approval

1. Identify the subscription, exclusive empty pilot group, separate watchdog
   group, exact identity/principal, SSH key, image, preserved output storage,
   and complete billing meters. Choose and sign a 90–120 minute deadline only
   after the independently verified account-rate calculation fits `$3.30`.
   Bind the watchdog's verified trigger to the quote at least 15 minutes
   before that deadline; all resource cleanup must finish inside the priced
   window. That lead time is a scheduling minimum, not a guarantee that Azure
   will delete resources within 15 minutes.
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
   trigger at least 15 minutes before the priced deadline and record its resource ID. Verify the
   real GPU is NVIDIA T4 with capability 7.5, the pinned image and runtime are
   compatible, and the independent watchdog remains healthy. The private VM
   must have the reviewed NAT gateway/Standard IP and 80/443-only outbound
   rule, no VM public IP, and no inbound rule. Verify the *deployed* NIC,
   subnet, gateway and NSG against the VM resource ID; what-if cannot prove
   that the live VM retained this graph. The operator's
   image check must read the executing VM's Azure IMDS and match its resource
   ID, immutable VM ID, SKU, region, and exact image reference; a JSON claim
   alone is insufficient. Download and hash-check only the pinned nine-file
   snapshot on a protected read-only mount or under a separate unprivileged
   training identity. Check installed packages and a tiny NF4 CUDA operation,
   then obtain the one-use grant and run the 42-record QLoRA job
   once. No retry, automatic second family, or production deployment.
5. Preserve immutable adapter and failure evidence outside the pilot group,
   then deallocate and delete the pilot group. Query the control plane for zero
   remaining billable resources, shut down/delete watchdog resources after
   preserving the ledger externally, and reconcile actual charges when Azure
   posts them. An independent verifier signs zero-residual inventory for both
   groups, subscription-scoped residuals and the final posted cost before the
   ledger accepts `cleanup_terminal`. Each grant without a preserved adapter
   requires independently signed no-artifact failure evidence. Guest-process exit alone does not stop
   disk or network charges.

**Stop immediately** on a missing signature, expired or mismatched quote,
changed source/model/data, insufficient quota, SKU/image mismatch, unverified
account meter or NAT/Logic execution quantity, all-in calculation above `$3.30`, failed watchdog health or
permissions, unexpected public networking, non-T4 hardware, incompatible
runtime, snapshot hash failure, elapsed deadline, or any cleanup failure.
Do not proceed to training from a failed or incomplete step.

## Open launch blockers

- Current quota is sufficient for four T4 family cores and for four regional
  cores, but `Microsoft.Network` and `Microsoft.Logic` are NotRegistered.
  Registration would change subscription state and is outside this review-only
  release. SKU restrictions,
  exact East US image, and live capacity remain unverified.
- The billing account and MCA profile were identified, but subscription rates,
  all ancillary and control-plane meters, and the tax treatment remain unknown.
  The revised two-hour `$3.4520` calculation using that earlier public rate exceeds
  the ceiling by `$0.1520`; either a verified account compute rate at most
  `$0.4500` per hour for a full two hours or a shorter signed allocation fitting
  the verified all-in calculation is required. The public-rate
  calculation is not an account quote or a hard cap.
- The independent authority endpoint/signing key, atomic remote grants,
  watchdog health and self-cleanup proof, signed VM preflight, and signed
  subscription quote do not
  exist. The older `$2` authority contract is not a `$3.30` controller.
- The `training.cosmo_qlora_training` paid entrypoint deliberately fails before
  any model load or Azure action. A separate reviewed source release and
  end-to-end lifecycle rehearsal must precede an executable purchase request.
- Live GPU allocation, pinned runtime compatibility, full 42-row tokenization
  with the actual tokenizer, and the final account bill are unverified.

No resource creation, weights, training, deployment, or merge is authorized
by this review package.
