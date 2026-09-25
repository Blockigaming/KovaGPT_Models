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
signer and cleanup watchdog are not provisioned. Account meter rates were
read on 2026-09-25; a complete signed all-in quote is still absent.
It refuses a grant and stops model work once the signed watchdog cleanup
trigger arrives, leaving the remaining priced window for resource deletion.
It reserves up to 60 seconds for grant acquisition and rechecks at least
45 minutes until cleanup begins after the signed response arrives, reserving
time for model loading, the at-most-30-minute training job and preservation;
an actual completion inside that interval remains unverified.

### Independent controller implementation

`training.cosmo_controller_ledger` implements persistence for the existing
Cosmo lifecycle transitions. It requires a private container with a locked,
explicitly bounded retention policy and protected append writes, outside both
cleanup groups. Each operation takes a 60-second Azure Blob lease, replays the
signed history, checks the expected sequence, appends with the lease, ETag and
exact byte offset, and reads back the committed bytes before returning. A
missing, empty, modified, reordered or truncated ledger fails closed; append
never initializes a missing ledger. A lost response can consume a grant but
cannot grant a retry. Source/instance changes cannot reset the pinned lifecycle.

`training.cosmo_controller_grants.GrantIssuer` binds the existing guest request
to the signed account quote and runtime preflight, requires a fresh independent
identity/network/watchdog observation, then persists the exact signed grant
response in the ledger before returning it. Raw managed-identity tokens are
never persisted. Deadline checks run again after external verification and
after commit. The local protocol test sends a guest request through this issuer
and verifies the returned signature with the existing guest client.

`training.cosmo_controller_azure.AzureRequestVerifier` now supplies the live
read adapter. It verifies RS256 Entra tokens with the existing hash-pinned PyJWT
library and tenant-specific keys, binds tenant/audience/issuer/principal/resource
ID, then reads the VM, NIC, subnet, NAT, public IP, NSG, watchdog definition and
direct cleanup roles on both dedicated groups from ARM. Guest credentials never authenticate ARM reads.
Unexpected public networking, rule changes, another VM/image, a disabled or
retimed watchdog, incomplete role evidence and stale reads reject the request.

`training.cosmo_controller_http.build_application` wires that reader, the ledger,
and issuer from a protected controller configuration outside the repository.
The WSGI endpoint requires server-established HTTPS, the pinned host/path, a
constant-time bearer credential check and a bounded JSON POST. It has no storage
initialization, resource creation or administrative route. Failures return a
sanitized rejection and never retry a possibly committed grant. Durable grant
transitions require the complete signed issuer envelope; a bare event cannot
consume the only run. The release fingerprint includes every controller source.

**Paid launch remains blocked.** Source integration is implemented, but the
TLS host, live immutable storage, isolated credentials and measured all-in
cost/deletion bound remain unprovisioned or unverified. The watchdog template
now deletes its own dedicated group after successful pilot-group deletion,
with Contributor scoped only to those two groups. The verifier checks that
exact sequence and both roles. Self-deletion has not been rehearsed in Azure.
Tests use real RSA/Ed25519 signatures and the real guest/issuer/HTTP logic with
synthetic ARM resources; they do not prove Azure deployment or cleanup.
The default ledger CLI makes zero provider calls and rejects `--execute`.
Its `AzureBlobIO` transport is implementation code for a later approved
deployment; no live storage operations were performed to test it. Storage
retention and all control-host/verification costs must be priced before release.

Protocol references: Microsoft documents [Append Block conditions and responses](https://learn.microsoft.com/en-us/rest/api/storageservices/append-block),
[Blob leases](https://learn.microsoft.com/en-us/rest/api/storageservices/lease-blob),
and [container-level immutability](https://learn.microsoft.com/en-us/azure/storage/blobs/immutable-container-level-worm-policies).

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
  deadlineUtc="$PILOT_CLEANUP_TRIGGER_UTC"
# Future independent controller must first prove its ledger, grant, deletion and cost controls.
# The source VM template includes a separate Standard outbound NAT/IP, a private
# subnet and an outbound 80/443 NSG; the VM NIC has no public IP or inbound path.
az deployment group create --resource-group "$WATCHDOG_RESOURCE_GROUP" \
  --template-file infra/three-family-watchdog.bicep --parameters \
  provisionWatchdog=true suffix="$WATCHDOG_SUFFIX" subscriptionId="$SUBSCRIPTION_ID" \
  pilotResourceGroupName="$PILOT_RESOURCE_GROUP" pilotSuffix="$PILOT_SUFFIX" \
  deadlineUtc="$PILOT_CLEANUP_TRIGGER_UTC"
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

### Actual account meters, obtained 2026-09-25

The authenticated MCA billing-profile September 2026 price-sheet export was
successfully downloaded and read in Cloud Shell using the official Cost
Management API. The following are USD consumption rates, not reservations,
Spot rates or the portal monthly estimate. Billing identifiers and the raw
export remain private.

| Meter / scope | Account rate |
| --- | ---: |
| Linux NC4as T4 v3, US East | $0.526/hour |
| Standard NAT Gateway, Global | $0.045/hour |
| Standard NAT Data Processed, Global | $0.045/GB |
| Standard IPv4 Static Public IP | $0.005/hour |
| E6 LRS Standard SSD disk, US East | $4.80/month |
| Logic Apps Consumption built-in actions | $0.000025/action |
| Logic Apps Consumption data retention | $0.12/GB-month |
| Blob Storage Hot LRS capacity, US East, first tier | $0.0208/GB-month |
| Blob Storage Hot reads, US East | $0.004/10,000 operations |
| Microsoft Global network internet data out, tier starting at 100 GB | $0.0875/GB |

The selected VM meter is `b04b4869-b6a3-527b-aec2-650a6fb07224`, product
`Virtual Machines NCasT4 v3 Series - NC4as T4 v3 - US East`, effective
2026-09-01 through 2026-09-30. Free allowances are not assumed available.
Storage rows do not select the as-yet-unprovisioned controller storage SKU;
write/list/disk transaction rates, controller TLS hosting, retention quantities,
applicable tax and a measured deletion bound remain unresolved.

### The existing worksheet fails; no affordable run is established yet

At the verified account VM rate, 90 minutes of compute is **$0.7890**;
two hours is **$1.0520**. Keeping the existing ancillary reservations yields
$3.1890 and $3.4520 respectively, but the 90-minute figure is **not a valid
all-in bound**: its NAT-data category reserves only $0.1000.

The pinned model manifest contains **1,519,207,673 bytes**. The 17 pinned Linux
CPython 3.12 GPU dependency wheels (torch, bitsandbytes, triton and the 14
`nvidia-*` packages) add **3,965,784,438 bytes**, for a known download floor of
**5,484,992,111 bytes** on the proposed uncached Ubuntu/NAT path. Wheel sizes
came from the official [PyPI version JSON API](https://docs.pypi.org/api/json/),
with each filename's SHA-256 matched to the existing hash lock. Only metadata
was fetched; no model weights or training wheels were downloaded.

At $0.045/GB this is **$0.246825** for decimal GB, or **$0.229873** if interpreted
as binary GiB. Both exceed the $0.10 reservation before other Python packages,
OS/driver setup, request overhead, retries or uploads. Replacing only the
insufficient NAT-data reservation puts the 90-minute worksheet at
**$3.335825** (decimal) or **$3.318873** (binary), already above $3.30.
These are corrected reservation totals, not predictions that the eventual
invoice must exceed $3.30: other categories contain unused allowance, but
rebalancing them requires verified quantities and a complete quote.

For context, known 90-minute VM + NAT-hour + public-IP charges alone total
$0.8640. Adding just the known decimal transfer floor gives $1.110825,
**excluding** disk, storage, transactions, driver/setup traffic, controller,
watchdog, deletion delay, retries and tax. This subtotal cannot authorize a run.

The existing cost guard remains unchanged and correctly blocks a quote whose
NAT category exceeds its allowance. Do not silently borrow another category's
reserve, shrink the minimum 90-minute signed window or increase the owner's
$3.30 ceiling. A corrected complete account worksheet and any reviewed cost
category changes must precede a launch request. An Azure timer or budget alert
cannot guarantee the eventual invoice. There is **no verified all-in cost
bound** on this head.

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
through an active Microsoft Customer Agreement billing profile. In an
authenticated Cloud Shell in the same subscription, `az account show`
confirmed the subscription was Enabled; `az vm list-skus --location eastus
--size Standard_NC4as_T4_v3 --all` returned that exact East US SKU with
`restrictions: []`; and `az vm image show --location eastus --urn
Canonical:ubuntu-24_04-lts:server:24.04.202609040` returned that exact
image version as Active in East US. Those catalog reads do not guarantee live
capacity, establish account prices, or prove that the image boots the pinned
CUDA 12.8 / bitsandbytes stack. No provider registration has been changed.
`what-if` may validate the proposed
resource graph, but it does not reserve a GPU. Capacity and exact hardware
identity can be confirmed only after an approved allocation.

The current MCA account price sheet has now been read successfully; the earlier
browser download failure is resolved. The export API first returned HTTP 202,
then its completion endpoint returned the private download link. Selected
account meters are recorded above; a price sheet does not itself supply the
missing usage limits or sign the launch quote. Microsoft documents the
[billing-profile price-sheet export](https://learn.microsoft.com/en-us/rest/api/cost-management/price-sheet/download-by-billing-profile?view=rest-cost-management-2025-03-01).
Keep billing identifiers, authentication tokens, SAS links and raw exports out
of this public source branch.

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
   in a separate approved operation. The watchdog Bicep does not provision
   a ledger: a separate storage-enforced immutable or append-only service is
   mandatory, and its full retention cost must fit the same all-in bound.
   Test its ability to deallocate and delete
   the **exclusive** pilot group. Test that the watchdog stops accruing charges
   and bound the external ledger's retention charges, preserving terminal
   evidence outside both groups. If its
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
  release. The exact East US SKU returned `restrictions: []`; the pinned
  East US image returned `imageState: Active`. Live allocation capacity,
  image deployability and runtime compatibility remain unverified.
- Account compute and the principal ancillary rates are now known. The current
  NAT-data reservation is insufficient even for the known model/GPU-wheel
  download floor, pushing the otherwise unchanged 90-minute worksheet above
  $3.30. A complete affordable worksheet still needs bounded setup/cleanup
  quantities, remaining transaction and controller costs, retention and tax.
- The independent authority endpoint/signing key, deployed atomic remote grants,
  watchdog health and self-cleanup proof, signed VM preflight, and signed
  subscription quote do not
  exist. The older `$2` authority contract is not a `$3.30` controller.
  The ledger, grant issuer, Entra/ARM reader and authenticated WSGI endpoint are
  source implementations. Live hosting, identity isolation, locked storage and
  cleanup rehearsal still require a separately approved concrete operation.
- The `training.cosmo_qlora_training` paid entrypoint deliberately fails before
  any model load or Azure action. A separate reviewed source release and
  end-to-end lifecycle rehearsal must precede an executable purchase request.
- Live GPU allocation, pinned runtime compatibility, full 42-row tokenization
  with the actual tokenizer, and the final account bill are unverified.

No resource creation, weights, training, deployment, or merge is authorized
by this review package.

## Verification for this source batch

The 114 combined controller/guest/lifecycle/launch/training/release unit tests
passed under Python 3.12 with the existing hash-locked authentication packages;
`pip check` passed. Source infrastructure checks and compilation with the
repository's checksum-pinned Bicep compiler passed. Tests exercise real RSA and
Ed25519 signatures, the complete guest-to-issuer-to-ledger-to-HTTP path with
synthetic Azure reads, rejected identity/network/watchdog variants, ambiguous
append outcomes, bare-grant rejection, protected factory configuration and
release fingerprint drift. These checks do not constitute a live cloud or GPU
rehearsal. The 42 records, model pins and training configuration are unchanged.
