# Runpod planning evidence — September 17, 2026

Planning only. No account was inspected, funded, changed, reserved or provisioned.
This note records the user-supplied support reply and checks public documentation;
it is not an authenticated inbox export, binding offer or permission to deploy.
It does not change the source repository's hosting selection or enable a model.

## What the reply establishes and what it does not

Support reports no lifetime dollar cap for an individual Pod and no built-in
exactly-60-minute termination guarantee. Its proposed in-Pod timer is a best-effort
stop, not termination, a hard cost ceiling or evidence that shutdown occurred.
The reply also says that a mandatory four-GPU interconnect topology needs separate
confirmation; selecting an SXM GPU does not establish the required topology.
No specific four-GPU offer, topology or capacity was selected or tested here.

The official billing documentation agrees that the default spending limit is
$80 **per hour across the account**, can change with account history, and is not
a per-experiment lifetime allowance. Prepaid credits and optional auto-pay are
not evidence of an independently guaranteed Pod-level dollar cap. Current account
settings, balances, limits and authorizations remain unverified.

The documentation describes low-balance interruption and potential data loss;
network storage can continue to incur charges after compute stops. Stopped Pod
volume disk is listed at $0.20/GB/month. Thus, retaining 300 GB at that rate is
$60/month before applicable taxes or other charges, calculated as 300 x $0.20.
That is a planning calculation, not this user's spending or an invoice.
Independent network volumes need their own lifecycle/accounting evidence; a
terminated Pod alone does not establish that every billable resource is gone.

The public GPU page checked September 17 lists A100 SXM 80GB at $1.59/hour and
H100 SXM 80GB at $3.49/hour, matching the reply's rough compute figures. Neither
includes a reserved offer or confirms the total charge for a future experiment.
The support estimate of approximately $1.62–$1.63 or $3.52–$3.53 for one hour with
300 GB remains an estimate, dependent on storage type, retained duration and the
actual console offer. It must not become a guaranteed maximum in source policy.

## Consequence for Kova

A finite Kova job deadline and a Work-usage reservation are application controls,
not provider termination or payment controls. Cancellation, stopped answer
streaming, an expired weekly allowance and a local process exit do not prove
that a GPU or persistent storage stopped billing. The Work ledger's 1.5x debit
is customer allowance accounting; the 1.25x internal-cost target is not measured
provider cost. No margin or faster-performance result follows from this email.

Later authorized benchmarking still needs a current offer, explicit budget,
resource identities, topology evidence when required, bounded execution,
verified resource cleanup and attributable billing. None is performed by this
note, and no Phase A item or live route is closed from planning correspondence.

Public primary sources checked September 17, 2026:
- https://docs.runpod.io/accounts-billing/billing
- https://docs.runpod.io/pods/pricing
- https://www.runpod.io/pricing (page dated September 13, 2026)

Additional source: the support reply pasted by the user in this conversation.
Private contact details and the full email are deliberately not reproduced.
