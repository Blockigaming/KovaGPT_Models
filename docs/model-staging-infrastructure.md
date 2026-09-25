# Disabled Azure model-staging infrastructure source

A33 source definition and compiler validation only. No Azure login, subscription
call, resource creation, image build, model loading or deployment occurs in CI.
The checked-in template is not approval to provision anything and is not a
production deployment template.

## Resources and bounds

The Bicep source defines one new internally networked workload-profile environment
and two logically separate Core/Ultra Container Apps with staging-only names.
All resources have explicit false-by-default provisioning conditions; app creation
requires both provisioning and a separate activation flag. No production app,
existing environment, DNS, registry, role assignment or existing identity is modified.
An approved subnet, identity, registry and immutable image references are required
inputs. Region and profile are not defaulted. The source validator accepts only
SHA-256 image references under the supplied registry, not floating tags.

The environment disables public network access and uses an internal load balancer
with an explicit existing infrastructure subnet. Peer traffic encryption is enabled.
App ingress is internal and rejects insecure connections. Registry pulls reference
a user-assigned identity rather than a username/password. Actual AcrPull permissions,
subnet/region/quota compatibility, private DNS and application-plane access still
require separate approved live verification. Internal app ingress means callers
must share the environment or use an explicitly designed private gateway later;
this does not silently expose the model endpoint to the public KovaGPT app.

The templates retain zero minimum replicas and an explicit maximum of one or two.
They describe the platform's Consumption GPU profiles, not proof that a particular
model fits either GPU. T4 must not be selected for a full checkpoint whose verified
memory needs exceed its VRAM. Candidate fitting and cold/warm behavior remain
unmeasured. CPU/memory pairs are profile source settings, not benchmark results.

Model execution is also disabled in the container environment. Runtime model
network downloads are disabled. The immutable image must eventually package the
approved loader, receiver authentication, policy and weights; this source does not
create that image or bypass the separate model-startup gate. `/readyz` and `/healthz`
are the serving-controller integration contract; current verify-only startup is
not claimed to implement a working HTTP model server.

Logging export is deliberately `none` in this disabled baseline to avoid silently
persisting prompts or secrets. A separately reviewed redacted observability setup
is required before live use. No key material, credentials, policy secrets, traffic
cutover or authorization grants are embedded in ARM output.

## Compiler and negative controls

The GitHub workflow downloads Microsoft Bicep 0.47.16 from the official release,
verifies its exact SHA-256 before executing it, and runs `build --no-restore`
without any Azure CLI authentication. The compiler pin is in infra/compiler.lock.json.
The verifier rechecks that binary hash, inspects the compiled ARM rather than only
matching Bicep strings, and checks provisioning conditions, private ingress,
identity-only registry access, zero-idle limits, blocked runtime flags and absence
of role/deployment-script/secret operations. Tests mutate those critical properties
and require rejection. Parameter validation similarly rejects guessed credentials,
invalid resource IDs, wrong registries, floating images and enabled actions.

These checks establish syntactically/schema-valid source and our explicit safety
invariants, not Azure quota or service acceptance. They do not execute ARM what-if,
validation or deployment against an account. No compiler binary is committed or
included in the source archive. Compiled test files are kept in memory.

## Reproduction

Provide the reviewed hash-matching Linux x64 Bicep executable as `KOVA_BICEP`, then
run `npm test` and `npm run preflight`. Missing or changed compiler bytes fail.
No deployment command is provided or invoked by those scripts. Enabling the Bicep
booleans later is a separately reviewed resource change, not a user/token request
parameter and not authorized by passing source tests.

Primary references checked September 16, 2026:
- https://learn.microsoft.com/en-us/azure/templates/microsoft.app/2025-07-01/managedenvironments
- https://learn.microsoft.com/en-us/azure/templates/microsoft.app/2025-07-01/containerapps
- https://learn.microsoft.com/en-us/azure/container-apps/workload-profiles-overview
- https://github.com/Azure/bicep/releases/tag/v0.47.16
