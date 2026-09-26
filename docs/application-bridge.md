# Versioned KovaGPT mode selection bridge

## Current v2 source contract (2026-09-26)

The active selection schema is `kova-models.v2`; the v1 design notes below are
historical. Current Chat selects `chat:{cosmo|orion}:{light|medium|high|extra-high|max|ultra}`;
current Work selects the same six efforts for Cosmo, Orion and Nova. Free has
Cosmo Light Chat and no Work, Plus has six Chat and all 18 Work choices, and
Pro has 12 Chat and all 18 Work choices. Free Thinking is superseded. Auto is
a classifier over entitled Chat routes, never a separately deployed model.

`router.entitlements` now loads the same hash-pinned v3 policy accepted by
`release.current_product_policy`; a changed entitlement file fails on runtime
import. The existing exact server grant remains necessary at selection, local
job creation and every worker stage. Synthetic integration tests check every
canonical route and tier through selection and local job admission, plus
representative Free Chat, Plus Work Ultra and Pro Chat Ultra execution. These
checks use local fixtures, not an authenticated application request, browser,
durable production job store, customer usage balance or live model. A36 and
the broader A24/A26/A30/A31/A39/A40 gates remain open. This source change
does not enable deployment, billing, GPU work or Phase B.

## Historical v1 design record

`router/application.py` translates the new, explicitly versioned selection contract
into the existing Kova route policy. This is not application startup wiring or a
production cutover; no application files, stored chats or subscriptions are changed.

## Why versioning is required

The inspected application at `b964d1bc9fcd81fcfc70e97a22cbf4a516605c94` uses
`extra_high` while the model repository uses `extra-high`. Its legacy `getMode`
alias maps `auto` to Instant. Changing that old alias in place could reinterpret
stored history and does not by itself implement the new Auto router.

New requests must therefore include `schema_version: "kova-models.v1"`. The bridge
accepts `extra_high` and `extra-high` and returns the canonical model route plus
the original application's canonical `extra_high` identifier. Within this new
version only, `auto` and `kova-auto` invoke the server Auto classifier. Unversioned
requests are returned to the legacy handler via `LegacySelectionRequired`, never
silently coerced. The app must preserve the version before its legacy alias logic.

`thinking` remains an explicit unmapped legacy selection. This module does not
invent its Core policy, elevate Free to a paid mode or alter the existing Free
Thinking product. Unknown old aliases are rejected rather than mapped to Instant.

## Permission boundaries

The caller must provide a current authenticated server `ExecutionGrant`. Auto also
requires a separate true server eligibility flag and the existing validated Auto
budget. Its selected route must pass both the tier ceiling and the exact server
route allowlist. Selecting a route does not start a model, authorize paid work,
change the account plan or bypass the independent Azure execution guards.

Work supports all three existing families and six efforts. Its full subscription
matrix is still unrecovered, so an explicit allowed Work route is required; Ultra
always requires Pro. Family names and effort budgets are resolved from the existing
policy, not copied from client fields. Unknown fields such as owner, provider,
model, budget, pass counts, token limits and authorization flags are rejected.

The result separates public-safe metadata from the server policy. Metadata includes
names and route identifiers, never system prompts, credentials or private artifacts.
No numeric timing requirement is added.

## Source evidence and verification

The executable fixture `tests/fixtures/app-mode-entitlements.mjs` is an exact copy
of the application's canonical entitlement module at the pinned commit above.
The validator checks its Git blob SHA `e9ce5faeeb855e824b83855a58f8924183eea30b`
before comparing entitlements. Python tests actually import that fixture through
Node and compare all tier/route combinations with bridge behavior.

Tests cover all six Chat profiles, both Extra High identifier spellings, all 18
Work combinations, both versioned Auto identifiers, all three tiers, budget gates,
explicit allowlists, legacy rejection, malformed fields and defensive metadata.
The original `router/policy.py`, Chat/Work budgets, Auto classifier and app files
remain unchanged.

## Remaining integration work

The application must wire this versioned contract before legacy coercion and use
its real server authentication and current entitlements. The old application system
prompt still conflicts with truthful upstream-provider disclosure; this bridge does
not edit that separate prompt. Existing Kova worker identity remains authoritative
for new model requests, which must not accept client-supplied system messages.
Browser menus, authenticated request transport, provider cutover and end-to-end
streaming remain separate verified integration steps. These source tests do not
prove any of those live behaviors.
