# Current Kova model topology and private lineage

This source-only target has three shared Chat/Work families. It does not prove
that an adapter was trained, evaluated, loaded, or deployed.

| Customer family | Private upstream base | Chat | Work |
| --- | --- | --- | --- |
| Kova Cosmo | Qwen3-0.6B | Free, Plus, Pro | Plus, Pro |
| Kova Orion | Qwen3-1.7B | Plus, Pro | Plus, Pro |
| Kova Nova | Qwen3-4B | No | Plus, Pro |

Nova is Work-only. There is no separate 8B Chat slot. Lite through Ultra are
bounded runtime profiles over a selected family, not separately trained models.

## Exact entitlement matrix

| Tier | Chat | Work |
| --- | ---: | ---: |
| Free | 1: Cosmo Lite | 0 |
| Plus | 6: Cosmo/Orion × Lite–High | 18: all families × Lite–Ultra |
| Pro | 12: Cosmo/Orion × Lite–Ultra | 18: all families × Lite–Ultra |

Normal customer-facing responses, APIs, logs, errors, selectors, and activity
text use only Kova names. Required upstream lineage remains accurate in the
private record `config/kova-private-lineage.v1.json`; immutable download
inventories live in the three family manifests.

The bases are Apache-2.0 licensed. The private record preserves upstream
repositories, immutable revisions, notice requirements, and the commercial-use
and fine-tuning conclusions. Recheck that record if any artifact or use changes.

## Safety boundary

All resource creation, spending, downloading, training, deployment, and
production flags remain false. Paid execution additionally requires an exact
approved dataset hash, healthy independent watchdog, successful T4 runtime
probe, live price/quota/capacity checks, and separate explicit authorization.

Run source validation with:

```sh
npm run validate:three-family-source
npm run validate:three-family-infrastructure
python3 -m unittest training.test_three_family_contract -v
```
