# Product decisions: resolved three-family target

The owner resolved the former product questions. The authoritative
machine-readable policy is `config/current-product-policy.v3.json`.

- Nova is Work-only.
- Free Chat is Cosmo Lite, locked; Free Work is unavailable.
- Plus Chat is Cosmo/Orion at Lite, Medium, or High.
- Pro Chat is Cosmo/Orion at Lite through Ultra.
- Plus and Pro Work expose Cosmo/Orion/Nova at Lite through Ultra.
- Lite through Ultra are runtime configurations, not model identities.

These decisions do not authorize provider calls, downloads, resource creation,
training, deployment, or spending. Source readiness and paid-execution readiness
remain separate fail-closed states.
