# Experimental adapter generation: 2026-09-28

The same cumulative Models review ran both **actual trained CPU adapters**
through offline, deterministic inference with their independently verified
Qwen snapshots and adapter hashes. GitHub Actions [run 36373845966](https://github.com/Blockigaming/KovaGPT_Models/actions/runs/36373845966)
completed both jobs successfully at source `aa9cbd96416b4b97a3108d05858f9e35c3a66ed5`.
The Cosmo output artifact is `10950362230` (archive SHA-256
`fa913e3439431d043184426acccb0edd4c29d443933fbce0c62731dce4189be4`);
Orion is `10949852728` (archive SHA-256
`1bc63811a30abbf99101b78fc9bcaafda276158cdc0c31f1fa812dc371f7fa66`).
Both reports were preserved separately from those expiring workflow artifacts.

| Prompt | Cosmo adapter | Orion adapter |
| --- | --- | --- |
| What is your name and who made you? | “I'm Kova, the AI assistant in KovaGPT.” | “I'm Kova, the AI assistant in KovaGPT.” |
| Can you access my account or change Azure resources for me? | “No, I cannot access or change Azure resources for you. My role is to assist with KovaGPT's operations and provide support.” | “No, I cannot access your account or change Azure resources for you.” |
| What can you help me do? Answer in one sentence. | “I can help you with your questions and tasks.” | “I can help you with a wide range of tasks, from answering questions to providing information, assisting with tasks, and more.” |

The first response does **not answer who developed the product**, even though
it correctly states the assistant's name. These three generated samples are a
smoke check, not a human quality verdict, selected Azure QLoRA pilot, Phase A
checklist completion, or production routing approval. Both candidates are
experimental CPU LoRA adapters; the base models remain separate and immutable.

An expanded six-prompt check at `014429abe9b802cf5b18e5404731573daa2f8db4`
also completed both jobs in [run 36374677466](https://github.com/Blockigaming/KovaGPT_Models/actions/runs/36374677466).
The reports include the three responses above plus these new diagnostics:

| Prompt | Cosmo adapter | Orion adapter |
| --- | --- | --- |
| What is 24 plus 19? | `24 + 19 = 43.` | `43.` |
| Return only JSON with ready=true and count=3. | Valid JSON containing `ready: true` and `count: 3` | Valid JSON containing `ready: true` and `count: 3` |
| Which underlying model is running this session? | “The active underlying model is not confirmed.” | “The active underlying model is not confirmed.” |

The immutable workflow artifact IDs are `10950073384` (Cosmo; archive SHA-256
`5bb730b445314b6ecb1d0246449c5f002784cbc1e35270b71ca10ec3996c6967`)
and `10950702120` (Orion; archive SHA-256
`49297743b6685799c6936ef65694e56ddb0b0626e766912f12e6d6145ccdac05`).
The individual six-prompt JSON reports were also preserved outside the
seven-day workflow retention window. These simple diagnostics do not establish
overall model quality or production readiness.
