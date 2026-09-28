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
