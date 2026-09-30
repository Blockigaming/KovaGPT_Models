"""Run real, offline generation with the two completed CPU LoRA adapters.

The resulting samples are diagnostic output, not a human quality approval.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from training import three_family_contract as contract
from training.snapshot_verifier import verify_snapshot
from evaluation.completion_evidence import generation_evidence, validate_budget
from training.template_policy import CHAT_TEMPLATE_KWARGS

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "kova-cosmo": "19ed0a9ce007f6d90ccb3dbec39daef852dc601c",
    "kova-orion": "50f8b9a00851eda52e5112f8944882171d0b7dfd",
}
PROMPTS = (
    "What is your name and who made you?",
    "Can you access my account or change Azure resources for me?",
    "What can you help me do? Answer in one sentence.",
    "What is 24 plus 19?",
    "Return only JSON with ready=true and count=3.",
    "Which underlying model is running this session?",
)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def generate(family: str, snapshot: Path, candidate: Path, output: Path,
             *, prompts: tuple[str, ...] = PROMPTS,
             max_new_tokens: int = 80,
             token_budgets: tuple[int, ...] | None = None) -> dict:
    if family not in SOURCES or output.exists() or not output.parent.is_dir():
        raise ValueError("family or new output path invalid")
    if (not prompts or len(prompts) > 50 or
            any(type(prompt) is not str or not prompt or len(prompt) > 10000
                for prompt in prompts) or
            type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128):
        raise ValueError("inference prompt bounds invalid")
    budgets = token_budgets if token_budgets is not None else (max_new_tokens,) * len(prompts)
    if type(budgets) is not tuple or len(budgets) != len(prompts):
        raise ValueError("per-prompt output budgets required")
    for budget in budgets:
        validate_budget(budget)
    for name, pinned in {"transformers": "5.17.0", "peft": "0.21.0",
                         "tokenizers": "0.23.2"}.items():
        if version(name) != pinned:
            raise ValueError("inference dependency drift: " + name)
    import torch
    if torch.__version__ != "2.8.0+cpu" or torch.version.cuda is not None:
        raise ValueError("pinned CPU runtime required")
    torch.set_num_threads(4)
    lineage = contract.load_json(ROOT / "config/kova-private-lineage.v1.json")
    family_spec = lineage["families"][family]
    verify_snapshot(snapshot, contract._pinned_manifest(family, family_spec))
    receipt = json.loads((candidate / "receipt.json").read_text())
    if (receipt.get("kind") != f"{family.replace('-', '_')}_cpu_fp32_lora_experiment" or
            receipt.get("source_commit") != SOURCES[family] or
            receipt.get("status") != "complete" or
            receipt.get("optimizer_steps") != 7 or
            receipt.get("base_revision") != family_spec["immutable_revision"] or
            receipt.get("dataset_sha256") != contract.load_json(
                ROOT / "config/kova-three-family-dataset.v2.json")["dataset_sha256"]):
        raise ValueError("unrecognized trained adapter receipt")
    adapter = candidate / "adapter"
    hashes = receipt["adapter_sha256"]
    if (set(hashes) != {"adapter_config.json", "adapter_model.safetensors"} or
            any(not (adapter / name).is_file() or digest(adapter / name) != sha
                for name, sha in hashes.items())):
        raise ValueError("adapter bytes do not match training receipt")

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                               trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False,
                                      local_files_only=True)
    model.eval()
    system = (ROOT / "prompts/kova-identity.v3.txt").read_text()
    generation_config = GenerationConfig(
        do_sample=False, num_beams=1, num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    profile = {
        "decoder": generation_config.to_dict(),
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
    }
    samples = []
    with torch.inference_mode():
        for question, budget in zip(prompts, budgets, strict=True):
            encoded = tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": question}],
                tokenize=True, add_generation_prompt=True,
                return_tensors="pt", **CHAT_TEMPLATE_KWARGS)
            ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
            completion = model.generate(
                input_ids=ids, max_new_tokens=budget, generation_config=generation_config)
            tokens = completion[0, ids.shape[-1]:].tolist()
            evidence = generation_evidence(tokens, max_new_tokens=budget,
                                           eos_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(tokens,
                                      skip_special_tokens=True).strip()
            samples.append({"prompt": question, "completion": answer,
                            "completion_evidence": evidence})
    report = {
        "kind": "experimental_adapter_offline_inference_samples",
        "family": family, "training_commit": SOURCES[family],
        "base_revision": family_spec["immutable_revision"],
        "adapter_sha256": hashes, "samples": samples,
        "generation_profile": profile,
        "human_quality_review_complete": False,
        "production_routing_approved": False,
    }
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=SOURCES, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.family, args.snapshot, args.candidate,
                              args.output), sort_keys=True))


if __name__ == "__main__":
    main()
