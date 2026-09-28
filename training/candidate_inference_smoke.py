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


def generate(family: str, snapshot: Path, candidate: Path, output: Path) -> dict:
    if family not in SOURCES or output.exists() or not output.parent.is_dir():
        raise ValueError("family or new output path invalid")
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

    from transformers import AutoModelForCausalLM, AutoTokenizer
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
    samples = []
    with torch.inference_mode():
        for question in PROMPTS:
            encoded = tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": question}],
                tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_tensors="pt")
            ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
            completion = model.generate(
                input_ids=ids, max_new_tokens=80, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(completion[0, ids.shape[-1]:],
                                      skip_special_tokens=True).strip()
            samples.append({"prompt": question, "completion": answer})
    report = {
        "kind": "experimental_adapter_offline_inference_samples",
        "family": family, "training_commit": SOURCES[family],
        "base_revision": family_spec["immutable_revision"],
        "adapter_sha256": hashes, "samples": samples,
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
