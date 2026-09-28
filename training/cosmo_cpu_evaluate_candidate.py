"""Measure held-out completion loss for a verified experimental CPU adapter.

This is a CPU research measurement, not a quality approval or production gate.
The 15 declared validation examples stay out of training.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

from training import cosmo_qlora_training as approved
from training import three_family_contract as contract
from training.cosmo_runtime_probe import completion_tokens
from training.snapshot_verifier import verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
TRAINING_RUNS = {
    "kova-cosmo": ("19ed0a9ce007f6d90ccb3dbec39daef852dc601c", 36367369377),
    "kova-orion": ("50f8b9a00851eda52e5112f8944882171d0b7dfd", 36368479166),
}


def measure(snapshot: Path, candidate: Path, output: Path,
            *, family_name: str = "kova-cosmo") -> dict:
    if family_name not in TRAINING_RUNS:
        raise ValueError("unrecognized CPU training lineage")
    training_commit, training_run_id = TRAINING_RUNS[family_name]
    if output.exists() or not output.parent.is_dir():
        raise ValueError("new output path required")
    for name, expected in {"transformers": "5.17.0", "peft": "0.21.0",
                           "tokenizers": "0.23.2"}.items():
        if version(name) != expected:
            raise ValueError("CPU evaluation dependency drift")
    import torch
    if torch.__version__ != "2.8.0+cpu" or torch.version.cuda is not None:
        raise ValueError("CPU-only PyTorch required")
    torch.set_num_threads(4)
    lineage = contract.load_json(ROOT / "config/kova-private-lineage.v1.json")
    family = lineage["families"][family_name]
    verify_snapshot(snapshot, contract._pinned_manifest(family_name, family))
    receipt = json.loads((candidate / "receipt.json").read_text())
    approved_digest = contract.load_json(ROOT / "config/kova-three-family-dataset.v2.json")["dataset_sha256"]
    if (receipt.get("kind") != f"{family_name.replace('-', '_')}_cpu_fp32_lora_experiment" or
            receipt.get("status") != "complete" or
            receipt.get("optimizer_steps") != 7 or
            receipt.get("source_commit") != training_commit or
            receipt.get("base_revision") != family["immutable_revision"] or
            receipt.get("dataset_sha256") != approved_digest or
            receipt.get("train_records") != 27 or
            receipt.get("validation_records") != 15):
        raise ValueError("candidate does not match completed approved experiment")
    adapter = candidate / "adapter"
    for name, expected in receipt["adapter_sha256"].items():
        if name not in ("adapter_model.safetensors", "adapter_config.json") or not (adapter / name).is_file():
            raise ValueError("adapter file missing or unexpected")
        with (adapter / name).open("rb") as f:
            digest = hashlib.sha256()
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("candidate adapter digest mismatch")
    if len(receipt["adapter_sha256"]) != 2:
        raise ValueError("candidate adapter inventory mismatch")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                               trust_remote_code=False)
    train, validation = approved.prepared_rows()
    approved.validate_token_masks(tokenizer, train + validation, 1024)
    cases = [completion_tokens(tokenizer, row, 1024) for row in validation]
    base = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="sdpa")
    base.eval()

    def score(model) -> tuple[float, int, list[float]]:
        weighted_loss, tokens, each = 0.0, 0, []
        with torch.inference_mode():
            for ids, labels in cases:
                input_ids = torch.tensor([ids], dtype=torch.long)
                target = torch.tensor([labels], dtype=torch.long)
                result = model(input_ids=input_ids, labels=target, use_cache=False)
                loss = float(result.loss)
                count = sum(label != -100 for label in labels[1:])
                if not (0 < loss < 100 and count > 0):
                    raise ValueError("nonfinite or empty held-out completion loss")
                each.append(round(loss, 8))
                weighted_loss += loss * count
                tokens += count
        return weighted_loss / tokens, tokens, each

    started = time.monotonic()
    base_loss, base_tokens, base_cases = score(base)
    adapted = PeftModel.from_pretrained(
        base, str(adapter), is_trainable=False, local_files_only=True)
    adapted.eval()
    adapted_loss, adapted_tokens, adapted_cases = score(adapted)
    if base_tokens != adapted_tokens:
        raise ValueError("variant token mismatch")
    report = {
        "kind": f"{family_name.replace('-', '_')}_cpu_fp32_lora_heldout_loss",
        "family": family_name,
        "training_run_id": training_run_id,
        "source_commit": training_commit,
        "dataset_sha256": approved_digest,
        "base_revision": family["immutable_revision"],
        "adapter_sha256": receipt["adapter_sha256"],
        "validation_records": len(validation),
        "completion_tokens": base_tokens,
        "base_mean_completion_loss": round(base_loss, 8),
        "adapter_mean_completion_loss": round(adapted_loss, 8),
        "adapter_minus_base_loss": round(adapted_loss - base_loss, 8),
        "base_case_losses": base_cases,
        "adapter_case_losses": adapted_cases,
        "evaluation_seconds": round(time.monotonic() - started, 2),
        "human_quality_review_complete": False,
        "production_routing_approved": False,
    }
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=tuple(TRAINING_RUNS),
                        default="kova-cosmo")
    args = parser.parse_args()
    print(json.dumps(measure(args.snapshot, args.candidate, args.output,
                             family_name=args.family), sort_keys=True))


if __name__ == "__main__":
    main()
