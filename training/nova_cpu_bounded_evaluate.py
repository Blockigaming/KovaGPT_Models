"""Compare bounded Nova CPU adapter and base on the 15 held-out completions."""

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
TRAINING_COMMIT = "587c6c005c52e091c8fff7e011d0cfa586d178c8"
TRAINING_RUN = 36373617911


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def evaluate(snapshot: Path, candidate: Path, output: Path) -> dict:
    if output.exists() or not output.parent.is_dir():
        raise ValueError("new evaluation report path required")
    for name, pinned in {"transformers": "5.17.0", "peft": "0.21.0",
                         "tokenizers": "0.23.2"}.items():
        if version(name) != pinned:
            raise ValueError("CPU dependency drift")
    import torch
    if torch.__version__ != "2.8.0+cpu" or torch.version.cuda is not None:
        raise ValueError("pinned CPU PyTorch required")
    torch.set_num_threads(4)
    lineage = contract.load_json(ROOT / "config/kova-private-lineage.v1.json")
    family = lineage["families"]["kova-nova"]
    verify_snapshot(snapshot, contract._pinned_manifest("kova-nova", family))
    receipt = json.loads((candidate / "receipt.json").read_text())
    dataset_sha = contract.load_json(
        ROOT / "config/kova-three-family-dataset.v2.json")["dataset_sha256"]
    if (receipt.get("kind") != "kova_nova_cpu_bf16_short_lora_experiment" or
            receipt.get("status") != "complete" or
            receipt.get("source_commit") != TRAINING_COMMIT or
            receipt.get("optimizer_steps") != 7 or
            receipt.get("train_records_available") != 27 or
            receipt.get("maximum_training_examples_seen") != 21 or
            receipt.get("heldout_validation_records") != 15 or
            receipt.get("dataset_sha256") != dataset_sha or
            receipt.get("base_revision") != family["immutable_revision"]):
        raise ValueError("candidate is not the completed bounded Nova experiment")
    adapter = candidate / "adapter"
    hashes = receipt["adapter_sha256"]
    if (set(hashes) != {"adapter_config.json", "adapter_model.safetensors"} or
            any(not (adapter / name).is_file() or digest(adapter / name) != sha
                for name, sha in hashes.items())):
        raise ValueError("adapter hash mismatch")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                               trust_remote_code=False)
    train, validation = approved.prepared_rows()
    for row in train:
        completion_tokens(tokenizer, row, 768)
    cases = [completion_tokens(tokenizer, row, 768) for row in validation]
    base = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa")
    base.eval()

    def score(model) -> tuple[float, int, list[float]]:
        total, count, each = 0.0, 0, []
        with torch.inference_mode():
            for ids, labels in cases:
                source = torch.tensor([ids], dtype=torch.long)
                target = torch.tensor([labels], dtype=torch.long)
                loss = float(model(input_ids=source, labels=target,
                                   use_cache=False).loss)
                n = sum(value != -100 for value in labels[1:])
                if not (0 < loss < 100 and n > 0):
                    raise ValueError("invalid held-out completion loss")
                total += loss * n
                count += n
                each.append(round(loss, 8))
        return total / count, count, each

    started = time.monotonic()
    base_loss, base_tokens, base_cases = score(base)
    adapted = PeftModel.from_pretrained(base, str(adapter), is_trainable=False,
                                        local_files_only=True)
    adapted.eval()
    adapter_loss, adapter_tokens, adapter_cases = score(adapted)
    if base_tokens != adapter_tokens:
        raise ValueError("held-out token mismatch")
    report = {
        "kind": "kova_nova_bounded_cpu_bf16_heldout_loss",
        "training_commit": TRAINING_COMMIT,
        "training_run_id": TRAINING_RUN,
        "base_revision": family["immutable_revision"],
        "dataset_sha256": dataset_sha,
        "adapter_sha256": hashes,
        "validation_records": len(validation),
        "completion_tokens": base_tokens,
        "base_mean_completion_loss": round(base_loss, 8),
        "adapter_mean_completion_loss": round(adapter_loss, 8),
        "adapter_minus_base_loss": round(adapter_loss - base_loss, 8),
        "base_case_losses": base_cases,
        "adapter_case_losses": adapter_cases,
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
    args = parser.parse_args()
    print(json.dumps(evaluate(args.snapshot, args.candidate,
                              args.output), sort_keys=True))


if __name__ == "__main__":
    main()
