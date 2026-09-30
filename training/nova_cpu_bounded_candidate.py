"""Train one bounded experimental Nova adapter on a free CPU runner.

This short CPU run consumes at most 21 of the 27 approved training examples.
It does not satisfy the selected three-family Azure QLoRA pilot or release gates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

from training import cosmo_qlora_training as approved
from training import three_family_contract as contract
from training.cosmo_runtime_probe import completion_tokens
from training.snapshot_verifier import verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
MAX_STEPS = 7
GRADIENT_ACCUMULATION = 3
MAX_TRAINING_SECONDS = 140 * 60


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def execute(snapshot: Path, output: Path) -> dict:
    if (sys.version_info[:2] != (3, 12) or not snapshot.is_dir() or
            output.exists() or output.is_symlink() or not output.parent.is_dir()):
        raise ValueError("Python 3.12, verified snapshot, new output required")
    pinned = {"transformers": "5.17.0", "trl": "1.13.0", "peft": "0.21.0",
              "datasets": "5.0.1", "tokenizers": "0.23.2"}
    if any(version(name) != expected for name, expected in pinned.items()):
        raise ValueError("CPU experiment dependency drift")
    import torch
    if torch.__version__ != "2.8.0+cpu" or torch.version.cuda is not None:
        raise ValueError("pinned CPU PyTorch required")
    torch.set_num_threads(min(os.cpu_count() or 1, 4))
    lineage = contract.load_json(ROOT / "config/kova-private-lineage.v1.json")
    family = lineage["families"]["kova-nova"]
    verify_snapshot(snapshot, contract._pinned_manifest("kova-nova", family))
    recipe = contract.load_json(ROOT / "config/kova-nova-qlora.v1.json")
    if (recipe["family"] != "kova-nova" or recipe["dataset"] !=
            "config/kova-three-family-dataset.v2.json" or
            recipe["training"]["maximum_optimizer_steps"] != MAX_STEPS or
            recipe["training"]["maximum_sequence_length"] != 768 or
            recipe["lora"]["rank"] != 8):
        raise ValueError("unexpected Nova data or LoRA recipe")
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                               trust_remote_code=False)
    train, validation = approved.prepared_rows()
    for row in train + validation:
        completion_tokens(tokenizer, row, 768)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa")
    lora = recipe["lora"]
    if not all(any(name.endswith("." + target) for name, _ in model.named_modules())
               for target in lora["target_modules"]):
        raise ValueError("Nova LoRA targets absent")
    output.mkdir(mode=0o700)
    started = time.monotonic()

    class DeadlineStop(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if time.monotonic() - started >= MAX_TRAINING_SECONDS:
                control.should_training_stop = True
            return control

    config = SFTConfig(
        output_dir=str(output / "checkpoints"), max_steps=MAX_STEPS,
        num_train_epochs=1, per_device_train_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=recipe["training"]["learning_rate"], max_length=768,
        completion_only_loss=True, packing=False, fp16=False, bf16=False,
        optim="adamw_torch", gradient_checkpointing=True,
        save_strategy="no", report_to="none", push_to_hub=False, seed=42)
    trainer = SFTTrainer(
        model=model, args=config,
        peft_config=LoraConfig(
            r=lora["rank"], lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"], bias=lora["bias"],
            target_modules=lora["target_modules"], task_type="CAUSAL_LM"),
        processing_class=tokenizer, train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(validation), callbacks=[DeadlineStop()])
    result = trainer.train()
    if not 1 <= result.global_step <= MAX_STEPS:
        raise ValueError("no bounded Nova optimizer step completed")
    adapter = output / "adapter"
    trainer.model.save_pretrained(adapter, safe_serialization=True)
    hashes = {name: digest(adapter / name)
              for name in ("adapter_config.json", "adapter_model.safetensors")}
    receipt = {
        "kind": "kova_nova_cpu_bf16_short_lora_experiment",
        "status": "complete" if result.global_step == MAX_STEPS else "partial",
        "source_commit": os.environ.get("GITHUB_SHA", "local"),
        "base_revision": family["immutable_revision"],
        "dataset_sha256": contract.load_json(
            ROOT / "config/kova-three-family-dataset.v2.json")["dataset_sha256"],
        "train_records_available": len(train),
        "maximum_training_examples_seen": result.global_step * GRADIENT_ACCUMULATION,
        "heldout_validation_records": len(validation),
        "optimizer_steps": result.global_step,
        "training_seconds": round(time.monotonic() - started, 2),
        "adapter_sha256": hashes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "azure_qlora_release_approved": False,
        "production_routing_approved": False,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True,
                                                     indent=2) + "\n")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "short_cpu_nova_experiment_only",
                          "maximum_steps": MAX_STEPS,
                          "maximum_training_examples_seen": 21,
                          "azure_qlora_release_approved": False}))
        return 0
    if not args.snapshot or not args.output:
        parser.error("snapshot and new output required")
    print(json.dumps(execute(args.snapshot, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
