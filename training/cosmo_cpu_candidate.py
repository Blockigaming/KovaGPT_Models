"""Train one experimental Cosmo LoRA adapter on a free CPU runner.

This is a bounded, public-dataset experiment. It does not release the selected
Azure T4 QLoRA pilot, count as a paid grant, or authorize production routing.
Only the adapter and its receipt leave the runner; the pinned base stays local.
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
from training.snapshot_verifier import verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
MAX_TRAINING_SECONDS = 3600


def execute(snapshot: Path, output: Path) -> dict:
    plan = approved.source_plan()
    if (sys.version_info[:2] != (3, 12) or not snapshot.is_dir() or
            output.exists() or not output.parent.is_dir() or
            output.is_symlink()):
        raise ValueError("Python 3.12, existing snapshot and new output required")
    expected = {"transformers": "5.17.0", "trl": "1.13.0", "peft": "0.21.0",
                "datasets": "5.0.1", "tokenizers": "0.23.2"}
    if any(version(name) != pinned for name, pinned in expected.items()):
        raise ValueError("CPU training dependency differs from pinned lock")
    import torch
    if torch.__version__ != "2.8.0+cpu" or torch.version.cuda is not None:
        raise ValueError("pinned CPU PyTorch required")
    torch.set_num_threads(min(os.cpu_count() or 1, 4))
    lineage = contract.load_json(ROOT / "config/kova-private-lineage.v1.json")
    manifest = contract._pinned_manifest(
        "kova-cosmo", lineage["families"]["kova-cosmo"])
    verify_snapshot(snapshot, manifest)

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False)
    train, validation = approved.prepared_rows()
    approved.validate_token_masks(tokenizer, train + validation, 1024)
    recipe = contract.load_json(ROOT / "config/kova-cosmo-qlora.v1.json")
    lora = recipe["lora"]
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="sdpa")
    if not all(any(name.endswith("." + target) for name, _ in model.named_modules())
               for target in lora["target_modules"]):
        raise ValueError("pinned LoRA target missing")
    output.mkdir(mode=0o700)
    started = time.monotonic()

    class StopAtDeadline(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if time.monotonic() - started >= MAX_TRAINING_SECONDS:
                control.should_training_stop = True
            return control

    settings = SFTConfig(
        output_dir=str(output / "checkpoints"),
        max_steps=7, num_train_epochs=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=1e-4, max_length=1024, completion_only_loss=True,
        packing=False, fp16=False, bf16=False, optim="adamw_torch",
        save_strategy="steps", save_steps=1, save_total_limit=1,
        report_to="none", push_to_hub=False, seed=42,
    )
    trainer = SFTTrainer(
        model=model, args=settings,
        peft_config=LoraConfig(
            r=lora["rank"], lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"], bias=lora["bias"],
            target_modules=lora["target_modules"], task_type="CAUSAL_LM"),
        processing_class=tokenizer, train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(validation), callbacks=[StopAtDeadline()],
    )
    result = trainer.train()
    if result.global_step < 1 or result.global_step > 7:
        raise ValueError("no bounded optimizer step completed")
    adapter = output / "adapter"
    trainer.model.save_pretrained(adapter, safe_serialization=True)
    files = {name: hashlib.sha256((adapter / name).read_bytes()).hexdigest()
             for name in ("adapter_config.json", "adapter_model.safetensors")}
    receipt = {
        "kind": "kova_cosmo_cpu_fp32_lora_experiment",
        "status": "complete" if result.global_step == 7 else "partial",
        "source_commit": os.environ.get("GITHUB_SHA", "local"),
        "base_revision": plan["model_revision"],
        "dataset_sha256": plan["dataset_sha256"],
        "train_records": len(train), "validation_records": len(validation),
        "optimizer_steps": result.global_step,
        "training_seconds": round(time.monotonic() - started, 2),
        "adapter_sha256": files,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "azure_qlora_release_approved": False,
        "production_routing_approved": False,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "ready_for_free_cpu_experiment",
                          "family": "kova-cosmo", "maximum_optimizer_steps": 7,
                          "azure_qlora_release_approved": False}))
        return 0
    if not args.snapshot or not args.output:
        parser.error("--snapshot and --output required")
    print(json.dumps(execute(args.snapshot, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
