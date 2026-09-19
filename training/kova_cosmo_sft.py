"""Kova Cosmo LoRA SFT recipe with source-only dry-run by default.

Execution is intentionally impossible while the checked-in authorization gates
remain false. This file does not provision Azure or alter quota/billing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from importlib.metadata import PackageNotFoundError, version

from release.model_revisions import MODEL_SOURCE_REFERENCES
from training.identity_pilot import load as load_identity_pilot
from training.identity_pilot import format_messages

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/kova-cosmo-sft.v1.json"

EXPECTED_SOFTWARE = {
    "python": "3.12",
    "torch": "2.8.0",
    "transformers": "5.17.0",
    "peft": "0.21.0",
    "trl": "1.13.0",
    "accelerate": "1.15.0",
    "datasets": "5.0.1",
}
EXPECTED_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class RecipeError(ValueError):
    pass


def need(value: bool) -> None:
    if not value:
        raise RecipeError("kova cosmo sft recipe rejected")


def load_recipe(root: Path = ROOT) -> dict:
    try:
        config_path = root / "config/kova-cosmo-sft.v1.json"
        value = json.loads(config_path.read_text(encoding="utf-8"))
        ref = MODEL_SOURCE_REFERENCES["work-cosmo"]
        need(value["schema_version"] == 1)
        need(value["status"] == "owner_requested_recipe_not_executed")
        need(value["display_name"] == "Kova Cosmo")
        need(value["model_slot"] == ref.slot)
        need(value["base_model"] == ref.model)
        need(value["base_revision"] == ref.revision)
        need(value["dataset_path"] == "data/kova-identity-pilot.v1.jsonl")
        need(value["method"] == "lora_sft")

        hardware = value["hardware"]
        need(hardware == {
            "region": "eastus",
            "vm_size": "Standard_NC4as_T4_v3",
            "gpu_family": "NVIDIA T4",
            "precision": "fp16",
            "bf16": False,
            "quantized_base": False,
        })
        need(value["software"] == EXPECTED_SOFTWARE)

        lora = value["lora"]
        need(lora["r"] == 16 and lora["alpha"] == 32)
        need(lora["dropout"] == 0.05 and lora["bias"] == "none")
        need(lora["target_modules"] == EXPECTED_TARGETS)

        training = value["training"]
        need(training == {
            "learning_rate": 0.0001,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "max_length": 1024,
            "optimizer": "adamw_torch",
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": False,
            "completion_only_loss": True,
            "packing": False,
            "seed": 42,
            "report_to": "none",
            "push_to_hub": False,
        })
        need(value["account_gates"] == {
            "microsoft_quota_provider_registration_authorized": True,
            "eastus_ncast4_quota_verified": False,
            "runtime_compatibility_verified": False,
            "approved_budget_usd": 2.0,
        })
        need(value["execution"] == {
            "model_download_authorized": True,
            "training_authorized": True,
            "deployment_authorized": False,
        })
        need(value["output"] == {
            "directory": "artifacts/kova-cosmo-sft",
            "adapter_sha256": None,
        })
        need(value["phase_b_ready"] is False)
        load_identity_pilot(root)
        return value
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise RecipeError("kova cosmo sft recipe rejected") from None


def dry_run() -> dict:
    value = load_recipe()
    return {
        "status": "validated_not_executed",
        "display_name": value["display_name"],
        "model_slot": value["model_slot"],
        "base_model": value["base_model"],
        "base_revision": value["base_revision"],
        "region": value["hardware"]["region"],
        "vm_size": value["hardware"]["vm_size"],
        "precision": value["hardware"]["precision"],
        "method": value["method"],
        "software": value["software"],
        "lora": value["lora"],
        "training": value["training"],
        "quota_verified": value["account_gates"]["eastus_ncast4_quota_verified"],
        "runtime_compatibility_verified": value["account_gates"]["runtime_compatibility_verified"],
        "approved_budget_usd": value["account_gates"]["approved_budget_usd"],
        "model_weights_downloaded": False,
        "training_started": False,
        "runtime_integrated": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    }


def prepare_sft_rows() -> tuple[list[dict], list[dict]]:
    """Build prompt/completion rows offline from the validated pilot corpus."""
    _, prompt, rows = load_identity_pilot()
    train_rows, eval_rows = [], []
    for row in rows:
        messages = format_messages(prompt, row)
        target = train_rows if row["split"] == "train" else eval_rows
        target.append({"prompt": messages[:-1], "completion": messages[-1:]})
    return train_rows, eval_rows


def verify_installed_software() -> None:
    """Reject missing or drifted recipe dependencies without importing them."""
    for package, expected in EXPECTED_SOFTWARE.items():
        if package == "python":
            need(f"{sys.version_info.major}.{sys.version_info.minor}" == expected)
            continue
        try:
            installed = version(package)
        except PackageNotFoundError:
            raise RecipeError("kova cosmo sft recipe rejected") from None
        need(installed == expected)


def execute() -> None:
    value = load_recipe()
    # Source control plus an operator acknowledgement are both required. The
    # checked-in config currently makes this branch unreachable.
    need(value["account_gates"]["eastus_ncast4_quota_verified"] is True)
    need(value["account_gates"]["runtime_compatibility_verified"] is True)
    need(type(value["account_gates"]["approved_budget_usd"]) in (int, float))
    need(value["account_gates"]["approved_budget_usd"] > 0)
    need(value["execution"]["model_download_authorized"] is True)
    need(value["execution"]["training_authorized"] is True)
    need(value["execution"]["deployment_authorized"] is False)
    need(os.environ.get("KOVA_CONFIRM_PAID_TRAINING") == "YES")
    verify_installed_software()

    # Heavy dependencies are imported only after every account/source guard.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    need(torch.cuda.is_available())
    need(torch.cuda.get_device_capability(0)[0:2] == (7, 5))

    train_rows, eval_rows = prepare_sft_rows()

    lora = value["lora"]
    training = value["training"]
    args = SFTConfig(
        output_dir=value["output"]["directory"],
        model_init_kwargs={
            "revision": value["base_revision"],
            "dtype": torch.float16,
            "trust_remote_code": False,
        },
        fp16=True,
        bf16=False,
        learning_rate=training["learning_rate"],
        num_train_epochs=training["num_train_epochs"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        max_length=training["max_length"],
        optim=training["optimizer"],
        lr_scheduler_type=training["lr_scheduler_type"],
        warmup_ratio=training["warmup_ratio"],
        weight_decay=training["weight_decay"],
        max_grad_norm=training["max_grad_norm"],
        gradient_checkpointing=training["gradient_checkpointing"],
        completion_only_loss=training["completion_only_loss"],
        packing=training["packing"],
        seed=training["seed"],
        report_to=training["report_to"],
        push_to_hub=training["push_to_hub"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
    )
    peft = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        target_modules=lora["target_modules"],
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=value["base_model"],
        args=args,
        peft_config=peft,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows),
    )
    trainer.train()
    trainer.save_model(value["output"]["directory"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute:
            execute()
            return 0
        print(json.dumps(dry_run(), sort_keys=True))
        return 0
    except RecipeError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
