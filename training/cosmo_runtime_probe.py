"""Guarded one-batch GPU compatibility probe for the selected Cosmo checkpoint.

The checked-in source remains non-executable. A future reviewed source change,
fresh external runtime evidence, and a distinct operator acknowledgement are all
required before this module can download or load the pinned model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from training.cosmo_artifacts import (
    ArtifactError,
    EXPECTED_SHA256,
    REQUIRED_ASSETS,
    verify_snapshot,
)
from training.cosmo_hardware import HardwareError, verify_nvidia_t4
from training.cosmo_lifecycle_authority import acquire_phase_grant
from training.cosmo_runtime_guard import RuntimeGuardError, require_ready
from training.kova_cosmo_sft import (
    EXPECTED_TARGETS,
    load_recipe,
    prepare_sft_rows,
    verify_installed_software,
    verify_source_checkout,
)

CONFIRMATION_ENV = "KOVA_CONFIRM_PAID_RUNTIME_PROBE"


class RuntimeProbeError(ValueError):
    pass


def need(condition: bool) -> None:
    if not condition:
        raise RuntimeProbeError("kova cosmo runtime probe rejected")


def target_inventory(model) -> dict[str, int]:
    counts = {target: 0 for target in EXPECTED_TARGETS}
    for name, _ in model.named_modules():
        for target in EXPECTED_TARGETS:
            if name.endswith("." + target):
                counts[target] += 1
    need(all(count > 0 for count in counts.values()))
    need(len(set(counts.values())) == 1)
    return counts


def completion_tokens(tokenizer, row: dict, max_length: int) -> tuple[list[int], list[int]]:
    prompt = tokenizer.apply_chat_template(
        row["prompt"], tokenize=True, add_generation_prompt=True,
        return_tensors=None,
    )
    full = tokenizer.apply_chat_template(
        row["prompt"] + row["completion"], tokenize=True,
        add_generation_prompt=False, return_tensors=None,
    )
    need(type(prompt) is list and type(full) is list)
    need(0 < len(prompt) < len(full) <= max_length)
    need(full[:len(prompt)] == prompt)
    labels = [-100] * len(prompt) + full[len(prompt):]
    need(any(label != -100 for label in labels))
    return full, labels


def authorize_probe() -> tuple[dict, dict, dict]:
    value = load_recipe()
    gates = value["account_gates"]
    permissions = value["execution"]
    need(gates["microsoft_quota_provider_registration_authorized"] is True)
    need(gates["eastus_ncast4_quota_verified"] is True)
    need(gates["runtime_compatibility_verified"] is False)
    need(type(gates["approved_budget_usd"]) in (int, float))
    need(gates["approved_budget_usd"] == 2.0)
    need(permissions["model_download_authorized"] is True)
    need(permissions["training_authorized"] is True)
    need(permissions["deployment_authorized"] is False)
    need(os.environ.get(CONFIRMATION_ENV) == "YES")
    source_commit = os.environ.get("KOVA_SOURCE_COMMIT")
    need(type(source_commit) is str)
    verify_source_checkout(source_commit)
    runtime = require_ready()
    grant = acquire_phase_grant(
        phase="runtime_probe",
        source_commit=source_commit,
        runtime_evidence_sha256=runtime["runtime_evidence_sha256"],
        lifecycle_id=runtime["lifecycle_id"],
        preflight_ledger_sequence=runtime["preflight_ledger_sequence"],
        runtime_deadline_utc=runtime["deadline_utc"],
        context={
            "operation": "one_batch_compatibility_probe",
            "base_model": value["base_model"],
            "base_revision": value["base_revision"],
            "region": value["hardware"]["region"],
            "vm_size": value["hardware"]["vm_size"],
        },
    )
    verify_installed_software()
    return value, runtime, grant


def execute_probe() -> dict:
    value, runtime, grant = authorize_probe()

    # Heavy imports and all model/network access occur only after every source,
    # account, runtime, budget and operator guard above has passed.
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        device_identity = verify_nvidia_t4(torch)
    except HardwareError:
        raise RuntimeProbeError("kova cosmo runtime probe rejected") from None
    torch.manual_seed(value["training"]["seed"])
    torch.cuda.manual_seed_all(value["training"]["seed"])
    torch.cuda.reset_peak_memory_stats(0)

    snapshot = Path(snapshot_download(
        repo_id=value["base_model"], revision=value["base_revision"],
        allow_patterns=list(REQUIRED_ASSETS),
    ))
    asset_inventory = verify_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False,
    )
    base = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True,
        dtype=torch.float16, device_map={"": 0},
        low_cpu_mem_usage=True, trust_remote_code=False,
    )
    targets = target_inventory(base)
    lora = value["lora"]
    adapter = get_peft_model(base, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"], bias=lora["bias"],
        target_modules=lora["target_modules"], task_type="CAUSAL_LM",
    ))
    trainable = {
        name: parameter for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    }
    need(len(trainable) > 0)
    need(all("lora_" in name for name in trainable))

    train_rows, _ = prepare_sft_rows()
    input_ids, labels = completion_tokens(
        tokenizer, train_rows[0], value["training"]["max_length"]
    )
    device = torch.device("cuda:0")
    inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    attention = torch.ones_like(inputs)
    adapter.train()
    output = adapter(
        input_ids=inputs, attention_mask=attention,
        labels=label_tensor, use_cache=False,
    )
    need(torch.isfinite(output.loss).item() and output.loss.item() > 0)
    output.loss.backward()
    need(all(
        parameter.grad is not None and
        torch.isfinite(parameter.grad).all().item()
        for parameter in trainable.values()
    ))
    need(all(
        parameter.grad is None for parameter in adapter.parameters()
        if not parameter.requires_grad
    ))

    return {
        "status": "selected_checkpoint_gpu_compatibility_verified",
        "base_model": value["base_model"],
        "base_revision": value["base_revision"],
        "device_name": device_identity["device_name"],
        "device_capability": device_identity["device_capability"],
        "gpu_family": device_identity["gpu_family"],
        "precision": value["hardware"]["precision"],
        "asset_inventory": asset_inventory,
        "target_module_counts": targets,
        "trainable_lora_tensors": len(trainable),
        "trainable_lora_parameters": sum(
            parameter.numel() for parameter in trainable.values()
        ),
        "tokens": len(input_ids),
        "completion_tokens": sum(label != -100 for label in labels),
        "finite_loss": True,
        "finite_lora_gradients": True,
        "compatibility_backward_passes": 1,
        "optimizer_steps": 0,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(0),
        "deallocation_deadline_utc": runtime["deadline_utc"],
        "lifecycle_id": grant["lifecycle_id"],
        "lifecycle_phase_grant_sha256": grant["phase_grant_sha256"],
        "lifecycle_ledger_sequence": grant["ledger_sequence"],
        "lifecycle_grant_id": grant["grant_id"],
        "lifecycle_ledger_commit_id": grant["ledger_commit_id"],
        "selected_checkpoint_loaded": True,
        "model_download_observed": None,
        "pilot_epoch_training_started": False,
        "training_weights_changed": False,
        "adapter_saved": False,
        "deployment_authorized": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    }


def dry_run() -> dict:
    value = load_recipe()
    return {
        "status": "blocked",
        "base_model": value["base_model"],
        "base_revision": value["base_revision"],
        "quota_verified": value["account_gates"][
            "eastus_ncast4_quota_verified"
        ],
        "runtime_compatibility_verified": value["account_gates"][
            "runtime_compatibility_verified"
        ],
        "spending_release": False,
        "model_weights_downloaded": False,
        "pilot_training_started": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        report = execute_probe() if arguments.execute else dry_run()
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception:
        # The CLI boundary never exposes provider, filesystem or dependency
        # details that could contain account or cache information.
        print("kova cosmo runtime probe rejected", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
