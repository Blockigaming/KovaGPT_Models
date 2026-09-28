"""Bounded experimental Nova NF4 LoRA job on a private Azure T4 VM.

This is not the selected signed-controller pilot or a production release.
The separate watchdog must already exist and own the VM group deadline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request
import zipfile

from training import cosmo_qlora_training as approved
from training import three_family_contract as contract
from training.cosmo_runtime_probe import completion_tokens
from training.snapshot_verifier import verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "kova-nova"
FAMILY_LIMITS = {
    "kova-cosmo": (1024, 4, 1800),
    "kova-orion": (1024, 4, 2700),
    "kova-nova": (768, 8, 4500),
}
ACCOUNT = "kova42c1a27"
CONTAINER = "cosmo-adapters"
PRESERVATION_LEAD = timedelta(minutes=12)


def deadline(value: str) -> datetime:
    end = datetime.fromisoformat(value.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    if (end.tzinfo is None or end <= now + timedelta(minutes=25) or
            end > now + timedelta(hours=3)):
        raise ValueError("watchdog deadline must be 25 minutes to 3 hours away")
    return end


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def put_with_vm_identity(archive: Path, archive_sha256: str, blob: str) -> str:
    # Azure IMDS is link-local. The bearer token is never logged or stored.
    metadata_url = ("http://169.254.169.254/metadata/identity/oauth2/token"
                    "?api-version=2018-02-01&resource=https%3A%2F%2Fstorage.azure.com%2F")
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    token_request = urllib.request.Request(metadata_url, headers={"Metadata": "true"})
    with direct.open(token_request, timeout=15) as response:
        token = json.load(response)["access_token"]
    url = (f"https://{ACCOUNT}.blob.core.windows.net/{CONTAINER}/"
           + urllib.parse.quote(blob, safe="/"))
    payload = archive.read_bytes()
    if not payload or len(payload) > 128 * 1024 * 1024:
        raise ValueError("adapter archive exceeds bounded Blob PUT")
    request = urllib.request.Request(url, data=payload, method="PUT", headers={
        "Authorization": "Bearer " + token,
        "x-ms-version": "2023-11-03",
        "x-ms-blob-type": "BlockBlob",
        "x-ms-blob-content-type": "application/zip",
        "x-ms-meta-sha256": archive_sha256,
        "If-None-Match": "*",
        "Content-Type": "application/zip",
    })
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 201:
            raise ValueError("Azure Blob did not create the experimental copy")
        return response.headers["ETag"]


def execute(snapshot: Path, output: Path, deadline_utc: str,
            source_commit: str, family: str = FAMILY) -> dict:
    end = deadline(deadline_utc)
    if (family not in FAMILY_LIMITS or sys.version_info[:2] != (3, 12) or
            not snapshot.is_dir() or
            output.exists() or output.is_symlink() or not output.parent.is_dir() or
            len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit)):
        raise ValueError("pinned Python, snapshot, new output and source SHA required")
    approved.verify_installed_stack()
    if version("bitsandbytes") != "0.48.2":
        raise ValueError("four-bit dependency drift")
    import torch
    import bitsandbytes as bnb
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, TrainerCallback)
    from trl import SFTConfig, SFTTrainer
    from training.cosmo_hardware import verify_nvidia_t4

    hardware = verify_nvidia_t4(torch)
    approved.verify_four_bit_runtime(torch, bnb)
    lineage = contract.load_json(ROOT / "config/kova-private-lineage.v1.json")
    model_family = lineage["families"][family]
    verify_snapshot(snapshot, contract._pinned_manifest(family, model_family))
    recipe = contract.load_json(ROOT / f"config/{family}-qlora.v1.json")
    if (recipe["family"] != family or recipe["dataset"] !=
            "config/kova-three-family-dataset.v2.json" or
            recipe["method"] != "four_bit_qlora_lora_sft" or
            recipe["quantization"] != {"bits": 4, "type": "nf4",
                                        "double_quant": True,
                                        "compute_dtype": "float16"}):
        raise ValueError("Nova recipe mismatch")
    train, validation = approved.prepared_rows()
    sequence_length, accumulation, max_training_seconds = FAMILY_LIMITS[family]
    training = recipe["training"]
    if (training["maximum_optimizer_steps"] != 7 or
            training["maximum_sequence_length"] != sequence_length or
            training["gradient_accumulation_steps"] != accumulation or
            training["maximum_elapsed_seconds"] != max_training_seconds):
        raise ValueError("bounded training config mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                               trust_remote_code=False)
    for row in train + validation:
        completion_tokens(tokenizer, row, sequence_length)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    quant = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    # SFTTrainer prepares the quantized model when peft_config is supplied.
    lora = recipe["lora"]
    if not all(any(name.endswith("." + target) for name, _ in model.named_modules())
               for target in lora["target_modules"]):
        raise ValueError("Nova LoRA target missing")
    if datetime.now(timezone.utc) + timedelta(minutes=25) >= end:
        raise ValueError("insufficient watchdog time after model loading")
    output.mkdir(mode=0o700)
    started = time.monotonic()

    class DeadlineStop(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if (time.monotonic() - started >= min(2700, max_training_seconds) or
                    datetime.now(timezone.utc) + PRESERVATION_LEAD >= end):
                control.should_training_stop = True
            return control

    settings = SFTConfig(
        output_dir=str(output / "checkpoints"), max_steps=7, num_train_epochs=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=accumulation,
        learning_rate=training["learning_rate"], max_length=sequence_length,
        # NF4 matmuls still compute in float16. Keep adapter optimization
        # outside AMP: this T4 stack exposes BF16 gradients that GradScaler
        # cannot unscale on CUDA capability 7.5.
        completion_only_loss=True, packing=False, fp16=False, bf16=False,
        gradient_checkpointing=True, optim="adamw_torch",
        save_strategy="no",
        report_to="none", push_to_hub=False, seed=42)
    trainer = SFTTrainer(
        model=model, args=settings,
        peft_config=LoraConfig(
            r=lora["rank"], lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"], bias=lora["bias"],
            target_modules=lora["target_modules"], task_type="CAUSAL_LM"),
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(validation), callbacks=[DeadlineStop()])
    result = trainer.train()
    if result.global_step < 1 or result.global_step > 7:
        raise ValueError("no bounded optimizer step completed")
    if datetime.now(timezone.utc) + timedelta(minutes=5) >= end:
        raise ValueError("too close to watchdog cleanup to preserve adapter")
    adapter = output / "adapter"
    trainer.model.save_pretrained(adapter, safe_serialization=True)
    hashes = {name: digest(adapter / name) for name in
              ("adapter_config.json", "adapter_model.safetensors")}
    receipt = {
        "kind": f"{family.replace('-', '_')}_gpu_t4_nf4_lora_experiment",
        "family": family,
        "status": "complete" if result.global_step == 7 else "partial",
        "source_commit": source_commit,
        "base_revision": model_family["immutable_revision"],
        "dataset_sha256": contract.load_json(
            ROOT / "config/kova-three-family-dataset.v2.json")["dataset_sha256"],
        "train_records": len(train), "validation_records": len(validation),
        "optimizer_steps": result.global_step,
        "training_seconds": round(time.monotonic() - started, 2),
        "adapter_sha256": hashes, "hardware": hardware,
        "watchdog_deadline_utc": end.isoformat(),
        "signed_controller_pilot": False,
        "production_routing_approved": False,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                     sort_keys=True) + "\n")
    archive = output / "adapter.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        for name in ("adapter_config.json", "adapter_model.safetensors",
                     "README.md"):
            z.write(adapter / name, "adapter/" + name)
        z.write(output / "receipt.json", "receipt.json")
    archive_sha256 = digest(archive)
    receipt["archive_sha256"] = archive_sha256
    blob = f"gpu-experimental/2026-09-28/{family}/adapter.zip"
    receipt["azure_blob"] = blob
    receipt["azure_etag"] = put_with_vm_identity(archive, archive_sha256, blob)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--family", choices=sorted(FAMILY_LIMITS), default=FAMILY)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deadline-utc")
    parser.add_argument("--source-commit")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "experimental_gpu_preparation_only",
                          "family": args.family, "watchdog_required": True,
                          "signed_controller_pilot": False}))
        return 0
    if not all((args.snapshot, args.output, args.deadline_utc,
                args.source_commit)):
        parser.error("snapshot, new output, deadline and source commit required")
    print(json.dumps(execute(args.snapshot, args.output, args.deadline_utc,
                             args.source_commit, args.family), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
