"""Prepare the approved 42-record Cosmo QLoRA job without loading model weights.

The currently checked-in paid gates remain false. The actual trainer is kept
behind those gates and an independent signed account admission. Its safety
cannot be established by the guest process alone; an external watchdog and
authority must be reviewed and tested before any paid run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import time

from training import cosmo_qlora_launch as launch
from training import cosmo_qlora_grant as grant
from training import cosmo_adapter_preservation as preservation
from training import three_family_contract as contract
from training.snapshot_verifier import verify_snapshot

ROOT = Path(__file__).resolve().parents[1]


class TrainingRejected(ValueError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise TrainingRejected(message)


def require_before_cleanup(admission: dict, *, now: datetime | None = None) -> datetime:
    """Keep grants and model work out of the watchdog's deletion window."""
    trigger = launch.authority.timestamp(admission["watchdog_cleanup_trigger_utc"])
    current = now or datetime.now(timezone.utc)
    need(current.tzinfo is not None and current.astimezone(timezone.utc) < trigger,
         "watchdog cleanup has started")
    return trigger


def prepared_rows() -> tuple[list[dict], list[dict]]:
    """Require exact approved bytes, then build prompt/completion conversations."""
    contract.validate_dataset()
    dataset = contract.load_json(ROOT / "config/kova-three-family-dataset.v2.json")
    system = (ROOT / dataset["prompt_path"]).read_text(encoding="utf-8")
    rows = (json.loads(line) for line in (ROOT / dataset["dataset_path"]).read_text(
        encoding="utf-8").splitlines())
    train, validation = [], []
    for row in rows:
        prompt = [{"role": "system", "content": system}, row["messages"][0]]
        if row.get("trusted_runtime"):
            need(row["split"] == "validation", "training includes a provenance fixture")
            prompt[0] = {"role": "system", "content": system + "\nTrusted test runtime: " +
                         json.dumps(row["trusted_runtime"], sort_keys=True)}
        prepared = {"prompt": prompt, "completion": [row["messages"][1]]}
        (train if row["split"] == "train" else validation).append(prepared)
    need(len(train) == 27 and len(validation) == 15,
         "42-record approved split mismatch")
    return train, validation


def validate_token_masks(tokenizer, rows: list[dict], max_length: int) -> None:
    """Reject token truncation and any training row without completion loss."""
    need(len(rows) == 42 and max_length == 1024, "invalid token check scope")
    from training.cosmo_runtime_probe import completion_tokens
    for row in rows:
        completion_tokens(tokenizer, row, max_length)


def verify_installed_stack() -> None:
    """Check the actual Python environment before opening model files."""
    stack = contract.load_json(ROOT / "config/kova-three-family-training-stack.v1.json")
    need(f"{sys.version_info.major}.{sys.version_info.minor}" == stack["python"],
         "Python version differs from the pinned stack")
    for name, expected in {**stack["packages"], "datasets": "5.0.1",
                           "cryptography": "50.0.1"}.items():
        try:
            actual = version(name)
        except PackageNotFoundError:
            raise TrainingRejected("missing QLoRA dependency: " + name) from None
        need(actual == expected, "QLoRA dependency drift: " + name)


def verify_four_bit_runtime(torch, bnb) -> None:
    """Exercise a tiny NF4 CUDA forward pass without opening model weights."""
    try:
        need(torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (7, 5),
             "T4 CUDA capability mismatch")
        layer = bnb.nn.Linear4bit(4, 4, bias=False, compute_dtype=torch.float16,
                                  compress_statistics=True, quant_type="nf4").to("cuda:0")
        output = layer(torch.ones((1, 4), device="cuda:0", dtype=torch.float16))
        need(output.shape == (1, 4) and bool(torch.isfinite(output).all().item()),
             "four-bit CUDA smoke test failed")
    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
        raise TrainingRejected("four-bit CUDA smoke test failed") from exc


def external_paths(snapshot: Path, output: Path) -> Path:
    need(snapshot.is_absolute() and output.is_absolute() and
         not output.exists() and not output.is_symlink(),
         "snapshot or new external output directory invalid")
    repository = ROOT.resolve(strict=True)
    model_root = snapshot.resolve(strict=True)
    parent = output.parent.resolve(strict=True)
    need(repository not in model_root.parents and repository != model_root and
         repository not in parent.parents and repository != parent and
         model_root != parent and model_root not in parent.parents and
         parent not in model_root.parents and parent != model_root,
         "training files must be outside source and verified snapshot trees")
    return model_root


def source_plan() -> dict:
    proposal = launch.proposal()
    train, validation = prepared_rows()
    recipe = contract.load_json(ROOT / "config/kova-cosmo-qlora.v1.json")
    need(recipe["method"] == "four_bit_qlora_lora_sft" and
         recipe["quantization"] == {"bits": 4, "type": "nf4",
                                    "double_quant": True, "compute_dtype": "float16"} and
         recipe["training"]["completion_only_masking"] is True and
         recipe["retry"] == {"automatic": False, "maximum_attempts": 1},
         "QLoRA method drifted")
    return {
        "status": "cosmo_42_record_qlora_prepared_execution_blocked",
        "family": "kova-cosmo", "train_records": len(train),
        "validation_records": len(validation),
        "dataset_sha256": proposal["dataset_sha256"],
        "model_revision": proposal["model_revision"],
        "maximum_optimizer_steps": recipe["training"]["maximum_optimizer_steps"],
        "maximum_training_seconds": recipe["training"]["maximum_elapsed_seconds"],
        "paid_actions_enabled": False,
    }


def execute(*, snapshot: Path, output: Path, quote: Path, subscription_id: str,
            runtime_evidence: Path | None = None) -> dict:
    """Future paid path. Admission is deliberately unreachable on this head."""
    source_plan()
    pilot = contract.load_json(ROOT / "config/kova-three-family-pilot.v1.json")
    dataset = contract.load_json(ROOT / "config/kova-three-family-dataset.v2.json")
    recipe = contract.load_json(ROOT / "config/kova-cosmo-qlora.v1.json")
    cost = contract.load_json(ROOT / "config/kova-three-family-cost-guard.v1.json")
    need(pilot["resource_creation_authorized"] is True and
         pilot["spending_authorized"] is True and
         pilot["model_download_authorized"] is True and
         pilot["training_authorized"] is True and
         dataset["training_authorized"] is True and
         cost["spending_authorized"] is True,
         "owner release and source authorization absent")
    need(recipe["training_authorized"] is True and
         os.environ.get("KOVA_CONFIRM_PAID_TRAINING") == "YES",
         "training authorization absent")
    source_commit = launch.clean_source_commit()
    admission = launch.assess_signed_quote(
        quote, source_commit=source_commit, subscription_id=subscription_id)
    # Source review must replace this hold only after the independent authority
    # and control-plane watchdog are live and verified in the chosen account.
    need(admission["paid_actions_enabled"] is True,
         "independent paid controller is not released")
    require_before_cleanup(admission)
    model_root = external_paths(snapshot, output)
    verify_installed_stack()
    lineage = contract.load_json(ROOT / "config/kova-private-lineage.v1.json")
    manifest = contract._pinned_manifest("kova-cosmo", lineage["families"]["kova-cosmo"])
    verify_snapshot(model_root, manifest, require_protected=True)
    os.environ.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "WANDB_DISABLED": "true"})
    import torch
    import bitsandbytes as bnb
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
    from training.cosmo_hardware import verify_nvidia_t4

    verify_nvidia_t4(torch)
    verify_four_bit_runtime(torch, bnb)
    need(runtime_evidence is not None, "signed VM runtime preflight absent")
    require_before_cleanup(admission)
    preflight = grant.read_runtime_preflight(
        runtime_evidence, quote_sha256=admission["quote_sha256"],
        source_commit=source_commit, subscription_id=subscription_id,
        deadline_utc=admission["allocation_deadline_utc"],
        cleanup_trigger_utc=admission["watchdog_cleanup_trigger_utc"])
    require_before_cleanup(admission)
    committed = grant.acquire_training_grant(
        quote=quote, source_commit=source_commit, subscription_id=subscription_id,
        lifecycle_id=preflight["lifecycle_id"],
        preflight_ledger_sequence=preflight["preflight_ledger_sequence"],
        azure_instance=preflight["azure_instance"], runtime_evidence=runtime_evidence)
    need(committed["training_runs_consumed"] == 1 and
         committed["allocation_deadline_utc"] == admission["allocation_deadline_utc"] and
         committed["watchdog_cleanup_trigger_utc"] ==
         admission["watchdog_cleanup_trigger_utc"],
         "single-use grant did not commit")
    require_before_cleanup(admission)
    tokenizer = AutoTokenizer.from_pretrained(str(model_root), local_files_only=True,
                                               trust_remote_code=False)
    train, validation = prepared_rows()
    validate_token_masks(tokenizer, train + validation, 1024)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_root), local_files_only=True, trust_remote_code=False,
        quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model)
    need(all(any(name.endswith("." + target) for name, _ in model.named_modules())
             for target in recipe["lora"]["target_modules"]),
         "base model LoRA target missing")
    lora = recipe["lora"]
    peft_config = LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"],
                             lora_dropout=lora["dropout"], bias=lora["bias"],
                             target_modules=lora["target_modules"],
                             task_type="CAUSAL_LM")
    training = recipe["training"]
    args = SFTConfig(
        output_dir=str(output / "checkpoints"),
        max_steps=training["maximum_optimizer_steps"], num_train_epochs=1,
        per_device_train_batch_size=training["per_device_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"], max_length=1024,
        completion_only_loss=True, packing=False, fp16=True, bf16=False,
        save_safetensors=True, save_strategy="steps", save_steps=7,
        report_to="none", push_to_hub=False, seed=42,
    )
    # The independent watchdog starts deletion before the priced deadline even
    # if the guest freezes. This callback stops between optimizer steps.
    from transformers import TrainerCallback
    cleanup_trigger = require_before_cleanup(admission)
    start = time.monotonic()

    class DeadlineCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if (time.monotonic() - start >= training["maximum_elapsed_seconds"] or
                    datetime.now(timezone.utc) >= cleanup_trigger):
                control.should_training_stop = True
            return control

    output.mkdir(mode=0o700)
    trainer = SFTTrainer(model=model, args=args, peft_config=peft_config,
                         processing_class=tokenizer,
                         train_dataset=Dataset.from_list(train),
                         eval_dataset=Dataset.from_list(validation),
                         callbacks=[DeadlineCallback()])
    require_before_cleanup(admission)
    result = trainer.train()
    need(result.global_step == 7 and time.monotonic() - start <
         training["maximum_elapsed_seconds"] and
         datetime.now(timezone.utc) < cleanup_trigger,
         "training did not complete inside the approved limit")
    require_before_cleanup(admission)
    trainer.model.save_pretrained(output / "adapter", safe_serialization=True)
    require_before_cleanup(admission)
    preserved = preservation.preserve_adapter(adapter=output / "adapter",
        grant_payload=committed, root=ROOT)
    return {"status": "candidate_preserved_not_released", "source_commit": source_commit,
            "optimizer_steps": result.global_step, "artifact_sha256": preserved["artifact_sha256"],
            "destination_uri": preserved["destination_uri"],
            "ledger_sequence": preserved["ledger_sequence"]}


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quote", type=Path)
    parser.add_argument("--runtime-evidence", type=Path)
    parser.add_argument("--subscription-id")
    args = parser.parse_args(argv)
    if args.execute:
        if not all((args.snapshot, args.output, args.quote, args.runtime_evidence,
                    args.subscription_id)):
            parser.error("the paid path requires all five explicit bindings")
        print(json.dumps(execute(snapshot=args.snapshot, output=args.output,
                                 quote=args.quote, subscription_id=args.subscription_id,
                                 runtime_evidence=args.runtime_evidence),
                         sort_keys=True))
    else:
        print(json.dumps(source_plan(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
