"""Exercise real TRL preprocessing/collation offline, without constructing a model.

Requires the separate Python 3.12/Linux CPU probe lock and two local, hash-verified
tokenizer assets. No weights, model forward pass, optimizer or training is used.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
from unittest.mock import patch

from training.identity_pilot import load as load_pilot
from training.kova_cosmo_sft import load_recipe, prepare_sft_rows
from training.probe_cosmo_tokenizer import ASSETS, REVISION

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements/cosmo-loss-mask-py312-linux-cpu.lock"


class ProbeError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def verify_environment() -> dict:
    require(sys.version_info[:2] == (3, 12), "probe requires Python 3.12")
    expected = dict(re.findall(r"^([\w.-]+)==([^\s\\]+)", LOCK.read_text(), re.M))
    expected["torch"] = "2.8.0+cpu"
    require(len(expected) == 57, "unexpected CPU lock package inventory")
    installed = {name: version(name) for name in expected}
    require(installed == expected, "probe dependency version mismatch")
    return installed


def read_assets(directory: Path) -> dict[str, bytes]:
    assets = {}
    for name, expected in ASSETS.items():
        with (directory / name).open("rb") as stream:
            body = stream.read(20_000_001)
        require(len(body) <= 20_000_000 and digest(body) == expected,
                "tokenizer asset mismatch: " + name)
        assets[name] = body
    return assets


@contextmanager
def offline_cpu():
    """Guard the probe process; dependency installation is a separate operation."""
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1", "CUDA_VISIBLE_DEVICES": "",
            "ACCELERATE_USE_CPU": "true", "WANDB_DISABLED": "true",
        }))
        for owner, attribute in ((socket.socket, "connect"),
                                 (socket.socket, "connect_ex"),
                                 (socket.socket, "sendto"),
                                 (socket, "create_connection"),
                                 (socket, "getaddrinfo")):
            stack.enter_context(patch.object(owner, attribute,
                                             side_effect=ProbeError("network is forbidden in probe")))
        yield


def verify_record(record: dict, full_ids: list, prompt_ids: list, max_length: int) -> int:
    """Check the token-level objective independently of TRL's constructed mask."""
    boundary = len(prompt_ids)
    require(0 < boundary < len(full_ids) <= max_length, "empty or truncated completion")
    require(full_ids[:boundary] == prompt_ids, "prompt token prefix mismatch")
    require(record["input_ids"] == full_ids, "preprocessing changed or truncated tokens")
    labels = record["labels"]
    require(len(labels) == len(full_ids), "label length mismatch")
    require(labels[:boundary] == [-100] * boundary, "prompt contributes to loss")
    require(labels[boundary:] == full_ids[boundary:], "completion labels lost or changed")
    return boundary


def verify_batch(batch: dict, records: list[dict], pad_id: int) -> int:
    """Verify tensors after the real collator, including attention and padding."""
    require(set(batch) == {"input_ids", "labels", "attention_mask"}, "unexpected batch fields")
    shape = batch["input_ids"].shape
    require(len(shape) == 2 and shape[0] == len(records), "batch row count mismatch")
    require(all(value.device.type == "cpu" and value.shape == shape for value in batch.values()),
            "non-CPU tensor or batch shape mismatch")
    padding = 0
    for index, record in enumerate(records):
        length = len(record["input_ids"])
        require(batch["input_ids"][index, :length].tolist() == record["input_ids"],
                "collator changed input tokens")
        require(batch["labels"][index, :length].tolist() == record["labels"],
                "collator changed completion objective")
        require(batch["attention_mask"][index, :length].eq(1).all().item(), "masked real token")
        require(batch["input_ids"][index, length:].eq(pad_id).all().item(), "wrong padding token")
        require(batch["labels"][index, length:].eq(-100).all().item(), "padding contributes to loss")
        require(batch["attention_mask"][index, length:].eq(0).all().item(), "padding receives attention")
        padding += shape[1] - length
    return padding


def probe(directory: Path) -> dict:
    installed = verify_environment()
    recipe = load_recipe()
    require(recipe["base_revision"] == REVISION, "tokenizer/model source revision mismatch")
    assets = read_assets(directory)
    plan, _, source_rows = load_pilot()
    train, validation = prepare_sft_rows()
    results, padding_tokens, batches = [], 0, 0

    with offline_cpu(), tempfile.TemporaryDirectory(prefix="kova-mask-probe-") as temporary:
        # Copy only verified bytes into an otherwise empty directory. Unrelated
        # files supplied alongside them cannot alter AutoTokenizer discovery.
        local = Path(temporary) / "tokenizer"
        local.mkdir()
        for name, body in assets.items():
            (local / name).write_bytes(body)

        import torch
        from datasets import Dataset, disable_progress_bars
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, Trainer
        from trl import SFTConfig, SFTTrainer
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

        require(torch.version.cuda is None, "probe requires CPU-only PyTorch")
        disable_progress_bars()
        with ExitStack() as guard:
            for owner, attribute in ((AutoModel, "from_pretrained"), (AutoModel, "from_config"),
                                     (AutoModelForCausalLM, "from_pretrained"),
                                     (AutoModelForCausalLM, "from_config"),
                                     (Trainer, "__init__"), (SFTTrainer, "__init__"),
                                     (SFTTrainer, "train")):
                guard.enter_context(patch.object(owner, attribute,
                                                  side_effect=ProbeError("model/training is forbidden in probe")))
            tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
            training = recipe["training"]
            args = SFTConfig(
                output_dir=str(Path(temporary) / "unused-output"),
                use_cpu=True, fp16=False, bf16=False,
                max_length=training["max_length"], packing=training["packing"],
                completion_only_loss=training["completion_only_loss"],
                seed=training["seed"], report_to="none", push_to_hub=False,
            )
            # Only the real dataset-preparation method is exercised. Constructing
            # a full Trainer would load model weights and is deliberately blocked.
            preparer = object.__new__(SFTTrainer)
            preparer._tokenizer = tokenizer
            preparer.chat_template = tokenizer.chat_template
            preparer.completion_only_loss = args.completion_only_loss
            collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
            for split, rows, batch_size in (
                ("train", train, training["per_device_train_batch_size"]),
                ("validation", validation, training["per_device_eval_batch_size"]),
            ):
                ids = [row["id"] for row in source_rows if row["split"] == split]
                require(len(ids) == len(rows), "source/prepared record count mismatch")
                originals = dict(zip(ids, rows, strict=True))
                dataset = Dataset.from_list([dict(row, probe_id=identifier)
                                             for identifier, row in originals.items()])
                processed = preparer._prepare_dataset(dataset, tokenizer, args, args.packing, None, split)
                require(len(processed) == len(rows), "TRL dropped or added examples")
                require(sorted(processed["probe_id"]) == sorted(ids), "duplicate or missing example")
                for record in processed:
                    row = originals[record["probe_id"]]
                    prompt_ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                                                                add_generation_prompt=True,
                                                                return_dict=False)
                    full_ids = tokenizer.apply_chat_template(row["prompt"] + row["completion"],
                                                              tokenize=True, add_generation_prompt=False,
                                                              return_dict=False)
                    boundary = verify_record(record, full_ids, prompt_ids, args.max_length)
                    # Preserve the assistant EOS even if it is also a pad token.
                    require(tokenizer.eos_token_id in record["labels"][boundary:], "completion EOS lost")
                    results.append({"id": record["probe_id"], "split": split,
                                    "tokens": len(full_ids), "prompt_tokens": boundary,
                                    "completion_tokens": len(full_ids) - boundary,
                                    "labels_sha256": digest(json.dumps(record["labels"]).encode())})
                for start in range(0, len(processed), batch_size):
                    records = [processed[index] for index in range(start, min(start + batch_size, len(processed)))]
                    padding_tokens += verify_batch(collator(records), records, tokenizer.pad_token_id)
                    batches += 1

    require(padding_tokens > 0, "padding was not exercised")
    results.sort(key=lambda row: row["id"])
    return {
        "status": "real_trl_preprocessing_and_collator_verified_on_cpu",
        "base_revision": REVISION, "tokenizer_asset_sha256": ASSETS,
        "dataset_sha256": plan["dataset_sha256"], "prompt_sha256": plan["prompt_sha256"],
        "recipe_sha256": digest((ROOT / "config/kova-cosmo-sft.v1.json").read_bytes()),
        "probe_sha256": digest(Path(__file__).read_bytes()), "cpu_lock_sha256": digest(LOCK.read_bytes()),
        "software": installed, "python": sys.version.split()[0],
        "records": results, "batches": batches, "padding_tokens_checked": padding_tokens,
        "max_tokens": max(row["tokens"] for row in results),
        "model_weights_loaded": False, "training_started": False,
        "model_forward_backward_verified": False, "gpu_verified": False,
        "phase_b_ready": False, "closed_checklist_ids": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(probe(arguments.assets), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
