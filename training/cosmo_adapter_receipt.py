"""Create and verify a hash-bound receipt for a retained Cosmo LoRA adapter.

The receipt is written only after training. It is not a release decision and
never loads model tensors, contacts a network, or invokes cloud services.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

from training.cosmo_artifacts import EXPECTED_SHA256
from training.cosmo_lifecycle_authority import AuthorityError, verify_envelope
from training import identity_pilot as pilot

ROOT = Path(__file__).resolve().parents[1]
RECEIPT_NAME = "adapter-receipt.v1.json"
TRAINING_GRANT_NAME = "training-grant-envelope.v1.json"
ADAPTER_DIRECTORY = "adapter"
HEX40 = re.compile(r"[0-9a-f]{40}")
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_GRANT_BYTES = 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
MAX_README_BYTES = 4 * 1024 * 1024
MAX_ADAPTER_BYTES = 512 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
ALLOWED_ARTIFACTS = {
    "README.md": MAX_README_BYTES,
    "adapter_config.json": MAX_CONFIG_BYTES,
    "adapter_model.safetensors": MAX_ADAPTER_BYTES,
}
REQUIRED_ARTIFACTS = {"adapter_config.json", "adapter_model.safetensors"}


class ReceiptError(ValueError):
    pass


def need(condition: bool) -> None:
    if not condition:
        raise ReceiptError("kova cosmo adapter receipt rejected")


def _unique(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        need(type(key) is str and key not in value)
        value[key] = item
    return value


def _reject_constant(_value: str) -> None:
    raise ReceiptError("kova cosmo adapter receipt rejected")


def parse_json(raw: bytes) -> object:
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, TypeError, RecursionError, ReceiptError):
        raise ReceiptError("kova cosmo adapter receipt rejected") from None


def serialize(value: object) -> bytes:
    try:
        return (json.dumps(
            value, indent=2, ensure_ascii=True, allow_nan=False
        ) + "\n").encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ReceiptError("kova cosmo adapter receipt rejected") from None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_digest(root: Path, relative: str) -> str:
    path = root / relative
    need(path.is_file() and not path.is_symlink())
    try:
        need(path.resolve(strict=True).is_relative_to(root.resolve(strict=True)))
        return file_sha256(path)
    except (OSError, ValueError):
        raise ReceiptError("kova cosmo adapter receipt rejected") from None


def _read_limited(path: Path, maximum: int) -> bytes:
    need(path.is_file() and not path.is_symlink())
    try:
        with path.open("rb") as stream:
            raw = stream.read(maximum + 1)
        need(0 < len(raw) <= maximum)
        return raw
    except OSError:
        raise ReceiptError("kova cosmo adapter receipt rejected") from None


def _validate_adapter_config(path: Path, recipe: dict) -> None:
    value = parse_json(_read_limited(path, MAX_CONFIG_BYTES))
    need(type(value) is dict)
    need(value.get("peft_type") == "LORA")
    need(value.get("task_type") == "CAUSAL_LM")
    need(value.get("r") == recipe["lora"]["r"])
    need(value.get("lora_alpha") == recipe["lora"]["alpha"])
    need(value.get("lora_dropout") == recipe["lora"]["dropout"])
    need(value.get("bias") == recipe["lora"]["bias"])
    need(value.get("modules_to_save") is None)
    base_path = value.get("base_model_name_or_path")
    need(type(base_path) is str and 0 < len(base_path) <= 4096)
    need(Path(base_path).is_absolute())
    targets = value.get("target_modules")
    need(type(targets) is list and len(targets) == len(recipe["lora"]["target_modules"]))
    need(all(type(item) is str for item in targets))
    need(set(targets) == set(recipe["lora"]["target_modules"]))


def validate_safetensors(path: Path, recipe: dict,
                         expected_layers: int = 28) -> None:
    """Validate a safe, complete LoRA-only tensor inventory without loading it."""
    need(type(expected_layers) is int and expected_layers > 0)
    raw_size = path.stat().st_size
    need(8 < raw_size <= MAX_ADAPTER_BYTES)
    try:
        with path.open("rb") as stream:
            header_size_raw = stream.read(8)
            need(len(header_size_raw) == 8)
            header_size = struct.unpack("<Q", header_size_raw)[0]
            need(1 < header_size <= MAX_SAFETENSORS_HEADER_BYTES)
            need(8 + header_size < raw_size)
            header = parse_json(stream.read(header_size))
    except (OSError, struct.error):
        raise ReceiptError("kova cosmo adapter receipt rejected") from None
    need(type(header) is dict)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    expected_count = expected_layers * len(recipe["lora"]["target_modules"]) * 2
    need(len(tensors) == expected_count)
    intervals = []
    item_bytes = {"F16": 2, "BF16": 2, "F32": 4}
    target_counts = {target: 0 for target in recipe["lora"]["target_modules"]}
    for name, metadata in tensors.items():
        need(type(name) is str and type(metadata) is dict)
        need(set(metadata) == {"dtype", "shape", "data_offsets"})
        matched = [target for target in target_counts
                   if f".{target}.lora_A.weight" in name or
                   f".{target}.lora_B.weight" in name]
        need(len(matched) == 1)
        target_counts[matched[0]] += 1
        need(metadata.get("dtype") in item_bytes)
        shape = metadata.get("shape")
        need(type(shape) is list and len(shape) == 2)
        need(all(type(item) is int and item > 0 for item in shape))
        offsets = metadata.get("data_offsets")
        need(type(offsets) is list and len(offsets) == 2)
        need(all(type(item) is int and item >= 0 for item in offsets))
        need(offsets[0] < offsets[1])
        elements = shape[0] * shape[1]
        need(offsets[1] - offsets[0] == elements * item_bytes[metadata["dtype"]])
        intervals.append(tuple(offsets))
    need(all(count == expected_layers * 2 for count in target_counts.values()))
    ordered = sorted(intervals)
    need(ordered[0][0] == 0)
    need(all(first[1] == second[0] for first, second in zip(ordered, ordered[1:])))
    need(ordered[-1][1] == raw_size - 8 - header_size)


def _load_sources(root: Path) -> tuple[dict, dict, str]:
    # Local import avoids an import cycle when the trainer imports this module.
    try:
        from training.kova_cosmo_sft import load_recipe
        from training.cosmo_sft_evaluation import load_plan

        recipe = load_recipe(root)
        identity, _, _ = pilot.load(root)
        _, _, evaluation_plan_sha256 = load_plan(root)
        return recipe, identity, evaluation_plan_sha256
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            UnicodeError, RecursionError):
        raise ReceiptError("kova cosmo adapter receipt rejected") from None


def _artifact_inventory(output: Path, recipe: dict) -> list[dict]:
    adapter = output / ADAPTER_DIRECTORY
    need(output.is_absolute() and output.is_dir() and not output.is_symlink())
    need(adapter.is_dir() and not adapter.is_symlink())
    try:
        children = sorted(adapter.iterdir(), key=lambda item: item.name)
    except OSError:
        raise ReceiptError("kova cosmo adapter receipt rejected") from None
    need(all(item.is_file() and not item.is_symlink() for item in children))
    names = {item.name for item in children}
    need(REQUIRED_ARTIFACTS <= names <= set(ALLOWED_ARTIFACTS))
    _validate_adapter_config(adapter / "adapter_config.json", recipe)
    validate_safetensors(adapter / "adapter_model.safetensors", recipe)
    inventory = []
    for path in children:
        size = path.stat().st_size
        need(0 < size <= ALLOWED_ARTIFACTS[path.name])
        inventory.append({
            "path": f"{ADAPTER_DIRECTORY}/{path.name}",
            "bytes": size,
            "sha256": file_sha256(path),
        })
    return inventory


def _verified_training_grant(
    output: Path, *, source_commit: str, runtime_evidence_sha256: str,
    lifecycle_phase_grant_sha256: str, lifecycle_id: str,
    lifecycle_grant_id: str, lifecycle_ledger_commit_id: str,
    root: Path,
) -> dict:
    try:
        raw = _read_limited(output / TRAINING_GRANT_NAME, MAX_GRANT_BYTES)
        envelope = parse_json(raw)
        payload, envelope_sha256 = verify_envelope(
            envelope, expected_kind="kova_cosmo_paid_phase_grant", root=root
        )
        need(envelope_sha256 == lifecycle_phase_grant_sha256)
        need(payload["phase"] == "training")
        need(payload["source_commit"] == source_commit)
        need(payload["runtime_evidence_sha256"] == runtime_evidence_sha256)
        need(payload["lifecycle_id"] == lifecycle_id)
        need(payload["grant_id"] == lifecycle_grant_id)
        need(payload["ledger_commit_id"] == lifecycle_ledger_commit_id)
        need(payload["lifecycle_terminal"] is False)
        need(type(payload["context_sha256"]) is str and
             re.fullmatch(r"[0-9a-f]{64}", payload["context_sha256"]) is not None)
        return payload
    except (AuthorityError, OSError, ValueError, TypeError, KeyError,
            AttributeError, UnicodeError, RecursionError):
        raise ReceiptError("kova cosmo adapter receipt rejected") from None


def expected_receipt(output: Path, source_commit: str, *,
                     runtime_evidence_sha256: str,
                     lifecycle_phase_grant_sha256: str,
                     lifecycle_id: str,
                     lifecycle_grant_id: str,
                     lifecycle_ledger_commit_id: str,
                     global_steps: int,
                     training_loss: float, root: Path = ROOT) -> dict:
    need(type(source_commit) is str and HEX40.fullmatch(source_commit) is not None)
    need(type(runtime_evidence_sha256) is str and
         re.fullmatch(r"[0-9a-f]{64}", runtime_evidence_sha256) is not None)
    need(type(lifecycle_phase_grant_sha256) is str and
         re.fullmatch(r"[0-9a-f]{64}",
                      lifecycle_phase_grant_sha256) is not None)
    for value in (lifecycle_id, lifecycle_grant_id,
                  lifecycle_ledger_commit_id):
        need(type(value) is str and 0 < len(value) <= 256)
    need(type(global_steps) is int and not isinstance(global_steps, bool) and
         0 < global_steps < 2**31)
    need(type(training_loss) in (int, float) and
         not isinstance(training_loss, bool) and
         math.isfinite(training_loss) and training_loss >= 0)
    recipe, identity, evaluation_plan_sha256 = _load_sources(root)
    grant = _verified_training_grant(
        output,
        source_commit=source_commit,
        runtime_evidence_sha256=runtime_evidence_sha256,
        lifecycle_phase_grant_sha256=lifecycle_phase_grant_sha256,
        lifecycle_id=lifecycle_id,
        lifecycle_grant_id=lifecycle_grant_id,
        lifecycle_ledger_commit_id=lifecycle_ledger_commit_id,
        root=root,
    )
    artifacts = _artifact_inventory(output, recipe)
    adapter_sha256 = next(item["sha256"] for item in artifacts
                          if item["path"] == "adapter/adapter_model.safetensors")
    return {
        "schema_version": 1,
        "kind": "kova_cosmo_lora_adapter",
        "status": "trained_adapter_not_evaluated",
        "source_commit": source_commit,
        "lineage": {
            "base_model": recipe["base_model"],
            "base_revision": recipe["base_revision"],
            "base_model_safetensors_sha256": EXPECTED_SHA256["model.safetensors"],
            "dataset_sha256": identity["dataset_sha256"],
            "prompt_sha256": identity["prompt_sha256"],
            "review_sha256": identity["review_sha256"],
            "recipe_sha256": _source_digest(root, "config/kova-cosmo-sft.v1.json"),
            "runtime_guard_sha256": _source_digest(
                root, "config/kova-cosmo-runtime-guard.v1.json"
            ),
            "lifecycle_trust_sha256": _source_digest(
                root, "config/kova-cosmo-lifecycle-trust.v1.json"
            ),
            "runtime_evidence_sha256": runtime_evidence_sha256,
            "lifecycle_id": lifecycle_id,
            "lifecycle_grant_id": lifecycle_grant_id,
            "lifecycle_ledger_commit_id": lifecycle_ledger_commit_id,
            "lifecycle_phase_grant_sha256":
                lifecycle_phase_grant_sha256,
            "training_grant_context_sha256": grant["context_sha256"],
            "training_grant_azure_resource_id": grant["azure_resource_id"],
            "evaluation_plan_sha256": evaluation_plan_sha256,
            "software_lock_sha256": _source_digest(
                root, "requirements/kova-cosmo-sft-py312-linux.lock"
            ),
        },
        "training": {
            "method": recipe["method"],
            "epochs": recipe["training"]["num_train_epochs"],
            "seed": recipe["training"]["seed"],
            "optimizer": recipe["training"]["optimizer"],
            "max_length": recipe["training"]["max_length"],
            "completion_only_loss": recipe["training"]["completion_only_loss"],
            "global_steps": global_steps,
            "training_loss": training_loss,
        },
        "runtime": {
            "region": recipe["hardware"]["region"],
            "vm_size": recipe["hardware"]["vm_size"],
            "gpu_family": recipe["hardware"]["gpu_family"],
            "precision": recipe["hardware"]["precision"],
            "quantized_base": recipe["hardware"]["quantized_base"],
        },
        "artifacts": artifacts,
        "adapter_sha256": adapter_sha256,
        "actual_model_outputs_evaluated": False,
        "deployment_authorized": False,
        "phase_b_ready": False,
    }


def write_receipt(output: Path, source_commit: str, *,
                  runtime_evidence_sha256: str,
                  lifecycle_phase_grant_sha256: str,
                  lifecycle_id: str,
                  lifecycle_grant_id: str,
                  lifecycle_ledger_commit_id: str,
                  signed_training_grant_envelope: dict,
                  global_steps: int,
                  training_loss: float, root: Path = ROOT) -> dict:
    """Create a new receipt without overwriting any existing evidence."""
    grant_created = False
    grant_path = output / TRAINING_GRANT_NAME
    try:
        need(type(signed_training_grant_envelope) is dict)
        need(not grant_path.exists() and not grant_path.is_symlink())
        grant_raw = serialize(signed_training_grant_envelope)
        need(0 < len(grant_raw) <= MAX_GRANT_BYTES)
        with grant_path.open("xb") as stream:
            stream.write(grant_raw)
        grant_created = True
        value = expected_receipt(
            output, source_commit,
            runtime_evidence_sha256=runtime_evidence_sha256,
            lifecycle_phase_grant_sha256=
                lifecycle_phase_grant_sha256,
            lifecycle_id=lifecycle_id,
            lifecycle_grant_id=lifecycle_grant_id,
            lifecycle_ledger_commit_id=lifecycle_ledger_commit_id,
            global_steps=global_steps,
            training_loss=training_loss,
            root=root,
        )
        raw = serialize(value)
        need(len(raw) <= MAX_RECEIPT_BYTES)
        with (output / RECEIPT_NAME).open("xb") as stream:
            stream.write(raw)
        return verify_receipt(output, expected_source_commit=source_commit, root=root)
    except ReceiptError:
        if grant_created:
            try:
                grant_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    except OSError:
        raise ReceiptError("kova cosmo adapter receipt rejected") from None


def verify_receipt(output: Path, *, expected_source_commit: str | None = None,
                   root: Path = ROOT) -> dict:
    try:
        path = output / RECEIPT_NAME
        raw = _read_limited(path, MAX_RECEIPT_BYTES)
        value = parse_json(raw)
        need(type(value) is dict)
        source_commit = value.get("source_commit")
        need(type(source_commit) is str and HEX40.fullmatch(source_commit) is not None)
        if expected_source_commit is not None:
            need(source_commit == expected_source_commit)
        lineage = value.get("lineage")
        training = value.get("training")
        need(type(lineage) is dict and type(training) is dict)
        expected = expected_receipt(
            output, source_commit,
            runtime_evidence_sha256=lineage.get("runtime_evidence_sha256"),
            lifecycle_phase_grant_sha256=lineage.get(
                "lifecycle_phase_grant_sha256"
            ),
            lifecycle_id=lineage.get("lifecycle_id"),
            lifecycle_grant_id=lineage.get("lifecycle_grant_id"),
            lifecycle_ledger_commit_id=lineage.get(
                "lifecycle_ledger_commit_id"
            ),
            global_steps=training.get("global_steps"),
            training_loss=training.get("training_loss"),
            root=root,
        )
        pilot.same(value, expected)
        need(raw == serialize(expected))
        return {
            "status": "trained_adapter_receipt_verified",
            "source_commit": source_commit,
            "receipt_sha256": hashlib.sha256(raw).hexdigest(),
            "adapter_sha256": expected["adapter_sha256"],
            "lifecycle_phase_grant_sha256": expected["lineage"][
                "lifecycle_phase_grant_sha256"
            ],
            "training_grant_context_sha256": expected["lineage"][
                "training_grant_context_sha256"
            ],
            "training_grant_azure_resource_id": expected["lineage"][
                "training_grant_azure_resource_id"
            ],
            "lifecycle_id": expected["lineage"]["lifecycle_id"],
            "lifecycle_grant_id": expected["lineage"][
                "lifecycle_grant_id"
            ],
            "lifecycle_ledger_commit_id": expected["lineage"][
                "lifecycle_ledger_commit_id"
            ],
            "artifact_files": len(expected["artifacts"]),
            "actual_model_outputs_evaluated": False,
            "deployment_authorized": False,
            "phase_b_ready": False,
            "closed_checklist_ids": [],
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            UnicodeError, RecursionError):
        raise ReceiptError("kova cosmo adapter receipt rejected") from None


def dry_run(root: Path = ROOT) -> dict:
    """Validate receipt source inputs without requiring or creating an adapter."""
    recipe, identity, evaluation_plan_sha256 = _load_sources(root)
    return {
        "status": "awaiting_trained_adapter",
        "base_model": recipe["base_model"],
        "base_revision": recipe["base_revision"],
        "dataset_sha256": identity["dataset_sha256"],
        "recipe_sha256": _source_digest(root, "config/kova-cosmo-sft.v1.json"),
        "runtime_guard_sha256": _source_digest(
            root, "config/kova-cosmo-runtime-guard.v1.json"
        ),
        "lifecycle_trust_sha256": _source_digest(
            root, "config/kova-cosmo-lifecycle-trust.v1.json"
        ),
        "evaluation_plan_sha256": evaluation_plan_sha256,
        "software_lock_sha256": _source_digest(
            root, "requirements/kova-cosmo-sft-py312-linux.lock"
        ),
        "adapter_receipt_created": False,
        "actual_model_outputs_evaluated": False,
        "deployment_authorized": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path,
                        help="Existing external run directory to verify")
    parser.add_argument("--source-commit",
                        help="Expected 40-character lowercase commit")
    arguments = parser.parse_args(argv)
    try:
        if arguments.output is None:
            need(arguments.source_commit is None)
            report = dry_run()
        else:
            need(arguments.output.is_absolute())
            report = verify_receipt(
                arguments.output,
                expected_source_commit=arguments.source_commit,
            )
        print(json.dumps(report, sort_keys=True))
        return 0
    except ReceiptError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
