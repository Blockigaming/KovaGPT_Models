"""Create and verify a hash-bound receipt for a retained Cosmo LoRA adapter.

The receipt is written only after training. It is not a release decision and
never loads model tensors, contacts a network, or invokes cloud services.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys

from training.cosmo_artifacts import EXPECTED_BYTES, EXPECTED_SHA256
from training import identity_pilot as pilot

ROOT = Path(__file__).resolve().parents[1]
RECEIPT_NAME = "adapter-receipt.v1.json"
RECEIPT_ATTESTATION_NAME = "adapter-receipt-attestation.v1.json"
TRAINING_GRANT_NAME = "training-grant.v1.json"
ADAPTER_DIRECTORY = "adapter"
HEX40 = re.compile(r"[0-9a-f]{40}")
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_GRANT_BYTES = 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
MAX_README_BYTES = 4 * 1024 * 1024
MAX_ADAPTER_BYTES = 512 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
EXPECTED_GLOBAL_STEPS = 18
EXPECTED_HIDDEN_SIZE = 1024
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


def _verify_training_grant(
    output: Path, *, expected_source_commit: str | None = None,
    expected_runtime_evidence_sha256: str | None = None,
    expected_lifecycle_id: str | None = None, root: Path = ROOT,
) -> dict:
    """Verify the persisted authority-signed training grant and its context."""
    try:
        # Keep the CPU-only adapter round-trip validator independent of the
        # cryptography package; paid-lineage verification imports it on demand.
        from training.cosmo_lifecycle_authority import (
            PHASE_RESERVED_SECONDS,
            PHASES,
            PILOT_ID,
            canonical,
            load_trust_policy,
            money,
            timestamp,
            validate_azure_instance,
            verify_envelope,
        )
        evidence = parse_json(_read_limited(
            output / TRAINING_GRANT_NAME, MAX_GRANT_BYTES
        ))
        need(type(evidence) is dict and list(evidence) == [
            "context", "grant_envelope",
        ])
        context = evidence["context"]
        need(type(context) is dict and list(context) == [
            "operation", "base_model", "base_revision", "output_directory",
            "snapshot_inventory",
        ])
        recipe, _, _ = _load_sources(root)
        need(context["operation"] == "single_lora_sft_run")
        need(context["base_model"] == recipe["base_model"])
        need(context["base_revision"] == recipe["base_revision"])
        need(context["output_directory"] == str(output))
        inventory = context["snapshot_inventory"]
        need(type(inventory) is dict and
             list(inventory) == list(EXPECTED_SHA256))
        for name, item in inventory.items():
            need(type(item) is dict and list(item) == ["bytes", "sha256"])
            need(item["bytes"] == EXPECTED_BYTES[name])
            need(item["sha256"] == EXPECTED_SHA256[name])

        envelope = evidence["grant_envelope"]
        payload, envelope_sha256 = verify_envelope(
            envelope, expected_kind="kova_cosmo_paid_phase_grant", root=root
        )
        need(type(payload) is dict and list(payload) == [
            "schema_version", "kind", "issuer", "pilot_id", "lifecycle_id",
            "preflight_ledger_sequence", "ledger_sequence",
            "ledger_commit_id", "ledger_append_only",
            "ledger_status", "grant_id", "phase", "source_commit",
            "runtime_evidence_sha256", "context_sha256", "request_nonce",
            "runtime_deadline_utc",
            "azure_instance", "azure_instance_identity",
            "issued_at_utc", "expires_at_utc", "grant_reserved_seconds",
            "grant_reserved_cost_usd", "phase_grants_committed",
            "training_runs_consumed", "aggregate_reserved_seconds",
            "aggregate_reserved_cost_usd", "approved_budget_usd",
            "deployment_authorized",
        ])
        need(payload["schema_version"] == 1)
        need(payload["pilot_id"] == PILOT_ID)
        need(payload["phase"] == "training")
        source_commit = payload["source_commit"]
        need(type(source_commit) is str and
             HEX40.fullmatch(source_commit) is not None)
        if expected_source_commit is not None:
            need(source_commit == expected_source_commit)
        runtime_sha256 = payload["runtime_evidence_sha256"]
        need(type(runtime_sha256) is str and
             re.fullmatch(r"[0-9a-f]{64}", runtime_sha256) is not None)
        if expected_runtime_evidence_sha256 is not None:
            need(runtime_sha256 == expected_runtime_evidence_sha256)
        need(payload["context_sha256"] ==
             hashlib.sha256(canonical(context)).hexdigest())
        lifecycle_id = payload["lifecycle_id"]
        need(type(lifecycle_id) is str and 0 < len(lifecycle_id) <= 256)
        if expected_lifecycle_id is not None:
            need(lifecycle_id == expected_lifecycle_id)
        for field in ("ledger_commit_id", "grant_id"):
            need(type(payload[field]) is str and
                 0 < len(payload[field]) <= 256)
        preflight_sequence = payload["preflight_ledger_sequence"]
        need(type(preflight_sequence) is int and
             0 < preflight_sequence < 2**63)
        need(type(payload["ledger_sequence"]) is int and
             preflight_sequence < payload["ledger_sequence"] < 2**63)
        need(payload["ledger_append_only"] is True)
        need(payload["ledger_status"] == "grant_committed_before_response")
        need(type(payload["request_nonce"]) is str and
             re.fullmatch(r"[0-9a-f]{64}", payload["request_nonce"]) is not None)
        need(payload["grant_reserved_seconds"] ==
             PHASE_RESERVED_SECONDS["training"])
        issued = timestamp(payload["issued_at_utc"])
        expires = timestamp(payload["expires_at_utc"])
        need(expires == timestamp(payload["runtime_deadline_utc"]))
        need(issued < expires)
        counts = payload["phase_grants_committed"]
        need(type(counts) is dict and list(counts) == list(PHASES))
        need(all(type(item) is int and 0 <= item <= 1
                 for item in counts.values()))
        need(counts["training"] == 1)
        need(type(payload["training_runs_consumed"]) is int and
             payload["training_runs_consumed"] == 1)
        aggregate_seconds = payload["aggregate_reserved_seconds"]
        need(type(aggregate_seconds) is int and
             aggregate_seconds == sum(
                 PHASE_RESERVED_SECONDS[phase] * counts[phase]
                 for phase in PHASES
             ))
        reserved_cost = money(payload["grant_reserved_cost_usd"])
        aggregate_cost = money(payload["aggregate_reserved_cost_usd"])
        need(reserved_cost >= (
            money("0.5260") *
            PHASE_RESERVED_SECONDS["training"] / 3600
        ))
        need(reserved_cost <= aggregate_cost <= money("0.5260"))
        need(aggregate_cost >= (
            money("0.5260") * aggregate_seconds / 3600
        ))
        need(money(payload["approved_budget_usd"]) == money("2.0000"))
        need(payload["deployment_authorized"] is False)
        azure_instance = validate_azure_instance(payload["azure_instance"])
        identity = payload["azure_instance_identity"]
        need(type(identity) is dict and list(identity) == [
            "verification_method", "token_sha256", "token_audience",
            "verified_at_utc", "token_expires_at_utc", "verified",
        ])
        need(identity["verification_method"] ==
             "microsoft_entra_system_assigned_managed_identity_token")
        need(type(identity["token_sha256"]) is str and
             re.fullmatch(r"[0-9a-f]{64}", identity["token_sha256"]) is not None)
        need(identity["token_audience"] == load_trust_policy(root)[
            "azure_managed_identity_token_audience"
        ])
        verified_at = timestamp(identity["verified_at_utc"])
        token_expires = timestamp(identity["token_expires_at_utc"])
        need(issued - timedelta(minutes=5) <= verified_at <= issued)
        need(verified_at < token_expires)
        need(token_expires >= expires)
        need(identity["verified"] is True)
        return {
            "source_commit": source_commit,
            "runtime_evidence_sha256": runtime_sha256,
            "lifecycle_id": lifecycle_id,
            "grant_id": payload["grant_id"],
            "ledger_sequence": payload["ledger_sequence"],
            "ledger_commit_id": payload["ledger_commit_id"],
            "context_sha256": payload["context_sha256"],
            "phase_grant_sha256": envelope_sha256,
            "azure_instance": azure_instance,
        }
    except (ImportError, OSError, ValueError, TypeError, KeyError,
            AttributeError, UnicodeError, RecursionError):
        raise ReceiptError("kova cosmo adapter receipt rejected") from None


def persist_training_grant(
    output: Path, *, source_commit: str, runtime_evidence_sha256: str,
    lifecycle_id: str, grant_context: dict, grant_envelope: dict,
    root: Path = ROOT,
) -> dict:
    """Persist verified grant evidence before imports or training begin."""
    try:
        need(output.is_absolute() and not output.exists() and
             not output.is_symlink())
        output.mkdir(mode=0o700)
        value = {
            "context": grant_context,
            "grant_envelope": grant_envelope,
        }
        raw = serialize(value)
        need(len(raw) <= MAX_GRANT_BYTES)
        with (output / TRAINING_GRANT_NAME).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return _verify_training_grant(
            output,
            expected_source_commit=source_commit,
            expected_runtime_evidence_sha256=runtime_evidence_sha256,
            expected_lifecycle_id=lifecycle_id,
            root=root,
        )
    except ReceiptError:
        raise
    except OSError:
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
                         expected_layers: int = 28, *,
                         expected_hidden_size: int = EXPECTED_HIDDEN_SIZE,
                         expected_intermediate_size: int = 3072,
                         expected_query_size: int = 2048,
                         expected_key_value_size: int = 1024) -> None:
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
    rank = recipe["lora"]["r"]
    need(all(type(item) is int and item > 0 for item in (
        expected_hidden_size, expected_intermediate_size,
        expected_query_size, expected_key_value_size,
    )))
    intermediate = expected_intermediate_size
    hidden = expected_hidden_size
    output_width = {
        "q_proj": expected_query_size,
        "k_proj": expected_key_value_size,
        "v_proj": expected_key_value_size,
        "o_proj": hidden,
        "gate_proj": intermediate,
        "up_proj": intermediate,
        "down_proj": hidden,
    }
    input_width = {
        "q_proj": hidden, "k_proj": hidden,
        "v_proj": hidden, "o_proj": hidden,
        "gate_proj": hidden, "up_proj": hidden,
        "down_proj": intermediate,
    }
    expected_shapes = {}
    for layer in range(expected_layers):
        for target in recipe["lora"]["target_modules"]:
            block = ("self_attn" if target in
                     ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp")
            prefix = (f"base_model.model.model.layers.{layer}.{block}."
                      f"{target}")
            expected_shapes[f"{prefix}.lora_A.weight"] = [rank, input_width[target]]
            expected_shapes[f"{prefix}.lora_B.weight"] = [output_width[target], rank]
    need(set(tensors) == set(expected_shapes))
    for name, metadata in tensors.items():
        need(type(name) is str and type(metadata) is dict)
        need(set(metadata) == {"dtype", "shape", "data_offsets"})
        need(metadata.get("dtype") in item_bytes)
        shape = metadata.get("shape")
        need(shape == expected_shapes[name])
        offsets = metadata.get("data_offsets")
        need(type(offsets) is list and len(offsets) == 2)
        need(all(type(item) is int and item >= 0 for item in offsets))
        need(offsets[0] < offsets[1])
        elements = shape[0] * shape[1]
        need(offsets[1] - offsets[0] == elements * item_bytes[metadata["dtype"]])
        intervals.append(tuple(offsets))
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


def expected_receipt(output: Path, source_commit: str, *,
                     global_steps: int, root: Path = ROOT) -> dict:
    need(type(source_commit) is str and HEX40.fullmatch(source_commit) is not None)
    need(type(global_steps) is int and not isinstance(global_steps, bool) and
         global_steps == EXPECTED_GLOBAL_STEPS)
    recipe, identity, evaluation_plan_sha256 = _load_sources(root)
    grant = _verify_training_grant(
        output, expected_source_commit=source_commit, root=root
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
            "runtime_evidence_sha256": grant["runtime_evidence_sha256"],
            "lifecycle_id": grant["lifecycle_id"],
            "lifecycle_grant_id": grant["grant_id"],
            "lifecycle_ledger_sequence": grant["ledger_sequence"],
            "lifecycle_ledger_commit_id": grant["ledger_commit_id"],
            "lifecycle_phase_grant_context_sha256": grant["context_sha256"],
            "lifecycle_phase_grant_sha256": grant["phase_grant_sha256"],
            "azure_instance": grant["azure_instance"],
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
                  global_steps: int, signing_key: object,
                  root: Path = ROOT) -> dict:
    """Create a new receipt without overwriting any existing evidence."""
    try:
        value = expected_receipt(
            output, source_commit,
            global_steps=global_steps,
            root=root,
        )
        raw = serialize(value)
        need(len(raw) <= MAX_RECEIPT_BYTES)
        with (output / RECEIPT_NAME).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        attested = {
            "schema_version": 1,
            "kind": "kova_cosmo_trained_adapter_attestation",
            "source_commit": source_commit,
            "receipt_sha256": hashlib.sha256(raw).hexdigest(),
            "adapter_sha256": value["adapter_sha256"],
        }
        attestation = {
            **attested,
            "signature": signing_key.sign(serialize(attested)).hex(),
        }
        attestation_raw = serialize(attestation)
        need(len(attestation_raw) <= MAX_RECEIPT_BYTES)
        with (output / RECEIPT_ATTESTATION_NAME).open("xb") as stream:
            stream.write(attestation_raw)
            stream.flush()
            os.fsync(stream.fileno())
        return verify_receipt(output, expected_source_commit=source_commit, root=root)
    except ReceiptError:
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
            global_steps=training.get("global_steps"),
            root=root,
        )
        pilot.same(value, expected)
        need(raw == serialize(expected))
        attestation_raw = _read_limited(
            output / RECEIPT_ATTESTATION_NAME, MAX_RECEIPT_BYTES
        )
        attestation = parse_json(attestation_raw)
        need(type(attestation) is dict and list(attestation) == [
            "schema_version", "kind", "source_commit", "receipt_sha256",
            "adapter_sha256", "signature",
        ])
        need(attestation["schema_version"] == 1)
        need(attestation["kind"] == "kova_cosmo_trained_adapter_attestation")
        need(attestation["source_commit"] == source_commit)
        need(attestation["receipt_sha256"] == hashlib.sha256(raw).hexdigest())
        need(attestation["adapter_sha256"] == expected["adapter_sha256"])
        signature = attestation["signature"]
        need(type(signature) is str and re.fullmatch(r"[0-9a-f]{128}", signature))
        from training.cosmo_generation_attestation import load_trust_policy
        trust = load_trust_policy(root)
        need(trust["status"] == "runner_signing_public_key_pinned")
        signed = {key: attestation[key] for key in list(attestation)[:-1]}
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        try:
            Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(trust["public_key_hex"])
            ).verify(bytes.fromhex(signature), serialize(signed))
        except InvalidSignature:
            raise ReceiptError("kova cosmo adapter receipt rejected") from None
        return {
            "status": "trained_adapter_receipt_verified",
            "source_commit": source_commit,
            "receipt_sha256": hashlib.sha256(raw).hexdigest(),
            "adapter_sha256": expected["adapter_sha256"],
            "lifecycle_phase_grant_sha256": expected["lineage"][
                "lifecycle_phase_grant_sha256"
            ],
            "lifecycle_id": expected["lineage"]["lifecycle_id"],
            "lifecycle_grant_id": expected["lineage"][
                "lifecycle_grant_id"
            ],
            "lifecycle_ledger_commit_id": expected["lineage"][
                "lifecycle_ledger_commit_id"
            ],
            "lifecycle_ledger_sequence": expected["lineage"][
                "lifecycle_ledger_sequence"
            ],
            "azure_instance": expected["lineage"]["azure_instance"],
            "artifact_files": len(expected["artifacts"]),
            "actual_model_outputs_evaluated": False,
            "deployment_authorized": False,
            "phase_b_ready": False,
            "closed_checklist_ids": [],
        }
    except (ImportError, OSError, ValueError, TypeError, KeyError, AttributeError,
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
        "authority_signed_training_grant_required": True,
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
