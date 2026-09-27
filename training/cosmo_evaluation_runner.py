"""Guarded three-way generation runner for the bounded Cosmo pilot.

Dry-run is the default. Execution requires promoted source gates, fresh external
runtime evidence, a separately verified adapter receipt, and an explicit paid
evaluation acknowledgement. Outputs are external and never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from training import cosmo_sft_evaluation as evaluation
from training import identity_pilot as pilot
from training.cosmo_adapter_receipt import ReceiptError, verify_receipt
from training.cosmo_artifacts import ArtifactError, verify_snapshot
from training.cosmo_generation_attestation import (
    AttestationError,
    create_attestation,
    load_signing_key,
    load_trust_policy as load_generation_trust_policy,
)
from training.cosmo_hardware import HardwareError, verify_nvidia_t4
from training.cosmo_lifecycle_authority import acquire_phase_grant
from training.cosmo_runtime_guard import require_ready
from training.kova_cosmo_sft import (
    load_recipe,
    verify_installed_software,
    verify_source_checkout,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION_ENV = "KOVA_CONFIRM_PAID_EVALUATION"
SNAPSHOT_ENV = "KOVA_COSMO_VERIFIED_SNAPSHOT"
ADAPTER_OUTPUT_ENV = "KOVA_COSMO_ADAPTER_OUTPUT"
EVALUATION_OUTPUT_ENV = "KOVA_COSMO_EVALUATION_OUTPUT"
SOURCE_COMMIT_ENV = "KOVA_SOURCE_COMMIT"
GENERATION_SIGNING_KEY_ENV = "KOVA_COSMO_GENERATION_SIGNING_KEY_FILE"
MAX_NEW_TOKENS = 256


class EvaluationRunnerError(ValueError):
    pass


def need(condition: bool) -> None:
    if not condition:
        raise EvaluationRunnerError("kova cosmo evaluation runner rejected")


def external_existing(raw: object) -> Path:
    need(type(raw) is str and 0 < len(raw) <= 4096)
    path = Path(raw)
    need(path.is_absolute())
    try:
        resolved = path.resolve(strict=True)
        repository = ROOT.resolve(strict=True)
    except OSError:
        raise EvaluationRunnerError("kova cosmo evaluation runner rejected") from None
    need(resolved.is_dir())
    need(repository != resolved and repository not in resolved.parents)
    return resolved


def external_new(raw: object) -> Path:
    need(type(raw) is str and 0 < len(raw) <= 4096)
    path = Path(raw)
    need(path.is_absolute() and path.name not in ("", ".", ".."))
    need(not path.exists() and not path.is_symlink())
    try:
        parent = path.parent.resolve(strict=True)
        repository = ROOT.resolve(strict=True)
    except OSError:
        raise EvaluationRunnerError("kova cosmo evaluation runner rejected") from None
    resolved = parent / path.name
    need(parent.is_dir())
    need(repository != resolved and repository not in resolved.parents)
    return resolved


def authorize() -> tuple[dict, dict, Path, Path, Path, object, dict, dict]:
    recipe = load_recipe()
    gates = recipe["account_gates"]
    permissions = recipe["execution"]
    need(gates["eastus_ncast4_quota_verified"] is True)
    need(gates["runtime_compatibility_verified"] is True)
    need(gates["approved_budget_usd"] == 2.0)
    need(permissions["model_download_authorized"] is True)
    need(permissions["training_authorized"] is True)
    need(permissions["deployment_authorized"] is False)
    need(os.environ.get(CONFIRMATION_ENV) == "YES")

    runtime = require_ready()
    verify_installed_software()
    snapshot = external_existing(os.environ.get(SNAPSHOT_ENV))
    try:
        verify_snapshot(snapshot)
    except ArtifactError:
        raise EvaluationRunnerError("kova cosmo evaluation runner rejected") from None
    adapter_output = external_existing(os.environ.get(ADAPTER_OUTPUT_ENV))
    evaluation_output = external_new(os.environ.get(EVALUATION_OUTPUT_ENV))
    generation_signing_key_path = Path(
        os.environ.get(GENERATION_SIGNING_KEY_ENV, "")
    )
    try:
        generation_trust = load_generation_trust_policy()
        need(generation_trust["status"] ==
             "runner_signing_public_key_pinned")
        generation_signing_key = load_signing_key(
            generation_signing_key_path,
            expected_public_key_hex=generation_trust["public_key_hex"],
        )
        generation_signing_key_path = generation_signing_key_path.resolve(
            strict=True
        )
    except AttestationError:
        raise EvaluationRunnerError(
            "kova cosmo evaluation runner rejected"
        ) from None
    source_commit = os.environ.get(SOURCE_COMMIT_ENV)
    need(type(source_commit) is str and len(source_commit) == 40)
    need(all(character in "0123456789abcdef" for character in source_commit))
    verify_source_checkout(source_commit)
    try:
        receipt = verify_receipt(
            adapter_output, expected_source_commit=source_commit
        )
    except ReceiptError:
        raise EvaluationRunnerError("kova cosmo evaluation runner rejected") from None
    need(receipt["lifecycle_id"] == runtime["lifecycle_id"])
    need(receipt["azure_instance"] == runtime["azure_instance"])
    need(snapshot != adapter_output)
    need(snapshot not in adapter_output.parents and
         adapter_output not in snapshot.parents)
    need(snapshot not in evaluation_output.parents)
    need(adapter_output not in evaluation_output.parents)
    need(generation_signing_key_path != snapshot and
         snapshot not in generation_signing_key_path.parents)
    need(generation_signing_key_path != adapter_output and
         adapter_output not in generation_signing_key_path.parents)
    need(generation_signing_key_path != evaluation_output and
         evaluation_output not in generation_signing_key_path.parents)
    phase_grant = acquire_phase_grant(
        phase="evaluation",
        source_commit=source_commit,
        runtime_evidence_sha256=runtime["runtime_evidence_sha256"],
        lifecycle_id=runtime["lifecycle_id"],
        preflight_ledger_sequence=runtime["preflight_ledger_sequence"],
        azure_instance=runtime["azure_instance"],
        runtime_deadline_utc=runtime["deadline_utc"],
        context={
            "operation": "three_way_guarded_generation",
            "base_model": recipe["base_model"],
            "base_revision": recipe["base_revision"],
            "adapter_receipt_sha256": receipt["receipt_sha256"],
            "evaluation_output": str(evaluation_output),
            "expected_attempts": 36,
        },
    )
    return (
        recipe, runtime, snapshot, adapter_output, evaluation_output,
        generation_signing_key, receipt, phase_grant,
    )


def messages_for_variant(case: dict, variant: str, prompt: str) -> list[dict]:
    need(variant in evaluation.VARIANTS)
    if variant == "base":
        return [dict(case["messages"][0])]
    return pilot.format_messages(prompt, case)[:-1]


def attempt_record(*, case: dict, variant: str, runtime: dict,
                   answer: str | None = None, latency_ms: int | None = None,
                   input_tokens: int | None = None,
                   output_tokens: int | None = None) -> dict:
    success = answer is not None
    if success:
        need(type(answer) is str and 0 < len(answer) <= 750000)
        need(type(latency_ms) is int and latency_ms >= 0)
        need(type(input_tokens) is int and input_tokens >= 0)
        need(type(output_tokens) is int and output_tokens > 0)
    else:
        need(latency_ms is None and input_tokens is None and output_tokens is None)
    return {
        "id": case["id"] + "-" + variant,
        "case_id": case["id"],
        "case_sha256": evaluation.digest(case),
        "variant": variant,
        "outcome": "success" if success else "failed",
        "answer": answer,
        "answer_sha256": (
            evaluation.answer_digest(answer) if success else None
        ),
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "runtime": dict(runtime),
        "scores": {dimension: "pending" for dimension in evaluation.DIMENSIONS},
    }


def _write_json_exclusive(path: Path, value: object) -> None:
    raw = (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(raw)


def _append_attempt(stream, row: dict) -> None:
    raw = (json.dumps(row, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")) + "\n").encode("ascii")
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())


def execute() -> dict:
    (recipe, runtime_guard, snapshot, adapter_output, output,
     generation_signing_key, receipt, phase_grant) = authorize()

    # Heavy dependencies are imported only after every source/runtime guard and
    # both immutable artifact sets have been verified.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        device = verify_nvidia_t4(torch)
    except HardwareError:
        raise EvaluationRunnerError(
            "kova cosmo evaluation runner rejected"
        ) from None
    device_name = device["device_name"]
    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "WANDB_DISABLED": "true",
    })

    _, prompt, _ = pilot.load()
    plan, cases, plan_sha256 = evaluation.load_plan()
    software_lock_sha256 = hashlib.sha256(
        (ROOT / "requirements/kova-cosmo-sft-py312-linux.lock").read_bytes()
    ).hexdigest()
    common_runtime = {
        "base_model": recipe["base_model"],
        "base_revision": recipe["base_revision"],
        "adapter_sha256": None,
        "adapter_receipt_sha256": receipt["receipt_sha256"],
        "software_lock_sha256": software_lock_sha256,
        "runtime_evidence_sha256": runtime_guard["runtime_evidence_sha256"],
        "lifecycle_id": phase_grant["lifecycle_id"],
        "lifecycle_grant_id": phase_grant["grant_id"],
        "lifecycle_ledger_commit_id": phase_grant["ledger_commit_id"],
        "lifecycle_phase_grant_sha256": phase_grant[
            "phase_grant_sha256"
        ],
        "hardware": "eastus/Standard_NC4as_T4_v3/" + device_name,
        "precision": "fp16",
        "quantization": "none",
    }
    bundle = {
        "schema_version": 1,
        "kind": "measured",
        "plan_sha256": plan_sha256,
        "source_commit": receipt["source_commit"],
        "adapter_sha256": receipt["adapter_sha256"],
        "adapter_receipt_sha256": receipt["receipt_sha256"],
        "runner_attestation": None,
        "attempts": [],
    }

    output.mkdir(mode=0o700)
    _write_json_exclusive(output / "run-metadata.json", {
        "schema_version": 1,
        "status": "generation_started",
        "source_commit": receipt["source_commit"],
        "plan_sha256": plan_sha256,
        "rubric_sha256": plan["rubric_sha256"],
        "adapter_sha256": receipt["adapter_sha256"],
        "adapter_receipt_sha256": receipt["receipt_sha256"],
        "runtime_evidence_sha256": runtime_guard["runtime_evidence_sha256"],
        "lifecycle_id": phase_grant["lifecycle_id"],
        "lifecycle_phase_grant_sha256": phase_grant[
            "phase_grant_sha256"
        ],
        "lifecycle_ledger_sequence": phase_grant["ledger_sequence"],
        "lifecycle_grant_id": phase_grant["grant_id"],
        "lifecycle_ledger_commit_id": phase_grant["ledger_commit_id"],
        "generation_signing_public_key_sha256":
            load_generation_trust_policy()["public_key_sha256"],
        "expected_attempts": len(cases) * len(evaluation.VARIANTS),
        "automatic_release_allowed": False,
        "phase_b_ready": False,
    })

    tokenizer = None
    base = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False,
        )
        base = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False,
            dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True,
        )
        base.eval()
    except Exception:
        tokenizer = None
        base = None

    active = base
    adapter_load_failed = False
    with (output / "attempts.jsonl").open("xb") as journal:
        for variant in evaluation.VARIANTS:
            if variant == "trained_adapter" and base is not None:
                try:
                    active = PeftModel.from_pretrained(
                        base, adapter_output / "adapter",
                        local_files_only=True, is_trainable=False,
                    )
                    active.eval()
                except Exception:
                    active = None
                    adapter_load_failed = True
            for case_id in plan["validation_ids"]:
                case = cases[case_id]
                runtime = dict(common_runtime)
                if variant == "trained_adapter":
                    runtime["adapter_sha256"] = receipt["adapter_sha256"]
                row = None
                if tokenizer is not None and active is not None:
                    try:
                        messages = messages_for_variant(case, variant, prompt)
                        inputs = tokenizer.apply_chat_template(
                            messages, tokenize=True, add_generation_prompt=True,
                            return_tensors="pt", enable_thinking=False,
                        ).to("cuda:0")
                        input_count = int(inputs.shape[-1])
                        need(0 < input_count <= recipe["training"]["max_length"])
                        started = time.monotonic_ns()
                        with torch.inference_mode():
                            generated = active.generate(
                                input_ids=inputs,
                                attention_mask=torch.ones_like(inputs),
                                max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=False,
                                use_cache=True,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        elapsed = max(0, (time.monotonic_ns() - started) // 1_000_000)
                        completion = generated[0, input_count:]
                        answer = tokenizer.decode(
                            completion, skip_special_tokens=True
                        ).strip()
                        row = attempt_record(
                            case=case, variant=variant, runtime=runtime,
                            answer=answer, latency_ms=int(elapsed),
                            input_tokens=input_count,
                            output_tokens=int(completion.shape[-1]),
                        )
                    except Exception:
                        row = None
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                if row is None:
                    row = attempt_record(case=case, variant=variant, runtime=runtime)
                bundle["attempts"].append(row)
                _append_attempt(journal, row)

    bundle["runner_attestation"] = create_attestation(
        bundle, generation_signing_key
    )
    _write_json_exclusive(output / "evaluation-bundle.v1.json", bundle)
    validated = evaluation.analyze(
        bundle, adapter_output=adapter_output,
        require_complete=False,
    )
    successes = sum(row["outcome"] == "success" for row in bundle["attempts"])
    return {
        "status": "outputs_recorded_pending_human_scores",
        "output_directory": str(output),
        "attempts": len(bundle["attempts"]),
        "successful_attempts": successes,
        "failed_attempts": len(bundle["attempts"]) - successes,
        "adapter_load_failed": adapter_load_failed,
        "rubric_sha256": plan["rubric_sha256"],
        "comparison_complete": validated["comparison_complete"],
        "actual_model_outputs_evaluated": False,
        "automatic_release_allowed": False,
        "deployment_authorized": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    }


def dry_run() -> dict:
    recipe = load_recipe()
    _, rubric_sha256 = evaluation.load_rubric()
    generation_trust = load_generation_trust_policy()
    return {
        "status": "blocked",
        "expected_attempts": 36,
        "variants": list(evaluation.VARIANTS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "rubric_sha256": rubric_sha256,
        "quota_verified": recipe["account_gates"]["eastus_ncast4_quota_verified"],
        "runtime_compatibility_verified": recipe["account_gates"][
            "runtime_compatibility_verified"
        ],
        "separate_paid_evaluation_confirmation_required": True,
        "external_runner_signing_private_key_required": True,
        "verifier_private_key_access_allowed": False,
        "generation_signing_public_key_pinned": (
            generation_trust["status"] ==
            "runner_signing_public_key_pinned"
        ),
        "model_outputs_generated": False,
        "actual_model_outputs_evaluated": False,
        "deployment_authorized": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        report = execute() if arguments.execute else dry_run()
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception:
        print("kova cosmo evaluation runner rejected", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
