"""Validate comparable base/configured-base/trained Cosmo evaluation evidence."""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from training import identity_pilot as pilot
from training.cosmo_adapter_receipt import ReceiptError, verify_receipt as verify_adapter_receipt
from training.cosmo_generation_attestation import (
    AttestationError,
    load_trust_policy as load_generation_trust_policy,
    verify_attestation as verify_runner_attestation,
)
from training.kova_cosmo_sft import load_recipe

PLAN_PATH = "config/kova-cosmo-evaluation-plan.v1.json"
RUBRIC_PATH = "config/kova-cosmo-evaluation-rubric.v1.json"
MAX_BYTES = 16 * 1024 * 1024
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
VARIANTS = ("base", "configured_base", "trained_adapter")
DIMENSIONS = (
    "identity", "instruction_adherence", "factuality", "format_adherence",
    "general_quality", "safety_truthfulness",
)


class EvaluationError(ValueError):
    pass


def need(condition: bool) -> None:
    if not condition:
        raise EvaluationError("cosmo evaluation evidence rejected")


def canonical(value: object) -> bytes:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
        need(len(raw) <= MAX_BYTES)
        return raw
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise EvaluationError("cosmo evaluation evidence rejected") from None


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def answer_digest(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def read_json(path: Path) -> object:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        need(0 < len(raw) <= MAX_BYTES)
        return pilot.parse(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeError, RecursionError):
        raise EvaluationError("cosmo evaluation evidence rejected") from None


def load_rubric(root: Path = pilot.ROOT) -> tuple[dict, str]:
    try:
        raw = pilot.read_asset(root, RUBRIC_PATH)
        value = pilot.parse(raw.decode("utf-8"))
        need(type(value) is dict and list(value) == [
            "schema_version", "status", "model_slot", "dimensions",
            "review_rules", "deployment_authorized", "phase_b_ready",
        ])
        need(value["schema_version"] == 1)
        need(value["status"] == "owner_review_protocol_not_release_authority")
        need(value["model_slot"] == "work-cosmo")
        need(type(value["dimensions"]) is dict and
             list(value["dimensions"]) == list(DIMENSIONS))
        for dimension in DIMENSIONS:
            criteria = value["dimensions"][dimension]
            need(type(criteria) is dict and list(criteria) == ["pass", "fail"])
            for verdict in ("pass", "fail"):
                text = criteria[verdict]
                need(type(text) is str and 20 <= len(text) <= 1000)
            need(criteria["pass"] != criteria["fail"])
        need(value["review_rules"] == {
            "score_each_attempt_independently": True,
            "use_case_variant_and_recorded_answer_only": True,
            "brand_keyword_alone_cannot_establish_identity_pass": True,
            "all_dimensions_require_explicit_pass_or_fail": True,
            "failed_or_missing_generation_cannot_be_scored_complete": True,
            "review_receipt_required_for_completion": True,
            "reviewer_label_is_not_verified_identity": True,
            "automatic_release_allowed": False,
        })
        need(value["deployment_authorized"] is False)
        need(value["phase_b_ready"] is False)
        return value, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError, KeyError, UnicodeError,
            RecursionError, AttributeError):
        raise EvaluationError("cosmo evaluation evidence rejected") from None


def load_plan(root: Path = pilot.ROOT) -> tuple[dict, dict[str, dict], str]:
    try:
        identity_plan, _, rows = pilot.load(root)
        recipe = load_recipe(root)
        _, rubric_sha256 = load_rubric(root)
        raw = pilot.read_asset(root, PLAN_PATH)
        plan = pilot.parse(raw.decode("utf-8"))
        validation = [row for row in rows if row["split"] == "validation"]
        expected = {
            "schema_version": 1,
            "status": "awaiting_model_outputs",
            "model_slot": recipe["model_slot"],
            "base_model": recipe["base_model"],
            "base_revision": recipe["base_revision"],
            "dataset_sha256": identity_plan["dataset_sha256"],
            "prompt_sha256": identity_plan["prompt_sha256"],
            "rubric_path": RUBRIC_PATH,
            "rubric_sha256": rubric_sha256,
            "validation_ids": [row["id"] for row in validation],
            "variants": [
                {"id": "base", "kova_system_prompt": False,
                 "adapter_required": False},
                {"id": "configured_base", "kova_system_prompt": True,
                 "adapter_required": False},
                {"id": "trained_adapter", "kova_system_prompt": True,
                 "adapter_required": True},
            ],
            "score_dimensions": list(DIMENSIONS),
            "comparison_rules": {
                "same_cases_required": True,
                "same_runtime_except_adapter_required": True,
                "missing_or_failed_attempt_blocks_completion": True,
                "pending_score_blocks_completion": True,
                "measured_completion_requires_review_receipt": True,
                "automatic_release_allowed": False,
            },
            "actual_model_outputs_evaluated": False,
            "phase_b_ready": False,
        }
        pilot.same(plan, expected)
        return plan, {row["id"]: row for row in validation}, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError, KeyError, UnicodeError,
            RecursionError, AttributeError):
        raise EvaluationError("cosmo evaluation evidence rejected") from None


def analyze(bundle: dict, *, root: Path = pilot.ROOT,
            require_complete: bool = False,
            adapter_output: Path | None = None) -> dict:
    plan, cases, plan_sha256 = load_plan(root)
    recipe = load_recipe(root)
    need(type(bundle) is dict and list(bundle) == [
        "schema_version", "kind", "plan_sha256", "source_commit",
        "adapter_sha256", "adapter_receipt_sha256", "runner_attestation",
        "attempts",
    ])
    need(bundle["schema_version"] == 1)
    need(bundle["kind"] in ("synthetic_fixture", "measured"))
    need(bundle["plan_sha256"] == plan_sha256)
    need(type(bundle["source_commit"]) is str and
         HEX40.fullmatch(bundle["source_commit"]) is not None)
    need(type(bundle["adapter_sha256"]) is str and
         HEX64.fullmatch(bundle["adapter_sha256"]) is not None)
    need(type(bundle["adapter_receipt_sha256"]) is str and
         HEX64.fullmatch(bundle["adapter_receipt_sha256"]) is not None)
    need(type(bundle["attempts"]) is list and len(bundle["attempts"]) <= 36)

    if bundle["kind"] == "measured":
        need(adapter_output is not None and adapter_output.is_absolute())
        try:
            trust = load_generation_trust_policy(root)
            need(trust["status"] ==
                 "runner_signing_public_key_pinned")
            receipt = verify_adapter_receipt(
                adapter_output,
                expected_source_commit=bundle["source_commit"],
                root=root,
            )
            attestation = verify_runner_attestation(
                bundle, public_key_hex_value=trust["public_key_hex"],
            )
        except (ReceiptError, AttestationError, EvaluationError):
            raise EvaluationError("cosmo evaluation evidence rejected") from None
        need(receipt["adapter_sha256"] == bundle["adapter_sha256"])
        need(receipt["receipt_sha256"] == bundle["adapter_receipt_sha256"])
    else:
        need(adapter_output is None)
        need(bundle["runner_attestation"] is None)
        attestation = None

    attempts = {}
    attempt_ids = set()
    runtime_fingerprint = None
    for row in bundle["attempts"]:
        need(type(row) is dict and list(row) == [
            "id", "case_id", "case_sha256", "variant", "outcome",
            "answer", "answer_sha256", "latency_ms", "input_tokens",
            "output_tokens", "runtime", "scores",
        ])
        need(type(row["id"]) is str and 0 < len(row["id"]) <= 256)
        need(row["id"] not in attempt_ids)
        need(row["case_id"] in cases and row["variant"] in VARIANTS)
        need(row["case_sha256"] == digest(cases[row["case_id"]]))
        key = (row["case_id"], row["variant"])
        need(key not in attempts)
        need(row["outcome"] in ("success", "failed"))

        runtime = row["runtime"]
        need(type(runtime) is dict and list(runtime) == [
            "base_model", "base_revision", "adapter_sha256", "adapter_receipt_sha256",
            "software_lock_sha256", "runtime_evidence_sha256",
            "lifecycle_id", "lifecycle_grant_id",
            "lifecycle_ledger_commit_id", "lifecycle_phase_grant_sha256",
            "hardware", "precision", "quantization",
        ])
        need(runtime["base_model"] == recipe["base_model"])
        need(runtime["base_revision"] == recipe["base_revision"])
        need(type(runtime["software_lock_sha256"]) is str and
             HEX64.fullmatch(runtime["software_lock_sha256"]) is not None)
        need(type(runtime["runtime_evidence_sha256"]) is str and
             HEX64.fullmatch(runtime["runtime_evidence_sha256"]) is not None)
        need(type(runtime["lifecycle_phase_grant_sha256"]) is str and
             HEX64.fullmatch(runtime["lifecycle_phase_grant_sha256"])
             is not None)
        for field in ("lifecycle_id", "lifecycle_grant_id",
                      "lifecycle_ledger_commit_id"):
            need(type(runtime[field]) is str and 0 < len(runtime[field]) <= 256)
        if bundle["kind"] == "measured":
            need(runtime["lifecycle_id"] == receipt["lifecycle_id"])
        for field in ("hardware", "precision", "quantization"):
            need(type(runtime[field]) is str and 0 < len(runtime[field]) <= 128)
        expected_adapter = bundle["adapter_sha256"] if row["variant"] == "trained_adapter" else None
        need(runtime["adapter_sha256"] == expected_adapter)
        need(runtime["adapter_receipt_sha256"] == bundle["adapter_receipt_sha256"])
        fingerprint = digest({key: value for key, value in runtime.items()
                              if key != "adapter_sha256"})
        runtime_fingerprint = runtime_fingerprint or fingerprint
        need(fingerprint == runtime_fingerprint)

        need(type(row["scores"]) is dict and list(row["scores"]) == list(DIMENSIONS))
        need(all(value in ("pass", "fail", "pending")
                 for value in row["scores"].values()))
        # Runner-produced measured evidence is generation evidence only.  Human
        # verdicts are accepted exclusively by cosmo_evaluation_review, which
        # binds the original bundle, rubric, score overlay, reviewed bundle and
        # receipt.  This blocks callers from editing scores in-place while
        # retaining a still-valid runner signature.
        if bundle["kind"] == "measured":
            need(all(value == "pending"
                     for value in row["scores"].values()))
        if row["outcome"] == "success":
            need(type(row["answer"]) is str and 0 < len(row["answer"]) <= 750000)
            need(row["answer_sha256"] == answer_digest(row["answer"]))
            for field in ("latency_ms", "input_tokens", "output_tokens"):
                need(type(row[field]) is int and 0 <= row[field] < 2**53)
            need(row["output_tokens"] > 0)
        else:
            need(row["answer"] is None and row["answer_sha256"] is None)
            need(row["latency_ms"] is None and row["input_tokens"] is None and
                 row["output_tokens"] is None)
            need(all(value == "pending" for value in row["scores"].values()))
        attempts[key] = row
        attempt_ids.add(row["id"])

    expected = {(case_id, variant) for case_id in cases for variant in VARIANTS}
    missing = sorted(expected - set(attempts))
    counts = {variant: Counter() for variant in VARIANTS}
    dimension_passes = {variant: {dimension: 0 for dimension in DIMENSIONS}
                        for variant in VARIANTS}
    complete = not missing
    for (_, variant), row in attempts.items():
        counts[variant][row["outcome"]] += 1
        for dimension, verdict in row["scores"].items():
            if verdict == "pass":
                dimension_passes[variant][dimension] += 1
            if verdict == "pending":
                complete = False
        if row["outcome"] != "success":
            complete = False
    if require_complete:
        need(complete)
    return {
        "status": "comparison_complete" if complete else "awaiting_comparable_outputs",
        "kind": bundle["kind"],
        "plan_sha256": plan_sha256,
        "dataset_sha256": plan["dataset_sha256"],
        "prompt_sha256": plan["prompt_sha256"],
        "adapter_sha256": bundle["adapter_sha256"],
        "adapter_receipt_sha256": bundle["adapter_receipt_sha256"],
        "runner_attestation_verified": attestation is not None,
        "review_receipt_verified": False,
        "measurement_sha256": (
            attestation["measurement_sha256"] if attestation else None
        ),
        "expected_attempts": len(expected),
        "provided_attempts": len(attempts),
        "missing_attempts": len(missing),
        "attempt_outcomes_by_variant": {variant: dict(counts[variant])
                                        for variant in VARIANTS},
        "dimension_passes_by_variant": dimension_passes,
        "comparison_complete": complete,
        "actual_model_outputs_evaluated": False,
        "human_reviewer_identity_verified": False,
        "automatic_release_allowed": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--adapter-output", type=Path,
                        help="External adapter run directory required for measured evidence")
    arguments = parser.parse_args(argv)
    try:
        if arguments.bundle is None:
            plan, cases, plan_sha256 = load_plan()
            trust = load_generation_trust_policy()
            report = {
                "status": plan["status"], "plan_sha256": plan_sha256,
                "validation_cases": len(cases), "variants": len(VARIANTS),
                "expected_attempts": len(cases) * len(VARIANTS),
                "actual_model_outputs_evaluated": False,
                "review_receipt_required_for_measured_completion": True,
                "generation_signing_key_pinned": (
                    trust["status"] ==
                    "runner_signing_public_key_pinned"
                ),
                "phase_b_ready": False, "closed_checklist_ids": [],
            }
        else:
            value = read_json(arguments.bundle)
            need(type(value) is dict)
            report = analyze(
                value,
                require_complete=arguments.require_complete,
                adapter_output=arguments.adapter_output,
            )
        print(json.dumps(report, sort_keys=True))
    except (EvaluationError, AttestationError):
        print("cosmo evaluation evidence rejected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
