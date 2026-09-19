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
from training.kova_cosmo_sft import load_recipe

PLAN_PATH = "config/kova-cosmo-evaluation-plan.v1.json"
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


def load_plan(root: Path = pilot.ROOT) -> tuple[dict, dict[str, dict], str]:
    try:
        identity_plan, _, rows = pilot.load(root)
        recipe = load_recipe(root)
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
            require_complete: bool = False) -> dict:
    plan, cases, plan_sha256 = load_plan(root)
    recipe = load_recipe(root)
    need(type(bundle) is dict and list(bundle) == [
        "schema_version", "kind", "plan_sha256", "source_commit",
        "adapter_sha256", "attempts",
    ])
    need(bundle["schema_version"] == 1)
    need(bundle["kind"] in ("synthetic_fixture", "measured"))
    need(bundle["plan_sha256"] == plan_sha256)
    need(type(bundle["source_commit"]) is str and
         HEX40.fullmatch(bundle["source_commit"]) is not None)
    need(type(bundle["adapter_sha256"]) is str and
         HEX64.fullmatch(bundle["adapter_sha256"]) is not None)
    need(type(bundle["attempts"]) is list and len(bundle["attempts"]) <= 36)

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
            "base_model", "base_revision", "adapter_sha256",
            "software_lock_sha256", "hardware", "precision", "quantization",
        ])
        need(runtime["base_model"] == recipe["base_model"])
        need(runtime["base_revision"] == recipe["base_revision"])
        need(type(runtime["software_lock_sha256"]) is str and
             HEX64.fullmatch(runtime["software_lock_sha256"]) is not None)
        for field in ("hardware", "precision", "quantization"):
            need(type(runtime[field]) is str and 0 < len(runtime[field]) <= 128)
        expected_adapter = bundle["adapter_sha256"] if row["variant"] == "trained_adapter" else None
        need(runtime["adapter_sha256"] == expected_adapter)
        fingerprint = digest({key: value for key, value in runtime.items()
                              if key != "adapter_sha256"})
        runtime_fingerprint = runtime_fingerprint or fingerprint
        need(fingerprint == runtime_fingerprint)

        need(type(row["scores"]) is dict and list(row["scores"]) == list(DIMENSIONS))
        need(all(value in ("pass", "fail", "pending")
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
        "expected_attempts": len(expected),
        "provided_attempts": len(attempts),
        "missing_attempts": len(missing),
        "attempt_outcomes_by_variant": {variant: dict(counts[variant])
                                        for variant in VARIANTS},
        "dimension_passes_by_variant": dimension_passes,
        "comparison_complete": complete,
        "actual_model_outputs_evaluated": complete and bundle["kind"] == "measured",
        "human_reviewer_identity_verified": False,
        "automatic_release_allowed": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        if arguments.bundle is None:
            plan, cases, plan_sha256 = load_plan()
            report = {
                "status": plan["status"], "plan_sha256": plan_sha256,
                "validation_cases": len(cases), "variants": len(VARIANTS),
                "expected_attempts": len(cases) * len(VARIANTS),
                "actual_model_outputs_evaluated": False,
                "phase_b_ready": False, "closed_checklist_ids": [],
            }
        else:
            value = read_json(arguments.bundle)
            need(type(value) is dict)
            report = analyze(value, require_complete=arguments.require_complete)
        print(json.dumps(report, sort_keys=True))
    except EvaluationError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
