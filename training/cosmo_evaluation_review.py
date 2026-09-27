"""Apply a score-only human review to immutable Cosmo generation evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from training import cosmo_sft_evaluation as evaluation
from training import identity_pilot as pilot
from training.cosmo_evaluation_runner import external_existing, external_new

MAX_BUNDLE_BYTES = 16 * 1024 * 1024
MAX_REVIEW_BYTES = 1024 * 1024
REVIEWED_BUNDLE_NAME = "reviewed-evaluation-bundle.v1.json"
REVIEW_RECEIPT_NAME = "review-receipt.v1.json"
ATTESTATION = "I reviewed every recorded answer against the declared rubric."


class ReviewError(ValueError):
    pass


def need(condition: bool) -> None:
    if not condition:
        raise ReviewError("kova cosmo evaluation review rejected")


def read_raw(path: Path, maximum: int) -> bytes:
    try:
        need(path.is_absolute() and path.is_file() and not path.is_symlink())
        resolved = path.resolve(strict=True)
        repository = pilot.ROOT.resolve(strict=True)
        need(resolved.is_file())
        need(repository != resolved and repository not in resolved.parents)
        with resolved.open("rb") as stream:
            raw = stream.read(maximum + 1)
        need(0 < len(raw) <= maximum)
        return raw
    except OSError:
        raise ReviewError("kova cosmo evaluation review rejected") from None


def parse(raw: bytes) -> dict:
    try:
        value = pilot.parse(raw.decode("utf-8"))
        need(type(value) is dict)
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ReviewError("kova cosmo evaluation review rejected") from None


def apply_scores(bundle: dict, review: dict, generation_sha256: str,
                 rubric_sha256: str) -> dict:
    need(type(bundle) is dict and bundle.get("kind") == "measured")
    attempts = bundle.get("attempts")
    need(type(attempts) is list and len(attempts) == 36)
    need(all(type(row) is dict and row.get("outcome") == "success"
             for row in attempts))
    need(all(type(row.get("scores")) is dict and
             list(row["scores"]) == list(evaluation.DIMENSIONS) and
             all(value == "pending" for value in row["scores"].values())
             for row in attempts))
    need(type(review) is dict and list(review) == [
        "schema_version", "kind", "generation_bundle_sha256", "rubric_sha256",
        "reviewer_label", "attestation", "scores",
    ])
    need(review["schema_version"] == 1)
    need(review["kind"] == "human_score_overlay")
    need(review["generation_bundle_sha256"] == generation_sha256)
    need(review["rubric_sha256"] == rubric_sha256)
    need(type(review["reviewer_label"]) is str and
         0 < len(review["reviewer_label"].strip()) <= 256)
    need(review["attestation"] == ATTESTATION)
    scores = review["scores"]
    need(type(scores) is list and len(scores) == len(attempts))

    scored = deepcopy(bundle)
    for attempt, row, target in zip(attempts, scores, scored["attempts"]):
        need(type(row) is dict and list(row) == ["attempt_id", *evaluation.DIMENSIONS])
        need(row["attempt_id"] == attempt["id"])
        verdicts = {dimension: row[dimension]
                    for dimension in evaluation.DIMENSIONS}
        need(all(verdict in ("pass", "fail") for verdict in verdicts.values()))
        target["scores"] = verdicts
    return scored


def _serialize(value: object) -> bytes:
    return (json.dumps(
        value, indent=2, ensure_ascii=True, allow_nan=False
    ) + "\n").encode("ascii")


def _write(path: Path, value: object) -> bytes:
    raw = _serialize(value)
    with path.open("xb") as stream:
        stream.write(raw)
    return raw


def _completed_report(original: dict, scored: dict) -> dict:
    need(original["kind"] == "measured")
    need(original["runner_attestation_verified"] is True)
    need(original["review_receipt_verified"] is False)
    need(original["comparison_complete"] is False)
    attempts = scored["attempts"]
    need(type(attempts) is list and len(attempts) == 36)
    passes = {
        variant: {dimension: 0 for dimension in evaluation.DIMENSIONS}
        for variant in evaluation.VARIANTS
    }
    for row in attempts:
        need(row["outcome"] == "success")
        need(row["variant"] in evaluation.VARIANTS)
        need(type(row["scores"]) is dict and
             list(row["scores"]) == list(evaluation.DIMENSIONS))
        need(all(value in ("pass", "fail")
                 for value in row["scores"].values()))
        for dimension, verdict in row["scores"].items():
            passes[row["variant"]][dimension] += int(verdict == "pass")
    report = dict(original)
    report.update({
        "status": "comparison_complete",
        "dimension_passes_by_variant": passes,
        "comparison_complete": True,
        "actual_model_outputs_evaluated": True,
        "review_receipt_verified": True,
        "human_reviewer_identity_verified": False,
        "automatic_release_allowed": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    })
    return report


def verify_finalized(generation_path: Path, review_path: Path,
                     adapter_output: Path, output: Path) -> dict:
    """Verify the complete hash-bound review chain before completion."""
    try:
        generation_raw = read_raw(generation_path, MAX_BUNDLE_BYTES)
        review_raw = read_raw(review_path, MAX_REVIEW_BYTES)
        generation = parse(generation_raw)
        overlay = parse(review_raw)
        destination = external_existing(str(output))
        children = sorted(destination.iterdir(), key=lambda item: item.name)
        need([item.name for item in children] == [
            REVIEW_RECEIPT_NAME, REVIEWED_BUNDLE_NAME,
        ])
        need(all(item.is_file() and not item.is_symlink()
                 for item in children))
        reviewed_raw = read_raw(
            destination / REVIEWED_BUNDLE_NAME, MAX_BUNDLE_BYTES
        )
        receipt_raw = read_raw(
            destination / REVIEW_RECEIPT_NAME, MAX_REVIEW_BYTES
        )
        reviewed = parse(reviewed_raw)
        receipt = parse(receipt_raw)
        generation_sha256 = hashlib.sha256(generation_raw).hexdigest()
        score_overlay_sha256 = hashlib.sha256(review_raw).hexdigest()
        reviewed_sha256 = hashlib.sha256(reviewed_raw).hexdigest()
        _, rubric_sha256 = evaluation.load_rubric()
        adapter = external_existing(str(adapter_output))
        original_report = evaluation.analyze(
            generation, adapter_output=adapter,
            require_complete=False,
        )
        need(original_report["actual_model_outputs_evaluated"] is False)
        need(original_report["review_receipt_verified"] is False)
        expected_reviewed = apply_scores(
            generation, overlay, generation_sha256, rubric_sha256
        )
        need(evaluation.canonical(reviewed) ==
             evaluation.canonical(expected_reviewed))
        need(reviewed_raw == _serialize(expected_reviewed))
        need(type(receipt) is dict and list(receipt) == [
            "schema_version", "kind", "generation_bundle_sha256",
            "rubric_sha256", "score_overlay_sha256",
            "reviewed_bundle_sha256", "runner_measurement_sha256",
            "reviewer_label", "comparison_complete",
            "actual_model_outputs_evaluated",
            "human_reviewer_identity_verified", "automatic_release_allowed",
            "deployment_authorized", "phase_b_ready",
        ])
        need(receipt["schema_version"] == 1)
        need(receipt["kind"] == "kova_cosmo_human_score_receipt")
        need(receipt["generation_bundle_sha256"] == generation_sha256)
        need(receipt["rubric_sha256"] == rubric_sha256)
        need(receipt["score_overlay_sha256"] == score_overlay_sha256)
        need(receipt["reviewed_bundle_sha256"] == reviewed_sha256)
        need(receipt["runner_measurement_sha256"] ==
             original_report["measurement_sha256"])
        need(receipt["reviewer_label"] == overlay["reviewer_label"])
        need(receipt["comparison_complete"] is True)
        need(receipt["actual_model_outputs_evaluated"] is True)
        need(receipt["human_reviewer_identity_verified"] is False)
        need(receipt["automatic_release_allowed"] is False)
        need(receipt["deployment_authorized"] is False)
        need(receipt["phase_b_ready"] is False)
        need(receipt_raw == _serialize(receipt))
        report = _completed_report(original_report, reviewed)
        report["review_receipt_sha256"] = hashlib.sha256(
            receipt_raw
        ).hexdigest()
        report["reviewed_bundle_sha256"] = reviewed_sha256
        report["score_overlay_sha256"] = score_overlay_sha256
        return report
    except ReviewError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            UnicodeError, RecursionError):
        raise ReviewError("kova cosmo evaluation review rejected") from None


def finalize(generation_path: Path, review_path: Path, adapter_output: Path,
             output: Path) -> dict:
    try:
        generation_raw = read_raw(generation_path, MAX_BUNDLE_BYTES)
        review_raw = read_raw(review_path, MAX_REVIEW_BYTES)
        bundle = parse(generation_raw)
        review = parse(review_raw)
        generation_sha256 = hashlib.sha256(generation_raw).hexdigest()
        _, rubric_sha256 = evaluation.load_rubric()
        adapter = external_existing(str(adapter_output))
        original_report = evaluation.analyze(
            bundle, adapter_output=adapter,
            require_complete=False,
        )
        need(original_report["comparison_complete"] is False)
        scored = apply_scores(
            bundle, review, generation_sha256, rubric_sha256
        )
        destination = external_new(str(output))
        need(adapter != destination and adapter not in destination.parents)
        destination.mkdir(mode=0o700)
        reviewed_raw = _write(
            destination / REVIEWED_BUNDLE_NAME, scored
        )
        receipt = {
            "schema_version": 1,
            "kind": "kova_cosmo_human_score_receipt",
            "generation_bundle_sha256": generation_sha256,
            "rubric_sha256": rubric_sha256,
            "score_overlay_sha256": hashlib.sha256(review_raw).hexdigest(),
            "reviewed_bundle_sha256": hashlib.sha256(reviewed_raw).hexdigest(),
            "runner_measurement_sha256": original_report[
                "measurement_sha256"
            ],
            "reviewer_label": review["reviewer_label"],
            "comparison_complete": True,
            "actual_model_outputs_evaluated": True,
            "human_reviewer_identity_verified": False,
            "automatic_release_allowed": False,
            "deployment_authorized": False,
            "phase_b_ready": False,
        }
        _write(destination / REVIEW_RECEIPT_NAME, receipt)
        verified = verify_finalized(
            generation_path, review_path, adapter, destination,
        )
        need(verified["actual_model_outputs_evaluated"] is True)
        return {
            "status": "human_scores_recorded_release_withheld",
            "output_directory": str(destination),
            "reviewed_attempts": len(scored["attempts"]),
            "generation_bundle_sha256": generation_sha256,
            "reviewed_bundle_sha256": receipt["reviewed_bundle_sha256"],
            "review_receipt_sha256": verified["review_receipt_sha256"],
            "review_receipt_verified": True,
            "comparison_complete": True,
            "actual_model_outputs_evaluated": True,
            "human_reviewer_identity_verified": False,
            "automatic_release_allowed": False,
            "deployment_authorized": False,
            "phase_b_ready": False,
            "closed_checklist_ids": [],
        }
    except ReviewError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            UnicodeError, RecursionError):
        raise ReviewError("kova cosmo evaluation review rejected") from None


def dry_run() -> dict:
    _, rubric_sha256 = evaluation.load_rubric()
    return {
        "status": "awaiting_measured_generation_bundle_and_human_scores",
        "expected_attempts": 36,
        "score_dimensions": list(evaluation.DIMENSIONS),
        "rubric_sha256": rubric_sha256,
        "verifier_uses_source_pinned_public_key_only": True,
        "verifier_private_key_access_allowed": False,
        "answer_editing_allowed": False,
        "review_receipt_required_for_completion": True,
        "human_reviewer_identity_verified": False,
        "automatic_release_allowed": False,
        "deployment_authorized": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation", nargs="?", type=Path)
    parser.add_argument("review", nargs="?", type=Path)
    parser.add_argument("adapter_output", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--verify-finalized", action="store_true",
                        help="Verify an existing reviewed output directory")
    arguments = parser.parse_args(argv)
    try:
        values = (arguments.generation, arguments.review,
                  arguments.adapter_output, arguments.output)
        if all(value is None for value in values):
            need(arguments.verify_finalized is False)
            report = dry_run()
        else:
            need(all(value is not None for value in values))
            action = verify_finalized if arguments.verify_finalized else finalize
            report = action(*values)
        print(json.dumps(report, sort_keys=True))
        return 0
    except ReviewError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
