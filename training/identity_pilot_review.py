"""Validate the human review ledger bound to the Kova identity pilot corpus."""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import re
import sys

from training import identity_pilot as pilot

REVIEW_PATH = pilot.REVIEW_PATH
CRITERIA = [
    "correct",
    "useful",
    "kova_identity_consistent",
    "truthful",
    "safe",
    "style_appropriate",
]
TIMESTAMP = re.compile(r"20\d\d-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])T"
                       r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\dZ")


class ReviewError(ValueError):
    pass


def need(condition: bool) -> None:
    if not condition:
        raise ReviewError("identity pilot review rejected")


def validate(root: Path = pilot.ROOT) -> dict:
    try:
        plan, _, source = pilot.load(root)
        review_bytes = pilot.read_asset(root, REVIEW_PATH)
        review = pilot.parse(review_bytes.decode("utf-8"))
        need(type(review) is dict and list(review) == [
            "schema_version", "status", "dataset_sha256", "criteria",
            "reviewer", "reviewed_at", "records", "human_review_complete",
        ])
        need(review["schema_version"] == 1)
        need(review["dataset_sha256"] == plan["dataset_sha256"])
        need(review["criteria"] == CRITERIA)
        need(type(review["records"]) is list)
        need(len(review["records"]) == len(source))
        need([item["id"] for item in review["records"]] ==
             [item["id"] for item in source])

        for item in review["records"]:
            need(type(item) is dict and list(item) == ["id", "verdict", "notes"])
            need(type(item["id"]) is str and 0 < len(item["id"]) <= 128)
            need(item["verdict"] in ("pending", "approve", "reject"))
            need(item["notes"] is None or
                 (type(item["notes"]) is str and 0 < len(item["notes"]) <= 2000))

        counts = Counter(item["verdict"] for item in review["records"])
        reviewer = review["reviewer"]
        reviewed_at = review["reviewed_at"]
        if counts == {"pending": len(source)}:
            expected_status = "awaiting_owner_review"
            complete = False
            need(reviewer is None and reviewed_at is None)
        elif counts["pending"]:
            expected_status = "in_progress"
            complete = False
            need(type(reviewer) is str and 0 < len(reviewer) <= 256)
            need(reviewed_at is None)
        else:
            expected_status = "changes_requested" if counts["reject"] else "approved"
            complete = counts["approve"] == len(source)
            need(type(reviewer) is str and 0 < len(reviewer) <= 256)
            need(type(reviewed_at) is str and TIMESTAMP.fullmatch(reviewed_at) is not None)
        need(review["status"] == expected_status)
        need(type(review["human_review_complete"]) is bool)
        need(review["human_review_complete"] is complete)
        need(plan["review_path"] == REVIEW_PATH)
        need(plan["review_sha256"] == pilot.digest(review_bytes))
        # This draft deliberately cannot promote the training plan merely by
        # changing review fields; source authorization remains separately gated.
        need(plan["provenance"]["human_review_complete"] is False)
        return {
            "status": expected_status,
            "review_sha256": plan["review_sha256"],
            "dataset_sha256": plan["dataset_sha256"],
            "records": len(source),
            "verdict_counts": {key: counts[key]
                               for key in ("pending", "approve", "reject")},
            "human_review_complete": complete,
            "training_authorized": False,
            "model_weights_downloaded": False,
            "training_started": False,
            "phase_b_ready": False,
            "closed_checklist_ids": [],
        }
    except (OSError, ValueError, TypeError, KeyError, UnicodeError,
            RecursionError, AttributeError):
        raise ReviewError("identity pilot review rejected") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        report = validate()
        if arguments.require_complete and not report["human_review_complete"]:
            raise ReviewError("identity pilot review incomplete")
        print(json.dumps(report, sort_keys=True))
    except ReviewError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
