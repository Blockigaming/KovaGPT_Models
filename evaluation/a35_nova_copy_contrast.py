"""Validate a Nova-only additive copy/alias coverage hypothesis, without models.

The measured failure does not isolate training-data causality. These independent
examples add an outer-copy/alias contrast; they neither repair model responses
nor change the existing shared corpus, strict targets or acceptance thresholds.
"""

import hashlib
import json
from pathlib import Path

from evaluation import a35_correction_data as existing
from evaluation.a35_input_similarity import audit
from evaluation.historical_suite_bridge import load_archived_suite
from evaluation.quality_evidence import canonical, need, strict_json

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = "data/a35-nova-copy-contrast.v1.draft.jsonl"
REVIEW_PATH = "data/a35-nova-copy-contrast-review.v1.json"
DATA_SHA256 = "bd5415471df1a8919bbea5dda1029fe1e1010eb10f9c3ef12d7463e6e77aa1dd"
REVIEW_SHA256 = "e02119fbdf99dbfd3efbb5a2628066c2359aace8495acc200a9badc692850f07"
SPLITS = {"a35-nova-copy-isolation-train": "train",
          "a35-nova-copy-isolation-validation": "validation"}


def _read_pinned(path, expected):
    raw = (ROOT / path).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == expected)
    return raw.decode("utf-8")


def _validated():
    rows = [strict_json(line) for line in _read_pinned(DATA_PATH, DATA_SHA256).splitlines()]
    review = strict_json(_read_pinned(REVIEW_PATH, REVIEW_SHA256))
    need(review["schema_version"] == 1)
    need(review["status"] == "owner_delegated_policy_conformance_review")
    need(review["scope"] == "nova_only_additive_draft" and review["model_calls_made"] == 0)
    need(review["provenance"] == {
        "source": "project_authored_synthetic_copy_contrast_examples", "license": "CC0-1.0",
        "private_customer_data": False, "private_conversations": False,
        "secrets": False, "third_party_training_text": False})
    records = review["records"]
    need(len(rows) == len(records) == 2)
    need({r["id"] for r in rows} == {r["id"] for r in records} == set(SPLITS))
    annotations = {r["id"]: r for r in records}
    for row in rows:
        need(set(row) == {"id", "split", "messages"})
        need(row["split"] == SPLITS[row["id"]])
        messages = row["messages"]
        need(len(messages) == 2 and [m["role"] for m in messages] == ["user", "assistant"])
        need(all(set(m) == {"role", "content"} and type(m["content"]) is str and m["content"]
                 for m in messages))
        annotation = annotations[row["id"]]
        need(set(annotation) == {"id", "group", "structure", "reference", "code_fixture",
                                 "criteria", "policy_review", "provenance"})
        need(annotation["group"] == "copy" and annotation["structure"])
        need(annotation["policy_review"] == "conformant")
        need(annotation["provenance"] == "project_authored_synthetic")
        need(annotation["code_fixture"] is None and annotation["criteria"] == [])
        reference = annotation["reference"]
        need(set(reference) == {"operation", "inputs"} and reference["operation"] == "python_trace")
        need(set(reference["inputs"]) == {"source"})
        expected = existing.authored_python(reference["inputs"]["source"])
        actual = strict_json(messages[1]["content"])
        need(type(actual) is list and canonical(actual) == canonical(expected))

    # Reuse the original immutable corpus and its explicit approved override.
    # Review/reference metadata remains outside all returned model messages.
    existing.validate_draft(require_mask_report=False)
    manifest = strict_json(existing.MANIFEST.read_text())
    old_reviews = strict_json((existing.ROOT / manifest["review_path"]).read_text())
    supplement = [strict_json(line) for line in
                  (existing.ROOT / manifest["dataset_path"]).read_text().splitlines()]
    approved = [strict_json(line) for line in
                (existing.ROOT / "data/kova-identity-shared.v2.jsonl").read_text().splitlines()]
    overrides = {r["id"]: r for r in (strict_json(line) for line in
                 (existing.ROOT / manifest["validation_overrides_path"]).read_text().splitlines())}
    approved = [overrides.get(r["id"], r) for r in approved]
    inherited = [{"id": r["id"], "group": "approved-v2", "structure": r["id"],
                  "reference": None, "code_fixture": None}
                 for r in approved if r["id"] not in overrides]
    combined = supplement + approved + rows
    need(len({r["id"] for r in combined}) == len(combined))
    prompts = [existing.normalize(r["messages"][0]["content"]) for r in combined]
    need(len(set(prompts)) == len(prompts))
    suite = load_archived_suite()
    need(not set(prompts) & {existing.normalize(c["prompt"]) for c in suite["cases"]})
    similarity = audit(combined, old_reviews["records"] + inherited +
                       old_reviews["approved_corpus_overrides"] + records, suite["cases"])
    need(not similarity["issues"])
    return rows, {"records_validated": 2, "reference_labels_verified": 2,
                  "new_training_records": 1, "new_validation_records": 1,
                  "existing_records_unchanged": len(combined) - 2,
                  "structural_similarity": similarity, "model_calls_made": 0,
                  "quality_improvement_proved": False}


def load_rows():
    """Return only two validated raw user/assistant records, never review fields."""
    return _validated()[0]


def validate():
    """Return source validation evidence; this is not a model-quality score."""
    return _validated()[1]


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
