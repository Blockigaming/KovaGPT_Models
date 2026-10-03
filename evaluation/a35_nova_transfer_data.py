"""Nova-only examples for observed failure classes; no model-quality claims.

Each class gains one independently authored train and held-out validation row.
The reference layer validates labels and stays outside model messages. Existing
inputs, prompt intent, benchmark targets and graders are never rewritten.
"""

from collections import Counter
import ast
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from evaluation import a35_correction_data as existing
from evaluation import a35_nova_copy_contrast as copy_data
from evaluation.a35_input_similarity import audit, graph_shape, benchmark_graph
from evaluation.historical_suite_bridge import load_archived_suite
from evaluation.quality_evidence import canonical, need, strict_json

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = "data/a35-nova-transfer.v1.draft.jsonl"
REVIEW_PATH = "data/a35-nova-transfer-review.v1.json"
DATA_SHA256 = "d8e8b606cce9e27572e8844a1518e5e46f1ae53edcb522d6006a4008469b114f"
REVIEW_SHA256 = "1ec53a4b083a0d0b4c2d63c6aaf78a7268b06666fd6baba6f08037f616ccf293"
GROUPS = {"arithmetic", "probability", "geometry", "predicate", "increment",
          "task_ids", "string_root", "integer_root", "boolean_root", "array_root"}


def reference_graph(reference):
    tree = ast.parse(reference["inputs"]["source"])
    graphs = [ast.literal_eval(node.value) for node in tree.body if
              isinstance(node, ast.Assign) and len(node.targets) == 1 and
              isinstance(node.targets[0], ast.Name) and node.targets[0].id == "deps"]
    need(len(graphs) == 1)
    return graphs[0]


def reference_answer(reference):
    operation, inputs = reference["operation"], reference["inputs"]
    if operation == "python_trace":
        need(set(inputs) == {"source"})
        return existing.authored_python(inputs["source"])
    if operation == "ordered_draws":
        need(set(inputs) == {"counts", "sequence"})
        counts = dict(inputs["counts"])
        need(all(type(n) is int and 0 < n <= 20 for n in counts.values()))
        need(0 < len(inputs["sequence"]) <= sum(counts.values()) <= 30)
        probability = Fraction(1)
        for color in inputs["sequence"]:
            need(color in counts and counts[color] > 0)
            probability *= Fraction(counts[color], sum(counts.values()))
            counts[color] -= 1
        return {"numerator": probability.numerator, "denominator": probability.denominator}
    need(operation == "unordered_hits")
    need(set(inputs) == {"eligible", "total", "draws", "hits"})
    e, n, d, h = (inputs[k] for k in ("eligible", "total", "draws", "hits"))
    need(all(type(x) is int for x in (e, n, d, h)))
    need(0 < e < n <= 30 and 0 < d <= n and 0 <= h <= min(e, d) and d-h <= n-e)
    probability = Fraction(math.comb(e, h) * math.comb(n-e, d-h), math.comb(n, d))
    return {"fraction": [probability.numerator, probability.denominator]}


def _pinned(relative, digest):
    body = (ROOT / relative).read_bytes()
    need(hashlib.sha256(body).hexdigest() == digest)
    return body.decode()


def _validated():
    rows = [strict_json(line) for line in _pinned(DATA_PATH, DATA_SHA256).splitlines()]
    review = strict_json(_pinned(REVIEW_PATH, REVIEW_SHA256))
    need(review["schema_version"] == 1 and review["model_calls_made"] == 0)
    need(review["status"] == "owner_delegated_policy_conformance_review")
    need(review["scope"] == "nova_only_additive_draft")
    need(review["provenance"] == {"source": "project_authored_synthetic_transfer_examples",
         "license": "CC0-1.0", "private_customer_data": False, "private_conversations": False,
         "secrets": False, "third_party_training_text": False})
    annotations = {r["id"]: r for r in review["records"]}
    need(len(rows) == len(annotations) == len(review["records"]) == 20)
    need({r["id"] for r in rows} == set(annotations))
    for row in rows:
        need(set(row) == {"id", "split", "messages"})
        annotation = annotations[row["id"]]
        group = annotation["group"]
        need(group in GROUPS and row["split"] in ("train", "validation"))
        need(row["id"] == f'a35-nova-transfer-{group}-{row["split"]}')
        need(annotation["structure"] and annotation["provenance"] == "project_authored_synthetic")
        need(annotation["policy_review"] == "conformant")
        need(annotation["code_fixture"] is None and annotation["criteria"] == [])
        messages = row["messages"]
        need(len(messages) == 2 and [m["role"] for m in messages] == ["user", "assistant"])
        need(all(set(m) == {"role", "content"} and type(m["content"]) is str and m["content"] for m in messages))
        expected = reference_answer(annotation["reference"])
        need(canonical(strict_json(messages[1]["content"])) == canonical(expected))
    for group in GROUPS:
        need(Counter(r["split"] for r in rows if annotations[r["id"]]["group"] == group)
             == {"train": 1, "validation": 1})

    # Reconstruct all old raw records, with the same explicit historical override.
    existing.validate_draft(require_mask_report=False)
    manifest = strict_json(existing.MANIFEST.read_text())
    old_reviews = strict_json((existing.ROOT / manifest["review_path"]).read_text())
    old = [strict_json(line) for line in (existing.ROOT / manifest["dataset_path"]).read_text().splitlines()]
    approved = [strict_json(line) for line in
                (existing.ROOT / "data/kova-identity-shared.v2.jsonl").read_text().splitlines()]
    overrides = {r["id"]: r for r in (strict_json(line) for line in
                 (existing.ROOT / manifest["validation_overrides_path"]).read_text().splitlines())}
    approved = [overrides.get(r["id"], r) for r in approved]
    inherited = [{"id": r["id"], "group": "approved-v2", "structure": r["id"],
                  "reference": None, "code_fixture": None} for r in approved if r["id"] not in overrides]
    copy_rows = copy_data.load_rows()
    copy_review = strict_json((copy_data.ROOT / copy_data.REVIEW_PATH).read_text())["records"]
    combined = old + approved + copy_rows + rows
    need(len(combined) == 132 and len({r["id"] for r in combined}) == 132)
    prompts = [existing.normalize(r["messages"][0]["content"]) for r in combined]
    need(len(set(prompts)) == 132)
    cases = load_archived_suite()["cases"]
    need(not set(prompts) & {existing.normalize(c["prompt"]) for c in cases})
    similarity = audit(combined, old_reviews["records"] + inherited +
                       old_reviews["approved_corpus_overrides"] + copy_review + review["records"], cases)
    need(not similarity["issues"])
    # Explicit name-independent graph check also covers graph tasks represented
    # by Python label oracles. Both new graphs differ structurally from the suite.
    graphs = [reference_graph(r["reference"]) for r in review["records"] if r["group"] == "task_ids"]
    heldout = [benchmark_graph(c["prompt"]) for c in cases]
    heldout = [graph_shape(g) for g in heldout if g]
    need(all(graph_shape(g) not in heldout for g in graphs))
    need(graph_shape(graphs[0]) != graph_shape(graphs[1]))
    return rows, {"records_validated": 20, "reference_labels_verified": 20,
        "failure_classes": sorted(GROUPS), "new_training_records": 10, "new_validation_records": 10,
        "existing_records_unchanged": 110, "copy_contrast_records": 2,
        "structural_similarity": similarity, "model_calls_made": 0,
        "quality_improvement_proved": False}


def load_rows():
    return _validated()[0]


def validate():
    return _validated()[1]


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
