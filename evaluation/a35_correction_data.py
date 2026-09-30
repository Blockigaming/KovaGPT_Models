"""Validate authored correction drafts; never repair or rescore model answers."""

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from evaluation.historical_suite_bridge import load_archived_suite
from evaluation.quality_evidence import canonical, need, strict_json

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config/a35-correction-supplement.v1.json"


def reference_answer(operation: str, inputs: dict):
    """Small authored-label oracles, independent of any pinned suite answers."""
    if operation == "add_multiply":
        return {"answer": (inputs["left"] + inputs["right"]) * inputs["factor"]}
    if operation == "probability":
        blue, total = inputs["blue"], inputs["blue"] + inputs["red"]
        need(2 <= blue < total)
        value = Fraction(blue * (blue - 1), total * (total - 1))
        return {"numerator": value.numerator, "denominator": value.denominator}
    if operation == "rectangle":
        width = Fraction(inputs["perimeter"], 2) - inputs["length"]
        need(width > 0 and width.denominator == 1)
        return {"area": int(width) * inputs["length"]}
    if operation == "lcm":
        return {"answer": math.lcm(*inputs["values"])}
    if operation == "equation":
        value = Fraction(inputs["rhs"] - inputs["offset"], inputs["coefficient"])
        need(value.denominator == 1)
        return {"x": int(value)}
    if operation == "filter_map":
        return [x * inputs["factor"] for x in inputs["values"] if x % inputs["divisor"] == 0]
    if operation == "increment":
        values = dict(inputs["values"])
        values[inputs["target"]] += values[inputs["source"]]
        return values
    if operation == "copy_or_alias":
        original = list(inputs["values"])
        other = original.copy() if inputs["copy"] else original
        other.append(inputs["append"])
        return original
    if operation == "range":
        return list(range(inputs["start"], inputs["stop"], inputs["step"]))
    if operation == "topological":
        todo = {name: set(required) for name, required in inputs["tasks"].items()}
        result = []
        while todo:
            ready = sorted(name for name, required in todo.items() if required <= set(result))
            need(bool(ready))
            name = ready[0]; result.append(name); del todo[name]
        return result
    if operation == "next_stage":
        done = inputs["completed"]
        need(inputs["stages"][:len(done)] == done and len(done) < len(inputs["stages"]))
        return inputs["stages"][len(done)]
    if operation == "odd_between":
        values = [x for x in range(inputs["lower"] + 1, inputs["upper"]) if x % 2]
        need(len(values) == 1)
        return values[0]
    if operation == "median":
        value = statistics.median(inputs["values"])
        need(type(value) is int)
        return {"median": value}
    if operation == "mean":
        value = Fraction(sum(inputs["values"]), len(inputs["values"]))
        need(value.denominator == 1)
        return {"mean": int(value), "count": len(inputs["values"])}
    if operation == "unique":
        return sorted(set(inputs["values"]))
    if operation == "repeated":
        counts = Counter(inputs["values"])
        return sorted(value for value, count in counts.items() if count > 1)
    if operation == "stable_unique":
        return list(dict.fromkeys(inputs["values"]))
    if operation == "null":
        return None
    if operation == "boolean":
        need(type(inputs["value"]) is bool)
        return inputs["value"]
    raise ValueError("unrecognised authored reference operation")


def normalize(prompt: str) -> str:
    return " ".join(re.findall(r"\w+", prompt.casefold()))


def validate_draft() -> dict:
    manifest = strict_json(MANIFEST.read_text())
    for flag in ("training_authorized", "paid_execution_authorized", "prompt_owner_approved",
                 "dataset_owner_approved", "integration_into_training_authorized"):
        need(manifest[flag] is False)
    dataset = ROOT / manifest["dataset_path"]
    prompt = ROOT / manifest["prompt_path"]
    need(hashlib.sha256(dataset.read_bytes()).hexdigest() == manifest["dataset_sha256"])
    need(hashlib.sha256(prompt.read_bytes()).hexdigest() == manifest["prompt_sha256"])
    rows = [strict_json(line) for line in dataset.read_text().splitlines()]
    need(len(rows) == manifest["record_count"])
    ids, prompts = set(), set()
    excluded = {normalize(case["prompt"]) for case in load_archived_suite()["cases"]}
    for original in (ROOT / "data/kova-identity-shared.v2.jsonl").read_text().splitlines():
        excluded.add(normalize(strict_json(original)["messages"][0]["content"]))
    operations = Counter()
    for row in rows:
        need(row["id"] not in ids and row["split"] in ("train", "validation"))
        ids.add(row["id"])
        normal = normalize(row["prompt"])
        need(normal not in excluded and normal not in prompts)
        prompts.add(normal)
        if row["kind"] == "reference_json":
            reference = row["reference"]
            expected = reference_answer(reference["operation"], reference["inputs"])
            need(canonical(strict_json(row["answer"])) == canonical(expected))
            operations[reference["operation"]] += 1
        else:
            need(row["kind"] == "manual_draft" and row["reference"] is None)
            need(bool(row["review_criteria"]) and row["review_complete"] is False)
    need(dict(Counter(row["split"] for row in rows)) == manifest["split_counts"])
    need(set(operations) == set(manifest["reference_operations"]))
    for operation in operations:
        need({row["split"] for row in rows if row["kind"] == "reference_json" and
              row["reference"]["operation"] == operation} == {"train", "validation"})
    return {"records_validated": len(rows), "reference_labels_verified": sum(operations.values()),
            "manual_targets_require_review": len(rows) - sum(operations.values()),
            "exact_prompt_overlap_with_pinned_suite_or_approved_corpus": 0,
            "semantic_overlap_review_complete": False,
            "active_training_dataset_changed": False, "model_calls_made": 0}


if __name__ == "__main__":
    print(json.dumps(validate_draft(), sort_keys=True))
