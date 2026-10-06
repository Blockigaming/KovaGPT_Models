"""Offline, inactive curriculum proposal; no model-output repair or execution.

Project-authored labels and process checks are outside the pinned quality suite.
The loader only prepares review inputs. Historical plans, graders, adapters,
identity policy and consumed authorizations are immutable.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from math import prod
from pathlib import Path

from evaluation.a35_correction_data import authored_python, normalize
from evaluation.a35_input_similarity import audit
from evaluation.historical_suite_bridge import load_archived_suite
from evaluation.quality_evidence import canonical, need, strict_json

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config/a35-nova-behavior-correction.v1.json"
GROUPS = {"arithmetic", "probability", "geometry", "equation", "predicate",
          "copy", "increment", "string_root", "median"}


def digest(body):
    return hashlib.sha256(body).hexdigest()


def reference_answer(reference):
    """Exact synthetic-label oracle; never takes model text or suite answers."""
    op, p = reference["operation"], reference["inputs"]
    if op == "python_trace":
        return authored_python(p["source"])
    if op == "ledger":
        subtotals = [base+extra for _,base,extra in p["groups"]]
        incoming = sum(group[0]*subtotal for group,subtotal in zip(p["groups"],subtotals,strict=True))
        balance = p["opening"] + incoming - sum(p["removals"])
        if p["checked"]:
            return {"per_lot_subtotals": subtotals, "incoming": incoming, "removed": sum(p["removals"]),
                    "balance": balance}
        return {p["key"]: balance}
    if op == "ordered_probability":
        counts = dict(p["counts"])
        eligible, population = [], []
        probability = Fraction(1)
        for color in p["sequence"]:
            need(color in counts and counts[color] > 0)
            eligible.append(counts[color]); population.append(sum(counts.values()))
            probability *= Fraction(counts[color], sum(counts.values()))
            counts[color] -= 1
        fraction = [probability.numerator, probability.denominator]
        return {"eligible": eligible, "population": population, "fraction": fraction} if p["checked"] else {"fraction": fraction}
    if op == "geometry":
        perimeter = p["fence"] + p["gate"]
        if "length" in p:
            length, width = Fraction(p["length"]), Fraction(perimeter, 2)-p["length"]
        else:
            a, b = p["ratio"]
            unit = Fraction(perimeter, 2*(a+b)); length, width = a*unit, b*unit
        area = length*width-prod(p["cutout"])
        need(width > 0 and area > 0 and all(v.denominator == 1 for v in (length, width, area)))
        return {"sides": [int(length), int(width)], "perimeter_check": int(2*(length+width)),
                "usable": int(area)} if p["checked"] else {"usable": int(area)}
    if op == "equation":
        a, b, c, d = (p[k] for k in ("a", "b", "c", "d"))
        solution = Fraction(d-b, a-c)
        need(solution.denominator == 1)
        x = int(solution)
        return {p["key"]: x, "left": a*x+b, "right": c*x+d} if p["checked"] else {p["key"]: x}
    raise ValueError("unknown authored operation")


def load_manifest():
    m = strict_json(MANIFEST.read_text())
    need(m["scope"] == "offline_review_only" and m["quality_status"] == "UNMEASURED")
    need(m["model_calls_made"] == m["new_training_runs"] == m["new_allocations"] == 0)
    need(all(m[k] is False for k in ("execution_authorized", "training_authorized",
         "paid_inference_authorized", "deployment_authorized", "merge_authorized",
         "quality_improvement_proved", "consumed_grant_reuse_authorized")))
    need(m["parent_source"] == "f63e24c36f30cf26b584dc903a5df95e16c36e72")
    need(m["preserved_recipe"] == {"epochs": 3, "optimizer_steps": 30, "training_cap_seconds": 600,
         "train_records": 73, "validation_records": 61})
    need(m["quality_gates"] == {"strict": 36, "manual_cases": 14, "manual_criteria": 48,
         "required_repetitions": 3, "strict_aggregate": 108, "averaging_failures_allowed": False,
         "applicable_identity_safety_grounding_failures_allowed": 0})
    for name, pin in m["file_sha256"].items():
        need(digest((ROOT/name).read_bytes()) == pin)
    return m


def load_rows():
    m = load_manifest()
    rows = [strict_json(line) for line in (ROOT/m["dataset_path"]).read_text().splitlines()]
    review = strict_json((ROOT/m["review_path"]).read_text())
    annotations = {r["id"]: r for r in review["records"]}
    need(len(rows) == len(annotations) == 36)
    need(review["provenance"] == {"source": "project_authored_synthetic", "license": "CC0-1.0",
         "third_party_training_text": False, "customer_data": False, "secrets": False})
    need(review["model_calls_made"] == 0 and review["quality_improvement_proved"] is False)
    slots = set()
    for row in rows:
        need(set(row) == {"id", "split", "messages"})
        a = annotations[row["id"]]
        need(a["group"] in GROUPS and a["split"] == row["split"] in ("train", "validation"))
        need(a["replaces"] not in slots); slots.add(a["replaces"])
        need(a["replaces"].endswith("-"+row["split"]))
        need(a["provenance"] == "project_authored_synthetic" and a["policy_review"] == "conformant")
        messages = row["messages"]
        need(len(messages) == 2 and [v["role"] for v in messages] == ["user", "assistant"])
        need(all(set(v) == {"role", "content"} and type(v["content"]) is str and v["content"] for v in messages))
        need(canonical(strict_json(messages[1]["content"])) == canonical(reference_answer(a["reference"])))
    for group in GROUPS:
        need(Counter(r["split"] for r in rows if annotations[r["id"]]["group"] == group)
             == {"train": 2, "validation": 2})
    return rows, annotations


def prepared_rows():
    """Replace only explicit content slots in a separate, disabled review pack."""
    from training.a35_nova_screen import load_plan, prepared_nova_rows
    from training.template_policy import template_row
    from core.public_identity import contains_prohibited
    parent = prepared_nova_rows(load_plan())
    rows, annotations = load_rows()
    replacements = {annotations[r["id"]]["replaces"]: r for r in rows}
    need(set(replacements) <= {rid for rid, _, _ in parent})
    identity = {r["id"] for r in map(strict_json,
        (ROOT/"data/a35-kovagpt-identity-overrides.v1.jsonl").read_text().splitlines())}
    need(not identity & set(replacements))
    result = []
    for rid, split, row in parent:
        if rid in replacements:
            new = replacements[rid]
            need(new["split"] == split)
            row = template_row([deepcopy(row["prompt"][0]), new["messages"][0]], [new["messages"][1]])
            need(not contains_prohibited(row["completion"]))
            rid = new["id"]
        else:
            row = deepcopy(row)
        result.append((rid, split, row))
    need(len(result) == len({rid for rid, _, _ in result}) == 134)
    need(Counter(split for _, split, _ in result) == {"train": 73, "validation": 61})
    need(all(row["chat_template_kwargs"] == {"enable_thinking": False} for _, _, row in result))
    return result


def validate():
    from training.a35_nova_screen import load_plan, prepared_nova_rows
    rows, annotations = load_rows()
    prepared = prepared_rows()
    # Audit the entire proposal, including preserved train/validation content.
    combined = [{"id": rid, "split": split, "messages": [row["prompt"][1], row["completion"][0]]}
                for rid, split, row in prepared]
    reviews = []
    for name in ("data/a35-correction-supplement-review.v1.json",
                 "data/a35-nova-copy-contrast-review.v1.json", "data/a35-nova-transfer-review.v1.json"):
        reviews.extend(strict_json((ROOT/name).read_text())["records"])
    base = {r["id"]: r for r in reviews}
    def review_for(row):
        rid = row["id"]
        return annotations.get(rid) or base.get(rid) or {
            "id": rid, "group": "preserved", "structure": rid, "reference": None, "code_fixture": None}
    kept = [review_for(row) for row in combined]
    cases = load_archived_suite()["cases"]
    need(not {normalize(r["messages"][0]["content"]) for r in combined}
         & {normalize(c["prompt"]) for c in cases})
    similarity = audit(combined, kept, cases)
    # Canonical product-identity/refusal targets intentionally repeat across
    # historical splits. Preserve that policy and disclose these inherited
    # pairs; never suppress a new overlap or change the audit/evaluator.
    parent = [{"id": rid, "split": split, "messages": [row["prompt"][1], row["completion"][0]]}
              for rid, split, row in prepared_nova_rows(load_plan())]
    inherited = audit(parent, [review_for(row) for row in parent], cases)
    identity_ids = {r["id"] for r in map(strict_json,
        (ROOT/"data/a35-kovagpt-identity-overrides.v1.jsonl").read_text().splitlines())}
    need(all(i["kind"] == "split_long_answer_copy" and
         i["id"] in identity_ids and i["other"] in identity_ids for i in inherited["issues"]))
    issue_key = lambda issue: tuple(sorted(issue.items()))
    need({issue_key(i) for i in similarity["issues"]} == {issue_key(i) for i in inherited["issues"]})
    encoded = (json.dumps(prepared, sort_keys=True, separators=(",", ":"))+"\n").encode()
    return {"status": "pass", "source_checks_only": True, "model_calls_made": 0,
            "quality_status": "UNMEASURED", "quality_improvement_proved": False,
            "synthetic_rows_verified": len(rows), "replaced_train": 18, "replaced_validation": 18,
            "preserved_rows": 98, "prepared_records": 134, "train_records": 73, "validation_records": 61,
            "optimizer_steps_at_existing_recipe": 30, "prepared_pack_sha256": digest(encoded),
            "similarity": similarity, "new_similarity_issues": 0,
            "preserved_identity_answer_duplicate_pairs": len(inherited["issues"]),
            "tokenizer_and_real_trl_masks": load_manifest()["tokenizer_and_real_trl_masks"],
            "execution_authorized": False}


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
