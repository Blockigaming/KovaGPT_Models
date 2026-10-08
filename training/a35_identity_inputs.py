"""Explicit identity-only edition of historical synthetic inputs; no model calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from core.public_identity import contains_prohibited

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = "data/a35-kovagpt-identity-overrides.v1.jsonl"
REVIEW_PATH = "data/a35-kovagpt-identity-review.v1.json"
PARENT_PACK = "d670792ddf8f38896ee9adcac338268a7677cd4b2c60079a33d80a018cdccdf8"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def need(condition):
    if not condition:
        raise ValueError("KovaGPT identity input contract rejected")


def apply_identity_policy(original, system, file_pins):
    encoded = (json.dumps(original,sort_keys=True,separators=(",",":"))+"\n").encode()
    need(sha(encoded) == PARENT_PACK)
    for name in (DATA_PATH, REVIEW_PATH):
        need(sha((ROOT/name).read_bytes()) == file_pins[name])
    overrides = [json.loads(line) for line in (ROOT/DATA_PATH).read_text().splitlines()]
    review = json.loads((ROOT/REVIEW_PATH).read_text())
    need(review["parent_pack_sha256"] == PARENT_PACK)
    need(review["dataset_sha256"] == sha((ROOT/DATA_PATH).read_bytes()))
    need(review["model_calls_made"] == 0 and review["training_authorized"] is False)
    need(review["human_quality_approval_claimed"] is False)
    mapping = {row["id"]:row for row in overrides}
    annotations = {row["id"]:row for row in review["records"]}
    need(len(mapping) == len(overrides) == len(annotations) == len(review["records"]) == 18)
    need(set(mapping) == set(annotations))
    need(not contains_prohibited(system))
    result = deepcopy(original)
    matched = set()
    for rid,split,row in result:
        # Nothing from synthetic runtime metadata enters the active model prompt.
        row["prompt"][0] = {"role":"system", "content":system}
        if rid in mapping:
            replacement, annotation = mapping[rid], annotations[rid]
            need(set(replacement) == {"id","split","messages"})
            need(replacement["split"] == annotation["split"] == split)
            need(len(replacement["messages"]) == 2)
            question, answer = replacement["messages"]
            need(question == row["prompt"][1] and answer["role"] == "assistant")
            need(set(answer) == {"role","content"} and type(answer["content"]) is str)
            need(sha(question["content"].encode()) == annotation["original_user_sha256"])
            need(sha(row["completion"][0]["content"].encode()) == annotation["original_target_sha256"])
            need(sha(answer["content"].encode()) == annotation["replacement_target_sha256"])
            need(annotation["policy_review"] == "conformant" and annotation["benchmark_answer_added"] is False)
            row["completion"] = [answer]
            matched.add(rid)
        need(not contains_prohibited(row["completion"]))
    need(matched == set(mapping))
    need(len(result) == 134 and sum(split == "train" for _,split,_ in result) == 73)
    return result
