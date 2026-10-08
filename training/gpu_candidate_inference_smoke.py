"""Verify the three archived T4 experiments and sample local adapted answers.

This operator-only tool never downloads weights, contacts Azure, selects a pilot,
or enables an application route. Generation requires an explicit local archive,
an already downloaded pinned snapshot, and a pinned offline T4 runtime.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import zipfile

from training import cosmo_qlora_training as training
from training import three_family_contract as contract
from training.candidate_inference_smoke import PROMPTS
from training.snapshot_verifier import verify_snapshot


ROOT = Path(__file__).resolve().parents[1]
MAX_ARCHIVE_BYTES = 40 * 1024 * 1024
MEMBERS = {"adapter/adapter_config.json", "adapter/adapter_model.safetensors",
           "adapter/README.md", "receipt.json"}
# These are the independent Azure readback digests from the GPU experiment.
# They identify experimental bytes, not selected or release-approved adapters.
ARCHIVES = {
    "kova-cosmo": (20245111,
                   "452aedd9b19a580799865e7c960e1037931a98cc8b225b4d2283b18ce958f815",
                   "240070e7db0cead3ce912780807f3091466bdf97"),
    "kova-orion": (34925359,
                   "0d76a2551e30910d28a20707bf6f92d8d5e37db25b98e2a92cd27894b2f05482",
                   "240070e7db0cead3ce912780807f3091466bdf97"),
    "kova-nova": (33104769,
                  "e87060e984c10e636a59d9e55dfaee1cb52ed47930dfaf2bf5b791435cbdf21b",
                  "288b7dd08c076bdbeaf7f2114049ae9fa4c3f439"),
}


def _require(condition):
    if not condition:
        raise ValueError("experimental GPU archive rejected")


def _json(raw):
    def unique(pairs):
        value = {}
        for name, item in pairs:
            _require(name not in value)
            value[name] = item
        return value

    def reject(_):
        raise ValueError("experimental GPU archive rejected")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=reject,
                           parse_float=lambda text: float(text) if math.isfinite(float(text))
                           else reject(text))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("experimental GPU archive rejected") from exc
    _require(type(value) is dict)
    return value


def inspect_archive(family: str, archive_path: Path) -> tuple[dict, dict[str, bytes]]:
    """Check published ZIP bytes, exact members, receipt and adapter settings.

    The published digest is a source pin for an experiment; this does not
    replace independent signing or approval for a selected production artifact.
    """
    _require(family in ARCHIVES and not archive_path.is_symlink())
    expected_bytes, expected_sha, source_commit = ARCHIVES[family]
    info = archive_path.stat()
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
             and 0 < info.st_size <= MAX_ARCHIVE_BYTES
             and info.st_size == expected_bytes)
    descriptor = os.open(archive_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        current = os.fstat(descriptor)
        _require(stat.S_ISREG(current.st_mode) and current.st_nlink == 1
                 and current.st_dev == info.st_dev and current.st_ino == info.st_ino
                 and current.st_size == expected_bytes)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(MAX_ARCHIVE_BYTES + 1)
        _require(len(raw) == expected_bytes and os.fstat(descriptor).st_size == expected_bytes)
    finally:
        os.close(descriptor)
    _require(hashlib.sha256(raw).hexdigest() == expected_sha)
    contents = {}
    try:
        with zipfile.ZipFile(BytesIO(raw)) as bundle:
            members = bundle.infolist()
            _require(len(members) == len(MEMBERS)
                     and {entry.filename for entry in members} == MEMBERS)
            _require(sum(entry.file_size for entry in members) <= MAX_ARCHIVE_BYTES)
            for entry in members:
                mode = entry.external_attr >> 16
                limit = (MAX_ARCHIVE_BYTES if entry.filename.endswith(".safetensors")
                         else 1024 * 1024)
                _require(entry.flag_bits & 1 == 0
                         and entry.compress_type == zipfile.ZIP_STORED
                         and entry.compress_size == entry.file_size
                         and 0 < entry.file_size <= limit
                         and (entry.create_system != 3 or stat.S_ISREG(mode)))
                with bundle.open(entry) as stream:
                    contents[entry.filename] = stream.read(limit + 1)
                _require(len(contents[entry.filename]) == entry.file_size)
    except (OSError, zipfile.BadZipFile, RuntimeError, EOFError, NotImplementedError) as exc:
        raise ValueError("experimental GPU archive rejected") from exc
    receipt = _json(contents["receipt.json"])
    config = _json(contents["adapter/adapter_config.json"])
    lineage = contract.load_json(ROOT / "config/kova-private-lineage.v1.json")
    recipe = contract.load_json(ROOT / f"config/{family}-qlora.v1.json")
    dataset = contract.load_json(ROOT / "config/kova-three-family-dataset.v2.json")
    hashes = receipt.get("adapter_sha256")
    # The pinned Nova experiment's receipt predates the `family` field.
    # Its exact ZIP digest, family-specific kind, source and base are still pinned.
    _require((receipt.get("family") == family or
              (family == "kova-nova" and "family" not in receipt))
             and receipt.get("kind") == f"{family.replace('-', '_')}_gpu_t4_nf4_lora_experiment"
             and receipt.get("status") == "complete"
             and receipt.get("source_commit") == source_commit
             and receipt.get("base_revision") == lineage["families"][family]["immutable_revision"]
             and receipt.get("dataset_sha256") == dataset["dataset_sha256"]
             and type(receipt.get("optimizer_steps")) is int
             and receipt["optimizer_steps"] == 7
             and type(receipt.get("train_records")) is int
             and receipt["train_records"] == 27
             and type(receipt.get("validation_records")) is int
             and receipt["validation_records"] == 15
             and receipt.get("signed_controller_pilot") is False
             and receipt.get("production_routing_approved") is False
             and type(hashes) is dict
             and set(hashes) == {"adapter_config.json", "adapter_model.safetensors"})
    for name, expected in hashes.items():
        _require(type(expected) is str and contract.HEX64.fullmatch(expected)
                 and hashlib.sha256(contents["adapter/" + name]).hexdigest() == expected)
    lora = recipe["lora"]
    targets = config.get("target_modules")
    _require(type(targets) in (list, tuple) and all(type(item) is str for item in targets))
    _require(config.get("peft_type") == "LORA"
             and config.get("task_type") == "CAUSAL_LM"
             and type(config.get("r")) is int and config["r"] == lora["rank"]
             and type(config.get("lora_alpha")) in (int, float)
             and config["lora_alpha"] == lora["alpha"]
             and type(config.get("lora_dropout")) in (int, float)
             and config["lora_dropout"] == lora["dropout"]
             and config.get("bias") == lora["bias"]
             and set(targets) == set(lora["target_modules"])
             and config.get("auto_mapping") in (None, {}))
    return receipt, contents


def _quality_results(cases: list[dict], samples: list[dict]) -> dict:
    """Score the exact archived prompts; leave rubric answers for human review."""
    from evaluation.cpu_candidate_quality import score_case
    from evaluation.historical_suite_bridge import HISTORICAL_CONTENT_SHA256

    _require(len(cases) == len(samples) == 50)
    counts = Counter()
    categories = defaultdict(Counter)
    results = []
    for case, sample in zip(cases, samples):
        _require(sample["prompt"] == case["prompt"])
        from evaluation.completion_evidence import completion_status
        evidence = sample.get("completion_evidence")
        result, safe_answer = score_case(case, sample["completion"], evidence)
        counts[result] += 1
        categories[case["category"]][result] += 1
        results.append({"case_id": case["id"], "category": case["category"],
                        "result": result, "answer": safe_answer,
                        "review_criteria": case["evaluation"].get("criteria", []),
                        "completion_evidence": evidence,
                        "completion_status": completion_status(evidence)})
    return {"suite_sha256": HISTORICAL_CONTENT_SHA256,
            "case_count": len(cases), "result_counts": dict(counts),
            "category_counts": {key: dict(value) for key, value in categories.items()},
            "cases": results, "release_thresholds_approved": False,
            "live_routes_verified": 0, "phase_a_item_closed": False}


def generate(family: str, archive_path: Path, snapshot: Path, output: Path,
             *, quality_suite: bool = False,
             manual_max_new_tokens: int | None = None) -> dict:
    from evaluation.completion_evidence import generation_evidence, output_budgets
    from training.template_policy import CHAT_TEMPLATE_KWARGS
    _require(not output.exists() and not output.is_symlink() and output.parent.is_dir())
    receipt, contents = inspect_archive(family, archive_path)
    if quality_suite:
        from evaluation.historical_suite_bridge import load_archived_suite
        cases = load_archived_suite()["cases"]
        prompts = tuple(case["prompt"] for case in cases)
        budgets = output_budgets(cases, manual_max_new_tokens)
    else:
        cases = []
        prompts = PROMPTS
        if manual_max_new_tokens is not None:
            raise ValueError("manual output budget requires quality-suite mode")
        budgets = (80,) * len(prompts)
    lineage = contract.load_json(ROOT / "config/kova-private-lineage.v1.json")
    verify_snapshot(snapshot, contract._pinned_manifest(family, lineage["families"][family]))
    _require(all(os.environ.get(name) == "1" for name in
                 ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")))
    training.verify_installed_stack()
    from importlib.metadata import version
    _require(version("bitsandbytes") == "0.48.2")
    import torch
    import bitsandbytes as bnb
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
    from training.cosmo_hardware import verify_nvidia_t4

    verify_nvidia_t4(torch)
    training.verify_four_bit_runtime(torch, bnb)
    with TemporaryDirectory(prefix="kova-gpu-experiment-") as folder:
        adapter = Path(folder) / "adapter"
        adapter.mkdir(mode=0o700)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            destination = adapter / name
            with destination.open("xb") as stream:
                stream.write(contents["adapter/" + name])
            destination.chmod(0o600)
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                                   trust_remote_code=False)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.float16)
        base = AutoModelForCausalLM.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False,
            quantization_config=quant, dtype=torch.float16, device_map={"": 0})
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False,
                                          local_files_only=True)
        model.eval()
        system = (ROOT / "prompts/kova-identity.v5.txt").read_text()
        generation_config = GenerationConfig(
            do_sample=False, num_beams=1, num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        profile = {
            "decoder": generation_config.to_dict(),
            "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS),
        }
        samples = []
        with torch.inference_mode():
            for question, budget in zip(prompts, budgets, strict=True):
                encoded = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system},
                     {"role": "user", "content": question}],
                    tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", **CHAT_TEMPLATE_KWARGS)
                ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
                generated = model.generate(input_ids=ids.to("cuda:0"),
                                           max_new_tokens=budget,
                                           generation_config=generation_config)
                tokens = generated[0, ids.shape[-1]:].tolist()
                evidence = generation_evidence(tokens, max_new_tokens=budget,
                                               eos_token_id=tokenizer.eos_token_id)
                samples.append({"prompt": question, "completion": tokenizer.decode(
                    tokens, skip_special_tokens=True).strip(),
                    "completion_evidence": evidence})
    report = {"kind": ("experimental_t4_quality_50_cases" if quality_suite else
                       "experimental_t4_offline_inference_samples"),
              "family": family, "archive_sha256": ARCHIVES[family][1],
              "training_commit": receipt["source_commit"],
              "base_revision": receipt["base_revision"],
              "adapter_sha256": receipt["adapter_sha256"],
              "human_quality_review_complete": False,
              "generation_profile": profile, "completion_evidence_authenticated": False,
              "signed_controller_pilot": False, "production_routing_approved": False}
    if quality_suite:
        report.update(_quality_results(cases, samples))
        report["generation"] = {"do_sample": False, "enable_thinking": False,
                                "strict_max_new_tokens": 128,
                                "manual_max_new_tokens": manual_max_new_tokens}
    else:
        report["samples"] = samples
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(ARCHIVES), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--generate", action="store_true")
    parser.add_argument("--quality-suite", action="store_true",
                        help="generate and score all 50 pinned quality cases")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manual-max-new-tokens", type=int)
    args = parser.parse_args(argv)
    if args.verify_only:
        if (args.snapshot is not None or args.output is not None or args.quality_suite
                or args.manual_max_new_tokens is not None):
            parser.error("verify-only accepts only family and archive")
        receipt, _ = inspect_archive(args.family, args.archive)
        print(json.dumps({"status": "experimental_archive_verified",
                          "family": args.family, "archive_sha256": ARCHIVES[args.family][1],
                          "source_commit": receipt["source_commit"],
                          "signed_controller_pilot": False,
                          "production_routing_approved": False}, sort_keys=True))
        return 0
    if args.snapshot is None or args.output is None:
        parser.error("generate requires local snapshot and new output")
    if args.quality_suite != (args.manual_max_new_tokens is not None):
        parser.error("quality-suite requires an explicit manual output budget")
    report = generate(args.family, args.archive, args.snapshot, args.output,
                      quality_suite=args.quality_suite,
                      manual_max_new_tokens=args.manual_max_new_tokens)
    if args.quality_suite:
        print(json.dumps({"family": args.family, "case_count": report["case_count"],
                          "result_counts": report["result_counts"],
                          "human_quality_review_complete": False}, sort_keys=True))
    else:
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
