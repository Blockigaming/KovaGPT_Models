"""Fail-closed source contract for the three-family, six-dollar T4 pilot.

Importing or validating this module performs no network, provider, download,
training, deployment, or filesystem mutation. Paid execution must use separate
signed evidence and the exact operator plan after an explicit owner release.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
FAMILIES = ("kova-cosmo", "kova-orion", "kova-nova")
LEVELS = ("light", "medium", "high", "extra-high", "max", "ultra")
FALSE_GATES = (
    "resource_creation_authorized", "spending_authorized",
    "model_download_authorized", "training_authorized",
    "evaluation_execution_authorized", "deployment_authorized",
    "production_routing_enabled",
)


class ContractError(ValueError):
    pass


def need(condition: bool, message: str = "three-family source contract rejected") -> None:
    if not condition:
        raise ContractError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def load_json(path: Path, *, maximum_bytes: int = 256 * 1024):
    try:
        raw = path.read_bytes()
        need(0 < len(raw) <= maximum_bytes, "invalid JSON size")
        return json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_unique,
                          parse_constant=lambda _: need(False, "nonfinite JSON"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, TypeError):
        raise ContractError("invalid UTF-8 JSON contract") from None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_false(value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FALSE_GATES:
                need(item is False, f"open safety gate at {path}.{key}")
            _all_false(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _all_false(item, f"{path}[{index}]")


def validate_policy():
    value = load_json(ROOT / "config/current-product-policy.v3.json")
    need(value["schema_version"] == 3)
    need(tuple(value["families"]) == ("cosmo", "orion", "nova"))
    need(tuple(value["processing_levels"]) == LEVELS)
    need(value["processing_levels_are_separate_models"] is False)
    expected = {
        ("chat", "free"): 1, ("chat", "plus"): 6, ("chat", "pro"): 12,
        ("work", "free"): 0, ("work", "plus"): 18, ("work", "pro"): 18,
    }
    counts = {}
    for surface in ("chat", "work"):
        for tier in ("free", "plus", "pro"):
            matrix = value["entitlements"][surface][tier]
            need(set(matrix) == {"cosmo", "orion", "nova"})
            routes = [(family, level) for family, levels in matrix.items() for level in levels]
            need(len(routes) == len(set(routes)))
            need(all(level in LEVELS for _, level in routes))
            counts[(surface, tier)] = len(routes)
    need(counts == expected)
    need(value["entitlements"]["chat"]["free"] == {"cosmo": ["light"], "orion": [], "nova": []})
    need(value["entitlements"]["chat"]["plus"]["nova"] == [])
    need(value["entitlements"]["chat"]["pro"]["nova"] == [])
    need(value["ordinary_surfaces"]["kova_only_names"] is True)
    need(value["ordinary_surfaces"]["upstream_identifiers_allowed"] is False)
    _all_false(value, "policy")
    return counts


def validate_lineage_and_manifests():
    lineage = load_json(ROOT / "config/kova-private-lineage.v1.json")
    need(lineage["license"] == "Apache-2.0")
    need(lineage["commercial_use_permitted"] is True)
    need(lineage["fine_tuning_permitted"] is True)
    need(lineage["license_notice_required_with_distributions"] is True)
    expected = {
        "kova-cosmo": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", 9),
        "kova-orion": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", 11),
        "kova-nova": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", 12),
    }
    totals = {}
    for family, (repository, revision, file_count) in expected.items():
        item = lineage["families"][family]
        need(item["upstream_repository"] == repository and item["immutable_revision"] == revision)
        manifest = load_json(ROOT / item["manifest"])
        need(manifest["model"] == repository and manifest["revision"] == revision)
        files = manifest["files"]
        need(len(files) == file_count)
        paths = [entry["path"] for entry in files]
        need(len(paths) == len(set(paths)))
        need({"LICENSE", "README.md", "config.json", "generation_config.json",
              "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"} <= set(paths))
        need(any(path.endswith(".safetensors") for path in paths))
        for entry in files:
            need(set(entry) == {"path", "bytes", "sha256"})
            need(type(entry["bytes"]) is int and entry["bytes"] > 0)
            need(isinstance(entry["sha256"], str) and HEX64.fullmatch(entry["sha256"]))
            need(not Path(entry["path"]).is_absolute() and ".." not in Path(entry["path"]).parts)
        totals[family] = sum(entry["bytes"] for entry in files)
    _all_false(lineage, "lineage")
    return totals


def verify_snapshot(family: str, directory: Path) -> None:
    """Reject every missing, extra, substituted, or modified model file."""
    lineage = load_json(ROOT / "config/kova-private-lineage.v1.json")
    need(family in FAMILIES, "unknown family")
    manifest = load_json(ROOT / lineage["families"][family]["manifest"])
    expected = {entry["path"]: entry for entry in manifest["files"]}
    actual = {path.relative_to(directory).as_posix(): path for path in directory.rglob("*") if path.is_file()}
    need(set(actual) == set(expected), "snapshot allowlist mismatch")
    for relative, path in actual.items():
        entry = expected[relative]
        need(path.stat().st_size == entry["bytes"], "snapshot size mismatch")
        need(sha256(path) == entry["sha256"], "snapshot hash mismatch")


def validate_dataset():
    contract = load_json(ROOT / "config/kova-three-family-dataset.v2.json")
    dataset = ROOT / contract["dataset_path"]
    prompt = ROOT / contract["prompt_path"]
    review = load_json(ROOT / contract["review_path"])
    need(sha256(dataset) == contract["dataset_sha256"])
    need(sha256(prompt) == contract["prompt_sha256"])
    rows = []
    for raw in dataset.read_bytes().splitlines():
        try:
            row = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_unique)
        except (UnicodeError, json.JSONDecodeError):
            raise ContractError("dataset must be strict UTF-8 JSONL") from None
        need(set(row).issubset({"id", "split", "trusted_runtime", "messages"}))
        need(row["split"] in ("train", "validation"))
        need([m["role"] for m in row["messages"]] == ["user", "assistant"])
        need(all(type(m["content"]) is str and m["content"] for m in row["messages"]))
        rows.append(row)
    need(len(rows) == 42)
    need(len({row["id"] for row in rows}) == 42)
    need(sum(row["split"] == "train" for row in rows) == 27)
    need(sum(row["split"] == "validation" for row in rows) == 15)
    normalized_prompts = {row["messages"][0]["content"].strip().casefold() for row in rows}
    need(len(normalized_prompts) == 42)
    reviewed = review["records"]
    need(len(reviewed) == 42 and {r["id"] for r in reviewed} == {r["id"] for r in rows})
    need(all(r["verdict"] == "approve" for r in reviewed))
    need(review["human_content_review_complete"] is True)
    need(review["current_reconstructed_dataset_sha256"] == contract["dataset_sha256"])
    need(review["approved_dataset_sha256"] == contract["dataset_sha256"])
    need(review["exact_current_dataset_owner_approved"] is True)
    need(review["cryptographic_dataset_approval_complete"] is True)
    need(contract["approval"]["approved_dataset_sha256"] == contract["dataset_sha256"])
    need(contract["approval"]["current_exact_dataset_owner_approved"] is True)
    need(contract["approval"]["reapproval_required"] is False)
    _all_false(contract, "dataset")
    _all_false(review, "review")
    return {"records": 42, "train": 27, "validation": 15, "approval_complete": True}


def validate_training():
    stack = load_json(ROOT / "config/kova-three-family-training-stack.v1.json")
    need(stack["gpu_name"] == "NVIDIA T4" and stack["compute_capability"] == "7.5")
    for lock in stack["locks"]:
        need(sha256(ROOT / lock["path"]) == lock["sha256"])
    expected = {"kova-cosmo": (1024, 1800, "1.2500"),
                "kova-orion": (1024, 2700, "1.5000"),
                "kova-nova": (768, 4500, "1.7500")}
    for family, (sequence, elapsed, allowance) in expected.items():
        cfg = load_json(ROOT / f"config/{family}-qlora.v1.json")
        need(cfg["family"] == family and cfg["method"] == "four_bit_qlora_lora_sft")
        need(cfg["quantization"] == {"bits": 4, "type": "nf4", "double_quant": True,
                                     "compute_dtype": "float16"})
        need(cfg["training"]["maximum_sequence_length"] == sequence)
        need(cfg["training"]["maximum_elapsed_seconds"] == elapsed)
        need(cfg["training"]["maximum_optimizer_steps"] == 7)
        need(cfg["training"]["maximum_epochs"] == 1)
        need(cfg["training"]["completion_only_masking"] is True)
        need(cfg["cost_allowance_usd"] == allowance)
        need(cfg["retry"] == {"automatic": False, "maximum_attempts": 1})
        need(cfg["continue_after_safety_failure"] is False)
        _all_false(cfg, family)
    _all_false(stack, "stack")


def validate_compatibility_evaluation_profiles():
    compat = load_json(ROOT / "config/kova-t4-compatibility.v1.json")
    need(compat["device"] == {"exact_name": "NVIDIA T4", "vram_bytes": 17179869184,
                              "cuda_compute_capability": "7.5", "four_bit_quantization_required": True})
    need(tuple(compat["families"]) == FAMILIES)
    need(compat["families"]["kova-nova"]["estimated_peak_vram_bytes"] < compat["device"]["vram_bytes"])
    need(compat["nova_fail_closed_if_probe_exceeds_estimate_or_available_vram"] is True)
    evaluation = load_json(ROOT / "config/kova-three-family-evaluation.v1.json")
    need(tuple(evaluation["families"]) == FAMILIES)
    need(evaluation["required_attempts_per_family"] == 40)
    need(len(evaluation["required_dimensions"]) == 12)
    need(evaluation["answer_binding"]["algorithm"] == "Ed25519")
    profiles = load_json(ROOT / "config/kova-runtime-profiles.v1.json")
    need(tuple(profiles["profiles"]) == LEVELS)
    need(profiles["separately_trained_models"] is False)
    need(profiles["hidden_chain_of_thought_exposed"] is False)
    for profile in profiles["profiles"].values():
        for field in ("maximum_output_tokens", "maximum_passes", "maximum_tool_calls", "maximum_elapsed_seconds"):
            need(type(profile[field]) is int and profile[field] > 0)
        need(profile["estimated_inference_cost_usd"] == "unmeasured")
    _all_false(compat, "compatibility")
    _all_false(evaluation, "evaluation")
    _all_false(profiles, "profiles")


def worst_case_total(*, hourly_compute_rate: Decimal, lifecycle_seconds: int = 21600) -> Decimal:
    need(type(hourly_compute_rate) is Decimal and hourly_compute_rate >= 0, "invalid live price")
    cost = load_json(ROOT / "config/kova-three-family-cost-guard.v1.json")
    increment = cost["provider_compute_meter_increment_seconds"]
    increments = (Decimal(lifecycle_seconds) / Decimal(increment)).to_integral_value(rounding=ROUND_CEILING)
    compute = (increments * Decimal(increment) / Decimal(3600)) * hourly_compute_rate
    ancillary = sum(Decimal(value) for value in cost["category_upper_bounds"].values())
    return compute + ancillary + Decimal(cost["emergency_cleanup_margin"])


def admit_bootstrap(hourly_compute_rate: Decimal) -> Decimal:
    total = worst_case_total(hourly_compute_rate=hourly_compute_rate)
    ceiling = Decimal(load_json(ROOT / "config/kova-three-family-cost-guard.v1.json")["combined_hard_ceiling"])
    need(total <= ceiling, "live-price worst case exceeds six-dollar ceiling")
    return total


def validate_live_price_evidence(path: Path) -> dict:
    """Consume an already-captured Azure Retail Prices response and fail closed."""
    value = load_json(path, maximum_bytes=64 * 1024)
    items = value.get("Items")
    need(type(items) is list and len(items) == 1, "live-price evidence must contain exactly one item")
    item = items[0]
    need(type(item) is dict, "invalid live-price item")
    need(item.get("armSkuName") == "Standard_NC4as_T4_v3", "live-price SKU mismatch")
    need(item.get("armRegionName") == "eastus", "live-price region mismatch")
    need(item.get("currencyCode") == "USD", "live-price currency mismatch")
    need(item.get("unitOfMeasure") == "1 Hour", "live-price unit mismatch")
    raw_rate = item.get("retailPrice")
    need(type(raw_rate) in (int, float, str) and not isinstance(raw_rate, bool), "invalid live price")
    try:
        rate = Decimal(str(raw_rate))
    except (InvalidOperation, ValueError):
        raise ContractError("invalid live price") from None
    need(rate.is_finite() and rate >= 0, "invalid live price")
    total = admit_bootstrap(rate)
    return {
        "status": "live_price_admitted",
        "hourly_compute_rate_usd": str(rate),
        "worst_case_total_usd": str(total),
        "hard_ceiling_usd": "6.0000",
    }


def validate_probe_evidence(path: Path) -> dict:
    """Validate supplied T4/CUDA/bitsandbytes probe evidence; never run a probe."""
    value = load_json(path, maximum_bytes=64 * 1024)
    compat = load_json(ROOT / "config/kova-t4-compatibility.v1.json")
    stack = load_json(ROOT / "config/kova-three-family-training-stack.v1.json")
    need(set(value) == {
        "schema_version", "device_name", "compute_capability", "cuda_version",
        "bitsandbytes_four_bit_available", "available_vram_bytes", "free_disk_bytes",
        "family_probes",
    }, "invalid T4 probe evidence shape")
    need(value["schema_version"] == 1, "invalid T4 probe evidence schema")
    need(value["device_name"] == compat["device"]["exact_name"], "unexpected GPU")
    need(value["compute_capability"] == compat["device"]["cuda_compute_capability"],
         "unexpected compute capability")
    need(value["cuda_version"] == stack["cuda"], "unexpected CUDA version")
    need(value["bitsandbytes_four_bit_available"] is True, "four-bit runtime unavailable")
    available = value["available_vram_bytes"]
    free_disk = value["free_disk_bytes"]
    need(type(available) is int and available > 0, "invalid available VRAM")
    need(type(free_disk) is int and free_disk >= compat["combined_base_and_adapter_disk_minimum_bytes"],
         "insufficient free disk")
    probes = value["family_probes"]
    need(type(probes) is dict and set(probes) == set(FAMILIES), "family probe set mismatch")
    for family in FAMILIES:
        measured = probes[family]
        expected = compat["families"][family]
        need(type(measured) is dict and set(measured) == {
            "peak_vram_bytes", "maximum_sequence_length", "probe_passed",
        }, "invalid family probe")
        peak = measured["peak_vram_bytes"]
        need(measured["probe_passed"] is True, "family probe failed")
        need(type(peak) is int and 0 < peak <= expected["estimated_peak_vram_bytes"],
             "measured peak exceeds estimate")
        need(peak <= available, "measured peak exceeds available VRAM")
        need(measured["maximum_sequence_length"] == expected["maximum_sequence_length"],
             "sequence-length probe mismatch")
    return {
        "status": "t4_probe_evidence_valid",
        "device_name": value["device_name"],
        "families": list(FAMILIES),
    }


def validate_cost_lifecycle_operator():
    cost = load_json(ROOT / "config/kova-three-family-cost-guard.v1.json")
    need(Decimal(cost["combined_hard_ceiling"]) == Decimal("6.0000"))
    need(Decimal(cost["emergency_cleanup_margin"]) == Decimal("1.2500"))
    need(len(cost["meter_categories"]) == 9)
    need(cost["azure_budget_alert_is_hard_stop"] is False)
    lifecycle = load_json(ROOT / "config/kova-three-family-lifecycle.v1.json")
    need(lifecycle["maximum_lifecycle_seconds"] == 21600)
    need(lifecycle["watchdog"]["recurrence_seconds"] == 60)
    need(lifecycle["watchdog"]["independent_control_plane"] is True)
    need(lifecycle["remote_ledger"]["gpu_runner_write_access"] is False)
    need(lifecycle["remote_ledger"]["exclusive_lease_seconds"] == 60)
    need(tuple(lifecycle["sequential_family_order"]) == FAMILIES)
    operator = load_json(ROOT / "config/kova-three-family-operator-plan.v1.json")
    need([step["id"] for step in operator["steps"]] == list(range(1, 19)))
    need(all(step["dry_run"] for step in operator["steps"]))
    need(operator["paid_execution_statement"].endswith("No production deployment is authorized."))
    pilot = load_json(ROOT / "config/kova-three-family-pilot.v1.json")
    need(tuple(pilot["families"]) == FAMILIES and tuple(pilot["sequential_order"]) == FAMILIES)
    need(pilot["single_shared_vm"]["sku"] == "Standard_NC4as_T4_v3")
    need(pilot["single_shared_vm"]["public_ip_allowed"] is False)
    need(pilot["combined_hard_ceiling_usd"] == "6.0000")
    for value, label in ((cost, "cost"), (lifecycle, "lifecycle"), (operator, "operator"), (pilot, "pilot")):
        _all_false(value, label)


def append_ledger_event(state: dict, event: dict, *, expected_sequence: int) -> dict:
    """Pure reference transition used by the independent leased remote ledger."""
    need(type(state) is dict and type(event) is dict, "invalid ledger state")
    need(state.get("terminal") is False, "ledger is terminal")
    need(type(expected_sequence) is int and expected_sequence == state.get("sequence"), "stale ledger sequence")
    need(event.get("sequence") is None, "caller cannot assign ledger sequence")
    state = json.loads(json.dumps(state))
    family = event.get("family")
    kind = event.get("kind")
    need(kind in ("watchdog_health", "cost_admission", "training_grant", "family_preserved", "cleanup_terminal"),
         "invalid ledger event")
    if family is not None:
        need(family in FAMILIES, "invalid ledger family")
        order = state.setdefault("family_order", [])
        if kind == "training_grant":
            need(family not in order, "duplicate family grant")
            need(family == FAMILIES[len(order)], "out-of-order family grant")
            order.append(family)
    assigned = expected_sequence + 1
    committed = json.loads(json.dumps(event))
    committed["sequence"] = assigned
    state["sequence"] = assigned
    state.setdefault("events", []).append(committed)
    if kind == "cleanup_terminal":
        state["terminal"] = True
    return state


def validate():
    counts = validate_policy()
    totals = validate_lineage_and_manifests()
    dataset = validate_dataset()
    validate_training()
    validate_compatibility_evaluation_profiles()
    validate_cost_lifecycle_operator()
    return {
        "status": "three_family_source_valid_dataset_approved_paid_execution_blocked",
        "families": list(FAMILIES),
        "chat_routes": {tier: counts[("chat", tier)] for tier in ("free", "plus", "pro")},
        "work_routes": {tier: counts[("work", tier)] for tier in ("free", "plus", "pro")},
        "manifest_total_bytes": totals,
        "dataset": dataset,
        "all_paid_and_production_gates_closed": True,
        "provider_calls_made": 0,
        "weights_downloaded": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument("--probe-evidence", type=Path)
    evidence.add_argument("--retail-price-evidence", type=Path)
    args = parser.parse_args(argv)
    if args.probe_evidence is not None:
        report = validate_probe_evidence(args.probe_evidence)
    elif args.retail_price_evidence is not None:
        report = validate_live_price_evidence(args.retail_price_evidence)
    else:
        report = validate()
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
