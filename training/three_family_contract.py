"""Fail-closed source contract for the three-family, six-dollar T4 pilot.

Importing or validating this module performs no network, provider, download,
training, deployment, or filesystem mutation. Paid execution must use separate
signed evidence and the exact operator plan after an explicit owner release.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
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
APPROVED_DATASET_SHA256 = "fa6406b2be2e607f8a40565bf9340db226a6fb8def4d5342d30dc38105696051"
APPROVED_REVIEW_SHA256 = "618afd809876d1b51b38f58999df6b8b492c823ab158c6e472083643c4004bf9"
APPROVED_PROMPT_SHA256 = "ed1b503f947cabc6a7c24a9395bd63b9eff2dd55d57570a0d5a8fd51df3c5bc8"
MANIFEST_SHA256 = {
    "kova-cosmo": "d1dd63b2ee120b0944a58021a21608a46bb03074f87adbafb067b2aedccaf162",
    "kova-orion": "d4eb95e29eef9a92445e3b7622d17f066678e5dc0915fd5a3c40189167c5e746",
    "kova-nova": "8fd1bac2209b5f1c4f1412fc1797b29e0bdc4609025bdd176660dd560fa97566",
}
RECIPE_SHA256 = {
    "kova-cosmo": "6f577e521d4ddeff1725f9c660825d6bd08f94d80b06acf6f0721b265236eb3d",
    "kova-orion": "467330ca8e7ee928ae76b88996b70264190512045e2d42c1d54b4262a9456885",
    "kova-nova": "46d44da0bcbcfc4fe15454059bbaba2c7e104e8d079348998e5075ee1fee3088",
}
COST_CATEGORY_BOUNDS = {
    "managed_disks": "0.3000", "snapshots": "0.0000",
    "storage_capacity": "0.1000", "storage_transactions": "0.1000",
    "network_transfer": "0.1000", "public_ip_and_network": "0.1000",
    "nat_gateway_hours": "0.1000", "nat_gateway_data_processed": "0.1000",
    "logic_app_executions": "0.0500",
    "shutdown_delay": "0.1000", "failed_allocation_attempts": "0.1000",
}
FAMILY_ALLOWANCES = {
    "kova-cosmo": "1.2500", "kova-orion": "1.5000", "kova-nova": "1.7500",
}
MANIFEST_PATHS = {
    "kova-cosmo": "config/qwen3-0.6b-download-manifest.v1.json",
    "kova-orion": "config/qwen3-1.7b-download-manifest.v1.json",
    "kova-nova": "config/qwen3-4b-download-manifest.v1.json",
}


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
                          parse_constant=lambda _: need(False, "nonfinite JSON"),
                          parse_float=_finite_json_float)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, TypeError):
        raise ContractError("invalid UTF-8 JSON contract") from None


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    need(Decimal(value).is_finite() and parsed != float("inf") and parsed != float("-inf"),
         "nonfinite JSON")
    return parsed


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
        manifest = _pinned_manifest(family, item)
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


def _pinned_manifest(family: str, lineage_item: dict) -> dict:
    need(lineage_item["manifest"] == MANIFEST_PATHS[family], "family manifest path mismatch")
    path = ROOT / MANIFEST_PATHS[family]
    need(sha256(path) == MANIFEST_SHA256[family], "complete manifest digest mismatch")
    return load_json(path)


def verify_snapshot(family: str, directory: Path) -> None:
    """Reject every missing, extra, substituted, or modified model file."""
    lineage = load_json(ROOT / "config/kova-private-lineage.v1.json")
    need(family in FAMILIES, "unknown family")
    manifest = _pinned_manifest(family, lineage["families"][family])
    from training.snapshot_verifier import verify_snapshot as verify_exact_snapshot
    try:
        verify_exact_snapshot(directory, manifest)
    except (ValueError, OSError) as error:
        raise ContractError(f"snapshot rejected: {error}") from None


def validate_dataset():
    contract = load_json(ROOT / "config/kova-three-family-dataset.v2.json")
    need(contract["dataset_path"] == "data/kova-identity-shared.v2.jsonl")
    need(contract["prompt_path"] == "prompts/kova-identity.v3.txt")
    need(contract["review_path"] == "data/kova-identity-shared-review.v2.json")
    need(contract["dataset_sha256"] == APPROVED_DATASET_SHA256, "unapproved dataset digest")
    dataset = ROOT / contract["dataset_path"]
    prompt = ROOT / contract["prompt_path"]
    need(sha256(ROOT / contract["review_path"]) == APPROVED_REVIEW_SHA256,
         "owner review ledger digest mismatch")
    review = load_json(ROOT / contract["review_path"])
    need(sha256(dataset) == APPROVED_DATASET_SHA256, "unapproved dataset bytes")
    need(contract["prompt_sha256"] == APPROVED_PROMPT_SHA256
         and sha256(prompt) == APPROVED_PROMPT_SHA256, "approved prompt mismatch")
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


def _locked_package_versions(stack: dict) -> dict[str, str]:
    versions = {}
    for lock in stack["locks"]:
        text = (ROOT / lock["path"]).read_text(encoding="utf-8", errors="strict")
        for match in re.finditer(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s\\]+)", text, re.MULTILINE):
            name = match.group(1).lower().replace("_", "-")
            version = match.group(2)
            need(name not in versions or versions[name] == version,
                 f"conflicting locked package version:{name}")
            versions[name] = version
    return versions


def validate_training():
    stack = load_json(ROOT / "config/kova-three-family-training-stack.v1.json")
    need(stack["gpu_name"] == "NVIDIA T4" and stack["compute_capability"] == "7.5")
    need(stack.get("python") == "3.12" and stack.get("cuda") == "12.8",
         "training runtime version drift")
    expected_packages = {
        "torch": "2.8.0",
        "transformers": "5.17.0",
        "trl": "1.13.0",
        "peft": "0.21.0",
        "accelerate": "1.15.0",
        "bitsandbytes": "0.48.2",
    }
    need(stack.get("packages") == expected_packages, "training package declaration drift")
    for lock in stack["locks"]:
        need(sha256(ROOT / lock["path"]) == lock["sha256"])
    locked = _locked_package_versions(stack)
    for package, version in expected_packages.items():
        need(locked.get(package) == version, f"training package lock mismatch:{package}")
    expected = {"kova-cosmo": (1024, 1800, "1.2500"),
                "kova-orion": (1024, 2700, "1.5000"),
                "kova-nova": (768, 4500, "1.7500")}
    for family, (sequence, elapsed, allowance) in expected.items():
        recipe_path = ROOT / f"config/{family}-qlora.v1.json"
        need(sha256(recipe_path) == RECIPE_SHA256[family], "complete QLoRA recipe digest mismatch")
        cfg = load_json(recipe_path)
        need(cfg["family"] == family and cfg["method"] == "four_bit_qlora_lora_sft")
        need(cfg["manifest"] == MANIFEST_PATHS[family], "recipe manifest mismatch")
        need(cfg["dataset"] == "config/kova-three-family-dataset.v2.json",
             "recipe dataset mismatch")
        need(cfg.get("output_directory") == f"outputs/{family}",
             "recipe output boundary mismatch")
        lineage_item = load_json(ROOT / "config/kova-private-lineage.v1.json")["families"][family]
        manifest = _pinned_manifest(family, lineage_item)
        need(cfg.get("upstream_repository") == lineage_item["upstream_repository"]
             and cfg.get("immutable_revision") == lineage_item["immutable_revision"]
             and manifest["model"] == cfg["upstream_repository"]
             and manifest["revision"] == cfg["immutable_revision"],
             "recipe model lineage mismatch")
        need(load_json(ROOT / cfg["dataset"])["dataset_sha256"] == APPROVED_DATASET_SHA256,
             "recipe dataset digest mismatch")
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
    declared_bindings = evaluation["answer_binding"]["binds"]
    need(type(declared_bindings) is list and tuple(declared_bindings) == (
        "source_commit", "family", "base_revision", "adapter_sha256",
        "adapter_bundle_sha256", "case_id", "prompt_sha256", "answer_sha256",
        "runtime_profile", "conversation_id", "session_id",
    ), "evaluation signature binding declaration drift")
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


def _validated_cost_guard() -> dict:
    cost = load_json(ROOT / "config/kova-three-family-cost-guard.v1.json")
    need(cost["combined_hard_ceiling"] == "6.0000")
    need(cost["conditional_cosmo_only_pilot"] == {
        "hard_ceiling_usd": "3.3000", "minimum_allocation_seconds": 5400,
        "maximum_allocation_seconds": 7200,
        "maximum_compute_reservation_usd": "0.9000",
        "other_families_authorized": False,
    }, "Cosmo-only owner ceiling drift")
    need(cost["emergency_cleanup_margin"] == "1.2500")
    need(type(cost.get("family_allowances")) is dict and
         cost["family_allowances"] == FAMILY_ALLOWANCES,
         "unapproved family allowances")
    for requirement in ("independent_signed_admission_required",
                        "post_run_cost_evidence_required",
                        "zero_residual_billable_resources_required"):
        need(cost.get(requirement) is True, f"mandatory cost control disabled: {requirement}")
    need(cost["provider_compute_meter_increment_seconds"] == 60)
    need(cost["meter_categories"] == ["compute_allocation_time", *COST_CATEGORY_BOUNDS])
    need(type(cost["category_upper_bounds"]) is dict
         and set(cost["category_upper_bounds"]) == set(COST_CATEGORY_BOUNDS), "cost categories mismatch")
    for category, expected in COST_CATEGORY_BOUNDS.items():
        raw = cost["category_upper_bounds"][category]
        need(type(raw) is str, f"invalid cost bound: {category}")
        try:
            bound = Decimal(raw)
        except InvalidOperation:
            raise ContractError(f"invalid cost bound: {category}") from None
        need(bound.is_finite() and bound >= 0 and bound == Decimal(expected),
             f"unapproved cost bound: {category}")
    return cost


def worst_case_total(*, hourly_compute_rate: Decimal, lifecycle_seconds: int = 21600) -> Decimal:
    need(type(hourly_compute_rate) is Decimal and hourly_compute_rate.is_finite()
         and hourly_compute_rate >= 0, "invalid live price")
    need(type(lifecycle_seconds) is int and 0 < lifecycle_seconds <= 21600, "invalid lifecycle")
    cost = _validated_cost_guard()
    increment = cost["provider_compute_meter_increment_seconds"]
    increments = (Decimal(lifecycle_seconds) / Decimal(increment)).to_integral_value(rounding=ROUND_CEILING)
    compute = ((increments * Decimal(increment) / Decimal(3600)) * hourly_compute_rate).quantize(
        Decimal("0.0001"), rounding=ROUND_CEILING)
    ancillary = sum(Decimal(value) for value in cost["category_upper_bounds"].values())
    return compute + ancillary + Decimal(cost["emergency_cleanup_margin"])


def admit_bootstrap(hourly_compute_rate: Decimal) -> Decimal:
    total = worst_case_total(hourly_compute_rate=hourly_compute_rate)
    need(total <= Decimal("6.0000"), "live-price worst case exceeds six-dollar ceiling")
    return total


def admit_conditional_cosmo_pilot(hourly_compute_rate: Decimal, *,
                                  lifecycle_seconds: int | None = None) -> Decimal:
    """Check the separate owner limit before a Cosmo-only pilot is proposed.

    This is a worst-case reservation check, not a release to spend or a real-time
    billing hard stop. The independent controller still needs verified meters.
    """
    cost = _validated_cost_guard()
    proposal = cost["conditional_cosmo_only_pilot"]
    seconds = (proposal["maximum_allocation_seconds"] if lifecycle_seconds is None
               else lifecycle_seconds)
    need(type(seconds) is int and proposal["minimum_allocation_seconds"] <= seconds <=
         proposal["maximum_allocation_seconds"], "Cosmo allocation window outside approved bounds")
    total = worst_case_total(hourly_compute_rate=hourly_compute_rate,
                             lifecycle_seconds=seconds)
    ancillary = sum(Decimal(value) for value in cost["category_upper_bounds"].values())
    compute = total - ancillary - Decimal(cost["emergency_cleanup_margin"])
    need(Decimal(proposal["maximum_compute_reservation_usd"]) + ancillary +
         Decimal(cost["emergency_cleanup_margin"]) == Decimal(proposal["hard_ceiling_usd"]),
         "Cosmo reservation does not cover the conditional ceiling")
    need(total <= Decimal(proposal["hard_ceiling_usd"]),
         "Cosmo worst case exceeds conditional $3.30 ceiling")
    need(compute <= Decimal(proposal["maximum_compute_reservation_usd"]),
         "Cosmo compute reservation exceeded")
    return total


def validate_live_price_evidence(path: Path, *, admission_scope: str = "three-family") -> dict:
    """Consume an already-captured Azure Retail Prices response and fail closed."""
    need(admission_scope in ("three-family", "cosmo-only"), "invalid price admission scope")
    value = load_json(path, maximum_bytes=64 * 1024)
    need(type(value) is dict and value.get("NextPageLink") in (None, ""),
         "incomplete live-price response")
    items = value.get("Items")
    need(type(items) is list and items, "empty live-price response")
    need(value.get("Count", len(items)) == len(items), "live-price response count mismatch")
    need(all(type(entry) is dict for entry in items), "invalid live-price item")
    for entry in items:
        need(entry.get("armSkuName") == "Standard_NC4as_T4_v3"
             and entry.get("armRegionName") == "eastus"
             and entry.get("currencyCode") == "USD"
             and entry.get("serviceName") == "Virtual Machines"
             and entry.get("serviceFamily") == "Compute"
             and entry.get("unitOfMeasure") == "1 Hour"
             and type(entry.get("productName")) is str
             and type(entry.get("skuName")) is str
             and type(entry.get("meterName")) is str
             and entry.get("type") in ("Consumption", "DevTestConsumption", "Reservation")
             and type(entry.get("isPrimaryMeterRegion")) is bool,
             "malformed or unrelated live-price meter")
    matches = [entry for entry in items
               if entry.get("armSkuName") == "Standard_NC4as_T4_v3"
               and entry.get("armRegionName") == "eastus"
               and entry.get("currencyCode") == "USD"
               and entry.get("serviceName") == "Virtual Machines"
               and entry.get("serviceFamily") == "Compute"
               and entry.get("productName") == "Virtual Machines NCasT4 v3 Series"
               and entry.get("skuName") == "NC4as T4 v3"
               and entry.get("meterName") == "NC4as T4 v3"
               and entry.get("type") == "Consumption"
               and entry.get("isPrimaryMeterRegion") is True
               and entry.get("unitOfMeasure") == "1 Hour"]
    need(len(matches) == 1, "missing or ambiguous Linux pay-as-you-go meter")
    item = matches[0]
    need(item.get("tierMinimumUnits") == 0, "tiered live price")
    raw_rate = item.get("retailPrice")
    need(type(raw_rate) in (int, float, str) and not isinstance(raw_rate, bool), "invalid live price")
    try:
        rate = Decimal(str(raw_rate))
        unit_rate = Decimal(str(item.get("unitPrice")))
    except (InvalidOperation, ValueError):
        raise ContractError("invalid live price") from None
    need(rate.is_finite() and rate > 0 and unit_rate == rate, "invalid live price")
    cosmo_total = worst_case_total(hourly_compute_rate=rate, lifecycle_seconds=7200)
    if admission_scope == "cosmo-only":
        total = admit_conditional_cosmo_pilot(rate)
        ceiling = "3.3000"
    else:
        total = admit_bootstrap(rate)
        ceiling = "6.0000"
    return {
        "status": "live_price_admitted",
        "admission_scope": admission_scope,
        "hourly_compute_rate_usd": str(rate),
        "worst_case_total_usd": str(total),
        "hard_ceiling_usd": ceiling,
        "conditional_cosmo_only_worst_case_usd": str(cosmo_total),
        "conditional_cosmo_only_hard_ceiling_usd": "3.3000",
        "conditional_cosmo_only_eligible": cosmo_total <= Decimal("3.3000"),
        "conditional_cosmo_only_maximum_allocation_seconds": 7200,
        "conditional_cosmo_only_compute_reservation_usd": "0.9000",
    }


def validate_probe_evidence(path: Path, *, require_live_imds: bool = False) -> dict:
    """Validate probe values, then bind the image to the executing Azure VM."""
    value = load_json(path, maximum_bytes=64 * 1024)
    compat = load_json(ROOT / "config/kova-t4-compatibility.v1.json")
    stack = load_json(ROOT / "config/kova-three-family-training-stack.v1.json")
    need(set(value) == {
        "schema_version", "device_name", "compute_capability", "cuda_version",
        "bitsandbytes_four_bit_available", "available_vram_bytes", "free_disk_bytes",
        "family_probes", "azure_vm_image_urn", "azure_vm_resource_id", "azure_vm_id",
    }, "invalid T4 probe evidence shape")
    need(value["schema_version"] == 1, "invalid T4 probe evidence schema")
    pilot = load_json(ROOT / "config/kova-three-family-pilot.v1.json")
    expected_image = "Canonical:ubuntu-24_04-lts:server:" + pilot["single_shared_vm"]["ubuntu_image_version"]
    need(value["azure_vm_image_urn"] == expected_image,
         "runtime image does not match pinned Azure image")
    need(type(value["azure_vm_resource_id"]) is str and
         value["azure_vm_resource_id"].startswith("/subscriptions/") and
         type(value["azure_vm_id"]) is str and len(value["azure_vm_id"]) == 36,
         "invalid Azure VM identity")
    if require_live_imds:
        from training.cosmo_lifecycle_authority import (
            AuthorityError, AZURE_COMPUTE_IMDS_URL, _imds_transport,
        )
        try:
            compute = _imds_transport(AZURE_COMPUTE_IMDS_URL)
            image = compute["storageProfile"]["imageReference"]
            need(compute["resourceId"].casefold() == value["azure_vm_resource_id"].casefold()
                 and compute["vmId"].casefold() == value["azure_vm_id"].casefold()
                 and compute["location"].casefold() == "eastus"
                 and compute["vmSize"] == "Standard_NC4as_T4_v3",
                 "probe ran on another Azure VM")
            need(image["publisher"] == "Canonical" and
                 image["offer"] == "ubuntu-24_04-lts" and image["sku"] == "server" and
                 image.get("exactVersion") == "24.04.202609040",
                 "live Azure VM image differs from the source pin")
        except (AuthorityError, KeyError, AttributeError, TypeError):
            raise ContractError("trusted Azure IMDS image proof unavailable") from None
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
        "status": "t4_probe_evidence_valid" if require_live_imds else "untrusted_probe_shape_valid",
        "device_name": value["device_name"],
        "azure_vm_image_urn": value["azure_vm_image_urn"],
        "live_azure_image_verified": require_live_imds,
        "families": list(FAMILIES),
    }


def validate_cost_lifecycle_operator():
    cost = _validated_cost_guard()
    need(cost["azure_budget_alert_is_hard_stop"] is False)
    lifecycle = load_json(ROOT / "config/kova-three-family-lifecycle.v1.json")
    need(lifecycle["maximum_lifecycle_seconds"] == 21600)
    need(lifecycle["watchdog"]["recurrence_seconds"] == 60)
    need(lifecycle["watchdog"]["independent_control_plane"] is True)
    need(lifecycle["remote_ledger"]["gpu_runner_write_access"] is False)
    need(lifecycle["remote_ledger"]["exclusive_lease_seconds"] == 60)
    need(tuple(lifecycle["sequential_family_order"]) == FAMILIES)
    operator = load_json(ROOT / "config/kova-three-family-operator-plan.v1.json")
    need([step["id"] for step in operator["steps"]] == list(range(1, 20)))
    need(operator["steps"][2] == {"id": 3,
         "name": "prepare_pilot_and_watchdog_resource_groups",
         "dry_run": "enumerate_required_creation_no_execution"})
    need(all(step["dry_run"] for step in operator["steps"]))
    need(operator["paid_execution_statement"].endswith("No production deployment is authorized."))
    pilot = load_json(ROOT / "config/kova-three-family-pilot.v1.json")
    need(tuple(pilot["families"]) == FAMILIES and tuple(pilot["sequential_order"]) == FAMILIES)
    need(pilot["single_shared_vm"]["sku"] == "Standard_NC4as_T4_v3")
    need(pilot["single_shared_vm"]["public_ip_allowed"] is False)
    need(pilot["combined_hard_ceiling_usd"] == "6.0000")
    need(pilot["single_shared_vm"]["ubuntu_image_version"] == "24.04.202609040",
         "exact East US Ubuntu image version mismatch")
    vm_template = (ROOT / "infra/three-family-pilot-vm.bicep").read_text(encoding="utf-8")
    need("param ubuntuImageVersion string" in vm_template
         and "version: ubuntuImageVersion" in vm_template
         and "version: 'latest'" not in vm_template,
         "VM image must require an immutable operator-supplied version")
    need("@allowed(['24.04.202609040'])" in vm_template,
         "VM image version must match reviewed East US listing")
    from training.three_family_operator import command_plan
    for step in command_plan():
        if "infra/three-family-pilot-vm.bicep" in step.get("argv", []):
            need("ubuntuImageVersion=${PINNED_UBUNTU_IMAGE_VERSION}" in step["argv"],
                 "VM image version missing from operator plan")
    for value, label in ((cost, "cost"), (lifecycle, "lifecycle"), (operator, "operator"), (pilot, "pilot")):
        _all_false(value, label)


def _fresh_admission(event: dict, *, now: datetime) -> None:
    need(type(event.get("observed_at_utc")) is str and
         type(event.get("expires_at_utc")) is str, "missing admission timestamps")
    try:
        observed = datetime.strptime(event["observed_at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
        expires = datetime.strptime(event["expires_at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
    except ValueError:
        raise ContractError("invalid admission timestamps") from None
    need(observed <= now < expires <= observed + timedelta(minutes=5),
         "stale admission")


def _verify_preservation_receipt(event: dict, *, family: str, grant_sequence: int,
                                 public_key: bytes | None, now: datetime) -> None:
    """Check a separately trusted verifier's receipt for a read-back artifact.

    The future remote authority must pin this verifier key independently of
    the ledger event and sign only after reading back the immutable export.
    With no provisioned key, preservation and the next grant fail closed.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    need(type(public_key) is bytes and len(public_key) == 32,
         "trusted preservation verifier key required")
    need(set(event) == {"kind", "family", "artifact_sha256", "preservation_receipt"},
         "artifact preservation event shape mismatch")
    receipt = event["preservation_receipt"]
    need(type(receipt) is dict and set(receipt) == {"payload", "signature_ed25519_hex"},
         "protected destination receipt required")
    payload = receipt["payload"]
    need(type(payload) is dict and set(payload) == {
        "schema_version", "family", "grant_sequence", "artifact_sha256",
        "verified_sha256", "destination_uri", "immutable_version",
        "protected_destination", "outside_pilot_group", "verification_succeeded",
        "verified_at_utc",
    }, "protected destination receipt shape mismatch")
    digest = event["artifact_sha256"]
    need(type(digest) is str and HEX64.fullmatch(digest) is not None and
         payload["schema_version"] == 1 and payload["family"] == family and
         type(payload["grant_sequence"]) is int and payload["grant_sequence"] == grant_sequence and
         payload["artifact_sha256"] == digest and payload["verified_sha256"] == digest,
         "artifact verification does not bind the grant")
    uri = payload["destination_uri"]
    need(type(uri) is str and uri.startswith("https://") and "?" not in uri and
         type(payload["immutable_version"]) is str and payload["immutable_version"] and
         payload["protected_destination"] is True and
         payload["outside_pilot_group"] is True and
         payload["verification_succeeded"] is True,
         "artifact destination is not verified and protected")
    try:
        verified = datetime.strptime(payload["verified_at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
    except (TypeError, ValueError):
        raise ContractError("invalid artifact verification timestamp") from None
    need(now - timedelta(minutes=5) <= verified <= now,
         "stale artifact verification receipt")
    signature = receipt["signature_ed25519_hex"]
    need(type(signature) is str and re.fullmatch(r"[0-9a-f]{128}", signature) is not None,
         "invalid artifact preservation signature")
    try:
        message = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")
        Ed25519PublicKey.from_public_bytes(public_key).verify(bytes.fromhex(signature), message)
    except (InvalidSignature, ValueError, TypeError, RecursionError):
        raise ContractError("untrusted artifact preservation receipt") from None


def _trusted_lifecycle(state: dict, trusted: dict | None) -> dict:
    """The remote ledger must load this immutable context from its own lease."""
    required = {"lifecycle_id", "ledger_id", "admission_scope",
                "pilot_resource_group_id", "watchdog_resource_group_id"}
    need(type(trusted) is dict and set(trusted) == required and
         all(state.get(key) == trusted[key] for key in required),
         "independently trusted lifecycle binding required")
    need(type(trusted["lifecycle_id"]) is str and trusted["lifecycle_id"] and
         type(trusted["ledger_id"]) is str and trusted["ledger_id"] and
         trusted["admission_scope"] in ("cosmo-only", "three-family"),
         "invalid original lifecycle admission scope")
    group = re.compile(
        r"/subscriptions/[0-9a-f-]{36}/resourceGroups/[a-zA-Z0-9_.()\-]{1,90}\Z",
        re.IGNORECASE)
    pilot = trusted["pilot_resource_group_id"]
    watchdog = trusted["watchdog_resource_group_id"]
    need(type(pilot) is str and type(watchdog) is str and
         group.fullmatch(pilot) is not None and group.fullmatch(watchdog) is not None and
         pilot.casefold() != watchdog.casefold() and
         pilot.split("/")[2].casefold() == watchdog.split("/")[2].casefold(),
         "trusted lifecycle must bind two groups in one subscription")
    return trusted


def _verify_cleanup_receipt(event: dict, *, expected_sequence: int,
                            lifecycle: dict, public_key: bytes | None,
                            now: datetime, unpreserved_grants: list[dict]) -> None:
    """Require independent, signed deletion inventory and final cost reconciliation.

    The verifier key must be provisioned outside this ledger and the two
    deleted groups. Delayed Azure charges prevent a terminal event until the
    verifier can attest that the final cost evidence is complete.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    need(type(public_key) is bytes and len(public_key) == 32,
         "trusted cleanup verifier key required")
    need(set(event) == {"kind", "cleanup_receipt"}, "cleanup event shape mismatch")
    receipt = event["cleanup_receipt"]
    need(type(receipt) is dict and set(receipt) == {"payload", "signature_ed25519_hex"},
         "signed cleanup receipt required")
    payload = receipt["payload"]
    need(type(payload) is dict and set(payload) == {
        "schema_version", "ledger_sequence", "lifecycle_id", "ledger_id",
        "admission_scope", "pilot_resource_group_id",
        "watchdog_resource_group_id", "pilot_remaining_resources",
        "watchdog_remaining_resources", "subscription_scoped_residual_resources",
        "pilot_deleted_at_utc", "watchdog_deleted_at_utc", "verified_at_utc",
        "cost_posting_complete", "final_cost_usd", "cost_evidence_sha256",
        "evidence_uri", "immutable_evidence_version", "outside_both_groups",
        "verification_succeeded", "unpreserved_grants",
    }, "cleanup receipt shape mismatch")
    need(type(payload["schema_version"]) is int and payload["schema_version"] == 1 and
         type(payload["ledger_sequence"]) is int and
         payload["ledger_sequence"] == expected_sequence and
         all(payload[key] == lifecycle[key] for key in (
             "lifecycle_id", "ledger_id", "admission_scope",
             "pilot_resource_group_id", "watchdog_resource_group_id")) and
         payload["pilot_remaining_resources"] == [] and
         payload["watchdog_remaining_resources"] == [] and
         payload["subscription_scoped_residual_resources"] == [] and
         payload["cost_posting_complete"] is True and
         payload["outside_both_groups"] is True and
         payload["verification_succeeded"] is True,
         "zero residual resources and reconciled final cost required")
    resource_group_id = re.compile(
        r"/subscriptions/[0-9a-f-]{36}/resourceGroups/[a-zA-Z0-9_.()\-]{1,90}\Z",
        re.IGNORECASE)
    pilot = payload["pilot_resource_group_id"]
    watchdog = payload["watchdog_resource_group_id"]
    need(type(pilot) is str and type(watchdog) is str and
         resource_group_id.fullmatch(pilot) is not None and
         resource_group_id.fullmatch(watchdog) is not None and
         pilot.casefold() != watchdog.casefold() and
         pilot.split("/")[2].casefold() == watchdog.split("/")[2].casefold(),
         "cleanup must bind two distinct groups in one subscription")
    try:
        pilot_deleted, watchdog_deleted, verified = (
            datetime.strptime(payload[field], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            for field in ("pilot_deleted_at_utc", "watchdog_deleted_at_utc", "verified_at_utc")
        )
        final_cost = Decimal(payload["final_cost_usd"])
    except (KeyError, TypeError, ValueError, InvalidOperation):
        raise ContractError("invalid cleanup timestamps or final cost") from None
    ceiling = Decimal("3.3000" if lifecycle["admission_scope"] == "cosmo-only" else "6.0000")
    need(pilot_deleted <= verified and watchdog_deleted <= verified and
         now - timedelta(minutes=5) <= verified <= now and
         final_cost.is_finite() and 0 <= final_cost <= ceiling,
         "cleanup evidence is stale or final cost exceeds owner ceiling")
    need(type(payload["cost_evidence_sha256"]) is str and
         HEX64.fullmatch(payload["cost_evidence_sha256"]) is not None and
         type(payload["evidence_uri"]) is str and
         payload["evidence_uri"].startswith("https://") and
         "?" not in payload["evidence_uri"] and
         type(payload["immutable_evidence_version"]) is str and
         bool(payload["immutable_evidence_version"]),
         "immutable external cleanup and cost evidence required")
    failures = payload["unpreserved_grants"]
    need(type(failures) is list and len(failures) == len(unpreserved_grants),
         "unpreserved grant requires signed failure evidence")
    for proof, grant in zip(failures, unpreserved_grants):
        need(type(proof) is dict and set(proof) == {
            "family", "grant_sequence", "training_failed", "no_artifact_produced",
            "failure_evidence_sha256", "failure_evidence_uri",
            "immutable_failure_evidence_version",
        } and proof["family"] == grant["family"] and
             type(proof["grant_sequence"]) is int and
             proof["grant_sequence"] == grant["grant_sequence"] and
             proof["training_failed"] is True and proof["no_artifact_produced"] is True and
             type(proof["failure_evidence_sha256"]) is str and
             HEX64.fullmatch(proof["failure_evidence_sha256"]) is not None and
             type(proof["failure_evidence_uri"]) is str and
             proof["failure_evidence_uri"].startswith("https://") and
             "?" not in proof["failure_evidence_uri"] and
             type(proof["immutable_failure_evidence_version"]) is str and
             bool(proof["immutable_failure_evidence_version"]),
             "independent immutable no-artifact failure proof required")
    signature = receipt["signature_ed25519_hex"]
    need(type(signature) is str and re.fullmatch(r"[0-9a-f]{128}", signature) is not None,
         "invalid cleanup verifier signature")
    try:
        message = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")
        Ed25519PublicKey.from_public_bytes(public_key).verify(bytes.fromhex(signature), message)
    except (InvalidSignature, ValueError, TypeError, RecursionError):
        raise ContractError("untrusted cleanup receipt") from None


def append_ledger_event(state: dict, event: dict, *, expected_sequence: int,
                        now: datetime | None = None,
                        preservation_public_key: bytes | None = None,
                        cleanup_public_key: bytes | None = None,
                        trusted_lifecycle: dict | None = None) -> dict:
    """Pure reference transition used by the independent leased remote ledger."""
    need(type(state) is dict and type(event) is dict, "invalid ledger state")
    need(state.get("terminal") is False, "ledger is terminal")
    need(type(expected_sequence) is int and expected_sequence == state.get("sequence"), "stale ledger sequence")
    need(event.get("sequence") is None, "caller cannot assign ledger sequence")
    now = now or datetime.now(timezone.utc)
    need(now.tzinfo is not None, "ledger requires a timezone-aware clock")
    now = now.astimezone(timezone.utc)
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
    if kind == "watchdog_health":
        need(len(state.get("family_order", [])) < len(FAMILIES) and
             family == FAMILIES[len(state.get("family_order", []))],
             "watchdog admission must bind the next family")
        _fresh_admission(event, now=now)
        need(event.get("healthy") is True and
             type(event.get("rule_id")) is str and event["rule_id"],
             "independent watchdog proof required")
    if kind == "cost_admission":
        lifecycle = _trusted_lifecycle(state, trusted_lifecycle)
        events = state.get("events", [])
        need(len(state.get("family_order", [])) < len(FAMILIES) and
             family == FAMILIES[len(state.get("family_order", []))] and
             events and events[-1].get("kind") == "watchdog_health" and
             events[-1].get("family") == family,
             "fresh family watchdog admission required")
        _fresh_admission(event, now=now)
        need(event.get("account_price_verified") is True and
             type(event.get("quote_sha256")) is str and
             HEX64.fullmatch(event["quote_sha256"]),
             "independent account quote required")
        try:
            remaining = Decimal(event["remaining_budget_usd"])
        except (KeyError, TypeError, InvalidOperation):
            raise ContractError("remaining family budget required") from None
        cost = _validated_cost_guard()
        ancillary = sum(Decimal(v) for v in cost["category_upper_bounds"].values())
        need(lifecycle["admission_scope"] == "three-family" or family == "kova-cosmo",
             "Cosmo-only ledger cannot admit another family")
        compute = (Decimal(cost["conditional_cosmo_only_pilot"]["maximum_compute_reservation_usd"])
                   if lifecycle["admission_scope"] == "cosmo-only" else
                   Decimal(cost["family_allowances"][family]))
        required = compute + ancillary + Decimal(cost["emergency_cleanup_margin"])
        need(remaining.is_finite() and required <= remaining <= Decimal("6.0000"),
             "remaining family budget insufficient")
    need(kind != "training_grant" or family in FAMILIES, "training grant requires a family")
    if kind == "family_preserved":
        need(family in FAMILIES and family in state.get("family_order", []),
             "preservation requires a granted family")
        need(not any(item.get("kind") == "family_preserved" and item.get("family") == family
                     for item in state.get("events", [])), "duplicate family preservation")
        grants = [item for item in state.get("events", [])
                  if item.get("kind") == "training_grant" and item.get("family") == family]
        need(len(grants) == 1 and type(grants[0].get("sequence")) is int,
             "preservation must bind one training grant")
        _verify_preservation_receipt(event, family=family,
                                     grant_sequence=grants[0]["sequence"],
                                     public_key=preservation_public_key, now=now)
    if kind == "training_grant":
        events = state.get("events", [])
        need(len(events) >= 2 and events[-2].get("kind") == "watchdog_health" and
             events[-1].get("kind") == "cost_admission" and
             events[-2].get("family") == family and events[-1].get("family") == family,
             "fresh family watchdog and cost admissions required before grant")
        _fresh_admission(events[-2], now=now)
        _fresh_admission(events[-1], now=now)
        previous = FAMILIES.index(family) - 1
        if previous >= 0:
            need(any(item.get("kind") == "family_preserved" and
                     item.get("family") == FAMILIES[previous] for item in events),
                 "previous family preservation required before grant")
    if kind == "cleanup_terminal":
        # A failed grant may terminate without an adapter, but only when the
        # independent cleanup verifier signs immutable no-artifact evidence.
        lifecycle = _trusted_lifecycle(state, trusted_lifecycle)
        events = state.get("events", [])
        unpreserved = [
            {"family": item["family"], "grant_sequence": item["sequence"]}
            for item in events if item.get("kind") == "training_grant" and
            not any(p.get("kind") == "family_preserved" and
                    p.get("family") == item["family"] for p in events)
        ]
        _verify_cleanup_receipt(event, expected_sequence=expected_sequence,
                                lifecycle=lifecycle,
                                public_key=cleanup_public_key, now=now,
                                unpreserved_grants=unpreserved)
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
    parser.add_argument("--admission-scope", choices=("three-family", "cosmo-only"),
                        default="three-family")
    parser.add_argument("--require-live-imds", action="store_true")
    args = parser.parse_args(argv)
    if args.probe_evidence is not None:
        report = validate_probe_evidence(args.probe_evidence,
                                         require_live_imds=args.require_live_imds)
    elif args.retail_price_evidence is not None:
        need(not args.require_live_imds, "IMDS proof requires probe evidence")
        report = validate_live_price_evidence(args.retail_price_evidence,
                                              admission_scope=args.admission_scope)
    else:
        need(not args.require_live_imds, "IMDS proof requires probe evidence")
        report = validate()
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
