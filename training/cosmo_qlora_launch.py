"""Cosmo-only launch admission. This module makes no Azure or training calls.

The quote is signed by the independent lifecycle authority, which must obtain
subscription prices and resource observations itself. A retail API response,
an operator-written JSON file, or a budget alert cannot authorize a launch.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
import subprocess
import sys

from training import cosmo_lifecycle_authority as authority
from training import three_family_contract as contract

ROOT = Path(__file__).resolve().parents[1]
KIND = "kova_cosmo_qlora_launch_quote"
SKU = "Standard_NC4as_T4_v3"
IMAGE = "Canonical:ubuntu-24_04-lts:server:24.04.202609040"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MAX_QUOTE_AGE = timedelta(minutes=10)
MAX_ALLOCATION_SECONDS = 7200
MIN_ALLOCATION_SECONDS = 5400
MIN_CLEANUP_LEAD = timedelta(minutes=15)
CEILING = Decimal("3.3000")
ADDITIONAL_COST_CATEGORIES = frozenset({
    "controller_runtime", "controller_registry_and_logs",
    "protected_evidence_retention", "external_archive_and_receipts",
    "tax_and_other_fees",
})
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")


class LaunchRejected(ValueError):
    pass


def need(value: bool, reason: str) -> None:
    if not value:
        raise LaunchRejected(reason)


def clean_source_commit(root: Path = ROOT) -> str:
    """Only a published, clean source tree can be bound to a signed quote."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              check=True, capture_output=True, text=True, timeout=5)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                                cwd=root, check=True, capture_output=True, text=True, timeout=5)
        commit = head.stdout.strip()
        need(bool(HEX40.fullmatch(commit)) and not head.stderr and
             not status.stdout and not status.stderr, "source checkout is not clean")
        return commit
    except (OSError, subprocess.SubprocessError) as exc:
        raise LaunchRejected("source checkout is not verifiable") from exc


def proposal() -> dict:
    """Show immutable source pins and the arithmetic, without claiming live readiness."""
    source = contract.validate()
    need(source["dataset"]["records"] == 42 and
         source["dataset"]["train"] == 27 and
         source["dataset"]["validation"] == 15 and
         source["dataset"]["approval_complete"] is True,
         "approved 42-record split drifted")
    cost = contract._validated_cost_guard()
    pilot = cost["conditional_cosmo_only_pilot"]
    need(pilot["maximum_allocation_seconds"] == MAX_ALLOCATION_SECONDS and
         pilot["minimum_allocation_seconds"] == MIN_ALLOCATION_SECONDS and
         Decimal(pilot["hard_ceiling_usd"]) == CEILING and
         pilot["other_families_authorized"] is False,
         "Cosmo-only limit drifted")
    lineage = contract.load_json(ROOT / "config/kova-private-lineage.v1.json")
    cosmo = lineage["families"]["kova-cosmo"]
    need(cosmo["immutable_revision"] == MODEL_REVISION and
         cosmo["manifest"] == contract.MANIFEST_PATHS["kova-cosmo"],
         "selected model revision drifted")
    recipe = contract.load_json(ROOT / "config/kova-cosmo-qlora.v1.json")
    need(recipe["family"] == "kova-cosmo" and
         recipe["dataset"] == "config/kova-three-family-dataset.v2.json" and
         recipe["training"]["maximum_optimizer_steps"] == 7 and
         recipe["training"]["maximum_elapsed_seconds"] == 1800,
         "Cosmo recipe drifted")
    ancillary = sum(Decimal(v) for v in cost["category_upper_bounds"].values())
    reserve = Decimal(cost["emergency_cleanup_margin"])
    compute = Decimal(pilot["maximum_compute_reservation_usd"])
    need(compute + ancillary + reserve == CEILING, "incomplete cost reservation")
    return {
        "status": "source_valid_live_admission_required",
        "family": "kova-cosmo", "model_revision": MODEL_REVISION,
        "model_manifest_sha256": contract.MANIFEST_SHA256["kova-cosmo"],
        "dataset_sha256": contract.APPROVED_DATASET_SHA256,
        "train_records": 27, "validation_records": 15,
        "image_urn": IMAGE, "sku": SKU,
        "maximum_allocation_seconds": MAX_ALLOCATION_SECONDS,
        "minimum_signed_allocation_seconds": MIN_ALLOCATION_SECONDS,
        "maximum_training_seconds": 1800,
        "maximum_optimizer_steps": 7,
        "compute_reservation_usd": str(compute),
        "ancillary_reservation_usd": str(ancillary),
        "cleanup_reservation_usd": str(reserve),
        "all_in_ceiling_usd": str(CEILING),
        "retail_price_is_account_quote": False,
        "capacity_guaranteed_by_quota_or_what_if": False,
        "paid_actions_enabled": False,
    }


def assess_signed_quote(path: Path, *, source_commit: str, subscription_id: str,
                        now: datetime | None = None, root: Path = ROOT) -> dict:
    """Verify the authority's independent account quote and resource preflight.

    This is a prerequisite report only: it does not release resource creation,
    weights, training, or deployment. The authority key is pinned in source.
    """
    proposal()
    need(type(source_commit) is str and HEX40.fullmatch(source_commit) is not None,
         "invalid source commit")
    need(type(subscription_id) is str and UUID.fullmatch(subscription_id) is not None,
         "invalid subscription ID")
    try:
        envelope = contract.load_json(path, maximum_bytes=65536)
        payload, digest = authority.verify_envelope(envelope, expected_kind=KIND, root=root)
    except (authority.AuthorityError, contract.ContractError) as exc:
        raise LaunchRejected("independent signed account quote is missing or invalid") from exc
    expected = {
        "schema_version", "kind", "issuer", "subscription_id", "source_commit",
        "observed_at_utc", "expires_at_utc", "allocation_deadline_utc",
        "watchdog_cleanup_trigger_utc",
        "region", "vm_sku", "image_urn", "model_revision", "model_manifest_sha256",
        "dataset_sha256", "train_records", "validation_records", "gpu_name",
        "quota", "sku_restrictions", "account_compute_hourly_usd",
        "account_meter_source", "category_upper_bounds_usd",
        "additional_cost_upper_bounds_usd", "evidence_scope",
        "all_category_rates_checked", "watchdog_health_tested",
        "watchdog_can_deallocate_and_delete", "exclusive_pilot_group_empty",
        "no_public_ip",
    }
    need(set(payload) == expected and payload["schema_version"] == 1,
         "account quote schema mismatch")
    current = now or datetime.now(timezone.utc)
    need(current.tzinfo is not None, "UTC clock required")
    current = current.astimezone(timezone.utc)
    try:
        observed = authority.timestamp(payload["observed_at_utc"])
        expires = authority.timestamp(payload["expires_at_utc"])
        deadline = authority.timestamp(payload["allocation_deadline_utc"])
        cleanup_trigger = authority.timestamp(payload["watchdog_cleanup_trigger_utc"])
    except authority.AuthorityError as exc:
        raise LaunchRejected("invalid UTC admission window") from exc
    need(observed <= current < expires <= observed + MAX_QUOTE_AGE,
         "account quote is stale or future-dated")
    allocation_seconds = int((deadline - observed).total_seconds())
    need(current < deadline and
         MIN_ALLOCATION_SECONDS <= allocation_seconds <= MAX_ALLOCATION_SECONDS,
         "signed allocation window outside approved 90-120 minute bounds")
    need(current < cleanup_trigger <= deadline - MIN_CLEANUP_LEAD,
         "watchdog cleanup trigger leaves insufficient deletion time")
    need(payload["subscription_id"] == subscription_id and
         payload["source_commit"] == source_commit and
         payload["region"] == "eastus" and payload["vm_sku"] == SKU and
         payload["image_urn"] == IMAGE and payload["gpu_name"] == "NVIDIA T4",
         "Azure instance or source binding mismatch")
    need(payload["model_revision"] == MODEL_REVISION and
         payload["model_manifest_sha256"] == contract.MANIFEST_SHA256["kova-cosmo"] and
         payload["dataset_sha256"] == contract.APPROVED_DATASET_SHA256 and
         type(payload["train_records"]) is int and payload["train_records"] == 27 and
         type(payload["validation_records"]) is int and payload["validation_records"] == 15,
         "model or 42-record dataset binding mismatch")
    quota = payload["quota"]
    need(type(quota) is dict and set(quota) == {
        "family_limit_vcpus", "family_used_vcpus", "regional_limit_vcpus",
        "regional_used_vcpus"}, "quota shape mismatch")
    need(all(type(v) is int and v >= 0 for v in quota.values()) and
         quota["family_limit_vcpus"] - quota["family_used_vcpus"] >= 4 and
         quota["regional_limit_vcpus"] - quota["regional_used_vcpus"] >= 4,
         "T4 or regional quota is insufficient")
    need(payload["sku_restrictions"] == [], "VM SKU has a restriction")
    for key in ("all_category_rates_checked", "watchdog_health_tested",
                "watchdog_can_deallocate_and_delete", "exclusive_pilot_group_empty",
                "no_public_ip"):
        need(payload[key] is True, "missing independent control: " + key)
    # A price list has a different provenance from Azure's public retail API.
    need(payload["account_meter_source"] == "subscription_specific_billing_price_sheet",
         "retail pricing is not an account quote")
    try:
        rate = authority.money(payload["account_compute_hourly_usd"])
    except authority.AuthorityError as exc:
        raise LaunchRejected("invalid account compute rate") from exc
    need(rate > 0, "missing account compute rate")
    cost = contract._validated_cost_guard()
    category = payload["category_upper_bounds_usd"]
    need(type(category) is dict and set(category) == set(cost["category_upper_bounds"]),
         "itemized ancillary cost evidence incomplete")
    try:
        bounds = {k: authority.money(v) for k, v in category.items()}
    except authority.AuthorityError as exc:
        raise LaunchRejected("invalid ancillary account rate") from exc
    need(all(bounds[k] <= Decimal(cost["category_upper_bounds"][k])
             for k in category), "ancillary cost exceeds category reservation")
    additional = payload["additional_cost_upper_bounds_usd"]
    need(type(additional) is dict and set(additional) == ADDITIONAL_COST_CATEGORIES,
         "controller, retention, archive or fees omitted from signed quote")
    try:
        additional_bounds = {k: authority.money(v) for k, v in additional.items()}
    except authority.AuthorityError as exc:
        raise LaunchRejected("invalid additional account cost") from exc
    need(all(value >= 0 for value in additional_bounds.values()) and
         additional_bounds["controller_runtime"] > 0 and
         additional_bounds["protected_evidence_retention"] > 0 and
         additional_bounds["external_archive_and_receipts"] > 0,
         "controller and durable evidence costs require positive reserves")
    scope = payload["evidence_scope"]
    need(type(scope) is dict and set(scope) == {
        "ledger_context_sha256", "ledger_retention_days", "artifact_container",
        "external_archive"}, "signed evidence scope missing")
    archive = scope["external_archive"]
    need(type(scope["ledger_context_sha256"]) is str and
         re.fullmatch(r"[0-9a-f]{64}", scope["ledger_context_sha256"]) and
         type(scope["ledger_retention_days"]) is int and
         1 <= scope["ledger_retention_days"] <= 90 and
         type(scope["artifact_container"]) is str and
         re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{1,61})[a-z0-9]", scope["artifact_container"]) and
         "--" not in scope["artifact_container"] and
         type(archive) is dict and set(archive) == {"uri", "retention_days", "maximum_bytes"} and
         type(archive["uri"]) is str and archive["uri"].startswith("https://") and
         type(archive["retention_days"]) is int and 1 <= archive["retention_days"] <= 365 and
         type(archive["maximum_bytes"]) is int and archive["maximum_bytes"] == 1024 * 1024,
         "invalid signed evidence scope")
    baseline = contract.admit_conditional_cosmo_pilot(
        rate, lifecycle_seconds=allocation_seconds)
    total = baseline + sum(additional_bounds.values())
    need(total <= CEILING, "complete signed cost reservation exceeds Cosmo ceiling")
    # Even when the signed quote passes, no paid operation is authorized here.
    return {
        "status": "signed_subscription_quote_checked_paid_execution_blocked",
        "quote_sha256": digest, "subscription_id": subscription_id,
        "account_hourly_compute_rate_usd": str(rate),
        "additional_cost_reservation_usd": str(sum(additional_bounds.values())),
        "evidence_scope": scope,
        "worst_case_all_in_usd": str(total),
        "all_in_ceiling_usd": str(CEILING),
        "allocation_deadline_utc": payload["allocation_deadline_utc"],
        "watchdog_cleanup_trigger_utc": payload["watchdog_cleanup_trigger_utc"],
        "signed_allocation_seconds": allocation_seconds,
        "paid_actions_enabled": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quote", type=Path)
    parser.add_argument("--subscription-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("paid execution is disabled until the independent authority and owner release exist")
    if args.quote:
        if not args.subscription_id:
            parser.error("--quote requires --subscription-id")
        result = assess_signed_quote(args.quote, source_commit=clean_source_commit(),
                                     subscription_id=args.subscription_id)
    else:
        result = proposal()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
