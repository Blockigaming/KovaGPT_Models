"""Render the reviewed Azure pilot sequence; dry-run is the only current mode."""

from __future__ import annotations

import argparse
import json


def command_plan() -> list[dict]:
    # Shell-free argv arrays make substitution and review explicit. Placeholders
    # are resolved only by a future authorized controller, never by this module.
    return [
        {"id": 1, "name": "recheck_live_price", "argv": ["az", "rest", "--method", "get", "--url", "${RETAIL_PRICE_URL}"], "mode": "read_only"},
        {"id": 2, "name": "recheck_quota", "argv": ["az", "vm", "list-usage", "--location", "eastus", "--output", "json"], "mode": "read_only"},
        {"id": 3, "name": "check_t4_capacity", "argv": ["az", "deployment", "sub", "what-if", "--location", "eastus", "--template-file", "infra/three-family-pilot-vm.bicep"], "mode": "what_if"},
        {"id": 4, "name": "provision_independent_watchdog", "argv": ["az", "deployment", "sub", "what-if", "--location", "eastus", "--template-file", "infra/three-family-watchdog.bicep"], "mode": "what_if"},
        {"id": 5, "name": "create_single_standard_nc4as_t4_v3", "argv": ["az", "deployment", "sub", "what-if", "--location", "eastus", "--template-file", "infra/three-family-pilot-vm.bicep"], "mode": "what_if"},
        {"id": 6, "name": "confirm_exact_nvidia_t4", "argv": ["python3", "-m", "training.three_family_contract", "--probe-evidence", "${PROBE_EVIDENCE}"], "mode": "validate_only"},
        {"id": 7, "name": "download_and_hash_verify_allowlist", "argv": ["python3", "-m", "training.snapshot_verifier", "--root", "${MODEL_SNAPSHOT_ROOT}"], "mode": "validate_only_no_download"},
        {"id": 8, "name": "run_compatibility_probes", "argv": ["python3", "-m", "training.three_family_contract", "--probe-evidence", "${PROBE_EVIDENCE}"], "mode": "validate_only"},
        {"id": 9, "name": "train_cosmo", "argv": ["python3", "-m", "training.three_family_runner", "--family", "kova-cosmo", "--dry-run"], "mode": "dry_run"},
        {"id": 10, "name": "evaluate_and_preserve_cosmo", "argv": ["python3", "-m", "evaluation.three_family_guard", "--family", "kova-cosmo", "--dry-run"], "mode": "dry_run"},
        {"id": 11, "name": "train_orion", "argv": ["python3", "-m", "training.three_family_runner", "--family", "kova-orion", "--dry-run"], "mode": "dry_run"},
        {"id": 12, "name": "evaluate_and_preserve_orion", "argv": ["python3", "-m", "evaluation.three_family_guard", "--family", "kova-orion", "--dry-run"], "mode": "dry_run"},
        {"id": 13, "name": "conditionally_train_nova", "argv": ["python3", "-m", "training.three_family_runner", "--family", "kova-nova", "--dry-run"], "mode": "dry_run_fail_closed"},
        {"id": 14, "name": "evaluate_and_preserve_nova", "argv": ["python3", "-m", "evaluation.three_family_guard", "--family", "kova-nova", "--dry-run"], "mode": "dry_run"},
        {"id": 15, "name": "package_adapters_and_evidence", "argv": ["python3", "-m", "training.three_family_runner", "--package", "--dry-run"], "mode": "dry_run"},
        {"id": 16, "name": "deallocate_and_delete_everything", "argv": ["az", "group", "delete", "--name", "${PILOT_RESOURCE_GROUP}", "--yes", "--no-wait"], "mode": "print_only_destructive"},
        {"id": 17, "name": "confirm_zero_residual_resources", "argv": ["az", "resource", "list", "--resource-group", "${PILOT_RESOURCE_GROUP}", "--output", "json"], "mode": "read_only"},
        {"id": 18, "name": "reconcile_final_charge", "argv": ["az", "costmanagement", "query", "--type", "ActualCost", "--scope", "${SUBSCRIPTION_SCOPE}"], "mode": "read_only"},
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("paid execution is source-blocked; a future authorized controller is required")
    print(json.dumps({"dry_run": True, "commands_executed": 0, "steps": command_plan()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
