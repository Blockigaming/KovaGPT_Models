"""Fail-closed cost and deallocation guard for the bounded Cosmo GPU pilot.

This module validates source policy and independently signed control-plane
evidence. It does not call Azure, create resources, download weights, or start
training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import sys

from release.model_revisions import MODEL_SOURCE_REFERENCES
from training.cosmo_lifecycle_authority import (
    AuthorityError,
    PHASE_RESERVED_SECONDS,
    PHASES,
    PILOT_ID,
    load_trust_policy as load_lifecycle_trust_policy,
    read_signed_record,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config/kova-cosmo-runtime-guard.v1.json"
EVIDENCE_ENV = "KOVA_COSMO_RUNTIME_EVIDENCE"
MONEY = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{4}")
AZURE_UUID = re.compile(
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
RESOURCE_GROUP_NAME = re.compile(r"[A-Za-z0-9_.()-]{1,90}")
UTC_TIMESTAMP = re.compile(
    r"20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z"
)


class RuntimeGuardError(ValueError):
    pass


def need(condition: bool) -> None:
    if not condition:
        raise RuntimeGuardError("kova cosmo runtime guard rejected")


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        need(type(key) is str and key not in value)
        value[key] = item
    return value


def reject_constant(_value: str) -> None:
    raise RuntimeGuardError("kova cosmo runtime guard rejected")


def parse_json(raw: str) -> object:
    return json.loads(
        raw, object_pairs_hook=unique_object, parse_constant=reject_constant
    )


def money(value: object) -> Decimal:
    need(type(value) is str and MONEY.fullmatch(value) is not None)
    try:
        return Decimal(value)
    except InvalidOperation:
        raise RuntimeGuardError("kova cosmo runtime guard rejected") from None


def timestamp(value: object) -> datetime:
    need(type(value) is str and UTC_TIMESTAMP.fullmatch(value) is not None)
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        raise RuntimeGuardError("kova cosmo runtime guard rejected") from None


def nonempty(value: object, maximum: int = 512) -> bool:
    return type(value) is str and 0 < len(value) <= maximum


def load_policy(root: Path = ROOT) -> dict:
    try:
        value = parse_json(
            (root / "config/kova-cosmo-runtime-guard.v1.json").read_text(
                encoding="utf-8"
            )
        )
        need(type(value) is dict and list(value) == [
            "schema_version", "status", "model_slot", "base_model",
            "base_revision", "azure", "pilot", "pricing",
            "required_controls", "owner_approvals", "runtime_evidence",
            "phase_b_ready",
        ])
        reference = MODEL_SOURCE_REFERENCES["work-cosmo"]
        need(value["schema_version"] == 1)
        need(value["model_slot"] == reference.slot)
        need(value["base_model"] == reference.model)
        need(value["base_revision"] == reference.revision)
        need(value["azure"] == {
            "region": "eastus",
            "vm_size": "Standard_NC4as_T4_v3",
            "required_family_quota_vcpus": 4,
        })
        pilot = value["pilot"]
        need(type(pilot) is dict and list(pilot) == [
            "pilot_id", "maximum_paid_phase_grants",
            "maximum_training_runs",
            "automatic_deallocation_required", "automatic_cleanup_required",
            "cleanup_scope", "production_deployment_authorized",
            "production_integration_authorized",
        ])
        need(pilot["pilot_id"] == PILOT_ID)
        need(pilot["maximum_paid_phase_grants"] == 3)
        need(pilot["maximum_training_runs"] == 1)
        need(pilot["automatic_deallocation_required"] is True)
        need(pilot["automatic_cleanup_required"] is True)
        need(pilot["cleanup_scope"] == "pilot_resource_group")
        need(pilot["production_deployment_authorized"] is False)
        need(pilot["production_integration_authorized"] is False)

        pricing = value["pricing"]
        need(type(pricing) is dict and list(pricing) == [
            "owner_verified_compute_usd_per_hour",
            "approved_all_in_budget_usd", "maximum_allocated_minutes",
            "maximum_compute_cost_usd", "minimum_ancillary_reserve_usd",
            "budget_alert_is_hard_stop",
        ])
        hourly = money(pricing["owner_verified_compute_usd_per_hour"])
        budget = money(pricing["approved_all_in_budget_usd"])
        compute = money(pricing["maximum_compute_cost_usd"])
        reserve = money(pricing["minimum_ancillary_reserve_usd"])
        need(hourly == Decimal("0.5260"))
        need(budget == Decimal("2.0000"))
        need(pricing["maximum_allocated_minutes"] == 60)
        need(compute == hourly)
        need(compute + reserve == budget)
        need(pricing["budget_alert_is_hard_stop"] is False)

        need(value["required_controls"] == {
            "remote_append_only_paid_phase_ledger": True,
            "exact_azure_vm_identity_grant_binding": True,
            "terminal_ledger_closure_after_cleanup": True,
            "control_plane_deallocation_deadline": True,
            "independent_watchdog": True,
            "watchdog_permission_test": True,
            "automatic_cleanup_execution_provenance": True,
            "scoped_resource_group_deleted_inventory": True,
            "signed_independent_control_plane_evidence": True,
            "no_public_ip": True,
            "post_run_power_state_verification": True,
            "post_run_residual_resource_inventory": True,
        })

        approvals = value["owner_approvals"]
        need(type(approvals) is dict and list(approvals) == [
            "budget_approved", "model_download_approved",
            "bounded_training_pilot_approved", "resource_creation_release",
            "spending_release", "deployment_authorized",
        ])
        for key, item in approvals.items():
            need(type(item) is bool)
        need(approvals["budget_approved"] is True)
        need(approvals["model_download_approved"] is True)
        need(approvals["bounded_training_pilot_approved"] is True)
        need(approvals["deployment_authorized"] is False)
        released = (approvals["resource_creation_release"] is True and
                    approvals["spending_release"] is True)
        need(value["status"] == (
            "operator_preflight_required" if released
            else "budget_approved_execution_withheld"
        ))
        need(value["runtime_evidence"] == {
            "environment_variable": EVIDENCE_ENV,
            "checked_in_evidence_allowed": False,
        })
        need(value["phase_b_ready"] is False)
        return value
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError,
            RecursionError, AttributeError):
        raise RuntimeGuardError("kova cosmo runtime guard rejected") from None


def load_evidence_record(path: Path, *,
                         repository_root: Path = ROOT) -> tuple[dict, str]:
    try:
        value, evidence_sha256 = read_signed_record(
            path, expected_kind="kova_cosmo_runtime_preflight",
            repository_root=repository_root,
        )
        need(type(value) is dict and list(value) == [
            "schema_version", "kind", "issuer", "pilot_id", "lifecycle_id",
            "ledger_sequence", "provider_observation_id",
            "azure_query_source", "captured_at_utc", "subscription_id",
            "resource_group", "cleanup_scope_resource_group_id",
            "resource_group_exclusive_to_pilot", "vm_name",
            "vm_resource_id", "vm_id",
            "vm_system_assigned_identity_principal_id", "region", "vm_size",
            "family_quota_limit_vcpus", "capacity_confirmed",
            "compute_usd_per_hour", "ancillary_cost_bound_usd",
            "preflight_power_state", "public_ip_attached",
            "control_plane_deallocation_deadline_utc",
            "control_plane_deallocation_rule_id",
            "control_plane_cleanup_rule_id", "watchdog_principal",
            "watchdog_permission_tested_at_utc", "watchdog_test_result",
            "cleanup_permission_tested_at_utc", "cleanup_test_result",
        ])
        need(value["schema_version"] == 1)
        need(value["pilot_id"] == PILOT_ID)
        need(nonempty(value["lifecycle_id"]))
        need(type(value["ledger_sequence"]) is int and
             0 < value["ledger_sequence"] < 2**63)
        need(nonempty(value["provider_observation_id"]))
        need(value["azure_query_source"] ==
             "independent_azure_control_plane_reader")
        timestamp(value["captured_at_utc"])
        for key in ("subscription_id", "resource_group", "vm_name",
                    "control_plane_deallocation_rule_id",
                    "control_plane_cleanup_rule_id",
                    "watchdog_principal"):
            need(nonempty(value[key]))
        need(AZURE_UUID.fullmatch(value["subscription_id"]) is not None)
        need(RESOURCE_GROUP_NAME.fullmatch(value["resource_group"]) is not None)
        need(value["cleanup_scope_resource_group_id"] ==
             f'/subscriptions/{value["subscription_id"]}/resourceGroups/'
             f'{value["resource_group"]}')
        need(value["resource_group_exclusive_to_pilot"] is True)
        need(value["vm_resource_id"] ==
             f'/subscriptions/{value["subscription_id"]}/resourceGroups/'
             f'{value["resource_group"]}/providers/Microsoft.Compute/'
             f'virtualMachines/{value["vm_name"]}')
        need(AZURE_UUID.fullmatch(value["vm_id"]) is not None)
        need(AZURE_UUID.fullmatch(
            value["vm_system_assigned_identity_principal_id"]
        ) is not None)
        need(value["region"] == "eastus")
        need(value["vm_size"] == "Standard_NC4as_T4_v3")
        need(type(value["family_quota_limit_vcpus"]) is int)
        need(value["family_quota_limit_vcpus"] >= 4)
        need(value["capacity_confirmed"] is True)
        money(value["compute_usd_per_hour"])
        money(value["ancillary_cost_bound_usd"])
        need(value["preflight_power_state"] == "deallocated")
        need(value["public_ip_attached"] is False)
        timestamp(value["control_plane_deallocation_deadline_utc"])
        timestamp(value["watchdog_permission_tested_at_utc"])
        timestamp(value["cleanup_permission_tested_at_utc"])
        need(value["watchdog_test_result"] == "passed")
        need(value["cleanup_test_result"] == "passed")
        return value, evidence_sha256
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError,
            RecursionError, AttributeError, AuthorityError):
        raise RuntimeGuardError("kova cosmo runtime guard rejected") from None


def load_evidence(path: Path, *, repository_root: Path = ROOT) -> dict:
    return load_evidence_record(path, repository_root=repository_root)[0]


def assess_preflight(policy: dict, evidence: dict, *, now: datetime) -> dict:
    need(now.tzinfo is not None and now.utcoffset() == timedelta(0))
    approvals = policy["owner_approvals"]
    need(approvals["resource_creation_release"] is True)
    need(approvals["spending_release"] is True)
    need(approvals["deployment_authorized"] is False)

    pricing = policy["pricing"]
    need(money(evidence["compute_usd_per_hour"]) ==
         money(pricing["owner_verified_compute_usd_per_hour"]))
    need(money(evidence["ancillary_cost_bound_usd"]) <=
         money(pricing["minimum_ancillary_reserve_usd"]))

    captured = timestamp(evidence["captured_at_utc"])
    tested = timestamp(evidence["watchdog_permission_tested_at_utc"])
    cleanup_tested = timestamp(evidence["cleanup_permission_tested_at_utc"])
    deadline = timestamp(evidence["control_plane_deallocation_deadline_utc"])
    need(captured <= now <= captured + timedelta(minutes=15))
    need(tested <= now <= tested + timedelta(hours=24))
    need(cleanup_tested <= now <= cleanup_tested + timedelta(hours=24))
    need(now + timedelta(minutes=5) <= deadline)
    need(deadline <= now + timedelta(
        minutes=pricing["maximum_allocated_minutes"]
    ))
    pilot = policy["pilot"]
    return {
        "status": "ready_for_single_bounded_pilot",
        "deadline_utc": evidence["control_plane_deallocation_deadline_utc"],
        "maximum_allocated_minutes": pricing["maximum_allocated_minutes"],
        "maximum_compute_cost_usd": pricing["maximum_compute_cost_usd"],
        "approved_all_in_budget_usd": pricing["approved_all_in_budget_usd"],
        "pilot_id": pilot["pilot_id"],
        "maximum_training_runs": pilot["maximum_training_runs"],
        "remote_paid_phase_grant_required": True,
        "automatic_deallocation_required": pilot[
            "automatic_deallocation_required"
        ],
        "automatic_cleanup_required": pilot[
            "automatic_cleanup_required"
        ],
        "deployment_authorized": False,
        "production_integration_authorized": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    }


def require_ready(*, root: Path = ROOT, evidence_path: Path | None = None,
                  now: datetime | None = None) -> dict:
    policy = load_policy(root)
    if evidence_path is None:
        raw = os.environ.get(EVIDENCE_ENV)
        need(nonempty(raw, 4096))
        evidence_path = Path(raw)
    evidence, evidence_sha256 = load_evidence_record(
        evidence_path, repository_root=root
    )
    report = assess_preflight(
        policy, evidence, now=now or datetime.now(timezone.utc)
    )
    report["runtime_evidence_sha256"] = evidence_sha256
    report["lifecycle_id"] = evidence["lifecycle_id"]
    report["preflight_ledger_sequence"] = evidence["ledger_sequence"]
    report["azure_instance"] = {
        "resource_id": evidence["vm_resource_id"],
        "vm_id": evidence["vm_id"],
        "system_assigned_identity_principal_id": evidence[
            "vm_system_assigned_identity_principal_id"
        ],
    }
    return report


def verify_post_run(preflight: dict, post_run_path: Path,
                    root: Path = ROOT) -> dict:
    """Verify deallocation evidence and enumerate every residual resource."""
    try:
        value, evidence_sha256 = read_signed_record(
            post_run_path,
            expected_kind="kova_cosmo_post_run_lifecycle",
            repository_root=root,
        )
        need(type(value) is dict and list(value) == [
            "schema_version", "kind", "issuer", "pilot_id", "lifecycle_id",
            "ledger_sequence", "ledger_commit_id", "ledger_append_only",
            "ledger_status", "last_paid_grant_ledger_sequence",
            "future_grants_allowed", "phase_grants_committed",
            "training_runs_consumed", "aggregate_reserved_seconds",
            "aggregate_reserved_cost_usd", "ledger_closed_at_utc",
            "provider_observation_id",
            "azure_query_source", "observed_at_utc",
            "allocation_started_at_utc",
            "deallocated_at_utc", "subscription_id",
            "resource_group", "vm_name", "vm_resource_id", "vm_id",
            "deallocation_deadline_utc",
            "power_state", "public_ip_attached", "allocated_seconds",
            "compute_cost_upper_bound_usd",
            "ancillary_cost_observed_or_bound_usd",
            "all_in_cost_upper_bound_usd", "residual_resources",
            "automatic_deallocation_execution",
            "automatic_cleanup_execution", "scoped_inventory",
        ])
        need(value["schema_version"] == 1)
        need(value["pilot_id"] == preflight["pilot_id"] == PILOT_ID)
        need(value["lifecycle_id"] == preflight["lifecycle_id"])
        need(type(value["ledger_sequence"]) is int and
             value["ledger_sequence"] > preflight["ledger_sequence"])
        need(nonempty(value["ledger_commit_id"]))
        need(value["ledger_append_only"] is True)
        need(value["ledger_status"] ==
             "terminal_cleanup_committed_no_future_grants")
        need(type(value["last_paid_grant_ledger_sequence"]) is int and
             preflight["ledger_sequence"] <
             value["last_paid_grant_ledger_sequence"] < 2**63)
        need(value["ledger_sequence"] ==
             value["last_paid_grant_ledger_sequence"] + 1)
        need(value["future_grants_allowed"] is False)
        counts = value["phase_grants_committed"]
        need(type(counts) is dict and list(counts) == list(PHASES))
        need(counts == {phase: 1 for phase in PHASES})
        need(value["training_runs_consumed"] == 1)
        need(value["aggregate_reserved_seconds"] == sum(
            PHASE_RESERVED_SECONDS.values()
        ))
        need(money(value["aggregate_reserved_cost_usd"]) ==
             Decimal("0.5260"))
        ledger_closed = timestamp(value["ledger_closed_at_utc"])
        need(nonempty(value["provider_observation_id"]))
        need(value["provider_observation_id"] !=
             preflight["provider_observation_id"])
        need(value["azure_query_source"] ==
             "independent_azure_control_plane_reader")
        observed = timestamp(value["observed_at_utc"])
        started = timestamp(value["allocation_started_at_utc"])
        deallocated = timestamp(value["deallocated_at_utc"])
        deadline = timestamp(value["deallocation_deadline_utc"])
        need(value["subscription_id"] == preflight["subscription_id"])
        need(value["resource_group"] == preflight["resource_group"])
        need(value["vm_name"] == preflight["vm_name"])
        need(value["vm_resource_id"] == preflight["vm_resource_id"])
        need(value["vm_id"] == preflight["vm_id"])
        need(value["deallocation_deadline_utc"] ==
             preflight["control_plane_deallocation_deadline_utc"])
        need(timestamp(preflight["captured_at_utc"]) <= started)
        need(started <= deallocated <= observed)
        need(deallocated <= deadline)
        need(observed <= deadline + timedelta(minutes=15))
        allocated_seconds = int((deallocated - started).total_seconds())
        need(type(value["allocated_seconds"]) is int and
             not isinstance(value["allocated_seconds"], bool))
        need(value["allocated_seconds"] == allocated_seconds)
        policy = load_policy(root)
        pricing = policy["pricing"]
        need(0 <= allocated_seconds <=
             pricing["maximum_allocated_minutes"] * 60)
        compute = money(value["compute_cost_upper_bound_usd"])
        ancillary = money(value["ancillary_cost_observed_or_bound_usd"])
        all_in = money(value["all_in_cost_upper_bound_usd"])
        minimum_compute = (
            money(pricing["owner_verified_compute_usd_per_hour"]) *
            Decimal(allocated_seconds) / Decimal(3600)
        )
        need(compute >= minimum_compute)
        need(compute <= money(pricing["maximum_compute_cost_usd"]))
        need(ancillary <= money(pricing["minimum_ancillary_reserve_usd"]))
        need(all_in == compute + ancillary)
        need(all_in <= money(pricing["approved_all_in_budget_usd"]))
        need(value["power_state"] == "deallocated")
        need(value["public_ip_attached"] is False)
        resources = value["residual_resources"]
        need(resources == [])

        deallocation_execution = value["automatic_deallocation_execution"]
        need(type(deallocation_execution) is dict and
             list(deallocation_execution) == [
                 "mechanism", "rule_id", "execution_id", "principal",
                 "trigger", "status", "completed_at_utc", "evidence",
             ])
        need(deallocation_execution["mechanism"] == "azure_control_plane")
        need(deallocation_execution["rule_id"] ==
             preflight["control_plane_deallocation_rule_id"])
        need(nonempty(deallocation_execution["execution_id"]))
        need(deallocation_execution["principal"] ==
             preflight["watchdog_principal"])
        need(deallocation_execution["trigger"] == "deadline_rule")
        need(deallocation_execution["status"] == "succeeded")
        need(timestamp(deallocation_execution["completed_at_utc"]) ==
             deallocated)
        need(nonempty(deallocation_execution["evidence"], 2048))

        cleanup_execution = value["automatic_cleanup_execution"]
        need(type(cleanup_execution) is dict and list(cleanup_execution) == [
            "mechanism", "rule_id", "execution_id", "principal", "trigger",
            "status", "scope_resource_group_id", "completed_at_utc",
            "evidence",
        ])
        need(cleanup_execution["mechanism"] == "azure_control_plane")
        need(cleanup_execution["rule_id"] ==
             preflight["control_plane_cleanup_rule_id"])
        need(nonempty(cleanup_execution["execution_id"]))
        need(cleanup_execution["principal"] == preflight["watchdog_principal"])
        need(cleanup_execution["trigger"] == "post_run_automatic_cleanup")
        need(cleanup_execution["status"] == "succeeded")
        need(cleanup_execution["scope_resource_group_id"] ==
             preflight["cleanup_scope_resource_group_id"])
        cleanup_completed = timestamp(cleanup_execution["completed_at_utc"])
        need(deallocated <= cleanup_completed <= observed)
        need(cleanup_completed <= ledger_closed <= observed)
        need(nonempty(cleanup_execution["evidence"], 2048))

        inventory = value["scoped_inventory"]
        need(type(inventory) is dict and list(inventory) == [
            "scope_resource_group_id", "query_id", "query_succeeded",
            "queried_at_utc", "resource_group_state",
            "remaining_resource_ids", "evidence",
        ])
        need(inventory["scope_resource_group_id"] ==
             preflight["cleanup_scope_resource_group_id"])
        need(nonempty(inventory["query_id"]))
        need(inventory["query_succeeded"] is True)
        inventory_time = timestamp(inventory["queried_at_utc"])
        need(cleanup_completed <= inventory_time <= observed)
        need(ledger_closed <= inventory_time)
        need(inventory["resource_group_state"] == "deleted")
        need(inventory["remaining_resource_ids"] == [])
        need(nonempty(inventory["evidence"], 2048))
        return {
            "status": "lifecycle_cleanup_verified",
            "observed_at_utc": value["observed_at_utc"],
            "power_state": value["power_state"],
            "allocated_seconds": allocated_seconds,
            "compute_cost_upper_bound_usd": value[
                "compute_cost_upper_bound_usd"
            ],
            "ancillary_cost_observed_or_bound_usd": value[
                "ancillary_cost_observed_or_bound_usd"
            ],
            "all_in_cost_upper_bound_usd": value[
                "all_in_cost_upper_bound_usd"
            ],
            "within_approved_budget": True,
            "terminal_ledger_verified": True,
            "terminal_ledger_sequence": value["ledger_sequence"],
            "future_grants_allowed": False,
            "automatic_deallocation_verified": True,
            "automatic_cleanup_verified": True,
            "deallocation_execution_id": deallocation_execution[
                "execution_id"
            ],
            "cleanup_execution_id": cleanup_execution["execution_id"],
            "inventory_query_id": inventory["query_id"],
            "cleanup_scope_resource_group_id": inventory[
                "scope_resource_group_id"
            ],
            "resource_group_deleted": True,
            "post_run_evidence_sha256": evidence_sha256,
            "residual_resource_count": len(resources),
            "billable_residual_resource_count": sum(
                item["billable"] for item in resources
            ),
            "deployment_authorized": False,
            "phase_b_ready": False,
            "closed_checklist_ids": [],
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError,
            RecursionError, AttributeError, AuthorityError):
        raise RuntimeGuardError("kova cosmo runtime guard rejected") from None


def dry_run(root: Path = ROOT) -> dict:
    policy = load_policy(root)
    approvals = policy["owner_approvals"]
    blockers = []
    if not approvals["resource_creation_release"]:
        blockers.append("resource_creation_release_withheld")
    if not approvals["spending_release"]:
        blockers.append("spending_release_withheld")
    lifecycle_trust = load_lifecycle_trust_policy(root)
    if lifecycle_trust["status"] != "authority_pinned":
        blockers.append("lifecycle_authority_unprovisioned")
    if not os.environ.get(EVIDENCE_ENV):
        blockers.append("runtime_evidence_missing")
    return {
        "status": "blocked" if blockers else "operator_preflight_required",
        "blockers": blockers,
        "approved_all_in_budget_usd": policy["pricing"][
            "approved_all_in_budget_usd"
        ],
        "maximum_allocated_minutes": policy["pricing"][
            "maximum_allocated_minutes"
        ],
        "pilot_id": policy["pilot"]["pilot_id"],
        "maximum_paid_phase_grants": policy["pilot"][
            "maximum_paid_phase_grants"
        ],
        "maximum_training_runs": policy["pilot"]["maximum_training_runs"],
        "remote_append_only_paid_phase_ledger_required": True,
        "automatic_deallocation_required": policy["pilot"][
            "automatic_deallocation_required"
        ],
        "automatic_cleanup_required": policy["pilot"][
            "automatic_cleanup_required"
        ],
        "deployment_authorized": False,
        "production_integration_authorized": False,
        "resource_created": False,
        "spending_started": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--require-ready", action="store_true")
    actions.add_argument("--verify-post-run", type=Path,
                         help="External post-run deallocation/cost evidence")
    parser.add_argument("--preflight-evidence", type=Path,
                        help="Exact external preflight evidence used by the run")
    arguments = parser.parse_args(argv)
    try:
        if arguments.verify_post_run is not None:
            need(arguments.preflight_evidence is not None)
            preflight, preflight_sha256 = load_evidence_record(
                arguments.preflight_evidence
            )
            report = verify_post_run(preflight, arguments.verify_post_run)
            report["preflight_evidence_sha256"] = preflight_sha256
        else:
            need(arguments.preflight_evidence is None)
            report = require_ready() if arguments.require_ready else dry_run()
        print(json.dumps(report, sort_keys=True))
        return 0
    except (RuntimeGuardError, AuthorityError):
        print("kova cosmo runtime guard rejected", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
