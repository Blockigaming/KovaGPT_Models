"""Resolved three-family product decisions; source-only and non-executable."""
import json

from release.current_product_policy import validate as validate_policy


class DecisionRejected(ValueError):
    pass


def validate():
    report = validate_policy()
    return {
        "schema_version": 3,
        "status": "three_family_product_decisions_resolved_source_only",
        "resolved_product_decision_ids": [
            "THREE_SHARED_CHAT_WORK_FAMILIES", "NOVA_WORK_ONLY",
            "LITE_THROUGH_ULTRA_ARE_PROCESSING_CONFIGURATIONS",
            "PLAN_ENTITLEMENT_MATRIX_1_6_12_AND_0_18_18", "NO_SEPARATE_CHAT_8B_SLOT",
        ],
        "chat_route_counts_by_tier": report["chat_route_counts_by_tier"],
        "work_route_counts_by_tier": report["work_route_counts_by_tier"],
        "family_slots": report["family_slots"],
        "resource_creation_authorized": False,
        "spending_authorized": False,
        "deployment_authorized": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    }


def main():
    print(json.dumps(validate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
