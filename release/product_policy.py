"""Compatibility CLI for the authoritative v3 product policy."""
import json
import sys

from release.current_product_policy import CurrentPolicyError, validate

ProductPolicyError = CurrentPolicyError


def validate_checked_in():
    report = validate()
    return {
        **report,
        "status": "approved_three_family_product_policy_source_valid",
        "resolved_product_decision_ids": ["A36", "A37"],
        "phase_a_total": 40,
        "product_policy_ready": True,
        "finite_server_job_budget_required": True,
        "execution_integration_verified": False,
        "independent_review_verified": False,
        "phase_b_ready": False,
        "closed_checklist_ids": [],
    }


def main(args=None):
    args = sys.argv[1:] if args is None else args
    try:
        if args not in ([], ["--require-ready"]):
            raise ProductPolicyError("approved product policy rejected")
        print(json.dumps(validate_checked_in(), sort_keys=True))
        return 0
    except (ProductPolicyError, ValueError, TypeError, KeyError):
        print("approved product policy rejected", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
