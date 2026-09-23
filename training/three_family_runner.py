"""Fail-closed entrypoint for the sequential adapter pilot.

Current source intentionally supports validation/dry-run only. Paid execution
must be introduced by a separately reviewed authorization bundle after every
source and live gate is satisfied.
"""

from __future__ import annotations

import argparse
import json

from training.three_family_contract import validate

FAMILIES = ("kova-cosmo", "kova-orion", "kova-nova")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.family) == bool(args.package):
        parser.error("select exactly one family or --package")
    report = validate()
    if args.execute:
        parser.error("training blocked: exact dataset approval and paid authority are absent")
    print(json.dumps({
        "status": "dry_run_validated_execution_blocked",
        "family": args.family,
        "package": args.package,
        "training_started": False,
        "provider_calls_made": 0,
        "source": report,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
