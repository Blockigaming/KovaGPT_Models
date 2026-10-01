"""Strict watchdog comparison with bounded, non-secret mismatch evidence."""

from copy import deepcopy
import hashlib
import json
import logging
import re

from training.cosmo_controller_ledger import LedgerRejected


COMPARISON_RULE = "canonical_json_exact_after_redundant_metadata_normalization"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def safe_value(value):
    # Never echo arbitrary workflow values, URLs, headers or credentials.
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is str and (value in {"Enabled", "Disabled", "Succeeded", "Failed",
            "String", "SecureString", "Http", "GET", "POST", "DELETE", "Minute"}
            or re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", value)):
        return value
    return {"type": type(value).__name__, "sha256": fingerprint(value)}


def verify_watchdog_properties(properties, deadline, definition):
    """Only redundant String typing / identical evaluated recurrence may vary.

    Azure's WorkflowParameter has optional type metadata. Its documented
    evaluatedRecurrence is redundant only when it exactly equals the pinned
    recurrence. Every other field in this safety projection remains exact.
    """
    expected = {"state": "Enabled", "provisioningState": "Succeeded",
                "parameters": {"deadlineUtc": {"value": deadline}}, "definition": definition}
    observed = {k: properties.get(k) for k in expected} if type(properties) is dict else properties
    actual = deepcopy(observed)
    if type(actual) is dict:
        parameters = actual.get("parameters")
        parameter = parameters.get("deadlineUtc") if type(parameters) is dict else None
        if type(parameter) is dict and parameter.get("type") == "String":
            parameter.pop("type")
        d = actual.get("definition")
        triggers = d.get("triggers") if type(d) is dict else None
        trigger = triggers.get("every_minute") if type(triggers) is dict else None
        if type(trigger) is dict and "evaluatedRecurrence" in trigger:
            pinned = definition["triggers"]["every_minute"]["recurrence"]
            if encoded(trigger["evaluatedRecurrence"]) == encoded(pinned):
                trigger.pop("evaluatedRecurrence")
    if encoded(actual) == encoded(expected):
        return
    mismatches = []

    def diff(want, got, path):
        if len(mismatches) >= 64:
            return
        if type(want) is dict and type(got) is dict:
            for key in sorted(set(want) | set(got)):
                child = path + "/" + key.replace("~", "~0").replace("/", "~1")
                if key not in want or key not in got:
                    mismatches.append({"path": child, "kind": "unexpected" if key not in want else "missing",
                        "expected": safe_value(want.get(key)), "actual": safe_value(got.get(key))})
                else:
                    diff(want[key], got[key], child)
                if len(mismatches) >= 64:
                    break
        elif type(want) is list and type(got) is list and len(want) == len(got):
            for i, (left, right) in enumerate(zip(want, got)):
                diff(left, right, path + "/" + str(i))
        elif encoded(want) != encoded(got):
            mismatches.append({"path": path, "kind": "changed",
                               "expected": safe_value(want), "actual": safe_value(got)})

    diff(expected, actual, "/properties")
    for mismatch in mismatches:
        mismatch["comparison_rule"] = COMPARISON_RULE
        if mismatch["path"] == "/properties/parameters/deadlineUtc/type":
            mismatch["comparison_rule"] = "optional_parameter_type_must_be_String"
        elif mismatch["path"] == "/properties/definition/triggers/every_minute/evaluatedRecurrence":
            mismatch["comparison_rule"] = "optional_evaluated_recurrence_must_exactly_equal_pinned_recurrence"
    evidence = {"event": "watchdog_readback_mismatch", "mismatches": mismatches,
                "expected_sha256": fingerprint(expected), "observed_sha256": fingerprint(observed),
                "normalized_actual_sha256": fingerprint(actual)}
    message = encoded(evidence)
    logging.getLogger(__name__).error("%s", message)
    error = LedgerRejected(message)
    error.evidence = evidence
    raise error
