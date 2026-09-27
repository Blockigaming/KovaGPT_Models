"""Supplied-evidence consistency checks, never actual tool execution/approval."""

from copy import deepcopy
import unittest

from evaluation.offline import SUITE, validate_activity_events, validate_response_artifact


def response():
    return deepcopy(next(c["artifact"] for c in SUITE["response_cases"] if c["id"] == "grounded_tool_claim"))


def activity():
    case = next(c for c in SUITE["activity_cases"] if c["id"] == "deep_grounded_event")
    return deepcopy(case["events"]), deepcopy(case["runtime_operations"])


class GroundingIntegrityTests(unittest.TestCase):
    def test_conflicting_receipts_cannot_overwrite_failed_attempts_in_either_order(self):
        for reverse in (False, True):
            value = response()
            good = value["runtime_tool_results"][0]
            value["runtime_tool_results"] = [{**good, "status": "failed", "source_urls": []}, good]
            if reverse:
                value["runtime_tool_results"].reverse()
            self.assertIn("duplicate_runtime_tool_operation_id", validate_response_artifact(value))

    def test_identical_duplicate_receipts_are_not_silently_collapsed(self):
        value = response()
        value["runtime_tool_results"] *= 2
        self.assertIn("duplicate_runtime_tool_operation_id", validate_response_artifact(value))

    def test_nonobject_or_missing_id_receipts_are_not_silently_dropped(self):
        for invalid in (None, 1, [], {}, {"operation_id": []}):
            value = response()
            value["runtime_tool_results"].append(invalid)
            self.assertTrue(validate_response_artifact(value))

    def test_tool_claims_need_unique_ids_and_explicit_nonempty_tools(self):
        for change in ("duplicate", "missing_tool", "empty_tool", "missing_id"):
            value = response()
            if change == "duplicate":
                value["tool_claims"] *= 2
            elif change == "missing_id":
                del value["tool_claims"][0]["operation_id"]
            else:
                for field in ("tool_claims", "runtime_tool_results"):
                    if change == "missing_tool":
                        del value[field][0]["tool"]
                    else:
                        value[field][0]["tool"] = ""
            self.assertTrue(validate_response_artifact(value))

    def test_malformed_response_collections_return_violations_without_type_errors(self):
        for field in ("tool_claims", "runtime_tool_results", "user_source_urls"):
            for invalid in (None, 7, "not-an-array", {}):
                value = response()
                value[field] = invalid
                self.assertTrue(validate_response_artifact(value))

    def test_receipt_url_container_cannot_use_substring_membership(self):
        value = response()
        value["runtime_tool_results"][0]["source_urls"] = "https://example.com/source-and-more"
        self.assertTrue(validate_response_artifact(value))
        events, operations = activity()
        operations[0]["source_urls"] = "https://example.com/source-and-more"
        self.assertTrue(validate_activity_events("high", events, operations))

    def test_invalid_user_urls_and_claim_identifiers_do_not_crash(self):
        for invalid in ([], {}, None, 1, "\ud800"):
            value = response()
            value["user_source_urls"] = [invalid]
            self.assertTrue(validate_response_artifact(value))
            value = response()
            value["tool_claims"][0]["operation_id"] = invalid
            self.assertTrue(validate_response_artifact(value))

    def test_requested_identity_flags_cannot_use_boolean_lookalikes(self):
        for field in ("identity_requested", "provider_disclosure_requested"):
            for invalid in (0, 1, "false", None):
                value = response()
                value[field] = invalid
                self.assertIn("invalid_" + field, validate_response_artifact(value))

    def test_populated_private_reasoning_fields_cannot_receive_a_pass(self):
        for field in ("reasoning", "reasoning_content", "reasoning_details"):
            value = response()
            value[field] = "SYNTHETIC-PRIVATE-MARKER"
            issues = validate_response_artifact(value)
            self.assertIn("private_response_field:" + field, issues)
            self.assertNotIn("SYNTHETIC-PRIVATE-MARKER", str(issues))

    def test_ambiguous_runtime_operations_cannot_ground_activity(self):
        events, operations = activity()
        original = operations[0]
        operations.insert(0, {**original, "started_at": "2026-09-14T01:00:00Z"})
        self.assertIn("duplicate_runtime_activity_operation_id", validate_activity_events("high", events, operations))

    def test_prestart_or_unknown_status_cannot_ground_started_activity(self):
        for state in ("planned", "queued", "pending", "skipped", "invented", None, True):
            events, operations = activity()
            operations[0]["status"] = state
            self.assertIn("event_1_runtime_not_started", validate_activity_events("high", events, operations))

    def test_failed_started_operation_can_still_ground_honest_non_success_activity(self):
        for state in ("started", "running", "success", "failed", "cancelled", "expired", "interrupted", "uncertain"):
            events, operations = activity()
            del events[0]["source_url"]
            operations[0]["status"] = state
            self.assertEqual(validate_activity_events("high", events, operations), [])

    def test_activity_sequence_requires_integer_not_bool_or_float(self):
        for invalid in (True, 1.0, "1"):
            events, operations = activity()
            events[0]["sequence"] = invalid
            self.assertIn("event_1_invalid_sequence", validate_activity_events("high", events, operations))

    def test_unhashable_event_and_grounding_ids_produce_violations_not_crashes(self):
        for field in ("event_id", "grounding_operation_id"):
            for invalid in ([], {}, None, "\ud800"):
                events, operations = activity()
                events[0][field] = invalid
                self.assertTrue(validate_activity_events("high", events, operations))

    def test_optional_grounding_fields_require_exact_observed_values_and_types(self):
        for field, good in (("result_count", 3), ("domain", "example.com"),
                            ("content_type", "text/html"), ("icon_key", "web-source")):
            events, operations = activity()
            events[0][field] = good
            self.assertTrue(validate_activity_events("high", events, operations))
            operations[0][field] = good
            self.assertEqual(validate_activity_events("high", events, operations), [])
            operations[0][field] = "different"
            self.assertTrue(validate_activity_events("high", events, operations))
        for invalid in (True, -1, 3.0, "3"):
            events, operations = activity()
            events[0]["result_count"] = operations[0]["result_count"] = invalid
            self.assertTrue(validate_activity_events("high", events, operations))

    def test_one_activity_delivery_cannot_mix_request_ids(self):
        events, operations = activity()
        other_event, other_operation = deepcopy(events[0]), deepcopy(operations[0])
        other_event.update(event_id="event-2", sequence=2, request_id="other-request", grounding_operation_id="other-op")
        other_operation.update(operation_id="other-op", request_id="other-request")
        self.assertIn("event_2_request_stream_mismatch", validate_activity_events("high", events + [other_event], operations + [other_operation]))

    def test_invalid_runtime_rows_do_not_disappear_from_activity_evaluation(self):
        events, operations = activity()
        for invalid in (None, 2, {}, {"operation_id": []}):
            self.assertTrue(validate_activity_events("high", events, operations + [invalid]))

    def test_work_activity_remains_allowed_while_instant_and_medium_remain_silent(self):
        events, operations = activity()
        for route in ("instant", "medium"):
            self.assertIn("route_forbids_activity", validate_activity_events(route, events, operations))
        for family in ("cosmo", "orion", "nova"):
            for effort in ("light", "medium", "high", "extra-high", "max", "ultra"):
                self.assertEqual(validate_activity_events(f"work:{family}:{effort}", events, operations), [])


if __name__ == "__main__":
    unittest.main()
