import unittest

from evaluation.offline import (
    build_route_manifest,
    run_offline_suite,
    validate_activity_events,
    validate_response_artifact,
)


class OfflineEvaluationTests(unittest.TestCase):
    def test_manifest_has_exact_product_complete_route_set(self):
        manifest = build_route_manifest()
        self.assertEqual(len(manifest), 37)
        self.assertEqual(len({route["route_id"] for route in manifest}), 37)
        self.assertEqual(sum(route["engine"] == "kova-core" for route in manifest), 30)
        self.assertEqual(sum(route["engine"] == "kova-ultra" for route in manifest), 6)
        self.assertEqual(sum(route["engine"] == "server-selected" for route in manifest), 1)

    def test_work_family_behavior_is_not_name_only(self):
        manifest = build_route_manifest()
        by_family = {}
        for route in manifest:
            if route["route_id"].startswith("work:"):
                by_family.setdefault(route["profile"], set()).add(route["behavior_contract_id"])
        self.assertEqual(set(by_family), {"cosmo", "orion", "nova"})
        self.assertTrue(all(len(contracts) == 1 for contracts in by_family.values()))
        self.assertEqual(len({next(iter(contracts)) for contracts in by_family.values()}), 3)

    def test_response_evaluator_requires_grounded_sources_and_private_reasoning(self):
        hidden = {
            "text": "<think>secret</think> answer",
            "identity_requested": False,
            "provider_disclosure_requested": False,
            "tool_claims": [],
            "runtime_tool_results": [],
            "user_source_urls": [],
        }
        fabricated = {
            "text": "I searched https://example.com/fake.",
            "identity_requested": False,
            "provider_disclosure_requested": False,
            "tool_claims": [{"operation_id": "missing", "tool": "web", "source_url": "https://example.com/fake"}],
            "runtime_tool_results": [],
            "user_source_urls": [],
        }
        self.assertIn("forbidden_response_pattern:<think", validate_response_artifact(hidden))
        self.assertTrue(any(item.startswith("ungrounded_source_url:") for item in validate_response_artifact(fabricated)))
        empty_disclosure = {
            "text": "arbitrary response", "identity_requested": False,
            "provider_disclosure_requested": True, "selected_provider": "",
            "selected_upstream_model": "", "tool_claims": [],
            "runtime_tool_results": [], "user_source_urls": [],
        }
        self.assertIn("selected_provider_missing", validate_response_artifact(empty_disclosure))
        self.assertIn("selected_upstream_model_missing", validate_response_artifact(empty_disclosure))

    def test_activity_evaluator_requires_real_started_operation(self):
        events = [{
            "event_id": "event-1",
            "request_id": "request-1",
            "sequence": 1,
            "phase": "checking",
            "title": "Checking",
            "summary": "Reviewing evidence.",
            "occurred_at": "2026-09-14T00:00:00Z",
            "grounding_operation_id": "op-1",
        }]
        runtime = [{
            "operation_id": "op-1",
            "request_id": "request-1",
            "phase": "checking",
            "started_at": "2026-09-14T00:00:01Z",
            "status": "started",
            "source_urls": [],
        }]
        self.assertIn("event_1_precedes_runtime_start", validate_activity_events("high", events, runtime))
        self.assertIn("route_forbids_activity", validate_activity_events("instant", events, runtime))
        timezone_less_runtime = [{**runtime[0], "started_at": "2026-09-14T00:00:00"}]
        self.assertIn("event_1_invalid_runtime_start", validate_activity_events("high", events, timezone_less_runtime))

    def test_complete_offline_suite_passes_without_releasing_routes(self):
        report = run_offline_suite()
        self.assertEqual(report["status"], "offline_contracts_passed_release_still_blocked")
        self.assertEqual(report["route_contracts_checked"], 37)
        self.assertEqual(report["paid_provider_calls"], 0)
        self.assertFalse(report["actual_model_outputs_evaluated"])
        self.assertFalse(report["quality_or_factuality_claimed"])
        self.assertEqual(report["passing_release_routes"], [])


if __name__ == "__main__":
    unittest.main()
