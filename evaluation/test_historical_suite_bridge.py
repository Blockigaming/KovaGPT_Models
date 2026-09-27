"""Archived cases are source data only; no provider or human review is run."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation import historical_suite_bridge as bridge
from evaluation.quality_evidence import EvidenceRejected, analyze, digest


class HistoricalSuiteBridgeTests(unittest.TestCase):
    def test_exact_source_and_lossless_case_fields(self):
        raw = bridge.SUITE_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), bridge.HISTORICAL_FILE_SHA256)
        original = bridge.load_archived_suite()
        self.assertEqual(digest(original), bridge.HISTORICAL_CONTENT_SHA256)
        mapped = bridge.load_mapped_suite()
        self.assertEqual(digest(mapped), bridge.MAPPED_SUITE_SHA256)
        self.assertEqual(mapped["provenance"], "operator_supplied_unverified")
        self.assertEqual(Counter(c["category"] for c in original["cases"]), bridge.EXPECTED_CATEGORIES)
        self.assertEqual(len(original["cases"]), len(mapped["cases"]))
        for historical, current in zip(original["cases"], mapped["cases"]):
            self.assertEqual((historical["id"], historical["prompt"]), (current["id"], current["prompt"]))
            old_score = historical["evaluation"]
            if old_score["kind"] == "exact_json":
                self.assertEqual(current["scorer"], "exact_json")
                self.assertEqual(current["expected_json"], old_score["expected"])
                self.assertEqual(current["rubric"], [])
            else:
                self.assertEqual(current["scorer"], "human_rubric")
                self.assertIsNone(current["expected_json"])
                self.assertEqual(current["rubric"], old_score["criteria"])
            self.assertEqual(current["source_urls"], [])

    def test_empty_bundle_keeps_every_case_route_condition_missing(self):
        suite = bridge.load_mapped_suite()
        commit = "a" * 40  # Source-only fixture pin, never claimed as a real commit.
        bundle = {"schema_version": 1, "kind": "recorded_unverified",
                  "suite_sha256": bridge.MAPPED_SUITE_SHA256,
                  "source_commit": commit, "attempts": [], "reviews": []}
        result = analyze(suite, bundle, expected_suite_sha256=bridge.MAPPED_SUITE_SHA256,
                         expected_source_commit=commit)
        self.assertEqual(result["expected_units"], 3700)
        self.assertEqual(result["unit_counts"], {"missing": 3700})
        self.assertEqual((result["attempts"], result["provided_reviews"], result["model_calls_made"]), (0, 0, 0))
        self.assertFalse(result["historical_suite_reconciled"])
        self.assertFalse(result["provenance_authenticated"])
        self.assertFalse(result["phase_b_ready"])
        self.assertEqual(result["live_routes_verified"], [])

    def test_modified_archived_bytes_fail_closed(self):
        raw = bridge.SUITE_PATH.read_bytes()
        with TemporaryDirectory() as folder:
            altered = Path(folder) / "altered.json"
            altered.write_bytes(raw.replace(b"kova-model-quality-v1", b"kova-model-quality-v2", 1))
            with patch.object(bridge, "SUITE_PATH", altered), self.assertRaises(EvidenceRejected):
                bridge.load_archived_suite()

    def test_source_cli_reports_counts_and_pins_without_prompt_bodies(self):
        result = subprocess.run([sys.executable, "-m", "evaluation.historical_suite_bridge"],
                                cwd=bridge.ROOT, capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report["case_count"], 50)
        self.assertEqual((report["exact_json_cases"], report["manual_review_cases"]), (36, 14))
        self.assertEqual((report["route_contracts"], report["expected_cold_warm_units"]), (37, 3700))
        self.assertEqual(report["model_calls_made"], 0)
        self.assertFalse(report["historical_suite_reconciled"])
        self.assertFalse(report["phase_b_ready"])
        self.assertNotIn(bridge.load_archived_suite()["cases"][0]["prompt"], result.stdout)


if __name__ == "__main__":
    unittest.main()
