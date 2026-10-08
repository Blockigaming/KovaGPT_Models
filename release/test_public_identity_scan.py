"""Disclosure scan regressions use artificial names only."""
from contextlib import ExitStack
from pathlib import Path
from subprocess import CompletedProcess
import tempfile
import unittest
from unittest.mock import patch
from release import public_identity_scan as scan


class PublicDiffTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(scan, "ROOT", self.root))
        self.stack.enter_context(patch.object(scan, "SURFACES", ("public.txt",)))
        (self.root / "public.txt").write_text("KovaGPT, built by Kova")
        self.policy = self.stack.enter_context(patch.object(scan, "load_catalog", return_value={
            "public_disclosure_terms": ["PRIVATE_MODEL_SOURCE"]}))
        self.stack.enter_context(patch.object(scan, "git", return_value="new.txt\n"))
        self.previous = self.stack.enter_context(patch.object(scan.subprocess, "run", return_value=
            CompletedProcess([], 128, "", "missing")))

    def test_neutral_public_source_passes(self):
        (self.root / "new.txt").write_text("internal-nova-base-v1")
        self.assertEqual(scan.scan("a" * 40)["status"], "PASS")

    def test_active_surface_fails_without_printing_matched_value(self):
        (self.root / "public.txt").write_text("PRIVATE_MODEL_SOURCE")
        with self.assertRaises(ValueError) as caught: scan.scan("a" * 40)
        self.assertNotIn("PRIVATE_MODEL_SOURCE", str(caught.exception))

    def test_new_diff_disclosure_fails(self):
        (self.root / "new.txt").write_text("private_model_source")
        with self.assertRaises(ValueError): scan.scan("a" * 40)

    def test_missing_private_rules_fail_closed(self):
        self.policy.return_value = {}
        with self.assertRaises(ValueError): scan.scan("a" * 40)

    def test_retained_identifier_in_touched_file_also_fails(self):
        (self.root / "new.txt").write_text("PRIVATE_MODEL_SOURCE\nKovaGPT")
        self.previous.return_value = CompletedProcess([], 0, "PRIVATE_MODEL_SOURCE", "")
        with self.assertRaises(ValueError): scan.scan("a" * 40)

    def test_exact_parent_required(self):
        with self.assertRaises(ValueError): scan.scan("branch-name")


if __name__ == "__main__": unittest.main()
