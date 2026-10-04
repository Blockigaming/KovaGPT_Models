"""Importing the test entry point in a spawned worker must not run tests again."""
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

RUNNER = Path(__file__).resolve().parents[1] / "scripts/run-source-tests.py"


class SourceTestRunnerTests(unittest.TestCase):
    def test_spawned_worker_import_does_not_restart_suites(self):
        with patch("unittest.main") as main:
            runpy.run_path(str(RUNNER), run_name="__mp_main__")
            main.assert_not_called()

    def test_normal_import_has_no_execution(self):
        with patch("unittest.main") as main:
            runpy.run_path(str(RUNNER), run_name="source_test_entrypoint")
            main.assert_not_called()

    def test_explicit_entrypoint_runs_every_discovered_suite_once(self):
        with patch("unittest.main") as main:
            namespace = runpy.run_path(str(RUNNER), run_name="__main__")
            main.assert_called_once()
            selected = main.call_args.kwargs["argv"][1:]
            self.assertEqual(selected, namespace["modules"])
            self.assertEqual(len(selected), len(set(selected)))
            self.assertIn("core.test_public_identity", selected)
            self.assertIn("execution.test_postgres_supervisor", selected)


if __name__ == "__main__": unittest.main()
