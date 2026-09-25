"""Synthetic CPU startup/CLI contracts; no image build, model loader or real weights."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import shutil
import tempfile
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch

from worker import model_artifact as artifact
from worker import model_startup as startup
from worker import test_model_artifact as fixtures


ROOT = Path(__file__).resolve().parents[1]


class ModelStartupTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ModelArtifactTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.controls = Path(self.fixture.directory.name) / "controls"
        self.controls.mkdir(mode=0o700)
        self.manifest = self.controls / "manifest.json"
        self.manifest.write_bytes(self.fixture.encode())
        self.policy = json.loads(startup.DEFAULT_POLICY.read_text())
        self.policy["verification_enabled"] = True
        self.policy["artifact"] = {
            "root": str(self.fixture.root), "manifest_path": str(self.manifest),
            "expected_manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "maximum_total_bytes": sum(map(len, self.fixture.files.values())),
            "timeout_seconds": 10,
        }
        self.policy_path = self.controls / "policy.json"
        self.save()

    def save(self):
        self.policy_path.write_text(json.dumps(self.policy))

    def run_gate(self, **kwargs):
        return startup.run_startup(self.policy_path, **kwargs)

    def cli(self, *args, env=None):
        return subprocess.run(
            [sys.executable, "-E", "-S", "-B", "-m", "worker.model_startup", *args],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=10, check=False,
        )

    def assert_blocked(self, report):
        self.assertIs(report["ready_for_serving"], False)
        self.assertIs(report["phase_b_ready"], False)
        self.assertIs(report["model_loaded"], False)
        self.assertEqual(report["safety"], dict.fromkeys(startup.SAFETY_FIELDS, False))

    def test_repository_policy_has_no_selected_artifact_or_execution_permission(self):
        policy = startup.validate_policy(startup.DEFAULT_POLICY.read_bytes())
        self.assertIs(policy["verification_enabled"], False)
        self.assertTrue(all(value is None for value in policy["artifact"].values()))
        with patch.object(artifact, "verify_model_artifact", side_effect=AssertionError("artifact opened")):
            for verify_only in (False, True):
                code, report = startup.run_startup(verify_only=verify_only)
                self.assertEqual(code, startup.BLOCKED_EXIT)
                self.assert_blocked(report)

    def test_source_check_succeeds_only_for_disabled_policy(self):
        code, report = startup.run_startup(check_source=True)
        self.assertEqual(code, 0)
        self.assert_blocked(report)
        with self.assertRaises(startup.StartupError):
            self.run_gate(check_source=True)

    def test_actual_artifact_verifier_is_wired_before_blocked_startup(self):
        with patch.object(artifact, "verify_model_artifact", wraps=artifact.verify_model_artifact) as verify:
            code, report = self.run_gate()
        self.assertEqual(code, startup.BLOCKED_EXIT)
        self.assert_blocked(report)
        verify.assert_called_once()
        self.assertEqual(verify.call_args.args, (self.fixture.root, self.manifest.read_bytes()))
        self.assertEqual(verify.call_args.kwargs["expected_manifest_sha256"],
                         self.policy["artifact"]["expected_manifest_sha256"])
        self.assertEqual(report["artifact_evidence"]["status"], "local_artifact_bytes_verified")

    def test_all_candidates_verify_without_becoming_a_serving_grant(self):
        for index, candidate in enumerate(fixtures.CATALOG["candidates"]):
            if index == 1:
                self.fixture.use_sharded_weights()
            elif index == 2:
                self.fixture.set_base_model(index)
            with self.subTest(candidate=candidate["id"]):
                self.manifest.write_bytes(self.fixture.encode(self.fixture.snapshot(index)))
                self.policy["artifact"]["expected_manifest_sha256"] = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
                self.policy["artifact"]["maximum_total_bytes"] = sum(map(len, self.fixture.files.values()))
                self.save()
                code, report = self.run_gate(verify_only=True)
                self.assertEqual(code, 0)
                self.assert_blocked(report)
                self.assertEqual(report["artifact_evidence"]["candidate_id"], candidate["id"])
                self.assertFalse(report["artifact_evidence"]["vendor_provenance_authenticated"])

    def test_changed_weights_fail_even_with_an_approved_manifest(self):
        (self.fixture.root / "README.md").write_bytes(b"x" * len(self.fixture.files["README.md"]))
        with self.assertRaises(artifact.ModelArtifactError):
            self.run_gate(verify_only=True)

    def test_missing_wrong_or_self_replaced_manifest_pin_fails(self):
        for pin in (None, "", "0" * 64, "sha256:" + "a" * 64):
            with self.subTest(pin=pin):
                self.policy["artifact"]["expected_manifest_sha256"] = pin
                self.save()
                with self.assertRaises((startup.StartupError, artifact.ModelArtifactError)):
                    self.run_gate(verify_only=True)

    def test_each_safety_field_is_required_and_strictly_false(self):
        for field in startup.SAFETY_FIELDS:
            for value in (True, 0, 1, None, "false", "delete"):
                policy = deepcopy(self.policy)
                if value == "delete":
                    del policy["safety"][field]
                else:
                    policy["safety"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(startup.StartupError):
                    startup.validate_policy(json.dumps(policy).encode())

    def test_unknown_or_missing_policy_fields_and_version_coercion_fail(self):
        for key in self.policy:
            policy = deepcopy(self.policy)
            del policy[key]
            with self.subTest(missing=key), self.assertRaises(startup.StartupError):
                startup.validate_policy(json.dumps(policy).encode())
        for key, value in (("schema_version", True), ("schema_version", "1"),
                           ("verification_enabled", 1), ("status", "ready"), ("execute", True)):
            policy = deepcopy(self.policy)
            policy[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(startup.StartupError):
                startup.validate_policy(json.dumps(policy).encode())

    def test_disabled_policy_cannot_hide_an_active_artifact_configuration(self):
        self.policy["verification_enabled"] = False
        with self.assertRaises(startup.StartupError):
            startup.validate_policy(json.dumps(self.policy).encode())

    def test_ambiguous_json_and_oversized_control_files_fail(self):
        for raw in (b'{"schema_version":1,"schema_version":1}', b'{"x":NaN}',
                    b'[]', b'{', b'\xff', b'x' * (startup.MAX_POLICY_BYTES + 1)):
            self.policy_path.write_bytes(raw)
            with self.subTest(raw=raw[:40]), self.assertRaises((startup.StartupError, artifact.ModelArtifactError)):
                self.run_gate()
        self.save()
        self.manifest.write_bytes(b'x' * (artifact.MAX_MANIFEST_BYTES + 1))
        with self.assertRaises(startup.StartupError):
            self.run_gate()

    def test_policy_and_manifest_cannot_live_inside_the_artifact(self):
        self.policy["artifact"]["manifest_path"] = str(self.fixture.root / "config.json")
        self.save()
        with self.assertRaises(startup.StartupError):
            self.run_gate()
        self.policy["artifact"]["manifest_path"] = str(self.manifest)
        embedded = self.fixture.root / "policy.json"
        embedded.write_text(json.dumps(self.policy))
        with self.assertRaises(startup.StartupError):
            startup.run_startup(embedded)

    def test_relative_traversal_and_url_paths_are_rejected(self):
        for key in ("root", "manifest_path"):
            for value in ("relative/file", "/tmp/a/../b", "https://example.invalid/model", "bad\x00path"):
                policy = deepcopy(self.policy)
                policy["artifact"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(startup.StartupError):
                    startup.validate_policy(json.dumps(policy).encode())

    def test_invalid_byte_and_deadline_limits_are_rejected(self):
        for key, values in (("maximum_total_bytes", (True, 0, -1, "1", 2**53)),
                            ("timeout_seconds", (True, 0, -1, "10", math.inf, math.nan, 3601))):
            for value in values:
                policy = deepcopy(self.policy)
                policy["artifact"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises((startup.StartupError, artifact.ModelArtifactError)):
                    startup.validate_policy(json.dumps(policy).encode())

    def test_policy_and_manifest_symlinks_are_rejected(self):
        for target in (self.policy_path, self.manifest):
            original = target.read_bytes()
            backing = self.controls / "backing"
            backing.write_bytes(original)
            target.unlink()
            target.symlink_to(backing)
            with self.subTest(target=target.name), self.assertRaises(startup.StartupError):
                self.run_gate()
            target.unlink()
            target.write_bytes(original)
            backing.unlink()

    def test_hard_links_and_other_writable_controls_are_rejected(self):
        for target in (self.policy_path, self.manifest):
            alias = self.controls / "hardlink"
            os.link(target, alias)
            with self.assertRaises(startup.StartupError):
                self.run_gate()
            alias.unlink()
            target.chmod(0o666)
            with self.assertRaises(startup.StartupError):
                self.run_gate()
            target.chmod(0o600)

    def test_writable_control_directory_and_fifo_do_not_pass_or_block(self):
        self.controls.chmod(0o777)
        with self.assertRaises(startup.StartupError):
            self.run_gate()
        self.controls.chmod(0o700)
        self.manifest.unlink()
        os.mkfifo(self.manifest)
        with self.assertRaises(startup.StartupError):
            self.run_gate()

    def test_control_file_mutation_is_detected_and_descriptors_are_closed(self):
        real_read, real_open, real_close = os.read, os.open, os.close
        opened, closed = [], []
        def read(fd, count):
            data = real_read(fd, count)
            if data:
                self.policy_path.write_bytes(b'x' * self.policy_path.stat().st_size)
            return data
        def tracked_open(*args, **kwargs):
            fd = real_open(*args, **kwargs)
            opened.append(fd)
            return fd
        def tracked_close(fd):
            closed.append(fd)
            return real_close(fd)
        with patch.object(os, "read", side_effect=read), patch.object(os, "open", side_effect=tracked_open), \
                patch.object(os, "close", side_effect=tracked_close):
            with self.assertRaises(startup.StartupError):
                self.run_gate()
        self.assertTrue(opened)
        self.assertEqual(sorted(opened), sorted(closed))

    def test_cancellation_and_deadline_never_report_verified_startup(self):
        with self.assertRaises(artifact.ModelArtifactCancelled):
            self.run_gate(cancelled=lambda: True)
        times = iter((0, 10))
        with self.assertRaises(artifact.ModelArtifactExpired):
            self.run_gate(clock=lambda: next(times))
        with self.assertRaises(startup.StartupError):
            self.run_gate(cancelled=lambda: "false")

    def test_run_has_no_network_subprocess_exec_or_model_import(self):
        with ExitStack() as stack:
            for owner, method in ((socket, "socket"), (subprocess, "Popen"), (os, "system"),
                                  (os, "execv"), (os, "execve"), (os, "spawnv")):
                stack.enter_context(patch.object(owner, method, side_effect=AssertionError(method)))
            stack.enter_context(patch.dict(sys.modules, dict.fromkeys(("torch", "vllm", "transformers", "huggingface_hub"))))
            code, report = self.run_gate(verify_only=True)
        self.assertEqual(code, 0)
        self.assert_blocked(report)

    def test_restarts_rehash_and_reject_new_corruption_without_cached_readiness(self):
        self.assertEqual(self.run_gate(verify_only=True)[0], 0)
        self.manifest.write_bytes(b"corrupted manifest")
        with self.assertRaises(artifact.ModelArtifactError):
            self.run_gate(verify_only=True)

    def test_no_control_or_artifact_bytes_or_permissions_are_mutated(self):
        paths = [self.policy_path, self.manifest, *self.fixture.root.iterdir()]
        before = [(path.read_bytes(), path.stat().st_mode) for path in paths]
        self.run_gate(verify_only=True)
        self.assertEqual(before, [(path.read_bytes(), path.stat().st_mode) for path in paths])

    def test_cli_default_startup_and_verification_have_distinct_exit_codes(self):
        for args, expected in (((), 78), (("--check-source",), 0),
                               (("--config", str(self.policy_path)), 1),
                               (("--verify-only", "--config", str(self.policy_path)), 1)):
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual(result.stderr, "")
                self.assert_blocked(json.loads(result.stdout))

    def test_cli_does_not_echo_failed_control_content_paths_or_tracebacks(self):
        secret = "synthetic-do-not-echo-credential"
        self.policy_path.write_text(secret)
        result = self.cli("--config", str(self.policy_path), "--verify-only")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "")
        self.assertNotIn(secret, result.stdout)
        self.assertNotIn(str(self.policy_path), result.stdout)
        self.assert_blocked(json.loads(result.stdout))

    def test_cli_environment_cannot_select_config_authorize_execution_or_inject_python(self):
        env = dict(os.environ, KOVA_MODEL_EXECUTION_ENABLED="true", KOVA_MODEL_STARTUP_CONFIG=str(self.policy_path),
                   MODEL_NAME="attacker", MODEL_REVISION="main", PYTHONPATH="/missing/injection",
                   PYTHONHOME="/missing/injection", PYTHONINSPECT="1")
        result = self.cli(env=env)
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "startup_blocked_no_approved_artifact")

    def test_cli_has_no_execute_or_loader_override(self):
        for option in ("--execute", "--serve", "--loader", "--model", "--manifest-sha256"):
            result = self.cli(option)
            self.assertEqual(result.returncode, 2)

    def test_signal_cancellation_uses_shared_verifier_control_and_restores_handlers(self):
        before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        real_verify = artifact.verify_model_artifact
        def interrupted(*args, **kwargs):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return real_verify(*args, **kwargs)
        with patch.object(artifact, "verify_model_artifact", side_effect=interrupted), redirect_stdout(io.StringIO()) as output:
            code = startup.main(["--config", str(self.policy_path), "--verify-only"])
        self.assertEqual(code, 130)
        self.assertEqual(json.loads(output.getvalue())["status"], "startup_cancelled")
        self.assertEqual(before, {sig: signal.getsignal(sig) for sig in before})

    def test_container_copy_payload_runs_standalone_without_repo_or_site_packages(self):
        lines = (ROOT / "container/model-startup.Dockerfile").read_text().splitlines()
        with tempfile.TemporaryDirectory(prefix="kova-startup-image-layout-") as folder:
            image = Path(folder)
            for line in lines:
                if line.startswith("COPY "):
                    for source in line.split()[3:-1]:
                        target = image / source
                        target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                        shutil.copyfile(ROOT / source, target)
                        target.chmod(0o444)
            command = json.loads(next(line.removeprefix("ENTRYPOINT ") for line in lines if line.startswith("ENTRYPOINT ")))
            command[0] = sys.executable
            result = subprocess.run(command, cwd=image, capture_output=True, text=True, timeout=10, check=False)
            self.assertEqual(result.returncode, 78, result.stderr)
            self.assert_blocked(json.loads(result.stdout))
            self.assertFalse(list(image.rglob("*.pyc")))
            self.assertEqual(sorted(str(p.relative_to(image)) for p in image.rglob("*") if p.is_file()),
                             sorted(("worker/__init__.py", "worker/model_artifact.py", "worker/model_startup.py",
                                     "config/core-serving.v1.json", "config/model-startup.v1.json",
                                     "core/__init__.py", "core/current_candidates.py",
                                     "release/__init__.py", "release/model_revisions.py")))

    def test_unknown_missing_artifact_fields_and_coerced_run_modes_fail(self):
        for key in self.policy["artifact"]:
            policy = deepcopy(self.policy)
            del policy["artifact"][key]
            with self.subTest(missing=key), self.assertRaises(startup.StartupError):
                startup.validate_policy(json.dumps(policy).encode())
        policy = deepcopy(self.policy)
        policy["artifact"]["loader"] = "unapproved"
        with self.assertRaises(startup.StartupError):
            startup.validate_policy(json.dumps(policy).encode())
        for options in ({"verify_only": 1}, {"check_source": "true"}, {"verify_only": True, "check_source": True}):
            with self.assertRaises(startup.StartupError):
                self.run_gate(**options)

    def test_linked_control_directory_and_missing_manifest_are_rejected(self):
        alias = Path(self.fixture.directory.name) / "alias"
        alias.symlink_to(self.controls, target_is_directory=True)
        with self.assertRaises(startup.StartupError):
            startup.run_startup(alias / self.policy_path.name)
        self.manifest.unlink()
        result = self.cli("--config", str(self.policy_path), "--verify-only")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "")
        self.assert_blocked(json.loads(result.stdout))

    def test_container_definition_uses_the_tested_entrypoint_and_no_build_or_serving_steps(self):
        lines = [line for line in (ROOT / "container/model-startup.Dockerfile").read_text().splitlines()
                 if line and not line.startswith("#")]
        self.assertEqual(lines[0:2], ["ARG APPROVED_PYTHON_BASE_IMAGE", "FROM ${APPROVED_PYTHON_BASE_IMAGE}"])
        self.assertIn("WORKDIR /opt/kova", lines)
        self.assertIn("USER 65532:65532", lines)
        entrypoint = json.loads(next(line.removeprefix("ENTRYPOINT ") for line in lines if line.startswith("ENTRYPOINT ")))
        self.assertEqual(entrypoint, ["python3", "-E", "-S", "-B", "-m", "worker.model_startup"])
        self.assertIn("CMD []", lines)
        self.assertFalse(any(line.split()[0] in ("RUN", "ADD", "HEALTHCHECK", "EXPOSE") for line in lines))
        sources = []
        for line in lines:
            if line.startswith("COPY "):
                self.assertIn("--chown=0:0 --chmod=0444", line)
                sources.extend(line.split()[3:-1])
        self.assertEqual(sorted(sources), sorted(("worker/__init__.py", "worker/model_artifact.py", "worker/model_startup.py",
                                                 "config/core-serving.v1.json", "config/model-startup.v1.json",
                                                 "core/__init__.py", "core/current_candidates.py",
                                                 "release/__init__.py", "release/model_revisions.py")))
        self.assertTrue(all((ROOT / path).is_file() for path in sources))


if __name__ == "__main__":
    unittest.main()
