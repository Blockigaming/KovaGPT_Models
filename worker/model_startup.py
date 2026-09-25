"""CPU-only container startup gate. There is deliberately no serving/exec path."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import stat
from time import monotonic

from worker import model_artifact as artifact


DEFAULT_POLICY = Path(__file__).resolve().parents[1] / "config/model-startup.v1.json"
MAX_POLICY_BYTES = 16 * 1024
SAFETY_FIELDS = (
    "model_loading_authorized", "network_access_authorized", "image_build_authorized",
    "gpu_execution_authorized", "deployment_authorized", "production_routing_authorized",
)
ARTIFACT_FIELDS = (
    "root", "manifest_path", "expected_manifest_sha256", "maximum_total_bytes", "timeout_seconds",
)
BLOCKED_EXIT = 78


class StartupError(ValueError):
    """Sanitized startup-policy or control-file failure."""


def _require(condition):
    if not condition:
        raise StartupError("startup configuration rejected")


def _canonical(value):
    _require(isinstance(value, str) and value and "\x00" not in value
             and os.path.isabs(value) and os.path.normpath(value) == value)
    return Path(value)


def validate_policy(encoded):
    """Permit only verification. No policy or environment setting enables serving."""
    _require(isinstance(encoded, bytes) and 0 < len(encoded) <= MAX_POLICY_BYTES)
    policy = artifact._json(encoded)
    _require(set(policy) == {"schema_version", "status", "verification_enabled", "artifact", "safety"})
    _require(type(policy["schema_version"]) is int and policy["schema_version"] == 1)
    _require(policy["status"] == "source_only_serving_blocked")
    _require(type(policy["verification_enabled"]) is bool)
    safety = policy["safety"]
    _require(isinstance(safety, dict) and set(safety) == set(SAFETY_FIELDS))
    _require(all(safety[field] is False for field in SAFETY_FIELDS))
    package = policy["artifact"]
    _require(isinstance(package, dict) and set(package) == set(ARTIFACT_FIELDS))
    if not policy["verification_enabled"]:
        _require(all(value is None for value in package.values()))
        return policy
    root = _canonical(package["root"])
    manifest = _canonical(package["manifest_path"])
    _require(manifest != root and root not in manifest.parents)
    pin = package["expected_manifest_sha256"]
    _require(isinstance(pin, str) and artifact.SHA256.fullmatch(pin))
    maximum = package["maximum_total_bytes"]
    _require(type(maximum) is int and 0 < maximum <= 2**53 - 1)
    timeout = package["timeout_seconds"]
    _require(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 3600)
    return policy


def _control_bytes(path, limit, cancelled):
    """Read a bounded, protected local policy/manifest without following file links.

    Parent hierarchy is trusted server configuration, as in the artifact verifier.
    This is not signature verification or protection against a privileged host.
    """
    path = _canonical(str(path))
    def check():
        state = cancelled()
        _require(type(state) is bool)
        if state:
            raise artifact.ModelArtifactCancelled("startup verification cancelled")
    check()
    _require(str(path.resolve(strict=True)) == str(path))
    parent = os.lstat(path.parent)
    _require(stat.S_ISDIR(parent.st_mode) and not parent.st_mode & 0o022)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _require(artifact._identity(os.fstat(descriptor)) == artifact._identity(parent))
        before = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                 and not before.st_mode & 0o022 and 0 < before.st_size <= limit)
        _require(before.st_uid in (0, os.geteuid()))
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        try:
            _require(artifact._identity(os.fstat(fd)) == artifact._identity(before))
            chunks, size = [], 0
            while True:
                check()
                chunk = os.read(fd, min(65536, limit - size + 1))
                check()
                if not chunk:
                    break
                size += len(chunk)
                _require(size <= limit)
                chunks.append(chunk)
            _require(size == before.st_size)
            _require(artifact._identity(os.fstat(fd)) == artifact._identity(before))
            _require(artifact._identity(os.stat(path.name, dir_fd=descriptor, follow_symlinks=False))
                     == artifact._identity(before))
            _require(artifact._identity(os.lstat(path.parent)) == artifact._identity(parent))
            return b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(descriptor)


def _report(status, evidence=None):
    report = {
        "status": status, "ready_for_serving": False, "phase_b_ready": False,
        "model_loaded": False, "safety": {field: False for field in SAFETY_FIELDS},
    }
    if evidence is not None:
        report["artifact_evidence"] = evidence
    return report


def run_startup(policy_path=DEFAULT_POLICY, *, verify_only=False, check_source=False,
                cancelled=lambda: False, clock=monotonic):
    """Startup always exits blocked; explicit offline verification can exit zero.

    Both paths return readiness=false. Call only from trusted server/operator code,
    never with a request-supplied path, manifest digest or verification budget.
    """
    _require(type(verify_only) is bool and type(check_source) is bool
             and not (verify_only and check_source) and callable(cancelled) and callable(clock))
    policy = validate_policy(_control_bytes(policy_path, MAX_POLICY_BYTES, cancelled))
    if check_source:
        _require(not policy["verification_enabled"])
        return 0, _report("source_contract_valid_serving_blocked")
    if not policy["verification_enabled"]:
        return BLOCKED_EXIT, _report("startup_blocked_no_approved_artifact")
    package = policy["artifact"]
    root = Path(package["root"])
    _require(Path(policy_path) != root and root not in Path(policy_path).parents)
    manifest = _control_bytes(package["manifest_path"], artifact.MAX_MANIFEST_BYTES, cancelled)
    evidence = artifact.verify_model_artifact(
        root, manifest, expected_manifest_sha256=package["expected_manifest_sha256"],
        maximum_total_bytes=package["maximum_total_bytes"], timeout_seconds=package["timeout_seconds"],
        cancelled=cancelled, clock=clock,
    )
    return (0 if verify_only else BLOCKED_EXIT), _report("artifact_verified_serving_blocked", evidence)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline artifact startup gate; never starts a model server.")
    parser.add_argument("--config", default=str(DEFAULT_POLICY), help="Trusted absolute server policy path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="Return zero for byte verification, NOT readiness")
    mode.add_argument("--check-source", action="store_true", help="Check disabled source policy without artifacts")
    args = parser.parse_args(argv)
    stopped = [False]
    previous = {}
    def stop(_signum, _frame):
        stopped[0] = True
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop)
        code, report = run_startup(args.config, verify_only=args.verify_only,
                                  check_source=args.check_source, cancelled=lambda: stopped[0])
    except artifact.ModelArtifactCancelled:
        code, report = 130, _report("startup_cancelled")
    except Exception:
        # Never print control-file content, filesystem paths, credentials or tracebacks.
        code, report = 1, _report("startup_verification_failed")
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    print(json.dumps(report, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
