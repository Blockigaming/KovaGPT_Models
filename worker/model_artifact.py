"""Read-only validation of a separately approved, locally packaged model snapshot.

No model loading, network, credentials, downloads, subprocesses or package installs.
The expected manifest digest MUST come from trusted server/release configuration,
not from the same untrusted artifact location or a user's request. Hash agreement
is byte integrity, not proof of vendor provenance, compatibility or GPU readiness.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from time import monotonic


MAX_MANIFEST_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_FILES = 512
READ_BYTES = 1024 * 1024
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}\Z")
REQUIRED_FILES = frozenset((
    "config.json", "tokenizer.json", "tokenizer_config.json", "adapter_config.json",
))
PARSED_FILES = frozenset(("config.json", "tokenizer_config.json", "adapter_config.json", "model.safetensors.index.json"))
OPTIONAL_FILES = frozenset((
    "chat_template.jinja", "generation_config.json", "preprocessor_config.json",
    "processor_config.json", "video_preprocessor_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "README.md", "LICENSE", "LICENSE.txt",
))


class ModelArtifactError(ValueError):
    """An artifact failed its approved byte-integrity contract."""


class ModelArtifactCancelled(ModelArtifactError):
    """Artifact verification was cancelled; no approval is returned."""


class ModelArtifactExpired(ModelArtifactError):
    """The explicit local verification deadline elapsed."""


def _require(condition, message):
    if not condition:
        raise ModelArtifactError(message)


def _integer(value, maximum=2**53 - 1):
    return type(value) is int and 0 < value <= maximum


def _json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate artifact JSON key")
            result[key] = value
        return result
    def reject(_value):
        raise ModelArtifactError("nonfinite artifact JSON")
    def finite(raw):
        value = float(raw)
        _require(math.isfinite(value), "nonfinite artifact JSON")
        return value
    try:
        result = json.loads(data.decode("utf-8"), object_pairs_hook=unique,
                            parse_constant=reject, parse_float=finite)
        _require(isinstance(result, dict), "artifact JSON must be an object")
        json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return result
    except (ValueError, UnicodeError, RecursionError):
        raise ModelArtifactError("invalid artifact JSON") from None


def validate_manifest(encoded, expected_sha256):
    """Validate the exact reviewed manifest bytes against the existing model catalog.

    No real manifest or per-file weight digest is selected in this source change.
    Tests use tiny synthetic files with synthetic reviewed digests.
    """
    _require(isinstance(encoded, bytes) and 0 < len(encoded) <= MAX_MANIFEST_BYTES,
             "invalid artifact manifest size")
    _require(isinstance(expected_sha256, str) and SHA256.fullmatch(expected_sha256),
             "trusted artifact manifest digest required")
    _require(hashlib.sha256(encoded).hexdigest() == expected_sha256,
             "artifact manifest digest mismatch")
    manifest = _json(encoded)
    _require(set(manifest) == {"schema_version", "candidate_id", "model", "revision", "adapter_sha256", "files"}
             and type(manifest["schema_version"]) is int and manifest["schema_version"] == 1,
             "unsupported artifact manifest")
    from core.current_candidates import CORE_SERVING
    catalog = CORE_SERVING
    matches = [c for c in catalog["candidates"] if c["id"] == manifest["candidate_id"]]
    _require(len(matches) == 1, "artifact candidate is not in the server allowlist")
    candidate = matches[0]
    _require(manifest["model"] == candidate["model"] and manifest["revision"] == candidate["revision"],
             "artifact identity differs from the pinned candidate")
    adapter = candidate["adapter_sha256"]
    _require(isinstance(adapter, str) and SHA256.fullmatch(adapter),
             "trained adapter digest is not pinned")
    _require(manifest["adapter_sha256"] == adapter,
             "artifact trained adapter differs from the pinned candidate")
    files = manifest["files"]
    _require(isinstance(files, list) and 1 <= len(files) <= MAX_FILES, "invalid artifact file list")
    names = set()
    total = 0
    for entry in files:
        _require(isinstance(entry, dict) and set(entry) == {"path", "bytes", "sha256"},
                 "invalid artifact file record")
        name = entry["path"]
        _require(isinstance(name, str) and NAME.fullmatch(name)
                 and name not in names and ".." not in name, "unsafe or duplicate artifact path")
        _require(name in REQUIRED_FILES | OPTIONAL_FILES | {"model.safetensors.index.json"} or name.endswith(".safetensors"),
                 "unapproved artifact file type")
        _require(_integer(entry["bytes"]), "invalid artifact byte count")
        _require(isinstance(entry["sha256"], str) and SHA256.fullmatch(entry["sha256"]),
                 "invalid artifact file digest")
        if name in PARSED_FILES:
            _require(entry["bytes"] <= MAX_METADATA_BYTES, "artifact metadata too large")
        total += entry["bytes"]
        _require(total <= 2**53 - 1, "artifact total is too large")
        names.add(name)
    _require(REQUIRED_FILES <= names and any(name.endswith(".safetensors") for name in names),
             "artifact lacks required weights or tokenizer files")
    if candidate["id"] == "kova-cosmo":
        _require("model.safetensors" in names and "model.safetensors.index.json" not in names,
                 "Cosmo requires approved monolithic base weights")
    else:
        _require("model.safetensors.index.json" in names and "model.safetensors" not in names,
                 "Orion and Nova require approved sharded base weights")
    adapter_files = [entry for entry in files if entry["path"] == "adapter_model.safetensors"]
    _require(len(adapter_files) == 1 and adapter_files[0]["sha256"] == adapter,
             "artifact trained adapter bytes differ from pinned digest")
    return manifest


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            info.st_mode, info.st_nlink)


def _regular(info, size):
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
             "artifact entries must be regular files without links")
    _require(info.st_size == size, "artifact file size mismatch")
    _require(not info.st_mode & 0o022, "artifact file is writable by another account")


def _metadata_contract(metadata, names):
    for filename in ("config.json", "tokenizer_config.json"):
        # Remote Python classes are not part of the approved packaging contract.
        _require(metadata[filename].get("auto_map") in (None, {}),
                 "artifact requires unapproved remote model code")
    adapter = metadata["adapter_config.json"]
    _require(adapter.get("peft_type") == "LORA" and adapter.get("task_type") == "CAUSAL_LM"
             and type(adapter.get("r")) is int
             and adapter["r"] > 0
             and isinstance(adapter.get("lora_alpha"), (int, float))
             and not isinstance(adapter["lora_alpha"], bool) and adapter["lora_alpha"] > 0
             and adapter.get("auto_mapping") in (None, {}), "invalid PEFT adapter configuration")
    if "model.safetensors.index.json" not in names:
        _require({name for name in names if name.endswith(".safetensors")}
                 == {"model.safetensors", "adapter_model.safetensors"}, "unexpected base weights")
        return
    index = metadata["model.safetensors.index.json"]
    _require(set(index) <= {"metadata", "weight_map"} and isinstance(index.get("weight_map"), dict)
             and index["weight_map"], "invalid artifact weight index")
    weight_map = index["weight_map"]
    _require(all(isinstance(key, str) and key and isinstance(value, str)
                 for key, value in weight_map.items()), "invalid artifact tensor reference")
    referenced = set(weight_map.values())
    shards = {name for name in names if name.endswith(".safetensors") and name != "adapter_model.safetensors"}
    _require(referenced == shards, "artifact weight index and packaged shards disagree")
    # Tensor contents, dimensions, dtype and engine compatibility require the
    # trusted safetensors loader/runtime rehearsal. A byte hash is not that test.


def verify_model_artifact(root, encoded_manifest, *, expected_manifest_sha256,
                          maximum_total_bytes, timeout_seconds, cancelled=lambda: False,
                          clock=monotonic):
    """Hash approved local files without following links or making network calls.

    This Linux-targeted implementation uses descriptor-relative reads and detects
    mutation while checking. A protected read-only mount and trusted loader are
    still required after verification; it does not prevent later host/root writes
    or interrupt an arbitrary blocking filesystem read. The deadline is cooperative
    around local reads, not a hard I/O deadline or a Kova reasoning-time budget.
    """
    manifest = validate_manifest(encoded_manifest, expected_manifest_sha256)
    _require(_integer(maximum_total_bytes), "explicit artifact byte budget required")
    total = sum(entry["bytes"] for entry in manifest["files"])
    _require(total <= maximum_total_bytes, "artifact exceeds the byte budget")
    _require(type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds)
             and 0 < timeout_seconds <= 3600, "explicit artifact verification deadline required")
    _require(callable(cancelled) and callable(clock), "artifact verification controls required")
    started = clock()
    _require(type(started) in (int, float) and math.isfinite(started), "invalid artifact clock")
    deadline = started + timeout_seconds
    _require(math.isfinite(deadline), "invalid artifact deadline")
    def check():
        try:
            stopped = cancelled()
            now = clock()
        except Exception:
            raise ModelArtifactError("artifact verification control failed") from None
        _require(type(stopped) is bool, "invalid artifact cancellation state")
        if stopped:
            raise ModelArtifactCancelled("artifact verification cancelled")
        _require(type(now) in (int, float) and math.isfinite(now) and now >= started,
                 "invalid artifact clock")
        if now >= deadline:
            raise ModelArtifactExpired("artifact verification deadline elapsed")
    check()
    _require(isinstance(root, (str, Path)), "trusted local artifact directory required")
    root = str(root)
    _require(os.path.isabs(root) and os.path.normpath(root) == root,
             "canonical absolute artifact directory required")
    _require(hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"),
             "artifact verification requires descriptor-safe filesystem support")
    descriptor = None
    try:
        # The parent hierarchy is trusted server configuration, never a user path.
        _require(str(Path(root).resolve(strict=True)) == root, "artifact root cannot contain links")
        root_info = os.lstat(root)
        _require(stat.S_ISDIR(root_info.st_mode) and not root_info.st_mode & 0o022,
                 "artifact root must be a protected directory")
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        _require(_identity(os.fstat(descriptor)) == _identity(root_info), "artifact root changed")
        names = {entry["path"] for entry in manifest["files"]}
        _require(set(os.listdir(descriptor)) == names, "artifact directory differs from manifest")
        checked = {}
        metadata = {}
        for entry in manifest["files"]:
            check()
            name = entry["path"]
            before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            _regular(before, entry["bytes"])
            file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
            try:
                _require(_identity(os.fstat(file_fd)) == _identity(before), "artifact file changed")
                sha = hashlib.sha256()
                consumed = 0
                chunks = []
                while True:
                    check()
                    chunk = os.read(file_fd, min(READ_BYTES, entry["bytes"] - consumed + 1))
                    check()
                    if not chunk:
                        break
                    consumed += len(chunk)
                    _require(consumed <= entry["bytes"], "artifact file grew during verification")
                    sha.update(chunk)
                    if name in PARSED_FILES:
                        chunks.append(chunk)
                _require(consumed == entry["bytes"] and sha.hexdigest() == entry["sha256"],
                         "artifact content does not match approved digest")
                _require(_identity(os.fstat(file_fd)) == _identity(before), "artifact changed while hashing")
                checked[name] = _identity(before)
                if name in PARSED_FILES:
                    metadata[name] = _json(b"".join(chunks))
            finally:
                os.close(file_fd)
        check()
        _metadata_contract(metadata, names)
        _require(set(os.listdir(descriptor)) == names, "artifact directory changed")
        for name, identity in checked.items():
            _require(_identity(os.stat(name, dir_fd=descriptor, follow_symlinks=False)) == identity,
                     "artifact changed after hashing")
        _require(_identity(os.lstat(root)) == _identity(root_info), "artifact root changed")
        check()
        return {
            "status": "local_artifact_bytes_verified",
            "candidate_id": manifest["candidate_id"], "model": manifest["model"],
            "revision": manifest["revision"], "manifest_sha256": expected_manifest_sha256,
            "adapter_sha256": manifest["adapter_sha256"],
            "file_count": len(names), "total_bytes": total,
            "vendor_provenance_authenticated": False,
            "model_loaded": False, "serving_compatibility_verified": False,
            "gpu_execution_authorized": False, "production_routing_authorized": False,
        }
    except OSError:
        raise ModelArtifactError("artifact filesystem verification failed") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
