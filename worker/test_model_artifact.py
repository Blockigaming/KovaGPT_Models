"""Small synthetic artifact files only; no real weights, downloads or GPU loads."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from worker import model_artifact as artifact
from core.current_candidates import CORE_SERVING
from release.model_revisions import MODEL_SOURCE_REFERENCES


CATALOG = CORE_SERVING


class ModelArtifactTests(unittest.TestCase):
    def setUp(self):
        self.adapter_bytes = b"synthetic fixture adapter bytes"
        self.adapter_digest = hashlib.sha256(self.adapter_bytes).hexdigest()
        self.original_pins = [candidate["adapter_sha256"] for candidate in CATALOG["candidates"]]
        for candidate in CATALOG["candidates"]:
            candidate["adapter_sha256"] = self.adapter_digest
        self.directory = tempfile.TemporaryDirectory(prefix="kova-artifact-fixture-")
        self.root = Path(self.directory.name) / "model"
        self.root.mkdir(mode=0o700)
        self.files = {
            "config.json": b'{"model_type":"synthetic_fixture"}',
            "tokenizer.json": b'{"fixture":true}',
            "tokenizer_config.json": b'{"tokenizer_class":"SyntheticFixture"}',
            "adapter_config.json": json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM",
                "r": 8, "lora_alpha": 16,
                "base_model_name_or_path": MODEL_SOURCE_REFERENCES["kova-cosmo"].model}).encode(),
            # Deliberately not a loadable model: only the byte-integrity layer is tested.
            "model.safetensors": b"synthetic fixture weight bytes",
            "README.md": b"Synthetic model-artifact fixture; not upstream weights.\n",
            "adapter_model.safetensors": self.adapter_bytes,
        }
        for name, content in self.files.items():
            (self.root / name).write_bytes(content)
        self.manifest = self.snapshot()

    def tearDown(self):
        for candidate, previous in zip(CATALOG["candidates"], self.original_pins):
            candidate["adapter_sha256"] = previous
        self.directory.cleanup()

    def snapshot(self, candidate_index=0):
        candidate = CATALOG["candidates"][candidate_index]
        return {
            "schema_version": 1, "candidate_id": candidate["id"],
            "model": candidate["model"], "revision": candidate["revision"],
            "adapter_sha256": candidate["adapter_sha256"],
            "files": [{"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                      for name, content in sorted(self.files.items())],
        }

    def encode(self, manifest=None):
        return json.dumps(self.manifest if manifest is None else manifest, sort_keys=True).encode()

    def verify(self, manifest=None, **changes):
        encoded = self.encode(manifest)
        options = {
            "expected_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            # The fixture's trusted byte budget is independent of malformed input.
            "maximum_total_bytes": sum(map(len, self.files.values())),
            "timeout_seconds": 10, "clock": lambda: 0,
        }
        options.update(changes)
        return artifact.verify_model_artifact(self.root, encoded, **options)

    def change_metadata(self, name, content):
        self.files[name] = content
        (self.root / name).write_bytes(content)
        self.manifest = self.snapshot()

    def set_base_model(self, candidate_index):
        value = json.loads(self.files["adapter_config.json"])
        value["base_model_name_or_path"] = MODEL_SOURCE_REFERENCES[CATALOG["candidates"][candidate_index]["id"]].model
        self.change_metadata("adapter_config.json", json.dumps(value).encode())

    def use_sharded_weights(self):
        (self.root / "model.safetensors").unlink()
        del self.files["model.safetensors"]
        self.change_metadata("model.safetensors.index.json",
                             b'{"metadata":{"total_size":16},"weight_map":{"fixture.weight":"model-00001-of-00001.safetensors"}}')
        self.change_metadata("model-00001-of-00001.safetensors", b"synthetic fixture weight bytes")
        self.set_base_model(1)
        self.manifest = self.snapshot(1)

    def test_all_pinned_candidates_produce_integrity_only_evidence(self):
        for index, candidate in enumerate(CATALOG["candidates"]):
            if index == 1:
                self.use_sharded_weights()
            elif index == 2:
                self.set_base_model(index)
            with self.subTest(candidate=candidate["id"]):
                result = self.verify(self.snapshot(index))
                self.assertEqual(result["status"], "local_artifact_bytes_verified")
                self.assertEqual(result["model"], candidate["model"])
                self.assertEqual(result["revision"], candidate["revision"])
                self.assertEqual(result["file_count"], len(self.files))
                self.assertEqual(result["total_bytes"], sum(map(len, self.files.values())))
                for field in ("vendor_provenance_authenticated", "model_loaded", "serving_compatibility_verified",
                              "gpu_execution_authorized", "production_routing_authorized"):
                    self.assertIs(result[field], False)

    def test_cosmo_monolithic_base_and_adapter_config_are_verified(self):
        self.assertEqual(self.verify()["status"], "local_artifact_bytes_verified")
        changed = json.loads(self.files["adapter_config.json"])
        changed["peft_type"] = "OTHER"
        self.change_metadata("adapter_config.json", json.dumps(changed).encode())
        with self.assertRaisesRegex(artifact.ModelArtifactError, "invalid PEFT adapter configuration"):
            self.verify()

    def test_adapter_config_is_required_and_tampering_rejected(self):
        altered = deepcopy(self.manifest)
        altered["files"] = [entry for entry in altered["files"]
                            if entry["path"] != "adapter_config.json"]
        with self.assertRaises(artifact.ModelArtifactError):
            self.verify(altered)
        config = self.root / "adapter_config.json"
        config.write_bytes(b"x" * len(self.files["adapter_config.json"]))
        with self.assertRaisesRegex(artifact.ModelArtifactError, "approved digest"):
            self.verify()

    def test_reviewed_adapter_config_must_be_a_safe_causal_lora(self):
        for change in ({"peft_type": "OTHER"}, {"task_type": "SEQ_CLS"}, {"r": True},
                       {"r": 0}, {"lora_alpha": -1},
                       {"auto_mapping": {"base_model_class": "RemoteModel"}}):
            value = json.loads(self.files["adapter_config.json"])
            value.update(change)
            self.change_metadata("adapter_config.json", json.dumps(value).encode())
            with self.subTest(change=change), \
                    self.assertRaisesRegex(artifact.ModelArtifactError, "invalid PEFT adapter configuration"):
                self.verify()

    def test_family_weight_layouts_must_match_pinned_manifests(self):
        for index in (1, 2):
            with self.subTest(candidate=CATALOG["candidates"][index]["id"]), \
                    self.assertRaisesRegex(artifact.ModelArtifactError, "sharded base weights"):
                self.verify(self.snapshot(index))
        self.use_sharded_weights()
        with self.assertRaisesRegex(artifact.ModelArtifactError, "monolithic base weights"):
            self.verify(self.snapshot(0))
        for index in (1, 2):
            with self.subTest(candidate=CATALOG["candidates"][index]["id"]):
                self.set_base_model(index)
                self.assertEqual(self.verify(self.snapshot(index))["status"],
                                 "local_artifact_bytes_verified")

    def test_adapter_base_must_match_trusted_family_even_with_a_reviewed_manifest(self):
        for base in ("Other/Model", "Qwen/Qwen3-1.7B", "/external/verified-snapshot", "", None):
            value = json.loads(self.files["adapter_config.json"])
            value["base_model_name_or_path"] = base
            self.change_metadata("adapter_config.json", json.dumps(value).encode())
            with self.subTest(base=base), self.assertRaisesRegex(
                artifact.ModelArtifactError, "adapter base model differs from pinned family source"
            ):
                self.verify()
        value.pop("base_model_name_or_path")
        self.change_metadata("adapter_config.json", json.dumps(value).encode())
        with self.assertRaisesRegex(artifact.ModelArtifactError, "adapter base model differs"):
            self.verify()

    def test_missing_or_mismatched_reviewed_manifest_pin_is_rejected(self):
        for digest in (None, "", "0" * 64, "A" * 64, "sha256:" + "a" * 64):
            with self.subTest(digest=digest), self.assertRaises(artifact.ModelArtifactError):
                self.verify(expected_manifest_sha256=digest)

    def test_adapter_file_must_match_separate_candidate_pin(self):
        changed = deepcopy(self.manifest)
        changed["adapter_sha256"] = "e" * 64
        with self.assertRaises(artifact.ModelArtifactError):
            self.verify(changed)
        changed = deepcopy(self.manifest)
        next(entry for entry in changed["files"] if entry["path"] == "adapter_model.safetensors")["sha256"] = "e" * 64
        with self.assertRaises(artifact.ModelArtifactError):
            self.verify(changed)
        CATALOG["candidates"][0]["adapter_sha256"] = None
        with self.assertRaisesRegex(artifact.ModelArtifactError, "trained adapter digest is not pinned"):
            self.verify()

    def test_unknown_candidate_and_changed_identity_do_not_pass_with_a_new_checksum(self):
        for field, value in (("candidate_id", "unknown"), ("model", "Unapproved/Model"),
                             ("revision", "0" * 40), ("revision", "main")):
            manifest = deepcopy(self.manifest)
            manifest[field] = value
            with self.subTest(field=field), self.assertRaises(artifact.ModelArtifactError):
                self.verify(manifest)

    def test_manifest_paths_reject_traversal_absolute_hidden_and_duplicate_names(self):
        for name in ("../config.json", "/config.json", "dir/config.json", ".env", "a\\b.json",
                     "a..b.json", "config.json\n", "config.json"):
            value = deepcopy(self.manifest)
            value["files"][-1]["path"] = name
            with self.subTest(name=name), self.assertRaises(artifact.ModelArtifactError):
                self.verify(value)

    def test_python_pickle_binaries_and_unapproved_extensions_are_rejected(self):
        for name in ("model.py", "model.bin", "model.pt", "model.pth", "module.so", "run.sh", "model.gguf"):
            value = deepcopy(self.manifest)
            value["files"][-1]["path"] = name
            with self.subTest(name=name), self.assertRaises(artifact.ModelArtifactError):
                self.verify(value)

    def test_invalid_manifest_types_sizes_and_extra_fields_fail_closed(self):
        for amount in (True, False, 0, -1, "10", 1.5, 2**53):
            value = deepcopy(self.manifest)
            value["files"][0]["bytes"] = amount
            with self.subTest(amount=amount), self.assertRaises(artifact.ModelArtifactError):
                self.verify(value, maximum_total_bytes=10000)
        for version in (True, 1.0, "1", 2):
            value = {**self.manifest, "schema_version": version}
            with self.assertRaises(artifact.ModelArtifactError):
                self.verify(value)
        with self.assertRaises(artifact.ModelArtifactError):
            self.verify({**self.manifest, "gpu_execution_authorized": True})

    def test_all_required_metadata_and_weights_must_be_present(self):
        for name in artifact.REQUIRED_FILES | {"model.safetensors"}:
            value = deepcopy(self.manifest)
            value["files"] = [item for item in value["files"] if item["path"] != name]
            with self.subTest(name=name), self.assertRaises(artifact.ModelArtifactError):
                self.verify(value)

    def test_missing_and_unlisted_files_are_rejected(self):
        path = self.root / "README.md"
        path.unlink()
        with self.assertRaises(artifact.ModelArtifactError):
            self.verify()
        path.write_bytes(self.files["README.md"])
        (self.root / "unexpected.json").write_text("{}")
        with self.assertRaises(artifact.ModelArtifactError):
            self.verify()

    def test_size_and_same_size_content_changes_are_rejected(self):
        path = self.root / "model.safetensors"
        original = path.read_bytes()
        for altered in (original + b"extra", b"x" * len(original)):
            path.write_bytes(altered)
            with self.assertRaises(artifact.ModelArtifactError):
                self.verify()

    def test_symlink_and_hardlink_entries_are_rejected(self):
        path = self.root / "README.md"
        external = Path(self.directory.name) / "external.txt"
        external.write_bytes(self.files["README.md"])
        path.unlink()
        path.symlink_to(external)
        with self.assertRaises(artifact.ModelArtifactError):
            self.verify()
        path.unlink()
        os.link(external, path)
        with self.assertRaises(artifact.ModelArtifactError):
            self.verify()

    def test_directories_and_fifos_fail_before_file_read(self):
        path = self.root / "README.md"
        path.unlink()
        path.mkdir()
        with self.assertRaises(artifact.ModelArtifactError):
            self.verify()
        path.rmdir()
        os.mkfifo(path)
        with self.assertRaises(artifact.ModelArtifactError):
            self.verify()

    def test_group_or_world_writable_packages_are_rejected(self):
        path = self.root / "config.json"
        path.chmod(0o666)
        with self.assertRaises(artifact.ModelArtifactError):
            self.verify()
        path.chmod(0o644)
        self.root.chmod(0o777)
        try:
            with self.assertRaises(artifact.ModelArtifactError):
                self.verify()
        finally:
            self.root.chmod(0o700)

    def test_remote_class_mapping_is_not_accepted_as_reviewed_model_code(self):
        for name in ("config.json", "tokenizer_config.json"):
            original = self.files[name]
            self.change_metadata(name, b'{"auto_map":{"AutoModel":"untrusted.Model"}}')
            with self.subTest(name=name), self.assertRaises(artifact.ModelArtifactError):
                self.verify()
            self.change_metadata(name, original)

    def test_weight_index_requires_exactly_the_packaged_shard_set(self):
        self.use_sharded_weights()
        name = "model.safetensors.index.json"
        for value in ({"weight_map": {}}, {"weight_map": {"x": "missing.safetensors"}},
                      {"weight_map": {"x": "../model.safetensors"}}, {"weight_map": {"x": []}},
                      {"weight_map": {"x": "tokenizer.json"}}, {"bad": True}):
            self.change_metadata(name, json.dumps(value).encode())
            with self.subTest(value=value), self.assertRaises(artifact.ModelArtifactError):
                self.verify(self.snapshot(1))

    def test_metadata_duplicate_keys_nonfinite_and_invalid_utf8_are_rejected(self):
        for content in (b'{"auto_map":{},"auto_map":{}}', b'{"x":NaN}', b'{"x":1e400}',
                        b"[]", b"\xff", b'{"x":"\\ud800"}'):
            self.change_metadata("config.json", content)
            with self.subTest(content=content), self.assertRaises(artifact.ModelArtifactError):
                self.verify()

    def test_cancellation_and_deadline_return_no_verification(self):
        with self.assertRaises(artifact.ModelArtifactCancelled):
            self.verify(cancelled=lambda: True)
        times = iter((0, 10))
        with self.assertRaises(artifact.ModelArtifactExpired):
            self.verify(clock=lambda: next(times))

    def test_invalid_limits_and_control_values_fail_before_opening_package(self):
        with patch.object(artifact.os, "open", side_effect=AssertionError("package opened")):
            for timeout in (None, True, 0, -1, math.inf, math.nan, "10", 3601):
                with self.subTest(timeout=timeout), self.assertRaises(artifact.ModelArtifactError):
                    self.verify(timeout_seconds=timeout)
            for maximum in (None, True, 0, -1, 1, "1000"):
                with self.subTest(maximum=maximum), self.assertRaises(artifact.ModelArtifactError):
                    self.verify(maximum_total_bytes=maximum)
            for state in (None, 0, "false"):
                with self.assertRaises(artifact.ModelArtifactError):
                    self.verify(cancelled=lambda: state)

    def test_concurrent_file_mutation_during_hashing_is_detected(self):
        original_read = os.read
        changed = []
        def mutating_read(fd, count):
            chunk = original_read(fd, count)
            if chunk and not changed:
                changed.append(True)
                (self.root / "README.md").write_bytes(b"x" * len(self.files["README.md"]))
            return chunk
        with patch.object(artifact.os, "read", side_effect=mutating_read):
            with self.assertRaises(artifact.ModelArtifactError):
                self.verify()

    def test_mutation_after_hashing_is_detected_before_success(self):
        original = artifact._metadata_contract
        def changed(metadata, names, candidate_id):
            original(metadata, names, candidate_id)
            (self.root / "config.json").write_bytes(b"x" * len(self.files["config.json"]))
        with patch.object(artifact, "_metadata_contract", side_effect=changed):
            with self.assertRaises(artifact.ModelArtifactError):
                self.verify()

    def test_manifest_parser_returns_an_independent_snapshot(self):
        encoded = self.encode()
        first = artifact.validate_manifest(encoded, hashlib.sha256(encoded).hexdigest())
        first["files"].clear()
        second = artifact.validate_manifest(encoded, hashlib.sha256(encoded).hexdigest())
        self.assertEqual(second, self.manifest)

    def test_hash_reads_are_bounded_and_no_network_or_model_loader_is_used(self):
        original_read = os.read
        reads = []
        def tracked(fd, amount):
            reads.append(amount)
            return original_read(fd, amount)
        with patch.object(artifact.os, "read", side_effect=tracked), \
                patch.object(socket, "socket", side_effect=AssertionError("network called")):
            result = self.verify()
        self.assertTrue(reads)
        self.assertTrue(all(0 < value <= artifact.READ_BYTES for value in reads))
        self.assertFalse(result["model_loaded"])

    def test_open_descriptors_are_closed_on_verification_failure(self):
        (self.root / "README.md").write_bytes(b"x" * len(self.files["README.md"]))
        original_open, original_close = os.open, os.close
        opened, closed = [], []
        def tracked_open(*args, **kwargs):
            fd = original_open(*args, **kwargs)
            opened.append(fd)
            return fd
        def tracked_close(fd):
            closed.append(fd)
            return original_close(fd)
        with patch.object(artifact.os, "open", side_effect=tracked_open), \
                patch.object(artifact.os, "close", side_effect=tracked_close):
            with self.assertRaises(artifact.ModelArtifactError):
                self.verify()
        self.assertTrue(opened)
        self.assertEqual(sorted(opened), sorted(closed))

    def test_root_symlinks_and_relative_paths_are_rejected(self):
        alias = Path(self.directory.name) / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        encoded = self.encode()
        for path in (str(alias), "relative/model", str(self.root) + "/../model"):
            with self.subTest(path=path), self.assertRaises(artifact.ModelArtifactError):
                artifact.verify_model_artifact(path, encoded,
                    expected_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
                    maximum_total_bytes=10000, timeout_seconds=10, clock=lambda: 0)

    def test_oversized_manifests_and_metadata_fail_without_reading_payloads(self):
        value = deepcopy(self.manifest)
        next(item for item in value["files"] if item["path"] == "config.json")["bytes"] = artifact.MAX_METADATA_BYTES + 1
        with self.assertRaises(artifact.ModelArtifactError):
            self.verify(value)
        encoded = b"x" * (artifact.MAX_MANIFEST_BYTES + 1)
        with self.assertRaises(artifact.ModelArtifactError):
            artifact.validate_manifest(encoded, hashlib.sha256(encoded).hexdigest())

    def test_verification_does_not_rewrite_artifact_bytes_or_permissions(self):
        before = {p.name: (p.read_bytes(), p.stat().st_mode) for p in self.root.iterdir()}
        self.verify()
        after = {p.name: (p.read_bytes(), p.stat().st_mode) for p in self.root.iterdir()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
