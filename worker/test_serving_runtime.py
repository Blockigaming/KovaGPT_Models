"""Actual artifact verification plus synthetic native-engine lifecycle tests.

No vLLM distribution, torch, model weights, network, CUDA or GPU is installed or
executed. One test substitutes native API modules to verify exact loader arguments.
"""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from worker import serving_runtime as runtime
from worker.model_startup import SAFETY_FIELDS
from core.current_candidates import CORE_SERVING
from release.model_revisions import MODEL_SOURCE_REFERENCES


class FakeBackend:
    def __init__(self, root, artifact, policy):
        self.engine = object()
        self.closed = []
        self.health_calls = 0
        self.bad_health = False
        self.bad_close = False
        self.observed = {"model_path": root, "tokenizer_path": root,
            "model_revision": artifact["revision"], "tokenizer_revision": artifact["revision"],
            "adapter_sha256": artifact["adapter_sha256"],
            "adapter_bundle_sha256": artifact["manifest_sha256"],
            "served_model_names": [artifact["model"]], "context_tokens": policy.context_tokens,
            "trust_remote_code": False}

    def observed_configuration(self):
        return dict(self.observed)

    async def health(self):
        self.health_calls += 1
        if self.bad_health:
            raise RuntimeError("PRIVATE ENGINE ERROR")

    def close(self):
        self.closed.append(True)
        if self.bad_close:
            raise RuntimeError("PRIVATE SHUTDOWN ERROR")


class ServingRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="kova-loader-fixture-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "model"
        self.model.mkdir(mode=0o700)
        self.candidate = CORE_SERVING["candidates"][0]
        previous_pin = self.candidate["adapter_sha256"]
        previous_bundle = self.candidate["adapter_bundle_sha256"]
        adapter_bytes = b"synthetic serving adapter"
        self.candidate["adapter_sha256"] = hashlib.sha256(adapter_bytes).hexdigest()
        self.addCleanup(self.candidate.update, adapter_sha256=previous_pin)
        self.addCleanup(self.candidate.update, adapter_bundle_sha256=previous_bundle)
        files = {"config.json": b'{"model_type":"synthetic"}', "tokenizer.json": b'{}',
            "tokenizer_config.json": b'{}',
            "adapter_config.json": json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM",
                "r": 8, "lora_alpha": 16,
                "base_model_name_or_path": MODEL_SOURCE_REFERENCES[self.candidate["id"]].model}).encode(),
            "model.safetensors": b'NOT REAL MODEL WEIGHTS',
            "adapter_model.safetensors": adapter_bytes,
        }
        for name, value in files.items():
            (self.model / name).write_bytes(value)
        manifest = {"schema_version":1, "candidate_id":self.candidate["id"], "model":self.candidate["model"],
            "revision":self.candidate["revision"], "adapter_sha256":self.candidate["adapter_sha256"],
            "files":[{"path":name, "bytes":len(value),
                "sha256":hashlib.sha256(value).hexdigest()} for name, value in sorted(files.items())]}
        manifest_path = self.root / "manifest.json"
        manifest_bytes = json.dumps(manifest).encode()
        self.candidate["adapter_bundle_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_path.write_bytes(manifest_bytes)
        manifest_path.chmod(0o600)
        startup = {"schema_version":1, "status":"source_only_serving_blocked", "verification_enabled":True,
            "artifact":{"root":str(self.model), "manifest_path":str(manifest_path),
                "expected_manifest_sha256":hashlib.sha256(manifest_bytes).hexdigest(),
                "maximum_total_bytes":sum(map(len,files.values())), "timeout_seconds":10},
            "safety":{field:False for field in SAFETY_FIELDS}}
        self.policy_path = self.root / "startup.json"
        policy_bytes = json.dumps(startup).encode()
        self.policy_path.write_bytes(policy_bytes)
        self.policy_path.chmod(0o600)
        self.policy = runtime.LoaderPolicy(str(self.policy_path), hashlib.sha256(policy_bytes).hexdigest(),
            self.candidate["id"], "sha256:" + "a"*64, 8192, 0.8, 2, 60, 2, 2, True, True, True)
        self.allowed = True
        self.backend = None
        def construct(root, artifact, policy):
            self.backend = FakeBackend(root, artifact, policy)
            return self.backend
        self.factory = patch.object(runtime, "NativeVllm", side_effect=construct).start()
        self.addCleanup(patch.stopall)
        # The temporary fixture directory is not a production read-only mount.
        self.environment = patch.object(runtime, "_environment_guard", return_value=None).start()
        self.controller = runtime.ServingRuntime(self.policy, lambda _:self.allowed,
            lambda:self.policy.container_image_digest)

    async def test_verified_artifact_loads_then_checks_identity_and_health_before_handoff(self):
        state = await self.controller.start()
        self.assertTrue(state["ready"])
        self.assertFalse(state["phase_b_ready"])
        identity = await self.controller.identity()
        self.assertEqual(identity["model"], self.candidate["model"])
        self.assertEqual(identity["model_revision"], self.candidate["revision"])
        self.assertEqual(identity["adapter_bundle_sha256"], self.candidate["adapter_bundle_sha256"])
        self.assertEqual(identity["context_tokens"], 8192)
        self.assertEqual(identity["container_image_digest"], self.policy.container_image_digest)
        self.assertNotIn("artifact_root", identity)
        engine = await self.controller.engine_for_server(identity["worker_lifecycle_id"])
        self.assertIs(engine, self.backend.engine)
        self.assertGreaterEqual(self.backend.health_calls, 3)
        self.assertFalse(self.controller.stop()["ready"])
        self.assertEqual(self.backend.closed, [True])

    async def test_loaded_base_without_observed_trained_adapter_cannot_be_ready(self):
        def base_only(root, artifact, policy):
            backend = FakeBackend(root, artifact, policy)
            backend.observed.pop("adapter_sha256")
            return backend
        self.factory.side_effect = base_only
        with self.assertRaises(runtime.ServingRuntimeError):
            await self.controller.start()
        self.assertFalse(self.controller.status()["ready"])

    async def test_each_disabled_loading_gate_blocks_before_artifacts_callbacks_or_engine(self):
        for field in ("enabled", "model_loading_authorized", "gpu_execution_authorized"):
            forbidden = Mock(side_effect=AssertionError("disabled callback called"))
            controller = runtime.ServingRuntime(replace(self.policy, **{field:False}), forbidden, forbidden)
            with patch.object(runtime.model_startup, "_control_bytes", side_effect=AssertionError("file read")):
                with self.assertRaises(runtime.ServingRuntimeError):
                    await controller.start()
            forbidden.assert_not_called()
        self.factory.assert_not_called()

    async def test_current_approval_and_observed_image_must_match(self):
        for allowed, digest in ((False,self.policy.container_image_digest), ("true",self.policy.container_image_digest),
                                (True,"sha256:"+"b"*64)):
            controller = runtime.ServingRuntime(self.policy, lambda _:allowed, lambda:digest)
            with self.assertRaises(runtime.ServingRuntimeError):
                await controller.start()
        self.factory.assert_not_called()

    async def test_policy_digest_tampering_rejects_before_loader_construction(self):
        self.policy_path.write_bytes(self.policy_path.read_bytes()+b" ")
        with self.assertRaises(runtime.ServingRuntimeError):
            await self.controller.start()
        self.factory.assert_not_called()
        self.assertEqual(self.controller.status()["state"], "failed")

    async def test_changed_weight_bytes_are_rejected_by_real_startup_verifier(self):
        path = self.model / "model.safetensors"
        path.write_bytes(b"x" * len(path.read_bytes()))
        with self.assertRaises(runtime.ServingRuntimeError):
            await self.controller.start()
        self.factory.assert_not_called()

    async def test_valid_changed_adapter_config_manifest_with_same_weights_rejects_unpinned_bundle(self):
        config = self.model / "adapter_config.json"
        changed = json.loads(config.read_bytes())
        changed["lora_alpha"] = 32
        config.write_bytes(json.dumps(changed).encode())
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        entry = next(entry for entry in manifest["files"] if entry["path"] == config.name)
        entry["bytes"] = len(config.read_bytes())
        entry["sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        manifest_path.write_bytes(json.dumps(manifest).encode())
        startup = json.loads(self.policy_path.read_bytes())
        startup["artifact"]["expected_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.policy_path.write_bytes(json.dumps(startup).encode())
        self.policy = replace(self.policy, startup_policy_sha256=hashlib.sha256(self.policy_path.read_bytes()).hexdigest())
        self.controller = runtime.ServingRuntime(self.policy, lambda _: True,
            lambda: self.policy.container_image_digest)
        self.assertNotEqual(startup["artifact"]["expected_manifest_sha256"],
                            self.candidate["adapter_bundle_sha256"])
        with self.assertRaises(runtime.ServingRuntimeError):
            await self.controller.start()
        self.factory.assert_not_called()

    async def test_candidate_mismatch_does_not_load_another_model(self):
        controller = runtime.ServingRuntime(replace(self.policy, candidate_id="unapproved"), lambda _:True,
                                            lambda:self.policy.container_image_digest)
        with self.assertRaises(runtime.ServingRuntimeError):
            await controller.start()
        self.factory.assert_not_called()

    async def test_current_candidate_context_limit_blocks_oversized_loader_before_native_allocation(self):
        for value in (self.candidate["context_tokens"] + 1, 262144):
            policy = replace(self.policy, context_tokens=value)
            controller = runtime.ServingRuntime(policy, lambda _: True,
                                                lambda: policy.container_image_digest)
            with self.subTest(context_tokens=value), self.assertRaises(runtime.ServingRuntimeError):
                await controller.start()
            self.assertEqual(controller.status()["state"], "failed")
            self.factory.assert_not_called()
        policy = replace(self.policy, context_tokens=self.candidate["context_tokens"])
        controller = runtime.ServingRuntime(policy, lambda _: True,
                                            lambda: policy.container_image_digest)
        await controller.start()
        self.assertEqual((await controller.identity())["context_tokens"], self.candidate["context_tokens"])
        controller.stop()

    async def test_observed_native_configuration_mismatch_closes_backend_and_refuses_readiness(self):
        original = self.factory.side_effect
        def altered(*args):
            backend = original(*args)
            backend.observed["context_tokens"] = 999
            return backend
        self.factory.side_effect = altered
        with self.assertRaises(runtime.ServingRuntimeError):
            await self.controller.start()
        self.assertEqual(self.backend.closed, [True])
        self.assertEqual(self.controller.status()["state"], "failed")

    async def test_health_failure_does_not_become_success_and_errors_are_sanitized(self):
        original = self.factory.side_effect
        def unhealthy(*args):
            backend = original(*args)
            backend.bad_health = True
            return backend
        self.factory.side_effect = unhealthy
        with self.assertRaises(runtime.ServingRuntimeError) as caught:
            await self.controller.start()
        self.assertNotIn("PRIVATE", str(caught.exception))
        self.assertEqual(self.backend.closed, [True])
        self.assertFalse(self.controller.status()["ready"])

    async def test_revoked_loading_authority_removes_ready_handoff(self):
        await self.controller.start()
        self.allowed = False
        with self.assertRaises(runtime.ServingRuntimeError):
            await self.controller.identity()
        self.assertFalse(self.controller.status()["ready"])
        self.assertEqual(self.backend.closed, [True])

    async def test_runtime_revision_drift_is_rejected_on_every_identity_probe(self):
        await self.controller.start()
        self.backend.observed["model_revision"] = "0" * 40
        with self.assertRaises(runtime.ServingRuntimeError):
            await self.controller.identity()
        self.assertEqual(self.backend.closed, [True])

    async def test_stale_lifecycle_id_cannot_obtain_engine_handle(self):
        await self.controller.start()
        with self.assertRaises(runtime.ServingRuntimeError):
            await self.controller.engine_for_server("stale-lifecycle")
        self.controller.stop()

    async def test_failed_shutdown_remains_failed_not_quiescent(self):
        await self.controller.start()
        self.backend.bad_close = True
        with self.assertRaises(runtime.ServingRuntimeError):
            self.controller.stop()
        self.assertEqual(self.controller.status()["state"], "failed")
        self.assertIsNotNone(self.controller._backend)
        self.backend.bad_close = False
        self.assertEqual(self.controller.stop()["state"], "stopped")

    async def test_restart_rehashes_artifacts_and_does_not_reuse_old_loader_success(self):
        await self.controller.start()
        self.controller.stop()
        path = self.model / "model.safetensors"
        path.write_bytes(b"changed")
        with self.assertRaises(runtime.ServingRuntimeError):
            await self.controller.start()
        self.assertEqual(self.factory.call_count, 1)

    async def test_repeated_start_cannot_allocate_two_engines(self):
        await self.controller.start()
        with self.assertRaises(runtime.ServingRuntimeError):
            await self.controller.start()
        self.assertEqual(self.factory.call_count, 1)
        self.controller.stop()

    async def test_explicit_elapsed_load_budget_rejects_late_initialization_and_closes(self):
        ticks = [0]
        original = self.factory.side_effect
        def slow(*args):
            backend = original(*args)
            ticks[0] = 61
            return backend
        self.factory.side_effect = slow
        controller = runtime.ServingRuntime(self.policy, lambda _:True, lambda:self.policy.container_image_digest,
                                            clock=lambda:ticks[0])
        with self.assertRaises(runtime.ServingRuntimeError):
            await controller.start()
        self.assertEqual(self.backend.closed, [True])

    async def test_cancelled_start_never_touches_model_files(self):
        controller = runtime.ServingRuntime(self.policy, lambda _:True, lambda:self.policy.container_image_digest,
                                            cancelled=lambda:True)
        with self.assertRaises(runtime.ServingRuntimeError):
            await controller.start()
        self.factory.assert_not_called()

    async def test_invalid_loader_limits_and_coerced_flags_fail_before_engine_import(self):
        for changes in ({"context_tokens":True}, {"context_tokens":0}, {"maximum_sequences":100},
                        {"gpu_memory_utilization":1}, {"gpu_memory_utilization":float("nan")},
                        {"load_timeout_seconds":0}, {"enabled":"true"}, {"container_image_digest":"latest"}):
            with self.assertRaises(runtime.ServingRuntimeError):
                replace(self.policy, **changes)
        self.factory.assert_not_called()


class NativeBindingTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_base_only_loader_rejects_before_gpu_allocation(self):
        policy = runtime.LoaderPolicy("/trusted/policy.json", "a"*64, "fixture", "sha256:"+"b"*64,
                                       8192, 0.8, 2, 60, 2, 2, True, True, True)
        with patch.dict("sys.modules", {"vllm": None}), self.assertRaisesRegex(
                runtime.ServingRuntimeError, "trained adapter loading is unavailable"):
            runtime.NativeVllm("/readonly/model", {"model":"fixture-model", "revision":"a"*40}, policy)

    async def test_disabled_native_policy_fails_before_loader(self):
        policy = runtime.LoaderPolicy("/trusted/policy.json", "a"*64, "fixture", "sha256:"+"b"*64,
                                       8192, 0.8, 2, 60, 2, 2, True, True, True)
        with self.assertRaises(runtime.ServingRuntimeError):
            runtime.NativeVllm("/readonly/model", {"model":"fixture", "revision":"a"*40},
                               replace(policy, enabled=False))

    async def test_concrete_environment_guard_requires_nonroot_readonly_and_offline_flags(self):
        environment = {name:"1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_NO_USAGE_STATS", "DO_NOT_TRACK")}
        environment["VLLM_PLUGINS"] = ""
        with patch.dict(os.environ, environment), patch.object(os, "geteuid", return_value=1000), \
                patch.object(os, "statvfs", return_value=SimpleNamespace(f_flag=os.ST_RDONLY)):
            runtime._environment_guard("/fixture")
            with patch.object(os, "geteuid", return_value=0):
                with self.assertRaises(runtime.ServingRuntimeError):
                    runtime._environment_guard("/fixture")
            with patch.object(os, "statvfs", return_value=SimpleNamespace(f_flag=0)):
                with self.assertRaises(runtime.ServingRuntimeError):
                    runtime._environment_guard("/fixture")
            with patch.dict(os.environ, {"HF_HUB_OFFLINE":"0"}):
                with self.assertRaises(runtime.ServingRuntimeError):
                    runtime._environment_guard("/fixture")
