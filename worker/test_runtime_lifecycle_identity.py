"""Additional lifecycle-fence controls; no real loader or model is imported."""

from dataclasses import replace
import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from worker import serving_runtime as runtime
from worker import test_serving_runtime as fixtures


class RuntimeFenceTests(unittest.IsolatedAsyncioTestCase):
    def policy(self):
        return runtime.LoaderPolicy("/fixture/startup.json", hashlib.sha256(b"fixture-policy").hexdigest(),
            "fixture-candidate", "sha256:"+"a"*64, 8192, 0.8, 1, 60, 2, 2, True, True, True)

    async def test_invalid_clock_does_not_strand_loading_state(self):
        policy = self.policy()
        controller = runtime.ServingRuntime(policy, lambda _:True, lambda:policy.container_image_digest,
                                            clock=lambda:float("nan"))
        with patch.object(runtime.model_startup, "_control_bytes", side_effect=AssertionError("file touched")):
            with self.assertRaises(runtime.ServingRuntimeError):
                await controller.start()
        self.assertEqual(controller.status()["state"], "failed")
        self.assertFalse(controller.status()["ready"])

    async def test_direct_native_constructor_refuses_default_disabled_policy(self):
        policy = replace(self.policy(), enabled=False)
        with self.assertRaises(runtime.ServingRuntimeError):
            runtime.NativeVllm("/fixture/model", {}, policy)

    async def test_reloading_the_same_artifact_invalidates_the_old_lifecycle_handle(self):
        policy = self.policy()
        artifact = {"model":"fixture-model", "revision":"a"*40, "candidate_id":policy.candidate_id,
                    "manifest_sha256":"b"*64, "adapter_sha256":"c"*64}
        report = {"status":"artifact_verified_serving_blocked", "artifact_evidence":artifact}
        controller = runtime.ServingRuntime(policy, lambda _:True, lambda:policy.container_image_digest)
        with patch.object(runtime.model_startup, "_control_bytes", return_value=b"fixture-policy"), \
                patch.object(runtime.model_startup, "validate_policy", return_value={"verification_enabled":True,"artifact":{"root":"/fixture/model"}}), \
                patch.object(runtime.model_startup, "run_startup", return_value=(0,report)), \
                patch.object(runtime, "_environment_guard"), patch.object(runtime, "NativeVllm", fixtures.FakeBackend):
            await controller.start()
            old = (await controller.identity())["worker_lifecycle_id"]
            controller.stop()
            await controller.start()
            new = (await controller.identity())["worker_lifecycle_id"]
            self.assertNotEqual(old, new)
            with self.assertRaises(runtime.ServingRuntimeError):
                await controller.engine_for_server(old)
            self.assertIsNotNone(await controller.engine_for_server(new))
            controller.stop()
