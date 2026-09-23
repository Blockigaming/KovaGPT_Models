"""Deterministic async lifecycle races; no native model, GPU or network calls."""

import asyncio
import hashlib
import unittest
from unittest.mock import patch

from worker import serving_runtime as runtime
from worker.test_serving_runtime import FakeBackend


class HeldHealthBackend(FakeBackend):
    def __init__(self, root, artifact, policy):
        super().__init__(root, artifact, policy)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def health(self):
        self.entered.set()
        await self.release.wait()
        await super().health()

    def hold_next_health(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()


class RuntimeInterleavingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.policy = runtime.LoaderPolicy(
            "/fixture/startup.json", hashlib.sha256(b"fixture-policy").hexdigest(),
            "fixture-candidate", "sha256:" + "a" * 64, 8192, 0.8, 1, 60, 2, 2,
            True, True, True,
        )
        artifact = {"model": "fixture-model", "revision": "a" * 40,
                    "candidate_id": self.policy.candidate_id, "manifest_sha256": "b" * 64,
                    "adapter_sha256": "c" * 64}
        report = {"status": "artifact_verified_serving_blocked", "artifact_evidence": artifact}
        self.created = asyncio.Queue()
        self.tasks = []
        self.controller = runtime.ServingRuntime(
            self.policy, lambda _: True, lambda: self.policy.container_image_digest,
        )

        def factory(root, evidence, policy):
            backend = HeldHealthBackend(root, evidence, policy)
            self.created.put_nowait(backend)
            return backend

        patches = (
            patch.object(runtime.model_startup, "_control_bytes", return_value=b"fixture-policy"),
            patch.object(runtime.model_startup, "validate_policy", return_value={
                "verification_enabled": True, "artifact": {"root": "/fixture/model"}}),
            patch.object(runtime.model_startup, "run_startup", return_value=(0, report)),
            patch.object(runtime, "_environment_guard"),
            patch.object(runtime, "NativeVllm", side_effect=factory),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.controller._backend is not None:
            self.controller._backend.bad_close = False
        self.controller.stop()

    def task(self, coroutine):
        value = asyncio.create_task(coroutine)
        self.tasks.append(value)
        return value

    async def pending_start(self):
        task = self.task(self.controller.start())
        backend = await asyncio.wait_for(self.created.get(), 1)
        await asyncio.wait_for(backend.entered.wait(), 1)
        return task, backend

    async def ready_backend(self):
        task, backend = await self.pending_start()
        backend.release.set()
        self.assertTrue((await task)["ready"])
        return backend

    async def pending_identity(self, *, handoff=False):
        backend = await self.ready_backend()
        old_id = self.controller.status()["instance_id"]
        backend.hold_next_health()
        task = self.task(self.controller.engine_for_server(old_id) if handoff else self.controller.identity())
        await asyncio.wait_for(backend.entered.wait(), 1)
        return task, backend

    async def test_old_start_cannot_mark_unchecked_replacement_ready(self):
        old_task, old = await self.pending_start()
        self.controller.stop()
        new_task, new = await self.pending_start()
        old.release.set()
        with self.assertRaises(runtime.ServingRuntimeError):
            await old_task
        self.assertEqual(self.controller.status()["state"], "loading")
        self.assertEqual(new.closed, [])
        new.release.set()
        self.assertTrue((await new_task)["ready"])

    async def test_failed_old_start_cannot_close_ready_replacement(self):
        task, old = await self.pending_start()
        self.controller.stop()
        new = await self.ready_backend()
        old.bad_health = True
        old.release.set()
        with self.assertRaises(runtime.ServingRuntimeError):
            await task
        self.assertTrue(self.controller.status()["ready"])
        self.assertEqual(new.closed, [])

    async def test_old_identity_cannot_return_new_lifecycle_metadata(self):
        task, old = await self.pending_identity()
        self.controller.stop()
        new = await self.ready_backend()
        old.release.set()
        with self.assertRaises(runtime.ServingRuntimeError):
            await task
        self.assertTrue(self.controller.status()["ready"])
        self.assertEqual(new.closed, [])

    async def test_old_engine_request_cannot_follow_replacement(self):
        task, old = await self.pending_identity(handoff=True)
        self.controller.stop()
        new = await self.ready_backend()
        old.release.set()
        with self.assertRaises(runtime.ServingRuntimeError):
            await task
        self.assertTrue(self.controller.status()["ready"])
        self.assertEqual(new.closed, [])

    async def test_old_identity_failure_does_not_close_replacement(self):
        task, old = await self.pending_identity()
        self.controller.stop()
        new = await self.ready_backend()
        old.bad_health = True
        old.release.set()
        with self.assertRaises(runtime.ServingRuntimeError):
            await task
        self.assertTrue(self.controller.status()["ready"])
        self.assertEqual(new.closed, [])

    async def test_cancelled_old_identity_does_not_close_replacement(self):
        task, _ = await self.pending_identity()
        self.controller.stop()
        new = await self.ready_backend()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.controller.status()["ready"])
        self.assertEqual(new.closed, [])

    async def test_cancelled_old_start_does_not_close_replacement(self):
        task, _ = await self.pending_start()
        self.controller.stop()
        new = await self.ready_backend()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.controller.status()["ready"])
        self.assertEqual(new.closed, [])

    async def test_pending_start_finishes_after_stop_without_overwriting_stopped_state(self):
        task, old = await self.pending_start()
        self.controller.stop()
        old.release.set()
        with self.assertRaises(runtime.ServingRuntimeError):
            await task
        self.assertEqual(self.controller.status()["state"], "stopped")
        self.assertEqual(old.closed, [True])

    async def test_pending_identity_finishes_after_stop_without_overwriting_stopped_state(self):
        task, old = await self.pending_identity()
        self.controller.stop()
        old.release.set()
        with self.assertRaises(runtime.ServingRuntimeError):
            await task
        self.assertEqual(self.controller.status()["state"], "stopped")
        self.assertEqual(old.closed, [True])

    async def test_current_start_cancellation_still_closes_owned_backend_and_propagates(self):
        task, backend = await self.pending_start()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.controller.status()["ready"])
        self.assertEqual(self.controller.status()["state"], "failed")
        self.assertEqual(backend.closed, [True])

    async def test_shutdown_failure_still_blocks_new_start_and_late_success(self):
        task, backend = await self.pending_start()
        backend.bad_close = True
        with self.assertRaises(runtime.ServingRuntimeError):
            self.controller.stop()
        with self.assertRaises(runtime.ServingRuntimeError):
            await self.controller.start()
        backend.release.set()
        with self.assertRaises(runtime.ServingRuntimeError):
            await task
        self.assertEqual(self.controller.status()["state"], "failed")
        backend.bad_close = False

    async def test_observed_context_and_remote_code_flags_require_exact_types(self):
        for field, value in (("context_tokens", 8192.0), ("trust_remote_code", 0)):
            backend = await self.ready_backend()
            backend.observed[field] = value
            with self.subTest(field=field), self.assertRaises(runtime.ServingRuntimeError):
                await self.controller.identity()
            self.controller.stop()


if __name__ == "__main__":
    unittest.main()
