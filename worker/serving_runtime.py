"""Verified local artifact -> native vLLM loader -> guarded lifecycle handoff.

No model import/load, networking or process is started on import/construction.
This component is not an HTTP listener; the application request/authentication
adapter must independently guard every inference call. Checked-in settings are
unselected and disabled. CPU tests never install or execute a real vLLM model.
"""

import asyncio
from dataclasses import dataclass
import hashlib
import math
import os
from time import monotonic
from uuid import uuid4

from worker import model_startup


SUPPORTED_VLLM_VERSION = "0.29.0"


class ServingRuntimeError(ValueError):
    """Loading, health, authorization or identity verification failed."""


def need(condition):
    if not condition:
        raise ServingRuntimeError("model runtime unavailable")


def _positive(value, maximum):
    return type(value) is int and 0 < value <= maximum


@dataclass(frozen=True)
class LoaderPolicy:
    startup_policy_path: str
    startup_policy_sha256: str
    candidate_id: str
    container_image_digest: str
    context_tokens: int
    gpu_memory_utilization: float
    maximum_sequences: int
    load_timeout_seconds: float
    health_timeout_seconds: float
    shutdown_timeout_seconds: float
    enabled: bool = False
    model_loading_authorized: bool = False
    gpu_execution_authorized: bool = False

    def __post_init__(self):
        need(type(self.startup_policy_path) is str and os.path.isabs(self.startup_policy_path)
             and os.path.normpath(self.startup_policy_path) == self.startup_policy_path)
        need(type(self.startup_policy_sha256) is str and len(self.startup_policy_sha256) == 64
             and all(c in "0123456789abcdef" for c in self.startup_policy_sha256))
        need(type(self.container_image_digest) is str and self.container_image_digest.startswith("sha256:")
             and len(self.container_image_digest) == 71
             and all(c in "0123456789abcdef" for c in self.container_image_digest[7:]))
        need(type(self.candidate_id) is str and self.candidate_id)
        need(_positive(self.context_tokens, 262144) and _positive(self.maximum_sequences, 32))
        need(type(self.gpu_memory_utilization) in (int, float)
             and math.isfinite(self.gpu_memory_utilization) and 0 < self.gpu_memory_utilization < 1)
        for value in (self.load_timeout_seconds, self.health_timeout_seconds, self.shutdown_timeout_seconds):
            need(type(value) in (int, float) and math.isfinite(value) and 0 < value <= 3600)
        for value in (self.enabled, self.model_loading_authorized, self.gpu_execution_authorized):
            need(type(value) is bool)


def _environment_guard(root):
    """A real loader requires a nonroot process and OS-enforced read-only mount.

    Offline environment flags are not a substitute for infrastructure egress
    policy. The deployment controller must separately verify network isolation.
    """
    need(os.geteuid() != 0)
    need(bool(os.statvfs(root).f_flag & os.ST_RDONLY))
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_NO_USAGE_STATS", "DO_NOT_TRACK"):
        need(os.environ.get(name) == "1")
    need(os.environ.get("VLLM_PLUGINS") == "")


class NativeVllm:
    """Block native allocation until the trained adapter is loaded on every request."""
    def __init__(self, root, artifact, policy):
        need(type(policy) is LoaderPolicy and policy.enabled
             and policy.model_loading_authorized and policy.gpu_execution_authorized)
        # The former base-only path never applied the verified adapter. A
        # manifest digest alone cannot make the native engine serve it.
        raise ServingRuntimeError("native trained adapter loading is unavailable")


class ServingRuntime:
    """Lifecycle owner for a verified immutable local model artifact.

    authorize(policy) returns literal True only for a current authenticated loading
    approval. observe_image() must come from the trusted container supervisor and
    return the actual immutable image digest, not a request-controlled environment
    string. Deployment/egress/role approval are not inferred from these values.
    Returned identity is a server observation chain, not hardware attestation or
    proof of model quality, latency, actual billing, or production readiness.
    """
    def __init__(self, policy, authorize, observe_image, *, cancelled=lambda: False, clock=monotonic):
        need(type(policy) is LoaderPolicy and all(callable(fn) for fn in (authorize, observe_image, cancelled, clock)))
        self.policy, self._authorize, self._image = policy, authorize, observe_image
        self._cancelled, self._clock = cancelled, clock
        self._backend = None
        self._identity = None
        self._state = "stopped"
        self._instance = str(uuid4())

    def status(self):
        return {"state": self._state, "ready": self._state == "ready",
                "instance_id": self._instance, "phase_b_ready": False,
                "production_routing_authorized": False}

    def _permission(self):
        need(self.policy.enabled and self.policy.model_loading_authorized and self.policy.gpu_execution_authorized)
        try:
            allowed, image, cancelled = self._authorize(self.policy), self._image(), self._cancelled()
        except Exception:
            raise ServingRuntimeError("model loading approval unavailable") from None
        need(allowed is True and image == self.policy.container_image_digest and type(cancelled) is bool and not cancelled)

    def _verify_observed(self):
        need(self._backend is not None and self._identity is not None)
        observed = self._backend.observed_configuration()
        expected = {"model_path": self._identity["artifact_root"], "tokenizer_path": self._identity["artifact_root"],
                    "model_revision": self._identity["model_revision"], "tokenizer_revision": self._identity["model_revision"],
                    "adapter_sha256": self._identity["adapter_sha256"],
                    "served_model_names": [self._identity["model"]],
                    "context_tokens": self.policy.context_tokens, "trust_remote_code": False}
        need(type(observed) is dict and set(observed) == set(expected))
        need(all(type(observed[key]) is type(value) for key, value in expected.items()))
        need(observed == expected)

    def _owns(self, instance, backend, states):
        """Awaited work may only affect the exact lifecycle that started it.

        This controller is owned by one event-loop thread. Synchronous stop may
        run between awaits; replacement engines must not inherit old health or
        be closed by an old task's timeout/cancellation/failure.
        """
        return (self._instance == instance and self._backend is backend
                and self._state in states)

    def _close_failed(self):
        self._state = "failed"
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                # Keep the backend reference for explicit supervisor cleanup;
                # never call a failed shutdown stopped or remotely quiescent.
                return False
            self._backend = None
        return True

    async def start(self):
        need(self._state == "stopped")
        self._permission()  # Before artifact access, native import or model workers.
        self._state = "verifying"
        self._instance = str(uuid4())  # Every new load has a distinct lifecycle fence.
        instance, backend = self._instance, None
        try:
            started = self._clock()
            need(type(started) in (float, int) and math.isfinite(started))
            encoded = model_startup._control_bytes(self.policy.startup_policy_path,
                model_startup.MAX_POLICY_BYTES, self._cancelled)
            need(hashlib.sha256(encoded).hexdigest() == self.policy.startup_policy_sha256)
            startup = model_startup.validate_policy(encoded)
            need(startup["verification_enabled"] is True)
            root = startup["artifact"]["root"]
            _environment_guard(root)
            code, report = model_startup.run_startup(self.policy.startup_policy_path, verify_only=True,
                                                     cancelled=self._cancelled, clock=self._clock)
            need(code == 0 and report["status"] == "artifact_verified_serving_blocked")
            artifact = report["artifact_evidence"]
            need(artifact["candidate_id"] == self.policy.candidate_id)
            again = model_startup._control_bytes(self.policy.startup_policy_path,
                model_startup.MAX_POLICY_BYTES, self._cancelled)
            need(again == encoded)
            self._permission()
            self._identity = {"model": artifact["model"], "model_revision": artifact["revision"],
                "adapter_sha256": artifact["adapter_sha256"],
                "candidate_id": artifact["candidate_id"], "manifest_sha256": artifact["manifest_sha256"],
                "artifact_root": root, "context_tokens": self.policy.context_tokens,
                "container_image_digest": self.policy.container_image_digest,
                "serving_engine": "vllm", "serving_version": SUPPORTED_VLLM_VERSION,
                "worker_lifecycle_id": self._instance}
            self._state = "loading"
            backend = NativeVllm(root, artifact, self.policy)
            self._backend = backend
            self._verify_observed()
            now = self._clock()
            need(type(now) in (int, float) and math.isfinite(now) and 0 <= now-started < self.policy.load_timeout_seconds)
            await asyncio.wait_for(backend.health(), self.policy.health_timeout_seconds)
            need(self._owns(instance, backend, ("loading",)))
            self._permission()
            self._verify_observed()
            need(self._owns(instance, backend, ("loading",)))
            self._state = "ready"
            return self.status()
        except BaseException as error:
            if self._owns(instance, backend, ("verifying", "loading")):
                self._close_failed()
            if not isinstance(error, Exception):
                raise
            raise ServingRuntimeError("model runtime startup failed") from None

    async def identity(self):
        """Recheck health/approval/configuration before exposing a server handoff.

        No prompt, reasoning, credential, wall-time price or fake GPU measurement
        is returned. The private artifact path is omitted from the handoff.
        """
        need(self._state == "ready")
        instance, backend = self._instance, self._backend
        try:
            self._permission()
            self._verify_observed()
            await asyncio.wait_for(backend.health(), self.policy.health_timeout_seconds)
            need(self._owns(instance, backend, ("ready",)))
            self._permission()
            self._verify_observed()
            need(self._owns(instance, backend, ("ready",)))
            return {key: value for key, value in self._identity.items() if key != "artifact_root"}
        except BaseException as error:
            if self._owns(instance, backend, ("ready",)):
                self._close_failed()
            if not isinstance(error, Exception):
                raise
            raise ServingRuntimeError("model runtime health or identity failed") from None

    async def engine_for_server(self, expected_lifecycle_id):
        """Internal server handoff only, never expose this object to untrusted code.

        Receiver authentication, user/plan admission, privacy-filtered protocol
        and request cancellation must still surround every actual generation call.
        This does not create an unprotected OpenAI-compatible public endpoint.
        """
        need(type(expected_lifecycle_id) is str and expected_lifecycle_id == self._instance)
        backend = self._backend
        await self.identity()
        need(self._owns(expected_lifecycle_id, backend, ("ready",)))
        return backend.engine

    def stop(self):
        if self._backend is None:
            self._state = "stopped"
            self._identity = None
            return self.status()
        self._state = "draining"
        try:
            self._backend.close()
        except Exception:
            self._state = "failed"
            raise ServingRuntimeError("model runtime shutdown requires supervisor verification") from None
        self._backend = None
        self._identity = None
        self._state = "stopped"
        return self.status()
