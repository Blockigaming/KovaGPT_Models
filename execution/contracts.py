"""Immutable execution snapshots and server-only authorization contracts.

These are internal APIs, not an authentication boundary for untrusted JSON. The
application must authenticate an owner and construct grants on the server.
There are no default production deadlines, spending limits or parallelism values.
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
import re

from router.entitlements import (
    CHAT_ALLOWED_BY_TIER, COMPAT_CHAT_ALLOWED_BY_TIER, WORK_ALLOWED_BY_TIER,
)
from router.policy import CHAT_FAMILIES, CHAT_POLICIES, WORK_FAMILIES, WORK_EFFORTS
from release.model_revisions import source_reference_for_route


class ExecutionError(ValueError):
    """An execution contract or state transition was rejected."""


class ExecutionBlocked(ExecutionError):
    """An execution or current-plan entitlement is not authorized."""


class ExecutionBusy(ExecutionError):
    """Another runner owns this job; automatic takeover is forbidden."""


class ExecutionCancelled(ExecutionError):
    """The current execution was cancelled."""


class ExecutionExpired(ExecutionError):
    """The original job or stage deadline has elapsed."""


class ExecutionInterrupted(ExecutionError):
    """An in-flight outcome is unknown and must not be silently retried."""


class ExecutionIntegrityError(ExecutionError):
    """A persisted snapshot, stage output or runtime identity failed validation."""


def require(condition, message):
    if not condition:
        raise ExecutionError(message)


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ExecutionError("execution snapshot must be finite UTF-8 JSON") from None


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def positive_integer(value, name, maximum=2**53 - 1):
    require(type(value) is int and 0 < value <= maximum, f"invalid {name}")


def identifier(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:@-]{1,128}", value),
            f"invalid {name}")


CANONICAL_CHAT_ROUTES = frozenset(
    f"chat:{family}:{effort.lower().replace(' ', '-')}"
    for family in CHAT_FAMILIES for effort in WORK_EFFORTS
)
ALL_ROUTES = frozenset(CHAT_POLICIES) | CANONICAL_CHAT_ROUTES | frozenset(
    f"work:{family}:{effort.lower().replace(' ', '-')}"
    for family in WORK_FAMILIES for effort in WORK_EFFORTS
)
# Direct Chat route entitlement. Free Thinking is a separately authorized app-mode
# alias to medium/Orion; it intentionally does not make direct Free `medium` valid.
CHAT_ALLOWED = COMPAT_CHAT_ALLOWED_BY_TIER


@dataclass(frozen=True)
class ExecutionGrant:
    """Current authenticated owner/entitlement, supplied only by trusted server code.

    Plan policy is an upper bound; the exact route must also be present in the
    server-built allowed_routes set. Free Thinking is the only approved alias that
    may authorize a route above the direct Free Chat set.
    """

    owner_id: str
    tier: str
    allowed_routes: frozenset[str]
    execution_authorized: bool = False

    def __post_init__(self):
        identifier(self.owner_id, "owner_id")
        require(isinstance(self.tier, str) and self.tier in CHAT_ALLOWED, "invalid tier")
        require(type(self.allowed_routes) is frozenset
                and self.allowed_routes <= ALL_ROUTES, "invalid allowed routes")
        require(type(self.execution_authorized) is bool, "invalid execution grant")

    def authorize(self, owner_id, route_id, *, application_mode_id=None):
        if not self.execution_authorized or owner_id != self.owner_id:
            raise ExecutionBlocked("execution is not authorized for this owner")
        if application_mode_id is not None and not isinstance(application_mode_id, str):
            raise ExecutionBlocked("invalid application mode authorization")
        if route_id not in self.allowed_routes:
            raise ExecutionBlocked("route is not in the current server entitlement")
        if route_id in CHAT_POLICIES:
            if route_id not in CHAT_ALLOWED[self.tier]:
                raise ExecutionBlocked("route exceeds the current Chat plan")
        elif route_id.startswith("chat:"):
            if route_id not in CHAT_ALLOWED_BY_TIER[self.tier]:
                raise ExecutionBlocked("route exceeds the current Chat plan")
        elif route_id.startswith("work:") and route_id not in WORK_ALLOWED_BY_TIER[self.tier]:
            raise ExecutionBlocked("route exceeds the current Work plan")
        if route_id == "ultra" and self.tier != "pro":
            raise ExecutionBlocked("Ultra requires current Pro entitlement")


@dataclass(frozen=True)
class ExecutionLimits:
    """Explicit per-job admission limits, not model response-time targets.

    deadline_unix_ms is a wall-clock expiry including queue time and pauses. A
    restart does not renew it. Cost reservations are conservative admission
    allocations, NOT measured invoices or guarantees that remote billing stops.
    """

    deadline_unix_ms: int
    token_limit: int
    cost_limit_microusd: int
    max_parallel: int
    stage_timeout_seconds: float

    def __post_init__(self):
        for name in ("deadline_unix_ms", "token_limit", "cost_limit_microusd"):
            positive_integer(getattr(self, name), name)
        positive_integer(self.max_parallel, "max_parallel", 5)
        require(type(self.stage_timeout_seconds) in (int, float)
                and math.isfinite(self.stage_timeout_seconds)
                and 0 < self.stage_timeout_seconds < 240, "invalid stage timeout")

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class Stage:
    id: str
    dependencies: tuple[str, ...]
    maximum_input_tokens: int
    maximum_output_tokens: int
    cost_cap_microusd: int
    public: bool
    activity: bool
    condition: str | None = None

    @property
    def token_reservation(self):
        return self.maximum_input_tokens + self.maximum_output_tokens

    def as_dict(self):
        value = {name: getattr(self, name) for name in self.__dataclass_fields__}
        value["dependencies"] = list(self.dependencies)
        return value


@dataclass(frozen=True)
class ExecutionSpec:
    """Immutable bytes keep prompt/policy snapshots independent of caller mutations.

    Construct with from_plan using a server-built Core/Ultra plan. The SHA-256
    checksum detects accidental corruption, not a malicious database administrator.
    """

    encoded: bytes = field(repr=False)

    def __post_init__(self):
        require(isinstance(self.encoded, bytes) and 0 < len(self.encoded) <= 8 * 1024 * 1024,
                "invalid execution snapshot size")
        try:
            value = json.loads(self.encoded)
            require(canonical(value) == self.encoded, "noncanonical execution snapshot")
            require(set(value) == {"schema_version", "plan", "limits", "stages", "runtime_identity"},
                    "invalid execution snapshot fields")
            require(value["schema_version"] == 1, "unsupported execution snapshot")
            ExecutionLimits(**value["limits"])
            identity = value["runtime_identity"]
            require(set(identity) == {"model", "model_revision", "adapter_sha256", "context_tokens"},
                    "invalid execution runtime identity")
            require(isinstance(identity["model"], str) and identity["model"].strip(), "missing model")
            require(isinstance(identity["model_revision"], str)
                    and re.fullmatch(r"[a-f0-9]{40}", identity["model_revision"]), "unpinned model")
            require(isinstance(identity["adapter_sha256"], str)
                    and re.fullmatch(r"[a-f0-9]{64}", identity["adapter_sha256"]),
                    "unpinned trained adapter")
            positive_integer(identity["context_tokens"], "context tokens")
            plan = value["plan"]
            require(plan["route_id"] in ALL_ROUTES, "invalid execution route")
            require(plan["production_ready"] is False and plan["endpoint_deployed"] is False,
                    "source-only execution plan required")
            require(plan["engine"] in ("kova-core", "kova-ultra"), "invalid engine")
            if plan["engine"] == "kova-core":
                require(plan["candidate_model"] == identity["model"]
                        and plan["candidate_revision"] == identity["model_revision"], "Core identity mismatch")
            else:
                require(plan["model_selection_required"] is True, "Ultra live model remains unselected")
                source = source_reference_for_route(plan["route_id"])
                require(identity["model"] == source.slot and identity["model_revision"] == source.revision,
                        "Ultra identity differs from selected route family")
            from core.current_candidates import CORE_SERVING
            candidates = [c for c in CORE_SERVING["candidates"] if c["model"] == identity["model"]
                          and c["revision"] == identity["model_revision"]]
            # Persisted snapshots retain the admitted adapter identity across pin rotation.
            # The current pin is enforced on admission and again at worker dispatch.
            require(len(candidates) == 1, "snapshot model differs from selected candidate")
            stages = value["stages"]
            require(isinstance(stages, list) and 1 <= len(stages) <= 16, "invalid stage count")
            seen = set()
            for raw, operation in zip(stages, plan["operations"], strict=True):
                require(set(raw) == set(Stage.__dataclass_fields__), "invalid stage fields")
                stage = Stage(**{**raw, "dependencies": tuple(raw["dependencies"])})
                identifier(stage.id, "stage ID")
                require(stage.id not in seen and isinstance(raw["dependencies"], list), "duplicate stage")
                require(len(set(stage.dependencies)) == len(stage.dependencies)
                        and set(stage.dependencies) <= seen, "invalid stage dependency order")
                for field_name in ("maximum_input_tokens", "maximum_output_tokens", "cost_cap_microusd"):
                    positive_integer(getattr(stage, field_name), field_name)
                require(type(stage.public) is bool and type(stage.activity) is bool, "invalid visibility")
                require(stage.token_reservation <= identity["context_tokens"], "stage exceeds served context")
                expected_id = operation.get("stage_id", operation.get("id"))
                expected_deps = operation.get("depends_on_stage_ids", operation.get("depends_on"))
                require(stage.id == expected_id and list(stage.dependencies) == expected_deps,
                        "stage differs from source plan")
                require(stage.maximum_input_tokens == operation["maximum_input_tokens"]
                        and stage.maximum_output_tokens == operation["maximum_output_tokens"]
                        and stage.public == operation.get("public_response", operation.get("public_output"))
                        and stage.activity == operation["activity_event_allowed_after_start"],
                        "stage budget or visibility differs from source plan")
                require(stage.condition == operation.get("condition"), "stage condition mismatch")
                require(stage.condition is None or (
                    plan["engine"] == "kova-ultra" and stage.id == "debate-round-1"
                    and stage.condition == "judge_detected_material_disagreement"
                    and operation["maximum_rounds"] == 1), "unsupported stage condition")
                seen.add(stage.id)
            require([s["id"] for s in stages if s["public"]] == [stages[-1]["id"]],
                    "exactly the final stage may publish")
            require(sum(s["maximum_input_tokens"] + s["maximum_output_tokens"] for s in stages)
                    <= value["limits"]["token_limit"], "route exceeds token admission")
            require(sum(s["cost_cap_microusd"] for s in stages) <= value["limits"]["cost_limit_microusd"],
                    "route exceeds cost reservation admission")
        except (KeyError, TypeError, ValueError, RecursionError):
            raise ExecutionError("invalid execution snapshot or admission limits") from None

    @classmethod
    def from_plan(cls, plan, *, limits, runtime_identity, stage_cost_caps):
        require(type(limits) is ExecutionLimits, "explicit execution limits required")
        require(isinstance(plan, dict) and isinstance(stage_cost_caps, dict), "trusted plan/cost bounds required")
        from core.current_candidates import CORE_SERVING
        require(isinstance(runtime_identity, dict), "runtime identity required")
        admitted = [c for c in CORE_SERVING["candidates"]
                    if c["model"] == runtime_identity.get("model")
                    and c["revision"] == runtime_identity.get("model_revision")]
        require(len(admitted) == 1 and runtime_identity.get("adapter_sha256") == admitted[0]["adapter_sha256"]
                and isinstance(admitted[0]["adapter_sha256"], str), "adapter differs from current candidate pin")
        operations = plan.get("operations", [])
        ids = [op.get("stage_id", op.get("id")) for op in operations]
        require(set(stage_cost_caps) == set(ids), "every stage requires an explicit server cost cap")
        stages = [Stage(
            id=stage_id,
            dependencies=tuple(op.get("depends_on_stage_ids", op.get("depends_on", []))),
            maximum_input_tokens=op["maximum_input_tokens"],
            maximum_output_tokens=op["maximum_output_tokens"],
            cost_cap_microusd=stage_cost_caps[stage_id],
            public=op.get("public_response", op.get("public_output")),
            activity=op["activity_event_allowed_after_start"],
            condition=op.get("condition"),
        ).as_dict() for stage_id, op in zip(ids, operations, strict=True)]
        return cls(canonical({"schema_version": 1, "plan": plan, "stages": stages,
                              "limits": limits.as_dict(), "runtime_identity": runtime_identity}))

    @property
    def fingerprint(self):
        return hashlib.sha256(self.encoded).hexdigest()

    def snapshot(self):
        return json.loads(self.encoded)

    @property
    def plan(self):
        return self.snapshot()["plan"]

    @property
    def limits(self):
        return ExecutionLimits(**self.snapshot()["limits"])

    @property
    def stages(self):
        return tuple(Stage(**{**raw, "dependencies": tuple(raw["dependencies"])})
                     for raw in self.snapshot()["stages"])
