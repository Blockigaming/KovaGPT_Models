"""Fail-closed cross-file checks for the authoritative three-family policy."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED = (
    "model-catalog.v1.json", "product-surface.v1.json", "route-policy.v1.json",
    "application-bridge.v1.json", "core-serving.v1.json",
)
PUBLIC_SOURCE = (
    "router/policy.py", "router/entitlements.py", "router/application.py",
)
CURRENT_RUNTIME_SOURCE = (
    "core/adapter.py", "core/current_candidates.py", "worker/handler.py",
    "worker/model_artifact.py", "release/rollout.py", "config/inference-contract.v1.json",
    "scripts/summarize-core-benchmark.mjs",
)
CURRENT_CANDIDATE_SOURCE = "core/current_candidates.py:CORE_SERVING.candidates"
CURRENT_CANDIDATE_ASSERTION = (
    'inference.candidate_selection.allowlist_source !== '
    f'"{CURRENT_CANDIDATE_SOURCE}"'
)
CURRENT_BENCHMARK_ASSERTION = (
    'inference.trusted_execution_context.benchmark_candidate_id_source !== '
    '"trusted_server_configuration_allowlisted_in_current_candidates"'
)
APPROVED_RUNTIME_IDENTITY_PATH = "prompts/kova-identity.v3.txt"
APPROVED_RUNTIME_IDENTITY_SHA256 = "ed1b503f947cabc6a7c24a9395bd63b9eff2dd55d57570a0d5a8fd51df3c5bc8"
IDENTITY_RUNTIME_CALLERS = {
    "core/adapter.py": ("IDENTITY", "build_core_plan"),
    "ultra/orchestrator.py": ("IDENTITY", "build_ultra_plan"),
    "worker/handler.py": ("TRUSTED_SYSTEM_IDENTITY", "build_engine_request"),
}
FORBIDDEN_PUBLIC = ("Qwen/", "Qwen3-", "qwen3-", "Kova 5.6", "chat-shared")


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate_key:{key}")
        value[key] = item
    return value


def _load(path: Path) -> dict:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"bom_forbidden:{path.name}")
    return json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_unique)


def _policy_routes(policy: dict, surface: str, tier: str) -> frozenset[str]:
    return frozenset(
        f"{surface}:{family}:{level}"
        for family, levels in policy["entitlements"][surface][tier].items()
        for level in levels
    )


def _runtime_entitlements(root: Path):
    path = root / "router/entitlements.py"
    spec = importlib.util.spec_from_file_location("_kova_source_policy_runtime_entitlements", path)
    if spec is None or spec.loader is None:
        raise ValueError("runtime_entitlements_unloadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CHAT_ALLOWED_BY_TIER, module.WORK_ALLOWED_BY_TIER


def _runtime_identity(root: Path) -> None:
    prompt_path = root / APPROVED_RUNTIME_IDENTITY_PATH
    try:
        prompt = prompt_path.read_bytes()
    except OSError as error:
        raise ValueError("approved_runtime_identity_prompt_unavailable") from error
    if hashlib.sha256(prompt).hexdigest() != APPROVED_RUNTIME_IDENTITY_SHA256:
        raise ValueError("approved_runtime_identity_prompt_drift")

    loader_path = root / "core/identity.py"
    source = loader_path.read_text("utf-8", errors="strict")
    if "identity.v1.json" in source:
        raise ValueError("runtime_reads_archived_identity:core/identity.py")
    spec = importlib.util.spec_from_file_location("_kova_source_policy_runtime_identity", loader_path)
    if spec is None or spec.loader is None:
        raise ValueError("runtime_identity_loader_unloadable")
    module = importlib.util.module_from_spec(spec)
    # Compile the inspected source so a copied tree cannot pass using stale bytecode.
    exec(compile(source, str(loader_path), "exec"), module.__dict__)
    if (module.PROMPT_PATH != prompt_path or
            module.APPROVED_PROMPT_SHA256 != APPROVED_RUNTIME_IDENTITY_SHA256 or
            module.load_runtime_identity() != prompt.decode("utf-8", "strict")):
        raise ValueError("runtime_identity_loader_drift")

    for relative, (binding, entrypoint) in IDENTITY_RUNTIME_CALLERS.items():
        source = (root / relative).read_text("utf-8", errors="strict")
        if "identity.v1.json" in source:
            raise ValueError(f"runtime_reads_archived_identity:{relative}")
        tree = ast.parse(source, filename=relative)
        imported = any(
            isinstance(node, ast.ImportFrom) and node.module == "core.identity" and
            any(alias.name == "load_runtime_identity" and alias.asname is None
                for alias in node.names)
            for node in tree.body
        )
        initialized = any(
            isinstance(node, ast.Assign) and
            any(isinstance(target, ast.Name) and target.id == binding
                for target in node.targets) and
            isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Name) and
            node.value.func.id == "load_runtime_identity"
            for node in tree.body
        )
        entrypoints = (
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entrypoint
        )
        called = any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
            node.func.id == "load_runtime_identity"
            for function in entrypoints for node in ast.walk(function)
        )
        if not (imported and initialized and called):
            raise ValueError(f"runtime_identity_loader_drift:{relative}")


def validate(root: Path = ROOT) -> dict:
    policy = _load(root / "config/current-product-policy.v3.json")
    expected = {"free": (1, 0), "plus": (6, 18), "pro": (12, 18)}
    if any(policy["entitlements"]["chat"][tier]["nova"] for tier in expected):
        raise ValueError("nova_must_be_work_only")
    for tier, (chat, work) in expected.items():
        actual_chat = sum(len(levels) for levels in policy["entitlements"]["chat"][tier].values())
        actual_work = sum(len(levels) for levels in policy["entitlements"]["work"][tier].values())
        if actual_chat != chat or actual_work != work:
            raise ValueError(f"entitlement_drift:{tier}")

    runtime_chat, runtime_work = _runtime_entitlements(root)
    for surface, runtime in (("chat", runtime_chat), ("work", runtime_work)):
        if set(runtime) != set(expected):
            raise ValueError(f"entitlement_runtime_tier_drift:{surface}")
        for tier in expected:
            if frozenset(runtime[tier]) != _policy_routes(policy, surface, tier):
                raise ValueError(f"entitlement_runtime_drift:{surface}:{tier}")

    if policy["processing_levels_are_separate_models"] is not False:
        raise ValueError("runtime_profiles_must_not_be_models")
    _runtime_identity(root)
    legacy_identity = _load(root / "config/identity.v1.json")
    if (legacy_identity.get("status") != "superseded_non_authoritative_history" or
            legacy_identity.get("superseded_by") != APPROVED_RUNTIME_IDENTITY_PATH):
        raise ValueError("legacy_identity_source_not_superseded")
    for name in ARCHIVED:
        value = _load(root / "config" / name)
        if value.get("status") != "superseded_non_authoritative_history":
            raise ValueError(f"legacy_file_not_superseded:{name}")
        if value.get("must_not_drive_current_routing") is not True:
            raise ValueError(f"legacy_file_can_drive_routing:{name}")
    for relative in CURRENT_RUNTIME_SOURCE:
        source = (root / relative).read_text("utf-8", errors="strict")
        for legacy in ("route-policy.v1.json", "core-serving.v1.json"):
            if legacy in source:
                raise ValueError(f"runtime_reads_archived_routing:{relative}:{legacy}")
    contract = _load(root / "config/inference-contract.v1.json")
    if (contract["candidate_selection"]["allowlist_source"] != CURRENT_CANDIDATE_SOURCE or
            contract["trusted_execution_context"]["benchmark_candidate_id_source"] !=
            "trusted_server_configuration_allowlisted_in_current_candidates"):
        raise ValueError("inference_candidate_source_drift")
    # The stack validator also verifies archived history; inspect only its active assertions.
    validator = (root / "scripts/validate-stack.mjs").read_text("utf-8", errors="strict")
    if (CURRENT_CANDIDATE_ASSERTION not in validator or
            CURRENT_BENCHMARK_ASSERTION not in validator):
        raise ValueError("inference_validator_candidate_source_drift")
    benchmark = (root / "scripts/summarize-core-benchmark.mjs").read_text(
        "utf-8", errors="strict")
    if ("from worker.handler import ALLOWED_SERVING_ENGINES, ALLOWED_ENDPOINT_TYPES"
            not in benchmark or
            "servingCapabilities = current.capabilities" not in benchmark):
        raise ValueError("benchmark_capability_source_drift")
    for relative in PUBLIC_SOURCE:
        source = (root / relative).read_text("utf-8", errors="strict")
        for token in FORBIDDEN_PUBLIC:
            if token in source:
                raise ValueError(f"private_upstream_leak:{Path(relative).name}:{token}")
    gate_names = (
        "resource_creation_authorized", "spending_authorized", "model_download_authorized",
        "training_authorized", "deployment_authorized", "production_routing_enabled",
    )
    if any(policy[name] is not False for name in gate_names):
        raise ValueError("current_policy_gate_promoted")
    return {
        "status": "authoritative_three_family_policy_no_drift",
        "chat_routes": {tier: counts[0] for tier, counts in expected.items()},
        "work_routes": {tier: counts[1] for tier, counts in expected.items()},
        "provider_calls_made": 0,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), sort_keys=True))
