"""Disabled release planning and durable, SYNTHETIC-ONLY rollout rehearsal.

No Azure client, subprocess, credentials, model call or apply operation exists.
The SQLite journal tests intent/observation separation; it is not a production
release-controller store. See docs/model-rollout-rehearsal.md for live gates.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from evaluation.offline import build_route_manifest

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ("core", "ultra")
MAX_BYTES = 1024 * 1024
SOURCE_CONFIG = {
    "schema_version": 1,
    "status": "disabled_source_only",
    "implementation": "release/rollout.py",
    "selection": {"release_plan": None, "canary_weights": None, "observation_validity_ms": None},
    "safety": {"apply_enabled": False, "azure_access_authorized": False,
               "production_routing_authorized": False, "gpu_execution_authorized": False,
               "phase_b_ready": False},
}


class RolloutRejected(ValueError):
    """Sanitized release contract failure; not permission to retry an operation."""


def need(condition):
    if not condition:
        raise RolloutRejected("release rehearsal contract rejected")


def keys(value, expected):
    need(type(value) is dict and set(value) == set(expected))


def integer(value, minimum=0, maximum=2**53 - 1):
    need(type(value) is int and minimum <= value <= maximum)
    return value


def canonical(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, RecursionError):
        raise RolloutRejected("release rehearsal contract rejected") from None
    need(len(raw) <= MAX_BYTES)
    return raw


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pin(value, length=64):
    need(type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None)


def policy_digest():
    # A staged plan must expire when any current family, entitlement, model,
    # planner, or evidence source changes. Archived routing files are excluded.
    paths = ("release/rollout.py", "config/current-product-policy.v3.json",
             "prompts/kova-identity.v3.txt", "core/identity.py",
             "config/kova-three-family-dataset.v2.json",
             "config/kova-runtime-profiles.v1.json", "router/policy.py",
             "config/ultra-orchestration.v1.json", "config/activity-event.v1.json",
             "config/kova-three-family-evaluation.v1.json", "config/model-rollout.v1.json",
             "router/entitlements.py", "router/auto.py", "router/application.py",
             "release/model_revisions.py", "core/current_candidates.py",
             "core/adapter.py", "ultra/orchestrator.py", "ultra/conversation.py",
             "worker/handler.py", "ultra/binding.py", "execution/workers.py",
             "execution/source_context.py", "evaluation/offline.py",
             "worker/model_artifact.py", "worker/model_startup.py",
             "worker/serving_runtime.py", "execution/contracts.py",
             "scripts/summarize-core-benchmark.mjs",
             "evaluation/three_family_guard.py", "config/evaluation-gates.v1.json",
             "evaluations/offline-suite.v1.json")
    return digest({path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths})


def gate_names():
    value = json.loads((ROOT / "config/evaluation-gates.v1.json").read_text())
    return tuple(value["required_per_route"])


def validate_source_config(value):
    # Exact canonical comparison rejects booleans coerced to numbers and new flags.
    need(canonical(value) == canonical(SOURCE_CONFIG))
    return {"status": "disabled_release_source_valid", "phase_b_ready": False,
            "azure_requests_made": 0, "production_traffic_changed": False}


def validate_plan(plan):
    canonical(plan)
    keys(plan, ("schema_version", "kind", "source_commit", "policy_sha256", "engines", "canary_weights"))
    need(type(plan["schema_version"]) is int and plan["schema_version"] == 1)
    # The only executable state machine is a rehearsal. Real inputs must never be
    # relabelled synthetic to run it; a separately reviewed live controller is needed.
    need(plan["kind"] == "synthetic_rehearsal")
    pin(plan["source_commit"], 40)
    need(plan["policy_sha256"] == policy_digest())
    keys(plan["engines"], ENGINES)
    for engine, deployment in plan["engines"].items():
        keys(deployment, ("app", "baseline", "candidate"))
        name = deployment["app"]
        need(type(name) is str and re.fullmatch(r"kova-" + engine + r"-staging-[a-z0-9]{3,12}", name) is not None)
        for slot in ("baseline", "candidate"):
            item = deployment[slot]
            keys(item, ("revision", "image_digest", "manifest_sha256", "model_revision"))
            need(type(item["revision"]) is str and re.fullmatch(re.escape(name) + r"--[a-z][a-z0-9-]{0,30}[a-z0-9]", item["revision"]) is not None)
            need("--" not in item["revision"][len(name) + 2:])
            pin(item["image_digest"])
            pin(item["manifest_sha256"])
            pin(item["model_revision"], 40)
        need(deployment["baseline"]["revision"] != deployment["candidate"]["revision"])
    weights = plan["canary_weights"]
    need(type(weights) is list and 2 <= len(weights) <= 100)
    for weight in weights:
        integer(weight, 1, 100)
    need(weights == sorted(set(weights)) and weights[0] < 100 and weights[-1] == 100)
    return deepcopy(plan)


def traffic_for(deployment, weight):
    integer(weight, 0, 100)
    return [
        {"revisionName": deployment["baseline"]["revision"], "latestRevision": False, "weight": 100 - weight},
        {"revisionName": deployment["candidate"]["revision"], "latestRevision": False, "weight": weight},
    ]


def validate_observation(plan, observation, now_ms):
    integer(now_ms, 1)
    keys(observation, ("kind", "plan_sha256", "observed_at_ms", "valid_until_ms", "intent_id", "settled_intent_id", "engines"))
    need(observation["kind"] == "synthetic_rehearsal" and observation["plan_sha256"] == digest(plan))
    need(observation["intent_id"] is None or (type(observation["intent_id"]) is str and re.fullmatch(r"[a-f0-9-]{36}", observation["intent_id"]) is not None))
    need(observation["settled_intent_id"] is None or (type(observation["settled_intent_id"]) is str and re.fullmatch(r"[a-f0-9-]{36}", observation["settled_intent_id"]) is not None))
    integer(observation["observed_at_ms"], 1)
    integer(observation["valid_until_ms"], 1)
    need(observation["observed_at_ms"] <= now_ms < observation["valid_until_ms"])
    keys(observation["engines"], ENGINES)
    weights = {}
    for engine, actual in observation["engines"].items():
        keys(actual, ("app", "active_revisions_mode", "external", "allow_insecure", "baseline", "candidate", "traffic"))
        deployment = plan["engines"][engine]
        need(actual["app"] == deployment["app"] and actual["active_revisions_mode"] == "Multiple")
        need(actual["external"] is False and actual["allow_insecure"] is False)
        for slot in ("baseline", "candidate"):
            keys(actual[slot], (*deployment[slot].keys(), "active", "healthy"))
            need(canonical({k: actual[slot][k] for k in deployment[slot]}) == canonical(deployment[slot]))
            need(actual[slot]["active"] is True and type(actual[slot]["healthy"]) is bool)
        rules = actual["traffic"]
        need(type(rules) is list and len(rules) == 2)
        by_name = {}
        for rule in rules:
            keys(rule, ("revisionName", "latestRevision", "weight"))
            need(rule["latestRevision"] is False and type(rule["revisionName"]) is str)
            need(rule["revisionName"] not in by_name)
            by_name[rule["revisionName"]] = integer(rule["weight"], 0, 100)
        need(set(by_name) == {deployment[s]["revision"] for s in ("baseline", "candidate")})
        need(sum(by_name.values()) == 100)
        weights[engine] = by_name[deployment["candidate"]["revision"]]
    return weights


def validate_gates(plan, state, evidence, now_ms):
    keys(evidence, ("kind", "plan_sha256", "state_version", "observed_at_ms", "valid_until_ms", "routes"))
    need(evidence["kind"] == "synthetic_rehearsal" and evidence["plan_sha256"] == digest(plan))
    integer(evidence["state_version"])
    need(evidence["state_version"] == state["version"])
    integer(evidence["observed_at_ms"], 1)
    integer(evidence["valid_until_ms"], 1)
    need(evidence["observed_at_ms"] <= now_ms < evidence["valid_until_ms"])
    routes = [item["route_id"] for item in build_route_manifest()]
    keys(evidence["routes"], routes)
    for result in evidence["routes"].values():
        keys(result, gate_names())
        need(all(value == "pass" for value in result.values()))
    return digest(evidence)


class RehearsalJournal:
    """A durable synthetic intent ledger. Nothing here applies Azure changes.

    A process crash preserves the pending intent. No timeout, closed connection,
    missing acknowledgement or fresh process makes promotion/retry safe. Both
    engines must be freshly observed at the exact target before confirmation.
    """
    def __init__(self, path, *, initialize=False):
        need(type(initialize) is bool)
        self.db = sqlite3.connect(path, timeout=2, isolation_level=None)
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            if initialize:
                self.db.execute("CREATE TABLE rehearsal (id INTEGER PRIMARY KEY CHECK(id=1), plan TEXT NOT NULL, state TEXT NOT NULL)")
            self.db.execute("SELECT state FROM rehearsal LIMIT 0")
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    @contextmanager
    def _transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def create(self, plan, observation, *, now_ms):
        plan = validate_plan(plan)
        weights = validate_observation(plan, observation, now_ms)
        need(observation["intent_id"] is None and observation["settled_intent_id"] is None and all(w == 0 for w in weights.values()))
        need(all(observation["engines"][e]["baseline"]["healthy"] for e in ENGINES))
        state = {"version": 0, "step": -1, "confirmed_weight": 0, "pending": None,
                 "rolled_back": False, "history": []}
        with self._transaction():
            need(self.db.execute("SELECT count(*) FROM rehearsal").fetchone()[0] == 0)
            self.db.execute("INSERT INTO rehearsal VALUES (1,?,?)", (canonical(plan).decode(), canonical(state).decode()))
        return self.status()

    def _read(self, expected_version=None):
        row = self.db.execute("SELECT plan,state FROM rehearsal WHERE id=1").fetchone()
        need(row is not None)
        plan, state = map(json.loads, row)
        validate_plan(plan)  # Running changed policy cannot reuse an old release.
        if expected_version is not None:
            integer(expected_version)
            need(state["version"] == expected_version)
        return plan, state

    def _save(self, state):
        state["version"] += 1
        need(len(state["history"]) <= 205)
        self.db.execute("UPDATE rehearsal SET state=? WHERE id=1", (canonical(state).decode(),))

    def status(self):
        _, state = self._read()
        return {**state, "kind": "synthetic_rehearsal", "phase_b_ready": False,
                "production_traffic_changed": False, "azure_requests_made": 0}

    def _intent(self, plan, state, action, weight, observation, now_ms, evidence_digest=None):
        intent = {"id": str(uuid4()), "action": action, "weight": weight,
                  "plan_sha256": digest(plan), "requested_at_ms": now_ms, "observation_sha256": digest(observation),
                  "evidence_sha256": evidence_digest,
                  "dry_run_only": True, "apply_authorized": False,
                  "traffic_by_engine": {e: traffic_for(plan["engines"][e], weight) for e in ENGINES}}
        state["pending"] = intent
        state["history"].append({"event": "intent_recorded", "intent_id": intent["id"], "action": action,
                                 "weight": weight, "observation_sha256": digest(observation),
                                 "evidence_sha256": evidence_digest})
        self._save(state)
        return deepcopy(intent)

    def promote(self, expected_version, observation, evidence, *, now_ms):
        with self._transaction():
            plan, state = self._read(expected_version)
            need(state["pending"] is None and not state["rolled_back"])
            next_step = state["step"] + 1
            need(next_step < len(plan["canary_weights"]))
            weights = validate_observation(plan, observation, now_ms)
            need(observation["intent_id"] is None and all(w == state["confirmed_weight"] for w in weights.values()))
            need(all(observation["engines"][e][slot]["healthy"] for e in ENGINES for slot in ("baseline", "candidate")))
            evidence_digest = validate_gates(plan, state, evidence, now_ms)
            return self._intent(plan, state, "promote", plan["canary_weights"][next_step], observation, now_ms, evidence_digest)

    def rollback(self, expected_version, observation, *, now_ms):
        with self._transaction():
            plan, state = self._read(expected_version)
            need(not state["rolled_back"])
            need(state["pending"] is None or state["pending"]["action"] != "rollback")
            validate_observation(plan, observation, now_ms)
            need(observation["intent_id"] is None)
            need(all(observation["engines"][e]["baseline"]["healthy"] for e in ENGINES))
            if state["pending"] is not None:
                # A prior asynchronous write must be settled before a rollback
                # supersedes it; an observed weight alone is not that evidence.
                need(observation["settled_intent_id"] == state["pending"]["id"])
                requested_at = integer(state["pending"].get("requested_at_ms"), 1)
                need(observation["observed_at_ms"] >= requested_at)
            # A bad candidate never needs to pass quality gates to leave traffic.
            # Partial application is recorded, not misrepresented as atomic.
            return self._intent(plan, state, "rollback", 0, observation, now_ms)

    def confirm(self, expected_version, intent_id, observation, *, now_ms):
        with self._transaction():
            plan, state = self._read(expected_version)
            intent = state["pending"]
            need(intent is not None and intent["id"] == intent_id)
            weights = validate_observation(plan, observation, now_ms)
            need(observation["intent_id"] == intent_id and observation["settled_intent_id"] == intent_id
                 and observation["observed_at_ms"] >= intent["requested_at_ms"])
            need(all(w == intent["weight"] for w in weights.values()))
            healthy_slots = ("baseline",) if intent["action"] == "rollback" else ("baseline", "candidate")
            need(all(observation["engines"][e][slot]["healthy"] for e in ENGINES for slot in healthy_slots))
            state["confirmed_weight"] = intent["weight"]
            if intent["action"] == "rollback":
                state["rolled_back"] = True
            else:
                state["step"] += 1
            state["history"].append({"event": "observation_confirmed", "intent_id": intent_id,
                                     "observation_sha256": digest(observation)})
            state["pending"] = None
            self._save(state)
        return self.status()


if __name__ == "__main__":
    # No --execute/--apply switch or arbitrary external plan input is accepted.
    import sys
    need(len(sys.argv) == 1)
    config = json.loads((ROOT / "config/model-rollout.v1.json").read_text())
    print(json.dumps(validate_source_config(config), sort_keys=True))
