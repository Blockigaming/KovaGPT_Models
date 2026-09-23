"""Synthetic CPU rehearsal only: no deployment, real models, tools or credentials."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import patch

from release.rollout import (ENGINES, ROOT, SOURCE_CONFIG, RehearsalJournal, RolloutRejected,
                            build_route_manifest, digest, gate_names, policy_digest,
                            traffic_for, validate_plan, validate_source_config)


def fixture_plan():
    engines = {}
    for engine in ENGINES:
        app = f"kova-{engine}-staging-fixture"
        slots = {}
        for slot, letter in (("baseline", "a"), ("candidate", "b")):
            slots[slot] = {"revision": app + "--" + slot, "image_digest": letter * 64,
                           "manifest_sha256": letter * 64, "model_revision": letter * 40}
        engines[engine] = {"app": app, **slots}
    return {"schema_version": 1, "kind": "synthetic_rehearsal", "source_commit": "c" * 40,
            "policy_sha256": policy_digest(), "engines": engines,
            "canary_weights": [10, 50, 100]}  # Fixture, NOT approved rollout percentages.


def observation(plan, weights=0, *, intent_id=None, observed_at=1000):
    if type(weights) is int:
        weights = {e: weights for e in ENGINES}
    engines = {}
    for e, deployment in plan["engines"].items():
        engines[e] = {"app": deployment["app"], "active_revisions_mode": "Multiple",
                      "external": False, "allow_insecure": False,
                      **{s: {**deployment[s], "active": True, "healthy": True} for s in ("baseline", "candidate")},
                      "traffic": traffic_for(deployment, weights[e])}
    return {"kind": "synthetic_rehearsal", "plan_sha256": digest(plan), "intent_id": intent_id,
            "settled_intent_id": intent_id, "observed_at_ms": observed_at, "valid_until_ms": 2000, "engines": engines}


def evidence(plan, version):
    return {"kind": "synthetic_rehearsal", "plan_sha256": digest(plan), "state_version": version,
            "observed_at_ms": 1000, "valid_until_ms": 2000,
            "routes": {r["route_id"]: {g: "pass" for g in gate_names()} for r in build_route_manifest()}}


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = str(Path(self.temp.name) / "rehearsal.sqlite3")
        self.plan = fixture_plan()
        self.store = RehearsalJournal(self.path, initialize=True)
        self.addCleanup(lambda: self.store.close())
        self.store.create(self.plan, observation(self.plan), now_ms=1100)

    def promote(self):
        state = self.store.status()
        return self.store.promote(state["version"], observation(self.plan, state["confirmed_weight"]),
                                  evidence(self.plan, state["version"]), now_ms=1100)

    def confirm(self, intent):
        return self.store.confirm(self.store.status()["version"], intent["id"],
            observation(self.plan, intent["weight"], intent_id=intent["id"], observed_at=1200), now_ms=1300)

    def test_complete_rollout_records_observations_not_just_intents(self):
        for weight in self.plan["canary_weights"]:
            previous = self.store.status()["confirmed_weight"]
            intent = self.promote()
            self.assertEqual(intent["weight"], weight)
            self.assertEqual(self.store.status()["confirmed_weight"], previous)
            self.assertFalse(intent["apply_authorized"])
            self.assertTrue(intent["dry_run_only"])
            state = self.confirm(intent)
            self.assertEqual(state["confirmed_weight"], weight)
            self.assertFalse(state["phase_b_ready"])
        self.assertEqual(len(self.store.status()["history"]), 6)
        with self.assertRaises(RolloutRejected):
            self.promote()

    def test_current_policy_or_family_revision_expires_a_staged_plan(self):
        original = Path.read_bytes
        for relative in ("config/current-product-policy.v3.json", "router/entitlements.py",
                         "release/model_revisions.py", "core/current_candidates.py",
                         "prompts/kova-identity.v3.txt", "core/identity.py",
                         "config/kova-three-family-dataset.v2.json", "worker/handler.py"):
            def changed(path):
                data = original(path)
                return data + b"\n# changed" if str(path).endswith(relative) else data
            with self.subTest(relative=relative), patch.object(Path, "read_bytes", changed):
                self.assertNotEqual(policy_digest(), self.plan["policy_sha256"])
                with self.assertRaises(RolloutRejected):
                    validate_plan(self.plan)

    def test_missing_failed_unknown_or_manual_pending_route_gates_block_every_route(self):
        for route in evidence(self.plan, 0)["routes"]:
            for bad in (None, "fail", "unknown", "pending_human_review", True):
                ev = evidence(self.plan, 0)
                ev["routes"][route][gate_names()[0]] = bad
                with self.subTest(route=route, bad=bad), self.assertRaises(RolloutRejected):
                    self.store.promote(0, observation(self.plan), ev, now_ms=1100)
            ev = evidence(self.plan, 0)
            del ev["routes"][route]
            with self.assertRaises(RolloutRejected):
                self.store.promote(0, observation(self.plan), ev, now_ms=1100)
        self.assertEqual(self.store.status()["version"], 0)

    def test_all_required_quality_privacy_latency_cost_and_tool_gates_are_required(self):
        for gate in gate_names():
            ev = evidence(self.plan, 0)
            del ev["routes"]["instant"][gate]
            with self.subTest(gate=gate), self.assertRaises(RolloutRejected):
                self.store.promote(0, observation(self.plan), ev, now_ms=1100)

    def test_expired_future_wrong_plan_or_stale_version_evidence_is_rejected(self):
        for changes in ({"valid_until_ms": 1100}, {"observed_at_ms": 1200},
                        {"plan_sha256": "d" * 64}, {"state_version": 1}, {"state_version": False},
                        {"kind": "live"}, {"valid_until_ms": float("inf")}):
            with self.subTest(changes=changes), self.assertRaises(RolloutRejected):
                self.store.promote(0, observation(self.plan), {**evidence(self.plan, 0), **changes}, now_ms=1100)

    def test_partial_two_engine_write_remains_pending_and_can_be_rolled_back(self):
        intent = self.promote()
        partial = observation(self.plan, {"core": 10, "ultra": 0}, intent_id=intent["id"], observed_at=1200)
        with self.assertRaises(RolloutRejected):
            self.store.confirm(1, intent["id"], partial, now_ms=1300)
        self.assertEqual(self.store.status()["pending"]["id"], intent["id"])
        partial["intent_id"] = None
        rollback = self.store.rollback(1, partial, now_ms=1300)
        state = self.store.confirm(2, rollback["id"],
            observation(self.plan, 0, intent_id=rollback["id"], observed_at=1400), now_ms=1500)
        self.assertTrue(state["rolled_back"])
        self.assertEqual(state["confirmed_weight"], 0)
        self.assertFalse(state["production_traffic_changed"])

    def test_unhealthy_candidate_does_not_block_rollback_to_healthy_baseline(self):
        intent = self.promote()
        self.confirm(intent)
        obs = observation(self.plan, 10)
        for e in ENGINES:
            obs["engines"][e]["candidate"]["healthy"] = False
        rollback = self.store.rollback(2, obs, now_ms=1100)
        self.assertEqual(rollback["weight"], 0)
        self.assertIsNone(rollback["evidence_sha256"])

    def test_unhealthy_or_inactive_baseline_blocks_rollback(self):
        for flag in ("healthy", "active"):
            obs = observation(self.plan)
            obs["engines"]["core"]["baseline"][flag] = False
            with self.assertRaises(RolloutRejected):
                self.store.rollback(0, obs, now_ms=1100)

    def test_unsettled_change_cannot_be_confirmed_or_superseded_by_rollback(self):
        intent = self.promote()
        obs = observation(self.plan, 10, intent_id=intent["id"], observed_at=1200)
        obs["settled_intent_id"] = None
        with self.assertRaises(RolloutRejected):
            self.store.confirm(1, intent["id"], obs, now_ms=1300)
        obs["intent_id"] = None
        with self.assertRaises(RolloutRejected):
            self.store.rollback(1, obs, now_ms=1300)
        self.assertEqual(self.store.status()["pending"]["id"], intent["id"])

    def test_baseline_health_loss_after_intent_prevents_confirmation(self):
        intent = self.promote()
        obs = observation(self.plan, 10, intent_id=intent["id"], observed_at=1200)
        obs["engines"]["ultra"]["baseline"]["healthy"] = False
        with self.assertRaises(RolloutRejected):
            self.store.confirm(1, intent["id"], obs, now_ms=1300)

    def test_pending_intent_is_durable_across_close_and_reopen_without_retry(self):
        intent = self.promote()
        self.store.close()
        self.store = RehearsalJournal(self.path)
        self.assertEqual(self.store.status()["pending"]["id"], intent["id"])
        with self.assertRaises(RolloutRejected):
            self.promote()
        self.assertEqual(self.confirm(intent)["confirmed_weight"], 10)

    def test_separate_connections_cannot_both_promote_the_same_version(self):
        barrier = Barrier(2)
        def try_promote(_):
            with_store = RehearsalJournal(self.path)
            try:
                barrier.wait(timeout=2)
                try:
                    return with_store.promote(0, observation(self.plan), evidence(self.plan, 0), now_ms=1100)
                except RolloutRejected:
                    return None
            finally:
                with_store.close()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(try_promote, range(2)))
        self.assertEqual(sum(r is not None for r in results), 1)
        self.assertEqual(self.store.status()["version"], 1)

    def test_child_exit_after_committed_intent_preserves_uncertain_change(self):
        # os._exit skips cleanup, approximating loss after the durable intent.
        script = """import os,sys
from release.test_rollout import *
s=RehearsalJournal(sys.argv[1]); p=fixture_plan()
s.promote(0,observation(p),evidence(p,0),now_ms=1100)
os._exit(9)
"""
        result = subprocess.run([sys.executable, "-c", script, self.path], cwd=ROOT, timeout=10, capture_output=True)
        self.assertEqual(result.returncode, 9, result.stderr.decode())
        self.assertIsNotNone(self.store.status()["pending"])
        with self.assertRaises(RolloutRejected):
            self.promote()

    def test_transaction_failure_does_not_publish_partial_intent(self):
        save = self.store._save
        def fail(state):
            save(state)
            raise RuntimeError("synthetic commit failure")
        with patch.object(self.store, "_save", side_effect=fail), self.assertRaises(RuntimeError):
            self.promote()
        self.assertEqual(self.store.status()["version"], 0)
        self.assertEqual(self.store.status()["history"], [])

    def test_duplicate_confirmation_and_stale_controller_versions_are_rejected(self):
        intent = self.promote()
        with self.assertRaises(RolloutRejected):
            self.store.rollback(0, observation(self.plan), now_ms=1100)
        self.confirm(intent)
        with self.assertRaises(RolloutRejected):
            self.confirm(intent)

    def test_old_unbound_observation_cannot_confirm_rollback(self):
        rollback = self.store.rollback(0, observation(self.plan), now_ms=1100)
        for obs in (observation(self.plan), observation(self.plan, intent_id=rollback["id"], observed_at=1000)):
            with self.assertRaises(RolloutRejected):
                self.store.confirm(1, rollback["id"], obs, now_ms=1300)
        self.assertIsNotNone(self.store.status()["pending"])

    def test_identity_image_manifest_and_revision_drift_block_promotion(self):
        for slot in ("baseline", "candidate"):
            for field in ("revision", "image_digest", "manifest_sha256", "model_revision"):
                obs = observation(self.plan)
                obs["engines"]["core"][slot][field] += "d"
                with self.subTest(slot=slot, field=field), self.assertRaises(RolloutRejected):
                    self.store.promote(0, obs, evidence(self.plan, 0), now_ms=1100)

    def test_private_ingress_multiple_revision_and_active_health_preconditions(self):
        for field, bad in (("external", True), ("allow_insecure", True), ("active_revisions_mode", "Single")):
            obs = observation(self.plan)
            obs["engines"]["core"][field] = bad
            with self.assertRaises(RolloutRejected):
                self.store.promote(0, obs, evidence(self.plan, 0), now_ms=1100)
        obs = observation(self.plan)
        obs["engines"]["ultra"]["candidate"]["healthy"] = False
        with self.assertRaises(RolloutRejected):
            self.store.promote(0, obs, evidence(self.plan, 0), now_ms=1100)

    def test_unknown_labels_latest_aliases_duplicate_and_invalid_weights_fail_closed(self):
        for mutate in (lambda t: t[0].update(latestRevision=True), lambda t: t[0].update(label="production"),
                       lambda t: t[0].update(weight=99), lambda t: t[1].update(weight=False),
                       lambda t: t.append(deepcopy(t[0])), lambda t: t[1].update(revisionName=t[0]["revisionName"])):
            obs = observation(self.plan)
            mutate(obs["engines"]["core"]["traffic"])
            with self.assertRaises(RolloutRejected):
                self.store.promote(0, obs, evidence(self.plan, 0), now_ms=1100)

    def test_policy_drift_rejects_reuse_of_saved_rehearsal(self):
        with patch("release.rollout.policy_digest", return_value="d" * 64), self.assertRaises(RolloutRejected):
            self.store.status()

    def test_invalid_plans_cannot_promote_defaults_or_guess_steps(self):
        for changes in ({"kind": "production"}, {"canary_weights": []}, {"canary_weights": [100]},
                        {"canary_weights": [10, 10, 100]}, {"canary_weights": [False, 100]},
                        {"canary_weights": [50, 10, 100]}, {"source_commit": "main"},
                        {"apply_enabled": True}, {"policy_sha256": "d" * 64}):
            with self.subTest(changes=changes), self.assertRaises(RolloutRejected):
                validate_plan({**self.plan, **changes})

    def test_actual_source_config_is_disabled_and_all_selections_unset(self):
        value = json.loads((ROOT / "config/model-rollout.v1.json").read_text())
        self.assertFalse(validate_source_config(value)["phase_b_ready"])
        for key in SOURCE_CONFIG["safety"]:
            for bad in (True, None, 0):
                changed = deepcopy(value)
                changed["safety"][key] = bad
                with self.assertRaises(RolloutRejected):
                    validate_source_config(changed)
        for key in value["selection"]:
            changed = deepcopy(value)
            changed["selection"][key] = "invented"
            with self.assertRaises(RolloutRejected):
                validate_source_config(changed)

    def test_cli_rejects_execute_and_defaults_to_disabled_validation(self):
        good = subprocess.run([sys.executable, "-m", "release.rollout"], cwd=ROOT, capture_output=True, timeout=10)
        self.assertEqual(good.returncode, 0, good.stderr.decode())
        self.assertEqual(json.loads(good.stdout)["azure_requests_made"], 0)
        bad = subprocess.run([sys.executable, "-m", "release.rollout", "--execute"], cwd=ROOT, capture_output=True, timeout=10)
        self.assertNotEqual(bad.returncode, 0)

    def test_rehearsal_does_not_open_network_or_spawn_cloud_tools(self):
        with patch("socket.socket", side_effect=AssertionError("network")), \
             patch("subprocess.Popen", side_effect=AssertionError("cloud subprocess")):
            self.confirm(self.promote())
        self.assertEqual(self.store.status()["azure_requests_made"], 0)

    def test_returned_intents_are_defensive_copies(self):
        intent = self.promote()
        intent["weight"] = 100
        self.assertEqual(self.store.status()["pending"]["weight"], 10)

    def test_rollback_is_terminal_and_cannot_restart_candidate_or_repeat_pending_rollback(self):
        intent = self.store.rollback(0, observation(self.plan), now_ms=1100)
        with self.assertRaises(RolloutRejected):
            self.store.rollback(1, observation(self.plan), now_ms=1100)
        self.confirm(intent)
        with self.assertRaises(RolloutRejected):
            self.promote()
        with self.assertRaises(RolloutRejected):
            self.store.rollback(self.store.status()["version"], observation(self.plan), now_ms=1100)


if __name__ == "__main__":
    unittest.main()
