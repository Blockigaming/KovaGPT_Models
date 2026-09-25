from dataclasses import replace
import unittest

from execution.contracts import (
    ALL_ROUTES, ExecutionBlocked, ExecutionError, ExecutionGrant, ExecutionLimits,
    ExecutionSpec, canonical,
)
from execution.test_support import IDENTITY, SyntheticAdapterTestCase, grant_for, make_plan, make_spec
from core.current_candidates import CORE_SERVING


class ExecutionContractTests(SyntheticAdapterTestCase):
    def test_new_admission_rejects_context_above_current_candidate_limit(self):
        spec = make_spec("high")
        identity = spec.snapshot()["runtime_identity"]
        candidate = next(c for c in CORE_SERVING["candidates"] if c["model"] == identity["model"])
        caps = {stage.id: stage.cost_cap_microusd for stage in spec.stages}
        inflated = {**identity, "context_tokens": candidate["context_tokens"] + 1}
        with self.assertRaisesRegex(ExecutionError, "served context exceeds current candidate context"):
            ExecutionSpec.from_plan(spec.plan, limits=spec.limits,
                                    runtime_identity=inflated, stage_cost_caps=caps)

        previous_limit = candidate["context_tokens"]
        self.addCleanup(candidate.__setitem__, "context_tokens", previous_limit)
        candidate["context_tokens"] = previous_limit - 1
        # Existing snapshots stay readable after a candidate context reduction.
        self.assertEqual(ExecutionSpec(spec.encoded).fingerprint, spec.fingerprint)
        with self.assertRaisesRegex(ExecutionError, "served context exceeds current candidate context"):
            ExecutionSpec.from_plan(spec.plan, limits=spec.limits,
                                    runtime_identity=identity, stage_cost_caps=caps)

    def test_saved_snapshot_survives_adapter_rotation_but_new_admission_does_not(self):
        spec = make_spec("instant")
        candidate = next(c for c in CORE_SERVING["candidates"] if c["model"] == spec.snapshot()["runtime_identity"]["model"])
        candidate["adapter_sha256"] = "e" * 64
        self.assertEqual(ExecutionSpec(spec.encoded).fingerprint, spec.fingerprint)
        with self.assertRaisesRegex(ExecutionError, "adapter differs from current candidate pin"):
            ExecutionSpec.from_plan(spec.plan, limits=spec.limits,
                                    runtime_identity=spec.snapshot()["runtime_identity"],
                                    stage_cost_caps={stage.id: stage.cost_cap_microusd for stage in spec.stages})

    def test_saved_snapshot_survives_bundle_rotation_but_new_admission_does_not(self):
        spec = make_spec("instant")
        candidate = next(c for c in CORE_SERVING["candidates"] if c["model"] == spec.snapshot()["runtime_identity"]["model"])
        candidate["adapter_bundle_sha256"] = "e" * 64
        self.assertEqual(ExecutionSpec(spec.encoded).fingerprint, spec.fingerprint)
        with self.assertRaisesRegex(ExecutionError, "adapter bundle differs from current candidate pin"):
            ExecutionSpec.from_plan(spec.plan, limits=spec.limits,
                                    runtime_identity=spec.snapshot()["runtime_identity"],
                                    stage_cost_caps={stage.id: stage.cost_cap_microusd for stage in spec.stages})

    def test_ultra_snapshot_rejects_another_family_identity(self):
        spec = make_spec("work:nova:ultra")
        changed = spec.snapshot()
        changed["runtime_identity"] = dict(IDENTITY)
        with self.assertRaisesRegex(ExecutionError, "invalid execution snapshot"):
            ExecutionSpec(canonical(changed))

    def test_all_36_explicit_and_migration_routes_keep_bounded_budgets(self):
        self.assertEqual(len(ALL_ROUTES), 36)
        for route in sorted(ALL_ROUTES):
            with self.subTest(route=route):
                spec = make_spec(route)
                self.assertEqual(spec.plan["route_id"], route)
                self.assertEqual(spec.limits.token_limit, sum(s.token_reservation for s in spec.stages))
                self.assertEqual(len([s for s in spec.stages if s.public]), 1)
                self.assertTrue(spec.stages[-1].public)

    def test_explicit_limits_reject_nonfinite_missing_boolean_and_invalid_values(self):
        baseline = make_spec().limits
        for name in ("deadline_unix_ms", "token_limit", "cost_limit_microusd", "max_parallel", "stage_timeout_seconds"):
            for value in (None, True, False, 0, -1, float("inf"), float("nan"), "10"):
                with self.subTest(name=name, value=value), self.assertRaises(ExecutionError):
                    replace(baseline, **{name: value})
        for value in (240, 600):
            with self.assertRaises(ExecutionError):
                replace(baseline, stage_timeout_seconds=value)
        with self.assertRaises(ExecutionError):
            replace(baseline, max_parallel=6)

    def test_source_and_budget_snapshots_are_defensive(self):
        plan = make_plan()
        expected = make_spec()
        caps = {s.id: 100 for s in expected.stages}
        spec = ExecutionSpec.from_plan(plan, limits=expected.limits,
                                       runtime_identity=expected.snapshot()["runtime_identity"],
                                       stage_cost_caps=caps)
        fingerprint = spec.fingerprint
        plan["display_name"] = "changed"
        caps.clear()
        copy = spec.plan
        copy["display_name"] = "changed again"
        self.assertEqual(spec.fingerprint, fingerprint)
        self.assertEqual(spec.plan["display_name"], "Kova Orion — High")
        self.assertNotIn("PRIVATE", repr(spec))

    def test_every_stage_needs_explicit_cost_bounds_and_total_admission(self):
        original = make_spec()
        caps = {s.id: 100 for s in original.stages}
        for changed in ({}, {**caps, "unknown": 1}, {**caps, original.stages[0].id: None}):
            with self.assertRaises(ExecutionError):
                ExecutionSpec.from_plan(original.plan, limits=original.limits, runtime_identity=IDENTITY,
                                        stage_cost_caps=changed)
        for field in ("token_limit", "cost_limit_microusd"):
            with self.subTest(field=field), self.assertRaises(ExecutionError):
                ExecutionSpec.from_plan(original.plan, limits=replace(original.limits, **{field: 1}),
                                        runtime_identity=IDENTITY, stage_cost_caps=caps)

    def test_snapshot_rejects_tampered_dependencies_budgets_visibility_and_pins(self):
        for mutate in (
            lambda v: v["stages"][0].update(dependencies=["verification-1"]),
            lambda v: v["stages"][0].update(id="forged"),
            lambda v: v["stages"][0].update(public=True),
            lambda v: v["stages"][0].update(maximum_output_tokens=999),
            lambda v: v["stages"][0].update(condition="always"),
            lambda v: v["runtime_identity"].update(model_revision="unpinned"),
            lambda v: v["runtime_identity"].update(adapter_sha256="not-a-digest"),
            lambda v: v["runtime_identity"].update(adapter_bundle_sha256="not-a-digest"),
            lambda v: v["runtime_identity"].update(context_tokens=1),
            lambda v: v["plan"].update(production_ready=True),
            lambda v: v["plan"].update(route_id="thinking"),
        ):
            value = make_spec().snapshot()
            mutate(value)
            with self.assertRaises(ExecutionError):
                ExecutionSpec(canonical(value))

    def test_direct_chat_plan_caps_cannot_be_widened_by_allowed_route_list(self):
        for tier, denied in (("free", ["medium", "high", "ultra"]), ("plus", ["extra-high", "max", "ultra"])):
            for route in denied:
                spec = make_spec(route)
                with self.subTest(tier=tier, route=route), self.assertRaises(ExecutionBlocked):
                    grant_for(spec, tier=tier).authorize("fixture-owner", route)

    def test_superseded_free_thinking_alias_is_rejected(self):
        grant = ExecutionGrant("fixture-owner", "free", frozenset(("medium",)), True)
        with self.assertRaises(ExecutionBlocked):
            grant.authorize("fixture-owner", "medium", application_mode_id="thinking")

    def test_complete_work_matrix_and_exact_route_allowlist_are_both_required(self):
        efforts = ("light", "medium", "high", "extra-high", "max", "ultra")
        for tier in ("free", "plus", "pro"):
            for family in ("cosmo", "orion", "nova"):
                for effort in efforts:
                    route = f"work:{family}:{effort}"
                    grant = ExecutionGrant("fixture-owner", tier, frozenset((route,)), True)
                    expected = tier in ("plus", "pro")
                    with self.subTest(tier=tier, route=route):
                        if expected:
                            grant.authorize("fixture-owner", route)
                        else:
                            with self.assertRaises(ExecutionBlocked):
                                grant.authorize("fixture-owner", route)
        route = "work:nova:high"
        with self.assertRaises(ExecutionBlocked):
            ExecutionGrant("fixture-owner", "pro", frozenset(), True).authorize("fixture-owner", route)

    def test_thinking_auto_and_application_aliases_are_not_execution_route_ids(self):
        for route in ("thinking", "kova-auto", "auto", "extra_high"):
            with self.subTest(route=route), self.assertRaises(ExecutionError):
                ExecutionGrant("fixture", "pro", frozenset((route,)), True)

    def test_grants_disabled_by_default_and_bound_to_the_owner(self):
        spec = make_spec()
        grant = ExecutionGrant("fixture-owner", "pro", frozenset(("high",)))
        with self.assertRaises(ExecutionBlocked):
            grant.authorize(grant.owner_id, "high")
        with self.assertRaises(ExecutionBlocked):
            grant_for(spec).authorize("different-owner", "high")


if __name__ == "__main__":
    unittest.main()
