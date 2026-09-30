"""Approved policy through selection, immutable job admission and fixture execution."""

import unittest

from execution.contracts import ExecutionBlocked, ExecutionGrant
from execution.runner import LocalRunner
from execution.store import LocalJobStore
from execution.test_support import ModelFixture, OWNER, SyntheticAdapterTestCase, make_spec, tokens
from release.current_product_policy import load_policy
from router.application import SELECTION_SCHEMA, resolve_application_selection
from router.entitlements import CHAT_ALLOWED_BY_TIER, WORK_ALLOWED_BY_TIER
from ultra.orchestrator import build_ultra_plan


def selection(route):
    surface, family, effort = route.split(":")
    return {"schema_version": SELECTION_SCHEMA, "surface": surface,
            "family": family, "effort": effort.replace("-", " ").title()}


class EntitlementJobAdmissionTests(SyntheticAdapterTestCase):
    def test_every_approved_route_reaches_job_admission_and_every_other_route_is_denied(self):
        policy = load_policy()
        routes = {f"{surface}:{family}:{effort}"
                  for surface in ("chat", "work")
                  for family in (("cosmo", "orion") if surface == "chat"
                                 else ("cosmo", "orion", "nova"))
                  for effort in ("light", "medium", "high", "extra-high", "max", "ultra")}
        specs = {route: make_spec(route) for route in routes}
        store = LocalJobStore()
        self.addCleanup(store.close)
        for tier in ("free", "plus", "pro"):
            for route in sorted(routes):
                with self.subTest(tier=tier, route=route):
                    surface, family, effort = route.split(":")
                    approved = effort in policy["entitlements"][surface][tier][family]
                    allowed = CHAT_ALLOWED_BY_TIER if surface == "chat" else WORK_ALLOWED_BY_TIER
                    self.assertEqual(route in allowed[tier], approved)
                    grant = ExecutionGrant(OWNER, tier, frozenset((route,)), True)
                    if approved:
                        chosen = resolve_application_selection(selection(route), grant=grant)
                        self.assertEqual(chosen.route_id, route)
                        job = store.create(grant, f"{tier}-{route}", specs[route])
                        self.assertEqual(store.status(OWNER, job)["state"], "queued")
                    else:
                        with self.assertRaises(ExecutionBlocked):
                            resolve_application_selection(selection(route), grant=grant)
                        with self.assertRaises(ExecutionBlocked):
                            store.create(grant, f"{tier}-{route}", specs[route])

    def test_representative_free_plus_and_pro_jobs_complete_through_current_grants(self):
        for tier, route in (("free", "chat:cosmo:light"),
                            ("plus", "work:nova:ultra"),
                            ("pro", "chat:orion:ultra")):
            with self.subTest(tier=tier, route=route):
                spec = make_spec(route)
                grant = ExecutionGrant(OWNER, tier, frozenset((route,)), True)
                chosen = resolve_application_selection(selection(route), grant=grant)
                self.assertEqual(chosen.route_id, spec.plan["route_id"])
                store = LocalJobStore()
                self.addCleanup(store.close)
                job = store.create(grant, f"fixture-{tier}", spec)
                fixture = ModelFixture()
                self.assertEqual(LocalRunner(store, lambda: grant, fixture.worker()).run(job)["state"], "succeeded")
                self.assertEqual(store.result(OWNER, job)["content"], "Kova final response")
                self.assertEqual(len(fixture.calls), len(set(fixture.calls)))

    def test_plus_work_ultra_planner_accepts_plus_but_chat_ultra_requires_pro(self):
        admission = {"entitlement": "plus", "ultra_authorized": True,
                     "remaining_usd": 1, "estimated_max_usd": 0.5,
                     "max_agents": 3, "max_total_tokens": 65536}
        plan = build_ultra_plan(
            {"request_id": "plus-work", "surface": "work", "family": "nova",
             "effort": "Ultra", "task": "Summarize the synthetic task."},
            admission=admission, token_counter=lambda messages: tokens("kova-nova", messages))
        self.assertEqual(plan["route_id"], "work:nova:ultra")
        with self.assertRaisesRegex(ValueError, "Chat Ultra requires Pro"):
            build_ultra_plan(
                {"request_id": "plus-chat", "surface": "chat", "family": "orion",
                 "effort": "Ultra", "task": "Summarize the synthetic task."},
                admission=admission, token_counter=lambda messages: tokens("kova-orion", messages))


if __name__ == "__main__":
    unittest.main()
