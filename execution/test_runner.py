from dataclasses import replace
import json
from pathlib import Path
import tempfile
from threading import Barrier, Event, Lock, Thread
from time import monotonic, sleep, time_ns
import unittest
from unittest.mock import patch

from execution.activity import activity_from_started_event
from execution.contracts import ExecutionBlocked, ExecutionError, ExecutionSpec, canonical
from execution.runner import LocalRunner
from execution.store import LocalJobStore
from execution.test_support import IDENTITY, OWNER, ModelFixture, grant_for, make_spec, tokens
from execution.workers import ModelStageWorker


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.store = LocalJobStore()

    def tearDown(self):
        self.store.close()

    def run_job(self, spec=None, fixture=None, *, worker=None, authorization=None, max_stages=None):
        spec = spec or make_spec()
        fixture = fixture or ModelFixture()
        grant = grant_for(spec)
        job = self.store.create(grant, str(time_ns()), spec)
        runner = LocalRunner(self.store, authorization or (lambda: grant), worker or fixture.worker())
        status = runner.run(job, max_stages=max_stages)
        return spec, fixture, job, runner, status

    def test_all_current_core_profiles_execute_all_171_stages(self):
        from execution.contracts import ALL_ROUTES
        count = 0
        for route in sorted(ALL_ROUTES - {r for r in ALL_ROUTES if r == "ultra" or r.endswith(":ultra")}):
            with self.subTest(route=route):
                spec, fixture, job, _, status = self.run_job(make_spec(route))
                self.assertEqual(status["state"], "succeeded")
                self.assertEqual(self.store.result(OWNER, job)["content"], "Kova final response")
                self.assertEqual(fixture.calls, [s.id for s in spec.stages])
                self.assertEqual(fixture.closed, fixture.calls)
                self.assertEqual(status["tokens_reserved"], spec.limits.token_limit)
                self.assertEqual(status["cost_reserved_microusd"], spec.limits.cost_limit_microusd)
                self.assertFalse(status["cost_is_measured"])
                for record, stage in zip(fixture.records, spec.stages, strict=True):
                    self.assertEqual(record["public_response"], stage.public)
                    if stage.public:
                        self.assertIsNotNone(record["time_to_first_token_ms"])
                    else:
                        self.assertIsNone(record["time_to_first_token_ms"])
                count += len(spec.stages)
        self.assertEqual(count, 171)

    def test_all_four_ultra_routes_execute_both_conditional_branches(self):
        for route in ("ultra", "work:cosmo:ultra", "work:orion:ultra", "work:nova:ultra"):
            for debate in (False, True):
                with self.subTest(route=route, debate=debate):
                    spec, fixture, job, _, status = self.run_job(make_spec(route), ModelFixture(debate=debate))
                    self.assertEqual(status["state"], "succeeded")
                    self.assertEqual(fixture.calls.count("debate-round-1"), int(debate))
                    self.assertEqual(fixture.calls[-1], "synthesis")
                    self.assertEqual(status["skipped_stages"], int(not debate))
                    expected = [s for s in spec.stages if debate or s.id != "debate-round-1"]
                    self.assertEqual(status["tokens_reserved"], sum(s.token_reservation for s in expected))
                    self.assertEqual(status["cost_reserved_microusd"], 100 * len(expected))
                    self.assertEqual(self.store.result(OWNER, job)["content"], "Kova final response")
                    synthesis = next(request for stage, request in fixture.requests if stage == "synthesis")
                    self.assertFalse(any("{{server_stage_output:" in msg["content"] for msg in synthesis["messages"]))

    def test_specialist_calls_overlap_and_respect_two_through_five_agent_caps(self):
        for agents, task in ((2, "Prove this equation."), (3, "Prove this equation."),
                             (4, "Research market competitors."), (5, "Research market competitors and code.")):
            gate = Barrier(agents)
            active, peak = [0], [0]
            lock = Lock()
            def hook(stage, control):
                if stage.startswith("specialist-"):
                    with lock:
                        active[0] += 1
                        peak[0] = max(peak[0], active[0])
                    gate.wait(timeout=3)
                    control.check()
                    with lock:
                        active[0] -= 1
            with self.subTest(agents=agents):
                spec = make_spec("ultra", agents=agents, parallel=agents, task=task)
                self.assertEqual(sum(s.id.startswith("specialist-") for s in spec.stages), agents)
                _, fixture, _, _, status = self.run_job(spec, ModelFixture(hook=hook))
                self.assertEqual(status["state"], "succeeded")
                self.assertEqual(peak[0], agents)  # Barrier proves concurrent invocations, not labels.
                self.assertEqual(active[0], 0)
                self.assertGreater(fixture.calls.index("disagreement-check"), agents - 1)

    def test_clean_pause_restart_and_resume_reuses_completed_stages(self):
        spec = make_spec("max")
        grant = grant_for(spec)
        fixture = ModelFixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.sqlite3"
            store = LocalJobStore(path)
            job = store.create(grant, "same-client-request", spec)
            status = LocalRunner(store, lambda: grant, fixture.worker()).run(job, max_stages=3)
            self.assertEqual(status["state"], "paused")
            self.assertEqual(len(fixture.calls), 3)
            cursor = store.replay(OWNER, job)["next_sequence"]
            store.close()
            store = LocalJobStore(path)
            self.assertEqual(store.create(grant, "same-client-request", spec), job)
            status = LocalRunner(store, lambda: grant, fixture.worker()).run(job)
            self.assertEqual(status["state"], "succeeded")
            self.assertEqual(fixture.calls, [s.id for s in spec.stages])
            events = store.replay(OWNER, job, after=cursor)["events"]
            self.assertTrue(all(e["sequence"] > cursor for e in events))
            self.assertEqual(status["tokens_reserved"], spec.limits.token_limit)
            store.close()

    def test_multi_stage_work_can_span_240_seconds_without_reusing_one_http_deadline(self):
        simulated_seconds = [0.0]
        base_ms = 1_900_000_000_000
        spec = make_spec("max", deadline=base_ms + 600_000, parallel=1)
        value = spec.snapshot()
        value["limits"]["stage_timeout_seconds"] = 35
        spec = ExecutionSpec(canonical(value))
        grant = grant_for(spec)
        store = LocalJobStore(clock_ms=lambda: base_ms + int(simulated_seconds[0] * 1000))
        job = store.create(grant, "long-fixture", spec)
        calls = []
        def worker(spec, stage, artifacts, control, job_id, attempt):
            control.check()
            calls.append(stage.id)
            simulated_seconds[0] += 30
            control.check()
            return {"content": "Kova fixture output", "tool_calls": [],
                    "input_tokens": 10, "output_tokens": 3, "debate_required": None}
        try:
            status = LocalRunner(store, lambda: grant, worker,
                                 clock=lambda: simulated_seconds[0],
                                 clock_ms=lambda: base_ms + int(simulated_seconds[0] * 1000)).run(job)
            self.assertEqual(status["state"], "succeeded")
            self.assertEqual(simulated_seconds[0], 300)
            self.assertEqual(len(calls), 10)
        finally:
            store.close()

    def test_running_a_completed_job_is_inert(self):
        _, fixture, job, runner, status = self.run_job()
        calls = list(fixture.calls)
        again = runner.run(job)
        self.assertEqual(again, status)
        self.assertEqual(fixture.calls, calls)

    def test_disabled_grant_never_dispatches_a_worker(self):
        spec = make_spec()
        grant = grant_for(spec)
        job = self.store.create(grant, "disabled-test", spec)
        calls = []
        runner = LocalRunner(self.store, lambda: replace(grant, execution_authorized=False), lambda *_: calls.append(1))
        with self.assertRaises(ExecutionBlocked):
            runner.run(job)
        self.assertEqual(calls, [])
        self.assertEqual(self.store.status(OWNER, job)["state"], "queued")

    def test_job_expiry_prevents_late_output_without_renewing_deadline(self):
        spec = make_spec("instant")
        value = spec.snapshot()
        value["limits"]["stage_timeout_seconds"] = 0.02
        spec = ExecutionSpec(canonical(value))
        def slow(stage, control):
            sleep(0.04)
            control.check()
        _, fixture, job, _, status = self.run_job(spec, ModelFixture(hook=slow))
        self.assertEqual(status["state"], "expired")
        self.assertEqual(len(fixture.calls), 1)
        with self.assertRaises(ExecutionError):
            self.store.result(OWNER, job)

    def test_cancellation_during_real_parallel_callbacks_drains_and_stops_later_stages(self):
        spec = make_spec("ultra")
        grant = grant_for(spec)
        job = self.store.create(grant, "cancel-parallel", spec)
        started = Barrier(3)
        def hook(stage, control):
            if stage.startswith("specialist-"):
                started.wait(timeout=3)
                if stage == "specialist-1":
                    self.store.request_cancel(OWNER, job)
                while not control.cancelled():
                    sleep(0.001)
                control.check()
        fixture = ModelFixture(hook=hook)
        status = LocalRunner(self.store, lambda: grant, fixture.worker()).run(job)
        self.assertEqual(status["state"], "cancelled")
        self.assertEqual(set(fixture.calls), {"specialist-1", "specialist-2", "specialist-3"})
        self.assertEqual(status["cost_reserved_microusd"], 300)
        self.assertNotIn("synthesis", fixture.calls)

    def test_entitlement_revocation_stops_current_job_and_blocks_resume(self):
        spec = make_spec("max")
        current = [grant_for(spec)]
        def hook(stage, control):
            current[0] = replace(current[0], tier="plus")
            control.check()
        _, fixture, job, runner, status = self.run_job(spec, ModelFixture(hook=hook), authorization=lambda: current[0])
        self.assertEqual(status["state"], "failed")
        self.assertEqual(len(fixture.calls), 1)
        with self.assertRaises(ExecutionBlocked):
            runner.run(job)

    def test_failure_in_one_specialist_prevents_judge_and_synthesis(self):
        fixture = ModelFixture()
        fixture.override["specialist-1"] = "<think>must not escape</think>"
        _, fixture, job, _, status = self.run_job(make_spec("ultra"), fixture)
        self.assertEqual(status["state"], "failed")
        self.assertNotIn("disagreement-check", fixture.calls)
        self.assertNotIn("synthesis", fixture.calls)
        self.assertEqual(sorted(fixture.closed), sorted(fixture.calls))
        self.assertNotIn("must not escape", json.dumps(self.store.replay(OWNER, job)))

    def test_malformed_judge_and_fabricated_references_never_trigger_debate(self):
        for content in ("I think yes", '{"material_disagreement":"true","disagreement_ids":["d1"],"summary":"x"}',
                        '{"material_disagreement":true,"disagreement_ids":["d99"],"summary":"x"}'):
            fixture = ModelFixture(debate=True)
            fixture.override["judge"] = content
            with self.subTest(content=content):
                _, fixture, _, _, status = self.run_job(make_spec("ultra"), fixture)
                self.assertEqual(status["state"], "failed")
                self.assertNotIn("debate-round-1", fixture.calls)
                self.assertNotIn("synthesis", fixture.calls)
                self.assertEqual(fixture.records[-1]["outcome"], "quarantined")

    def test_progress_formats_only_actual_started_stages_and_preserves_instant_silence(self):
        for route in ("instant", "medium", "high", "ultra"):
            spec, _, job, _, status = self.run_job(make_spec(route))
            events = self.store.replay(OWNER, job)["events"]
            activity = [a for e in events if (a := activity_from_started_event(spec, e))]
            with self.subTest(route=route):
                self.assertEqual(status["state"], "succeeded")
                if route in ("instant", "medium"):
                    self.assertEqual(activity, [])
                else:
                    self.assertEqual(len(activity), status["completed_stages"])
                    self.assertTrue(all(a["grounding_operation_id"].startswith(job + ":") for a in activity))
                    self.assertTrue(all(a["title"] and a["summary"] for a in activity))
                self.assertNotIn("PRIVATE", json.dumps(events) + json.dumps(activity))
                self.assertFalse(any("source_url" in a or "icon_key" in a for a in activity))

    def test_tool_requests_are_not_executed_or_mistaken_for_a_completed_answer(self):
        calls = []
        def worker(spec, stage, artifacts, control, job, attempt):
            calls.append(stage.id)
            return {"content": "", "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query":"fixture"}'}}],
                    "input_tokens": 10, "output_tokens": 3, "debate_required": None}
        _, _, job, runner, status = self.run_job(make_spec("instant"), worker=worker)
        self.assertEqual(status["state"], "waiting_tools")
        self.assertEqual(calls, ["answer-1"])
        runner.run(job)
        self.assertEqual(calls, ["answer-1"])
        with self.assertRaises(ExecutionError):
            self.store.result(OWNER, job)

    def test_runtime_identity_change_quarantines_attempt_and_prevents_final_output(self):
        fixture = ModelFixture()
        original = fixture.probe
        def changing(stage, phase):
            probe = original(stage, phase)
            if phase == "after":
                probe["loaded_model_revision"] = "0" * 40
            return probe
        fixture.probe = changing
        for route in ("instant", "ultra"):
            with self.subTest(route=route):
                _, fixture, job, _, status = self.run_job(make_spec(route), fixture)
                self.assertEqual(status["state"], "failed")
                self.assertTrue(any(r["outcome"] == "quarantined" for r in fixture.records))
                with self.assertRaises(ExecutionError):
                    self.store.result(OWNER, job)

    def test_unknown_inflight_job_is_not_replayed_by_new_runner(self):
        spec = make_spec()
        grant = grant_for(spec)
        job = self.store.create(grant, "crash", spec)
        fence = self.store.begin(grant, job)
        self.store.claim(grant, job, *fence, spec.stages[0].id)
        self.store.recover_abandoned(OWNER, job, runner=fence[0], epoch=fence[1], supervisor_confirmed_stopped=True)
        fixture = ModelFixture()
        status = LocalRunner(self.store, lambda: grant, fixture.worker()).run(job)
        self.assertEqual(status["state"], "interrupted")
        self.assertEqual(fixture.calls, [])

    def test_bad_runtime_clock_fails_before_acquiring_runner(self):
        spec = make_spec()
        grant = grant_for(spec)
        job = self.store.create(grant, "clock", spec)
        fixture = ModelFixture()
        for clock in (lambda: float("nan"), lambda: True, lambda: "time"):
            with self.assertRaises(ExecutionError):
                LocalRunner(self.store, lambda: grant, fixture.worker(), clock=clock).run(job)
        self.assertEqual(fixture.calls, [])
        self.assertIsNotNone(self.store.begin(grant, job))


    def test_changed_core_planning_contract_is_rejected_before_any_provider_call(self):
        from core import adapter
        spec = make_spec("high")
        grant = grant_for(spec)
        job = self.store.create(grant, "stale-core-plan", spec)
        fixture = ModelFixture()
        with patch.dict(adapter.STAGE_INSTRUCTIONS, {"planning": "Changed source planning instructions"}):
            status = LocalRunner(self.store, lambda: grant, fixture.worker()).run(job)
        self.assertEqual(status["state"], "failed")
        self.assertEqual(fixture.calls, [])

    def test_invalid_wall_clock_never_acquires_a_runner_fence(self):
        spec = make_spec()
        grant = grant_for(spec)
        job = self.store.create(grant, "wall-clock-invalid", spec)
        fixture = ModelFixture()
        for clock in (lambda: None, lambda: float("nan"), lambda: True, lambda: "time"):
            with self.subTest(clock=clock), self.assertRaises(ExecutionError):
                LocalRunner(self.store, lambda: grant, fixture.worker(), clock_ms=clock).run(job)
            self.assertEqual(self.store.status(OWNER, job)["state"], "queued")
        self.assertEqual(fixture.calls, [])
        self.assertIsNotNone(self.store.begin(grant, job))

    def test_changed_ultra_planning_contract_is_rejected_before_client_construction(self):
        from ultra import orchestrator
        spec = make_spec("ultra")
        grant = grant_for(spec)
        job = self.store.create(grant, "stale-ultra-plan", spec)
        fixture = ModelFixture()
        with patch.dict(orchestrator.ROLE_INSTRUCTIONS, {"judge": "Changed judge instruction"}), \
                patch.object(fixture, "client_factory", side_effect=AssertionError("client constructed")) as factory:
            status = LocalRunner(self.store, lambda: grant, fixture.worker()).run(job)
            factory.assert_not_called()
        self.assertEqual(status["state"], "failed")
        self.assertEqual(fixture.calls, [])

    def test_resume_with_changed_core_policy_preserves_completed_artifacts_without_new_calls(self):
        from core import adapter
        spec, fixture, job, runner, status = self.run_job(make_spec("max"), max_stages=2)
        self.assertEqual(status["state"], "paused")
        before = list(fixture.calls)
        with patch.dict(adapter.STAGE_INSTRUCTIONS, {"answer": "Changed answer instruction"}):
            status = runner.run(job)
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["completed_stages"], 2)
        self.assertEqual(fixture.calls, before)
        with self.assertRaises(ExecutionError):
            self.store.result(OWNER, job)

    def test_wall_clock_corruption_after_start_releases_runner_and_withholds_result(self):
        wall = [time_ns() // 1_000_000]
        spec = make_spec("instant")
        grant = grant_for(spec)
        job = self.store.create(grant, "wall-clock-changes", spec)
        def worker(spec, stage, artifacts, control, logical_id, attempt):
            wall[0] = None
            control.check()
            raise AssertionError("invalid wall clock was accepted")
        status = LocalRunner(self.store, lambda: grant, worker, clock_ms=lambda: wall[0]).run(job)
        self.assertEqual(status["state"], "failed")
        row = self.store._db.execute("SELECT runner FROM jobs WHERE id=?", (job,)).fetchone()
        self.assertIsNone(row["runner"])
        with self.assertRaises(ExecutionError):
            self.store.result(OWNER, job)


if __name__ == "__main__":
    unittest.main()
