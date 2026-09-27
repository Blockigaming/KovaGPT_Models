import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const safetyKeys = [
  "runtime_execution_authorized", "paid_execution_authorized", "deployment_authorized",
  "production_routing_authorized", "private_customer_data_persistence_authorized",
];
const liveKeys = [
  "azure_job_durability_verified", "real_specialist_gpu_parallelism_verified",
  "browser_streaming_and_reconnect_verified", "actual_model_quality_evaluated",
  "actual_route_latency_measured", "actual_route_cost_measured",
];
const unselectedKeys = [
  "production_durable_store", "production_job_queue", "worker_supervisor",
  "per_mode_response_timing_targets", "per_mode_active_work_durations",
  "production_job_deadlines", "production_per_route_cost_caps", "ultra_worker_model",
];

export function validateExecutionKernel(value) {
  assert.deepEqual(Object.keys(value ?? {}).sort(), [
    "schema_version", "status", "implementation", "verified_source_contracts",
    "execution_rules", "unselected", "safety", "live_verification",
  ].sort());
  assert.equal(value.schema_version, 1);
  assert.equal(value.status, "source_only_reference_runtime");
  assert.deepEqual(value.implementation, {
    contracts: "execution/contracts.py",
    journal: "execution/store.py",
    scheduler: "execution/runner.py",
    workers: "execution/workers.py",
    ultra_binding: "ultra/binding.py",
    activity: "execution/activity.py",
    reference_storage: "local_sqlite_synthetic_data_only",
    background_service_installed: false,
  });
  assert.deepEqual(value.verified_source_contracts, {
    core_profiles: 20, core_stages: 113, ultra_profiles: 4,
    ultra_specialist_range: [2, 5], maximum_conditional_debate_rounds: 1,
    parallel_callbacks_exercised: true, safe_frontier_restart_exercised: true,
    owner_scoped_replay_exercised: true,
  });
  assert.deepEqual(value.execution_rules, {
    server_owned_current_entitlement_required: true,
    explicit_work_route_entitlement_required: true,
    whole_route_input_output_reservation_required: true,
    cost_reservation_is_measured_billing: false,
    original_deadline_survives_pause_and_restart: true,
    deadline_includes_queue_and_pause_time: true,
    unknown_inflight_retry_allowed: false,
    automatic_runner_takeover_allowed: false,
    private_artifacts_in_public_events_allowed: false,
    unstarted_operation_activity_allowed: false,
    model_tool_calls_are_executed: false,
    unhandled_tool_calls_state: "waiting_tools",
    cancellation_model: "cooperative_workers_and_bounded_transport",
    remote_gpu_abort_proven: false,
  });
  assert.deepEqual(value.unselected, Object.fromEntries(unselectedKeys.map((key) => [key, null])));
  assert.deepEqual(value.safety, Object.fromEntries(safetyKeys.map((key) => [key, false])));
  assert.deepEqual(value.live_verification, Object.fromEntries(liveKeys.map((key) => [key, false])));
  return { status: "execution_kernel_source_checks_passed_live_release_blocked", provider_calls_made: 0 };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const value = JSON.parse(readFileSync(new URL("../config/execution-kernel.v1.json", import.meta.url), "utf8"));
  console.log(JSON.stringify(validateExecutionKernel(value)));
}
