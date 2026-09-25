import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const SAFETY = [
  "paid_execution_authorized", "image_build_authorized", "deployment_authorized",
  "endpoint_creation_authorized", "production_routing_authorized",
];
const READINESS = [
  "quota_verified", "gpu_profile_selected", "container_compatibility_verified",
  "private_auth_verified", "streaming_verified", "scale_to_zero_verified",
  "cold_start_measured", "token_throughput_measured", "cost_measured",
];

export function validateAzureConfig(azure, architecture) {
  assert.equal(azure?.schema_version, 1, "Azure contract schema");
  assert.equal(azure.status, "planning_only", "Azure status remains blocked");
  assert.equal(azure.provider, "azure_container_apps_consumption_gpu");
  assert.equal(azure.workload_profile, "consumption_gpu");
  assert.equal(azure.scale_to_zero_required, true);
  assert.equal(azure.per_second_billing_required, true);
  assert.equal(azure.physical_app_count, 2);
  assert.deepEqual(azure.apps?.map((app) => app.id), ["kova-core", "kova-ultra"]);
  for (const app of azure.apps) {
    assert.equal(app.container_app_name_reserved, app.id);
    assert.equal(app.deployed, false);
    assert.equal(app.min_replicas, 0);
    assert.equal(app.max_replicas, null, "replica budget not approved");
    assert.equal(app.gpu_profile_selected, null);
    assert.equal(app.model_selected, false);
    assert.equal(app.container_image_digest, null);
    for (const field of ["streaming_required", "cold_start_benchmark_required", "throughput_benchmark_required"]) {
      assert.equal(app[field], true, `${app.id}:${field}`);
    }
  }
  assert.equal(azure.apps[1].multi_agent_capacity_benchmark_required, true);
  assert.deepEqual(Object.keys(azure.safety ?? {}).sort(), [...SAFETY].sort());
  for (const flag of SAFETY) assert.equal(azure.safety[flag], false, flag);
  assert.deepEqual(Object.keys(azure.selection_guards ?? {}).sort(), ["region", ...READINESS].sort());
  assert.equal(azure.selection_guards.region, null);
  for (const flag of READINESS) assert.equal(azure.selection_guards[flag], false, flag);
  assert.equal(azure.migration?.source_provider, "runpod_serverless");
  assert.equal(azure.migration.source_transport_status, "legacy_source_only_not_production_live");
  assert.equal(azure.migration.preserve_existing_source_until_azure_adapter_verified, true);
  assert.equal(azure.migration.delete_existing_azure_deployments, false);
  assert.equal(azure.migration.production_cutover_authorized, false);

  const transport = azure.transport;
  assert.equal(transport?.adapter, "worker/azure_container_apps.py");
  assert.equal(transport.shared_protocol, "worker/openai_protocol.py");
  assert.equal(transport.route, "/v1/chat/completions");
  assert.equal(transport.request_shape, "direct_openai_compatible_json_no_queue_envelope");
  assert.equal(transport.credential_source, "trusted_server_provider_only");
  assert.equal(transport.injected_http_boundary_implemented, true);
  assert.equal(transport.cpu_fixture_contract_verified, true);
  for (const flag of ["network_client_bound", "live_transport_verified", "client_transport_overrides_allowed", "redirects_allowed", "automatic_retries_allowed"]) {
    assert.equal(transport[flag], false, flag);
  }
  assert.equal(transport.tls_verification_required, true);
  assert.equal(transport.close_on_completion_failure_and_cancellation, true);
  assert.equal(transport.sdk_authentication_integration, null);
  assert.equal(transport.configured_origin, null);

  const timing = azure.execution_timing;
  assert.equal(timing?.documented_http_ingress_limit_seconds, 240);
  assert.equal(timing.selected_http_hop_timeout_seconds, null, "no unapproved timeout default");
  assert.equal(timing.selected_total_model_work_budget_seconds, null, "mode work budgets remain unresolved");
  assert.equal(timing.transport_io_timeout_required, true);
  assert.equal(timing.streaming_bypasses_ingress_limit_claimed, false);
  assert.equal(timing.long_work_transport_implemented, false);
  assert.equal(timing.deadline_checks, "cooperative_plus_required_transport_io_timeout");

  const target = architecture?.inference_migration_target;
  assert.equal(target?.provider, azure.provider);
  assert.equal(target.config_source, "config/azure-container-apps-gpu.v1.json");
  assert.equal(target.migration_status, "cpu_adapter_verified_live_transport_blocked");
  for (const flag of ["paid_execution_authorized", "deployment_authorized", "production_routing_authorized"]) {
    assert.equal(target[flag], false, flag);
  }
  assert.equal(architecture.application_plane?.delete_existing_azure_deployments, false);
  assert.equal(architecture.application_plane.production_routing_authorized, false);
  assert.equal(architecture.global_guards?.paid_or_production_actions_authorized, false);
  return { status: "azure_cpu_contract_passed_live_execution_blocked", paid_actions_started: false };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const load = (name) => JSON.parse(readFileSync(new URL(`../config/${name}`, import.meta.url), "utf8"));
  console.log(JSON.stringify(validateAzureConfig(
    load("azure-container-apps-gpu.v1.json"), load("provider-architecture.v1.json"),
  )));
}
