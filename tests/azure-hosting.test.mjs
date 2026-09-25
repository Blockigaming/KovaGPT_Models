import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const root = fileURLToPath(new URL("..", import.meta.url));
const load = (name) => JSON.parse(readFileSync(`${root}/config/${name}`, "utf8"));

test("Azure Container Apps Consumption GPU is the blocked inference migration target", () => {
  const architecture = load("provider-architecture.v1.json");
  const azure = load("azure-container-apps-gpu.v1.json");

  assert.equal(architecture.schema_version, 3);
  assert.equal(architecture.status, "planning_only");
  assert.equal(architecture.application_plane.provider, "azure_container_apps");
  assert.equal(architecture.inference_decision.status, "legacy_source_only_pending_azure_migration");
  assert.equal(architecture.inference_decision.production_live, false);

  assert.equal(architecture.inference_migration_target.provider, "azure_container_apps_consumption_gpu");
  assert.equal(architecture.inference_migration_target.config_source, "config/azure-container-apps-gpu.v1.json");
  assert.equal(architecture.inference_migration_target.scale_to_zero_required, true);
  assert.equal(architecture.inference_migration_target.per_second_billing_required, true);
  assert.equal(architecture.inference_migration_target.migration_status, "cpu_adapter_verified_live_transport_blocked");
  assert.equal(architecture.inference_migration_target.paid_execution_authorized, false);
  assert.equal(architecture.inference_migration_target.deployment_authorized, false);
  assert.equal(architecture.inference_migration_target.production_routing_authorized, false);

  assert.equal(azure.provider, "azure_container_apps_consumption_gpu");
  assert.equal(azure.status, "planning_only");
  assert.equal(azure.workload_profile, "consumption_gpu");
  assert.equal(azure.scale_to_zero_required, true);
  assert.equal(azure.per_second_billing_required, true);
  assert.equal(azure.physical_app_count, 2);
  assert.deepEqual(azure.apps.map((app) => app.id), ["kova-core", "kova-ultra"]);
  assert.ok(azure.apps.every((app) =>
    app.deployed === false &&
    app.min_replicas === 0 &&
    app.gpu_profile_selected === null &&
    app.model_selected === false &&
    app.container_image_digest === null &&
    app.streaming_required === true
  ));
  assert.ok(azure.apps.every((app) =>
    app.gpu_candidates.includes("Consumption-GPU-NC8as-T4") &&
    app.gpu_candidates.includes("Consumption-GPU-NC24-A100")
  ));

  assert.equal(azure.migration.source_provider, "runpod_serverless");
  assert.equal(azure.migration.source_transport_status, "legacy_source_only_not_production_live");
  assert.equal(azure.migration.preserve_existing_source_until_azure_adapter_verified, true);
  assert.equal(azure.migration.production_cutover_authorized, false);

  assert.equal(azure.selection_guards.region, null);
  assert.equal(azure.selection_guards.quota_verified, false);
  assert.equal(azure.selection_guards.gpu_profile_selected, false);
  assert.equal(azure.selection_guards.container_compatibility_verified, false);
  assert.equal(azure.selection_guards.streaming_verified, false);
  assert.equal(azure.selection_guards.scale_to_zero_verified, false);
  assert.equal(azure.selection_guards.cold_start_measured, false);
  assert.equal(azure.selection_guards.token_throughput_measured, false);
  assert.equal(azure.selection_guards.cost_measured, false);

  assert.ok(Object.values(azure.safety).every((value) => value === false));
  assert.ok(architecture.engines.every((engine) =>
    engine.provider === "runpod_serverless" &&
    engine.migration_target_provider === "azure_container_apps_consumption_gpu" &&
    engine.endpoint_deployed === false &&
    engine.deployment_authorized === false &&
    engine.production_routing_authorized === false
  ));
});
