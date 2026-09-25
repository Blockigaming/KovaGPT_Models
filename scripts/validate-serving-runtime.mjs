import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

export function validateServingRuntime(value) {
  assert.deepEqual(value, {
    schema_version: 1, status: "source_only_disabled_loader",
    implementation: "worker/serving_runtime.py", supported_api_version: "vllm_0.29.0",
    selection: {
      startup_policy_path: null, startup_policy_sha256: null, candidate_id: null,
      container_image_digest: null, context_tokens: null, gpu_memory_utilization: null,
      maximum_sequences: null, load_timeout_seconds: null, health_timeout_seconds: null,
      shutdown_timeout_seconds: null,
    },
    safety: {
      enabled: false, model_loading_authorized: false, gpu_execution_authorized: false,
      public_listener_installed: false, production_routing_authorized: false,
      real_model_loaded_in_tests: false,
    },
  });
  return { status: "loader_source_config_valid", model_loading_enabled: false, phase_b_ready: false };
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const value = JSON.parse(readFileSync(new URL("../config/serving-runtime.v1.json", import.meta.url), "utf8"));
  console.log(JSON.stringify(validateServingRuntime(value)));
}
