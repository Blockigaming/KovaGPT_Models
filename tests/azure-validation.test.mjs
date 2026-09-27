import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { validateAzureConfig } from "../scripts/validate-azure.mjs";

const load = (name) => JSON.parse(readFileSync(new URL(`../config/${name}`, import.meta.url), "utf8"));
const fixture = () => [load("azure-container-apps-gpu.v1.json"), load("provider-architecture.v1.json")];

test("Azure validator and CLI pass without live execution", () => {
  const [azure, architecture] = fixture();
  assert.equal(validateAzureConfig(azure, architecture).paid_actions_started, false);
  const result = spawnSync(process.execPath, [new URL("../scripts/validate-azure.mjs", import.meta.url).pathname], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(JSON.parse(result.stdout).paid_actions_started, false);
});

test("every paid authorization and runtime verification field rejects removal or promotion", () => {
  for (const section of ["safety", "selection_guards"]) {
    for (const field of Object.keys(fixture()[0][section]).filter((key) => key !== "region")) {
      for (const mutation of [undefined, true, "false", null]) {
        const [azure, architecture] = fixture();
        if (mutation === undefined) delete azure[section][field];
        else azure[section][field] = mutation;
        assert.throws(() => validateAzureConfig(azure, architecture), `${section}.${field}`);
      }
    }
  }
});

test("vacuous safety objects and new authorization keys cannot pass", () => {
  for (const value of [{}, undefined, { unexpected_authorization: false }]) {
    const [azure, architecture] = fixture();
    azure.safety = value;
    assert.throws(() => validateAzureConfig(azure, architecture));
  }
});

test("both apps remain unselected and scale-to-zero", () => {
  for (let index = 0; index < 2; index += 1) {
    for (const [field, badValue] of [
      ["deployed", true], ["min_replicas", 1], ["max_replicas", 5], ["gpu_profile_selected", "T4"],
      ["model_selected", true], ["container_image_digest", "sha256:unverified"],
    ]) {
      const [azure, architecture] = fixture();
      azure.apps[index][field] = badValue;
      assert.throws(() => validateAzureConfig(azure, architecture), field);
    }
  }
});

test("transport cannot claim networking, auth, retries or redirect readiness", () => {
  for (const [field, value] of [
    ["network_client_bound", true], ["live_transport_verified", true],
    ["redirects_allowed", true], ["automatic_retries_allowed", true],
    ["client_transport_overrides_allowed", true], ["tls_verification_required", false],
    ["credential_source", "client_body"], ["configured_origin", "https://unverified.example"],
    ["sdk_authentication_integration", "unverified"],
  ]) {
    const [azure, architecture] = fixture();
    azure.transport[field] = value;
    assert.throws(() => validateAzureConfig(azure, architecture), field);
  }
});

test("infrastructure timing never becomes a guessed model work budget", () => {
  for (const [field, value] of [
    ["documented_http_ingress_limit_seconds", 600], ["selected_http_hop_timeout_seconds", 30],
    ["selected_total_model_work_budget_seconds", 600], ["streaming_bypasses_ingress_limit_claimed", true],
    ["long_work_transport_implemented", true], ["transport_io_timeout_required", false],
  ]) {
    const [azure, architecture] = fixture();
    azure.execution_timing[field] = value;
    assert.throws(() => validateAzureConfig(azure, architecture), field);
  }
});

test("architecture cannot enable paid routing or cutover", () => {
  for (const field of ["paid_execution_authorized", "deployment_authorized", "production_routing_authorized"]) {
    const [azure, architecture] = fixture();
    architecture.inference_migration_target[field] = true;
    assert.throws(() => validateAzureConfig(azure, architecture), field);
  }
  const [azure, architecture] = fixture();
  azure.migration.production_cutover_authorized = true;
  assert.throws(() => validateAzureConfig(azure, architecture));
});
