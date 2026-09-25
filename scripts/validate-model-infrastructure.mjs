import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

export function compileInfrastructure(binary = process.env.KOVA_BICEP) {
  assert.ok(typeof binary === "string" && binary.startsWith("/"), "hash-pinned Bicep executable required");
  const lock = JSON.parse(readFileSync(new URL("../infra/compiler.lock.json", import.meta.url), "utf8"));
  assert.equal(createHash("sha256").update(readFileSync(binary)).digest("hex"), lock.sha256);
  const result = spawnSync(binary, ["build", "--no-restore", "--stdout",
    fileURLToPath(new URL("../infra/model-staging.bicep", import.meta.url))], {
    encoding: "utf8", timeout: 30000, maxBuffer: 4 * 1024 * 1024,
    env: { PATH: process.env.PATH, HOME: process.env.HOME, DOTNET_CLI_TELEMETRY_OPTOUT: "1", BICEP_TRACING_ENABLED: "false" },
  });
  assert.equal(result.status, 0, result.stderr || String(result.error));
  assert.doesNotMatch(result.stderr, /BCP081|BCP037|BCP036/, "resource schema must be typechecked");
  return JSON.parse(result.stdout);
}

export function validateInfrastructure(arm) {
  assert.equal(arm.parameters.provisionStaging.defaultValue, false);
  assert.equal(arm.parameters.activateModelApps.defaultValue, false);
  assert.equal(arm.parameters.maxReplicas.minValue, 1);
  assert.equal(arm.parameters.maxReplicas.maxValue, 2);
  for (const key of ["location", "infrastructureSubnetId", "imagePullIdentityId", "registryServer", "coreImage", "ultraImage", "gpuProfileType", "stagingSuffix", "maxReplicas"]) {
    assert.ok(!Object.hasOwn(arm.parameters[key], "defaultValue"), `${key} must not be guessed`);
  }
  const resources = Array.isArray(arm.resources) ? arm.resources : Object.values(arm.resources);
  assert.equal(resources.length, 2);
  const environment = resources.find((r) => r.type === "Microsoft.App/managedEnvironments");
  const app = resources.find((r) => r.type === "Microsoft.App/containerApps");
  assert.ok(environment && app);
  assert.equal(environment.apiVersion, "2025-07-01");
  assert.equal(app.apiVersion, "2025-07-01");
  assert.equal(environment.condition, "[parameters('provisionStaging')]");
  assert.equal(app.condition, "[and(parameters('provisionStaging'), parameters('activateModelApps'))]");
  assert.match(environment.name, /kova-model-staging-/);
  assert.match(app.name, /staging-/);
  assert.equal(environment.properties.publicNetworkAccess, "Disabled");
  assert.equal(environment.properties.vnetConfiguration.internal, true);
  assert.equal(environment.properties.vnetConfiguration.infrastructureSubnetId, "[parameters('infrastructureSubnetId')]");
  assert.equal(environment.properties.peerTrafficConfiguration.encryption.enabled, true);
  assert.equal(environment.properties.appLogsConfiguration.destination, "none");
  assert.equal(environment.properties.workloadProfiles[1].workloadProfileType, "[parameters('gpuProfileType')]");
  assert.equal(app.identity.type, "UserAssigned");
  assert.equal(app.properties.workloadProfileName, "model-gpu");
  assert.deepEqual(app.properties.configuration.registries, [{
    server: "[parameters('registryServer')]", identity: "[parameters('imagePullIdentityId')]",
  }]);
  assert.equal(app.properties.configuration.activeRevisionsMode, "Single");
  const ingress = app.properties.configuration.ingress;
  assert.equal(ingress.external, false);
  assert.equal(ingress.allowInsecure, false);
  assert.equal(ingress.targetPort, 8080);
  const template = app.properties.template;
  assert.equal(template.scale.minReplicas, 0);
  assert.equal(template.scale.maxReplicas, "[parameters('maxReplicas')]");
  assert.equal(template.containers.length, 1);
  const container = template.containers[0];
  assert.match(container.image, /parameters\('coreImage'\)/);
  assert.match(container.image, /parameters\('ultraImage'\)/);
  assert.deepEqual(Object.fromEntries(container.env.map((e) => [e.name, e.value])), {
    KOVA_MODEL_EXECUTION_ENABLED: "false", HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1", DO_NOT_TRACK: "1",
  });
  assert.equal(container.probes.find((p) => p.type === "Readiness").httpGet.path, "/readyz");
  assert.equal(container.probes.find((p) => p.type === "Liveness").httpGet.path, "/healthz");
  assert.equal(arm.outputs.phaseBReady.value, false);
  assert.equal(arm.outputs.modelExecutionEnabled.value, false);
  assert.doesNotMatch(JSON.stringify(arm), /roleAssignments|deploymentScripts|listKeys\(|passwordSecretRef|clientSecret/);
  return { status: "compiled_staging_source_valid", azure_requests_made: 0, phase_b_ready: false };
}

export function validateStagingParameters(value) {
  const keys = ["provisionStaging", "activateModelApps", "stagingSuffix", "location", "infrastructureSubnetId",
    "imagePullIdentityId", "registryServer", "coreImage", "ultraImage", "gpuProfileType", "maxReplicas"];
  assert.deepEqual(Object.keys(value).sort(), keys.sort());
  assert.equal(value.provisionStaging, false, "source-only preparation cannot authorize deployment");
  assert.equal(value.activateModelApps, false, "model apps remain disabled");
  assert.match(value.stagingSuffix, /^[a-z][a-z0-9-]{2,11}$/);
  assert.match(value.location, /^[a-z][a-z0-9]{1,31}$/);
  assert.match(value.registryServer, /^[a-z0-9]{5,50}\.azurecr\.io$/);
  const root = "/subscriptions/[a-f0-9-]{36}/resourceGroups/[A-Za-z0-9_.()-]{1,90}/providers/";
  assert.match(value.infrastructureSubnetId, new RegExp("^" + root + "Microsoft.Network/virtualNetworks/[A-Za-z0-9_.-]+/subnets/[A-Za-z0-9_.-]+$"));
  assert.match(value.imagePullIdentityId, new RegExp("^" + root + "Microsoft.ManagedIdentity/userAssignedIdentities/[A-Za-z0-9_.-]+$"));
  for (const name of ["coreImage", "ultraImage"]) {
    assert.ok(typeof value[name] === "string" && value[name].startsWith(value.registryServer + "/"));
    assert.match(value[name], /^[a-z0-9.]+\/[a-z0-9/-]+@sha256:[a-f0-9]{64}$/);
  }
  assert.ok(["Consumption-GPU-NC24-A100", "Consumption-GPU-NC8as-T4"].includes(value.gpuProfileType));
  assert.ok(Number.isSafeInteger(value.maxReplicas) && value.maxReplicas >= 1 && value.maxReplicas <= 2);
  return { resource_creation_authorized: false, model_fit_verified: false };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  console.log(JSON.stringify(validateInfrastructure(compileInfrastructure())));
}
