import assert from "node:assert/strict";
import test from "node:test";
import { compileInfrastructure, validateInfrastructure, validateStagingParameters } from "../scripts/validate-model-infrastructure.mjs";

const compiled = process.env.KOVA_BICEP ? compileInfrastructure() : null;
const resources = (arm) => Array.isArray(arm.resources) ? arm.resources : Object.values(arm.resources);
const app = (arm) => resources(arm).find((r) => r.type === "Microsoft.App/containerApps");
const environment = (arm) => resources(arm).find((r) => r.type === "Microsoft.App/managedEnvironments");
const prefix = "/subscriptions/10000000-0000-4000-8000-000000000001/resourceGroups/fixture-only/providers/";
const params = () => ({ provisionStaging: false, activateModelApps: false, stagingSuffix: "fixture", location: "eastus",
  infrastructureSubnetId: prefix + "Microsoft.Network/virtualNetworks/fixture/subnets/models",
  imagePullIdentityId: prefix + "Microsoft.ManagedIdentity/userAssignedIdentities/fixture",
  registryServer: "fixtureonly.azurecr.io", coreImage: "fixtureonly.azurecr.io/core@sha256:" + "a".repeat(64),
  ultraImage: "fixtureonly.azurecr.io/ultra@sha256:" + "b".repeat(64),
  gpuProfileType: "Consumption-GPU-NC24-A100", maxReplicas: 1 });

test("the real pinned Bicep compiler validates the staging-only resource source", {skip: !compiled}, () => {
  assert.equal(validateInfrastructure(compiled).phase_b_ready, false);
  assert.equal(validateInfrastructure(compiled).azure_requests_made, 0);
});
test("default deployment and model-app guards cannot be promoted or removed", {skip: !compiled}, () => {
  for (const key of ["provisionStaging", "activateModelApps"]) {
    for (const bad of [true, undefined, "false"]) {
      const changed = structuredClone(compiled); changed.parameters[key].defaultValue = bad;
      assert.throws(() => validateInfrastructure(changed));
    }
  }
});
test("every deployed resource stays behind its explicit condition", {skip: !compiled}, () => {
  for (const index of [0, 1]) {
    const changed = structuredClone(compiled); delete resources(changed)[index].condition;
    assert.throws(() => validateInfrastructure(changed));
  }
});
test("public environment or insecure/external inference ingress is rejected", {skip: !compiled}, () => {
  for (const mutate of [
    (a) => environment(a).properties.publicNetworkAccess = "Enabled",
    (a) => environment(a).properties.vnetConfiguration.internal = false,
    (a) => app(a).properties.configuration.ingress.external = true,
    (a) => app(a).properties.configuration.ingress.allowInsecure = true,
  ]) {
    const changed = structuredClone(compiled); mutate(changed);
    assert.throws(() => validateInfrastructure(changed));
  }
});
test("scale-to-zero and maximum replica controls cannot become always-on or unbounded", {skip: !compiled}, () => {
  for (const mutate of [
    (a) => app(a).properties.template.scale.minReplicas = 1,
    (a) => app(a).properties.template.scale.maxReplicas = 100,
    (a) => a.parameters.maxReplicas.maxValue = 100,
  ]) {
    const changed = structuredClone(compiled); mutate(changed);
    assert.throws(() => validateInfrastructure(changed));
  }
});
test("model activation and runtime downloads remain disabled inside app definitions", {skip: !compiled}, () => {
  for (const key of ["KOVA_MODEL_EXECUTION_ENABLED", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]) {
    const changed = structuredClone(compiled);
    app(changed).properties.template.containers[0].env.find((e) => e.name === key).value = "unsafe";
    assert.throws(() => validateInfrastructure(changed));
  }
});
test("additional role assignments and deployment scripts cannot be hidden in the template", {skip: !compiled}, () => {
  const changed = structuredClone(compiled);
  if (Array.isArray(changed.resources)) changed.resources.push({type:"Microsoft.Authorization/roleAssignments"});
  else changed.resources.unexpected = {type:"Microsoft.Authorization/roleAssignments"};
  assert.throws(() => validateInfrastructure(changed));
});
test("parameter validation accepts only disabled pinned-image preparation", () => {
  assert.deepEqual(validateStagingParameters(params()), {resource_creation_authorized:false, model_fit_verified:false});
});
test("preparation does not accept approval flags, floating image tags or another registry", () => {
  for (const changes of [{provisionStaging:true}, {activateModelApps:true}, {maxReplicas:99}, {maxReplicas:true},
    {coreImage:"fixtureonly.azurecr.io/core:latest"}, {ultraImage:"evil.azurecr.io/ultra@sha256:"+"a".repeat(64)},
    {gpuProfileType:"Flexible"}, {unexpectedCredential:"secret"}]) {
    assert.throws(() => validateStagingParameters({...params(), ...changes}));
  }
});
test("invalid subnet identity location and names are not coerced into deployment parameters", () => {
  for (const changes of [{infrastructureSubnetId:"https://evil.invalid"}, {imagePullIdentityId:"owner-supplied-string"},
    {stagingSuffix:"production/../"}, {location:"eastus; command"}, {registryServer:"user:password@registry"}]) {
    assert.throws(() => validateStagingParameters({...params(), ...changes}));
  }
});
