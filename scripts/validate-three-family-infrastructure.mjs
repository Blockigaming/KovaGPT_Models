import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const files = ["three-family-pilot-vm.bicep", "three-family-watchdog.bicep"];

export function validateSource() {
  const vm = readFileSync(new URL("../infra/three-family-pilot-vm.bicep", import.meta.url), "utf8");
  const watchdog = readFileSync(new URL("../infra/three-family-watchdog.bicep", import.meta.url), "utf8");
  const watchdogRole = readFileSync(new URL("../infra/three-family-watchdog-pilot-role.bicep", import.meta.url), "utf8");
  assert.match(vm, /param provisionPilot bool = false/);
  assert.match(vm, /Standard_NC4as_T4_v3/);
  assert.match(vm, /disablePasswordAuthentication: true/);
  assert.match(vm, /publisher: 'Microsoft\.HpcCompute'/);
  assert.match(vm, /type: 'NvidiaGpuDriverLinux'/);
  assert.match(vm, /typeHandlerVersion: '1\.10'/);
  assert.match(vm, /enableAutomaticUpgrade: false/);
  assert.doesNotMatch(vm, /publicIPAddress\s*:/);
  assert.match(vm, /resource egressNat 'Microsoft\.Network\/natGateways@2024-05-01'/);
  assert.match(vm, /natGateway: \{ id: egressNat!\.id \}/);
  assert.match(vm, /defaultOutboundAccess: false/);
  assert.match(vm, /name: 'deny-all-inbound'/);
  assert.match(vm, /name: 'deny-other-egress'/);
  assert.match(watchdog, /param provisionWatchdog bool = false/);
  assert.match(watchdog, /frequency: 'Minute'/);
  assert.match(watchdog, /interval: 1/);
  assert.match(watchdog, /param deadlineUtc string/);
  assert.match(watchdog, /param pilotSuffix string/);
  assert.match(watchdog, /virtualMachines\/kova-t4-\$\{pilotSuffix\}\/deallocate/);
  assert.match(watchdog, /greaterOrEquals\(ticks\(utcNow\(\)\), ticks\(parameters/);
  assert.doesNotMatch(watchdog, /Microsoft\.Storage\/storageAccounts|controllerLedgerWriter/);
  assert.match(watchdog, /output externalAppendOnlyLedgerRequired bool = true/);
  assert.match(watchdog, /deallocate_after_deadline/);
  assert.match(watchdog, /delete_pilot_group/);
  assert.match(watchdog, /delete_watchdog_group/);
  assert.match(watchdog, /watchdogSelfCleanup/);
  assert.match(watchdog, /watchdogPilotContributor/);
  assert.match(watchdogRole, /watchdogPrincipalId/);
  assert.match(watchdogRole, /b24988ac-6180-42a0-ab88-20f7382dd24c/);
  return {status: "three_family_bicep_source_contracts_valid"};
}

export function compileAll(binary = process.env.KOVA_BICEP) {
  if (!binary) return {compiler_validation_deferred_to_exact_head_ci: true};
  assert.ok(binary.startsWith("/"), "absolute hash-pinned compiler required");
  const lock = JSON.parse(readFileSync(new URL("../infra/compiler.lock.json", import.meta.url), "utf8"));
  assert.equal(createHash("sha256").update(readFileSync(binary)).digest("hex"), lock.sha256);
  for (const file of files) {
    const result = spawnSync(binary, ["build", "--no-restore", "--stdout",
      fileURLToPath(new URL(`../infra/${file}`, import.meta.url))], {
      encoding: "utf8", timeout: 30000, maxBuffer: 8 * 1024 * 1024,
      env: {PATH: process.env.PATH, HOME: process.env.HOME,
            DOTNET_CLI_TELEMETRY_OPTOUT: "1", BICEP_TRACING_ENABLED: "false"},
    });
    assert.equal(result.status, 0, result.stderr || String(result.error));
    assert.doesNotMatch(result.stderr, /BCP081|BCP037|BCP036/);
    const arm = JSON.parse(result.stdout);
    assert.equal(arm.outputs.resourceCreationAuthorized.value, false);
    if (file.includes("watchdog")) assert.equal(arm.outputs.spendingAuthorized.value, false);
    else assert.equal(arm.outputs.deploymentAuthorized.value, false);
  }
  return {compiler_validation_deferred_to_exact_head_ci: false};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  console.log(JSON.stringify({source: validateSource(), ...compileAll(), azure_requests_made: 0,
    resource_creation_authorized: false}));
}
