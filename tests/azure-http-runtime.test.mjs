import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { validateAzureHTTPRuntime } from "../scripts/validate-azure-http-runtime.mjs";

const load = () => JSON.parse(readFileSync(new URL("../config/azure-http-runtime.v1.json", import.meta.url), "utf8"));

test("concrete HTTP runtime CLI and contract pass without live execution", () => {
  assert.equal(validateAzureHTTPRuntime(load()).network_calls_made, 0);
  const result = spawnSync(process.execPath, [fileURLToPath(new URL("../scripts/validate-azure-http-runtime.mjs", import.meta.url))], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(JSON.parse(result.stdout).network_calls_made, 0);
});

test("live authorization or readiness cannot be removed or promoted by source edits", () => {
  for (const section of ["safety", "verification"]) {
    for (const [key, current] of Object.entries(load()[section])) {
      for (const changed of [undefined, !current, null, "false"]) {
        const config = load();
        if (changed === undefined) delete config[section][key];
        else config[section][key] = changed;
        assert.throws(() => validateAzureHTTPRuntime(config), `${section}.${key}`);
      }
    }
  }
});

test("unapproved address audience identity role and timing selections remain unset", () => {
  for (const key of Object.keys(load().selection)) {
    for (const changed of [undefined, "guessed-selection", false, 30]) {
      const config = load();
      if (changed === undefined) delete config.selection[key];
      else config.selection[key] = changed;
      assert.throws(() => validateAzureHTTPRuntime(config), key);
    }
  }
});

test("security and timeout contract cannot be weakened", () => {
  for (const [key, current] of Object.entries(load().transport_contract)) {
    for (const changed of [undefined, !current]) {
      const config = load();
      if (changed === undefined) delete config.transport_contract[key];
      else config.transport_contract[key] = changed;
      assert.throws(() => validateAzureHTTPRuntime(config), key);
    }
  }
});

test("empty and unknown safety fields cannot create vacuous verification", () => {
  for (const changed of [{}, { unintended_network_authorization: true }, undefined]) {
    const config = load();
    config.safety = changed;
    assert.throws(() => validateAzureHTTPRuntime(config));
  }
  const config = load();
  config.enable_paid_execution = true;
  assert.throws(() => validateAzureHTTPRuntime(config));
});
