import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { validateExecutionKernel } from "../scripts/validate-execution.mjs";

const load = () => JSON.parse(readFileSync(new URL("../config/execution-kernel.v1.json", import.meta.url), "utf8"));

test("execution kernel validator and CLI are provider-free", () => {
  assert.equal(validateExecutionKernel(load()).provider_calls_made, 0);
  const result = spawnSync(process.execPath, [fileURLToPath(new URL("../scripts/validate-execution.mjs", import.meta.url))], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(JSON.parse(result.stdout).provider_calls_made, 0);
});

test("all execution permissions and live claims remain false, including missing-field cases", () => {
  for (const section of ["safety", "live_verification"]) {
    for (const field of Object.keys(load()[section])) {
      for (const changed of [undefined, true, "false", null]) {
        const value = load();
        if (changed === undefined) delete value[section][field];
        else value[section][field] = changed;
        assert.throws(() => validateExecutionKernel(value), `${section}.${field}`);
      }
    }
  }
});

test("unresolved storage, model and timing choices cannot be silently selected", () => {
  for (const field of Object.keys(load().unselected)) {
    for (const changed of [undefined, "guessed", 600, false]) {
      const value = load();
      if (changed === undefined) delete value.unselected[field];
      else value.unselected[field] = changed;
      assert.throws(() => validateExecutionKernel(value), field);
    }
  }
});

test("runtime safety semantics and source/live distinction cannot be weakened", () => {
  for (const [field, current] of Object.entries(load().execution_rules)) {
    for (const changed of [undefined, typeof current === "boolean" ? !current : "unsafe"] ) {
      const value = load();
      if (changed === undefined) delete value.execution_rules[field];
      else value.execution_rules[field] = changed;
      assert.throws(() => validateExecutionKernel(value), field);
    }
  }
});

test("no vacuous empty safety object, undeclared action, or background service passes", () => {
  for (const changed of [{}, undefined, { unauthorized_extra_action: true }]) {
    const value = load();
    value.safety = changed;
    assert.throws(() => validateExecutionKernel(value));
  }
  const value = load();
  value.implementation.background_service_installed = true;
  assert.throws(() => validateExecutionKernel(value));
});
