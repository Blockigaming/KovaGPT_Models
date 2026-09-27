import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { validateServingRuntime } from "../scripts/validate-serving-runtime.mjs";
const load = () => JSON.parse(readFileSync(new URL("../config/serving-runtime.v1.json", import.meta.url), "utf8"));

test("serving source CLI checks disabled configuration without importing a model", () => {
  assert.equal(validateServingRuntime(load()).model_loading_enabled, false);
  const result = spawnSync(process.execPath, [fileURLToPath(new URL("../scripts/validate-serving-runtime.mjs", import.meta.url))], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(JSON.parse(result.stdout).phase_b_ready, false);
});
test("model loading permissions and readiness claims cannot silently change", () => {
  for (const field of Object.keys(load().safety)) {
    for (const bad of [true, "false", undefined]) {
      const value = load(); value.safety[field] = bad;
      assert.throws(() => validateServingRuntime(value), field);
    }
  }
});
test("unverified models images paths contexts and budgets remain unselected", () => {
  for (const field of Object.keys(load().selection)) {
    const value = load(); value.selection[field] = "unapproved";
    assert.throws(() => validateServingRuntime(value), field);
  }
  const value = load(); value.supported_api_version = "latest";
  assert.throws(() => validateServingRuntime(value));
});
