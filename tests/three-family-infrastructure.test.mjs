import assert from "node:assert/strict";
import test from "node:test";
import {compileAll, validateSource} from "../scripts/validate-three-family-infrastructure.mjs";

test("three-family paid infrastructure is source-valid and disabled", () => {
  assert.equal(validateSource().status, "three_family_bicep_source_contracts_valid");
  const result = compileAll();
  assert.equal(typeof result.compiler_validation_deferred_to_exact_head_ci, "boolean");
});
