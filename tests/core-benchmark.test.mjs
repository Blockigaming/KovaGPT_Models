import assert from "node:assert/strict";
import { test } from "node:test";
import { coreConfigurationKey, summarizeCoreBenchmark } from "../scripts/summarize-core-benchmark.mjs";

const revision = "c1899de289a04d12100db370d81485cdf75e47ca";
const fp8Revision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e";
const digest = `sha256:${"a".repeat(64)}`;
const logicalId = (value) => `kova-exec-00000000-0000-4000-8000-${String(value).padStart(12, "0")}`;
const common = {
  worker_lifecycle_id: "lifecycle-1",
  model: "kova-cosmo",
  model_revision: revision,
  gpu_type_id: "NVIDIA A100 80GB PCIe",
  gpu_count: 1,
  serving_engine: "vllm",
  endpoint_type: "queue_based",
  container_image_digest: digest,
  gpu_rate_per_second_usd: 0.001,
  measurement_source: "server_provider_runtime",
};
const attempt = (overrides = {}) => ({
  ...common,
  record_type: "attempt",
  request_id: logicalId(1),
  correlation_id: "client-request-1",
  attempt_id: "attempt-1",
  outcome: "success",
  route_id: "instant",
  stage_id: "answer-1",
  public_response: true,
  cold_start: true,
  reasoning_effort: "low",
  worker_start_ms: 1000,
  model_load_ms: 1000,
  queue_ms: 100,
  inference_ms: 2000,
  time_to_first_token_ms: 500,
  input_tokens: 1000,
  output_tokens: 500,
  ...overrides,
});
const close = (overrides = {}) => ({
  ...common,
  record_type: "lifecycle_close",
  close_event_id: "close-1",
  billed_lifecycle_ms: 9000,
  attributed_idle_timeout_ms: 5000,
  ...overrides,
});
const warmAttempt = (overrides = {}) => attempt({
  cold_start: false,
  worker_start_ms: 0,
  model_load_ms: 0,
  ...overrides,
});
const configuration = (overrides = {}) => coreConfigurationKey({...common, ...overrides});
const group = (result, route = "instant", config = configuration()) =>
  result.by_configuration[config].by_route[route];

test("Core benchmark prices one complete RunPod lifecycle", () => {
  const result = summarizeCoreBenchmark([attempt(), close()]);
  const instant = group(result);
  assert.equal(result.schema_version, 6);
  assert.equal(result.worker_lifecycles, 1);
  assert.equal(instant.successful_requests, 1);
  assert.ok(Math.abs(instant.total_attributable_compute_cost_usd - 0.009) < 1e-12);
  assert.ok(Math.abs(instant.average_compute_cost_per_successful_request_usd - 0.009) < 1e-12);
  assert.ok(Math.abs(instant.startup_share_of_compute_percent - 2 / 9 * 100) < 1e-9);
  assert.ok(Math.abs(instant.active_share_of_compute_percent - 2 / 9 * 100) < 1e-9);
  assert.ok(Math.abs(instant.idle_share_of_compute_percent - 5 / 9 * 100) < 1e-9);
});

test("Core benchmark separates model, GPU, server, endpoint type, and image", () => {
  const fp8Config = {
    worker_lifecycle_id: "lifecycle-2",
    model: "kova-cosmo",
    model_revision: revision,
    gpu_type_id: "NVIDIA L40S",
    serving_engine: "sglang",
    endpoint_type: "load_balancing",
    container_image_digest: `sha256:${"b".repeat(64)}`,
  };
  const result = summarizeCoreBenchmark([
    attempt(), close(),
    attempt({...fp8Config, request_id: logicalId(2), attempt_id: "attempt-2"}),
    close({...fp8Config, close_event_id: "close-2"}),
  ]);
  assert.equal(Object.keys(result.by_configuration).length, 2);
  assert.equal(group(result).attempts, 1);
  assert.equal(group(result, "instant", configuration(fp8Config)).attempts, 1);
});

test("Core benchmark attributes failed retries to a successful request", () => {
  const result = summarizeCoreBenchmark([
    attempt({attempt_id: "failed", outcome: "failed", inference_ms: 1000, output_tokens: 0, time_to_first_token_ms: 0}),
    warmAttempt({attempt_id: "success", inference_ms: 1000}),
    close(),
  ]);
  const instant = group(result);
  assert.equal(instant.successful_requests, 1);
  assert.equal(instant.failed_attempts, 1);
  assert.ok(Math.abs(instant.average_compute_cost_per_successful_request_usd - 0.009) < 1e-12);
});

test("Core request is not successful when only a private stage succeeded", () => {
  const result = summarizeCoreBenchmark([
    attempt({route_id: "chat:cosmo:medium", stage_id: "planning-1", public_response: false, reasoning_effort: "medium", time_to_first_token_ms: null}),
    close(),
  ]);
  const medium = group(result, "chat:cosmo:medium");
  assert.equal(medium.successful_requests, 0);
  assert.equal(medium.average_compute_cost_per_successful_request_usd, null);
});

test("Core request succeeds only after every declared route stage succeeds", () => {
  const result = summarizeCoreBenchmark([
    attempt({attempt_id: "planning", route_id: "chat:cosmo:medium", stage_id: "planning-1", public_response: false, reasoning_effort: "medium", time_to_first_token_ms: null}),
    warmAttempt({attempt_id: "answer", route_id: "chat:cosmo:medium", stage_id: "answer-1", public_response: false, reasoning_effort: "medium", time_to_first_token_ms: null}),
    warmAttempt({attempt_id: "verification", route_id: "chat:cosmo:medium", stage_id: "verification-1", reasoning_effort: "medium"}),
    close({billed_lifecycle_ms: 13000}),
  ]);
  assert.equal(group(result, "chat:cosmo:medium").successful_requests, 1);
  assert.equal(group(result, "chat:cosmo:medium").average_request_time_to_first_token_ms, 6800);
});

test("Core benchmark rejects retries crossing a serving configuration or route", () => {
  assert.throws(() => summarizeCoreBenchmark([
    attempt(),
    warmAttempt({
      attempt_id: "retry", worker_lifecycle_id: "lifecycle-2",
      gpu_type_id: "NVIDIA L40S",
    }),
  ]), /mixes serving configuration or route/);
  assert.throws(() => summarizeCoreBenchmark([
    attempt(),
    warmAttempt({
      attempt_id: "retry", worker_lifecycle_id: "lifecycle-2",
      route_id: "chat:cosmo:medium", stage_id: "verification-1", reasoning_effort: "medium",
    }),
  ]), /mixes serving configuration or route/);
});

test("Core benchmark validates stage visibility against Chat and Work DAGs", () => {
  assert.throws(() => summarizeCoreBenchmark([
    attempt({route_id: "work:cosmo:max", stage_id: "planning-1", public_response: true, reasoning_effort: "xhigh"}), close(),
  ]), /public_response does not match route DAG/);
  const result = summarizeCoreBenchmark([
    attempt({route_id: "work:cosmo:light", stage_id: "answer-1"}), close(),
  ]);
  assert.equal(group(result, "work:cosmo:light").successful_requests, 1);
});

test("shared Core lifecycle can span routes and conserves allocated overhead", () => {
  const result = summarizeCoreBenchmark([
    attempt({inference_ms: 1000}),
    warmAttempt({request_id: logicalId(2), attempt_id: "planning", route_id: "chat:cosmo:medium", stage_id: "planning-1", public_response: false, reasoning_effort: "medium", inference_ms: 1000, time_to_first_token_ms: null}),
    warmAttempt({request_id: logicalId(2), attempt_id: "answer", route_id: "chat:cosmo:medium", stage_id: "answer-1", public_response: false, reasoning_effort: "medium", inference_ms: 1000, time_to_first_token_ms: null}),
    warmAttempt({request_id: logicalId(2), attempt_id: "verify", route_id: "chat:cosmo:medium", stage_id: "verification-1", reasoning_effort: "medium", inference_ms: 1000}),
    close({billed_lifecycle_ms: 11000}),
  ]);
  assert.ok(Math.abs(group(result).total_attributable_compute_cost_usd - 0.0045) < 1e-12);
  assert.ok(Math.abs(group(result, "chat:cosmo:medium").total_attributable_compute_cost_usd - 0.0065) < 1e-12);
  assert.ok(Math.abs(result.total_attributable_compute_cost_usd - 0.011) < 1e-12);
  assert.equal(result.lifecycle_allocations.length, 2);
});

test("Core benchmark requires one measured cold start and idle tail per lifecycle", () => {
  assert.throws(() => summarizeCoreBenchmark([
    attempt({cold_start: true, worker_start_ms: 0, model_load_ms: 0}), close(),
  ]), /cold attempt missing measured startup/);
  assert.throws(() => summarizeCoreBenchmark([
    attempt(), close({attributed_idle_timeout_ms: 0}),
  ]), /lifecycle close missing measured idle tail/);
  assert.throws(() => summarizeCoreBenchmark([
    attempt(), close({billed_lifecycle_ms: 6000}),
  ]), /startup and idle exceed billed wall time/);
  assert.throws(() => summarizeCoreBenchmark([attempt()]), /exactly one close event/);
  assert.throws(() => summarizeCoreBenchmark([attempt(), close(), close({close_event_id: "close-2"})]), /exactly one close event/);
});

test("Core benchmark rejects malformed model, rates, timing, and serving identity", () => {
  assert.throws(() => summarizeCoreBenchmark([
    attempt({route_id: "work:nova:light"}), close(),
  ]), /model differs from route family/);
  assert.throws(() => summarizeCoreBenchmark([attempt({model: "attacker/model"}), close()]), /unverified model revision/);
  assert.throws(() => summarizeCoreBenchmark([attempt({model_revision: "0".repeat(40)}), close()]), /unverified model revision/);
  assert.throws(() => summarizeCoreBenchmark([attempt({gpu_rate_per_second_usd: 0}), close()]), /gpu_rate/);
  assert.throws(() => summarizeCoreBenchmark([attempt({input_tokens: 0}), close()]), /successful attempt missing input tokens/);
  assert.throws(() => summarizeCoreBenchmark([attempt({output_tokens: 0}), close()]), /successful attempt missing output tokens/);
  assert.throws(() => summarizeCoreBenchmark([attempt({time_to_first_token_ms: 3000}), close()]), /first token exceeds inference/);
  assert.throws(() => summarizeCoreBenchmark([attempt({serving_engine: "unknown"}), close()]), /serving_engine/);
  assert.throws(() => summarizeCoreBenchmark([attempt({container_image_digest: "latest"}), close()]), /container_image_digest/);
  assert.throws(() => summarizeCoreBenchmark([attempt({request_id: "caller-id"}), close()]), /server logical execution ID/);
});

test("Core benchmark treats TTFT as public-stream-only and nullable before first output", () => {
  assert.throws(() => summarizeCoreBenchmark([
    attempt({route_id: "chat:cosmo:medium", stage_id: "planning-1", public_response: false, reasoning_effort: "medium"}), close(),
  ]), /private stage must not claim first-token timing/);
  assert.throws(() => summarizeCoreBenchmark([
    attempt({time_to_first_token_ms: null}), close(),
  ]), /successful public stage missing first-token timing/);
  const result = summarizeCoreBenchmark([
    attempt({outcome: "failed", time_to_first_token_ms: null, output_tokens: 0}), close(),
  ]);
  assert.equal(group(result).failed_attempts, 1);
  assert.equal(group(result).successful_requests, 0);
});

test("Core benchmark prices billed wall time instead of summed overlapping attempts", () => {
  const result = summarizeCoreBenchmark([
    attempt({inference_ms: 8000, time_to_first_token_ms: 1000}),
    warmAttempt({request_id: logicalId(2), attempt_id: "attempt-2", inference_ms: 8000, time_to_first_token_ms: 1000}),
    close({billed_lifecycle_ms: 15000}),
  ]);
  assert.equal(result.total_billed_lifecycle_ms, 15000);
  assert.equal(result.total_observed_attempt_inference_ms, 16000);
  assert.equal(group(result).logical_requests, 2);
  assert.equal(group(result).successful_requests, 2);
  assert.ok(Math.abs(result.total_attributable_compute_cost_usd - 0.015) < 1e-12);
  assert.ok(Math.abs(result.lifecycle_allocations.reduce(
    (total, item) => total + item.total_lifecycle_cost_usd, 0,
  ) - result.total_attributable_compute_cost_usd) < 1e-12);
});

test("Core benchmark rejects a lifecycle shorter than its longest attempt", () => {
  assert.throws(() => summarizeCoreBenchmark([
    attempt({inference_ms: 8000, time_to_first_token_ms: 1000}),
    close({billed_lifecycle_ms: 9000}),
  ]), /longest attempt exceeds billed active window/);
});

test("Core benchmark requires active time for the longest sequential request path", () => {
  assert.throws(() => summarizeCoreBenchmark([
    attempt({attempt_id: "planning", route_id: "chat:cosmo:medium", stage_id: "planning-1", public_response: false, reasoning_effort: "medium", inference_ms: 2500, time_to_first_token_ms: null}),
    warmAttempt({attempt_id: "answer", route_id: "chat:cosmo:medium", stage_id: "answer-1", public_response: false, reasoning_effort: "medium", inference_ms: 2500, time_to_first_token_ms: null}),
    warmAttempt({attempt_id: "verification", route_id: "chat:cosmo:medium", stage_id: "verification-1", reasoning_effort: "medium", inference_ms: 2500}),
    close({billed_lifecycle_ms: 11000}),
  ]), /sequential request path exceeds billed active window/);
});

test("Core benchmark counts quarantined attempts without treating them as success", () => {
  const result = summarizeCoreBenchmark([
    attempt({outcome: "quarantined", time_to_first_token_ms: 500}), close(),
  ]);
  assert.equal(group(result).successful_requests, 0);
  assert.equal(group(result).failed_attempts, 0);
  assert.equal(group(result).quarantined_attempts, 1);
  assert.equal(group(result).non_successful_attempts, 1);
});

test("Core lifecycle rejects mixed hardware even when model and route match", () => {
  assert.throws(() => summarizeCoreBenchmark([
    attempt(),
    warmAttempt({request_id: logicalId(2), attempt_id: "attempt-2", gpu_type_id: "NVIDIA H100 PCIe"}),
    close(),
  ]), /mixes serving configurations/);
});
