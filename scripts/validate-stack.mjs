import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";

const load = async (name) => JSON.parse(await readFile(new URL(`../config/${name}`, import.meta.url)));
const [
  candidate, stack, runpod, catalog, economics, currentIdentity, archivedIdentity, inference, hardware, surface,
  activity, completion, evaluations, nova, cosmo, architecture, routes, ultraPlan,
  coreServing, coreContainer,
] = await Promise.all([
  "candidate.v1.json", "training-stack.v1.json", "runpod-serverless.v1.json",
  "model-catalog.v1.json", "economics.v1.json", "kova-three-family-dataset.v2.json", "identity.v1.json",
  "inference-contract.v1.json", "hardware-benchmark.v1.json", "product-surface.v1.json",
  "activity-event.v1.json", "completion-target.v1.json", "evaluation-gates.v1.json",
  "nova-candidate.v1.json", "cosmo-candidate.v1.json", "provider-architecture.v1.json",
  "route-policy.v1.json", "ultra-orchestration.v1.json",
  "core-serving.v1.json", "core-container.v1.json",
].map(load));

if (candidate.base_model !== "Qwen/Qwen3.8-27B" || !/^[a-f0-9]{40}$/u.test(candidate.base_revision)) {
  throw new Error("unexpected_or_unpinned_self_hosted_candidate");
}
if (candidate.execution.authorized !== false || stack.status !== "planning_only" || stack.execution_authorized !== false) {
  throw new Error("planning_only_required");
}
if (runpod.provider !== "runpod_serverless" || runpod.worker_type !== "flex" || runpod.active_workers !== 0) {
  throw new Error("runpod_must_scale_to_zero");
}
if (
  runpod.physical_endpoint_count !== 2 || runpod.endpoints.length !== 2 ||
  runpod.endpoints.map((endpoint) => endpoint.id).join(",") !== "kova-core,kova-ultra" ||
  runpod.endpoints.some((endpoint) =>
    endpoint.name_reserved !== endpoint.id ||
    endpoint.deployed !== false || endpoint.worker_type !== "flex" || endpoint.active_workers !== 0 ||
    endpoint.max_workers !== 1 || endpoint.flashboot_required !== true ||
    endpoint.cached_model_required !== true || endpoint.streaming_required !== true
  )
) throw new Error("two_blocked_scale_to_zero_runpod_endpoints_required");
if (runpod.billing.usage_model !== "metered_pay_per_second" || runpod.billing.flat_rate_plan !== false) {
  throw new Error("runpod_must_use_metered_billing");
}
if (runpod.billing.auto_pay_enabled !== false || runpod.billing.automatic_credit_reload_allowed !== false) {
  throw new Error("automatic_credit_reload_must_be_disabled");
}
if (
  runpod.safety.paid_execution_authorized !== false ||
  runpod.safety.deployment_authorized !== false ||
  runpod.safety.production_routing_authorized !== false
) {
  throw new Error("runpod_paid_actions_must_be_blocked");
}

if (economics.target_gross_margin !== 0.426 || economics.cost_fraction !== 0.574) throw new Error("unexpected_margin_target");
if (
  economics.enforcement.pricing_status !== "blocked_until_benchmarked" ||
  economics.enforcement.require_per_model_benchmark !== true ||
  economics.enforcement.require_per_effort_benchmark !== true ||
  economics.enforcement.require_usage_telemetry !== true ||
  economics.enforcement.require_periodic_recalibration !== true ||
  economics.enforcement.allow_unmeasured_price_publication !== false
) throw new Error("unmeasured_prices_must_be_blocked");

const approvedIdentityDigest = "ed1b503f947cabc6a7c24a9395bd63b9eff2dd55d57570a0d5a8fd51df3c5bc8";
const identityPrompt = await readFile(new URL("../prompts/kova-identity.v3.txt", import.meta.url));
if (
  archivedIdentity.status !== "superseded_non_authoritative_history" ||
  archivedIdentity.superseded_by !== "prompts/kova-identity.v3.txt" ||
  currentIdentity.prompt_path !== "prompts/kova-identity.v3.txt" ||
  currentIdentity.prompt_sha256 !== approvedIdentityDigest ||
  createHash("sha256").update(identityPrompt).digest("hex") !== approvedIdentityDigest
) throw new Error("truthful_kova_identity_required");

if (
  inference.candidate_selection.source !== "trusted_server_configuration" ||
  inference.candidate_selection.allowlist_source !== "core/current_candidates.py:CORE_SERVING.candidates" ||
  inference.candidate_selection.client_selectable !== false ||
  inference.candidate_selection.benchmark_job_must_select_exactly_one !== true ||
  inference.candidate_selection.production_candidate_selected !== false
) throw new Error("benchmark_worker_candidate_selection_must_be_trusted_and_blocked");
for (const field of [
  "record_type", "request_id", "correlation_id", "attempt_id", "outcome", "model", "model_revision", "adapter_sha256", "adapter_bundle_sha256", "route_id",
  "stage_id", "public_response", "worker_lifecycle_id", "measurement_source", "cold_start",
  "time_to_first_token_ms", "gpu_rate_per_second_usd", "gpu_type_id", "gpu_count",
  "serving_engine", "endpoint_type", "container_image_digest",
]) {
  if (!inference.telemetry.attempt_record_required.includes(field)) throw new Error(`inference_attempt_telemetry_missing:${field}`);
}
for (const field of ["record_type", "close_event_id", "worker_lifecycle_id", "adapter_sha256", "adapter_bundle_sha256", "billed_lifecycle_ms", "attributed_idle_timeout_ms"]) {
  if (!inference.telemetry.lifecycle_close_record_required.includes(field)) throw new Error(`inference_lifecycle_telemetry_missing:${field}`);
}
if (inference.telemetry.lifecycle_close_source !== "trusted_runtime_shutdown_observation") throw new Error("trusted_lifecycle_close_required");
if (
  inference.safety.paid_execution_authorized !== false ||
  inference.safety.accept_arbitrary_model_from_request !== false ||
  inference.safety.return_hidden_reasoning !== false
) throw new Error("inference_must_remain_source_only_and_pinned");
if (
  inference.request.allowed_client_message_roles.join(",") !== "user,assistant" ||
  inference.request.caller_supplied_tool_results_allowed !== false ||
  inference.request.request_id_semantics !== "caller_correlation_only_not_logical_benchmark_identity" ||
  inference.request.reasoning_effort_and_output_limit_must_match_trusted_stage !== true ||
  inference.trusted_execution_context.required.join(",") !==
    "logical_request_id,benchmark_candidate_id,route_id,stage_id,public_response,prior_stage_outputs" ||
  inference.trusted_execution_context.source !== "server_router_and_stage_store_only" ||
  inference.trusted_execution_context.logical_request_id_source !== "server_generated_uuid4_once_per_route_execution" ||
  inference.trusted_execution_context.logical_request_id_format !== "kova-exec-{uuid4}" ||
  inference.trusted_execution_context.benchmark_candidate_id_source !== "trusted_server_configuration_allowlisted_in_current_candidates" ||
  inference.trusted_execution_context.prior_stage_outputs !== "exact_declared_core_dag_dependencies_only" ||
  inference.trusted_execution_context.artifact_trust !== "server_recorded_untrusted_model_output" ||
  inference.trusted_execution_context.trusted_token_recount_after_binding !== true ||
  inference.streaming.public_stage_requires_chunk_iterable !== true ||
  inference.streaming.private_stage_requires_non_stream_response !== true ||
  inference.streaming.assemble_content_tool_calls_and_usage_before_sanitizing !== true ||
  inference.streaming.time_to_first_token_source !== "worker_monotonic_clock_first_visible_public_delta" ||
  inference.streaming.time_to_first_token_unavailable_value !== null ||
  inference.streaming.private_stage_time_to_first_token_is_null !== true ||
  inference.streaming.preserve_measured_time_to_first_token_on_stream_failure !== true ||
  inference.streaming.tool_time_to_first_token_requires_nonempty_fragment_data !== true ||
  inference.streaming.accepted_finish_reasons.join(",") !== "stop,tool_calls" ||
  inference.streaming.non_tool_success_requires_non_whitespace_content !== true ||
  inference.streaming.positive_input_usage_required_for_success !== true ||
  inference.streaming.positive_completion_usage_required_for_success !== true ||
  inference.runtime_identity.source !== "server_provider_runtime" ||
  inference.runtime_identity.required.join(",") !==
    "loaded_model,loaded_model_revision,loaded_adapter_sha256,loaded_adapter_bundle_sha256,worker_lifecycle_id,gpu_type_id,gpu_count,serving_engine,endpoint_type,container_image_digest" ||
  inference.runtime_identity.loaded_model_must_match_selected_pinned_candidate !== true ||
  inference.runtime_identity.loaded_revision_must_match_selected_pinned_candidate !== true ||
  inference.runtime_identity.loaded_adapter_must_match_selected_pinned_candidate !== true ||
  inference.runtime_identity.loaded_adapter_bundle_must_match_selected_pinned_candidate !== true ||
  inference.runtime_identity.validated_before_and_after_each_attempt !== true ||
  inference.runtime_identity.lifecycle_close_must_match_pinned_candidate !== true ||
  inference.telemetry.attempt_outcomes.join(",") !== "success,failed,quarantined" ||
  inference.telemetry.request_id_semantics !== "server_generated_logical_route_execution_id" ||
  inference.telemetry.correlation_id_semantics !== "untrusted_caller_value_never_used_for_grouping" ||
  inference.telemetry.postflight_integrity_failure_outcome !== "quarantined" ||
  inference.telemetry.lifecycle_cost_source !== "provider_measured_billed_lifecycle_wall_time" ||
  inference.telemetry.gpu_rate_per_second_usd_semantics !== "total_worker_gpu_rate_for_configured_gpu_count" ||
  inference.telemetry.lifecycle_cost_allocation !== "startup_idle_equal_per_request_active_proportional_to_observed_attempt_inference" ||
  inference.telemetry.billed_active_window_must_cover_longest_attempt !== true ||
  inference.telemetry.billed_active_window_must_cover_longest_sequential_request_path !== true ||
  inference.telemetry.attempt_durations_are_cost_diagnostics_only !== true ||
  !inference.telemetry.route_time_to_first_token.includes("pre_public_stage_durations")
) throw new Error("inference_executor_contract_invalid");
if (
  hardware.schema_version !== 2 || hardware.status !== "candidate_aware_benchmark_required" ||
  hardware.engine !== "kova-core" ||
  hardware.candidate_source !== "config/core-serving.v1.json:candidates" ||
  hardware.selected_candidate_id !== null || hardware.selected_provider_hardware_id !== null ||
  hardware.provider_inventory_snapshot !== null || hardware.provider_price_snapshot !== null ||
  hardware.inventory_must_be_refreshed_at_benchmark_time !== true ||
  hardware.paid_benchmark_authorized !== false || hardware.deployment_authorized !== false ||
  hardware.production_routing_authorized !== false
) throw new Error("candidate_aware_hardware_benchmark_must_stay_unselected_and_blocked");
if (hardware.candidate_matrices.length !== coreServing.candidates.length) {
  throw new Error("every_core_candidate_requires_a_hardware_matrix");
}
for (const servingCandidate of coreServing.candidates) {
  const matrix = hardware.candidate_matrices.find((item) => item.candidate_id === servingCandidate.id);
  if (
    !matrix || matrix.model !== servingCandidate.model || matrix.model_revision !== servingCandidate.revision ||
    matrix.published_weight_bytes !== servingCandidate.stored_bytes ||
    matrix.minimum_benchmark_vram_gb !== servingCandidate.minimum_benchmark_vram_gb ||
    matrix.selected_provider_hardware_id !== null || matrix.compatibility_verified !== false ||
    matrix.benchmark_complete !== false || !Array.isArray(matrix.eligible_vram_tiers_gb) ||
    matrix.eligible_vram_tiers_gb.length === 0 ||
    matrix.eligible_vram_tiers_gb[0] !== matrix.minimum_benchmark_vram_gb ||
    matrix.eligible_vram_tiers_gb.some((tier) =>
      !Number.isInteger(tier) || tier < matrix.minimum_benchmark_vram_gb
    )
  ) throw new Error(`invalid_hardware_matrix:${servingCandidate.id}`);
}

if (
  architecture.status !== "planning_only" || architecture.application_plane.provider !== "azure_container_apps" ||
  architecture.edge_plane.provider !== "cloudflare" || architecture.edge_plane.inference_allowed !== false
) {
  throw new Error("azure_application_plane_must_be_preserved");
}
if (
  architecture.application_plane.delete_existing_azure_deployments !== false ||
  architecture.application_plane.production_routing_authorized !== false ||
  architecture.global_guards.paid_or_production_actions_authorized !== false
) throw new Error("unverified_migration_actions_must_be_blocked");
const core = architecture.engines.find((engine) => engine.id === "kova-core");
const ultra = architecture.engines.find((engine) => engine.id === "kova-ultra");
if (
  architecture.inference_decision.provider !== "runpod_serverless" ||
  architecture.inference_decision.physical_endpoint_count !== 2 ||
  architecture.inference_decision.cloudflare_workers_ai_status !== "rejected_as_primary_inference_backend" ||
  architecture.inference_decision.do_not_claim_cloudflare_benchmark_evidence !== true ||
  architecture.global_guards.one_core_endpoint_serves_all_core_profiles !== true
) throw new Error("two_runpod_engine_decision_required");
if (
  !core || !ultra || [core, ultra].some((engine) =>
    engine.provider !== "runpod_serverless" || engine.endpoint_deployed !== false ||
    engine.worker_type !== "flex" || engine.active_workers !== 0 || engine.selected_model !== null ||
    engine.selected_quantization !== null || engine.selected_gpu !== null ||
    engine.selected_serving_engine !== null || engine.paid_execution_authorized !== false ||
    engine.deployment_authorized !== false || engine.production_routing_authorized !== false
  ) || core.endpoint_name_reserved !== "kova-core" || ultra.endpoint_name_reserved !== "kova-ultra" ||
  ultra.hidden_chain_of_thought_exposed !== false
) throw new Error("both_runpod_engines_must_be_unselected_scale_to_zero_and_blocked");
if (
  coreServing.status !== "superseded_non_authoritative_history" ||
  coreServing.superseded_by !== "config/kova-three-family-pilot.v1.json" ||
  coreServing.must_not_drive_current_routing !== true || coreServing.engine !== "kova-core" ||
  coreServing.provider !== "runpod_serverless" || coreServing.endpoint_name_reserved !== "kova-core" ||
  coreServing.endpoint_deployed !== false || coreServing.selected_serving_engine !== null ||
  coreServing.selected_candidate_id !== null || coreServing.selected_gpu !== null ||
  coreServing.container_image_digest !== null || coreServing.native_context_tokens !== 262144 ||
  coreServing.reasoning_efforts.join(",") !== "low,medium,xhigh" ||
  coreServing.endpoint_type_candidates.join(",") !== "queue_based,load_balancing" ||
  coreServing.container_policy.prebuilt_image_required !== true ||
  coreServing.container_policy.runtime_package_installs_allowed !== false ||
  coreServing.container_policy.streaming_required !== true ||
  coreServing.container_policy.hidden_reasoning_must_be_filtered !== true ||
  Object.values(coreServing.safety).some((value) => value !== false)
) throw new Error("runpod_core_serving_selection_must_stay_blocked");
const bf16Core = coreServing.candidates.find((model) => model.model === candidate.base_model);
const fp8Core = coreServing.candidates.find((model) => model.model === cosmo.base_model);
if (
  coreServing.candidates.length !== 2 || !bf16Core || !fp8Core ||
  coreServing.candidates.map((model) => model.id).join(",") !== "qwen3.8-27b-bf16,qwen3.8-27b-fp8" ||
  bf16Core.revision !== candidate.base_revision || bf16Core.license !== candidate.base_license ||
  fp8Core.revision !== cosmo.base_revision || fp8Core.license !== cosmo.base_license ||
  coreServing.candidates.some((model) =>
    model.context_tokens !== 262144 || model.compatibility_verified !== false || model.benchmark_complete !== false
  )
) throw new Error("verified_unbenchmarked_core_candidates_required");
const requiredContainerSafetyKeys = [
  "compatibility_tested", "deployment_authorized", "endpoint_creation_authorized",
  "image_built", "image_pulled", "paid_benchmark_authorized",
  "production_routing_authorized", "runtime_package_installs_allowed",
];
if (
  coreContainer.status !== "source_only_build_blocked" || coreContainer.engine !== "kova-core" ||
  coreContainer.provider !== "runpod_serverless" ||
  coreContainer.role !== "upstream_inference_server_candidate" ||
  coreContainer.upstream_worker.repository !== "runpod-workers/worker-vllm" ||
  coreContainer.upstream_worker.release_tag !== "v2.27.0" ||
  coreContainer.upstream_worker.source_commit !== "76054c22c79c515f07065f523598d8efb2f9b682" ||
  coreContainer.upstream_worker.bundled_vllm_version !== "0.29.0" ||
  coreContainer.upstream_worker.candidate_image_reference !== "runpod/worker-v1-vllm:v2.27.0" ||
  coreContainer.upstream_worker.resolved_image_digest !== null ||
  coreContainer.integration.selected !== false ||
  coreContainer.integration.kova_benchmark_contract !== "worker/handler.py" ||
  coreContainer.integration.queue_protocol_adapter !== "worker/runpod_vllm.py" ||
  coreContainer.integration.queue_protocol_adapter_implemented !== true ||
  coreContainer.integration.cpu_protocol_fixture_verified !== true ||
  coreContainer.integration.live_provider_envelope_verified !== false ||
  coreContainer.integration.openai_compatible_api_required !== true ||
  coreContainer.integration.streaming_required !== true ||
  coreContainer.integration.hidden_reasoning_filter_required !== true ||
  coreContainer.integration.model_compatibility_verified !== false ||
  coreContainer.integration.runtime_identity_probe_verified !== false ||
  coreContainer.integration.provider_tokenizer_recount_verified !== false ||
  coreContainer.context.native_context_tokens !== 262144 ||
  coreContainer.context.selected_max_model_len !== null ||
  coreContainer.context.selection_status !== "benchmark_required" ||
  coreContainer.context.full_native_context_fit_claimed !== false ||
  coreContainer.endpoint.type_candidates.join(",") !== "queue_based,load_balancing" ||
  coreContainer.endpoint.selected_type !== null ||
  coreContainer.endpoint.cold_scale_to_zero_compatibility_verified !== false ||
  Object.keys(coreContainer.safety).sort().join(",") !== requiredContainerSafetyKeys.join(",") ||
  requiredContainerSafetyKeys.some((key) => coreContainer.safety[key] !== false)
) throw new Error("pinned_core_container_candidate_must_stay_unbuilt_and_blocked");
if (
  coreContainer.transport.target !== "runpod_serverless_queue" ||
  coreContainer.transport.queue_input_shape !== "openai_passthrough" ||
  coreContainer.transport.openai_route !== "/v1/chat/completions" ||
  coreContainer.transport.stream_worker_output !== "raw_openai_sse" ||
  coreContainer.transport.implemented_boundary !==
    "pinned_worker_yielded_output_after_provider_transport" ||
  coreContainer.transport.full_runpod_http_envelope_supported !== false ||
  coreContainer.transport.network_client_implemented !== false ||
  coreContainer.transport.live_transport_verified !== false
) throw new Error("runpod_queue_transport_must_remain_cpu_only_and_live_unverified");
if (
  coreContainer.candidate_runtime_configuration.source !== "trusted_server_configuration" ||
  coreContainer.candidate_runtime_configuration.client_overrides_allowed !== false ||
  coreContainer.candidate_runtime_configuration.required_environment.join(",") !== "MODEL_NAME,MODEL_REVISION" ||
  coreContainer.candidate_runtime_configuration.allowed_tuning_environment.join(",") !==
    "MAX_MODEL_LEN,GPU_MEMORY_UTILIZATION,MAX_NUM_SEQS,TENSOR_PARALLEL_SIZE" ||
  coreContainer.candidate_runtime_configuration.profiles.length !== coreServing.candidates.length
) throw new Error("core_container_candidate_configuration_must_be_trusted");
for (const servingCandidate of coreServing.candidates) {
  const profile = coreContainer.candidate_runtime_configuration.profiles.find(
    (item) => item.candidate_id === servingCandidate.id,
  );
  if (
    !profile || profile.MODEL_NAME !== servingCandidate.model ||
    profile.MODEL_REVISION !== servingCandidate.revision
  ) throw new Error(`core_container_profile_mismatch:${servingCandidate.id}`);
}
if (
  coreContainer.weights.runtime_model_download_allowed !== false ||
  runpod.model_storage.runtime_model_download_allowed !== false ||
  coreContainer.weights.network_volume_enabled !== false ||
  runpod.model_storage.network_volume_enabled !== false ||
  coreContainer.weights.selected_packaging_strategy !== null ||
  coreContainer.weights.packaging_strategy_candidates.join(",") !==
    "model_baked_into_derived_image,immutable_cached_artifact" ||
  coreContainer.weights.model_baked_into_image !== false ||
  coreContainer.weights.immutable_cached_artifact_present !== false ||
  coreContainer.weights.cache_warmth_verified !== false
) throw new Error("core_container_weights_must_stay_uncached_and_selection_blocked");
if (
  ultraPlan.status !== "source_only" || ultraPlan.engine !== "kova-ultra" ||
  ultraPlan.provider !== "runpod_serverless" || ultraPlan.worker_type !== "flex" ||
  ultraPlan.endpoint_name_reserved !== "kova-ultra" || ultraPlan.endpoint_deployed !== false ||
  ultraPlan.active_workers !== 0 || ultraPlan.required_entitlement !== "pro" ||
  ultraPlan.minimum_specialists !== 2 || ultraPlan.maximum_specialists !== 5 ||
  ultraPlan.maximum_debate_rounds !== 1 || ultraPlan.selected_model !== null ||
  ultraPlan.required_stages.join(",") !== "specialists,disagreement_check,judge,conditional_debate,synthesis" ||
  ultraPlan.hidden_chain_of_thought_exposed !== false ||
  ultraPlan.paid_execution_authorized !== false || ultraPlan.deployment_authorized !== false ||
  ultraPlan.production_routing_authorized !== false
) throw new Error("ultra_orchestration_must_be_bounded_and_blocked");

const expectedChatModes = ["instant", "medium", "high", "extra-high", "max", "ultra"];
if (
  surface.assistant_name !== "Kova" || surface.auto_route.id !== "kova-auto" ||
  surface.auto_route.classifier_implemented !== true || surface.auto_route.deployment_ready !== false
) {
  throw new Error("blocked_kova_auto_surface_required");
}
if (surface.chat_modes.map((mode) => mode.id).join(",") !== expectedChatModes.join(",")) throw new Error("six_ordered_chat_modes_required");
if (
  surface.chat_modes.map((mode) => mode.display_name).join(",") !==
  "Kova Cosmo,Kova Orion,Kova Nova,Nova Extra High,Nova Max,Kova Ultra"
) throw new Error("product_surface_kova_names_invalid");
if (surface.chat_modes.slice(0, 5).some((mode) => mode.engine !== "kova-core")) throw new Error("auto_through_max_must_use_core");
if (surface.chat_modes.find((mode) => mode.id === "ultra").engine !== "kova-ultra") throw new Error("ultra_must_change_engine");
if (surface.chat_modes.find((mode) => mode.id === "instant").activity_updates !== false) throw new Error("instant_must_respond_directly");
for (const id of ["high", "extra-high", "max", "ultra"]) {
  if (surface.chat_modes.find((mode) => mode.id === id).activity_updates !== true) throw new Error(`deep_mode_requires_activity:${id}`);
}
if (surface.chat_modes.some((mode) => mode.deployment_ready !== false)) throw new Error("all_chat_modes_must_stay_blocked");
if (surface.work_families.map((family) => family.display_name).join(",") !== "Kova Cosmo,Kova Orion,Kova Nova") {
  throw new Error("three_kova_work_families_required");
}
if (surface.work_efforts.length !== 6 || surface.work_families.length * surface.work_efforts.length !== 18) {
  throw new Error("eighteen_work_combinations_required");
}
if (surface.effort_profiles.length !== 6 || surface.effort_profiles.map((profile) => profile.name).join(",") !== surface.work_efforts.join(",")) {
  throw new Error("six_distinct_work_effort_profiles_required");
}
if (surface.effort_profiles.some((profile, index, profiles) => index > 0 && profile.maximum_output_tokens <= profiles[index - 1].maximum_output_tokens)) {
  throw new Error("work_effort_budgets_must_increase");
}
if (surface.effort_profiles.slice(0, 5).some((profile) => profile.engine !== "kova-core") || surface.effort_profiles[5].engine !== "kova-ultra") {
  throw new Error("work_ultra_must_change_engine");
}

if (
  routes.shared_core_weights !== true || routes.chat.length !== 6 ||
  routes.auto.classifier_implemented !== true || routes.auto.classifier_type !== "deterministic_server_rules_v1" ||
  routes.auto.free_plan_route_cap !== "instant" || routes.auto.ultra_entitlement !== "pro" ||
  routes.auto.ultra_budget_gate_required !== true || routes.auto.production_ready !== false
) {
  throw new Error("route_policy_must_be_source_only_and_fail_closed");
}
if (routes.chat.slice(0, 5).some((route) => route.engine !== "kova-core") || routes.chat[5].engine !== "kova-ultra") {
  throw new Error("route_policy_engine_boundary_invalid");
}
if (routes.chat[0].answer_passes !== 1 || routes.chat[0].activity_updates !== false) throw new Error("instant_route_must_be_one_pass");
if (
  !routes.caller_forbidden_fields.includes("model") ||
  !routes.caller_forbidden_fields.includes("engine") ||
  !routes.caller_forbidden_fields.includes("behavior_contract_id")
) {
  throw new Error("route_provider_and_model_must_be_server_controlled");
}
if (
  routes.work.family_profiles.length !== 3 ||
  new Set(routes.work.family_profiles.map((profile) => profile.behavior_contract_id)).size !== 3 ||
  new Set(routes.work.family_profiles.map((profile) => profile.answer_style)).size !== 3 ||
  new Set(routes.work.family_profiles.map((profile) => profile.tool_posture)).size !== 3
) throw new Error("work_families_require_distinct_behavior_contracts");
if (
  routes.work.effort_profiles.length !== 6 ||
  routes.work.effort_profiles.map((profile) => profile.name).join(",") !== routes.work.efforts.join(",") ||
  routes.work.effort_profiles.slice(0, 5).some((profile) => profile.engine !== "kova-core") ||
  routes.work.effort_profiles[5].engine !== "kova-ultra"
) throw new Error("work_effort_route_contracts_invalid");

if (surface.deep_mode_experience.hidden_chain_of_thought_exposed !== false || activity.rules.may_expose_hidden_reasoning !== false) {
  throw new Error("hidden_reasoning_must_stay_private");
}
if (activity.rules.must_follow_real_runtime_or_tool_event !== true || activity.rules.may_claim_unstarted_action !== false) {
  throw new Error("activity_must_be_truthfully_grounded");
}
if (!activity.required_fields.includes("grounding_operation_id")) throw new Error("activity_runtime_grounding_id_required");
if (completion.baseline_percent !== 0 || completion.current_verified_percent !== 22 || completion.live_model_routes !== 0 || completion.target_model_routes !== 37) {
  throw new Error("completion_progress_contract_mismatch");
}
if (
  evaluations.status !== "all_routes_blocked" || evaluations.target_routes !== 37 ||
  evaluations.passing_routes.length !== 0 ||
  evaluations.offline_contract_evidence.status !== "passed" ||
  evaluations.offline_contract_evidence.route_contracts_checked !== 37 ||
  evaluations.offline_contract_evidence.engine_split.auto !== 1 ||
  evaluations.offline_contract_evidence.engine_split.core !== 30 ||
  evaluations.offline_contract_evidence.engine_split.ultra !== 6 ||
  evaluations.offline_contract_evidence.actual_model_outputs_evaluated !== false ||
  evaluations.offline_contract_evidence.quality_or_factuality_claimed !== false ||
  evaluations.offline_contract_evidence.paid_provider_calls !== 0 ||
  Object.values(evaluations.release_policy).some((value) => value !== false) ||
  !evaluations.required_per_route.includes("truthful_selected_provider_and_upstream_model_disclosure_when_asked")
) {
  throw new Error("all_routes_require_real_evaluation_evidence");
}

const evaluated = catalog.evaluated_self_hosted_candidates;
for (const source of [candidate, cosmo, nova]) {
  if (!evaluated.some((item) => item.model === source.base_model && item.revision === source.base_revision && item.license === source.base_license)) {
    throw new Error(`catalog_missing_verified_candidate:${source.base_model}`);
  }
}
if (cosmo.runpod.paid_benchmark_authorized !== false || nova.execution.authorized !== false) {
  throw new Error("candidate_specific_paid_execution_must_stay_blocked");
}
if (catalog.physical_engines.some((engine) => engine.deployment_ready !== false || engine.upstream_model !== null)) {
  throw new Error("unselected_physical_engines_must_stay_blocked");
}
if (catalog.public_profiles.some((profile) => profile.deployment_ready !== false || profile.separate_foundation_weights !== false)) {
  throw new Error("public_profiles_must_be_truthful_and_blocked");
}
if (
  catalog.user_facing_hierarchy.map((mode) => mode.display_name).join(",") !==
  "Kova Auto,Kova Cosmo,Kova Orion,Kova Nova,Nova Extra High,Nova Max,Kova Ultra"
) throw new Error("final_kova_mode_hierarchy_required");

console.log("Validated Kova two-engine planning stack; paid execution and production routing remain blocked.");
