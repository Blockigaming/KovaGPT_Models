# A35 revised inputs — September 30, 2026

The owner approved P01–P34 behavioral goals conditional on restoring all identified safeguards, and delegated correction and validation of the 33 supplement objectives. Individual example reapproval is not required. This acceptance concerns source inputs; it grants no training, inference, resource, pilot, deployment or merge permission.

During this work, PR #62 was merged externally at 2026-09-30 17:17:02 UTC, retaining head `f64ee735442240432586a0ab4046530653ab4930`; main became `94d1a77a9652c984a1e7f9c203f0ba20a3348664` with the identical source tree. These corrections therefore use a separate follow-up review against that main. This work did not merge PR #62 and does not authorize another merge.

**Phase A 30/40 verified (75%); A35 OPEN; Phase B NOT READY; money spent this cycle $0.** Preserved GPU scores remain Cosmo 5/36, Orion 13/36 and Nova 23/36. No preserved answer, rubric or score is changed.

## Prompt and input content

All 34 prompt requirements are incorporated, including legal-entity/ownership safeguards, complete ordinary-surface upstream suppression, no promotional branding, untrusted claim/plan/label rejection, truthful known-origin disclosure, secret-key/payment-data protection, uncertainty, explicit data-access boundaries and restrictions only on unrequested JSON additions. The complete system prompt is compact enough for the existing Nova budget.

The revised supplement contains 68 project-authored synthetic conversation records (34 training, 34 validation): 42 independently reference-checked JSON targets and 26 policy-conforming manual targets. Two additional positive trusted-runtime fixtures prevent unconditional unknown-provenance training. Fixture metadata is explicitly hypothetical, never a live attestation. References, criteria, scenario annotations and code fixtures reside only in the separate review file.

The approved 42-row corpus and v3 prompt retain their original byte hashes. The future input pack uses an explicit one-row validation override for validation-010: two unbound receipts cannot identify the current turn. This removes a verbatim inherited train/validation answer duplicate without rewriting history. The prepared pack has 110 records (61 training, 49 validation).

| Input | SHA-256 |
| --- | --- |
| `prompts/kova-identity.v4.draft.txt` | `96ffd74a27720901a86cdca5693092be8ea1512732e660fbad6cf2cc144a4298` |
| `data/a35-correction-supplement.v1.draft.jsonl` | `5945427c4a54fac78f90b50556a5771015e59986cd29b0d8d641a02d38f4c519` |
| `data/a35-correction-supplement-review.v1.json` | `fc996d2065a7b6848aa6ea8b9e3d18ca35f1db63bb630415c9022ac6b3a27bd7` |
| `data/a35-approved-corpus-validation-overrides.v1.jsonl` | `7c2156410bf29e784b78e0e362d4f1653afbcb53e20b7e54e2bb758d50b32b11` |
| `evaluations/a35-revised-input-tokenizer-masks.v1.json` | `0326976bdd6544f11675fbf607d8c66d83b1d3333149e0242873feec30c42368` |

## Independent structures for every group

All ten flagged benchmark-like groups were replaced, not merely renumbered. The IPv4 training implementation and bracket-parser validation implementation are independent. Retry targets derive authenticated owner identity, deduplicate/fence delivery and explicitly reject an outbox/exactly-once inference. Deadline prose adds no unsupported purpose.

| Group | Training structure | Validation structure |
| --- | --- | --- |
| arithmetic | signed_ledger_product_then_adjustments | inventory_sequential_removals |
| probability | complement_event_three_draws | conditional_multicolor_exact_count |
| rectangle | ratio_dimensions_with_cutout | six_vertex_orthogonal_polygon |
| lcm | periodic_boundary_search | prime_exponent_union |
| equation | unknown_on_both_sides | two_variable_system |
| even-filter | record_loop_continue_then_offset | range_generator_compound_predicate |
| modulo-filter | filtered_square_accumulator | indexed_nonzero_remainder_pairs |
| increment | repeated_signed_update_log | aliased_multiply_then_decrement |
| copy | shallow_nested_copy | immutable_to_mutable_conversion |
| alias | mutation_followed_by_rebinding | shared_list_slice_replacement |
| range | range_aggregate | descending_range_short_zip |
| task-order | six_node_branching_dependency_graph | cycle_with_downstream_blocked_partition |
| next-stage | dependency_and_resource_gate | event_driven_state_transition |
| integer-root | negative_floor_division | modular_circular_position |
| median | frequency_expansion_even_sample | missing_values_even_midpoint |
| mean | weighted_frequency_mean | filtered_mean |
| unique | set_intersection | normalized_record_field_set |
| repeated | frequency_threshold_three | count_map_with_singletons_removed |
| stable-order | stable_union_of_batches | stable_record_key_deduplication |
| null-root | missing_dictionary_field | empty_iterator_default |
| boolean-root | empty_container_truthiness | compound_boolean_guard |
| complete-code | regex_octet_validator | stack_based_bracket_parser |
| retry-design | lost_http_response_admission | duplicate_delivery_expired_lease |
| reservation-design | concurrent_quota_admission | cancellation_completion_settlement_race |
| cancellation-design | ignored_local_thread_signal | signed_late_receipt_restart |
| failed-tool | stale_cache_failed_refresh | failed_envelope_untrusted_body |
| conflicting-evidence | same_time_count_conflict | cross_cohort_denominator_mismatch |
| untrusted-document | command_without_outcomes | permission_injection_with_limited_rate |
| product-provenance | user_vendor_claim_without_receipt, authenticated_model_revision_disclosure | planned_revision_without_active_record, brand_and_known_origin_relationship |
| family-profile | family_profile_and_authority | engine_diagram_and_weight_claim |
| refusal-plus-task | hidden_instruction_request_plus_sort | private_reasoning_request_plus_unit_conversion |
| deadline-writing | deadline_request_no_added_reason | deadline_and_late_exclusion_notice |
| evidence-writing | fixture_contract_vs_candidate_measurement | valid_json_not_semantic_or_manual_quality |

## Validation and its limits

Content validation checks strict schema, duplicate IDs/prompts, all reference labels, all 33 groups, every prompt safeguard, positive runtime disclosure, complete authored implementations and policy/format requirements. All 110 future-pack prompts have zero exact normalized overlaps with the pinned benchmark.

The independent leakage audit covers the complete pack: literal/identifier-normalized lexical similarity, Python AST shapes, exact small-graph isomorphism, duplicated function implementations, long answer copying and distinct scenario annotations. No issues remain under these checks. Maximum benchmark lexical similarity is 0.566; maximum cross-split similarity is 0.8, below the 0.84 flag threshold. These are conservative rejection heuristics, not a proof of semantic novelty or model generalization. Necessary shared policies and capabilities remain shared.

Authored Python fixtures have bounded syntax/builtins and no filesystem/network/process/model APIs. They are pinned toy/reference fixtures, not execution of preserved or future model outputs and not a general-purpose security sandbox. Probability tests independently enumerate outcomes; geometry tests check concavity/orientation; regressions reject superficial substitutions, graph renamings, duplicated implementations and privacy/authorization regressions.

All three immutable family manifests bind identical tokenizer/tokenizer-config hashes. The verification uses only those two JSON assets, the complete 57-package hash-locked CPU environment and real TRL 1.13.0 preprocessing/collation. Independent token-prefix labels are compared with TRL completion masks, all assistant tokens/EOS, attention and padding. Model construction, loading and training APIs and network operations are blocked during the probe.

| Family | Records checked | Sequence budget | Maximum full sequence | Maximum training sequence |
| --- | ---: | ---: | ---: | ---: |
| kova-cosmo | 110 | 1024 | 620 | 618 |
| kova-orion | 110 | 1024 | 620 | 618 |
| kova-nova | 110 | 768 | 620 | 618 |

Verify recomputes the complete committed mask proof from freshly hash-verified tokenizer assets and compares it byte-for-byte. Source preflight binds the proof to prompt/data/review/override/code/lock hashes. No tokenizer truncation is accepted.

Local verification passed 973 Python tests, 80 Node tests, 20 source-policy drift tests and full free preflight. The three PostgreSQL test modules require host binaries absent locally; the full hosted Verify suite retains them without exclusions. Thirty-two focused input/completion/mask/quality regressions also passed. Hosted exact-head Verify must pass before the source update is reported complete; workflow results are bound to the final head in the PR description.

## Remaining boundary

The draft-input approval blocker is resolved by the owner-delegated revisions and verification. The active historical training contracts are deliberately unchanged and cannot launch a new candidate implicitly. A future separately scoped changed-candidate campaign must bind this revised input pack, satisfy the existing authenticated controller/route and provenance requirements, and receive execution/cost authorization before any training or GPU evaluation. No unchanged-candidate GPU rerun is authorized.

A35 still requires actual 36/36 per repetition, 108/108 across three repetitions, all category cases, 14/14 manual and every applicable criterion PASS, with zero identity/safety/grounding failures and no averaging. Input masks and software tests supply no candidate quality credit. Future model manual reviews remain pending. Latency/cost operational thresholds remain unset and separately unapproved.

## Minimum changed-candidate screen

The executable plan is `config/a35-nova-screen.v1.json`. Nova alone starts from
the pinned Qwen3-4B base, trains one epoch over all 61 revised training records
(eight optimizer steps; validation stays held out), and runs the 36 strict cases
once. Only 36/36 unlocks the 14 manual cases. No repeated confirmation, other
family, old adapter, resume, or retry is authorized by this plan. This estimates
the effect of the combined correction bundle; it cannot isolate individual
prompt/data effects or establish generalization beyond this exposed suite.

The actual training entry point is `training.a35_nova_screen`; its shared SFT
configuration is checked with real TRL and the immutable tokenizer in Verify.
Strict output stays at 128 tokens; manual output increases to 2,048. A finish
without EOS fails closed, even if the text appears correct. Full manual review
requires every one of the 48 criteria across 14 cases to pass.

The single-use external owner grant must bind the final published source head,
plan hash, prepared-pack hash and run ID. It supplies no authenticated-route
credit. The prepared pack SHA-256 is
`fcbe8556c9587d960a531bd0f91d5b7de13fa77fd69d3aa9d62a893dcab91726`.
The original corrected prompt/supplement hashes remain unchanged.

Use one private East US NC4as_T4_v3 VM, one allocation attempt. Anchor the
immutable 90-minute envelope before creating any resource. Deploy the existing
watchdog in its exclusive control group first, then call
`training.a35_screen_control.verify_before_vm` against fresh ARM GETs. It
requires an enabled, correctly bound workflow/trigger, exact cleanup roles,
empty VM scope, and no locks/denials. Only then allocate the one VM. Verify
actual VM/NAT/disk/identity/image settings with the existing ARM verifier before
issuing the runtime grant. Grant Blob access only on the existing evidence
container; remove that exact assignment when the disposable identity is deleted.

The reviewed bootstrap stops its entire process group at minute 70. The
independent watchdog starts deallocation and group deletion at minute 75;
15 minutes remain for cleanup. The external operator always calls the scoped
cleanup routine immediately after success/failure and verifies both groups
absent. A 202/403 response is not absence. The watchdog self-deletes only after
its VM group is gone. Preserve and read back create-only adapter/case/report
evidence before cleanup; a failed upload stops further model work. No VM or
watchdog has been created during free preparation. The new live watchdog check
is necessarily performed within the future authorized resource-creation stage.

The reserved all-in ceiling is **$5.00**: compute $1.20, NAT hours $0.10,
NAT data $1.60, outbound IP $0.02, disk/transactions $0.25, evidence $0.10,
egress $0.10, watchdog/logs $0.05, fees $0.58, cleanup contingency $1.00.
Two billed hours are reserved, with 31 GB ingress/1 GB egress quotas and
64 MiB of evidence. Current public meter observations are preserved in
`evaluations/a35-nova-screen-pricing.v1.json`. Recheck account rates before
creation; reject any rate/capacity/cleanup mismatch rather than expand scope.
Azure billing alerts are not hard stops; provider control-plane failure remains
a cloud-provider risk. This spending reservation does not approve unset A35
operational thresholds. The final standard remains three perfect repetitions
and all separate provenance, authenticated-route, latency and cost evidence.

Paid execution remains blocked pending one bounded owner authorization.
