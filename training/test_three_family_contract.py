import base64
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import copy

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from evaluation import three_family_guard as guard
from training.three_family_operator import command_plan

from release.source_policy_drift import validate as validate_policy_drift
from unittest.mock import patch

from training import three_family_contract as contract


class ThreeFamilyContractTests(unittest.TestCase):
    def test_operator_has_exact_dry_run_sequence(self):
        plan = command_plan()
        self.assertEqual([step["id"] for step in plan], list(range(1, 20)))
        self.assertTrue(all(step["mode"] != "execute" for step in plan))
        self.assertEqual(plan[16]["mode"], "print_only_destructive")
        self.assertIn("--retail-price-evidence", plan[0]["verify_argv"])
        self.assertEqual(plan[2]["mode"], "print_only_resource_creation")
        self.assertFalse(plan[2]["resource_creation_authorized"])
        self.assertEqual(plan[2]["must_succeed_before_step"], 4)
        self.assertEqual({argv[4] for argv in plan[2]["argvs"]},
                         {"${PILOT_RESOURCE_GROUP}", "${WATCHDOG_RESOURCE_GROUP}"})
        self.assertTrue(all(argv[:3] == ["az", "group", "create"] for argv in plan[2]["argvs"]))
        self.assertIn("provisionPilot=true", plan[3]["argv"])
        self.assertIn("provisionWatchdog=true", plan[4]["argv"])
        self.assertIn("pilotSuffix=${PILOT_SUFFIX}", plan[4]["argv"])
        self.assertIn("provisionPilot=true", plan[5]["argv"])
        self.assertIn("--all-families", plan[7]["argv"])
        cleanup = json.dumps(plan[16:18])
        self.assertIn("${PILOT_RESOURCE_GROUP}", cleanup)
        self.assertIn("${WATCHDOG_RESOURCE_GROUP}", cleanup)

    def test_generated_answer_is_bound_to_pinned_key_and_runtime_context(self):
        digest = "a" * 64
        private_key = Ed25519PrivateKey.generate()
        trusted_public = base64.b64encode(private_key.public_key().public_bytes_raw()).decode("ascii")
        bindings = {
            "expected_source_commit": "d" * 40,
            "expected_family": "kova-cosmo",
            "expected_base_revision": "e" * 40,
            "expected_base_manifest_sha256": digest,
            "expected_adapter_sha256": "b" * 64,
            "expected_adapter_bundle_sha256": "f" * 64,
            "expected_runner_sha256": "c" * 64,
            "expected_case_id": "case-1",
            "expected_variant": "trained_adapter",
            "expected_case_category": "generated_answer",
            "expected_runtime_profile": "high",
            "expected_conversation_id": "conversation-1",
            "expected_session_id": "session-1",
        }
        evidence = guard.create_evidence(
            source_commit=bindings["expected_source_commit"], family="kova-cosmo",
            base_revision=bindings["expected_base_revision"], base_manifest_sha256=digest,
            adapter_sha256="b" * 64, adapter_bundle_sha256="f" * 64,
            runner_sha256="c" * 64, case_id="case-1",
            prompt="Who are you?", answer="I am Kova.", runtime_profile="high",
            conversation_id="conversation-1", session_id="session-1",
            dimensions=["kova_identity_consistency"], private_key=private_key,
            variant="trained_adapter", case_category="generated_answer",
            created_at="2026-09-21T00:00:00+00:00",
        )
        self.assertNotIn("public_key_ed25519_b64", evidence)
        with patch.object(guard, "PINNED_RUNNER_PUBLIC_KEY_B64", trusted_public):
            verified = guard.verify_evidence(evidence, **bindings)
            self.assertEqual(verified["answer"], "I am Kova.")
            with self.assertRaisesRegex(ValueError, "evidence_binding_mismatch:session_id"):
                guard.verify_evidence(evidence, **{**bindings, "expected_session_id": "session-2"})
            with self.assertRaisesRegex(ValueError, "evidence_binding_mismatch:adapter_bundle_sha256"):
                guard.verify_evidence(
                    evidence, **{**bindings, "expected_adapter_bundle_sha256": "9" * 64})
            tampered = copy.deepcopy(evidence)
            tampered["payload"]["answer"] = "altered"
            tampered["payload"]["answer_sha256"] = hashlib.sha256(b"altered").hexdigest()
            with self.assertRaisesRegex(ValueError, "invalid_evidence_signature"):
                guard.verify_evidence(tampered, **bindings)

    def test_evaluation_dimensions_are_configured_unique_and_coverable(self):
        private_key = Ed25519PrivateKey.generate()
        kwargs = {
            "source_commit": "d" * 40, "family": "kova-cosmo", "base_revision": "e" * 40,
            "base_manifest_sha256": "a" * 64, "adapter_sha256": "b" * 64,
            "adapter_bundle_sha256": "f" * 64,
            "runner_sha256": "c" * 64, "case_id": "case-1", "prompt": "p", "answer": "a",
            "runtime_profile": "light", "conversation_id": "conversation-1",
            "session_id": "session-1", "private_key": private_key,
            "variant": "trained_adapter", "case_category": "generated_answer",
        }
        for dimensions in (["not-a-required-dimension"], ["kova_identity_consistency"] * 2, [1]):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                guard.create_evidence(**kwargs, dimensions=dimensions)
        payloads = [
            {"family": family, "variant": variant, "case_category": category,
             "case_id": f"{family}:{variant}:{category}:{index}",
             "runtime_profile": (
                 guard.PROFILE_ORDER[index - 1] if category == "runtime_profile" else "light"),
             "dimensions": list(guard._expected_case_dimensions(
                 f"{family}:{variant}:{category}:{index}"))}
            for family in sorted(guard.FAMILIES) for variant in guard.VARIANTS
            for category, count in guard.CASE_CATEGORIES.items()
            for index in range(1, count + 1)
        ]
        self.assertEqual(
            set(guard.validate_verified_dimension_coverage(payloads)),
            set(guard.REQUIRED_DIMENSIONS),
        )
        with self.assertRaisesRegex(ValueError, "incomplete_evaluation_matrix"):
            guard.validate_verified_dimension_coverage(payloads[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate_evaluation_case"):
            guard.validate_verified_dimension_coverage(payloads[:-1] + [payloads[0]])
        mislabeled = deepcopy(payloads)
        mislabeled[0]["dimensions"] = ["cross_user_isolation", "exact_adapter_base_binding"]
        with self.assertRaisesRegex(ValueError, "evaluation_dimension_binding_mismatch"):
            guard.validate_verified_dimension_coverage(mislabeled)
        for field, value in (("variant", "fake"), ("family", "kova-unknown"),
                             ("case_id", "other"), ("runtime_profile", "unknown")):
            changed = deepcopy(payloads)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                guard.validate_verified_dimension_coverage(changed)

    def test_evaluation_declaration_requires_bundle_binding(self):
        original_load = contract.load_json

        def without_bundle_binding(path, *args, **kwargs):
            value = original_load(path, *args, **kwargs)
            if path.name == "kova-three-family-evaluation.v1.json":
                value["answer_binding"]["binds"].remove("adapter_bundle_sha256")
            return value

        with patch.object(contract, "load_json", side_effect=without_bundle_binding), \
             self.assertRaisesRegex(ValueError, "evaluation signature binding declaration drift"):
            contract.validate_compatibility_evaluation_profiles()

    def test_unconfigured_runner_key_fails_closed(self):
        private_key = Ed25519PrivateKey.generate()
        evidence = guard.create_evidence(
            source_commit="d" * 40, family="kova-cosmo", base_revision="e" * 40,
            base_manifest_sha256="a" * 64, adapter_sha256="b" * 64,
            adapter_bundle_sha256="f" * 64,
            runner_sha256="c" * 64, case_id="case-1", prompt="p", answer="a",
            runtime_profile="light", conversation_id="conversation-1", session_id="session-1",
            dimensions=["kova_identity_consistency"], private_key=private_key,
            variant="trained_adapter", case_category="generated_answer",
        )
        with patch.object(guard, "PINNED_RUNNER_PUBLIC_KEY_B64", None), self.assertRaisesRegex(
                ValueError, "trusted_runner_key_not_configured"):
            guard.verify_evidence(
                evidence, expected_source_commit="d" * 40, expected_family="kova-cosmo",
                expected_base_revision="e" * 40, expected_base_manifest_sha256="a" * 64,
                expected_adapter_sha256="b" * 64,
                expected_adapter_bundle_sha256="f" * 64,
                expected_runner_sha256="c" * 64,
                expected_case_id="case-1", expected_runtime_profile="light",
                expected_variant="trained_adapter", expected_case_category="generated_answer",
                expected_conversation_id="conversation-1", expected_session_id="session-1",
            )

    def test_public_evaluation_requires_source_pinned_full_adapter_bundles(self):
        weights = {
            family: hashlib.sha256(f"weights:{family}".encode()).hexdigest()
            for family in guard.FAMILIES
        }
        bundles = {
            family: hashlib.sha256(f"bundle:{family}".encode()).hexdigest()
            for family in guard.FAMILIES
        }
        families = {
            family: {"base_revision": guard.MODEL_SOURCE_REFERENCES[family].revision,
                     "base_manifest_sha256": guard.MANIFEST_SHA256[family],
                     "variant_adapters": {"configured_base": None,
                                         "trained_adapter": weights[family]},
                     "variant_adapter_bundles": {"configured_base": None,
                                                 "trained_adapter": bundles[family]}}
            for family in guard.FAMILIES
        }
        with patch.object(guard, "_trusted_reviewed_case_bindings", return_value={}):
            with self.assertRaisesRegex(ValueError, "trusted_adapter_bundle_not_pinned"):
                guard.validate_evidence_matrix(
                    [], source_commit="d" * 40, family_bindings=families,
                    case_bindings={}, review_verdicts={})
            registry = deepcopy(guard.CORE_SERVING)
            for candidate in registry["candidates"]:
                family = candidate["id"]
                candidate["adapter_sha256"] = weights[family]
                candidate["adapter_bundle_sha256"] = bundles[family]
            with patch.object(guard, "CORE_SERVING", registry):
                guard._require_trusted_family_adapter_bindings(families)
                changed_bundle = deepcopy(families)
                changed_bundle["kova-cosmo"]["variant_adapter_bundles"]["trained_adapter"] = "9" * 64
                with self.assertRaisesRegex(ValueError, "untrusted_variant_artifact_binding"):
                    guard.validate_evidence_matrix(
                        [], source_commit="d" * 40, family_bindings=changed_bundle,
                        case_bindings={}, review_verdicts={})
                changed_weights = deepcopy(families)
                changed_weights["kova-cosmo"]["variant_adapters"]["trained_adapter"] = "9" * 64
                with self.assertRaisesRegex(ValueError, "untrusted_variant_artifact_binding"):
                    guard.validate_evidence_matrix(
                        [], source_commit="d" * 40, family_bindings=changed_weights,
                        case_bindings={}, review_verdicts={})
                wrong_revision = deepcopy(families)
                wrong_revision["kova-cosmo"]["base_revision"] = families["kova-orion"]["base_revision"]
                with self.assertRaisesRegex(ValueError, "untrusted_base_revision_binding"):
                    guard.validate_evidence_matrix(
                        [], source_commit="d" * 40, family_bindings=wrong_revision,
                        case_bindings={}, review_verdicts={})
                wrong_manifest = deepcopy(families)
                wrong_manifest["kova-cosmo"]["base_manifest_sha256"] = families["kova-orion"][
                    "base_manifest_sha256"]
                with self.assertRaisesRegex(ValueError, "untrusted_base_manifest_binding"):
                    guard.validate_evidence_matrix(
                        [], source_commit="d" * 40, family_bindings=wrong_manifest,
                        case_bindings={}, review_verdicts={})
                wrong_registry_revision = deepcopy(registry)
                next(candidate for candidate in wrong_registry_revision["candidates"]
                     if candidate["id"] == "kova-cosmo")["revision"] = families["kova-orion"]["base_revision"]
                with patch.object(guard, "CORE_SERVING", wrong_registry_revision), \
                     self.assertRaisesRegex(ValueError, "untrusted_base_revision_binding"):
                    guard.validate_evidence_matrix(
                        [], source_commit="d" * 40, family_bindings=families,
                        case_bindings={}, review_verdicts={})
                with patch.object(guard, "sha256_file", return_value="0" * 64), \
                     self.assertRaisesRegex(ValueError, "trusted_base_manifest_source_mismatch"):
                    guard.validate_evidence_matrix(
                        [], source_commit="d" * 40, family_bindings=families,
                        case_bindings={}, review_verdicts={})
                duplicate_weights = deepcopy(registry)
                next(candidate for candidate in duplicate_weights["candidates"]
                     if candidate["id"] == "kova-orion")["adapter_sha256"] = weights["kova-cosmo"]
                duplicate_weight_bindings = deepcopy(families)
                duplicate_weight_bindings["kova-orion"]["variant_adapters"]["trained_adapter"] = weights[
                    "kova-cosmo"]
                with patch.object(guard, "CORE_SERVING", duplicate_weights), \
                     self.assertRaisesRegex(ValueError, "duplicate_trusted_family_adapter_artifact"):
                    guard.validate_evidence_matrix(
                        [], source_commit="d" * 40, family_bindings=duplicate_weight_bindings,
                        case_bindings={}, review_verdicts={})
                duplicate_bundles = deepcopy(registry)
                next(candidate for candidate in duplicate_bundles["candidates"]
                     if candidate["id"] == "kova-orion")["adapter_bundle_sha256"] = bundles["kova-cosmo"]
                duplicate_bundle_bindings = deepcopy(families)
                duplicate_bundle_bindings["kova-orion"]["variant_adapter_bundles"]["trained_adapter"] = bundles[
                    "kova-cosmo"]
                with patch.object(guard, "CORE_SERVING", duplicate_bundles), \
                     self.assertRaisesRegex(ValueError, "duplicate_trusted_family_adapter_artifact"):
                    guard.validate_evidence_matrix(
                        [], source_commit="d" * 40, family_bindings=duplicate_bundle_bindings,
                        case_bindings={}, review_verdicts={})

    def test_signed_evaluation_matrix_rejects_missing_replayed_and_substituted_cases(self):
        private_key = Ed25519PrivateKey.generate()
        trusted_public = base64.b64encode(private_key.public_key().public_bytes_raw()).decode("ascii")
        def bundle_digest(adapter_config: bytes) -> str:
            inventory = {"adapter_config.json": hashlib.sha256(adapter_config).hexdigest(),
                         "adapter_model.safetensors": "b" * 64}
            return hashlib.sha256(guard._canonical(inventory)).hexdigest()
        adapter_bundle_sha256 = bundle_digest(b'{"r":16}')
        changed_adapter_bundle_sha256 = bundle_digest(b'{"r":32}')
        self.assertNotEqual(adapter_bundle_sha256, changed_adapter_bundle_sha256)
        families = {
            family: {"base_revision": "e" * 40, "base_manifest_sha256": "a" * 64,
                     "variant_adapters": {"configured_base": None,
                                          "trained_adapter": "b" * 64},
                     "variant_adapter_bundles": {"configured_base": None,
                                                 "trained_adapter": adapter_bundle_sha256},
                     "runner_sha256": "c" * 64}
            for family in guard.FAMILIES
        }
        case_pins, envelopes = {}, []
        for family in sorted(guard.FAMILIES):
            for variant in guard.VARIANTS:
                for category, count in guard.CASE_CATEGORIES.items():
                    for index in range(1, count + 1):
                        case_id = f"{family}:{variant}:{category}:{index}"
                        profile = (guard.PROFILE_ORDER[index - 1]
                                   if category == "runtime_profile" else "light")
                        prompt = f"Question {case_id}"
                        case_pins[case_id] = {
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "runtime_profile": profile,
                            "conversation_id": f"conversation-{index}",
                            "session_id": f"session-{index}",
                            "expected_dimensions": list(guard._expected_case_dimensions(case_id)),
                        }
                        envelopes.append(guard.create_evidence(
                            source_commit="d" * 40, family=family,
                            base_revision="e" * 40, base_manifest_sha256="a" * 64,
                            adapter_sha256=None if variant == "configured_base" else "b" * 64,
                            adapter_bundle_sha256=(None if variant == "configured_base"
                                                   else adapter_bundle_sha256),
                            runner_sha256="c" * 64,
                            variant=variant, case_category=category, case_id=case_id,
                            prompt=prompt, answer="I am Kova.", runtime_profile=profile,
                            conversation_id=f"conversation-{index}", session_id=f"session-{index}",
                            dimensions=list(guard._expected_case_dimensions(case_id)),
                            private_key=private_key,
                        ))
        reviewer_key = Ed25519PrivateKey.generate()
        verdicts = {}
        for envelope in envelopes:
            payload = envelope["payload"]
            expected_dimensions = case_pins[payload["case_id"]]["expected_dimensions"]
            review = {"case_id": payload["case_id"], "source_commit": "d" * 40,
                      "reviewed_case_manifest_sha256": "f" * 64,
                      "adapter_bundle_sha256": payload["adapter_bundle_sha256"],
                      "prompt_sha256": payload["prompt_sha256"],
                      "answer_sha256": payload["answer_sha256"],
                      "expected_dimensions": expected_dimensions,
                      "dimension_verdicts": {name: True for name in expected_dimensions},
                      "passed": True}
            verdicts[payload["case_id"]] = {
                "payload": review,
                "signature_ed25519_b64": base64.b64encode(
                    reviewer_key.sign(guard._canonical(review))).decode("ascii")}
        review_args = {"review_verdicts": verdicts}
        reviewer_public = base64.b64encode(reviewer_key.public_key().public_bytes_raw()).decode("ascii")
        with self.assertRaisesRegex(ValueError, "reviewed_evaluation_case_manifest_not_pinned"):
            guard.validate_evidence_matrix(
                envelopes, source_commit="d" * 40, family_bindings=families,
                case_bindings=case_pins, **review_args)
        with patch.object(guard, "PINNED_REVIEWED_CASE_MANIFEST_SHA256", "f" * 64), \
             patch.object(guard, "REVIEWED_CASE_MANIFEST_PATH", Path("/nonexistent/reviewed-cases.json")), \
             self.assertRaisesRegex(ValueError, "invalid_reviewed_evaluation_case_manifest"):
            guard.validate_evidence_matrix(
                envelopes, source_commit="d" * 40, family_bindings=families,
                case_bindings=case_pins, **review_args)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "unapproved-cases.json"
            raw = b'{"schema_version":1,"status":"independently_reviewed","cases":[]}'
            manifest_path.write_bytes(raw)
            with patch.object(guard, "PINNED_REVIEWED_CASE_MANIFEST_SHA256", "f" * 64), \
                 patch.object(guard, "REVIEWED_CASE_MANIFEST_PATH", manifest_path), \
                 self.assertRaisesRegex(ValueError, "reviewed_evaluation_case_manifest_mismatch"):
                guard.validate_evidence_matrix(
                    envelopes, source_commit="d" * 40, family_bindings=families,
                    case_bindings=case_pins, **review_args)
            with patch.object(guard, "PINNED_REVIEWED_CASE_MANIFEST_SHA256", hashlib.sha256(raw).hexdigest()), \
                 patch.object(guard, "REVIEWED_CASE_MANIFEST_PATH", manifest_path), \
                 self.assertRaisesRegex(ValueError, "invalid_reviewed_evaluation_case_manifest"):
                guard.validate_evidence_matrix(
                    envelopes, source_commit="d" * 40, family_bindings=families,
                    case_bindings=case_pins, **review_args)
        def validate_synthetic_matrix(rows, **kwargs):
            return guard._validate_evidence_matrix_authenticated(
                rows, reviewed_case_manifest_sha256="f" * 64, **kwargs)

        with patch.object(guard, "PINNED_RUNNER_PUBLIC_KEY_B64", trusted_public), \
             patch.object(guard, "PINNED_REVIEWER_PUBLIC_KEY_B64", reviewer_public):
            with patch.object(guard, "PINNED_REVIEWER_PUBLIC_KEY_B64", None), self.assertRaisesRegex(
                    ValueError, "trusted_reviewer_key_not_configured"):
                validate_synthetic_matrix(envelopes, source_commit="d" * 40,
                                               family_bindings=families, case_bindings=case_pins,
                                               **review_args)
            with patch.object(guard, "PINNED_REVIEWER_PUBLIC_KEY_B64", trusted_public), self.assertRaisesRegex(
                    ValueError, "reviewer_must_be_independent"):
                validate_synthetic_matrix(envelopes, source_commit="d" * 40,
                                               family_bindings=families, case_bindings=case_pins,
                                               **review_args)
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
            alternate = (trusted_public[:-2] +
                         alphabet[alphabet.index(trusted_public[-2]) ^ 1] + "=")
            self.assertNotEqual(alternate, trusted_public)
            self.assertEqual(base64.b64decode(alternate, validate=True),
                             base64.b64decode(trusted_public, validate=True))
            with patch.object(guard, "PINNED_REVIEWER_PUBLIC_KEY_B64", alternate), self.assertRaisesRegex(
                    ValueError, "reviewer_must_be_independent"):
                validate_synthetic_matrix(envelopes, source_commit="d" * 40,
                                               family_bindings=families, case_bindings=case_pins,
                                               **review_args)
            self.assertEqual(set(validate_synthetic_matrix(
                envelopes, source_commit="d" * 40, family_bindings=families,
                case_bindings=case_pins, **review_args)), guard.REQUIRED_DIMENSIONS)
            missing_bundle_pin = deepcopy(families)
            del missing_bundle_pin["kova-cosmo"]["variant_adapter_bundles"]
            with self.assertRaisesRegex(ValueError, "invalid_variant_artifact_binding"):
                validate_synthetic_matrix(
                    envelopes, source_commit="d" * 40,
                    family_bindings=missing_bundle_pin, case_bindings=case_pins,
                    **review_args)
            changed_family = deepcopy(families)
            changed_family["kova-cosmo"]["variant_adapter_bundles"]["trained_adapter"] = (
                changed_adapter_bundle_sha256)
            with self.assertRaisesRegex(ValueError, "evidence_binding_mismatch:adapter_bundle_sha256"):
                validate_synthetic_matrix(
                    envelopes, source_commit="d" * 40,
                    family_bindings=changed_family, case_bindings=case_pins,
                    **review_args)
            changed_evidence = deepcopy(envelopes)
            for envelope in changed_evidence:
                payload = envelope["payload"]
                if payload["family"] == "kova-cosmo" and payload["variant"] == "trained_adapter":
                    self.assertEqual(payload["adapter_sha256"], "b" * 64)
                    payload["adapter_bundle_sha256"] = changed_adapter_bundle_sha256
                    envelope["signature_ed25519_b64"] = base64.b64encode(
                        private_key.sign(guard._canonical(payload))).decode("ascii")
            with self.assertRaisesRegex(ValueError, "failed_or_unbound_independent_review"):
                validate_synthetic_matrix(
                    changed_evidence, source_commit="d" * 40,
                    family_bindings=changed_family, case_bindings=case_pins,
                    **review_args)
            mislabeled = deepcopy(envelopes)
            claimed = mislabeled[0]["payload"]
            self.assertIn(":generated_answer:", claimed["case_id"])
            claimed["dimensions"] = ["cross_user_isolation", "exact_adapter_base_binding"]
            mislabeled[0]["signature_ed25519_b64"] = base64.b64encode(
                private_key.sign(guard._canonical(claimed))).decode("ascii")
            complicit_review = deepcopy(verdicts)
            review = complicit_review[claimed["case_id"]]["payload"]
            review["expected_dimensions"] = list(claimed["dimensions"])
            review["dimension_verdicts"] = {name: True for name in claimed["dimensions"]}
            complicit_review[claimed["case_id"]]["signature_ed25519_b64"] = base64.b64encode(
                reviewer_key.sign(guard._canonical(review))).decode("ascii")
            with self.assertRaisesRegex(ValueError, "evaluation_dimension_binding_mismatch"):
                validate_synthetic_matrix(
                    mislabeled, source_commit="d" * 40, family_bindings=families,
                    case_bindings=case_pins, review_verdicts=complicit_review)
            relabeled_pins = deepcopy(case_pins)
            relabeled_pins[claimed["case_id"]]["expected_dimensions"] = list(claimed["dimensions"])
            with self.assertRaisesRegex(ValueError, "untrusted_evaluation_dimensions"):
                validate_synthetic_matrix(
                    envelopes, source_commit="d" * 40, family_bindings=families,
                    case_bindings=relabeled_pins, **review_args)
            unsigned_dimensions = deepcopy(verdicts)
            unsigned_dimensions[claimed["case_id"]]["payload"].pop("dimension_verdicts")
            with self.assertRaisesRegex(ValueError, "failed_or_unbound_independent_review"):
                validate_synthetic_matrix(
                    envelopes, source_commit="d" * 40, family_bindings=families,
                    case_bindings=case_pins, review_verdicts=unsigned_dimensions)
            fake_base = deepcopy(envelopes)
            base_payload = fake_base[0]["payload"]
            self.assertEqual(base_payload["variant"], "configured_base")
            base_payload["adapter_sha256"] = "b" * 64
            fake_base[0]["signature_ed25519_b64"] = base64.b64encode(
                private_key.sign(guard._canonical(base_payload))).decode("ascii")
            with self.assertRaisesRegex(ValueError, "evidence_binding_mismatch:adapter_sha256"):
                validate_synthetic_matrix(fake_base, source_commit="d" * 40,
                                               family_bindings=families, case_bindings=case_pins,
                                               **review_args)
            failed = deepcopy(verdicts)
            first_case = envelopes[0]["payload"]["case_id"]
            failed[first_case]["payload"]["passed"] = False
            with self.assertRaisesRegex(ValueError, "failed_or_unbound_independent_review"):
                validate_synthetic_matrix(envelopes, source_commit="d" * 40,
                                               family_bindings=families, case_bindings=case_pins,
                                               review_verdicts=failed)
            forged = deepcopy(verdicts)
            forged[first_case]["signature_ed25519_b64"] = verdicts[envelopes[1]["payload"]["case_id"]][
                "signature_ed25519_b64"]
            with self.assertRaisesRegex(ValueError, "invalid_independent_review_signature"):
                validate_synthetic_matrix(envelopes, source_commit="d" * 40,
                                               family_bindings=families, case_bindings=case_pins,
                                               review_verdicts=forged)
            with self.assertRaisesRegex(ValueError, "incomplete_evaluation_matrix"):
                validate_synthetic_matrix(envelopes[:-1], source_commit="d" * 40,
                                               family_bindings=families, case_bindings=case_pins, **review_args)
            with self.assertRaisesRegex(ValueError, "duplicate_or_unconfigured_evaluation_case"):
                validate_synthetic_matrix(envelopes[:-1] + [envelopes[0]],
                                               source_commit="d" * 40,
                                               family_bindings=families, case_bindings=case_pins, **review_args)
            altered = deepcopy(envelopes)
            altered[0]["payload"]["prompt"] = "substituted"
            altered[0]["payload"]["prompt_sha256"] = hashlib.sha256(b"substituted").hexdigest()
            with self.assertRaisesRegex(ValueError, "invalid_evidence_signature"):
                validate_synthetic_matrix(altered, source_commit="d" * 40,
                                               family_bindings=families, case_bindings=case_pins, **review_args)
            changed_pins = deepcopy(case_pins)
            changed_pins[envelopes[0]["payload"]["case_id"]]["prompt_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "substituted_evaluation_prompt"):
                validate_synthetic_matrix(envelopes, source_commit="d" * 40,
                                               family_bindings=families, case_bindings=changed_pins, **review_args)
            signed_substitution = deepcopy(envelopes)
            first = signed_substitution[0]["payload"]
            first["prompt"] = "A different signed case under the same ID"
            first["prompt_sha256"] = hashlib.sha256(first["prompt"].encode()).hexdigest()
            signed_substitution[0]["signature_ed25519_b64"] = base64.b64encode(
                private_key.sign(guard._canonical(first))).decode("ascii")
            with self.assertRaisesRegex(ValueError, "substituted_evaluation_prompt"):
                validate_synthetic_matrix(signed_substitution, source_commit="d" * 40,
                                               family_bindings=families, case_bindings=case_pins, **review_args)

    def test_authoritative_policy_cannot_drift(self):
        self.assertEqual(
            validate_policy_drift()["status"],
            "authoritative_three_family_policy_no_drift",
        )

    def test_archived_routing_sources_cannot_reenter_runtime(self):
        self.assertEqual(validate_policy_drift()["status"],
                         "authoritative_three_family_policy_no_drift")
        original = Path.read_text
        for relative in ("core/adapter.py", "core/current_candidates.py",
                         "worker/handler.py", "worker/model_artifact.py", "release/rollout.py"):
            def changed(path, *args, **kwargs):
                source = original(path, *args, **kwargs)
                return source + "\n# loads core-serving.v1.json\n" if str(path).endswith(relative) else source
            with self.subTest(relative=relative), patch.object(Path, "read_text", changed):
                with self.assertRaisesRegex(ValueError, "runtime_reads_archived_routing"):
                    validate_policy_drift()

    def test_complete_source_contract_is_valid_dataset_approved_execution_blocked(self):
        report = contract.validate()
        self.assertEqual(report["families"], list(contract.FAMILIES))
        self.assertEqual(report["chat_routes"], {"free": 1, "plus": 6, "pro": 12})
        self.assertEqual(report["work_routes"], {"free": 0, "plus": 18, "pro": 18})
        self.assertTrue(report["dataset"]["approval_complete"])
        self.assertTrue(report["all_paid_and_production_gates_closed"])
        self.assertEqual(report["provider_calls_made"], 0)

    def test_policy_has_no_nova_chat_or_separate_8b_slot(self):
        value = contract.load_json(contract.ROOT / "config/current-product-policy.v3.json")
        self.assertEqual(value["entitlements"]["chat"]["plus"]["nova"], [])
        self.assertEqual(value["entitlements"]["chat"]["pro"]["nova"], [])
        raw = json.dumps(value)
        self.assertNotIn("Qwen3-8B", raw)
        self.assertFalse(value["processing_levels_are_separate_models"])

    def test_all_active_authorization_gates_are_false(self):
        for name in (
            "current-product-policy.v3.json", "kova-private-lineage.v1.json",
            "kova-three-family-dataset.v2.json", "kova-three-family-pilot.v1.json",
            "kova-three-family-cost-guard.v1.json", "kova-three-family-lifecycle.v1.json",
            "kova-three-family-evaluation.v1.json", "kova-three-family-operator-plan.v1.json",
        ):
            with self.subTest(name=name):
                contract._all_false(contract.load_json(contract.ROOT / "config" / name), name)

    def test_invalid_utf8_and_duplicate_json_keys_fail_closed(self):
        for raw in (b'\xff', b'{"x":1,"x":2}', b'{"x":NaN}', b''):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "bad.json"
                path.write_bytes(raw)
                with self.assertRaises(contract.ContractError):
                    contract.load_json(path)

    def test_manifest_inventory_is_immutable_and_complete(self):
        totals = contract.validate_lineage_and_manifests()
        self.assertEqual(totals["kova-cosmo"], 1519207673)
        self.assertEqual(totals["kova-orion"], 4079448540)
        self.assertEqual(totals["kova-nova"], 8060925056)

    def test_snapshot_rejects_missing_extra_and_modified_files(self):
        manifest = {"files": [{"path": "config.json", "bytes": 2,
                                "sha256": hashlib.sha256(b"{}").hexdigest()}]}
        lineage = {"families": {family: {"manifest": "ignored"} for family in contract.FAMILIES}}
        def fake_load(path, **_):
            return lineage if path.name == "kova-private-lineage.v1.json" else manifest
        with tempfile.TemporaryDirectory() as folder, patch.object(contract, "load_json", fake_load), patch.object(
                contract, "_pinned_manifest", return_value=manifest):
            root = Path(folder)
            (root / "config.json").write_bytes(b"{}")
            contract.verify_snapshot("kova-cosmo", root)
            (root / "extra").write_bytes(b"x")
            with self.assertRaises(contract.ContractError):
                contract.verify_snapshot("kova-cosmo", root)
            (root / "extra").unlink()
            (root / "config.json").write_bytes(b"[]")
            with self.assertRaises(contract.ContractError):
                contract.verify_snapshot("kova-cosmo", root)
            (root / "config.json").unlink()
            with self.assertRaises(contract.ContractError):
                contract.verify_snapshot("kova-cosmo", root)

    def test_snapshot_rejects_links_and_nonregular_entries(self):
        from training.snapshot_verifier import verify_snapshot
        manifest = {"files": [{"path": "config.json", "bytes": 2,
                               "sha256": hashlib.sha256(b"{}").hexdigest()}]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "config.json"
            target.write_bytes(b"{}")
            self.assertTrue(verify_snapshot(root, manifest)["verified"])
            (root / "extra").symlink_to(target)
            with self.assertRaises(ValueError):
                verify_snapshot(root, manifest)
            (root / "extra").unlink()
            os.link(target, root / "extra")
            with self.assertRaises(ValueError):
                verify_snapshot(root, manifest)
            (root / "extra").unlink()
            target.unlink()
            target.symlink_to(root / "absent")
            with self.assertRaises(ValueError):
                verify_snapshot(root, manifest)

    def test_snapshot_cli_requires_independent_complete_manifest_pins(self):
        from training import snapshot_verifier
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "config"
            config.mkdir()
            paths = ("config/kova-private-lineage.v1.json", *contract.MANIFEST_PATHS.values())
            for relative in paths:
                (root / relative).write_bytes((contract.ROOT / relative).read_bytes())
            result = {"verified": True, "files": 0}
            with patch.object(contract, "ROOT", root), \
                    patch.object(snapshot_verifier, "verify_snapshot", return_value=result) as verify, \
                    patch("builtins.print"):
                self.assertEqual(snapshot_verifier.main(["--all-families", "--root", str(root)]), 0)
                self.assertEqual(verify.call_count, 3)
                verify.reset_mock()
                manifest = config / "qwen3-0.6b-download-manifest.v1.json"
                manifest.write_bytes(manifest.read_bytes() + b" ")  # Still valid JSON; bytes no longer pinned.
                for args in (["--family", "kova-cosmo"], ["--all-families"]):
                    with self.subTest(args=args), self.assertRaisesRegex(
                            contract.ContractError, "complete manifest digest mismatch"):
                        snapshot_verifier.main([*args, "--root", str(root)])
                verify.assert_not_called()

    def test_six_dollar_bootstrap_uses_live_rate_and_60_second_rounding(self):
        self.assertLessEqual(contract.admit_bootstrap(Decimal("0.526")), Decimal("6.0000"))
        with self.assertRaisesRegex(contract.ContractError, "six-dollar"):
            contract.admit_bootstrap(Decimal("1.00"))
        with self.assertRaises(contract.ContractError):
            contract.admit_bootstrap(Decimal("-0.01"))
        with self.assertRaises(contract.ContractError):
            contract.admit_bootstrap(Decimal("NaN"))

    def test_conditional_cosmo_only_limit_is_enforced_separately(self):
        self.assertEqual(contract.admit_conditional_cosmo_pilot(Decimal("0.526")),
                         Decimal("3.2020"))
        self.assertEqual(contract.admit_conditional_cosmo_pilot(Decimal("0.575")),
                         Decimal("3.3000"))
        with self.assertRaisesRegex(contract.ContractError, "Cosmo worst case"):
            contract.admit_conditional_cosmo_pilot(Decimal("0.5751"))
        self.assertLessEqual(contract.admit_bootstrap(Decimal("0.60")), Decimal("6.0000"))
        with self.assertRaisesRegex(contract.ContractError, "Cosmo worst case"):
            contract.admit_conditional_cosmo_pilot(Decimal("0.60"))
        original = contract.load_json
        cost = original(contract.ROOT / "config/kova-three-family-cost-guard.v1.json")
        for update in ({"hard_ceiling_usd": "6.0000"},
                       {"maximum_allocation_seconds": 21600},
                       {"maximum_compute_reservation_usd": "1.2500"},
                       {"other_families_authorized": True}):
            altered = deepcopy(cost)
            altered["conditional_cosmo_only_pilot"].update(update)
            def fake_load(path, **kwargs):
                return altered if path.name == "kova-three-family-cost-guard.v1.json" else original(path, **kwargs)
            with self.subTest(update=update), patch.object(contract, "load_json", side_effect=fake_load):
                with self.assertRaisesRegex(contract.ContractError, "Cosmo-only owner ceiling"):
                    contract.admit_conditional_cosmo_pilot(Decimal("0.526"))

    def test_cost_guard_rejects_understated_or_nonfinite_ancillary_bounds(self):
        original = contract.load_json
        cost = original(contract.ROOT / "config/kova-three-family-cost-guard.v1.json")
        for change in (lambda cfg: cfg["category_upper_bounds"].update(managed_disks="-1"),
                       lambda cfg: cfg["category_upper_bounds"].update(managed_disks="NaN"),
                       lambda cfg: cfg["category_upper_bounds"].pop("managed_disks"),
                       lambda cfg: cfg["meter_categories"].pop(),
                       lambda cfg: cfg.update(independent_signed_admission_required=False),
                       lambda cfg: cfg.update(post_run_cost_evidence_required=False),
                       lambda cfg: cfg.update(zero_residual_billable_resources_required=False)):
            altered = deepcopy(cost)
            change(altered)
            def fake_load(path, **kwargs):
                return altered if path.name == "kova-three-family-cost-guard.v1.json" else original(path, **kwargs)
            with patch.object(contract, "load_json", side_effect=fake_load), self.assertRaises(contract.ContractError):
                contract.admit_bootstrap(Decimal("0.526"))

    def test_captured_live_price_is_parsed_and_admitted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "price.json"
            path.write_text(json.dumps({"Items": [{
                "armSkuName": "Standard_NC4as_T4_v3", "armRegionName": "eastus",
                "currencyCode": "USD", "unitOfMeasure": "1 Hour", "retailPrice": 0.526,
                "unitPrice": 0.526, "tierMinimumUnits": 0, "type": "Consumption",
                "serviceName": "Virtual Machines", "serviceFamily": "Compute",
                "productName": "Virtual Machines NCasT4 v3 Series",
                "skuName": "NC4as T4 v3", "meterName": "NC4as T4 v3",
                "isPrimaryMeterRegion": True,
            }]}))
            admitted = contract.validate_live_price_evidence(path)
            self.assertEqual(admitted["status"], "live_price_admitted")
            self.assertEqual(admitted["conditional_cosmo_only_worst_case_usd"], "3.2020")
            self.assertTrue(admitted["conditional_cosmo_only_eligible"])
            self.assertEqual(contract.validate_live_price_evidence(
                path, admission_scope="cosmo-only")["hard_ceiling_usd"], "3.3000")
            value = json.loads(path.read_text())
            value["Items"][0]["retailPrice"] = 0.60
            value["Items"][0]["unitPrice"] = 0.60
            path.write_text(json.dumps(value))
            full_plan = contract.validate_live_price_evidence(path)
            self.assertEqual(full_plan["admission_scope"], "three-family")
            self.assertFalse(full_plan["conditional_cosmo_only_eligible"])
            with self.assertRaisesRegex(contract.ContractError, "Cosmo worst case"):
                contract.validate_live_price_evidence(path, admission_scope="cosmo-only")
            with self.assertRaisesRegex(contract.ContractError, "scope"):
                contract.validate_live_price_evidence(path, admission_scope="unspecified")
            value = json.loads(path.read_text())
            value["Items"][0]["retailPrice"] = 1.0
            value["Items"][0]["unitPrice"] = 1.0
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(contract.ContractError, "six-dollar"):
                contract.validate_live_price_evidence(path)

    def test_live_price_rejects_discounted_secondary_incomplete_and_ambiguous_meters(self):
        valid = {
            "armSkuName": "Standard_NC4as_T4_v3", "armRegionName": "eastus",
            "currencyCode": "USD", "unitOfMeasure": "1 Hour",
            "retailPrice": 0.526, "unitPrice": 0.526, "tierMinimumUnits": 0,
            "type": "Consumption", "serviceName": "Virtual Machines",
            "serviceFamily": "Compute", "productName": "Virtual Machines NCasT4 v3 Series",
            "skuName": "NC4as T4 v3", "meterName": "NC4as T4 v3",
            "isPrimaryMeterRegion": True,
        }
        mutations = (
            {"skuName": "NC4as T4 v3 Spot", "meterName": "NC4as T4 v3 Spot"},
            {"skuName": "NC4as T4 v3 Low Priority", "meterName": "NC4as T4 v3 Low Priority"},
            {"productName": "Virtual Machines NCasT4 v3 Series Windows"},
            {"type": "DevTestConsumption"}, {"type": "Reservation"},
            {"isPrimaryMeterRegion": False}, {"currencyCode": "EUR"},
            {"unitOfMeasure": "1 Month"}, {"armRegionName": "westus"},
            {"unitPrice": 0.105}, {"tierMinimumUnits": 1},
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "price.json"
            for changed in mutations:
                with self.subTest(changed=changed):
                    path.write_text(json.dumps({"Items": [{**valid, **changed}]}))
                    with self.assertRaises(contract.ContractError):
                        contract.validate_live_price_evidence(path)
            for response in (
                {"Items": [valid, valid]},
                {"Items": [valid], "NextPageLink": "https://prices.azure.com/next"},
                {"Items": [valid], "Count": 2},
                {"Items": [valid, "malformed"]},
            ):
                with self.subTest(response=response):
                    path.write_text(json.dumps(response))
                    with self.assertRaises(contract.ContractError):
                        contract.validate_live_price_evidence(path)

    def test_approval_pin_rejects_coordinated_dataset_contract_review_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config").mkdir()
            (root / "data").mkdir()
            source = contract.ROOT
            cfg = contract.load_json(source / "config/kova-three-family-dataset.v2.json")
            data_path = root / cfg["dataset_path"]
            review_path = root / cfg["review_path"]
            data_path.write_bytes((source / cfg["dataset_path"]).read_bytes() + b"\n")
            review = contract.load_json(source / cfg["review_path"])
            cfg["dataset_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
            cfg["approval"]["approved_dataset_sha256"] = cfg["dataset_sha256"]
            review["approved_dataset_sha256"] = cfg["dataset_sha256"]
            review["current_reconstructed_dataset_sha256"] = cfg["dataset_sha256"]
            review_path.write_text(json.dumps(review))
            (root / "config/kova-three-family-dataset.v2.json").write_text(json.dumps(cfg))
            with patch.object(contract, "ROOT", root), self.assertRaises(contract.ContractError):
                contract.validate_dataset()

    def test_complete_manifest_pins_reject_coordinated_substitution_for_each_family(self):
        original = contract.load_json
        for family in contract.FAMILIES:
            lineage = original(contract.ROOT / "config/kova-private-lineage.v1.json")["families"][family]
            self.assertEqual(contract._pinned_manifest(family, lineage)["revision"],
                             lineage["immutable_revision"])
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                target = root / contract.MANIFEST_PATHS[family]
                target.parent.mkdir(parents=True)
                manifest = original(contract.ROOT / contract.MANIFEST_PATHS[family])
                manifest["files"][0]["sha256"] = "0" * 64
                target.write_text(json.dumps(manifest))
                with self.subTest(family=family), patch.object(contract, "ROOT", root), self.assertRaisesRegex(
                        contract.ContractError, "complete manifest digest mismatch"):
                    contract._pinned_manifest(family, lineage)

    def test_qlora_recipe_rejects_swapped_manifest_and_dataset_for_each_family(self):
        original = contract.load_json
        for family in contract.FAMILIES:
            name = f"{family}-qlora.v1.json"
            source = original(contract.ROOT / "config" / name)
            for field, bad in (
                ("manifest", contract.MANIFEST_PATHS[next(f for f in contract.FAMILIES if f != family)]),
                ("dataset", "config/kova-cosmo-sft.v1.json"),
                ("output_directory", f"outputs/{next(f for f in contract.FAMILIES if f != family)}"),
                ("upstream_repository", "Qwen/Other"),
                ("immutable_revision", "0" * 40),
            ):
                changed = deepcopy(source)
                changed[field] = bad
                def fake_load(path, **kwargs):
                    return changed if path.name == name else original(path, **kwargs)
                with self.subTest(family=family, field=field), patch.object(
                        contract, "load_json", side_effect=fake_load), self.assertRaises(contract.ContractError):
                    contract.validate_training()

    def test_qlora_recipe_rejects_changed_hyperparameters(self):
        original = Path.read_bytes
        for family in contract.FAMILIES:
            name = f"{family}-qlora.v1.json"
            def changed(path):
                raw = original(path)
                if path.name == name:
                    cfg = json.loads(raw)
                    cfg["training"]["maximum_optimizer_steps"] = 6
                    return json.dumps(cfg).encode()
                return raw
            with self.subTest(family=family), patch.object(Path, "read_bytes", changed), self.assertRaisesRegex(
                    contract.ContractError, "complete QLoRA recipe digest mismatch"):
                contract.validate_training()

    def test_json_numeric_overflow_is_rejected_at_any_depth(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "value.json"
            for value in ("1e999", "-1e999", '{"deep":[{"value":1e999}]}',
                          '{"deep":[{"value":-1e999}]}'):
                path.write_text(value)
                with self.subTest(value=value), self.assertRaises(contract.ContractError):
                    contract.load_json(path)
            path.write_text('{"value":1.25}')
            self.assertEqual(contract.load_json(path)["value"], 1.25)

    def test_familyless_grants_cannot_advance_ledger(self):
        state = {"sequence": 0, "terminal": False, "family_order": [], "events": []}
        for family in (None, "", "other"):
            for event in ({"kind": "training_grant"}, {"kind": "training_grant", "family": family}):
                with self.subTest(event=event), self.assertRaises(contract.ContractError):
                    contract.append_ledger_event(state, event, expected_sequence=0)
                self.assertEqual(state["sequence"], 0)
                self.assertEqual(state["family_order"], [])

    def test_training_stack_versions_are_bound_to_hash_locked_requirements(self):
        base = contract.load_json(contract.ROOT / "config/kova-three-family-training-stack.v1.json")
        original_load = contract.load_json
        changes = []
        changed = deepcopy(base); changed["python"] = "3.11"; changes.append(changed)
        changed = deepcopy(base); changed["cuda"] = "12.7"; changes.append(changed)
        changed = deepcopy(base); changed["packages"]["torch"] = "2.7.0"; changes.append(changed)
        for changed in changes:
            def fake_load(path, **kwargs):
                if path.name == "kova-three-family-training-stack.v1.json":
                    return changed
                return original_load(path, **kwargs)
            with self.subTest(changed=changed), patch.object(contract, "load_json", side_effect=fake_load):
                with self.assertRaises(contract.ContractError):
                    contract.validate_training()

    def test_t4_probe_evidence_is_actually_consumed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "probe.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "device_name": "NVIDIA T4",
                "compute_capability": "7.5",
                "cuda_version": "12.8",
                "bitsandbytes_four_bit_available": True,
                "available_vram_bytes": 16000000000,
                "free_disk_bytes": 40000000000,
                "family_probes": {
                    "kova-cosmo": {"peak_vram_bytes": 6000000000, "maximum_sequence_length": 1024, "probe_passed": True},
                    "kova-orion": {"peak_vram_bytes": 9000000000, "maximum_sequence_length": 1024, "probe_passed": True},
                    "kova-nova": {"peak_vram_bytes": 15000000000, "maximum_sequence_length": 768, "probe_passed": True},
                },
            }))
            self.assertEqual(contract.validate_probe_evidence(path)["status"], "t4_probe_evidence_valid")
            value = json.loads(path.read_text())
            value["device_name"] = "not-a-t4"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(contract.ContractError, "unexpected GPU"):
                contract.validate_probe_evidence(path)

    def test_ledger_assigns_sequence_and_rejects_stale_duplicate_out_of_order(self):
        state = {"sequence": 0, "terminal": False, "family_order": [], "events": []}
        state = contract.append_ledger_event(state, {"kind": "watchdog_health"}, expected_sequence=0)
        self.assertEqual(state["events"][-1]["sequence"], 1)
        with self.assertRaises(contract.ContractError):
            contract.append_ledger_event(state, {"kind": "training_grant", "family": "kova-orion"},
                                         expected_sequence=1)
        with self.assertRaisesRegex(contract.ContractError, "cost admission required"):
            contract.append_ledger_event(state, {"kind": "training_grant", "family": "kova-cosmo"},
                                         expected_sequence=1)
        state = contract.append_ledger_event(state, {"kind": "cost_admission"}, expected_sequence=1)
        before_grant = deepcopy(state)
        state = contract.append_ledger_event(state, {"kind": "training_grant", "family": "kova-cosmo"},
                                             expected_sequence=2)
        self.assertEqual(before_grant["family_order"], [])
        self.assertEqual(state["family_order"], ["kova-cosmo"])
        with self.assertRaises(contract.ContractError):
            contract.append_ledger_event(state, {"kind": "training_grant", "family": "kova-cosmo"},
                                         expected_sequence=3)
        with self.assertRaisesRegex(contract.ContractError, "previous family preservation required"):
            contract.append_ledger_event(state, {"kind": "training_grant", "family": "kova-orion"},
                                         expected_sequence=3)
        state = contract.append_ledger_event(state, {"kind": "family_preserved", "family": "kova-cosmo"},
                                             expected_sequence=3)
        state = contract.append_ledger_event(state, {"kind": "training_grant", "family": "kova-orion"},
                                             expected_sequence=4)
        self.assertEqual(state["family_order"], ["kova-cosmo", "kova-orion"])
        with self.assertRaises(contract.ContractError):
            contract.append_ledger_event(state, {"kind": "cost_admission"}, expected_sequence=1)
        with self.assertRaises(contract.ContractError):
            contract.append_ledger_event(state, {"kind": "cost_admission", "sequence": 3},
                                         expected_sequence=2)

    def test_terminal_ledger_rejects_post_cleanup_events(self):
        state = {"sequence": 0, "terminal": False, "family_order": [], "events": []}
        state = contract.append_ledger_event(state, {"kind": "cleanup_terminal"}, expected_sequence=0)
        self.assertTrue(state["terminal"])
        with self.assertRaises(contract.ContractError):
            contract.append_ledger_event(state, {"kind": "watchdog_health"}, expected_sequence=1)

    def test_runner_dry_run_uses_existing_contract_and_blocks_execution(self):
        from training.three_family_runner import main as runner_main
        from io import StringIO
        from contextlib import redirect_stdout
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(runner_main(["--family", "kova-cosmo", "--dry-run"]), 0)
        self.assertFalse(json.loads(output.getvalue())["training_started"])
        with self.assertRaises(SystemExit):
            runner_main(["--family", "kova-cosmo", "--execute"])

    def test_infrastructure_has_no_public_ip_and_source_gates_false(self):
        vm = (contract.ROOT / "infra/three-family-pilot-vm.bicep").read_text()
        watchdog = (contract.ROOT / "infra/three-family-watchdog.bicep").read_text()
        self.assertNotIn("publicIPAddresses", vm)
        self.assertIn("Standard_NC4as_T4_v3", vm)
        self.assertIn("disablePasswordAuthentication: true", vm)
        self.assertIn("Microsoft.HpcCompute", vm)
        self.assertIn("NvidiaGpuDriverLinux", vm)
        self.assertIn("version: ubuntuImageVersion", vm)
        self.assertIn("@allowed(['24.04.202609040'])", vm)
        self.assertNotIn("version: 'latest'", vm)
        self.assertIn("ubuntuImageVersion=${PINNED_UBUNTU_IMAGE_VERSION}", command_plan()[3]["argv"])
        self.assertIn("enableAutomaticUpgrade: false", vm)
        self.assertIn("frequency: 'Minute'", watchdog)
        self.assertIn("param pilotSuffix string", watchdog)
        self.assertIn("kova-t4-${pilotSuffix}/deallocate", watchdog)
        self.assertIn("delete_pilot_group", watchdog)
        self.assertIn("allowSharedKeyAccess: false", watchdog)

    def test_tampering_with_a_safety_gate_is_rejected(self):
        value = contract.load_json(contract.ROOT / "config/kova-three-family-pilot.v1.json")
        changed = deepcopy(value)
        changed["spending_authorized"] = True
        with self.assertRaisesRegex(contract.ContractError, "open safety gate"):
            contract._all_false(changed, "pilot")


if __name__ == "__main__":
    unittest.main()
