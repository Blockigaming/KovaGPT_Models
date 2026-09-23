"""Cryptographic evidence envelopes for three-family adapter evaluation."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"kova-cosmo", "kova-orion", "kova-nova"}
PROFILES = {"light", "medium", "high", "extra-high", "max", "ultra"}
VARIANTS = ("configured_base", "trained_adapter")
CASE_CATEGORIES = {"generated_answer": 7, "isolation_integrity": 7, "runtime_profile": 6}
PROFILE_ORDER = ("light", "medium", "high", "extra-high", "max", "ultra")
SCHEMA = "kova-three-family-evaluation-evidence.v1"
_CONFIG = json.loads((ROOT / "config/kova-three-family-evaluation.v1.json").read_text(encoding="utf-8"))
PINNED_RUNNER_PUBLIC_KEY_B64 = _CONFIG["answer_binding"]["runner_public_key_ed25519_b64"]
PINNED_REVIEWER_PUBLIC_KEY_B64 = _CONFIG["reviewer_public_key_ed25519_b64"]
# No independently reviewed 120-case corpus or structured isolation observations
# exist yet. A future source review must pin exact manifest bytes before the
# public matrix verifier can accept any generated evidence.
PINNED_REVIEWED_CASE_MANIFEST_SHA256 = None
REVIEWED_CASE_MANIFEST_PATH = ROOT / "config/kova-three-family-reviewed-evaluation-cases.v1.json"
_REQUIRED_DIMENSIONS = _CONFIG.get("required_dimensions")
if (
    type(_REQUIRED_DIMENSIONS) is not list
    or len(_REQUIRED_DIMENSIONS) != 12
    or any(type(value) is not str or not value for value in _REQUIRED_DIMENSIONS)
    or len(set(_REQUIRED_DIMENSIONS)) != len(_REQUIRED_DIMENSIONS)
):
    raise RuntimeError("invalid required evaluation dimensions")
REQUIRED_DIMENSIONS = frozenset(_REQUIRED_DIMENSIONS)
# Case meanings are fixed independently of runner evidence and supplied pins.
# The reviewer must assess the actual prompt/answer against these dimensions.
CASE_DIMENSIONS_BY_CATEGORY = {
    "generated_answer": (
        ("kova_identity_consistency",),
        ("kovagpt_ownership_behavior",),
        ("ordinary_upstream_name_suppression",),
        ("instruction_following",),
        ("response_quality",),
        ("refusal_and_safety",),
        ("instruction_following", "response_quality"),
    ),
    "isolation_integrity": (
        ("cross_user_isolation",),
        ("conversation_session_isolation",),
        ("tampered_evidence_rejection",),
        ("exact_adapter_base_binding",),
        ("family_separation",),
        ("cross_user_isolation", "conversation_session_isolation"),
        ("tampered_evidence_rejection", "exact_adapter_base_binding"),
    ),
    "runtime_profile": (("runtime_profile_behavior",),) * 6,
}
if (set(CASE_DIMENSIONS_BY_CATEGORY) != set(CASE_CATEGORIES)
        or any(len(CASE_DIMENSIONS_BY_CATEGORY[category]) != count
               for category, count in CASE_CATEGORIES.items())
        or {dimension for cases in CASE_DIMENSIONS_BY_CATEGORY.values()
            for dimensions in cases for dimension in dimensions} != REQUIRED_DIMENSIONS):
    raise RuntimeError("invalid configured evaluation case dimensions")


def _canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8", "strict")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "strict")).hexdigest()


def _hex(value: str, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _identifier(value: str) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 128 and value.strip() == value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_dimensions(dimensions) -> tuple[str, ...]:
    if type(dimensions) is not list or not dimensions:
        raise ValueError("invalid_evaluation_dimensions")
    if any(type(value) is not str or value not in REQUIRED_DIMENSIONS for value in dimensions):
        raise ValueError("invalid_evaluation_dimensions")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("duplicate_evaluation_dimensions")
    return tuple(sorted(dimensions))


def _expected_case_dimensions(case_id: str) -> tuple[str, ...]:
    if type(case_id) is not str:
        raise ValueError("invalid_evaluation_case")
    parts = case_id.split(":")
    if (len(parts) != 4 or parts[0] not in FAMILIES or parts[1] not in VARIANTS
            or parts[2] not in CASE_CATEGORIES or not parts[3].isdecimal()):
        raise ValueError("invalid_evaluation_case")
    index = int(parts[3])
    if str(index) != parts[3] or not 1 <= index <= CASE_CATEGORIES[parts[2]]:
        raise ValueError("invalid_evaluation_case")
    return tuple(sorted(CASE_DIMENSIONS_BY_CATEGORY[parts[2]][index - 1]))


def _unique_json_object(pairs):
    value = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("duplicate_reviewed_case_manifest_key")
        value[name] = item
    return value


def _trusted_reviewed_case_bindings() -> dict:
    """Read an exact, independently reviewed source manifest; absent today."""
    if not _hex(PINNED_REVIEWED_CASE_MANIFEST_SHA256, 64):
        raise ValueError("reviewed_evaluation_case_manifest_not_pinned")
    try:
        raw = REVIEWED_CASE_MANIFEST_PATH.read_bytes()
        if not 0 < len(raw) <= 512 * 1024 or hashlib.sha256(raw).hexdigest() != PINNED_REVIEWED_CASE_MANIFEST_SHA256:
            raise ValueError("reviewed_evaluation_case_manifest_mismatch")
        manifest = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_unique_json_object)
        if (type(manifest) is not dict or set(manifest) != {"schema_version", "status", "cases"}
                or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
                or manifest["status"] != "independently_reviewed"
                or type(manifest["cases"]) is not list or len(manifest["cases"]) != 120):
            raise ValueError("invalid_reviewed_evaluation_case_manifest")
        expected_ids = {
            f"{family}:{variant}:{category}:{index}"
            for family in FAMILIES for variant in VARIANTS
            for category, count in CASE_CATEGORIES.items() for index in range(1, count + 1)
        }
        bindings = {}
        for case in manifest["cases"]:
            if (type(case) is not dict or set(case) != {"case_id", "prompt_sha256",
                    "expected_dimensions", "runtime_profile", "conversation_id", "session_id"}):
                raise ValueError("invalid_reviewed_evaluation_case_manifest")
            case_id = case["case_id"]
            if (type(case_id) is not str or case_id not in expected_ids or case_id in bindings
                    or not _hex(case["prompt_sha256"], 64)
                    or case["runtime_profile"] not in PROFILES
                    or not all(_identifier(case[field]) for field in ("conversation_id", "session_id"))
                    or _validated_dimensions(case["expected_dimensions"]) != _expected_case_dimensions(case_id)):
                raise ValueError("invalid_reviewed_evaluation_case_manifest")
            category = case_id.split(":")[2]
            if category == "runtime_profile" and case["runtime_profile"] != PROFILE_ORDER[int(case_id.split(":")[3]) - 1]:
                raise ValueError("invalid_reviewed_evaluation_case_manifest")
            bindings[case_id] = {key: value for key, value in case.items() if key != "case_id"}
        if set(bindings) != expected_ids:
            raise ValueError("incomplete_reviewed_evaluation_case_manifest")
        return bindings
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError, RecursionError):
        raise ValueError("invalid_reviewed_evaluation_case_manifest") from None


def validate_verified_dimension_coverage(payloads: list[dict]) -> tuple[str, ...]:
    """Require source-assigned dimensions in every complete family/variant cell.

    Test fixtures may use synthetic prompts. A real independent reviewer must
    inspect each pinned prompt and answer before signing its dimensions. Those
    signatures attest human assessment, not independent provider isolation
    telemetry; no paid evaluation or release is enabled by this source check.
    """
    if type(payloads) is not list or not payloads:
        raise ValueError("missing_evaluation_evidence")
    covered = set()
    covered_by_cell = {(family, variant): set() for family in FAMILIES for variant in VARIANTS}
    observed = set()
    for payload in payloads:
        if type(payload) is not dict:
            raise ValueError("invalid_evaluation_payload")
        family = payload.get("family")
        variant = payload.get("variant")
        category = payload.get("case_category")
        case_id = payload.get("case_id")
        profile = payload.get("runtime_profile")
        if (type(family) is not str or family not in FAMILIES
                or type(variant) is not str or variant not in VARIANTS
                or type(category) is not str or category not in CASE_CATEGORIES):
            raise ValueError("invalid_evaluation_matrix_cell")
        if type(profile) is not str or profile not in PROFILES:
            raise ValueError("invalid_evaluation_profile")
        index = (PROFILE_ORDER.index(profile) + 1) if category == "runtime_profile" else None
        valid_ids = {f"{family}:{variant}:{category}:{n}" for n in range(1, CASE_CATEGORIES[category] + 1)}
        if type(case_id) is not str or case_id not in valid_ids:
            raise ValueError("invalid_evaluation_case")
        if index is not None and case_id != f"{family}:{variant}:{category}:{index}":
            raise ValueError("runtime_profile_case_mismatch")
        key = (family, variant, category, case_id, profile if category == "runtime_profile" else "")
        if key in observed or any(row[:4] == key[:4] for row in observed):
            raise ValueError("duplicate_evaluation_case")
        observed.add(key)
        expected_dimensions = _expected_case_dimensions(case_id)
        if _validated_dimensions(payload.get("dimensions")) != expected_dimensions:
            raise ValueError("evaluation_dimension_binding_mismatch")
        covered.update(expected_dimensions)
        covered_by_cell[(family, variant)].update(expected_dimensions)
    expected = {
        (family, variant, category, f"{family}:{variant}:{category}:{n}")
        for family in FAMILIES for variant in VARIANTS
        for category, count in CASE_CATEGORIES.items() for n in range(1, count + 1)
    }
    if {row[:4] for row in observed} != expected or len(payloads) != 120:
        raise ValueError("incomplete_evaluation_matrix")
    missing = REQUIRED_DIMENSIONS - covered
    if missing:
        raise ValueError("missing_required_dimensions:" + ",".join(sorted(missing)))
    if any(dimensions != REQUIRED_DIMENSIONS for dimensions in covered_by_cell.values()):
        raise ValueError("incomplete_evaluation_dimension_coverage_per_variant")
    return tuple(sorted(covered))


def validate_evidence_matrix(envelopes: list[dict], *, source_commit: str,
                             family_bindings: dict, case_bindings: dict,
                             review_verdicts: dict) -> tuple[str, ...]:
    """Accept only cases bound to a separately approved exact source manifest."""
    trusted_cases = _trusted_reviewed_case_bindings()
    if type(case_bindings) is not dict or case_bindings != trusted_cases:
        raise ValueError("untrusted_or_incomplete_evaluation_inputs")
    return _validate_evidence_matrix_authenticated(
        envelopes, source_commit=source_commit, family_bindings=family_bindings,
        case_bindings=trusted_cases, review_verdicts=review_verdicts,
        reviewed_case_manifest_sha256=PINNED_REVIEWED_CASE_MANIFEST_SHA256)


def _validate_evidence_matrix_authenticated(envelopes: list[dict], *, source_commit: str,
                                            family_bindings: dict, case_bindings: dict,
                                            review_verdicts: dict,
                                            reviewed_case_manifest_sha256: str) -> tuple[str, ...]:
    """Testable cryptographic mechanics; public acceptance also needs the manifest."""
    if not _hex(reviewed_case_manifest_sha256, 64):
        raise ValueError("reviewed_evaluation_case_manifest_not_pinned")
    if type(envelopes) is not list or len(envelopes) != 120:
        raise ValueError("incomplete_evaluation_matrix")
    expected_cases = {
        f"{family}:{variant}:{category}:{n}"
        for family in FAMILIES for variant in VARIANTS
        for category, count in CASE_CATEGORIES.items() for n in range(1, count + 1)
    }
    if set(family_bindings) != FAMILIES or set(case_bindings) != expected_cases:
        raise ValueError("untrusted_or_incomplete_evaluation_inputs")
    if type(review_verdicts) is not dict or set(review_verdicts) != expected_cases:
        raise ValueError("independent_review_required")
    reviewer_public_key = _trusted_reviewer_public_key()
    verified = []
    seen = set()
    for envelope in envelopes:
        if type(envelope) is not dict or type(envelope.get("payload")) is not dict:
            raise ValueError("invalid_evaluation_envelope")
        claimed = envelope["payload"]
        case_id = claimed.get("case_id")
        if type(case_id) is not str or case_id not in expected_cases or case_id in seen:
            raise ValueError("duplicate_or_unconfigured_evaluation_case")
        seen.add(case_id)
        family, variant, category, _ = case_id.split(":")
        pin = case_bindings[case_id]
        family_pin = family_bindings[family]
        if type(pin) is not dict or type(family_pin) is not dict:
            raise ValueError("invalid_evaluation_input_pin")
        expected_dimensions = _expected_case_dimensions(case_id)
        if _validated_dimensions(pin.get("expected_dimensions")) != expected_dimensions:
            raise ValueError("untrusted_evaluation_dimensions")
        variant_adapters = family_pin.get("variant_adapters")
        if (type(variant_adapters) is not dict or set(variant_adapters) != set(VARIANTS)
                or variant_adapters["configured_base"] is not None
                or not _hex(variant_adapters["trained_adapter"], 64)
                or variant_adapters["trained_adapter"] == "0" * 64):
            raise ValueError("invalid_variant_artifact_binding")
        payload = verify_evidence(
            envelope, expected_source_commit=source_commit, expected_family=family,
            expected_base_revision=family_pin["base_revision"],
            expected_base_manifest_sha256=family_pin["base_manifest_sha256"],
            expected_adapter_sha256=variant_adapters[variant],
            expected_runner_sha256=family_pin["runner_sha256"],
            expected_case_id=case_id, expected_variant=variant,
            expected_case_category=category, expected_runtime_profile=pin["runtime_profile"],
            expected_conversation_id=pin["conversation_id"],
            expected_session_id=pin["session_id"],
        )
        if payload["prompt_sha256"] != pin["prompt_sha256"]:
            raise ValueError("substituted_evaluation_prompt")
        if _validated_dimensions(payload.get("dimensions")) != expected_dimensions:
            raise ValueError("evaluation_dimension_binding_mismatch")
        review = review_verdicts[case_id]
        expected_review = {"case_id": case_id, "source_commit": source_commit,
                           "reviewed_case_manifest_sha256": reviewed_case_manifest_sha256,
                           "prompt_sha256": payload["prompt_sha256"],
                           "answer_sha256": payload["answer_sha256"],
                           "expected_dimensions": list(expected_dimensions),
                           "dimension_verdicts": {name: True for name in expected_dimensions},
                           "passed": True}
        if (type(review) is not dict or set(review) != {"payload", "signature_ed25519_b64"}
                or review["payload"] != expected_review):
            raise ValueError("failed_or_unbound_independent_review")
        try:
            reviewer_public_key.verify(
                base64.b64decode(review["signature_ed25519_b64"], validate=True),
                _canonical(expected_review))
        except Exception as exc:
            raise ValueError("invalid_independent_review_signature") from exc
        verified.append(payload)
    return validate_verified_dimension_coverage(verified)


def create_evidence(*, source_commit: str, family: str, base_revision: str,
                    base_manifest_sha256: str, adapter_sha256: str | None,
                    runner_sha256: str, case_id: str, prompt: str, answer: str,
                    runtime_profile: str, conversation_id: str, session_id: str,
                    dimensions: list[str], private_key: Ed25519PrivateKey,
                    variant: str, case_category: str, created_at: str | None = None) -> dict:
    if family not in FAMILIES:
        raise ValueError("unknown_family")
    if not _hex(source_commit, 40) or not _hex(base_revision, 40):
        raise ValueError("invalid_source_or_base_revision")
    if variant not in VARIANTS or case_category not in CASE_CATEGORIES:
        raise ValueError("invalid_evaluation_matrix_cell")
    if (adapter_sha256 is None) != (variant == "configured_base"):
        raise ValueError("variant_adapter_binding_mismatch")
    for name, value in (("base_manifest", base_manifest_sha256),
                        ("runner", runner_sha256)):
        if not _hex(value, 64):
            raise ValueError(f"invalid_{name}_sha256")
    if variant == "trained_adapter" and (not _hex(adapter_sha256, 64) or adapter_sha256 == "0" * 64):
        raise ValueError("invalid_adapter_sha256")
    if not all(_identifier(value) for value in (case_id, conversation_id, session_id)):
        raise ValueError("invalid_evaluation_context")
    if runtime_profile not in PROFILES:
        raise ValueError("invalid_runtime_profile")
    if not isinstance(prompt, str) or not prompt or not isinstance(answer, str) or not answer:
        raise ValueError("empty_evaluation_binding")
    dimensions = list(_validated_dimensions(dimensions))
    payload = {
        "schema": SCHEMA,
        "source_commit": source_commit,
        "family": family,
        "base_revision": base_revision,
        "base_manifest_sha256": base_manifest_sha256,
        "adapter_sha256": adapter_sha256,
        "runner_sha256": runner_sha256,
        "case_id": case_id,
        "variant": variant,
        "case_category": case_category,
        "prompt": prompt,
        "prompt_sha256": _sha256_text(prompt),
        "answer": answer,
        "answer_sha256": _sha256_text(answer),
        "runtime_profile": runtime_profile,
        "conversation_id": conversation_id,
        "session_id": session_id,
        "dimensions": dimensions,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    signature = private_key.sign(_canonical(payload))
    return {
        "payload": payload,
        "signature_ed25519_b64": base64.b64encode(signature).decode("ascii"),
    }


def _trusted_runner_public_key() -> Ed25519PublicKey:
    if not isinstance(PINNED_RUNNER_PUBLIC_KEY_B64, str) or not PINNED_RUNNER_PUBLIC_KEY_B64:
        raise ValueError("trusted_runner_key_not_configured")
    try:
        raw = base64.b64decode(PINNED_RUNNER_PUBLIC_KEY_B64, validate=True)
        if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != PINNED_RUNNER_PUBLIC_KEY_B64:
            raise ValueError("wrong key length")
        return Ed25519PublicKey.from_public_bytes(raw)
    except Exception as exc:
        raise ValueError("invalid_pinned_runner_key") from exc


def _trusted_reviewer_public_key() -> Ed25519PublicKey:
    if not isinstance(PINNED_REVIEWER_PUBLIC_KEY_B64, str) or not PINNED_REVIEWER_PUBLIC_KEY_B64:
        raise ValueError("trusted_reviewer_key_not_configured")
    try:
        raw = base64.b64decode(PINNED_REVIEWER_PUBLIC_KEY_B64, validate=True)
        runner_raw = base64.b64decode(PINNED_RUNNER_PUBLIC_KEY_B64, validate=True)
        if len(raw) != 32 or len(runner_raw) != 32:
            raise ValueError("wrong key length")
        if raw == runner_raw:
            raise ValueError("reviewer_must_be_independent_of_runner")
        if (base64.b64encode(raw).decode("ascii") != PINNED_REVIEWER_PUBLIC_KEY_B64
                or base64.b64encode(runner_raw).decode("ascii") != PINNED_RUNNER_PUBLIC_KEY_B64):
            raise ValueError("noncanonical pinned public key")
        return Ed25519PublicKey.from_public_bytes(raw)
    except ValueError as exc:
        if str(exc) == "reviewer_must_be_independent_of_runner":
            raise
        raise ValueError("invalid_pinned_reviewer_key") from exc
    except Exception as exc:
        raise ValueError("invalid_pinned_reviewer_key") from exc


def verify_evidence(evidence: dict, *, expected_source_commit: str,
                    expected_family: str, expected_base_revision: str,
                    expected_base_manifest_sha256: str, expected_adapter_sha256: str,
                    expected_runner_sha256: str, expected_case_id: str,
                    expected_runtime_profile: str, expected_conversation_id: str,
                    expected_session_id: str, expected_variant: str,
                    expected_case_category: str) -> dict:
    if set(evidence) != {"payload", "signature_ed25519_b64"}:
        raise ValueError("invalid_evidence_shape")
    payload = evidence["payload"]
    expected = {
        "source_commit": expected_source_commit,
        "family": expected_family,
        "base_revision": expected_base_revision,
        "base_manifest_sha256": expected_base_manifest_sha256,
        "adapter_sha256": expected_adapter_sha256,
        "runner_sha256": expected_runner_sha256,
        "case_id": expected_case_id,
        "variant": expected_variant,
        "case_category": expected_case_category,
        "runtime_profile": expected_runtime_profile,
        "conversation_id": expected_conversation_id,
        "session_id": expected_session_id,
    }
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("invalid_evidence_schema")
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"evidence_binding_mismatch:{field}")
    prompt, answer = payload.get("prompt"), payload.get("answer")
    if not isinstance(prompt, str) or payload.get("prompt_sha256") != _sha256_text(prompt):
        raise ValueError("evidence_binding_mismatch:prompt_sha256")
    if not isinstance(answer, str) or payload.get("answer_sha256") != _sha256_text(answer):
        raise ValueError("evidence_binding_mismatch:answer_sha256")
    _validated_dimensions(payload.get("dimensions"))
    try:
        signature = base64.b64decode(evidence["signature_ed25519_b64"], validate=True)
        _trusted_runner_public_key().verify(signature, _canonical(payload))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("invalid_evidence_signature") from exc
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error("evaluation execution blocked until a signed adapter and authority archive exist")
    print(json.dumps({"family": args.family, "dry_run": True, "generated_answers": 0,
                      "provider_calls_made": 0, "signature_scheme": "Ed25519",
                      "trusted_runner_key_configured": bool(PINNED_RUNNER_PUBLIC_KEY_B64)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
