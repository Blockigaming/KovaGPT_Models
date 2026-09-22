"""Regression tests for asymmetric guarded-generation attestations."""
from copy import deepcopy
import inspect
from pathlib import Path
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from training import cosmo_generation_attestation as attestation


class CosmoGenerationAttestationTests(unittest.TestCase):
    def setUp(self):
        self.private = Ed25519PrivateKey.from_private_bytes(b"g" * 32)
        self.public = attestation.public_key_hex(self.private)

    def bundle(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "measured",
            "plan_sha256": "a" * 64,
            "source_commit": "b" * 40,
            "adapter_sha256": "c" * 64,
            "adapter_receipt_sha256": "d" * 64,
            "evaluation_grant": {"fixture": True},
            "runner_attestation": None,
            "attempts": [{
                "id": "case-trained_adapter",
                "case_id": "case",
                "case_sha256": "e" * 64,
                "variant": "trained_adapter",
                "outcome": "success",
                "answer": "runner-produced answer",
                "answer_sha256": "f" * 64,
                "latency_ms": 1,
                "input_tokens": 2,
                "output_tokens": 3,
                "runtime": {"guard": "bound"},
                "scores": {"identity": "pending"},
            }],
        }

    def signed_bundle(self) -> dict:
        value = self.bundle()
        value["runner_attestation"] = attestation.create_attestation(
            value, self.private
        )
        return value

    def test_public_key_verifies_runner_signature_without_private_key(self):
        value = self.signed_bundle()
        report = attestation.verify_attestation(
            value, public_key_hex_value=self.public
        )
        self.assertEqual(report["status"], "guarded_runner_signature_verified")
        self.assertEqual(report["algorithm"], "ed25519")
        parameters = inspect.signature(attestation.verify_attestation).parameters
        self.assertEqual(tuple(parameters), ("bundle", "public_key_hex_value"))

    def test_tampered_answer_runtime_or_lineage_is_rejected(self):
        mutations = [
            lambda value: value["attempts"][0].update(
                answer="hand-authored replacement"
            ),
            lambda value: value["attempts"][0]["runtime"].update(
                guard="different"
            ),
            lambda value: value.update(adapter_sha256="0" * 64),
        ]
        for mutate in mutations:
            value = self.signed_bundle()
            mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(
                attestation.AttestationError
            ):
                attestation.verify_attestation(
                    value, public_key_hex_value=self.public
                )

    def test_fake_signer_and_wrong_public_key_are_rejected(self):
        fake = Ed25519PrivateKey.from_private_bytes(b"f" * 32)
        value = self.bundle()
        value["runner_attestation"] = attestation.create_attestation(value, fake)
        with self.assertRaises(attestation.AttestationError):
            attestation.verify_attestation(
                value, public_key_hex_value=self.public
            )
        with self.assertRaises(attestation.AttestationError):
            attestation.verify_attestation(
                self.signed_bundle(),
                public_key_hex_value=attestation.public_key_hex(fake),
            )

    def test_scores_are_outside_runner_authority_but_answers_are_not(self):
        value = self.signed_bundle()
        changed = deepcopy(value)
        changed["attempts"][0]["scores"]["identity"] = "pass"
        report = attestation.verify_attestation(
            changed, public_key_hex_value=self.public
        )
        self.assertEqual(report["status"], "guarded_runner_signature_verified")

    def test_runner_private_seed_must_be_protected_external_and_match_public(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            external = Path(directory) / "runner.key"
            external.write_text((b"g" * 32).hex(), encoding="ascii")
            external.chmod(0o600)
            loaded = attestation.load_signing_key(
                external,
                expected_public_key_hex=self.public,
                repository_root=root,
            )
            self.assertEqual(attestation.public_key_hex(loaded), self.public)

            inside = root / "runner.key"
            inside.write_text((b"g" * 32).hex(), encoding="ascii")
            inside.chmod(0o600)
            permissive = Path(directory) / "permissive.key"
            permissive.write_text((b"g" * 32).hex(), encoding="ascii")
            permissive.chmod(0o644)
            wrong_public = "0" * 64
            for path, public in (
                (inside, self.public),
                (permissive, self.public),
                (external, wrong_public),
            ):
                with self.subTest(path=path, public=public), self.assertRaises(
                    attestation.AttestationError
                ):
                    attestation.load_signing_key(
                        path,
                        expected_public_key_hex=public,
                        repository_root=root,
                    )


if __name__ == "__main__":
    unittest.main()
