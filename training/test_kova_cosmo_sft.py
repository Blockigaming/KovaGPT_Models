"""Offline checks for the guarded Kova Cosmo SFT recipe."""
from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from training import kova_cosmo_sft as recipe
from training import identity_pilot as pilot


class KovaCosmoSftTests(unittest.TestCase):
    def test_installed_versions_must_match_every_recipe_pin(self):
        with patch.object(recipe, "version", side_effect=recipe.EXPECTED_SOFTWARE.__getitem__):
            recipe.verify_installed_software()

    def test_declared_source_commit_must_match_a_clean_checkout(self):
        clean = [
            SimpleNamespace(stdout="a" * 40 + "\n", stderr=""),
            SimpleNamespace(stdout="", stderr=""),
        ]
        with patch.object(recipe.subprocess, "run", side_effect=clean) as run:
            recipe.verify_source_checkout("a" * 40)
        self.assertEqual(run.call_count, 2)

        for responses in (
            [SimpleNamespace(stdout="b" * 40 + "\n", stderr=""),
             SimpleNamespace(stdout="", stderr="")],
            [SimpleNamespace(stdout="a" * 40 + "\n", stderr=""),
             SimpleNamespace(stdout=" M training/file.py\n", stderr="")],
        ):
            with self.subTest(responses=responses), patch.object(
                recipe.subprocess, "run", side_effect=responses
            ), self.assertRaises(recipe.RecipeError):
                recipe.verify_source_checkout("a" * 40)

    def test_each_missing_training_dependency_is_rejected(self):
        for missing in recipe.EXPECTED_SOFTWARE:
            if missing == "python":
                continue
            def installed(package):
                if package == missing:
                    raise recipe.PackageNotFoundError(package)
                return recipe.EXPECTED_SOFTWARE[package]
            with self.subTest(package=missing), patch.object(recipe, "version", side_effect=installed):
                with self.assertRaises(recipe.RecipeError):
                    recipe.verify_installed_software()

    def test_each_drifted_training_dependency_is_rejected(self):
        for drifted in recipe.EXPECTED_SOFTWARE:
            if drifted == "python":
                continue
            def installed(package):
                return "0.0.0" if package == drifted else recipe.EXPECTED_SOFTWARE[package]
            with self.subTest(package=drifted), patch.object(recipe, "version", side_effect=installed):
                with self.assertRaises(recipe.RecipeError):
                    recipe.verify_installed_software()

    def test_interpreter_is_checked_separately_from_distributions(self):
        def installed(package):
            self.assertNotEqual(package, "python")
            return recipe.EXPECTED_SOFTWARE[package]
        with patch.object(recipe, "version", side_effect=installed):
            recipe.verify_installed_software()
            with patch.object(recipe.sys, "version_info", SimpleNamespace(major=3, minor=11)):
                with self.assertRaises(recipe.RecipeError):
                    recipe.verify_installed_software()

    def test_blocked_execution_does_not_probe_installed_packages(self):
        with patch.object(recipe, "version", side_effect=AssertionError("premature probe")):
            with self.assertRaises(recipe.RecipeError):
                recipe.execute()

    def test_sft_rows_match_compiled_messages_for_every_example(self):
        train, validation = recipe.prepare_sft_rows()
        self.assertEqual((len(train), len(validation)), (24, 12))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "compiled"
            pilot.prepare(output=output)
            for split, records in (("train", train), ("validation", validation)):
                compiled = [json.loads(line)["messages"] for line in
                            (output / (split + ".jsonl")).read_text().splitlines()]
                self.assertEqual([row["prompt"] + row["completion"]
                                  for row in records], compiled)
                for row in records:
                    self.assertEqual([m["role"] for m in row["prompt"]],
                                     ["system", "user"])
                    self.assertEqual([m["role"] for m in row["completion"]],
                                     ["assistant"])

    def test_provenance_evaluation_receives_its_hypothetical_evidence(self):
        train, validation = recipe.prepare_sft_rows()
        fixture = validation[10]
        system = fixture["prompt"][0]["content"]
        self.assertIn("Offline evaluation fixture only", system)
        self.assertIn("Qwen/Qwen3-0.6B", system)
        self.assertIn(recipe.load_recipe()["base_revision"], system)
        _, ordinary_prompt, _ = pilot.load()
        for row in train + validation[:10] + validation[11:]:
            self.assertEqual(row["prompt"][0]["content"], ordinary_prompt)

    def test_row_preparation_needs_no_training_dependencies(self):
        with patch.dict("sys.modules", {"torch": None, "datasets": None,
                                       "peft": None, "trl": None}):
            train, validation = recipe.prepare_sft_rows()
        self.assertEqual((len(train), len(validation)), (24, 12))

    def test_dry_run_is_nonexecuting_and_pinned(self):
        report = recipe.dry_run()
        self.assertEqual(report["display_name"], "Kova Cosmo")
        self.assertEqual(report["region"], "eastus")
        self.assertEqual(report["vm_size"], "Standard_NC4as_T4_v3")
        self.assertEqual(report["precision"], "fp16")
        self.assertFalse(report["model_weights_downloaded"])
        self.assertFalse(report["training_started"])
        self.assertTrue(report["external_output_directory_required"])
        self.assertFalse(report["adapter_receipt_created"])
        self.assertFalse(report["phase_b_ready"])

    def test_external_output_directory_and_source_commit_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-run"
            with patch.dict("os.environ", {
                "KOVA_COSMO_OUTPUT_DIR": str(output),
                "KOVA_SOURCE_COMMIT": "a" * 40,
            }, clear=True):
                resolved, commit = recipe.resolve_output_directory()
            self.assertEqual(resolved, output)
            self.assertEqual(commit, "a" * 40)

            output.mkdir()
            with patch.dict("os.environ", {
                "KOVA_COSMO_OUTPUT_DIR": str(output),
                "KOVA_SOURCE_COMMIT": "a" * 40,
            }, clear=True), self.assertRaises(recipe.RecipeError):
                recipe.resolve_output_directory()

        with tempfile.TemporaryDirectory(dir=recipe.ROOT) as directory:
            internal = Path(directory) / "new-run"
            with patch.dict("os.environ", {
                "KOVA_COSMO_OUTPUT_DIR": str(internal),
                "KOVA_SOURCE_COMMIT": "a" * 40,
            }, clear=True), self.assertRaises(recipe.RecipeError):
                recipe.resolve_output_directory()

        invalid = [
            ("relative-run", "a" * 40),
            (str(Path(tempfile.gettempdir()) / "new-run"), "A" * 40),
            (str(Path(tempfile.gettempdir()) / "new-run"), "a" * 39),
        ]
        for output, commit in invalid:
            with self.subTest(output=output, commit=commit), patch.dict(
                "os.environ", {
                    "KOVA_COSMO_OUTPUT_DIR": output,
                    "KOVA_SOURCE_COMMIT": commit,
                }, clear=True
            ), self.assertRaises(recipe.RecipeError):
                recipe.resolve_output_directory()

    def test_current_package_versions_are_exact(self):
        self.assertEqual(recipe.load_recipe()["software"], {
            "python": "3.12",
            "torch": "2.8.0",
            "transformers": "5.17.0",
            "peft": "0.21.0",
            "trl": "1.13.0",
            "accelerate": "1.15.0",
            "datasets": "5.0.1",
            "cryptography": "50.0.1",
        })

    def test_lora_targets_are_explicit_and_stable(self):
        value = recipe.load_recipe()
        self.assertEqual(value["lora"]["target_modules"], recipe.EXPECTED_TARGETS)
        self.assertEqual(value["lora"]["r"], 16)
        self.assertEqual(value["lora"]["alpha"], 32)

    def test_t4_recipe_uses_fp16_not_bf16(self):
        value = recipe.load_recipe()
        self.assertEqual(value["hardware"]["precision"], "fp16")
        self.assertIs(value["hardware"]["bf16"], False)
        self.assertIs(value["hardware"]["quantized_base"], False)

    def test_first_cosmo_pilot_is_plain_lora_not_q_lora(self):
        value = recipe.load_recipe()
        self.assertEqual(value["method"], "lora_sft")
        self.assertFalse(value["hardware"]["quantized_base"])

    def test_completion_only_loss_keeps_prompt_tokens_out_of_objective(self):
        value = recipe.load_recipe()
        self.assertIs(value["training"]["completion_only_loss"], True)
        self.assertIs(value["training"]["packing"], False)

    def test_remote_reporting_and_hub_push_are_disabled(self):
        training = recipe.load_recipe()["training"]
        self.assertEqual(training["report_to"], "none")
        self.assertIs(training["push_to_hub"], False)

    def test_approved_pilot_records_quota_and_remains_blocked_on_runtime(self):
        gates = recipe.load_recipe()["account_gates"]
        self.assertIs(gates["microsoft_quota_provider_registration_authorized"], True)
        self.assertIs(gates["eastus_ncast4_quota_verified"], True)
        self.assertIs(gates["runtime_compatibility_verified"], False)
        self.assertEqual(gates["approved_budget_usd"], 2.0)

    def test_pilot_permissions_do_not_authorize_deployment(self):
        permissions = recipe.load_recipe()["execution"]
        self.assertEqual(permissions, {
            "model_download_authorized": True,
            "training_authorized": True,
            "deployment_authorized": False,
        })

    def test_execute_fails_before_heavy_library_imports(self):
        with patch.dict("sys.modules", {
            "torch": None,
            "datasets": None,
            "peft": None,
            "trl": None,
        }):
            with self.assertRaises(recipe.RecipeError):
                recipe.execute()

    def test_training_reserves_remote_phase_before_heavy_imports(self):
        value = deepcopy(recipe.load_recipe())
        value["account_gates"]["eastus_ncast4_quota_verified"] = True
        value["account_gates"]["runtime_compatibility_verified"] = True
        runtime = {
            "runtime_evidence_sha256": "e" * 64,
            "deadline_utc": "2026-09-19T19:40:00Z",
            "lifecycle_id": "lifecycle-001",
            "preflight_ledger_sequence": 1,
            "azure_instance": {
                "resource_id": "/subscriptions/11111111-2222-3333-4444-555555555555/resourceGroups/kova-cosmo-pilot/providers/Microsoft.Compute/virtualMachines/kova-cosmo-t4",
                "vm_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "system_assigned_identity_principal_id":
                    "99999999-8888-7777-6666-555555555555",
            },
        }
        grant = {
            "phase_grant_sha256": "f" * 64,
            "phase_grant_context": {"operation": "single_lora_sft_run"},
            "phase_grant_envelope": {"payload": {}, "signature": ""},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            output = root / "output"
            with patch.object(recipe, "load_recipe", return_value=value), \
                 patch.dict("os.environ", {
                     "KOVA_CONFIRM_PAID_TRAINING": "YES",
                     "KOVA_COSMO_VERIFIED_SNAPSHOT": str(snapshot),
                 }, clear=True), \
                 patch.object(recipe, "require_runtime_ready", return_value=runtime), \
                 patch.object(recipe, "verify_installed_software"), \
                 patch.object(recipe, "verify_snapshot", return_value={
                     "model.safetensors": {"sha256": "0" * 64, "bytes": 1}
                 }), \
                 patch.object(
                     recipe, "resolve_output_directory",
                     return_value=(output, "a" * 40),
                 ), \
                 patch.object(recipe, "verify_source_checkout"), \
                 patch.object(
                     recipe, "reserve_training_phase", return_value=grant
                 ) as acquire, \
                 patch.object(
                     recipe, "persist_training_grant",
                     return_value={"phase_grant_sha256": "f" * 64},
                 ) as persist, \
                 patch.dict("sys.modules", {"torch": None}):
                with self.assertRaises(ModuleNotFoundError):
                    recipe.execute()
        self.assertEqual(acquire.call_args.kwargs["phase"], "training")
        self.assertEqual(
            acquire.call_args.kwargs["runtime_evidence_sha256"], "e" * 64
        )
        self.assertEqual(
            acquire.call_args.kwargs["azure_instance"],
            runtime["azure_instance"],
        )
        self.assertEqual(
            persist.call_args.kwargs["grant_envelope"],
            grant["phase_grant_envelope"],
        )

    def test_operator_environment_variable_cannot_override_source_guards(self):
        with patch.dict("os.environ", {"KOVA_CONFIRM_PAID_TRAINING": "YES"}):
            with self.assertRaises(recipe.RecipeError):
                recipe.execute()

    def test_dry_run_never_closes_phase_a_items(self):
        report = recipe.dry_run()
        self.assertEqual(report["closed_checklist_ids"], [])
        self.assertFalse(report["runtime_integrated"])

    def test_cli_execute_returns_nonzero_while_blocked(self):
        error = io.StringIO()
        with redirect_stderr(error):
            self.assertEqual(recipe.main(["--execute"]), 1)
        self.assertEqual(error.getvalue(), "kova cosmo sft recipe rejected\n")


if __name__ == "__main__":
    unittest.main()
