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
        self.assertFalse(report["phase_b_ready"])

    def test_current_package_versions_are_exact(self):
        self.assertEqual(recipe.load_recipe()["software"], {
            "python": "3.12",
            "torch": "2.8.0",
            "transformers": "5.17.0",
            "peft": "0.21.0",
            "trl": "1.13.0",
            "accelerate": "1.15.0",
            "datasets": "5.0.1",
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

    def test_approved_pilot_remains_blocked_on_quota_and_runtime(self):
        gates = recipe.load_recipe()["account_gates"]
        self.assertIs(gates["microsoft_quota_provider_registration_authorized"], True)
        self.assertIs(gates["eastus_ncast4_quota_verified"], False)
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
