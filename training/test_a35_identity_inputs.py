"""Identity edition integrity, historical separation and raw scoring regressions."""
from copy import deepcopy
import json
import unittest

from core.public_identity import contains_prohibited
from evaluation.completion_evidence import generation_evidence
from evaluation.historical_suite_bridge import load_archived_suite
from training import a35_nova_screen as s
from training.a35_identity_inputs import DATA_PATH, REVIEW_PATH, PARENT_PACK, apply_identity_policy


class IdentityInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = s.load_plan()
        cls.old = s.historical_nova_rows(cls.plan)
        cls.new = s.prepared_nova_rows(cls.plan)

    def test_historical_parent_is_unchanged_and_new_pack_is_distinct(self):
        self.assertEqual(s.sha(s.encoded(self.old)), PARENT_PACK)
        self.assertNotEqual(s.sha(s.encoded(self.new)), PARENT_PACK)
        self.assertEqual(s.sha(s.encoded(self.new)), self.plan["prepared_pack_sha256"])

    def test_only_18_identity_targets_changed_no_tasks_or_splits_changed(self):
        changed = []
        for (old_id, old_split, old), (new_id, new_split, new) in zip(self.old, self.new):
            self.assertEqual((old_id, old_split), (new_id, new_split))
            self.assertEqual(old["prompt"][1:], new["prompt"][1:])
            self.assertEqual(old["chat_template_kwargs"], new["chat_template_kwargs"])
            self.assertEqual(new["prompt"][0]["content"], s.system_prompt(self.plan))
            self.assertFalse(contains_prohibited(new["completion"]))
            if old["completion"] != new["completion"]: changed.append(old_id)
        review = json.loads((s.ROOT / REVIEW_PATH).read_text())
        self.assertEqual(set(changed), {row["id"] for row in review["records"]})
        self.assertEqual(len(changed), 18)
        self.assertEqual(len(self.new) - len(changed), 116)

    def test_no_train_validation_duplicate_or_strict_prompt_leakage(self):
        train = {r["prompt"][1]["content"] for _, split, r in self.new if split == "train"}
        val = {r["prompt"][1]["content"] for _, split, r in self.new if split == "validation"}
        strict = {c["prompt"] for c in load_archived_suite()["cases"] if c["evaluation"]["kind"] == "exact_json"}
        self.assertFalse(train & val)
        self.assertFalse((train | val) & strict)
        self.assertEqual((len(train), len(val)), (73, 61))

    def test_override_and_review_hash_drift_rejected(self):
        for name in (DATA_PATH, REVIEW_PATH):
            pins = dict(self.plan["file_sha256"], **{name: "0" * 64})
            with self.subTest(name=name), self.assertRaises(ValueError):
                apply_identity_policy(self.old, s.system_prompt(self.plan), pins)

    def test_parent_drift_or_unsafe_system_rejected(self):
        old = deepcopy(self.old); old[0][2]["completion"][0]["content"] = "Changed"
        with self.assertRaises(ValueError):
            apply_identity_policy(old, s.system_prompt(self.plan), self.plan["file_sha256"])
        with self.assertRaises(ValueError):
            apply_identity_policy(self.old, "You are INTERNAL_UPSTREAM_MODEL", self.plan["file_sha256"])

    def test_raw_identity_leak_stops_quality_and_never_unlocks_manual(self):
        cases, saved, calls = load_archived_suite()["cases"], [], []
        raw = "INTERNAL_UPSTREAM_MODEL"
        def generate(prompt, budget):
            calls.append(prompt)
            return raw, generation_evidence([7, 9], max_new_tokens=budget, eos_token_id=9), 0.001
        rows, state = s.run_screen(cases, generate, saved.append)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state, "safety_contract_failed")
        self.assertEqual(rows, saved)
        self.assertEqual(rows[0]["answer"], raw)
        self.assertEqual(rows[0]["answer_sha256"], s.sha(raw.encode()))
        self.assertEqual(rows[0]["result"], "identity_policy_blocked")
        self.assertFalse(s.strict_passed(cases, rows))
        with self.assertRaises(ValueError): s.manual_packet(cases, rows)

    def test_synthetic_catalog_cannot_admit_real_execution(self):
        from core.private_provenance import load_catalog, PrivateCatalogError
        with self.assertRaises(PrivateCatalogError):
            load_catalog(self.plan["private_catalog_sha256"])

    def test_existing_recipe_quality_and_six_rejections_remain(self):
        self.assertEqual(self.plan["training"]["epochs"], 3)
        self.assertEqual(self.plan["training"]["maximum_seconds"], 600)
        self.assertEqual(self.plan["training"]["expected_optimizer_steps"], 30)
        self.assertEqual(self.plan["rejected_adapter_sha256"], list(s.MEASURED_ADAPTERS))
        self.assertEqual(len(s.MEASURED_ADAPTERS), 6)
        self.assertFalse(self.plan["execution_authorized"])

    def test_new_receipt_exactly_binds_inputs_without_quality_credit(self):
        receipt = json.loads((s.ROOT / "evaluations/a35-kovagpt-identity-receipt.v1.json").read_text())
        for key, path in (("dataset", DATA_PATH), ("review", REVIEW_PATH),
                          ("tokenizer", "evaluations/a35-nova-task-checks-tokenizer.v1.json"),
                          ("plan", "config/a35-nova-screen.v1.json")):
            self.assertEqual(receipt["hashes"][key], s.sha((s.ROOT / path).read_bytes()))
        self.assertEqual(receipt["hashes"]["pack"], self.plan["prepared_pack_sha256"])
        self.assertEqual(receipt["hashes"]["prompt"], s.sha(s.system_prompt(self.plan).encode()))
        self.assertEqual(receipt["latest_measured"]["strict"], 25)
        self.assertEqual(receipt["a35"], "OPEN")
        self.assertEqual(receipt["rejected_adapters"], list(s.MEASURED_ADAPTERS))
        self.assertFalse(receipt["next_proposal"]["execution_authorized"])
        self.assertFalse(receipt["source_tests_are_model_quality_evidence"])


if __name__ == "__main__":
    unittest.main()
