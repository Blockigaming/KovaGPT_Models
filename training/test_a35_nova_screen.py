"""Free synthetic regressions; stubbed completions are never quality evidence."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.completion_evidence import generation_evidence
from evaluation.historical_suite_bridge import load_archived_suite
from training import a35_nova_screen as s
from training.a35_screen_storage import Storage
from training import a35_screen_control as control


class ScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = s.load_plan()
        cls.cases = load_archived_suite()["cases"]

    def test_new_pack_uses_v4_and_all_61_training_rows(self):
        train, val = s.prepared_inputs(self.plan)
        self.assertEqual((len(train), len(val)), (61, 49))
        prompt = (s.ROOT / "prompts/kova-identity.v4.draft.txt").read_text()
        self.assertTrue(all(row["prompt"][0]["content"].startswith(prompt) for row in train + val))
        self.assertTrue(all(set(row) == {"prompt", "completion", "chat_template_kwargs"}
                            for row in train + val))
        self.assertFalse({r["prompt"][1]["content"] for r in train} &
                         {r["prompt"][1]["content"] for r in val})

    def test_pack_mutation_rejected(self):
        plan = deepcopy(self.plan)
        plan["prepared_pack_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "pack drift"):
            s.prepared_inputs(plan)

    def test_exact_sft_one_epoch_no_seven_step_truncation(self):
        settings = s.sft_kwargs(self.plan, "/unused")
        self.assertEqual(settings["num_train_epochs"], 1)
        self.assertEqual(settings["max_steps"], -1)
        self.assertEqual(settings["max_length"], 768)
        self.assertTrue(settings["completion_only_loss"])
        self.assertFalse(settings["packing"] or settings["dataloader_drop_last"])
        self.assertEqual(settings["eval_strategy"], "no")

    def grant(self):
        return dict(owner_authorized=True, experiment_id=self.plan["experiment_id"],
            source_commit="c" * 40, plan_sha256=s.sha(s.PLAN.read_bytes()),
            prepared_pack_sha256=self.plan["prepared_pack_sha256"], family="kova-nova",
            training_runs=1, evaluation_sweeps=1, spend_ceiling_usd="5.00", run_id="d" * 32,
            allocation_started_epoch=1000, watchdog_deadline_epoch=5500,
            allocation_deadline_epoch=6400, live_watchdog_verified=True,
            account_rates_within_reserved_bounds=True)

    def test_admission_valid_exact_scope(self):
        self.assertEqual(s.admit(self.grant(), self.plan, "c" * 40, now=1100), 5200)

    def test_admission_rejects_drift_expiry_or_scope_expansion(self):
        for field, value in (("owner_authorized", False), ("family", "kova-cosmo"),
             ("source_commit", "e" * 40), ("prepared_pack_sha256", "0" * 64),
             ("training_runs", 2), ("evaluation_sweeps", 3), ("spend_ceiling_usd", "6.00"),
             ("watchdog_deadline_epoch", 5550), ("allocation_deadline_epoch", 6500),
             ("live_watchdog_verified", False), ("account_rates_within_reserved_bounds", False)):
            with self.subTest(field=field):
                grant = self.grant(); grant[field] = value
                with self.assertRaises(ValueError): s.admit(grant, self.plan, "c" * 40, now=1100)
        for now in (999, 5200, 6400):
            with self.assertRaises(ValueError): s.admit(self.grant(), self.plan, "c" * 40, now=now)

    def sample(self, bad=None):
        by_prompt = {c["prompt"]: c for c in self.cases}
        calls = []
        def generate(prompt, budget):
            case = by_prompt[prompt]
            calls.append((case["id"], budget))
            answer = json.dumps(case["evaluation"].get("expected", "synthetic manual stub"))
            tokens = [7, 9]
            if bad:
                answer, tokens = bad(case, answer, tokens, budget)
            return answer, generation_evidence(tokens, max_new_tokens=budget, eos_token_id=9), 0.001
        return generate, calls

    def test_strict_failure_stops_before_manual_or_repeat(self):
        generate, calls = self.sample(lambda c, a, t, b: ("null", t))
        rows, state = s.run_screen(self.cases, generate, lambda row: None)
        self.assertEqual((len(calls), len(rows), state), (36, 36, "strict_threshold_failed_manual_skipped"))
        self.assertTrue(all(b == 128 for _, b in calls))

    def test_perfect_synthetic_strict_unlocks_only_one_manual_sweep(self):
        generate, calls = self.sample()
        saved = []
        rows, state = s.run_screen(self.cases, generate, saved.append)
        self.assertEqual(len(calls), 50)
        self.assertEqual(rows, saved)
        self.assertEqual(state, "manual_review_pending")
        self.assertEqual([b for _, b in calls[36:]], [2048] * 14)
        self.assertEqual(sum(r["result"] == "exact_json_pass" for r in rows), 36)
        self.assertEqual(sum(r["result"] == "pending_human_review" for r in rows), 14)

    def test_length_cap_and_unexplained_stop_never_earn_quality(self):
        for tokens in ([7] * 128, [7, 8]):
            generate, calls = self.sample(lambda c, a, t, b: (a, tokens))
            rows, state = s.run_screen(self.cases, generate, lambda row: None)
            self.assertEqual(len(calls), 1)
            self.assertEqual(state, "completion_failed")
            self.assertNotEqual(rows[0]["result"], "exact_json_pass")

    def test_private_text_is_redacted_and_immediately_stops(self):
        generate, calls = self.sample(lambda c, a, t, b: ("<think>private</think>" + a, t))
        rows, state = s.run_screen(self.cases, generate, lambda row: None)
        self.assertEqual((len(calls), state), (1, "safety_contract_failed"))
        self.assertIsNone(rows[0]["answer"])
        self.assertNotIn("private</think>", json.dumps(rows))

    def test_manual_packet_keeps_all_14_cases_and_48_pending_criteria(self):
        text = s.manual_packet(self.cases, [])
        self.assertEqual(text.count("\n## "), 14)
        self.assertEqual(text.count("- PENDING:"), 48)
        self.assertEqual(text.count("NOT RUN"), 14)

    def test_failed_preservation_stops_before_next_generation(self):
        generate, calls = self.sample()
        def failed(_): raise OSError("storage unavailable")
        with self.assertRaises(OSError): s.run_screen(self.cases, generate, failed)
        self.assertEqual(len(calls), 1)

    def test_old_or_partial_adapter_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter = Path(folder)
            (adapter / "adapter_model.safetensors").write_bytes(b"synthetic-new-adapter")
            (adapter / "adapter_config.json").write_text("{}")
            receipt = dict(experiment_id=self.plan["experiment_id"], source_commit="c" * 40,
                run_id="d" * 32, base_revision=self.plan["base_revision"],
                prepared_pack_sha256=self.plan["prepared_pack_sha256"], optimizer_steps=8,
                completed_epochs=1.0, training_records=61,
                adapter_sha256={f.name: s.sha(f.read_bytes()) for f in adapter.iterdir()})
            s.new_candidate(receipt, self.plan, "c" * 40, "d" * 32, adapter)
            for field, value in (("optimizer_steps", 7), ("training_records", 27),
                                 ("prepared_pack_sha256", "0" * 64), ("completed_epochs", 0.5)):
                altered = receipt | {field: value}
                with self.assertRaises(ValueError): s.new_candidate(altered, self.plan, "c" * 40, "d" * 32, adapter)
            old = self.plan | {"preserved_adapter_sha256": receipt["adapter_sha256"]["adapter_model.safetensors"]}
            with self.assertRaisesRegex(ValueError, "unchanged"):
                s.new_candidate(receipt, old, "c" * 40, "d" * 32, adapter)

    def test_outputs_and_run_claims_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"
            s.write_new(path, {"first": True})
            with self.assertRaises(FileExistsError): s.write_new(path, {"first": False})

    def test_storage_rejects_escape_and_oversized_before_token_access(self):
        store = Storage("a" * 32)
        for path, body in (("../overwrite", b"x"), ("other/../../x.json", b"x"),
                           ("x.json", b"")):
            with self.assertRaises(ValueError): store.put(path, body)
        store.total = 64 * 1024 * 1024
        with self.assertRaises(ValueError): store.put("x.json", b"x")


class CleanupTests(unittest.TestCase):
    sub = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    run_id = "b" * 32

    def documents(self):
        from training.cosmo_controller_azure import ARM, CONTRIBUTOR, watchdog_definition
        from urllib.parse import urlencode
        ctx = control.scopes(self.sub, self.run_id)
        pilot, watch = ctx["pilot_resource_group_id"], ctx["watchdog_resource_group_id"]
        principal = "cccccccc-cccc-cccc-cccc-cccccccccccc"
        docs = {ARM + ctx["watchdog_id"] + "?api-version=2019-05-01": {
            "id": ctx["watchdog_id"], "location": "eastus",
            "identity": {"type": "SystemAssigned", "principalId": principal},
            "properties": {"state": "Enabled", "provisioningState": "Succeeded",
                "parameters": {"deadlineUtc": {"value": "1970-01-01T01:31:40Z"}},
                "definition": watchdog_definition(ctx["vm_id"], pilot, watch)}},
            ARM + ctx["watchdog_id"] + "/triggers/every_minute?api-version=2019-05-01": {
                "properties": {"state": "Enabled"}}}
        for group in (pilot, watch):
            docs[ARM + group + "/providers/Microsoft.Authorization/roleAssignments?api-version=2022-04-01"] = {
                "value": [{"properties": {"principalId": principal, "scope": group,
                    "roleDefinitionId": "/roles/" + CONTRIBUTOR}}]}
            docs[ARM + group + "/providers/Microsoft.Authorization/denyAssignments?api-version=2022-04-01&" +
                 urlencode({"$filter": "atScope()"})] = {"value": []}
            docs[ARM + group + "/resources?api-version=2021-04-01"] = {
                "value": [] if group == pilot else [{"id": ctx["watchdog_id"]}]}
        for suffix in ("/providers/Microsoft.Authorization/locks?api-version=2016-09-01",
                       "/providers/Microsoft.Authorization/denyAssignments?api-version=2022-04-01"):
            docs[ARM + "/subscriptions/" + self.sub + suffix] = {"value": []}
        return docs

    def test_exact_external_watchdog_can_admit_empty_vm_group(self):
        docs = self.documents()
        result = control.verify_before_vm(self.sub, self.run_id, 1000, docs.__getitem__, now=1100)
        self.assertTrue(result["live_watchdog_verified"])
        self.assertEqual((result["watchdog_deadline_epoch"], result["allocation_deadline_epoch"]), (5500, 6400))

    def test_watchdog_mutation_disabled_trigger_scope_or_existing_vm_rejected(self):
        original = self.documents()
        watchdog_url = next(u for u in original if u.endswith("?api-version=2019-05-01") and "/triggers/" not in u)
        ctx = control.scopes(self.sub, self.run_id)
        mutations = [
            (watchdog_url, lambda x: x["properties"].update(state="Disabled")),
            (watchdog_url, lambda x: x["properties"]["parameters"]["deadlineUtc"].update(value="2099-01-01T00:00:00Z")),
            (watchdog_url, lambda x: x["properties"]["definition"]["actions"].pop("delete_pilot_group")),
            ("https://management.azure.com" + ctx["pilot_resource_group_id"] + "/resources?api-version=2021-04-01",
             lambda x: x["value"].append({"id": ctx["vm_id"]})),
        ]
        for url, mutate in mutations:
            docs = deepcopy(original); mutate(docs[url])
            with self.assertRaises(ValueError):
                control.verify_before_vm(self.sub, self.run_id, 1000, docs.__getitem__, now=1100)

    def test_incomplete_lock_denial_and_role_inventory_fail_closed(self):
        original = self.documents()
        for fragment in ("/locks?", "/denyAssignments?", "/roleAssignments?"):
            docs = deepcopy(original)
            key = next(u for u in docs if fragment in u)
            docs[key]["nextLink"] = "unread-page"
            with self.assertRaises(ValueError):
                control.verify_before_vm(self.sub, self.run_id, 1000, docs.__getitem__, now=1100)

    def test_cleanup_deletes_even_when_deallocation_errors(self):
        calls = []
        def call(method, url):
            calls.append((method, url))
            if method == "POST": raise OSError("provider deallocation failed")
            return (404, {}) if method == "GET" else (202, {})
        result = control.cleanup(self.sub, self.run_id, call, sleep=lambda _: None)
        self.assertTrue(result["vm_group_absent"] and result["control_group_absent"])
        self.assertEqual([m for m, _ in calls], ["POST", "DELETE", "GET", "DELETE", "GET"])

    def test_accepted_delete_or_forbidden_get_is_not_absence(self):
        for status in (200, 202, 403):
            calls = []
            clock = iter([0, 0, 901]).__next__
            def call(method, url):
                calls.append((method, url)); return status, {}
            with self.assertRaises(TimeoutError):
                control.cleanup(self.sub, self.run_id, call, clock=clock, sleep=lambda _: None)
            self.assertFalse(any("kova-a35-watch" in u for _, u in calls))


if __name__ == "__main__":
    unittest.main()
