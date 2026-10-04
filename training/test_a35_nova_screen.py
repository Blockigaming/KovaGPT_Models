"""Free synthetic regressions; stubbed completions are never quality evidence."""

from copy import deepcopy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

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

    def test_new_pack_uses_v4_and_all_72_training_rows(self):
        train, val = s.prepared_inputs(self.plan)
        self.assertEqual((len(train), len(val)), (72, 60))
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

    def test_copy_pair_is_additive_and_preserves_every_measured_input(self):
        rows = s.prepared_nova_rows(self.plan)
        self.assertEqual(len(rows), 132)
        measured_pack = "272d2a6e98fa57fec743618591d054e31298f6d36f01e5eacdc61ece61e1e591"
        self.assertEqual(s.sha(s.encoded(rows[:110])), measured_pack)
        self.assertNotEqual(self.plan["prepared_pack_sha256"], measured_pack)
        self.assertEqual(s.sha(s.system_prompt(self.plan).encode()),
                         "258b48ccdc670cfd32312983de0ca4e71b136de1e71aa4ef97fb88d026e554d8")
        self.assertEqual([(identifier, split) for identifier, split, _ in rows[110:112]],
                         [("a35-nova-copy-isolation-train", "train"),
                          ("a35-nova-copy-isolation-validation", "validation")])
        for _, _, row in rows[110:]:
            self.assertEqual(set(row), {"prompt", "completion", "chat_template_kwargs"})
            self.assertEqual(row["prompt"][0]["content"], s.system_prompt(self.plan))

    def test_nova_only_pair_does_not_modify_shared_family_inputs(self):
        shared = s.prepared_revision_rows()
        self.assertEqual(len(shared), 110)
        self.assertEqual(s.sha(s.encoded(shared)),
                         "fcbe8556c9587d960a531bd0f91d5b7de13fa77fd69d3aa9d62a893dcab91726")

    def test_exact_sft_two_epochs_no_step_truncation(self):
        settings = s.sft_kwargs(self.plan, "/unused")
        self.assertEqual(settings["num_train_epochs"], 2)
        self.assertEqual(settings["max_steps"], -1)
        self.assertEqual(settings["max_length"], 768)
        self.assertTrue(settings["completion_only_loss"])
        self.assertFalse(settings["packing"] or settings["dataloader_drop_last"])
        self.assertEqual(settings["eval_strategy"], "no")

    def test_exposure_probe_preserves_inputs_bounds_and_one_run(self):
        self.assertEqual(self.plan["training"]["expected_optimizer_steps"], 18)
        self.assertEqual(self.plan["training"]["maximum_seconds"], 600)
        self.assertEqual(self.plan["prepared_pack_sha256"],
                         "d0a7bbd518601ef95f9f04736aaad02fa5a20f92677042ae6d4ae3f8ce447d52")
        self.assertEqual(self.plan["resources"]["allocation_attempts"], 1)
        self.assertEqual(self.plan["evaluation"]["screening_repetitions"], 1)
        self.assertFalse(self.plan["execution_authorized"])

    def test_exposure_probe_rejects_epoch_or_step_drift(self):
        for field, value in (("epochs", 1), ("epochs", 3),
                             ("expected_optimizer_steps", 9), ("expected_optimizer_steps", 17)):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as folder:
                plan = deepcopy(self.plan)
                plan["training"][field] = value
                path = Path(folder) / "plan.json"
                path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, "training recipe drift"):
                    s.load_plan(path)

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
        generate, _ = self.sample()
        rows, _ = s.run_screen(self.cases, generate, lambda row: None)
        text = s.manual_packet(self.cases, rows)
        self.assertEqual(text.count("\n## "), 14)
        self.assertEqual(text.count("- PENDING:"), 48)
        self.assertNotIn("NOT RUN", text)

    def preserve(self, report, *, manual_expected=False):
        uploaded = {}
        storage = Mock()
        storage.put.side_effect = lambda name, body: uploaded.__setitem__(name, body)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            receipt = b'{"synthetic_training_receipt":true}'
            (output / "training-receipt.json").write_bytes(receipt)
            with patch.object(s, "manual_packet", wraps=s.manual_packet) as packet:
                s.preserve_screen(output, storage, report)
                self.assertEqual(packet.call_count, int(manual_expected))
            self.assertEqual((output / "manual-review.md").exists(), manual_expected)
            self.assertEqual("manual-review.md" in uploaded, manual_expected)
            self.assertEqual(json.loads((output / "screen.json").read_bytes()), report)
            self.assertEqual(json.loads(uploaded["screen.json"]), report)
            self.assertEqual((output / "training-receipt.json").read_bytes(), receipt)
            if manual_expected:
                self.assertEqual((output / "manual-review.md").read_bytes(), uploaded["manual-review.md"])
        return uploaded

    def test_35_of_36_never_generates_writes_or_uploads_manual_packet(self):
        first = next(c["id"] for c in self.cases if c["evaluation"]["kind"] == "exact_json")
        generate, calls = self.sample(lambda c, a, t, b: ("false" if c["id"] == first else a, t))
        rows, state = s.run_screen(self.cases, generate, lambda row: None)
        self.assertEqual(sum(r["result"] == "exact_json_pass" for r in rows), 35)
        self.assertEqual(state, "strict_threshold_failed_manual_skipped")
        self.assertEqual(len(calls), 36)
        self.assertTrue(all(budget == 128 for _, budget in calls))
        self.preserve({"status": state, "cases": rows})
        with self.assertRaisesRegex(ValueError, "strict 36/36"):
            s.manual_packet(self.cases, rows)

    def test_36_of_36_generates_exactly_14_manual_and_publishes_48_criteria_once(self):
        generate, calls = self.sample()
        rows, state = s.run_screen(self.cases, generate, lambda row: None)
        self.assertEqual([budget for _, budget in calls], [128] * 36 + [2048] * 14)
        self.assertEqual(len({case_id for case_id, _ in calls}), 50)
        uploaded = self.preserve({"status": state, "cases": rows}, manual_expected=True)
        packet = uploaded["manual-review.md"].decode()
        self.assertEqual(packet.count("\n## "), 14)
        self.assertEqual(packet.count("- PENDING:"), 48)

    def test_evaluator_exception_preserves_partial_results_without_manual(self):
        for fail_at in (1, 36):
            with self.subTest(fail_at=fail_at):
                generate, calls = self.sample()
                saved = []
                scorer = s.score_case
                def evaluate(*args):
                    if len(calls) == fail_at:
                        raise ValueError("synthetic evaluator error")
                    return scorer(*args)
                with patch.object(s, "score_case", side_effect=evaluate):
                    with self.assertRaisesRegex(ValueError, "evaluator error"):
                        s.run_screen(self.cases, generate, saved.append)
                self.assertEqual(len(calls), fail_at)
                self.assertTrue(all(budget == 128 for _, budget in calls))
                self.preserve({"status": "failed", "cases": saved})

    def test_incomplete_or_missing_completion_evidence_never_unlocks_manual(self):
        for evidence in (None, {}, generation_evidence([7] * 128, max_new_tokens=128, eos_token_id=9),
                         generation_evidence([7], max_new_tokens=128, eos_token_id=9)):
            with self.subTest(evidence=evidence):
                sample, calls = self.sample()
                def generate(prompt, budget):
                    answer, _, latency = sample(prompt, budget)
                    return answer, evidence, latency
                rows, state = s.run_screen(self.cases, generate, lambda row: None)
                self.assertEqual((len(calls), state), (1, "completion_failed"))
                self.preserve({"status": state, "cases": rows})

    def test_failed_36th_checkpoint_blocks_manual_despite_36_scored_passes(self):
        generate, calls = self.sample()
        saved = []
        def checkpoint(row):
            saved.append(row)
            if len(saved) == 36:
                raise OSError("independent evidence readback failed")
        with self.assertRaisesRegex(OSError, "readback failed"):
            s.run_screen(self.cases, generate, checkpoint)
        self.assertEqual((len(calls), len(saved)), (36, 36))
        self.assertTrue(s.strict_passed(self.cases, saved))
        self.preserve({"status": "failed", "cases": saved})

    def test_safety_contract_stop_prevents_manual_and_packet_even_during_manual(self):
        for stop_result in ("private_output_blocked", "output_contract_failed"):
            for fail_at in (1, 36, 37):
                with self.subTest(stop_result=stop_result, fail_at=fail_at):
                    generate, calls = self.sample()
                    scorer = s.score_case
                    def evaluate(*args):
                        return (stop_result, None) if len(calls) == fail_at else scorer(*args)
                    with patch.object(s, "score_case", side_effect=evaluate):
                        rows, state = s.run_screen(self.cases, generate, lambda row: None)
                    self.assertEqual((len(calls), state), (fail_at, "safety_contract_failed"))
                    self.preserve({"status": state, "cases": rows})

    def test_missing_duplicate_or_mismatched_strict_evidence_cannot_create_packet(self):
        generate, _ = self.sample()
        valid, _ = s.run_screen(self.cases, generate, lambda row: None)
        mutations = [lambda rows: rows.pop(0), lambda rows: rows.__setitem__(1, rows[0]),
            lambda rows: rows[0].pop("completion_evidence"),
            lambda rows: rows[0].update(completion_status="unverified"),
            lambda rows: rows[0].update(prompt_sha256="0" * 64),
            lambda rows: rows[0].update(result="exact_json_fail")]
        for mutate in mutations:
            rows = deepcopy(valid)
            mutate(rows)
            self.assertFalse(s.strict_passed(self.cases, rows))
            self.preserve({"status": "manual_review_pending", "cases": rows})
        self.preserve({"status": "failed", "cases": valid})

    def test_stale_local_attempt_is_rejected_before_runtime_or_storage(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "candidate"
            output.mkdir()
            old_packet = output / "manual-review.md"
            old_packet.write_text("old candidate must not count")
            with patch.object(s, "load_plan", return_value=self.plan), \
                 patch.object(s, "clean_head", return_value="c" * 40), \
                 patch.object(s, "admit", return_value=time.time() + 100), \
                 patch("training.a35_screen_storage.Storage") as storage:
                with self.assertRaisesRegex(ValueError, "new output required"):
                    s.execute(Path(folder), output, self.grant())
                storage.assert_not_called()
            self.assertEqual(old_packet.read_text(), "old candidate must not count")

    def test_stale_packet_cannot_be_overwritten_or_uploaded_as_current(self):
        generate, _ = self.sample()
        rows, state = s.run_screen(self.cases, generate, lambda row: None)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "manual-review.md").write_bytes(b"previous candidate")
            storage = Mock()
            with self.assertRaises(FileExistsError):
                s.preserve_screen(output, storage, {"status": state, "cases": rows})
            self.assertEqual((output / "manual-review.md").read_bytes(), b"previous candidate")
            self.assertEqual([call.args[0] for call in storage.put.call_args_list], ["screen.json"])
        self.assertNotEqual(Storage("a" * 32).prefix, Storage("b" * 32).prefix)

    def test_execute_failure_preserves_evidence_and_propagates_for_cleanup(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            output = Path(folder) / "candidate"
            stack.enter_context(patch.object(s, "load_plan", return_value=self.plan))
            stack.enter_context(patch.object(s, "clean_head", return_value="c" * 40))
            stack.enter_context(patch.object(s, "admit", return_value=time.time() + 100))
            stack.enter_context(patch.dict(os.environ, {key: "1" for key in
                ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")}))
            stack.enter_context(patch("training.cosmo_qlora_training.verify_installed_stack"))
            stack.enter_context(patch("training.snapshot_verifier.verify_snapshot"))
            stack.enter_context(patch("importlib.metadata.version", return_value="0.48.2"))
            stack.enter_context(patch.dict(sys.modules, {"torch": None}))
            storage = stack.enter_context(patch("training.a35_screen_storage.Storage")).return_value
            with self.assertRaises(ModuleNotFoundError):
                s.execute(Path(folder), output, self.grant())
            report = json.loads((output / "screen.json").read_text())
            self.assertEqual((report["status"], report["cases"]), ("failed", []))
            self.assertFalse((output / "manual-review.md").exists())
            self.assertEqual([call.args[0] for call in storage.put.call_args_list],
                             ["run-claim.json", "screen.json"])

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
                prepared_pack_sha256=self.plan["prepared_pack_sha256"], optimizer_steps=18,
                completed_epochs=2.0, training_records=72,
                adapter_sha256={f.name: s.sha(f.read_bytes()) for f in adapter.iterdir()})
            s.new_candidate(receipt, self.plan, "c" * 40, "d" * 32, adapter)
            for field, value in (("optimizer_steps", 7), ("optimizer_steps", 9), ("training_records", 27),
                                 ("prepared_pack_sha256", "0" * 64), ("completed_epochs", 0.5),
                                 ("completed_epochs", 1.0)):
                altered = receipt | {field: value}
                with self.assertRaises(ValueError): s.new_candidate(altered, self.plan, "c" * 40, "d" * 32, adapter)
            old = self.plan | {"rejected_adapter_sha256": [receipt["adapter_sha256"]["adapter_model.safetensors"]]}
            with self.assertRaisesRegex(ValueError, "unchanged"):
                s.new_candidate(receipt, old, "c" * 40, "d" * 32, adapter)

    def test_every_measured_adapter_including_latest_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter = Path(folder)
            (adapter / "adapter_model.safetensors").write_bytes(b"measured fixture")
            for digest in s.MEASURED_ADAPTERS:
                with self.subTest(digest=digest), patch.object(s, "sha", return_value=digest):
                    receipt = dict(experiment_id=self.plan["experiment_id"], source_commit="c" * 40,
                        run_id="d" * 32, base_revision=self.plan["base_revision"],
                        prepared_pack_sha256=self.plan["prepared_pack_sha256"], optimizer_steps=18,
                        completed_epochs=2.0, training_records=72,
                        adapter_sha256={"adapter_model.safetensors": digest})
                    with self.assertRaisesRegex(ValueError, "unchanged/corrupt"):
                        s.new_candidate(receipt, self.plan, "c" * 40, "d" * 32, adapter)

    def test_plan_cannot_omit_or_replace_a_measured_adapter_binding(self):
        for rejected in (None, list(s.MEASURED_ADAPTERS[:2]),
                         list(s.MEASURED_ADAPTERS[:3]), ["a" * 64] * 4):
            with self.subTest(rejected=rejected), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "plan.json"
                path.write_text(json.dumps(self.plan | {"rejected_adapter_sha256": rejected}))
                with self.assertRaisesRegex(ValueError, "all measured candidates"):
                    s.load_plan(path)

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
