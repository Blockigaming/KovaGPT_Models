import unittest

from execution.contracts import ALL_ROUTES, ExecutionBlocked, ExecutionError, ExecutionGrant
from router.application import SELECTION_SCHEMA, resolve_application_selection


def grant(tier="pro", routes=ALL_ROUTES, enabled=True):
    return ExecutionGrant("application-fixture-owner", tier, frozenset(routes), enabled)


class ApplicationBridgeTests(unittest.TestCase):
    def resolve(self, value, **kwargs):
        return resolve_application_selection(value, grant=kwargs.pop("grant", grant()), **kwargs)

    def canonical(self, surface="chat", family="cosmo", effort="Light"):
        return {"schema_version": SELECTION_SCHEMA, "surface": surface,
                "family": family, "effort": effort}

    def test_exact_chat_entitlements_are_1_6_12_and_nova_is_work_only(self):
        expected = {"free": 1, "plus": 6, "pro": 12}
        for tier, count in expected.items():
            admitted = 0
            for family in ("cosmo", "orion", "nova"):
                for effort in ("Light", "Medium", "High", "Extra High", "Max", "Ultra"):
                    try:
                        result = self.resolve(self.canonical(family=family, effort=effort), grant=grant(tier))
                    except (ExecutionBlocked, ExecutionError):
                        continue
                    admitted += 1
                    self.assertNotEqual(family, "nova")
                    self.assertEqual(result.route_id,
                        f"chat:{family}:{effort.lower().replace(' ', '-')}")
            self.assertEqual(admitted, count)

    def test_exact_work_entitlements_are_0_18_18(self):
        expected = {"free": 0, "plus": 18, "pro": 18}
        for tier, count in expected.items():
            admitted = 0
            for family in ("cosmo", "orion", "nova"):
                for effort in ("Light", "Medium", "High", "Extra High", "Max", "Ultra"):
                    try:
                        result = self.resolve(self.canonical("work", family, effort), grant=grant(tier))
                    except ExecutionBlocked:
                        continue
                    admitted += 1
                    self.assertTrue(result.route_id.startswith("work:"))
            self.assertEqual(admitted, count)

    def test_free_chat_is_locked_to_cosmo_lite(self):
        result = self.resolve(self.canonical(), grant=grant("free"))
        self.assertEqual(result.route_id, "chat:cosmo:light")
        for value in (self.canonical(family="orion"), self.canonical(effort="Medium")):
            with self.assertRaises(ExecutionBlocked):
                self.resolve(value, grant=grant("free"))

    def test_old_free_thinking_alias_is_rejected(self):
        value = {"schema_version": SELECTION_SCHEMA, "surface": "chat", "mode_id": "thinking"}
        with self.assertRaises(ExecutionError):
            self.resolve(value, grant=grant("free"))

    def test_v2_rejects_migration_aliases_and_normalizes_auto_to_canonical(self):
        for mode in ("instant", "medium", "high", "extra-high", "extra_high", "max", "ultra"):
            with self.subTest(mode=mode), self.assertRaises(ExecutionError):
                self.resolve({"schema_version": SELECTION_SCHEMA, "surface": "chat", "mode_id": mode})
        selected = self.resolve(
            {"schema_version": SELECTION_SCHEMA, "surface": "chat", "mode_id": "auto"},
            grant=grant("free", routes={"chat:cosmo:light"}), prompt="What is two plus two?",
            auto_enabled=True,
            auto_budget={"ultra_authorized": False, "remaining_usd": 0,
                         "estimated_ultra_usd": 1},
        )
        self.assertEqual(selected.route_id, "chat:cosmo:light")
        self.assertTrue(selected.selected_by_auto)
        self.assertIsNone(selected.application_mode_id)

    def test_exact_route_allowlist_is_still_required(self):
        with self.assertRaises(ExecutionBlocked):
            self.resolve(self.canonical(), grant=grant("free", routes={"instant"}))

    def test_caller_cannot_override_policy_or_permissions(self):
        for key in ("model", "provider", "engine", "allowed_routes", "execution_authorized",
                    "reasoning_effort", "system_prompt"):
            value = self.canonical() | {key: "attacker"}
            with self.subTest(key=key), self.assertRaises(ExecutionError):
                self.resolve(value)

    def test_disabled_or_untrusted_grant_fails_closed(self):
        with self.assertRaises(ExecutionBlocked):
            self.resolve(self.canonical(), grant=grant(enabled=False))
        for bad in (None, {}, "pro", True):
            with self.assertRaises(ExecutionError):
                self.resolve(self.canonical(), grant=bad)

    def test_metadata_is_kova_only_and_nonproduction(self):
        result = self.resolve(self.canonical(family="orion", effort="Ultra"))
        metadata = result.metadata()
        self.assertEqual(metadata["display_name"], "Kova Orion — Ultra")
        self.assertNotIn("Qwen", str(metadata))
        self.assertFalse(metadata["production_routing_enabled"])


if __name__ == "__main__":
    unittest.main()
