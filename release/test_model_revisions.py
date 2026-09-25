import unittest

from release.model_revisions import MODEL_SOURCE_REFERENCES, ROUTE_MODEL_SLOTS, source_reference_for_route


class ModelRevisionTests(unittest.TestCase):
    def test_only_three_family_sources_exist(self):
        self.assertEqual(set(MODEL_SOURCE_REFERENCES), {"kova-cosmo","kova-orion","kova-nova"})
        self.assertNotIn("chat-shared", MODEL_SOURCE_REFERENCES)

    def test_chat_uses_cosmo_or_orion_and_nova_is_work_only(self):
        chat_slots = {slot for route, slot in ROUTE_MODEL_SLOTS.items() if route.startswith("chat:")}
        self.assertEqual(chat_slots, {"kova-cosmo","kova-orion"})
        self.assertEqual(source_reference_for_route("work:nova:high").slot, "kova-nova")
        with self.assertRaises(ValueError):
            source_reference_for_route("chat:nova:high")


if __name__ == "__main__":
    unittest.main()
