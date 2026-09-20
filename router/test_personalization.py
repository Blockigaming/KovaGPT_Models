from copy import deepcopy
import unittest

from execution.contracts import ExecutionError
from router.personalization import build_personalization_context, render_personalization_instructions


OWNER = "owner-123"


def item(kind, text="Prefer concise answers.", **changes):
    value = {"id": f"{kind}-1", "owner_id": OWNER, "text": text, "authorized": True}
    if kind == "saved_memory":
        value["approved"] = True
    if kind == "task_context":
        value["project_id"] = "project-1"
    value.update(changes)
    return value


def payload():
    return {
        "schema_version": "kova-personalization.v1",
        "owner_id": OWNER,
        "current_conversation": [item("current_conversation", "The user asked for a short answer.")],
        "saved_memory": [item("saved_memory", "The user prefers direct language.")],
        "custom_instructions": [item("custom_instructions", "Use compact explanations.")],
        "task_context": [item("task_context", "Use the attached project specification.")],
        "session_corrections": [item("session_corrections", "Be shorter in this chat.")],
    }


class PersonalizationTests(unittest.TestCase):
    def test_all_approved_context_layers_are_preserved_without_weight_changes(self):
        source = payload()
        saved = deepcopy(source)
        result = build_personalization_context(source, authenticated_owner_id=OWNER)
        self.assertEqual(source, saved)
        self.assertFalse(result["changes_model_weights"])
        self.assertFalse(result["may_expand_entitlements"])
        rendered = render_personalization_instructions(source, authenticated_owner_id=OWNER)
        for text in ("short answer", "direct language", "compact explanations",
                     "project specification", "shorter in this chat"):
            self.assertIn(text, rendered)

    def test_cross_owner_unapproved_or_unauthorized_context_is_rejected(self):
        mutations = (
            lambda value: value.update(owner_id="other"),
            lambda value: value["saved_memory"][0].update(owner_id="other"),
            lambda value: value["saved_memory"][0].update(approved=False),
            lambda value: value["task_context"][0].update(authorized=False),
        )
        for mutate in mutations:
            value = payload()
            mutate(value)
            with self.subTest(value=value), self.assertRaises(ExecutionError):
                build_personalization_context(value, authenticated_owner_id=OWNER)

    def test_render_requires_independently_authenticated_owner(self):
        value = payload()
        with self.assertRaises(ExecutionError):
            render_personalization_instructions(value, authenticated_owner_id="other")

    def test_context_cannot_override_plan_model_effort_or_execution(self):
        for field in ("tier", "model", "provider", "effort", "allowed_routes", "execution_authorized"):
            value = payload()
            value["custom_instructions"][0][field] = "pro"
            with self.subTest(field=field), self.assertRaises(ExecutionError):
                build_personalization_context(value, authenticated_owner_id=OWNER)

    def test_unknown_shapes_and_oversized_context_fail_closed(self):
        values = [None, [], {}, payload() | {"unknown": True}]
        for value in values:
            with self.subTest(value=value), self.assertRaises(ExecutionError):
                build_personalization_context(value, authenticated_owner_id=OWNER)
        value = payload()
        value["session_corrections"][0]["text"] = "x" * 4097
        with self.assertRaises(ExecutionError):
            build_personalization_context(value, authenticated_owner_id=OWNER)


if __name__ == "__main__":
    unittest.main()
