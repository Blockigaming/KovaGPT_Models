from copy import deepcopy
import unittest

from execution.contracts import ExecutionError
from router.personalization import build_personalization_context, render_personalization_instructions


OWNER = "owner-123"
CONVERSATION = "conversation-123"
DIGEST = "a" * 64


def item(kind, text="Prefer concise answers.", **changes):
    value = {"id": f"{kind}-1", "owner_id": OWNER, "text": text, "authorized": True}
    if kind == "saved_memory":
        value["approved"] = True
    if kind == "task_context":
        value.update({
            "project_id": "project-1",
            "source_type": "project_file",
            "source_id": "spec.md",
            "source_revision": "revision-1",
            "source_digest_sha256": DIGEST,
        })
    if kind in {"current_conversation", "session_corrections"}:
        value["conversation_id"] = CONVERSATION
    value.update(changes)
    return value


def payload():
    return {
        "schema_version": "kova-personalization.v1",
        "owner_id": OWNER,
        "conversation_id": CONVERSATION,
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
        result = build_personalization_context(
            source, authenticated_owner_id=OWNER,
            active_conversation_id=CONVERSATION)
        self.assertEqual(source, saved)
        self.assertFalse(result["changes_model_weights"])
        self.assertFalse(result["may_expand_entitlements"])
        rendered = render_personalization_instructions(
            source, authenticated_owner_id=OWNER,
            active_conversation_id=CONVERSATION)
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
                build_personalization_context(
                    value, authenticated_owner_id=OWNER,
                    active_conversation_id=CONVERSATION)

    def test_render_requires_independently_authenticated_owner(self):
        value = payload()
        with self.assertRaises(ExecutionError):
            render_personalization_instructions(
                value, authenticated_owner_id="other",
                active_conversation_id=CONVERSATION)

    def test_request_local_context_requires_active_conversation(self):
        for section in ("current_conversation", "session_corrections"):
            value = payload()
            value[section][0]["conversation_id"] = "other-conversation"
            with self.subTest(section=section), self.assertRaises(ExecutionError):
                build_personalization_context(
                    value, authenticated_owner_id=OWNER,
                    active_conversation_id=CONVERSATION)

    def test_tool_results_require_structured_attribution(self):
        value = payload()
        value["task_context"] = [item(
            "task_context", "The tool returned verified evidence.",
            source_type="tool_result", tool_name="search",
            tool_call_id="call-123")]
        rendered = render_personalization_instructions(
            value, authenticated_owner_id=OWNER,
            active_conversation_id=CONVERSATION)
        self.assertIn("tool_result:spec.md@revision-1", rendered)
        self.assertIn("tool:search call:call-123", rendered)
        value["task_context"][0].pop("tool_call_id")
        with self.assertRaises(ExecutionError):
            build_personalization_context(
                value, authenticated_owner_id=OWNER,
                active_conversation_id=CONVERSATION)

    def test_every_string_counts_toward_total_limit(self):
        value = payload()
        value["saved_memory"] = [item("saved_memory", "x", id="i" * 4096)
                                 for _ in range(16)]
        with self.assertRaises(ExecutionError):
            build_personalization_context(
                value, authenticated_owner_id=OWNER,
                active_conversation_id=CONVERSATION)

    def test_non_utf8_strings_fail_closed(self):
        value = payload()
        value["saved_memory"][0]["text"] = "\ud800"
        with self.assertRaises(ExecutionError):
            build_personalization_context(
                value, authenticated_owner_id=OWNER,
                active_conversation_id=CONVERSATION)

    def test_unhashable_task_source_type_fails_as_execution_error(self):
        for malformed in ([], {}):
            value = payload()
            value["task_context"][0]["source_type"] = malformed
            with self.subTest(malformed=malformed), self.assertRaises(ExecutionError):
                build_personalization_context(
                    value, authenticated_owner_id=OWNER,
                    active_conversation_id=CONVERSATION)

    def test_section_shape_is_bounded_before_character_counting(self):
        value = payload()
        nested = []
        for _ in range(1200):
            nested = [nested]
        value["saved_memory"] = [nested]
        with self.assertRaises(ExecutionError):
            build_personalization_context(
                value, authenticated_owner_id=OWNER,
                active_conversation_id=CONVERSATION)

        value = payload()
        value["saved_memory"] = [None] * 65
        with self.assertRaises(ExecutionError):
            build_personalization_context(
                value, authenticated_owner_id=OWNER,
                active_conversation_id=CONVERSATION)

    def test_context_cannot_override_plan_model_effort_or_execution(self):
        for field in ("tier", "model", "provider", "effort", "allowed_routes", "execution_authorized"):
            value = payload()
            value["custom_instructions"][0][field] = "pro"
            with self.subTest(field=field), self.assertRaises(ExecutionError):
                build_personalization_context(
                    value, authenticated_owner_id=OWNER,
                    active_conversation_id=CONVERSATION)

    def test_unknown_shapes_and_oversized_context_fail_closed(self):
        values = [None, [], {}, payload() | {"unknown": True}]
        for value in values:
            with self.subTest(value=value), self.assertRaises(ExecutionError):
                build_personalization_context(
                    value, authenticated_owner_id=OWNER,
                    active_conversation_id=CONVERSATION)
        value = payload()
        value["session_corrections"][0]["text"] = "x" * 4097
        with self.assertRaises(ExecutionError):
            build_personalization_context(
                value, authenticated_owner_id=OWNER,
                active_conversation_id=CONVERSATION)


if __name__ == "__main__":
    unittest.main()
