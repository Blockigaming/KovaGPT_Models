"""Reject private reasoning and decoded-token side channels at the Azure boundary.

This is not a reasoning-to-answer converter. Visible answers and numeric usage
are left intact; a provider that ignores suppression is rejected, not silently
laundered into successful output. There are no logs, callbacks, or network calls.
"""

from worker.openai_protocol import OpenAIProtocolError


_REASONING_FIELDS = ("reasoning", "reasoning_content", "reasoning_details")
_TOKEN_FIELDS = ("logprobs", "prompt_logprobs", "token_ids", "prompt_token_ids")


def _check_mapping(value):
    if not isinstance(value, dict):
        return
    for field in _REASONING_FIELDS:
        if field in value:
            private = value[field]
            if private is not None and not (type(private) is str and private == ""):
                raise OpenAIProtocolError("engine returned hidden reasoning")
    for field in _TOKEN_FIELDS:
        if field in value and value[field] is not None:
            raise OpenAIProtocolError("engine returned private token metadata")


def require_private_reasoning_absent(value):
    """Inspect known Chat Completions envelope/message/delta locations.

    Do not recurse into user-visible text, serialized tool arguments, or numeric
    usage details: a user can legitimately ask about the word 'reasoning'. This
    check complements, and never substitutes for, the existing response sanitizer.
    No message, choice, tool call, usage count or completion marker is modified.
    """
    _check_mapping(value)
    choices = value.get("choices") if isinstance(value, dict) else None
    if not isinstance(choices, list):
        return
    for choice in choices:
        _check_mapping(choice)
        if isinstance(choice, dict):
            _check_mapping(choice.get("message"))
            _check_mapping(choice.get("delta"))
