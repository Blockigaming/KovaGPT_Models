"""Validate bounded user/assistant text history without promoting its authority.

This is not authentication, project retrieval or a tool-result transport. The
application must select history belonging to the authenticated conversation.
System/developer/tool messages and all metadata are deliberately unsupported.
"""

MAX_MESSAGES = 256
MAX_MESSAGE_CHARS = 250_000
MAX_TOTAL_CHARS = 750_000
ARTIFACT_PREFIX = "{{server_stage_output:"


def validated_conversation(value):
    """Copy exact text/order or reject; never summarize or truncate silently.

    The final user turn identifies the current task. Consecutive turns are allowed
    so existing applications are not forced to fabricate alternating messages.
    Reserved internal artifact markers are rejected before template construction.
    """
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_MESSAGES:
        raise ValueError("Ultra messages must contain 1 to 256 text turns")
    cleaned = []
    total = 0
    for message in value:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("unsupported Ultra conversation message fields")
        role, content = message["role"], message["content"]
        if not isinstance(role, str) or role not in ("user", "assistant"):
            raise ValueError("unsupported Ultra conversation role")
        if not isinstance(content, str) or len(content) > MAX_MESSAGE_CHARS:
            raise ValueError("invalid Ultra conversation text")
        total += len(content)
        if total > MAX_TOTAL_CHARS:
            raise ValueError("Ultra conversation exceeds the aggregate text limit")
        if ARTIFACT_PREFIX in content:
            raise ValueError("reserved artifact marker in Ultra conversation")
        try:
            content.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("Ultra conversation is not valid UTF-8") from None
        cleaned.append({"role": role, "content": content})
    if cleaned[-1]["role"] != "user" or not cleaned[-1]["content"].strip():
        raise ValueError("Ultra conversation must end with a nonempty user task")
    return cleaned
