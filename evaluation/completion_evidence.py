"""Bounded generation evidence; never infers completion from answer wording."""

from __future__ import annotations

import hashlib
import json

MAX_DIAGNOSTIC_TOKENS = 4096
STRICT_MAX_NEW_TOKENS = 128


def output_budgets(cases: list[dict], manual_max_new_tokens: int) -> tuple[int, ...]:
    """A manual budget must be explicitly supplied, even for an offline run."""
    validate_budget(manual_max_new_tokens)
    return tuple(manual_max_new_tokens if case["evaluation"]["kind"] ==
                 "review_required" else STRICT_MAX_NEW_TOKENS for case in cases)


def validate_budget(value: int) -> None:
    if type(value) is not int or not 1 <= value <= MAX_DIAGNOSTIC_TOKENS:
        raise ValueError("explicit bounded output budget required")


def generation_evidence(token_ids: list[int], *, max_new_tokens: int,
                        eos_token_id: int | list[int]) -> dict:
    """Count raw generated tokens, including EOS, before special-token stripping.

    Greedy single-sequence generation must end in a configured EOS to establish
    completion. A shorter response without EOS is an unexplained stop, not a
    successful completion. EOS at the final budget token is a completed answer.
    """
    validate_budget(max_new_tokens)
    eos = [eos_token_id] if type(eos_token_id) is int else eos_token_id
    if (type(eos) is not list or not eos or
            any(type(token) is not int or token < 0 for token in eos) or
            len(set(eos)) != len(eos) or type(token_ids) is not list or
            not 0 < len(token_ids) <= max_new_tokens or
            any(type(token) is not int or token < 0 for token in token_ids)):
        raise ValueError("invalid generation token evidence")
    if any(token in eos for token in token_ids[:-1]):
        raise ValueError("tokens generated after EOS")
    reason = ("eos" if token_ids[-1] in eos else "length_limit" if
              len(token_ids) == max_new_tokens else "unexplained_stop")
    return {"generated_token_count": len(token_ids),
            "max_new_tokens": max_new_tokens, "finish_reason": reason,
            "terminal_token_id": token_ids[-1], "eos_token_ids": eos,
            "generated_token_ids_sha256": hashlib.sha256(json.dumps(
                token_ids, separators=(",", ":")).encode()).hexdigest()}


def completion_status(evidence: dict | None) -> str:
    """Fail closed for absent, inconsistent, or unrecognised evidence."""
    if type(evidence) is not dict:
        return "unverified"
    fields = {"generated_token_count", "max_new_tokens", "finish_reason",
              "terminal_token_id", "eos_token_ids", "generated_token_ids_sha256"}
    if set(evidence) != fields:
        return "unverified"
    count, budget = evidence["generated_token_count"], evidence["max_new_tokens"]
    eos, terminal = evidence["eos_token_ids"], evidence["terminal_token_id"]
    sha = evidence["generated_token_ids_sha256"]
    if (type(count) is not int or type(budget) is not int or
            not 0 < count <= budget <= MAX_DIAGNOSTIC_TOKENS or
            type(terminal) is not int or terminal < 0 or
            type(eos) is not list or not eos or
            any(type(token) is not int or token < 0 for token in eos) or
            len(set(eos)) != len(eos) or
            type(sha) is not str or len(sha) != 64 or
            any(c not in "0123456789abcdef" for c in sha)):
        return "unverified"
    if evidence["finish_reason"] == "eos" and terminal in eos:
        return "verified_complete"
    if (evidence["finish_reason"] == "length_limit" and terminal not in eos
            and count == budget):
        return "length_limit"
    return "unverified"
