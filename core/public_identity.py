"""Product identity at delivery boundaries; never a model-quality answer repair.

Raw evaluation output must be scored and retained before this delivery-only
guard. No model calls, regeneration, network activity or raw-output logging.
"""
import html
import base64
import re
import unicodedata
from urllib.parse import unquote

IDENTITY = "I’m KovaGPT, built by Kova."
PROVENANCE_REFUSAL = IDENTITY + " I don’t provide internal model-provenance details."
SAFE_ERROR = IDENTITY + " I couldn’t complete that request."
from core.private_provenance import load_catalog, PrivateCatalogError

_CONFUSABLES = str.maketrans({"а":"a", "е":"e", "о":"o", "р":"p", "с":"c",
    "х":"x", "у":"y", "і":"i", "ј":"j", "ԛ":"q", "ԝ":"w", "ԁ":"d", "п":"n"})


def _normalized(text):
    # Decode standard visible encodings before matching; do not execute content.
    for _ in range(3):
        decoded = html.unescape(unquote(text))
        decoded = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)), decoded)
        if decoded == text:
            break
        text = decoded
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold().translate(_CONFUSABLES)
                   if c.isalnum())


def _alias_keys():
    return tuple(_normalized(alias) for alias in load_catalog()["aliases"])


def contains_prohibited(value):
    """Check complete text or a structured visible payload, including keys."""
    if type(value) is str:
        try:
            keys = _alias_keys()
        except PrivateCatalogError:
            return True
        folded = _normalized(value)
        if any(alias in folded for alias in keys):
            return True
        for token in re.findall(r"[A-Za-z0-9+/]{4,512}={0,2}", value):
            if len(token) % 4 == 0:
                try:
                    decoded = base64.b64decode(token, validate=True).decode("utf-8")
                except (ValueError, UnicodeError):
                    continue
                if any(alias in _normalized(decoded) for alias in keys):
                    return True
        # Repository paths/revisions labelled as internal origins must not leak.
        return bool(re.search(r"(?i)(?:upstream|base[- ]?model|model[- ]?repository|"
                              r"model[- ]?revision)\s*[:=]\s*\S+|"
                              r"(?:i am|i['’]m|this (?:model|assistant))\s+(?:based on|"
                              r"powered by|fine[- ]?tuned from|built on)\s+\S+", value))
    if type(value) is dict:
        return any(contains_prohibited(k) or contains_prohibited(v) for k, v in value.items())
    if type(value) in (list, tuple):
        return any(contains_prohibited(v) for v in value)
    return False


def identity_reply(messages):
    """Deterministic common identity intents; the system policy covers all turns."""
    users = [m.get("content", "") for m in (messages or []) if m.get("role") == "user"]
    question = users[-1].casefold().strip() if users else ""
    if not question:
        return None
    if contains_prohibited(question) and re.search(r"are you|made by|built by|created by", question):
        return IDENTITY
    if re.search(r"(?:your|you|model|weights|runtime).*(?:base model|upstream|provenance|"
                 r"originally|built on|weights)|(?:base model|upstream|provenance).*"
                 r"(?:your|you|model|weights)", question):
        return PROVENANCE_REFUSAL
    if re.search(r"who (?:are|built|made|created|developed) you|what (?:model|assistant) are you|"
                 r"what(?:'s| is) your (?:name|identity)|"
                 r"are you made by|qui es.tu|qui (?:vous|t.a) (?:êtes|créé)|"
                 r"quién eres|quién te (?:creó|construyó)|wer bist du|wer hat dich|"
                 r"你是誰|你是谁|你是什么模型|你是什麼模型|誰が作|あなたは誰|"
                 r"من أنت|من صنعك|кто ты|кто тебя создал|आप कौन|किसने बनाया", question):
        return IDENTITY
    return None


def guard_message(content, tool_calls, messages=None):
    """Replace the entire rejected message; never edit answers or tool arguments."""
    blocked = contains_prohibited({"content": content, "tool_calls": tool_calls})
    reply = identity_reply(messages)
    if blocked:
        return (reply or PROVENANCE_REFUSAL, [], True)
    if reply is not None:
        return reply, [], False
    if (isinstance(content, str) and not content.startswith(IDENTITY)
            and re.match(r"(?i)^\s*(?:i['’]m|i am|you.re speaking with)\s+kova(?:gpt)?\b", content)):
        return IDENTITY, [], True
    return content, tool_calls, False


def public_usage(usage):
    """Numeric accounting only: engine strings/unknown metadata are not public."""
    def numbers(value):
        if type(value) is int and value >= 0:
            return value
        if type(value) is dict:
            return {k: cleaned for k, v in value.items()
                    if type(k) is str and re.fullmatch(r"[a-z_]{1,64}", k)
                    and not contains_prohibited(k) and (cleaned := numbers(v)) is not None}
        return None
    return {k: cleaned for k in ("prompt_tokens", "completion_tokens", "total_tokens",
                                "prompt_tokens_details", "completion_tokens_details")
            if k in usage and (cleaned := numbers(usage[k])) is not None}


def public_error_payload(_error):
    """Never serialize exception text, class, traceback or provider response."""
    return {"error": {"code": "kova_request_failed", "message": SAFE_ERROR}}


def public_answer_payload(value):
    """Allowlist the complete customer response; private runtime data stays private."""
    content, calls, blocked = guard_message(value.get("content", ""), value.get("tool_calls", []))
    payload = {"content": content, "tool_calls": calls}
    if "usage" in value:
        payload["usage"] = public_usage(value["usage"])
    return payload
