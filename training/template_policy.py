"""One explicit template mode for SFT, loss masks and diagnostic generation.

TRL 1.13.0 reads chat_template_kwargs from each conversational dataset row,
not from SFTConfig. This changes future source preparation only; historical
adapters and receipts retain their original training configuration.
"""

CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def template_row(prompt: list[dict], completion: list[dict]) -> dict:
    return {"prompt": prompt, "completion": completion,
            "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS)}
