"""Offline template/token boundary probe; no weights, downloads or training.

Run with tokenizers==0.23.1 and jinja2==3.1.6:
python -m training.probe_cosmo_tokenizer /path/to/tokenizer-assets
This does not verify TRL collator loss masks or GPU compatibility.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from training.kova_cosmo_sft import prepare_sft_rows

ASSETS = {
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def probe(directory):
    for package, expected in (("tokenizers", "0.23.1"), ("jinja2", "3.1.6")):
        if version(package) != expected:
            raise ValueError("probe dependency mismatch")
    data = {}
    for name, expected in ASSETS.items():
        body = (directory / name).read_bytes()
        if hashlib.sha256(body).hexdigest() != expected:
            raise ValueError("tokenizer asset mismatch")
        data[name] = body
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def reject(message):
        raise ValueError(message)

    environment.globals["raise_exception"] = reject
    template = environment.from_string(json.loads(data["tokenizer_config.json"])["chat_template"])
    tokenizer = Tokenizer.from_str(data["tokenizer.json"].decode("utf-8"))
    train, validation = prepare_sft_rows()
    results = []
    for split, rows in (("train", train), ("validation", validation)):
        for index, row in enumerate(rows):
            prompt = template.render(messages=row["prompt"], add_generation_prompt=True)
            full = template.render(messages=row["prompt"] + row["completion"], add_generation_prompt=False)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            full_ids = tokenizer.encode(full, add_special_tokens=False).ids
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError("prompt token prefix mismatch")
            if not 0 < len(prompt_ids) < len(full_ids) <= 1024:
                raise ValueError("empty completion or sequence exceeds recipe limit")
            results.append({"split": split, "index": index, "tokens": len(full_ids),
                            "prompt_tokens": len(prompt_ids),
                            "completion_tokens": len(full_ids) - len(prompt_ids)})
    return {"status": "offline_token_boundaries_verified", "source_revision": REVISION,
            "asset_sha256": ASSETS, "records": results,
            "trl_loss_masks_verified": False, "gpu_verified": False,
            "training_started": False, "phase_b_ready": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    args = parser.parse_args()
    print(json.dumps(probe(args.assets), sort_keys=True))
