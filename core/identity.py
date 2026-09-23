"""Load the exact owner-approved Kova runtime identity without trusting caller input."""

import hashlib
from pathlib import Path
from release.model_revisions import source_reference_for_route


APPROVED_PROMPT_SHA256 = "ed1b503f947cabc6a7c24a9395bd63b9eff2dd55d57570a0d5a8fd51df3c5bc8"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "kova-identity.v3.txt"


def load_runtime_identity():
    """Fail closed if the active identity differs from the reviewed prompt."""
    try:
        prompt = PROMPT_PATH.read_bytes()
    except OSError as error:
        raise ValueError("approved Kova identity prompt unavailable") from error
    if hashlib.sha256(prompt).hexdigest() != APPROVED_PROMPT_SHA256:
        raise ValueError("approved Kova identity prompt mismatch")
    try:
        return prompt.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("approved Kova identity prompt is not UTF-8") from error


def trusted_model_provenance(route_id):
    """Only the server's route registry can supply model lineage to the prompt."""
    source = source_reference_for_route(route_id)
    return ("Verified runtime provenance for direct user questions: "
            f"the selected model family is {source.slot}; its upstream model is "
            f"{source.model} at revision {source.revision}. "
            "Disclose these facts accurately when asked; do not volunteer provider details.")
