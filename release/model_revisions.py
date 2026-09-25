"""Immutable private upstream lineage for the three shared Kova families."""
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ModelSourceReference:
    slot: str
    model: str
    revision: str

    @property
    def source_url(self):
        return f"https://huggingface.co/{self.model}/commit/{self.revision}"


MODEL_SOURCE_REFERENCES = MappingProxyType({
    "kova-cosmo": ModelSourceReference("kova-cosmo", "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca"),
    "kova-orion": ModelSourceReference("kova-orion", "Qwen/Qwen3-1.7B",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "kova-nova": ModelSourceReference("kova-nova", "Qwen/Qwen3-4B",
        "1cfa9a7208912126459214e8b04321603b3df60c"),
})
# Historical pilot evidence used this slot label. It is retained only to verify
# the immutable, superseded Cosmo pilot and is never a current route/model slot.
LEGACY_WORK_COSMO_REFERENCE = ModelSourceReference(
    "work-cosmo", MODEL_SOURCE_REFERENCES["kova-cosmo"].model,
    MODEL_SOURCE_REFERENCES["kova-cosmo"].revision,
)
LEVELS = ("light", "medium", "high", "extra-high", "max", "ultra")
ROUTE_MODEL_SLOTS = MappingProxyType({
    **{f"chat:{family}:{level}": f"kova-{family}"
       for family in ("cosmo", "orion") for level in LEVELS},
    **{f"work:{family}:{level}": f"kova-{family}"
       for family in ("cosmo", "orion", "nova") for level in LEVELS},
    "instant": "kova-cosmo", "medium": "kova-orion", "high": "kova-orion",
    "extra-high": "kova-orion", "max": "kova-orion", "ultra": "kova-orion",
})


def source_reference_for_route(route_id):
    if type(route_id) is not str or route_id not in ROUTE_MODEL_SLOTS:
        raise ValueError("model source route rejected")
    return MODEL_SOURCE_REFERENCES[ROUTE_MODEL_SLOTS[route_id]]
