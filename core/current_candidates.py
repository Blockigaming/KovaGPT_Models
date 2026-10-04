"""Inactive three-family source registry. No model or provider is activated here."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ModelSourceReference:
    slot: str
    model: str
    revision: str



MODEL_SOURCE_REFERENCES = MappingProxyType({
    "kova-cosmo": ModelSourceReference("kova-cosmo", "internal-cosmo-base-v1",
        "c1899de289a04d12100db370d81485cdf75e47ca"),
    "kova-orion": ModelSourceReference("kova-orion", "internal-orion-base-v1",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "kova-nova": ModelSourceReference("kova-nova", "internal-nova-base-v1",
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



# Native upstream context, without extended-context scaling. Paid probing must
# still confirm the actual server limit before any model is served.
NATIVE_CONTEXT_TOKENS = 32768
# A trained adapter digest is populated only after a separately reviewed,
# preserved artifact exists. None blocks serving every family today.
TRAINED_ADAPTER_SHA256 = {family: None for family in MODEL_SOURCE_REFERENCES}
# SHA-256 of the independently reviewed full artifact manifest bytes. This
# binds adapter_config.json and adapter_model.safetensors (and the base bundle)
# to one candidate; the weights digest alone cannot identify a loaded adapter.
ADAPTER_BUNDLE_SHA256 = {family: None for family in MODEL_SOURCE_REFERENCES}
CORE_SERVING = {
    "endpoint_name_reserved": "kova-core",
    "candidates": [
        {
            "id": family,
            "model": family,
            "revision": source.revision,
            "adapter_sha256": TRAINED_ADAPTER_SHA256[family],
            "adapter_bundle_sha256": ADAPTER_BUNDLE_SHA256[family],
            "quantization": "four_bit_nf4",
            "context_tokens": NATIVE_CONTEXT_TOKENS,
        }
        for family, source in MODEL_SOURCE_REFERENCES.items()
    ],
}
