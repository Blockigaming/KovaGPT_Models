"""Inactive three-family source registry. No model or provider is activated here."""

from release.model_revisions import MODEL_SOURCE_REFERENCES


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
