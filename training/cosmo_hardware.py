"""Fail-closed NVIDIA T4 identity checks shared by Cosmo GPU entrypoints."""
from __future__ import annotations


ALLOWED_T4_DEVICE_NAMES = frozenset({
    "tesla t4",
    "nvidia t4",
    "nvidia tesla t4",
})


class HardwareError(ValueError):
    pass


def need(condition: bool) -> None:
    if not condition:
        raise HardwareError("kova cosmo hardware rejected")


def verify_nvidia_t4(torch_module) -> dict:
    """Require CUDA capability 7.5 and an exact NVIDIA T4 device identity."""
    try:
        cuda = torch_module.cuda
        need(cuda.is_available() is True)
        capability = cuda.get_device_capability(0)
        need(type(capability) in (tuple, list) and
             tuple(capability) == (7, 5))
        device_name = cuda.get_device_name(0)
        need(type(device_name) is str)
        normalized = " ".join(device_name.strip().casefold().split())
        need(normalized in ALLOWED_T4_DEVICE_NAMES)
        return {
            "device_name": device_name,
            "device_capability": [7, 5],
            "gpu_family": "NVIDIA T4",
        }
    except (AttributeError, TypeError, ValueError, RuntimeError):
        raise HardwareError("kova cosmo hardware rejected") from None
