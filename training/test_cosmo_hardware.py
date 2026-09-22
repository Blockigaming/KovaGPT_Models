"""Regression tests for exact NVIDIA T4 runtime identity enforcement."""
from types import SimpleNamespace
import unittest

from training import cosmo_hardware as hardware


class FakeCuda:
    def __init__(self, name="Tesla T4", capability=(7, 5), available=True):
        self.name = name
        self.capability = capability
        self.available = available

    def is_available(self):
        return self.available

    def get_device_capability(self, _index):
        return self.capability

    def get_device_name(self, _index):
        return self.name


class CosmoHardwareTests(unittest.TestCase):
    def torch(self, **changes):
        return SimpleNamespace(cuda=FakeCuda(**changes))

    def test_exact_nvidia_t4_names_and_capability_are_accepted(self):
        for name in ("Tesla T4", "NVIDIA T4", "NVIDIA Tesla T4"):
            with self.subTest(name=name):
                report = hardware.verify_nvidia_t4(self.torch(name=name))
                self.assertEqual(report["gpu_family"], "NVIDIA T4")
                self.assertEqual(report["device_capability"], [7, 5])

    def test_other_turing_gpu_with_same_capability_is_rejected(self):
        for name in ("GeForce RTX 2080 Ti", "Quadro RTX 8000", "T4-like GPU"):
            with self.subTest(name=name), self.assertRaises(
                hardware.HardwareError
            ):
                hardware.verify_nvidia_t4(self.torch(name=name))

    def test_name_alone_cannot_override_capability_or_cuda_availability(self):
        for changes in (
            {"name": "Tesla T4", "capability": (8, 0)},
            {"name": "Tesla T4", "available": False},
        ):
            with self.subTest(changes=changes), self.assertRaises(
                hardware.HardwareError
            ):
                hardware.verify_nvidia_t4(self.torch(**changes))


if __name__ == "__main__":
    unittest.main()
