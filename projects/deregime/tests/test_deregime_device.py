import unittest

from deregime.config import ConfigDict, resolve_torch_device
from deregime.run import configure_runtime_device
from scripts.build_deregime_rolling_experiments import DEFAULT_BASE_OVERRIDES


class DeRegimeDeviceTests(unittest.TestCase):
    def test_auto_prefers_mps_when_cuda_is_unavailable(self):
        device, meta = resolve_torch_device(
            "auto", cuda_available=False, mps_available=True
        )

        self.assertEqual(str(device), "mps")
        self.assertFalse(meta["fallback_used"])
        self.assertTrue(meta["mps_available"])

    def test_explicit_cpu_resolves_without_fallback(self):
        device, meta = resolve_torch_device(
            "cpu", cuda_available=False, mps_available=True
        )

        self.assertEqual(str(device), "cpu")
        self.assertFalse(meta["fallback_used"])

    def test_explicit_mps_fails_when_unavailable(self):
        with self.assertRaises(RuntimeError):
            resolve_torch_device("mps", cuda_available=False, mps_available=False)

    def test_configure_runtime_device_records_request_and_resolution(self):
        cfg = ConfigDict({"device": "cpu", "allow_device_fallback": False})

        configure_runtime_device(cfg)

        self.assertEqual(cfg.requested_device, "cpu")
        self.assertEqual(cfg.resolved_device, "cpu")
        self.assertFalse(cfg.device_fallback_used)

    def test_rolling_defaults_are_resume_safe_and_auto_device(self):
        self.assertEqual(DEFAULT_BASE_OVERRIDES["device"], "auto")
        self.assertEqual(DEFAULT_BASE_OVERRIDES["checkpoint_every"], 1)
        self.assertTrue(DEFAULT_BASE_OVERRIDES["tensorize_dataset"])


if __name__ == "__main__":
    unittest.main()
