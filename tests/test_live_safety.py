import tempfile
import unittest
from datetime import date
from pathlib import Path

from deepm.live.config import assert_live_mode_allowed, ledger_path_for_mode, starting_cash_for_mode
from deepm.live.exceptions import SafetyError


class LiveSafetyTests(unittest.TestCase):
    def test_live_mode_refuses_without_config_gate(self):
        config = {"mode_defaults": {"live_enabled": False, "require_confirmation_file": True}}
        with self.assertRaises(SafetyError):
            assert_live_mode_allowed(config, "ibkr-live", date(2026, 6, 25))

    def test_live_mode_requires_same_day_confirmation_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "mode_defaults": {
                    "live_enabled": True,
                    "require_confirmation_file": True,
                    "confirmation_dir": tmp,
                }
            }
            with self.assertRaises(SafetyError):
                assert_live_mode_allowed(config, "ibkr-live", date(2026, 6, 25))

            Path(tmp, "2026-06-25.confirm").write_text("confirm\n", encoding="utf-8")
            assert_live_mode_allowed(config, "ibkr-live", date(2026, 6, 25))

    def test_sim_mode_does_not_need_live_confirmation(self):
        config = {"mode_defaults": {"live_enabled": False}}
        assert_live_mode_allowed(config, "sim", date(2026, 6, 25))

    def test_ledger_path_is_mode_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"paths": {"ledger": str(Path(tmp) / "trading_ledger.sqlite")}}
            self.assertEqual(
                ledger_path_for_mode(config, "sim"),
                Path(tmp) / "trading_ledger.sqlite",
            )
            self.assertEqual(
                ledger_path_for_mode(config, "ibkr-paper"),
                Path(tmp) / "trading_ledger_ibkr_paper.sqlite",
            )

    def test_paper_simulator_can_use_larger_fake_cash(self):
        config = {
            "ibkr": {"paper_simulation": True},
            "risk": {"simulator_cash": 100_000.0, "paper_simulator_cash": 1_000_000.0},
        }
        self.assertEqual(starting_cash_for_mode(config, "sim"), 100_000.0)
        self.assertEqual(starting_cash_for_mode(config, "ibkr-paper"), 1_000_000.0)


if __name__ == "__main__":
    unittest.main()
