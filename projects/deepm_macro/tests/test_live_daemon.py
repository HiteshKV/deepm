import argparse
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from deepm.live.daemon import run_once, should_run_now


class LiveDaemonTests(unittest.TestCase):
    def test_schedule_waits_until_configured_utc_time(self):
        config = {"schedule": {"run_after_utc": "22:15", "weekdays_only": True}}
        self.assertFalse(should_run_now(config, datetime(2026, 6, 26, 22, 14, tzinfo=timezone.utc)))
        self.assertTrue(should_run_now(config, datetime(2026, 6, 26, 22, 15, tzinfo=timezone.utc)))

    def test_schedule_skips_weekends(self):
        config = {"schedule": {"run_after_utc": "00:00", "weekdays_only": True}}
        self.assertFalse(should_run_now(config, datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc)))

    @patch("deepm.live.daemon.catch_up", return_value=Path("/tmp/backfill.csv"))
    @patch("deepm.live.daemon.load_live_config")
    def test_local_paper_daemon_resumes_from_ledger(self, load_config, catch_up):
        load_config.return_value = {"ibkr": {"paper_simulation": True}}
        args = argparse.Namespace(
            mode="ibkr-paper",
            config="paper.yaml",
            date=None,
            run_now=False,
            send_email=True,
            no_place_orders=False,
            force=False,
        )

        self.assertEqual(run_once(args), Path("/tmp/backfill.csv"))
        catch_up.assert_called_once()


if __name__ == "__main__":
    unittest.main()
