import unittest
from datetime import datetime, timezone

from deepm.live.daemon import should_run_now


class LiveDaemonTests(unittest.TestCase):
    def test_schedule_waits_until_configured_utc_time(self):
        config = {"schedule": {"run_after_utc": "22:15", "weekdays_only": True}}
        self.assertFalse(should_run_now(config, datetime(2026, 6, 26, 22, 14, tzinfo=timezone.utc)))
        self.assertTrue(should_run_now(config, datetime(2026, 6, 26, 22, 15, tzinfo=timezone.utc)))

    def test_schedule_skips_weekends(self):
        config = {"schedule": {"run_after_utc": "00:00", "weekdays_only": True}}
        self.assertFalse(should_run_now(config, datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc)))


if __name__ == "__main__":
    unittest.main()
