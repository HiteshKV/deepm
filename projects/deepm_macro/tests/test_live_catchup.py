import unittest
from datetime import date, datetime, timezone

from deepm.live.catchup import latest_eligible_date, next_business_date


class LiveCatchupTests(unittest.TestCase):
    def setUp(self):
        self.config = {"schedule": {"run_after_utc": "22:15"}}

    def test_before_close_uses_previous_weekday(self):
        now = datetime(2026, 7, 24, 10, 48, tzinfo=timezone.utc)
        self.assertEqual(latest_eligible_date(self.config, now), date(2026, 7, 23))

    def test_after_close_uses_current_weekday(self):
        now = datetime(2026, 7, 24, 22, 15, tzinfo=timezone.utc)
        self.assertEqual(latest_eligible_date(self.config, now), date(2026, 7, 24))

    def test_weekend_uses_friday(self):
        now = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(latest_eligible_date(self.config, now), date(2026, 7, 24))

    def test_next_business_date_skips_weekend(self):
        self.assertEqual(next_business_date(date(2026, 7, 24)), date(2026, 7, 27))


if __name__ == "__main__":
    unittest.main()
