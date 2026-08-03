import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from deepm.live.data_update import latest_snapshot_path, persist_live_snapshot


class LiveDataUpdateTests(unittest.TestCase):
    def test_persist_live_snapshot_writes_dated_and_latest_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "data": {
                    "live_snapshot_dir": tmp,
                    "latest_snapshot_name": "latest_prices.parquet",
                }
            }
            prices = pd.DataFrame(
                {"ES": [100.0, 101.0]},
                index=pd.to_datetime(["2026-06-24", "2026-06-25"]),
            )
            latest = persist_live_snapshot(prices, config, date(2026, 6, 25), "unit")

            self.assertEqual(latest, latest_snapshot_path(config))
            self.assertTrue(latest.exists())
            self.assertTrue((Path(tmp) / "prices_20260625.parquet").exists())
            self.assertTrue((Path(tmp) / "latest_snapshot.json").exists())


if __name__ == "__main__":
    unittest.main()
