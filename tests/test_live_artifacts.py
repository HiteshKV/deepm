import tempfile
import unittest
from pathlib import Path

import pandas as pd

from deepm.live.artifacts import assert_required_runs, completed_windows, latest_completed_window
from deepm.live.exceptions import LiveTradingError


class LiveArtifactTests(unittest.TestCase):
    def test_completed_windows_require_ranked_runs_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            window = base / "2020"
            (window / "models").mkdir(parents=True)
            pd.DataFrame({"valid_loss_best": [1.0, 0.5]}, index=["run-a", "run-b"]).to_csv(
                window / "all_runs.csv"
            )
            (window / "models" / "run-a").write_text("checkpoint", encoding="utf-8")

            windows = completed_windows(base, top_n=2)
            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0].start_year, 2020)
            self.assertEqual(windows[0].run_names, ["run-a"])

    def test_latest_window_uses_asof_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for year in (2015, 2020):
                window = base / str(year)
                (window / "models").mkdir(parents=True)
                pd.DataFrame({"valid_loss_best": [1.0]}, index=[f"run-{year}"]).to_csv(
                    window / "all_runs.csv"
                )
                (window / "models" / f"run-{year}").write_text("checkpoint", encoding="utf-8")

            selected = latest_completed_window("unused", "DeePM", top_n=1, asof_year=2026, base_dir=base)
            self.assertEqual(selected.start_year, 2020)

    def test_required_run_count_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            window = completed_windows(Path(tmp), top_n=1)
            self.assertEqual(window, [])
        with self.assertRaises(LiveTradingError):
            from deepm.live.types import ModelWindow

            assert_required_runs(ModelWindow(2020, None, Path("."), ["run-a"]), top_n=2)


if __name__ == "__main__":
    unittest.main()
