import json
import tempfile
import unittest
from pathlib import Path

from scripts.finalize_sweep_results import usable_runs, write_window_outputs


class FinalizeSweepResultsTests(unittest.TestCase):
    def _write_run(self, root: Path, name: str, valid: float, complete: bool = True):
        (root / "settings").mkdir(parents=True, exist_ok=True)
        (root / "models").mkdir(parents=True, exist_ok=True)
        (root / "data-params").mkdir(parents=True, exist_ok=True)
        (root / "settings" / f"{name}.json").write_text(
            json.dumps(
                {
                    "valid_sharpe": valid,
                    "test_sharpe": valid / 2,
                    "test_sharpe_net": valid / 3,
                }
            ),
            encoding="utf-8",
        )
        if complete:
            (root / "models" / name).write_text("checkpoint", encoding="utf-8")
            (root / "data-params" / f"{name}.pkl").write_bytes(b"params")

    def test_usable_runs_excludes_incomplete_and_ranks_by_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, "run-low", 0.1)
            self._write_run(root, "run-high", 0.9)
            self._write_run(root, "run-incomplete", 2.0, complete=False)

            frame = usable_runs(root, summaries={})

            self.assertEqual(frame.index.tolist(), ["run-high", "run-low"])
            self.assertEqual(float(frame.loc["run-high", "valid_loss_best"]), 0.9)

    def test_write_window_outputs_writes_all_and_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, "run-a", 0.1)
            self._write_run(root, "run-b", 0.2)
            frame = usable_runs(root, summaries={})

            best = write_window_outputs(root, frame, top_n=1)

            self.assertEqual(best.index.tolist(), ["run-b"])
            self.assertTrue((root / "all_runs.csv").exists())
            self.assertTrue((root / "best_runs.csv").exists())
            self.assertTrue((root / "best_runs_mean.json").exists())


if __name__ == "__main__":
    unittest.main()
