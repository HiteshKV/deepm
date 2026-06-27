import tempfile
import unittest
from pathlib import Path

from scripts.generate_deepm_report import validate_inputs


class GenerateDeepmReportTests(unittest.TestCase):
    def test_validate_inputs_fails_on_missing_required_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SystemExit) as ctx:
                validate_inputs(
                    diagnostics_dir=root / "diagnostics",
                    metrics_csv=root / "metrics.csv",
                    report_dir=root / "report",
                )
            self.assertIn("Missing required report inputs", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
