import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from deepm.live.exceptions import LiveTradingError
from deepm.live.reporting import send_email_report, write_run_outputs
from deepm.live.types import RiskCheck, Signal


class LiveReportingTests(unittest.TestCase):
    def test_report_writer_creates_audit_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "2026-06-25"
            report = write_run_outputs(
                run_dir=run_dir,
                run_date=date(2026, 6, 25),
                mode="sim",
                status="completed",
                summary={"account_equity": 100000.0},
                input_snapshot={"provider": "local_parquet"},
                signals=[Signal(date(2026, 6, 25), "ES", 0.5)],
                targets=[],
                orders=[],
                fills=[],
                checks=[RiskCheck("gross_leverage", True, 0.1, 1.0)],
                errors=[],
            )
            self.assertTrue(report.exists())
            self.assertTrue((run_dir / "signals.csv").exists())
            self.assertTrue((run_dir / "risk_checks.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertIn("DeePM Daily Report", report.read_text(encoding="utf-8"))

    def test_email_requires_env_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.html"
            report.write_text("<html></html>", encoding="utf-8")
            with patch.dict("os.environ", {"DEEPM_DISABLE_DOTENV": "1"}, clear=True), self.assertRaises(LiveTradingError):
                send_email_report(report, {"to": ["hkvelakaturi@gmail.com"]}, date(2026, 6, 25), "sim")

    def test_resend_provider_requires_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.html"
            report.write_text("<html></html>", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {"DEEPM_EMAIL_PROVIDER": "resend", "DEEPM_DISABLE_DOTENV": "1"},
                clear=True,
            ):
                with self.assertRaisesRegex(LiveTradingError, "RESEND_API_KEY"):
                    send_email_report(
                        report,
                        {"to": ["hkvelakaturi@gmail.com"]},
                        date(2026, 6, 25),
                        "ibkr-paper",
                    )

    def test_unknown_email_provider_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.html"
            report.write_text("<html></html>", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {"DEEPM_EMAIL_PROVIDER": "unknown", "DEEPM_DISABLE_DOTENV": "1"},
                clear=True,
            ):
                with self.assertRaisesRegex(LiveTradingError, "Unknown DEEPM_EMAIL_PROVIDER"):
                    send_email_report(
                        report,
                        {"to": ["hkvelakaturi@gmail.com"]},
                        date(2026, 6, 25),
                        "ibkr-paper",
                    )


if __name__ == "__main__":
    unittest.main()
