import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from deepm.live.exceptions import LiveTradingError
from deepm.live.reporting import (
    regenerate_reports_from_run_dir,
    send_daily_report_emails,
    send_email_report,
    write_run_outputs,
)
from deepm.live.types import Fill, Instrument, OrderIntent, Position, RiskCheck, Signal


def _instrument_map():
    return {
        "ES": Instrument(
            ticker="ES",
            bloomberg_ticker="ES1 Index",
            yahoo_symbol="ES=F",
            description="S&P 500 E-mini",
            ibkr_symbol="ES",
            sec_type="FUT",
            exchange="CME",
            currency="USD",
            point_value=50.0,
            tick_size=0.25,
            contract_type="FUT",
            max_contracts=2,
            reviewed=True,
        )
    }


class LiveReportingTests(unittest.TestCase):
    def test_report_writer_creates_audit_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "2026-06-25"
            report = write_run_outputs(
                run_dir=run_dir,
                run_date=date(2026, 6, 25),
                mode="sim",
                status="completed",
                summary={
                    "account_equity": 100000.0,
                    "ending_equity": 100250.0,
                    "daily_pnl": 250.0,
                    "daily_return": 0.0025,
                    "gross_exposure": 50000.0,
                    "fills": 1,
                },
                input_snapshot={"provider": "local_parquet"},
                signals=[Signal(date(2026, 6, 25), "ES", 0.5)],
                targets=[],
                orders=[
                    OrderIntent(
                        run_date=date(2026, 6, 25),
                        ticker="ES",
                        side="BUY",
                        quantity=1,
                        current_quantity=0,
                        target_quantity=1,
                        limit_price=5000.0,
                        market_price=4999.0,
                        point_value=50.0,
                        contract_value=249950.0,
                        exchange="CME",
                        currency="USD",
                    )
                ],
                fills=[
                    Fill(
                        fill_id="f1",
                        run_date=date(2026, 6, 25),
                        mode="sim",
                        ticker="ES",
                        side="BUY",
                        quantity=1,
                        price=5000.0,
                        point_value=50.0,
                        commission=0.0,
                        status="filled",
                    )
                ],
                checks=[RiskCheck("gross_leverage", True, 0.1, 1.0)],
                errors=[],
                positions=[Position("ES", 1, 5000.0, 50.0, 5010.0)],
                instruments=_instrument_map(),
            )
            self.assertTrue(report.exists())
            self.assertTrue((run_dir / "signals.csv").exists())
            self.assertTrue((run_dir / "risk_checks.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "report_detailed.html").exists())
            self.assertTrue((run_dir / "report_legacy.html").exists())
            friendly = report.read_text(encoding="utf-8")
            detailed = (run_dir / "report_detailed.html").read_text(encoding="utf-8")
            legacy = (run_dir / "report_legacy.html").read_text(encoding="utf-8")
            self.assertIn("DeePM Daily Trade Summary", friendly)
            self.assertIn("S&amp;P 500 E-mini", friendly)
            self.assertIn("Top model signals", friendly)
            self.assertIn("Risk Checks", detailed)
            self.assertIn("DeePM Daily Report", legacy)

    def test_report_regeneration_uses_existing_audit_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "2026-06-25"
            write_run_outputs(
                run_dir=run_dir,
                run_date=date(2026, 6, 25),
                mode="sim",
                status="completed",
                summary={"account_equity": 100000.0, "orders": 0, "fills": 0},
                input_snapshot={},
                signals=[Signal(date(2026, 6, 25), "ES", 0.25)],
                targets=[],
                orders=[],
                fills=[],
                checks=[],
                errors=[],
                instruments=_instrument_map(),
            )
            (run_dir / "report.html").unlink()
            report = regenerate_reports_from_run_dir(
                run_dir,
                date(2026, 6, 25),
                "sim",
                instruments=_instrument_map(),
            )
            self.assertTrue(report.exists())
            self.assertIn("S&amp;P 500 E-mini", report.read_text(encoding="utf-8"))

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

    def test_daily_report_bundle_sends_friendly_and_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "2026-06-25"
            run_dir.mkdir()
            (run_dir / "report.html").write_text("<html>friendly</html>", encoding="utf-8")
            (run_dir / "report_detailed.html").write_text("<html>detailed</html>", encoding="utf-8")
            (run_dir / "report_legacy.html").write_text("<html>legacy</html>", encoding="utf-8")
            sent = []

            def fake_send(path, email_config, run_date, mode, report_label=None):
                sent.append((path.name, report_label))

            with patch("deepm.live.reporting.send_email_report", side_effect=fake_send):
                sent_paths = send_daily_report_emails(
                    run_dir,
                    {"to": ["hkvelakaturi@gmail.com"], "send_legacy": True},
                    date(2026, 6, 25),
                    "ibkr-paper",
                )

            self.assertEqual([path.name for path in sent_paths], ["report.html", "report_legacy.html"])
            self.assertEqual(
                sent,
                [
                    ("report.html", "Friendly Summary"),
                    ("report_legacy.html", "Legacy Technical Report"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
