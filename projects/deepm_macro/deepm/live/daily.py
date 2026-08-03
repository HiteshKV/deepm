"""Daily simulator / paper / live trading CLI."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from deepm.live.brokers import IBKRBroker, SimBroker
from deepm.live.config import (
    DEFAULT_CONFIG,
    assert_live_mode_allowed,
    ledger_path_for_mode,
    load_live_config,
    mode_provider,
    path_from_config,
    resolve_project_path,
    starting_cash_for_mode,
)
from deepm.live.data_update import persist_live_snapshot
from deepm.live.features import feature_freshness_check, latest_feature_rows, load_or_build_features
from deepm.live.instruments import load_instruments
from deepm.live.ledger import PortfolioStore
from deepm.live.providers import provider_from_name
from deepm.live.reporting import send_daily_report_emails, write_run_outputs
from deepm.live.signal import SignalEngine
from deepm.live.sizing import assert_risk_checks_pass, build_orders, build_targets, risk_checks
from deepm.live.types import (
    Fill,
    Instrument,
    OrderIntent,
    Position,
    RiskCheck,
    Signal,
    TargetPosition,
    dataclass_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DeePM daily simulator/paper/live pipeline")
    parser.add_argument("--mode", choices=["sim", "ibkr-paper", "ibkr-live"], default="sim")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--send-email", action="store_true")
    parser.add_argument("--no-place-orders", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _latest_price_maps(rows: pd.DataFrame, instruments: dict[str, Instrument]) -> tuple[dict[str, float], dict[str, float]]:
    latest_prices: dict[str, float] = {}
    point_values: dict[str, float] = {}
    for ticker, instrument in instruments.items():
        if ticker not in rows.index:
            continue
        row = rows.loc[ticker]
        price = float(row.get("close", row.get("srs", 0.0)) or 0.0)
        if price > 0:
            latest_prices[ticker] = price
            point_values[ticker] = instrument.point_value
    return latest_prices, point_values


def _uses_local_ledger(config: dict[str, object], mode: str) -> bool:
    if mode == "sim":
        return True
    if mode == "ibkr-paper":
        return bool(config.get("ibkr", {}).get("paper_simulation", True))
    return False


def _broker_for_mode(
    mode: str,
    config: dict[str, object],
    store: PortfolioStore,
    instruments: dict[str, Instrument],
):
    if _uses_local_ledger(config, mode):
        return SimBroker(
            store,
            instruments,
            mode=mode,
            slippage_bps=float(config["risk"].get("default_slippage_bps", 5.0)),
        ), ("local_simulator" if mode == "sim" else "ibkr_paper_simulator")
    return IBKRBroker(mode, config.get("ibkr", {}), instruments), "ibkr_gateway"


def _risk_config_for_adapter(config: dict[str, object], broker_adapter: str) -> dict[str, object]:
    """Return risk settings with local-paper-simulator overrides applied."""
    risk = dict(config["risk"])
    if broker_adapter == "ibkr_paper_simulator":
        overrides = {
            "max_daily_turnover": risk.get("paper_simulator_max_daily_turnover"),
            "max_gross_leverage": risk.get("paper_simulator_max_gross_leverage"),
            "max_order_notional": risk.get("paper_simulator_max_order_notional"),
        }
        for key, value in overrides.items():
            if value is not None:
                risk[key] = value
    return risk


def run_daily(
    mode: str,
    config_path: str | Path,
    run_date: date,
    send_email: bool = False,
    no_place_orders: bool = False,
    force: bool = False,
) -> Path:
    config = load_live_config(config_path)
    assert_live_mode_allowed(config, mode, run_date)

    run_root = path_from_config(config, "paths", "run_dir")
    run_dir = run_root / run_date.isoformat()
    state_dir = path_from_config(config, "paths", "state_dir")
    state_dir.mkdir(parents=True, exist_ok=True)

    instruments = load_instruments(path_from_config(config, "paths", "instrument_map"))
    raw_price_path = resolve_project_path(config["data"]["raw_price_parquet"])
    provider_name = mode_provider(config, mode)
    provider = provider_from_name(provider_name, raw_price_path)
    store = PortfolioStore(ledger_path_for_mode(config, mode), starting_cash_for_mode(config, mode))

    signals: list[Signal] = []
    targets: list[TargetPosition] = []
    orders: list[OrderIntent] = []
    fills: list[Fill] = []
    positions: list[Position] = []
    checks: list[RiskCheck] = []
    errors: list[str] = []
    summary: dict[str, object] = {"mode": mode, "date": run_date.isoformat()}
    input_snapshot: dict[str, object] = {
        "config": str(resolve_project_path(config_path)),
        "provider": provider_name,
        "raw_price_parquet": str(raw_price_path),
        "ledger": str(ledger_path_for_mode(config, mode)),
    }

    status = "failed"
    try:
        if _uses_local_ledger(config, mode) and store.has_completed_run(run_date, mode) and not force:
            status = "already_ran"
            summary["message"] = "Simulator run was already completed; no duplicate fills were created."
            report_path = write_run_outputs(
                run_dir, run_date, mode, status, summary, input_snapshot,
                signals, targets, orders, fills, checks, errors, positions,
                instruments=instruments,
            )
            return report_path

        raw_prices = provider.load_history(
            instruments,
            start_date=str(config["data"]["history_start_date"]),
            end_date=run_date,
        )
        feature_source_path = raw_price_path
        if bool(config.get("data", {}).get("persist_live_snapshot", True)):
            feature_source_path = persist_live_snapshot(raw_prices, config, run_date, provider_name)
            input_snapshot["live_snapshot_parquet"] = str(feature_source_path)
        features = load_or_build_features(raw_prices, feature_source_path, run_date)
        fresh, stale_days = feature_freshness_check(
            features,
            run_date,
            int(config["schedule"].get("max_data_staleness_days", 3)),
        )
        checks.append(
            RiskCheck(
                name="feature_freshness",
                passed=fresh,
                value=stale_days,
                limit=config["schedule"].get("max_data_staleness_days", 3),
            )
        )
        if not fresh:
            raise RuntimeError(f"Feature data is stale by {stale_days} days")

        latest_rows = latest_feature_rows(features, run_date)
        latest_prices, point_values = _latest_price_maps(latest_rows, instruments)

        engine = SignalEngine(config["model"])
        signals, window = engine.generate(features, list(instruments), run_date)
        input_snapshot["model_window"] = dataclass_to_dict(window)
        input_snapshot["feature_latest_date"] = str(pd.Timestamp(features.index.max()).date())

        broker, broker_adapter = _broker_for_mode(mode, config, store, instruments)
        summary["broker_adapter"] = broker_adapter
        current_positions = broker.current_positions(latest_prices)
        account_equity = broker.account_equity(latest_prices)
        previous_equity = store.previous_equity(run_date, mode) if _uses_local_ledger(config, mode) else None
        summary["account_equity"] = round(account_equity, 2)
        summary["previous_equity"] = None if previous_equity is None else round(previous_equity, 2)
        active_risk = _risk_config_for_adapter(config, broker_adapter)

        targets, input_checks = build_targets(
            signals,
            latest_rows,
            instruments,
            current_positions,
            account_equity,
            active_risk,
        )
        checks.extend(input_checks)

        orders = build_orders(targets, instruments, run_date, active_risk, account_equity)
        risk_mode = "sim" if broker_adapter == "ibkr_paper_simulator" else mode
        checks.extend(risk_checks(targets, orders, account_equity, active_risk, risk_mode))
        assert_risk_checks_pass(checks)

        if no_place_orders:
            status = "dry_run"
        elif orders:
            fills = broker.execute_orders(orders, run_date)
            status = "completed"
        else:
            status = "completed"
        positions = list(broker.current_positions(latest_prices).values())
        if _uses_local_ledger(config, mode) and status == "completed":
            end_equity = broker.account_equity(latest_prices)
            pnl_base = previous_equity if previous_equity is not None else account_equity
            summary["ending_equity"] = round(end_equity, 2)
            summary["daily_pnl"] = round(end_equity - pnl_base, 2)
            summary["daily_return"] = round((end_equity / pnl_base) - 1.0, 8) if pnl_base else None
            summary["cash"] = round(store.cash(), 2)
            summary["position_count"] = len(positions)
            summary["gross_exposure"] = round(
                sum(abs(position.market_value) for position in positions),
                2,
            )
            store.record_equity(run_date, mode, end_equity)
            store.mark_run(run_date, mode, status)
        summary["signals"] = len(signals)
        summary["orders"] = len(orders)
        summary["fills"] = len(fills)
    except Exception as exc:
        errors.append(str(exc))
        status = "failed"
        if _uses_local_ledger(config, mode):
            store.mark_run(run_date, mode, status)

    report_path = write_run_outputs(
        run_dir,
        run_date,
        mode,
        status,
        summary,
        input_snapshot,
        signals,
        targets,
        orders,
        fills,
        checks,
        errors,
        positions,
        instruments=instruments,
    )
    email_status = "not_requested"
    if send_email:
        try:
            sent_reports = send_daily_report_emails(
                run_dir,
                config.get("email", {}),
                run_date,
                mode,
            )
            email_status = f"sent:{len(sent_reports)}"
        except Exception as exc:
            email_status = "failed"
            errors.append(f"Email failed: {exc}")
    status_path = run_dir / "status.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    status_payload["errors"] = errors
    status_payload["email_status"] = email_status
    status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    report_path = run_daily(
        mode=args.mode,
        config_path=args.config,
        run_date=date.fromisoformat(args.date),
        send_email=args.send_email,
        no_place_orders=args.no_place_orders,
        force=args.force,
    )
    print(report_path)


if __name__ == "__main__":
    main()
