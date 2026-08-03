# Cleanup Manifest - 2026-07-10

## Moved active artifacts
- Root DeePM macro code/configs/scripts/tests/data/results/live state moved into projects/deepm_macro/.
- Standalone DeRegiME project moved into projects/deregime/.
- Stock top-10 strategy moved into projects/stock_top10_strategy/.
- Hourly equity DeePM pilot moved into projects/equity_intraday_deepm/.
- Active DeRegiME run outputs moved from deregime_runs/ to projects/deregime/deregime_runs/.

## Deleted generated caches and stale outputs

### Python caches
projects/deepm_macro/deepm/__pycache__
projects/deepm_macro/deepm/backtest/__pycache__
projects/deepm_macro/deepm/backtest/models/__pycache__
projects/deepm_macro/deepm/configs/__pycache__
projects/deepm_macro/deepm/data/__pycache__
projects/deepm_macro/deepm/experiments/__pycache__
projects/deepm_macro/deepm/live/__pycache__
projects/deepm_macro/deepm/models/__pycache__
projects/deepm_macro/deepm/training/__pycache__
projects/deepm_macro/deepm/utils/__pycache__
projects/deepm_macro/scripts/__pycache__
projects/deepm_macro/tests/__pycache__
projects/deregime/__pycache__
projects/deregime/deregime/__pycache__
projects/deregime/scripts/__pycache__
projects/equity_intraday_deepm/equity_intraday_deepm/__pycache__
projects/stock_top10_strategy/stock_top10_strategy/__pycache__

### DeepM smoke model directories
projects/deepm_macro/models_torch/smoke-deepm-gat
projects/deepm_macro/models_torch/smoke-deepm-gat-cascading
projects/deepm_macro/models_torch/smoke-deepm-gat-flip
projects/deepm_macro/models_torch/smoke-deepm-gat-full-cost
projects/deepm_macro/models_torch/smoke-deepm-gat-macd
projects/deepm_macro/models_torch/smoke-deepm-gat-no-rezero
projects/deepm_macro/models_torch/smoke-deepm-gat-pooled-only
projects/deepm_macro/models_torch/smoke-deepm-gat-softmin-t005
projects/deepm_macro/models_torch/smoke-deepm-gat-softmin-t1
projects/deepm_macro/models_torch/smoke-deepm-gat-zero-cost
projects/deepm_macro/models_torch/smoke-deepm-gcn
projects/deepm_macro/models_torch/smoke-deepm-graph-only
projects/deepm_macro/models_torch/smoke-deepm-independent
projects/deepm_macro/models_torch/smoke-deepm-no-graph
projects/deepm_macro/models_torch/smoke-temporal-baseline
projects/deepm_macro/models_torch/smoke-temporal-baseline-zero-cost

### DeepM smoke backtest directories
projects/deepm_macro/backtest_diagnostics/bt-smoke-deepm-gat
projects/deepm_macro/backtest_diagnostics/bt-smoke-deepm-gat-cascading
projects/deepm_macro/backtest_diagnostics/bt-smoke-deepm-gat-flip
projects/deepm_macro/backtest_diagnostics/bt-smoke-deepm-gat-full-cost
projects/deepm_macro/backtest_diagnostics/bt-smoke-deepm-gat-macd
projects/deepm_macro/backtest_diagnostics/bt-smoke-deepm-gat-no-rezero
projects/deepm_macro/backtest_diagnostics/bt-smoke-deepm-gat-pooled-only
projects/deepm_macro/backtest_diagnostics/bt-smoke-deepm-gat-softmin-t005
projects/deepm_macro/backtest_diagnostics/bt-smoke-deepm-gat-softmin-t1
projects/deepm_macro/backtest_diagnostics/bt-smoke-deepm-gat-zero-cost
projects/deepm_macro/backtest_diagnostics/bt-smoke-deepm-gcn
projects/deepm_macro/backtest_diagnostics/bt-smoke-deepm-graph-only
projects/deepm_macro/backtest_diagnostics/bt-smoke-deepm-independent
projects/deepm_macro/backtest_diagnostics/bt-smoke-deepm-no-graph
projects/deepm_macro/backtest_diagnostics/bt-smoke-temporal-baseline
projects/deepm_macro/backtest_diagnostics/bt-smoke-temporal-baseline-zero-cost
projects/deepm_macro/backtest_results/bt-smoke-deepm-gat
projects/deepm_macro/backtest_results/bt-smoke-deepm-gat-cascading
projects/deepm_macro/backtest_results/bt-smoke-deepm-gat-flip
projects/deepm_macro/backtest_results/bt-smoke-deepm-gat-full-cost
projects/deepm_macro/backtest_results/bt-smoke-deepm-gat-macd
projects/deepm_macro/backtest_results/bt-smoke-deepm-gat-no-rezero
projects/deepm_macro/backtest_results/bt-smoke-deepm-gat-pooled-only
projects/deepm_macro/backtest_results/bt-smoke-deepm-gat-softmin-t005
projects/deepm_macro/backtest_results/bt-smoke-deepm-gat-softmin-t1
projects/deepm_macro/backtest_results/bt-smoke-deepm-gat-zero-cost
projects/deepm_macro/backtest_results/bt-smoke-deepm-gcn
projects/deepm_macro/backtest_results/bt-smoke-deepm-graph-only
projects/deepm_macro/backtest_results/bt-smoke-deepm-independent
projects/deepm_macro/backtest_results/bt-smoke-deepm-no-graph
projects/deepm_macro/backtest_results/bt-smoke-temporal-baseline
projects/deepm_macro/backtest_results/bt-smoke-temporal-baseline-zero-cost

### Stock strategy stale sample/check runs
projects/stock_top10_strategy/stock_top10_strategy/runs/sample_check
projects/stock_top10_strategy/stock_top10_strategy/runs/sample_check_2
projects/stock_top10_strategy/stock_top10_strategy/runs/top10_buy_the_move_sample_20260708_121135
projects/stock_top10_strategy/stock_top10_strategy/runs/wikipedia_yahoo_2000_current_check

### Other generated local state
wandb/
projects/deepm_macro/deepm.egg-info/
./.DS_Store
./projects/deepm_macro/live_runs/.DS_Store
./projects/stock_top10_strategy/stock_top10_strategy/runs/.DS_Store

## Retained active artifacts
- projects/deepm_macro/models_torch/deepm-gat*/
- projects/deepm_macro/models_torch/deepm-mt-vsn/
- projects/deepm_macro/backtest_results/ and backtest_diagnostics/ non-smoke runs
- projects/deepm_macro/live_runs/ and live_state/
- projects/deregime/deregime_runs/
- projects/stock_top10_strategy/stock_top10_strategy/runs/wikipedia_yahoo_2000_current/

## Post-verification cleanup
- Removed __pycache__/ directories recreated by compile/tests.
- Removed project *.egg-info/ directories recreated by editable install checks.

## Removed nested repository metadata
- Removed projects/deregime/.git so the parent repository can track DeRegiME as normal source.
- Previous nested DeRegiME remote: origin	https://github.com/kieranjwood/deregime.git (fetch);origin	https://github.com/kieranjwood/deregime.git (push);
- Previous nested DeRegiME status:
 M .gitignore
 M deregime/config.py
 M deregime/data.py
 M deregime/models.py
 M deregime/run.py
 M deregime/training.py
 M pyproject.toml
 M requirements-tested.txt
?? __init__.py
?? data/
?? run.py
?? scripts/benchmark_deregime_runtime.py
?? scripts/build_deregime_rolling_experiments.py
?? scripts/check_deregime_device.py
?? tests/

## Removed empty local W&B artifact tree
- Removed projects/deepm_macro/hkvelakaturi-e/ because it contained only empty local run directories.

## Removed obsolete compatibility wrapper
- Removed projects/deregime/__init__.py; standalone imports now use the inner deregime package directly.
