import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from .config import CONFIGS, ConfigDict
from .time_features import get_time_mark


def split_time(data: pd.DataFrame, index: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return data.iloc[:index, :], data.iloc[index:, :]


def train_val_split(train_data: pd.DataFrame, ratio: float, seq_len: int):
    if ratio == 1:
        return train_data, None
    border = int(train_data.shape[0] * ratio)
    if seq_len is not None:
        train_value, _ = split_time(train_data, border)
        train_rest, valid_data = split_time(train_data, border - seq_len)
        return train_value, valid_data
    else:
        train_value, valid_rest = split_time(train_data, border)
        return train_value, valid_rest


class DatasetForTransformer:
    """
    Alignment:
      - seq_x: series values at times [s .. e-1]
      - seq_xm: marks at times [s+1 .. e] (seq_x is the previous values)
      - Label window [r0 .. r1-1] ends at e; r1=e+1; size = gp_label_len
      - seq_ym[-1] == seq_xm[-1] = timestamp of seq_y[-1]
      - Multi-horizon (H>1): length H-1; empty tensors if H==1
      - gp_label_len is retained for compatibility with decoder-style paths;
        sequence_flatten models use direct future forecast locations.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        target: pd.DataFrame,
        history_len: int,
        prediction_len: int,
        gp_label_len: int,
        timeenc: int,
        freq: str,
    ):
        self.dataset = dataset
        self.target = target
        self.history_length = history_len
        self.prediction_length = prediction_len
        self.gp_label_length = gp_label_len
        self.timeenc = timeenc
        self.freq = freq
        self.tensorize = True
        self._read()

    def __len__(self):
        return len(self.dataset) - self.history_length - self.prediction_length + 1

    @property
    def _total_seq_len(self):
        return self.history_length + self.prediction_length

    def _read(self):
        df_stamp = self.dataset.reset_index()
        df_stamp = df_stamp[[CONFIGS["date_col"]]].values.T
        data_stamp = get_time_mark(df_stamp, self.timeenc, self.freq)[0]
        self.data_values = torch.as_tensor(
            self.dataset.to_numpy(dtype=np.float32, copy=True), dtype=torch.float32
        )
        self.target_values = torch.as_tensor(
            self.target.to_numpy(dtype=np.float32, copy=True), dtype=torch.float32
        )
        self.data_stamp = torch.as_tensor(data_stamp, dtype=torch.float32)
        self.time_index = torch.as_tensor(self.dataset.index.asi8, dtype=torch.int64)

    def __getitem__(self, idx):
        s, e = idx, idx + self.history_length
        dec_end = e
        r0 = dec_end - self.gp_label_length + 1
        r1 = dec_end + 1

        seq_x = self.data_values[s:e]
        seq_xm = self.data_stamp[s + 1 : e + 1]
        t0 = seq_xm[0]
        if self.timeenc == 2:
            seq_xm = (seq_xm - t0) / self._total_seq_len

        seq_y = self.target_values[r0:r1]
        seq_ym = self.data_stamp[r0:r1]
        if self.timeenc == 2:
            seq_ym = (seq_ym - t0) / self._total_seq_len

        seq_y_time_idx = self.time_index[r0:r1]

        if self.prediction_length > 1:
            mh_start = r1
            mh_end = r1 + (self.prediction_length - 1)
            seq_multihorizon_y = self.target_values[mh_start:mh_end]
            seq_multihorizon_m = self.data_stamp[mh_start:mh_end]
            if self.timeenc == 2:
                seq_multihorizon_m = (seq_multihorizon_m - t0) / self._total_seq_len
            seq_mh_time_idx = self.time_index[mh_start:mh_end]
        else:
            seq_multihorizon_y = torch.empty(0, seq_y.shape[-1], dtype=torch.float32)
            seq_multihorizon_m = torch.empty(
                0, self.data_stamp.shape[1], dtype=torch.float32
            )
            seq_mh_time_idx = torch.empty(0, dtype=torch.int64)
            # if self.timeenc == 2:
            #     seq_multihorizon_m = (seq_multihorizon_m - t0)/self._total_seq_len

        return (
            seq_x,
            seq_y,
            seq_xm,
            seq_ym,
            seq_y_time_idx,
            seq_multihorizon_y,
            seq_multihorizon_m,
            seq_mh_time_idx,
        )


def _dataloader_kwargs(config):
    num_workers = int(config.get("num_workers", 0) or 0)
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(config.get("pin_memory", False)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", False))
        prefetch_factor = config.get("prefetch_factor", None)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def forecasting_data_provider(
    data, target, config, timeenc, batch_size, shuffle, drop_last
):
    ds = DatasetForTransformer(
        data,
        target,
        config.seq_len,
        config.pred_len,
        config.gp_label_len,
        timeenc,
        config.freq,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **_dataloader_kwargs(config),
    )
    return ds, dl


# ========================== HELPERS: Loading, aggregation, metrics ===============================
def _validate_io_dims(x_train_df: pd.DataFrame, y_train_df: pd.DataFrame):
    """
    Enforce currently supported mappings:
      ✓ 1D → 1D
      ✓ MD → MD (same D)
      ✗ 1D → MD  (NotImplemented)
      ✗ MD → 1D  (NotImplemented)
    """
    Din = x_train_df.shape[1]
    Dout = y_train_df.shape[1]
    if Din == Dout:
        return
    if Din == 1 and Dout > 1:
        raise NotImplementedError("Path 1D inputs → MD targets not implemented.")
    if Din > 1 and Dout == 1:
        raise NotImplementedError("Path MD inputs → 1D targets not implemented.")
    # If it's "MD→MD but different D", also not allowed
    raise NotImplementedError(
        f"Path MD→MD with different dims ({Din} → {Dout}) not implemented."
    )


def load_long_to_wide(
    file: str, date_col: str, series_id_col: str, value_col: str
) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=[date_col])
    wide = df.pivot(
        index=date_col, columns=series_id_col, values=value_col
    ).sort_index()
    return wide.ffill().bfill()


def choose_target(df: pd.DataFrame, enable_target_aggregation: bool, how) -> pd.Series:
    """
    Returns a Series (always).
    how can be:
      - "sum": sum across all columns
      - "mean": mean across all columns
      - str column name
      - list/tuple/array of column names -> sum across those columns
      - dict like {"cols":[...], "agg":"sum"|"mean"} for explicit control
    """
    # dict form for explicit selection
    if not enable_target_aggregation or how is None:
        return df.copy()
    if isinstance(how, dict):
        cols = how.get("cols", None)
        agg = how.get("agg", "sum").lower()
        if cols is None:
            raise ValueError("choose_target: dict form must include 'cols'.")
        sub = df[cols]
        if agg == "sum":
            return sub.sum(axis=1)
        elif agg == "mean":
            return sub.mean(axis=1)
        else:
            raise ValueError(f"choose_target: unknown agg '{agg}'.")

    # list-like column selection
    if isinstance(how, (list, tuple, np.ndarray, pd.Index)):
        sub = df[list(how)]
        # default to sum across the chosen columns
        return sub.sum(axis=1)

    # string cases
    if isinstance(how, str):
        h = how.strip().lower()
        if h == "sum":
            return df.sum(axis=1)
        if h == "mean":
            return df.mean(axis=1)
        # treat as column name (preserve original case for lookup)
        if how in df.columns:
            s = df[how]
            # ensure Series (if someone passed a duplicate column name producing DF)
            return s.squeeze() if isinstance(s, pd.DataFrame) else s
        raise ValueError(
            f"choose_target: target_aggregation '{how}' not found in columns"
        )

    # fallback: single column already provided as Series?
    if isinstance(how, pd.Series):
        return how

    raise TypeError(f"choose_target: unsupported type for 'how': {type(how)}")


def choose_inputs(
    df: pd.DataFrame, enable_input_aggregation: bool, aggregation: str | None
) -> pd.DataFrame:
    if not enable_input_aggregation:
        return df.copy()  # <-- MV inputs (D cols)
    if aggregation is None:
        raise ValueError("enable_input_aggregation=True requires an aggregation")
    if aggregation == "sum":
        return df.sum(axis=1).to_frame("sum")
    if aggregation == "mean":
        return df.mean(axis=1).to_frame("mean")
    if aggregation in df.columns:
        return df[[aggregation]].copy()
    raise ValueError(f"input_aggregation='{aggregation}' invalid")


def _apply_transform(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Applies a returns transformation to a DataFrame."""
    if mode == "none":
        return df

    print(f"Applying '{mode}' transformation...")
    if mode == "log_returns":
        # .clip() avoids log(0) if prices are 0
        # .fillna(0.0) handles the first NaN value at the start of the split
        return np.log(df.clip(lower=1e-9)).diff(periods=1).fillna(0.0)
    elif mode == "returns":
        # .fillna(0.0) handles the first NaN value at the start of the split
        return df.pct_change(periods=1).fillna(0.0)
    else:
        raise ValueError(f"Unknown transform mode: {mode}")


def make_splits_and_loaders(wide: pd.DataFrame, config: ConfigDict):
    # --- 1. Global Feature Selection & Transformation ---
    # We act on the WHOLE dataframe to preserve history for differencing

    # Choose TARGETS (from raw global data)
    raw_targets = choose_target(
        wide, config.enable_target_aggregation, config.target_aggregation
    )
    # Ensure DataFrame
    if not isinstance(raw_targets, pd.DataFrame):
        raw_targets = raw_targets.to_frame("target")

    # Choose INPUTS (from raw global data)
    raw_inputs = choose_inputs(
        wide, config.enable_input_aggregation, config.input_aggregation
    )

    # Apply TRANSFORMATIONS (Global) - This fixes the "Ghost Zero" bug
    trgt_transform_mode = config.get("target_transform", "none")
    inpt_transform_mode = config.get("input_transform", "none")

    # The diffs will now be correct across split boundaries
    all_targets_trans = _apply_transform(raw_targets, trgt_transform_mode)
    all_inputs_trans = _apply_transform(raw_inputs, inpt_transform_mode)

    # --- 2. Splitting ---
    # Default DeRegiME benchmarks use ratio splits. DeePM sidecar experiments
    # can opt into explicit date cutoffs so each rolling test window is forecast
    # by a DeRegiME model trained only on dates before the test window.
    if config.get("test_start_date"):
        date_index = pd.to_datetime(all_inputs_trans.index)
        test_start = pd.Timestamp(config.test_start_date)
        test_end = (
            pd.Timestamp(config.test_end_date)
            if config.get("test_end_date")
            else date_index.max() + pd.Timedelta(days=1)
        )
        if test_end <= test_start:
            raise ValueError("test_end_date must be after test_start_date")

        test_start_pos = int(date_index.searchsorted(test_start, side="left"))
        test_end_pos = int(date_index.searchsorted(test_end, side="left"))
        context = int(config.seq_len or 0)
        test_context_start_pos = max(0, test_start_pos - context)

        if test_start_pos <= context:
            raise ValueError(
                "Not enough pre-test history for the requested seq_len context"
            )
        if test_end_pos <= test_start_pos:
            raise ValueError("No test rows found inside the requested date window")
        if test_end_pos - test_context_start_pos < context + int(config.pred_len):
            raise ValueError("Date split does not contain enough rows for pred_len")

        x_train_valid = all_inputs_trans.iloc[:test_start_pos]
        y_train_valid = all_targets_trans.iloc[:test_start_pos]
        val_border = int(x_train_valid.shape[0] * (1 - config.valid_ratio))
        valid_start = max(0, val_border - context)

        x_train_df = x_train_valid.iloc[:val_border]
        x_valid_df = x_train_valid.iloc[valid_start:]
        x_test_df = all_inputs_trans.iloc[test_context_start_pos:test_end_pos]

        trgt_train = y_train_valid.iloc[:val_border]
        trgt_valid = y_train_valid.iloc[valid_start:]
        trgt_test = all_targets_trans.iloc[test_context_start_pos:test_end_pos]
    else:
        # --- 2a. Pre-calculate ratio split borders ---
        test_border = int(wide.shape[0] * (1 - config.test_ratio))

        # Calculate validation border based on the remaining train_valid set.
        valid_ratio_adj = 1 - (
            config.valid_ratio / max(1e-12, (1 - config.test_ratio))
        )
        val_border = int(test_border * valid_ratio_adj)

        # Now we slice the transformed data.
        x_train_valid = all_inputs_trans.iloc[:test_border]
        x_test_df = all_inputs_trans.iloc[test_border:]

        x_train_df = x_train_valid.iloc[:val_border]

        # Handle validation lookback (if seq_len is provided, we need overlap).
        if config.seq_len is not None:
            x_valid_df = x_train_valid.iloc[val_border - config.seq_len :]
        else:
            x_valid_df = x_train_valid.iloc[val_border:]

        # Split targets (logic mirrors inputs).
        y_train_valid = all_targets_trans.iloc[:test_border]
        trgt_test = all_targets_trans.iloc[test_border:]

        trgt_train = y_train_valid.iloc[:val_border]

        if config.seq_len is not None:
            trgt_valid = y_train_valid.iloc[val_border - config.seq_len :]
        else:
            trgt_valid = y_train_valid.iloc[val_border:]

    # --- 3. Scaling (Fit on Train only) ---
    # data_scaler is fit on TRANSFORMED inputs
    # targ_scaler is fit on TRANSFORMED targets
    data_scaler = StandardScaler().fit(x_train_df)
    targ_scaler = StandardScaler().fit(trgt_train)

    # --- 5. Apply SCALERS ---
    z_train = pd.DataFrame(
        data_scaler.transform(x_train_df),
        index=x_train_df.index,
        columns=x_train_df.columns,
    )
    z_valid = pd.DataFrame(
        data_scaler.transform(x_valid_df),
        index=x_valid_df.index,
        columns=x_valid_df.columns,
    )
    z_test = pd.DataFrame(
        data_scaler.transform(x_test_df),
        index=x_test_df.index,
        columns=x_test_df.columns,
    )

    y_train = pd.DataFrame(
        targ_scaler.transform(trgt_train),
        index=trgt_train.index,
        columns=trgt_train.columns,
    )
    y_valid = pd.DataFrame(
        targ_scaler.transform(trgt_valid),
        index=trgt_valid.index,
        columns=trgt_valid.columns,
    )
    y_test = pd.DataFrame(
        targ_scaler.transform(trgt_test),
        index=trgt_test.index,
        columns=trgt_test.columns,
    )

    # --- Batch Size and DataLoader Logic ---
    accumulation_steps = config.gradient_accumulation_steps
    total_batch_size = config.batch_size

    if accumulation_steps == 0:
        # Auto-detect mode: use full batch for now; run.py will probe and rebuild
        mini_batch_size = total_batch_size
        print("-" * 50)
        print(f"Batch Size: {total_batch_size} (gradient_accumulation_steps=0 -> auto-detect)")
        print("-" * 50)
    elif total_batch_size < accumulation_steps:
        warnings.warn(
            f"Total 'batch_size' ({total_batch_size}) is less than "
            f"'gradient_accumulation_steps' ({accumulation_steps}). "
            f"Setting mini_batch_size=1."
        )
        mini_batch_size = 1
    elif total_batch_size % accumulation_steps != 0:
        warnings.warn(
            f"Total 'batch_size' ({total_batch_size}) is not divisible by "
            f"'gradient_accumulation_steps' ({accumulation_steps}). "
            f"Effective batch size will be slightly smaller than requested."
        )
        mini_batch_size = total_batch_size // accumulation_steps
    else:
        mini_batch_size = total_batch_size // accumulation_steps

    if accumulation_steps != 0:
        print("-" * 50)
        print(f"Effective Batch Size Configured: {total_batch_size}")
        print(f"Gradient Accumulation Steps: {accumulation_steps}")
        print(f"  -> DataLoader Mini-Batch Size: {mini_batch_size}")
        print("-" * 50)

    train_dataset, train_loader = forecasting_data_provider(
        z_train,
        y_train,
        config,
        timeenc=2,
        batch_size=mini_batch_size,
        shuffle=True,
        drop_last=config.train_drop_last,
    )
    valid_dataset, valid_loader = forecasting_data_provider(
        z_valid,
        y_valid,
        config,
        timeenc=2,
        batch_size=mini_batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_dataset, test_loader = forecasting_data_provider(
        z_test,
        y_test,
        config,
        timeenc=2,
        batch_size=config.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

    tmark_dim = get_time_mark(
        z_train.reset_index()[[config.date_col]].values.T, 2, config.freq
    )[0].shape[1]

    print(f"INFO: Running with Din={z_train.shape[1]}, Dout={y_train.shape[1]}")
    return (
        z_train,
        z_valid,
        z_test,
        y_train,
        y_valid,
        y_test,
        data_scaler,
        targ_scaler,
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
        tmark_dim,
        z_train.shape[1],
        y_train.shape[1],
    )
