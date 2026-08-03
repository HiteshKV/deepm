import unittest

import numpy as np
import pandas as pd

from deregime.config import ConfigDict
from deregime.data import make_splits_and_loaders


class DeRegimeRollingProtocolTests(unittest.TestCase):
    def test_date_cutoff_split_trains_before_test_start(self):
        dates = pd.date_range("2000-01-03", periods=24, freq="B")
        wide = pd.DataFrame(
            {
                "A": np.linspace(0.0, 1.0, len(dates)),
                "B": np.linspace(1.0, 2.0, len(dates)),
            },
            index=dates,
        )
        wide.index.name = "date"
        cutoff = dates[12]
        end = dates[20]
        cfg = ConfigDict(
            {
                "date_col": "date",
                "freq": "b",
                "seq_len": 5,
                "pred_len": 3,
                "gp_label_len": 2,
                "valid_ratio": 0.25,
                "test_ratio": 0.2,
                "batch_size": 4,
                "eval_batch_size": 4,
                "gradient_accumulation_steps": 1,
                "num_workers": 0,
                "train_drop_last": False,
                "enable_input_aggregation": False,
                "enable_target_aggregation": False,
                "input_aggregation": None,
                "target_aggregation": None,
                "input_transform": "none",
                "target_transform": "none",
                "test_start_date": cutoff.strftime("%Y-%m-%d"),
                "test_end_date": end.strftime("%Y-%m-%d"),
            }
        )

        (
            z_train,
            z_valid,
            z_test,
            _y_train,
            _y_valid,
            _y_test,
            *_rest,
            test_dataset,
            _test_loader,
            _tmark_dim,
            _din,
            _dout,
        ) = make_splits_and_loaders(wide, cfg)

        self.assertLess(z_train.index.max(), cutoff)
        self.assertLess(z_valid.index.max(), cutoff)
        self.assertEqual(z_test.index[0], dates[7])

        first = test_dataset[0]
        seq_y_time_idx = first[4]
        first_h1_time = pd.to_datetime(seq_y_time_idx[-1].item(), unit="us")
        self.assertEqual(first_h1_time, cutoff)


if __name__ == "__main__":
    unittest.main()
