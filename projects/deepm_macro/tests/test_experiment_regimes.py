import unittest

from deepm.configs.load import load_backtest_settings, load_settings_for_architecture
from deepm.experiments.regimes import (
    REGIMES,
    combined_aggregate_command,
    commands_for_regime,
)
from deepm.training.train import resolve_training_device


class ExperimentRegimeTests(unittest.TestCase):
    def test_regimes_are_distinct_and_commandable(self):
        self.assertEqual(set(REGIMES), {"original", "deregime", "momentum"})

        original = REGIMES["original"]
        deregime = REGIMES["deregime"]
        momentum = REGIMES["momentum"]

        self.assertEqual(original.experiments[0].train_yaml, "deepm-gat")
        self.assertTrue(
            all(exp.train_yaml.startswith("deepm-gat-dg-") for exp in deregime.experiments)
        )
        self.assertEqual(momentum.experiments[0].architecture, "MT_DEEPM")

        commands = commands_for_regime("momentum", ["train", "backtest"])
        self.assertIn("-r deepm-mt-vsn -a MT_DEEPM --device mps", commands[0])
        self.assertIn("--name bt-deepm-mt-vsn-k10-current", commands[1])

        filtered = commands_for_regime(
            "momentum", ["train"], filter_start_years=[2020]
        )
        self.assertIn("-fsy 2020", filtered[0])

    def test_combined_aggregate_includes_each_lane(self):
        command = combined_aggregate_command()
        self.assertIn("bt-deepm-gat-k10-current", command)
        self.assertIn("bt-deepm-gat-dg-full", command)
        self.assertIn("bt-deepm-mt-vsn-k10-current", command)

    def test_momentum_config_loads_as_cross_sectional_lane(self):
        settings = load_settings_for_architecture("deepm-mt-vsn", "MT_DEEPM")
        self.assertEqual(settings["run_name"], "MT_DEEPM")
        self.assertTrue(settings["cross_section"])
        self.assertEqual(settings["data_parquet"], "data_20260625.parquet")
        self.assertEqual(settings["device"], "mps")

        backtest = load_backtest_settings("bt-deepm-mt-vsn-k10-current")
        self.assertEqual(backtest["model"]["train_yaml"], "deepm-mt-vsn")
        self.assertEqual(backtest["model"]["architecture"], "MT_DEEPM")

    def test_device_resolver_accepts_explicit_cpu(self):
        self.assertEqual(str(resolve_training_device("cpu")), "cpu")


if __name__ == "__main__":
    unittest.main()
