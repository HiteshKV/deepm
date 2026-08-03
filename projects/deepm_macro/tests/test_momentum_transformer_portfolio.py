import unittest

import torch

from deepm.models.momentum_transformer import PortfolioMomentumTransformer


class PortfolioMomentumTransformerTests(unittest.TestCase):
    def test_forward_and_importance_preserve_cross_section_contract(self):
        model = PortfolioMomentumTransformer(
            input_dim=3,
            num_tickers=2,
            hidden_dim=8,
            dropout=0.0,
            num_heads=2,
            seq_len=5,
            pre_loss_steps=2,
            use_transaction_costs=False,
        )

        target_x = torch.randn(4, 2, 5, 3)
        target_tickers = torch.tensor([[0, 1]] * 4)
        mask_single_date = torch.zeros(4, 2, 5, dtype=torch.bool)

        output = model.forward_candidate_arch(
            target_x,
            target_tickers,
            mask_single_date=mask_single_date,
            batch_size=4,
        )
        self.assertEqual(tuple(output.shape), (4, 2, 5, 8))

        importance = model.variable_importance(
            target_x,
            target_tickers,
            mask_single_date=mask_single_date,
            batch_size=4,
        )
        self.assertEqual(tuple(importance.shape), (4, 2, 5, 4))

    def test_temporal_attention_can_be_recorded_on_demand(self):
        model = PortfolioMomentumTransformer(
            input_dim=2,
            num_tickers=3,
            hidden_dim=12,
            dropout=0.0,
            num_heads=3,
            seq_len=6,
            pre_loss_steps=2,
            use_transaction_costs=False,
        )

        target_x = torch.randn(2, 3, 6, 2)
        target_tickers = torch.tensor([[0, 1, 2]] * 2)
        mask_single_date = torch.zeros(2, 3, 6, dtype=torch.bool)

        _ = model.forward_candidate_arch(
            target_x,
            target_tickers,
            mask_single_date=mask_single_date,
            batch_size=2,
            return_temporal_attention=True,
        )

        attention = model.last_temporal_attention()
        self.assertIsNotNone(attention)
        self.assertEqual(tuple(attention.shape), (6, 3, 6, 6))


if __name__ == "__main__":
    unittest.main()
