import torch
from torch import nn

from deepm.models.common import (
    causal_attention_mask,
    GateAddNorm,
    GatedResidualNetwork,
)
from deepm.models.base import (
    DeepMomentumNetwork,
    SequenceRepresentation,
)


class MomentumTransformer(DeepMomentumNetwork):
    """Momentum Transformer with self-attention and GRN post-processing."""

    def __init__(
        self,
        input_dim: int,
        num_tickers: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear=0,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
            **kwargs,
        )

        assert isinstance(num_heads, int)
        self.num_heads = num_heads

        self.seq_rep = SequenceRepresentation(
            self.input_dim,
            hidden_dim,
            dropout,
            num_tickers,
            fuse_encoder_input=False,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
        )

        self.gate_add_norm_mha = GateAddNorm(hidden_dim, dropout=dropout)
        self.ffn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout=dropout
        )
        self.self_att = nn.MultiheadAttention(
            hidden_dim, self.num_heads, batch_first=True,
        )
        self.gate_add_norm_block = GateAddNorm(hidden_dim, dropout=dropout)

    def forward_candidate_arch(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        representation, lstm_hidden_state = self.seq_rep(
            target_x, target_tickers, pos_encoding_batch=pos_encoding_batch
        )

        mask = causal_attention_mask(self.seq_len).to(representation.device)
        mha, _ = self.self_att(
            representation, representation, representation, attn_mask=mask
        )
        add = self.gate_add_norm_mha(mha, representation)
        ffn_representation = self.ffn(add)
        transformer_representation = self.gate_add_norm_block(
            ffn_representation, lstm_hidden_state
        )
        return transformer_representation

    def variable_importance(self, target_x, target_tickers, **kwargs):
        importance = self.seq_rep(target_x, target_tickers, variable_importance=True)
        return importance.squeeze(-2).swapaxes(0, 1)


class PortfolioMomentumTransformer(DeepMomentumNetwork):
    """Portfolio-level Momentum Transformer variant for DeePM ablations.

    This keeps the original DeePM training/backtest objective and panel layout,
    but removes cross-asset graph/attention mixing. Each asset is encoded with a
    Momentum-Transformer-style VSN, LSTM, causal self-attention, and GRN block;
    the shared base class then turns hidden states into positions and aggregates
    portfolio returns exactly like the DeePM cross-sectional models.
    """

    def __init__(
        self,
        input_dim: int,
        num_tickers: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear: int = 0,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
            **kwargs,
        )

        self.num_tickers = num_tickers
        self.num_heads = int(num_heads)
        self.record_temporal_attention = bool(
            kwargs.get("record_temporal_attention", False)
        )
        self._last_temporal_attention = None

        seq_input_dim = self.input_dim + getattr(self, "extra_tcost_channels", 0) + 1
        self.seq_rep = SequenceRepresentation(
            seq_input_dim,
            hidden_dim,
            dropout,
            num_tickers,
            fuse_encoder_input=False,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
        )

        self.gate_add_norm_mha = GateAddNorm(hidden_dim, dropout=dropout)
        self.ffn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout=dropout
        )
        self.self_att = nn.MultiheadAttention(
            hidden_dim,
            self.num_heads,
            batch_first=True,
        )
        self.gate_add_norm_block = GateAddNorm(hidden_dim, dropout=dropout)

        self.is_cross_section = True

    def _cross_section_inputs(
        self,
        target_x,
        target_tickers,
        mask_single_date=None,
        batch_size=None,
    ):
        if target_x.dim() != 4:
            raise ValueError(
                "PortfolioMomentumTransformer expects cross-sectional input "
                f"[B, N, S, F], got {tuple(target_x.shape)}"
            )
        batch_size = batch_size or target_x.shape[0]
        if mask_single_date is None:
            mask_single_date = torch.zeros(
                target_x.shape[:3], dtype=torch.bool, device=target_x.device
            )
        x_in = torch.cat([target_x, mask_single_date.float().unsqueeze(-1)], dim=-1)
        x_in = x_in.view(batch_size * self.num_tickers, self.seq_len, -1)
        tickers_flat = target_tickers.view(-1)
        return x_in, tickers_flat, batch_size

    def _maybe_add_transaction_cost_inputs(
        self,
        target_x,
        vol_scaling_amount=None,
        vol_scaling_amount_prev=None,
        trans_cost_bp=None,
    ):
        """Mirror base.forward preprocessing for interpretability calls."""
        if (
            self.use_transaction_costs
            and self.tcost_inputs
            and target_x.shape[-1] == self.input_dim
        ):
            if (
                vol_scaling_amount is None
                or vol_scaling_amount_prev is None
                or trans_cost_bp is None
            ):
                raise ValueError(
                    "Transaction-cost inputs are enabled; pass vol_scaling_amount, "
                    "vol_scaling_amount_prev, and trans_cost_bp."
                )
            return self.concat_transaction_cost_inputs(
                target_x,
                vol_scaling_amount,
                vol_scaling_amount_prev,
                trans_cost_bp,
            )
        return target_x

    def forward_candidate_arch(
        self,
        target_x,
        target_tickers,
        mask_single_date=None,
        batch_size=None,
        pos_encoding_batch=None,
        **kwargs,
    ):
        x_in, tickers_flat, batch_size = self._cross_section_inputs(
            target_x,
            target_tickers,
            mask_single_date=mask_single_date,
            batch_size=batch_size,
        )

        representation, lstm_hidden_state = self.seq_rep(
            x_in,
            tickers_flat,
            pos_encoding_batch=pos_encoding_batch,
        )

        mask = causal_attention_mask(self.seq_len).to(representation.device)
        need_attention = bool(
            kwargs.get("return_temporal_attention", False)
            or self.record_temporal_attention
        )
        mha, attn_weights = self.self_att(
            representation,
            representation,
            representation,
            attn_mask=mask,
            need_weights=need_attention,
            average_attn_weights=False,
        )
        self._last_temporal_attention = (
            attn_weights.detach().cpu() if need_attention else None
        )

        add = self.gate_add_norm_mha(mha, representation)
        ffn_representation = self.ffn(add)
        transformer_representation = self.gate_add_norm_block(
            ffn_representation, lstm_hidden_state
        )

        return transformer_representation.view(
            batch_size,
            self.num_tickers,
            self.seq_len,
            self.hidden_dim,
        )

    def variable_importance(
        self,
        target_x,
        target_tickers,
        mask_single_date=None,
        batch_size=None,
        **kwargs,
    ):
        target_x = self._maybe_add_transaction_cost_inputs(
            target_x,
            kwargs.get("vol_scaling_amount"),
            kwargs.get("vol_scaling_amount_prev"),
            kwargs.get("trans_cost_bp"),
        )
        x_in, tickers_flat, batch_size = self._cross_section_inputs(
            target_x,
            target_tickers,
            mask_single_date=mask_single_date,
            batch_size=batch_size,
        )
        importance = self.seq_rep(x_in, tickers_flat, variable_importance=True)
        importance = importance.squeeze(-2).swapaxes(0, 1)
        return importance.view(batch_size, self.num_tickers, self.seq_len, -1)

    def last_temporal_attention(self):
        """Return the most recent attention tensor captured for diagnostics."""
        return self._last_temporal_attention
