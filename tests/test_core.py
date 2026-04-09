"""Core tests for the v2 dual-branch architecture."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch


class TestPriceBranch:
    def test_forward(self):
        from src.encoder.price_branch import PriceBranch

        model = PriceBranch(input_dim=6, d_model=64, inception_channels=32, tcn_layers=2)
        x = torch.randn(4, 120, 6)
        pooled, seq_features = model(x)

        assert pooled.shape == (4, 64)
        assert seq_features.shape == (4, 120, 64)
        assert not torch.isnan(pooled).any()

    def test_receptive_field(self):
        from src.encoder.price_branch import PriceBranch

        model = PriceBranch(tcn_layers=4)
        assert model.receptive_field() > 1


class TestSentimentBranch:
    def test_forward(self):
        from src.encoder.sentiment_branch import SentimentBranch

        model = SentimentBranch(input_dim=768, d_model=64)
        x = torch.randn(4, 768)
        out = model(x)
        assert out.shape == (4, 64)

    def test_null_embedding(self):
        from src.encoder.sentiment_branch import SentimentBranch

        model = SentimentBranch(d_model=64)
        x = torch.zeros(2, 768)  # no news
        out = model(x)
        assert out.shape == (2, 64)
        assert not torch.isnan(out).any()
        # Null embeddings should be non-zero (learned)
        assert out.abs().sum() > 0


class TestDualBranchModel:
    def test_forward_no_sentiment(self):
        from src.encoder.fusion import DualBranchModel

        model = DualBranchModel(d_model=64, inception_channels=32, tcn_layers=2)
        x = torch.randn(4, 120, 6)
        logits, confidence = model(x)

        assert logits.shape == (4, 3)
        assert confidence.shape == (4, 1)
        assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_forward_with_sentiment(self):
        from src.encoder.fusion import DualBranchModel

        model = DualBranchModel(d_model=64, inception_channels=32, tcn_layers=2)
        x = torch.randn(2, 120, 6)
        s = torch.randn(2, 768)
        logits, confidence = model(x, s)

        assert logits.shape == (2, 3)
        assert confidence.shape == (2, 1)

    def test_predict(self):
        from src.encoder.fusion import DualBranchModel

        model = DualBranchModel(d_model=64, inception_channels=32, tcn_layers=2)
        x = torch.randn(4, 120, 6)
        result = model.predict(x)

        assert "action" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["action"].shape == (4,)
        assert result["probabilities"].shape == (4, 3)


class TestLabeling:
    def test_triple_barrier(self):
        from src.training.labels import LabelConfig, TripleBarrierLabeler

        cfg = LabelConfig(profit_target_pips=300, stop_loss_pips=150, max_holding_bars=120, pip_value=0.10)
        labeler = TripleBarrierLabeler(cfg)
        prices = np.cumsum(np.random.randn(500) * 0.5) + 2000
        labels = labeler.label(prices)

        assert len(labels) == 500
        assert set(np.unique(labels)).issubset({0, 1, 2})

    def test_macd_rsi(self):
        from src.training.labels import LabelConfig, MACDRSILabeler

        cfg = LabelConfig()
        labeler = MACDRSILabeler(cfg)
        prices = np.cumsum(np.random.randn(500) * 0.5) + 2000
        labels = labeler.label(prices)

        assert len(labels) == 500
        # MACD/RSI should produce more balanced labels
        counts = np.bincount(labels, minlength=3)
        assert counts[1] > 0  # should have holds

    def test_hybrid(self):
        from src.training.labels import LabelConfig, HybridLabeler

        cfg = LabelConfig(profit_target_pips=300, stop_loss_pips=150, max_holding_bars=120, pip_value=0.10)
        labeler = HybridLabeler(cfg)
        prices = np.cumsum(np.random.randn(1000) * 0.5) + 2000
        labels = labeler.label(prices)

        assert len(labels) == 1000

    def test_create_sequences(self):
        from src.training.labels import create_sequences

        features = np.random.randn(200, 6).astype(np.float32)
        labels = np.random.randint(0, 3, 200)
        sentiment = np.random.randn(200, 768).astype(np.float32)

        X, y, S = create_sequences(features, labels, 50, sentiment)
        assert X.shape == (150, 50, 6)
        assert y.shape == (150,)
        assert S.shape == (150, 768)


class TestBacktesting:
    def test_with_confidence(self):
        from src.backtesting.engine import BacktestEngine, BacktestConfig

        prices = np.linspace(2000, 2010, 100)
        signals = np.random.choice([0, 1, 2], 100, p=[0.2, 0.6, 0.2])
        confs = np.random.uniform(0.3, 0.9, 100)

        engine = BacktestEngine(BacktestConfig())
        result = engine.run(prices, signals, confs)

        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0

    def test_hitl_disabled(self):
        from src.backtesting.engine import BacktestEngine, BacktestConfig

        prices = np.ones(50) * 2000
        signals = np.ones(50, dtype=int)  # all hold
        engine = BacktestEngine(BacktestConfig(human_exit_approval=False))
        result = engine.run(prices, signals)
        assert result.total_trades == 0


class TestHITL:
    def test_auto_approve_high_confidence(self):
        from src.hitl.mt5_interface import HITLGate, SignalContext

        gate = HITLGate(enabled=True, auto_approve_confidence=0.7)
        ctx = SignalContext(
            action="BUY", confidence=0.9, current_price=2000,
            position_size_lots=0.01,
        )
        assert gate.check_entry(ctx) is True
        assert gate.stats["auto_approved"] == 1

    def test_auto_approve_profitable_exit(self):
        from src.hitl.mt5_interface import HITLGate, SignalContext

        gate = HITLGate(enabled=True)
        ctx = SignalContext(
            action="CLOSE", confidence=0.5, current_price=2010,
            entry_price=2000, unrealized_pnl=10.0,
        )
        assert gate.check_exit(ctx) is True


class TestGANMarket:
    def test_generate(self):
        from src.meta_policy.gan_market import GANMarketSimulator

        gan = GANMarketSimulator(seq_len=60, feature_dim=6, device="cpu")
        samples = gan.generate(8)
        assert samples.shape == (8, 60, 6)


class TestRLAgent:
    def test_select_action(self):
        from src.meta_policy.rl_agent import ConfidenceSACAgent

        agent = ConfidenceSACAgent(obs_dim=9)
        obs = np.random.randn(9).astype(np.float32)
        action = agent.select_action(obs, confidence=0.8)
        assert -1 <= action <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
