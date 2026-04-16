#!/usr/bin/env python3
"""Backtest with confidence filtering, equity export, and trade analysis.

Usage:
    python scripts/backtest.py model=dual_branch data=xauusd
    python scripts/backtest.py model=dual_branch data=xauusd ++min_confidence=0.5
    python scripts/backtest.py model=dual_branch data=xauusd ++min_confidence=0.7 ++export_csv=true
    python scripts/backtest.py model=dual_branch data=xauusd ++risk.human_exit_approval=true
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from loguru import logger

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.data.preprocessing import prepare_features
from src.data.tick_store import TickStore
from src.encoder.fusion import DualBranchModel
from src.training.labels import LabelConfig, get_labeler, create_sequences
from src.utils.config import get_device, set_seed
from src.utils.logger import setup_logger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logger()
    set_seed(cfg.project.seed)
    device = get_device(cfg.project.device)

    # pre-trade confidence gate removed in v2.3 — reward gate is the sole filter
    export_csv = cfg.get("export_csv", True)

    # Load data
    store = TickStore(cfg.paths.data_dir + "/ticks.duckdb")
    df = store.query_ohlcv(cfg.data.symbol, cfg.data.timeframe)
    store.close()

    if df.is_empty():
        logger.error("No data.")
        return

    features, close_prices = prepare_features(
        df, cfg.data.preprocessing.scaling, cfg.data.preprocessing.window_size,
    )

    # Label
    label_cfg = LabelConfig(
        method=cfg.data.labeling.method,
        profit_target_pips=cfg.data.labeling.profit_target_pips,
        stop_loss_pips=cfg.data.labeling.stop_loss_pips,
        max_holding_bars=cfg.data.labeling.get("max_holding_bars", 120),
        pip_value=cfg.data.labeling.pip_value,
    )
    labeler = get_labeler(label_cfg)
    labels = labeler.label(close_prices)

    ws = cfg.data.preprocessing.window_size
    features = features[ws:]
    labels = labels[ws:]

    # Sentiment
    sentiment = None
    if cfg.data.sentiment.enabled:
        emb_path = cfg.paths.data_dir + "/sentiment_embeddings.npy"
        if Path(emb_path).exists():
            all_emb = np.load(emb_path, allow_pickle=True)
            if isinstance(all_emb, np.ndarray) and all_emb.ndim == 2:
                sentiment = all_emb[ws:]

    seq_len = cfg.model.input.sequence_length
    X, y, S = create_sequences(features, labels, seq_len, sentiment)
    logger.info(f"Dataset: X={X.shape}, classes={np.bincount(y, minlength=3)}")

    # Test split
    n = len(X)
    test_start = int(n * (1 - cfg.training.test_split))
    X_test = X[test_start:]
    S_test = S[test_start:] if S is not None else None

    offset = ws + seq_len
    test_prices = close_prices[offset + test_start : offset + test_start + len(X_test)]

    # Load model
    model = DualBranchModel.from_config(cfg.model)
    ckpt_path = f"{cfg.paths.model_dir}/{cfg.model.name}_best.pt"

    if Path(ckpt_path).exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded: {ckpt_path}")
    else:
        logger.warning(f"No checkpoint at {ckpt_path}")

    # Predict
    model.to(device)
    model.eval()
    predictions, confidences = [], []

    with torch.no_grad():
        for i in range(0, len(X_test), 512):
            bx = torch.FloatTensor(X_test[i:i+512]).to(device)
            bs = torch.FloatTensor(S_test[i:i+512]).to(device) if S_test is not None else None
            logits, conf = model(bx, bs)
            predictions.extend(logits.argmax(dim=-1).cpu().numpy())
            confidences.extend(conf.squeeze(-1).cpu().numpy())

    signals = np.array(predictions)
    confs = np.array(confidences)

    min_len = min(len(signals), len(test_prices))
    signals, confs, test_prices = signals[:min_len], confs[:min_len], test_prices[:min_len]

    logger.info(
        f"Predictions: {len(signals)} | "
        f"Conf: mean={confs.mean():.3f} std={confs.std():.3f} | "
        f"Signals: sell={np.mean(signals==0):.1%} hold={np.mean(signals==1):.1%} buy={np.mean(signals==2):.1%}"
    )

    # Backtest
    bt_config = BacktestConfig(
        initial_balance=10_000.0,
        human_exit_approval=cfg.get("risk", {}).get("human_exit_approval", False),
    )
    engine = BacktestEngine(bt_config)
    result = engine.run(test_prices, signals, confs)

    # Results
    logger.info(f"\n{result.summary()}")
    logger.info(f"\n{result.profitable_trades_summary()}")

    if result.total_trades > 5:
        _confidence_analysis(result)

    # Export
    output_dir = Path(cfg.paths.log_dir) / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(output_dir / f"{cfg.model.name}_backtest.npz"),
        equity_curve=np.array(result.equity_curve),
        signals=signals, confidences=confs, prices=test_prices,
    )

    if export_csv:
        result.export_equity_csv(str(output_dir / "equity_curve.csv"))
        result.export_trades_csv(str(output_dir / "trades.csv"))

        # Simple equity plot
        _plot_equity(result.equity_curve, str(output_dir / "equity_curve.png"))

    logger.info(f"Results saved to {output_dir}")


def _confidence_analysis(result):
    """Performance by confidence band."""
    trades = result.trades
    bands = [
        (0.0, 0.3, "Low  (0-30%)"),
        (0.3, 0.5, "Med  (30-50%)"),
        (0.5, 0.7, "High (50-70%)"),
        (0.7, 1.01, "Best (70%+)"),
    ]

    logger.info(f"\n  CONFIDENCE-STRATIFIED ANALYSIS")
    logger.info(f"  {'Band':<16} {'N':>5} {'Win%':>7} {'AvgPnL':>10} {'TotPnL':>12} {'AvgHold':>8}")
    logger.info(f"  {'─'*60}")

    for lo, hi, name in bands:
        bt = [t for t in trades if lo <= t.confidence < hi]
        if not bt:
            logger.info(f"  {name:<16} {'—':>5}")
            continue
        n = len(bt)
        wins = sum(1 for t in bt if t.pnl_usd > 0)
        avg_pnl = np.mean([t.pnl_usd for t in bt])
        tot_pnl = sum(t.pnl_usd for t in bt)
        avg_hold = np.mean([t.hold_time for t in bt])
        logger.info(f"  {name:<16} {n:>5} {wins/n:>6.1%} ${avg_pnl:>9.2f} ${tot_pnl:>11.2f} {avg_hold:>7.0f}")

    # Key insight
    best_band = [t for t in trades if t.confidence >= 0.7]
    if best_band:
        best_pf = sum(t.pnl_usd for t in best_band if t.pnl_usd > 0) / max(abs(sum(t.pnl_usd for t in best_band if t.pnl_usd <= 0)), 1e-8)
        logger.info(f"\n  High-confidence (70%+) profit factor: {best_pf:.2f}")


def _plot_equity(equity_curve, path):
    """Save a simple equity curve plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(equity_curve, linewidth=0.8, color="#2563eb")
        ax.axhline(y=equity_curve[0], color="#94a3b8", linewidth=0.5, linestyle="--")
        ax.fill_between(range(len(equity_curve)), equity_curve[0], equity_curve,
                         where=[e >= equity_curve[0] for e in equity_curve],
                         alpha=0.15, color="#22c55e")
        ax.fill_between(range(len(equity_curve)), equity_curve[0], equity_curve,
                         where=[e < equity_curve[0] for e in equity_curve],
                         alpha=0.15, color="#ef4444")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Equity ($)")
        ax.set_title("Equity Curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info(f"Equity plot saved: {path}")
    except ImportError:
        logger.warning("matplotlib not available — skipping equity plot")


if __name__ == "__main__":
    main()
