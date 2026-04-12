#!/usr/bin/env python3
"""Backtest the dual-branch model with confidence filtering.

Usage:
    # Standard backtest
    python scripts/backtest.py model=dual_branch data=xauusd

    # Only trade when confidence > 0.5
    python scripts/backtest.py model=dual_branch data=xauusd ++min_confidence=0.5

    # With HITL approval on exits
    python scripts/backtest.py model=dual_branch data=xauusd ++risk.human_exit_approval=true

    # Show only profitable trade analysis
    python scripts/backtest.py model=dual_branch data=xauusd ++show_winners=true
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

    # Parse extra flags
    min_confidence = cfg.get("min_confidence", 0.0)
    show_winners = cfg.get("show_winners", False)

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

    # Label (for test split calculation)
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
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning(f"No checkpoint at {ckpt_path}")

    # Generate predictions
    model.to(device)
    model.eval()
    predictions = []
    confidences = []

    with torch.no_grad():
        batch_size = 512
        for i in range(0, len(X_test), batch_size):
            batch_x = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            batch_s = None
            if S_test is not None:
                batch_s = torch.FloatTensor(S_test[i:i+batch_size]).to(device)
            logits, conf = model(batch_x, batch_s)
            predictions.extend(logits.argmax(dim=-1).cpu().numpy())
            confidences.extend(conf.squeeze(-1).cpu().numpy())

    signals = np.array(predictions)
    confs = np.array(confidences)

    min_len = min(len(signals), len(test_prices))
    signals = signals[:min_len]
    confs = confs[:min_len]
    test_prices = test_prices[:min_len]

    # Log confidence distribution
    logger.info(
        f"Predictions: {len(signals)} signals | "
        f"Confidence: mean={confs.mean():.3f} min={confs.min():.3f} max={confs.max():.3f} | "
        f"Signal dist: sell={np.mean(signals==0):.1%} hold={np.mean(signals==1):.1%} buy={np.mean(signals==2):.1%}"
    )

    if min_confidence > 0:
        n_filtered = np.sum((confs < min_confidence) & (signals != 1))
        logger.info(f"Confidence filter: {n_filtered} signals below {min_confidence:.2f} will be skipped")

    # Run backtest
    bt_config = BacktestConfig(
        initial_balance=10_000.0,
        human_exit_approval=cfg.get("risk", {}).get("human_exit_approval", False),
        min_confidence=min_confidence,
    )
    engine = BacktestEngine(bt_config)
    result = engine.run(test_prices, signals, confs)

    # Print result
    logger.info(f"\n{result.summary()}")

    # Show profitable trades analysis
    if show_winners or True:  # always show this
        logger.info(f"\n{result.profitable_trades_summary()}")

    # Confidence-stratified analysis
    if result.total_trades > 5:
        _confidence_analysis(result)

    # Save
    output_dir = Path(cfg.paths.log_dir) / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_dir / f"{cfg.model.name}_backtest.npz"),
        equity_curve=np.array(result.equity_curve),
        signals=signals,
        confidences=confs,
        prices=test_prices,
    )
    logger.info(f"Results saved to {output_dir}")


def _confidence_analysis(result):
    """Show performance stratified by confidence bands."""
    trades = result.trades
    if not trades:
        return

    bands = [(0, 0.3, "Low"), (0.3, 0.5, "Med"), (0.5, 0.7, "High"), (0.7, 1.0, "V.High")]
    logger.info("\n  Confidence-stratified analysis:")
    logger.info(f"  {'Band':<10} {'Trades':>7} {'Win%':>7} {'Avg PnL':>10} {'Tot PnL':>12}")
    logger.info(f"  {'─'*48}")

    for lo, hi, name in bands:
        band_trades = [t for t in trades if lo <= t.confidence < hi]
        if not band_trades:
            continue
        n = len(band_trades)
        wins = sum(1 for t in band_trades if t.pnl_usd > 0)
        avg = np.mean([t.pnl_usd for t in band_trades])
        total = sum(t.pnl_usd for t in band_trades)
        logger.info(f"  {name:<10} {n:>7} {wins/n:>6.1%} ${avg:>9.2f} ${total:>11.2f}")


if __name__ == "__main__":
    main()
