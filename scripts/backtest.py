#!/usr/bin/env python3
"""Backtest the dual-branch model.

Usage:
    python scripts/backtest.py model=dual_branch data=xauusd
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
from src.utils.config import get_device, set_seed, setup_logger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logger()
    set_seed(cfg.project.seed)
    device = get_device(cfg.project.device)

    # Load data
    store = TickStore(cfg.paths.data_dir + "/ticks.duckdb")
    df = store.query_ohlcv(cfg.data.symbol, cfg.data.timeframe)
    store.close()

    if df.is_empty():
        logger.error("No data. Run scripts/download_data.py first.")
        return

    # Preprocess
    features, close_prices = prepare_features(
        df, cfg.data.preprocessing.scaling, cfg.data.preprocessing.window_size,
    )

    # Label
    label_cfg = LabelConfig(
        method=cfg.data.labeling.method,
        profit_target_pips=cfg.data.labeling.profit_target_pips,
        stop_loss_pips=cfg.data.labeling.stop_loss_pips,
        max_holding_bars=cfg.data.labeling.max_holding_bars,
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

    # Test split only
    n = len(X)
    test_start = int(n * (1 - cfg.training.test_split))
    X_test = X[test_start:]
    S_test = S[test_start:] if S is not None else None

    # Corresponding prices
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
        logger.warning(f"No checkpoint at {ckpt_path} — using random weights")

    # Generate predictions
    model.to(device)
    model.eval()
    predictions = []
    confidences = []

    with torch.no_grad():
        batch_size = 512
        for i in range(0, len(X_test), batch_size):
            batch_x = torch.FloatTensor(X_test[i : i + batch_size]).to(device)
            batch_s = None
            if S_test is not None:
                batch_s = torch.FloatTensor(S_test[i : i + batch_size]).to(device)

            logits, conf = model(batch_x, batch_s)
            preds = logits.argmax(dim=-1).cpu().numpy()
            confs = conf.squeeze(-1).cpu().numpy()
            predictions.extend(preds)
            confidences.extend(confs)

    signals = np.array(predictions)
    confs = np.array(confidences)

    # Align lengths
    min_len = min(len(signals), len(test_prices))
    signals = signals[:min_len]
    confs = confs[:min_len]
    test_prices = test_prices[:min_len]

    # Run backtest
    bt_config = BacktestConfig(
        initial_balance=10_000.0,
        human_exit_approval=cfg.get("risk", {}).get("human_exit_approval", False),
    )
    engine = BacktestEngine(bt_config)
    result = engine.run(test_prices, signals, confs)

    logger.info(f"\n{result.summary()}")

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


if __name__ == "__main__":
    main()
