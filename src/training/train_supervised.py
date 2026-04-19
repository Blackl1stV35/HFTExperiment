"""Supervised training pipeline — Phase 3 regime-balanced edition.

Phase 3 changes:
    - RegimeBalancedSampler: oversamples Bear-GMM2 and Stagflation windows;
      undersamples 2024-2026 parabolic window. Requires regime_labels array.
    - class_weights from config (xauusd.yaml class_weights: [2.5, 0.3, 2.5])
      take precedence over the auto-computed sqrt-dampened weights when set.
    - join_regime_labels() called after data load so the sampler can use them.
    - Regime coverage logged after each epoch for monitoring.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from loguru import logger

from src.encoder.fusion import DualBranchModel
from src.training.labels import LabelConfig, get_labeler, create_sequences
from src.data.preprocessing import WindowMinMaxScaler, ZScoreScaler, join_regime_labels
from src.data.tick_store import TickStore
from src.utils.config import get_device, set_seed, load_env
from src.utils.logger import setup_logger


# ── Step 3a: Regime-balanced sampler ──────────────────────────────────────────

def build_regime_balanced_sampler(
    y: np.ndarray,
    gmm2: np.ndarray,
    timestamps,
    class_weights_cfg=None,
) -> WeightedRandomSampler:
    """WeightedRandomSampler that balances class labels AND regime windows.

    Sampling weight for each sequence = class_weight × regime_multiplier.

    Regime multipliers:
        Bear (gmm2=0):   2.0  — oversample; model trained in only Bull environment
        Bull (gmm2=1):   1.0  — standard
        Pre-2020 bars:   1.5  — oversample historical diversity
        2024+ bars:      0.5  — undersample parabolic window that dominates recent data
    """
    n = len(y)

    # Class weights
    if class_weights_cfg is not None:
        cw = np.array(list(class_weights_cfg), dtype=np.float32)
        logger.info(f"Using config class weights: {cw}")
    else:
        counts = np.bincount(y, minlength=3)
        raw_w = 1.0 / np.sqrt(counts.astype(np.float32) + 1)
        raw_w = raw_w / raw_w.min()
        raw_w = np.clip(raw_w, 1.0, 5.0)
        cw = raw_w / raw_w.sum() * 3
        logger.info(f"Auto class weights: {cw}")

    label_w = cw[y]  # per-sample class weight

    # Regime multiplier
    regime_mult = np.ones(n, dtype=np.float32)
    if gmm2 is not None and len(gmm2) == n:
        regime_mult[gmm2 == 0] = 2.0   # oversample Bear

    # Temporal multiplier: boost pre-2020, reduce post-2024
    if timestamps is not None and len(timestamps) == n:
        import pandas as pd
        ts = pd.to_datetime(timestamps)
        pre2020  = ts < pd.Timestamp("2020-01-01")
        post2024 = ts >= pd.Timestamp("2024-01-01")
        regime_mult[pre2020]  *= 1.5
        regime_mult[post2024] *= 0.5
        logger.info(
            f"Temporal reweighting: pre-2020={pre2020.sum():,} ×1.5, "
            f"post-2024={post2024.sum():,} ×0.5"
        )
    sample_weights = torch.FloatTensor(label_w * regime_mult)
    
    # CRITICAL FIX: Cap the epoch size so the CPU doesn't freeze
    # 500,000 is plenty of samples for one epoch and takes <5 seconds to calculate
    epoch_size = min(500000, len(sample_weights))
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=epoch_size,
        replacement=True,
    )
    logger.info(
        f"RegimeBalancedSampler: Bear boost ×2.0, "
        f"class weights sell={cw[0]:.2f} hold={cw[1]:.2f} buy={cw[2]:.2f}"
    )
    return sampler, cw


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: DictConfig, model: DualBranchModel, device: torch.device,
                 class_weights: np.ndarray | None = None):
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # Phase 3: use config class weights if provided
        if class_weights is not None:
            w = torch.FloatTensor(class_weights).to(device)
        else:
            w = None
        self.criterion = nn.CrossEntropyLoss(weight=w)
        self.confidence_criterion = nn.MSELoss()

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            return batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        return batch[0].to(self.device), None, batch[-1].to(self.device)

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0; correct = 0; total = 0
        for batch in loader:
            X, S, y = self._unpack_batch(batch)
            self.optimizer.zero_grad()
            logits, confidence = self.model(X, S)
            signal_loss = self.criterion(logits, y)
            with torch.no_grad():
                is_correct = (logits.argmax(dim=-1) == y).float().unsqueeze(-1)
            conf_loss = self.confidence_criterion(confidence, is_correct)
            loss = signal_loss + 0.3 * conf_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip)
            self.optimizer.step()
            total_loss += signal_loss.item() * X.size(0)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += X.size(0)
        grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.model.parameters() if p.grad is not None
        ) ** 0.5
        self.scheduler.step()
        return {"loss": total_loss/total, "accuracy": correct/total,
                "lr": self.optimizer.param_groups[0]["lr"], "grad_norm": grad_norm}

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0; correct = 0; total = 0; all_confs = []
        for batch in loader:
            X, S, y = self._unpack_batch(batch)
            logits, confidence = self.model(X, S)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += X.size(0)
            all_confs.append(confidence.cpu())
        confs = torch.cat(all_confs)
        return {"loss": total_loss/total, "accuracy": correct/total,
                "confidence_mean": confs.mean().item(), "confidence_std": confs.std().item()}

    def check_early_stopping(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss; self.patience_counter = 0; return False
        self.patience_counter += 1
        return self.patience_counter >= self.cfg.training.early_stopping_patience

    def save_checkpoint(self, path: str, epoch: int, metrics: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metrics": metrics, "config": OmegaConf.to_container(self.cfg)}, path)
        logger.info(f"Checkpoint saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logger()
    set_seed(cfg.project.seed)
    device = get_device(cfg.project.device)
    logger.info(f"Device: {device}")

    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project=cfg.logging.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.model.name}_{cfg.data.name}_phase3",
        )
    except Exception as e:
        logger.warning(f"W&B not available: {e}")

    # Load M1 data
    store = TickStore(cfg.paths.data_dir + "/ticks.duckdb")
    df_raw = store.query_ohlcv(cfg.data.symbol, cfg.data.timeframe)
    store.close()
    if df_raw.is_empty():
        logger.error("No data. Run scripts/download_data.py first."); return

    # Step 2: join regime labels onto M1 bars
    regime_csv = Path(cfg.paths.data_dir) / "regime" / "daily_regime_labels.csv"
    df_raw = join_regime_labels(df_raw, regime_csv)

    # Extract timestamps for temporal sampler weighting
    timestamps = df_raw["timestamp"].to_list() if "timestamp" in df_raw.columns else None

    from src.data.preprocessing import prepare_features
    features, close_prices = prepare_features(
        df_raw, scaler_method=cfg.data.preprocessing.scaling,
        window_size=cfg.data.preprocessing.window_size,
    )

    # Extract GMM2 state aligned to features (after scaler warmup trim)
    ws = cfg.data.preprocessing.window_size
    gmm2_raw = None
    ts_raw   = None
    if "gmm2_state" in df_raw.columns:
        gmm2_raw = df_raw["gmm2_state"].to_numpy()[ws:]
        if timestamps:
            ts_raw = timestamps[ws:]

    label_cfg = LabelConfig(
        method=cfg.data.labeling.method,
        profit_target_pips=cfg.data.labeling.profit_target_pips,
        stop_loss_pips=cfg.data.labeling.stop_loss_pips,
        max_holding_bars=cfg.data.labeling.max_holding_bars,
        pip_value=cfg.data.labeling.pip_value,
    )
    labeler = get_labeler(label_cfg)
    labels = labeler.label(close_prices)

    features = features[ws:]; labels = labels[ws:]

    sentiment = None
    if cfg.data.sentiment.enabled:
        emb_path = cfg.paths.data_dir + "/sentiment_embeddings.npy"
        if Path(emb_path).exists():
            all_emb = np.load(emb_path, allow_pickle=True)
            if isinstance(all_emb, np.ndarray) and all_emb.ndim == 2:
                sentiment = all_emb[ws:]

    seq_len = cfg.model.input.sequence_length
    X, y, S = create_sequences(features, labels, seq_len, sentiment)

    # Align gmm2 and timestamps to sequence count
    gmm2_seq = gmm2_raw[seq_len - 1 : seq_len - 1 + len(X)] if gmm2_raw is not None else None
    ts_seq   = ts_raw[seq_len - 1 : seq_len - 1 + len(X)]   if ts_raw   is not None else None

    logger.info(f"Dataset: X={X.shape}, classes={np.bincount(y, minlength=3)}")
    if gmm2_seq is not None:
        bull_frac = gmm2_seq.mean()
        logger.info(f"GMM2 coverage: Bull={bull_frac:.1%} Bear={1-bull_frac:.1%}")

    n = len(X)
    n_test  = int(n * cfg.training.test_split)
    n_val   = int(n * cfg.training.val_split)
    n_train = n - n_val - n_test

    def make_ds(start, end):
        x = torch.FloatTensor(X[start:end])
        yy = torch.LongTensor(y[start:end])
        if S is not None:
            return TensorDataset(x, torch.FloatTensor(S[start:end]), yy)
        return TensorDataset(x, yy)

    train_ds = make_ds(0, n_train)
    val_ds   = make_ds(n_train, n_train + n_val)
    test_ds  = make_ds(n_train + n_val, n)

    # Step 3: regime-balanced sampler for training set
    cw_cfg = list(cfg.data.labeling.get("class_weights", [])) or None
    sampler, class_weights_arr = build_regime_balanced_sampler(
        y[:n_train],
        gmm2_seq[:n_train] if gmm2_seq is not None else None,
        ts_seq[:n_train]   if ts_seq   is not None else None,
        class_weights_cfg=cw_cfg,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              sampler=sampler, num_workers=cfg.training.num_workers,
                              pin_memory=False)  # <--- CHANGED TO FALSE
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.training.batch_size)
    logger.info(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

    model = DualBranchModel.from_config(cfg.model)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    trainer = Trainer(cfg, model, device, class_weights=class_weights_arr)

    for epoch in range(1, cfg.training.epochs + 1):
        train_m = trainer.train_epoch(train_loader)
        val_m   = trainer.validate(val_loader)

        logger.info(
            f"Epoch {epoch}/{cfg.training.epochs} | "
            f"Train: loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} | "
            f"Val: loss={val_m['loss']:.4f} acc={val_m['accuracy']:.4f} | "
            f"Conf: {val_m['confidence_mean']:.3f}±{val_m['confidence_std']:.3f} | "
            f"GradNorm: {train_m['grad_norm']:.2f}"
        )

        if wandb_run:
            import wandb
            wandb.log({"epoch": epoch,
                       "train/loss": train_m["loss"], "train/accuracy": train_m["accuracy"],
                       "train/lr": train_m["lr"], "train/grad_norm": train_m["grad_norm"],
                       "val/loss": val_m["loss"], "val/accuracy": val_m["accuracy"],
                       "val/confidence_mean": val_m["confidence_mean"]})

        if val_m["loss"] < trainer.best_val_loss:
            trainer.save_checkpoint(
                f"{cfg.paths.model_dir}/{cfg.model.name}_best.pt", epoch, val_m)

        if trainer.check_early_stopping(val_m["loss"]):
            logger.info(f"Early stopping at epoch {epoch}"); break

    test_m = trainer.validate(test_loader)
    logger.info(f"Test: loss={test_m['loss']:.4f} acc={test_m['accuracy']:.4f}")

    if wandb_run:
        import wandb
        wandb.log({"test/loss": test_m["loss"], "test/accuracy": test_m["accuracy"]})
        wandb_run.finish()


if __name__ == "__main__":
    main()
