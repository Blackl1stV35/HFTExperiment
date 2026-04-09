"""Supervised training pipeline for the dual-branch model.

Handles:
    - Class-weighted loss with sqrt dampening + max 5:1 cap
    - Separate sentiment tensor (3-element TensorDataset: X, S, y)
    - Confidence head trained alongside signal head
    - Gradient norm logging for stability monitoring
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from src.encoder.fusion import DualBranchModel
from src.training.labels import LabelConfig, get_labeler, create_sequences
from src.data.preprocessing import WindowMinMaxScaler, ZScoreScaler
from src.data.tick_store import TickStore
from src.utils.config import get_device, set_seed, load_env
from src.utils.logger import setup_logger


class Trainer:
    """Training loop for the dual-branch model."""

    def __init__(self, cfg: DictConfig, model: DualBranchModel, device: torch.device):
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
        self.criterion = nn.CrossEntropyLoss()

        # Confidence loss: MSE against "correctness" (1 if prediction matches label, 0 otherwise)
        self.confidence_criterion = nn.MSELoss()

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def set_class_weights(self, y_train: np.ndarray) -> None:
        """Compute and apply class weights with sqrt dampening and 5:1 cap."""
        counts = np.bincount(y_train, minlength=3)
        if counts.min() == 0:
            logger.warning(f"Zero-count class detected: {counts}. Using uniform weights.")
            return

        weights = 1.0 / np.sqrt(counts.astype(np.float32))
        weights = weights / weights.min()
        weights = np.clip(weights, 1.0, 5.0)
        weights = weights / weights.sum() * len(weights)

        self.criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(weights).to(self.device)
        )
        logger.info(f"Class weights: sell={weights[0]:.3f}, hold={weights[1]:.3f}, buy={weights[2]:.3f}")

    def _unpack_batch(self, batch):
        """Unpack (X, S, y) or (X, y) batch."""
        if len(batch) == 3:
            return batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        else:
            return batch[0].to(self.device), None, batch[-1].to(self.device)

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in loader:
            X, S, y = self._unpack_batch(batch)

            self.optimizer.zero_grad()
            logits, confidence = self.model(X, S)

            # Signal loss
            signal_loss = self.criterion(logits, y)

            # Confidence loss: target = 1 if correct, 0 if wrong
            with torch.no_grad():
                is_correct = (logits.argmax(dim=-1) == y).float().unsqueeze(-1)
            conf_loss = self.confidence_criterion(confidence, is_correct)

            loss = signal_loss + 0.3 * conf_loss  # confidence is auxiliary

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip)
            self.optimizer.step()

            total_loss += signal_loss.item() * X.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += X.size(0)

        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5

        self.scheduler.step()
        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_confs = []

        for batch in loader:
            X, S, y = self._unpack_batch(batch)
            logits, confidence = self.model(X, S)
            loss = self.criterion(logits, y)

            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += X.size(0)
            all_confs.append(confidence.cpu())

        confs = torch.cat(all_confs)
        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
            "confidence_mean": confs.mean().item(),
            "confidence_std": confs.std().item(),
        }

    def check_early_stopping(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.cfg.training.early_stopping_patience

    def save_checkpoint(self, path: str, epoch: int, metrics: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": OmegaConf.to_container(self.cfg),
        }, path)
        logger.info(f"Checkpoint saved: {path}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logger()
    set_seed(cfg.project.seed)
    device = get_device(cfg.project.device)
    logger.info(f"Device: {device}")

    # W&B
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project=cfg.logging.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.model.name}_{cfg.data.name}",
        )
    except Exception as e:
        logger.warning(f"W&B not available: {e}")

    # Load data
    store = TickStore(cfg.paths.data_dir + "/ticks.duckdb")
    df = store.query_ohlcv(cfg.data.symbol, cfg.data.timeframe)
    store.close()

    if df.is_empty():
        logger.error("No data. Run scripts/download_data.py first.")
        return

    # Preprocess
    from src.data.preprocessing import prepare_features
    features, close_prices = prepare_features(
        df,
        scaler_method=cfg.data.preprocessing.scaling,
        window_size=cfg.data.preprocessing.window_size,
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

    # Trim scaler warmup
    ws = cfg.data.preprocessing.window_size
    features = features[ws:]
    labels = labels[ws:]

    # Load sentiment if enabled
    sentiment = None
    if cfg.data.sentiment.enabled:
        emb_path = cfg.paths.data_dir + "/sentiment_embeddings.npy"
        if Path(emb_path).exists():
            all_emb = np.load(emb_path, allow_pickle=True)
            if isinstance(all_emb, np.ndarray) and all_emb.ndim == 2:
                sentiment = all_emb[ws:]
                logger.info(f"Sentiment loaded: {sentiment.shape}")
            else:
                logger.warning("Bad sentiment format — rebuild with scripts/build_embeddings.py")
        else:
            logger.warning(f"Sentiment file not found: {emb_path}")

    # Create sequences
    seq_len = cfg.model.input.sequence_length
    X, y, S = create_sequences(features, labels, seq_len, sentiment)

    logger.info(f"Dataset: X={X.shape}, y={y.shape}, classes={np.bincount(y, minlength=3)}")

    # Split (chronological — no shuffle across time)
    n = len(X)
    n_test = int(n * cfg.training.test_split)
    n_val = int(n * cfg.training.val_split)
    n_train = n - n_val - n_test

    def make_ds(start, end):
        x = torch.FloatTensor(X[start:end])
        yy = torch.LongTensor(y[start:end])
        if S is not None:
            return TensorDataset(x, torch.FloatTensor(S[start:end]), yy)
        return TensorDataset(x, yy)

    train_ds = make_ds(0, n_train)
    val_ds = make_ds(n_train, n_train + n_val)
    test_ds = make_ds(n_train + n_val, n)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,
                              num_workers=cfg.training.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size)

    logger.info(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Build model
    model = DualBranchModel.from_config(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} params")

    # Trainer
    trainer = Trainer(cfg, model, device)
    trainer.set_class_weights(y[:n_train])

    for epoch in range(1, cfg.training.epochs + 1):
        train_m = trainer.train_epoch(train_loader)
        val_m = trainer.validate(val_loader)

        logger.info(
            f"Epoch {epoch}/{cfg.training.epochs} | "
            f"Train: loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} | "
            f"Val: loss={val_m['loss']:.4f} acc={val_m['accuracy']:.4f} | "
            f"Conf: {val_m['confidence_mean']:.3f}±{val_m['confidence_std']:.3f} | "
            f"GradNorm: {train_m['grad_norm']:.2f}"
        )

        if wandb_run:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_m["loss"], "train/accuracy": train_m["accuracy"],
                "train/lr": train_m["lr"], "train/grad_norm": train_m["grad_norm"],
                "val/loss": val_m["loss"], "val/accuracy": val_m["accuracy"],
                "val/confidence_mean": val_m["confidence_mean"],
            })

        if val_m["loss"] < trainer.best_val_loss:
            trainer.save_checkpoint(f"{cfg.paths.model_dir}/{cfg.model.name}_best.pt", epoch, val_m)

        if trainer.check_early_stopping(val_m["loss"]):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Test
    test_m = trainer.validate(test_loader)
    logger.info(f"Test: loss={test_m['loss']:.4f} acc={test_m['accuracy']:.4f}")

    if wandb_run:
        wandb.log({"test/loss": test_m["loss"], "test/accuracy": test_m["accuracy"]})
        wandb_run.finish()


if __name__ == "__main__":
    main()
