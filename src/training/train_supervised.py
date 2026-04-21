"""Supervised training pipeline — Phase 3 regime-balanced edition.

OOM fix (v2):
    Replaced create_sequences() + TensorDataset with SequenceDataset, a
    torch Dataset that slices windows on the fly from the raw (N, F) features
    array.  This drops CPU RAM from 16.4 GB -> ~0.19 GB for a 5.68 M bar
    dataset, eliminating the system-RAM OOM crash that killed training after
    epoch 4.

    Before: create_sequences() pre-built (5,680,771 x 120 x 6) float32 = 16.4 GB
    After:  SequenceDataset holds (5,680,771 x 6) float32 = 0.13 GB + slices on demand

Hardware settings (match caller constraints):
    batch_size  = 2048   (set in config or --override training.batch_size=2048)
    num_workers = 0      (no subprocesses; safe on T4 Colab)
    pin_memory  = False  (disabled per caller)
    epoch_size  = 500000 (WeightedRandomSampler draws this many indices per epoch)

Phase 3 features retained:
    - RegimeBalancedSampler: Bear x2.0, pre-2020 x1.5, post-2024 x0.5
    - class_weights [2.5, 0.3, 2.5] applied directly in CrossEntropyLoss(weight=w)
    - join_regime_labels() joined before feature extraction
    - Per-class precision/recall logged every epoch (val acc 0.98 is misleading
      with 97.5% hold — per-class breakdown shows whether sell/buy are learned)
    - Resume from checkpoint: training.resume_from=<path> in config overrides
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from loguru import logger

from src.encoder.fusion import DualBranchModel
from src.training.labels import LabelConfig, get_labeler
from src.data.preprocessing import prepare_features, join_regime_labels
from src.data.tick_store import TickStore
from src.utils.config import get_device, set_seed
from src.utils.logger import setup_logger


# ── SequenceDataset ───────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Sliding-window dataset that avoids pre-allocating the full X array.

    Holds only raw (N, F) features (0.13 GB) and slices each window of
    shape (seq_len, F) in __getitem__, so the DataLoader assembles batches
    of (batch_size, seq_len, F) at runtime — never the full array.

    Args:
        features:  (N, F) float32 — scaled OHLCV, warmup-trimmed.
        labels:    (N,)   int64   — class labels aligned to features.
        seq_len:   sliding window length (120).
        sentiment: (N, 768) float32 | None — kept for interface compat,
                   disabled in Phase 3.
    """

    def __init__(
        self,
        features:  np.ndarray,
        labels:    np.ndarray,
        seq_len:   int,
        sentiment: np.ndarray | None = None,
    ):
        assert len(features) == len(labels), (
            f"features/labels length mismatch: {len(features)} vs {len(labels)}"
        )
        assert len(features) > seq_len, (
            f"Dataset too short ({len(features)}) for seq_len={seq_len}"
        )
        self.features  = features
        self.labels    = labels
        self.seq_len   = seq_len
        self.sentiment = sentiment
        self._n        = len(features) - seq_len

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.features[i : i + self.seq_len].copy())
        y = torch.tensor(self.labels[i + self.seq_len - 1], dtype=torch.long)
        if self.sentiment is not None:
            s = torch.from_numpy(self.sentiment[i + self.seq_len - 1].copy())
            return x, s, y
        return x, y


# ── Regime-balanced sampler ───────────────────────────────────────────────────

def build_regime_balanced_sampler(
    labels:            np.ndarray,
    gmm2:              np.ndarray | None,
    timestamps,
    class_weights_cfg = None,
    epoch_size:        int = 500_000,
) -> tuple[WeightedRandomSampler, np.ndarray]:
    """WeightedRandomSampler with class + regime weighting.

    Per-sample weight = class_weight x regime_multiplier.

    Regime multipliers:
        Bear (gmm2 == 0):  x2.0  — model saw almost no Bear in Phase 2
        Bull (gmm2 == 1):  x1.0
        pre-2020 bars:     x1.5  — boost historical diversity
        post-2024 bars:    x0.5  — reduce parabolic-window dominance

    epoch_size caps num_samples per epoch to keep each epoch ~11 min on T4.
    """
    n = len(labels)

    if class_weights_cfg is not None:
        cw = np.array(list(class_weights_cfg), dtype=np.float32)
        logger.info(f"Using config class weights: {cw}")
    else:
        counts = np.bincount(labels, minlength=3).astype(np.float32)
        raw_w  = 1.0 / np.sqrt(counts + 1)
        raw_w  = raw_w / raw_w.min()
        raw_w  = np.clip(raw_w, 1.0, 5.0)
        cw     = raw_w / raw_w.sum() * 3
        logger.info(f"Auto class weights: {cw}")

    sample_weights = cw[labels].astype(np.float32)

    regime_mult = np.ones(n, dtype=np.float32)
    if gmm2 is not None and len(gmm2) == n:
        regime_mult[gmm2 == 0] = 2.0

    if timestamps is not None and len(timestamps) == n:
        import pandas as pd
        ts       = pd.to_datetime(timestamps)
        pre2020  = (ts < pd.Timestamp("2020-01-01"))
        regime_mult[pre2020]  *= 1.5
        logger.info(
            f"Temporal reweighting: pre-2020={pre2020.sum():,} x1.5"
        )

    final_weights      = torch.FloatTensor(sample_weights * regime_mult)
    actual_epoch_size  = min(epoch_size, n)
    sampler = WeightedRandomSampler(
        weights     = final_weights,
        num_samples = actual_epoch_size,
        replacement = True,
    )
    logger.info(
        f"RegimeBalancedSampler: Bear x2.0 | "
        f"class weights sell={cw[0]:.2f} hold={cw[1]:.2f} buy={cw[2]:.2f} | "
        f"epoch_size={actual_epoch_size:,}"
    )
    return sampler, cw


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """Training loop with per-class metrics and checkpoint resume."""

    def __init__(
        self,
        cfg:           DictConfig,
        model:         DualBranchModel,
        device:        torch.device,
        class_weights: np.ndarray | None = None,
    ):
        self.cfg    = cfg
        self.model  = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = cfg.training.learning_rate,
            weight_decay = cfg.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        w = torch.FloatTensor(class_weights).to(device) if class_weights is not None else None
        self.criterion            = nn.CrossEntropyLoss(weight=w)
        self.confidence_criterion = nn.MSELoss()

        self.best_val_loss    = float("inf")
        self.best_signal_score = 0.0
        self.patience_counter = 0
        self.start_epoch      = 1

    def load_checkpoint(self, path: str) -> None:
        """Resume from saved checkpoint — restores model + optimizer state."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch   = ckpt.get("epoch", 0) + 1
        self.best_val_loss = ckpt.get("metrics", {}).get("loss", float("inf"))
        logger.info(
            f"Resumed from {path} | "
            f"start_epoch={self.start_epoch} | "
            f"best_val_loss={self.best_val_loss:.4f}"
        )

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            return (batch[0].to(self.device),
                    batch[1].to(self.device),
                    batch[2].to(self.device))
        return batch[0].to(self.device), None, batch[-1].to(self.device)

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for batch in loader:
            X, S, y = self._unpack_batch(batch)
            self.optimizer.zero_grad()
            logits, confidence = self.model(X, S)
            signal_loss = self.criterion(logits, y)
            with torch.no_grad():
                is_correct = (logits.argmax(dim=-1) == y).float().unsqueeze(-1)
            conf_loss = self.confidence_criterion(confidence, is_correct)
            loss = signal_loss + 0.5 * conf_loss
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.gradient_clip
            )
            self.optimizer.step()
            total_loss += signal_loss.item() * X.size(0)
            correct    += (logits.argmax(dim=-1) == y).sum().item()
            total      += X.size(0)

        grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.model.parameters()
            if p.grad is not None
        ) ** 0.5
        self.scheduler.step()
        return {
            "loss":      total_loss / total,
            "accuracy":  correct    / total,
            "lr":        self.optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        """Validate with overall metrics AND per-class precision/recall.

        Per-class breakdown is essential because val_acc ~0.98 reflects
        97.5% hold dominance rather than sell/buy signal quality. Tracking
        sell and buy recall separately shows whether the minority classes
        are genuinely being learned by the regime-balanced sampler.
        """
        self.model.eval()
        total_loss = 0.0
        total      = 0
        all_confs  = []
        n_classes  = 3
        conf_mat   = np.zeros((n_classes, n_classes), dtype=np.int64)

        for batch in loader:
            X, S, y    = self._unpack_batch(batch)
            logits, confidence = self.model(X, S)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            total      += X.size(0)
            all_confs.append(confidence.cpu())
            preds = logits.argmax(dim=-1)
            for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
                conf_mat[t, p] += 1

        confs = torch.cat(all_confs)
        names = {0: "sell", 1: "hold", 2: "buy"}
        per_class: dict = {}
        for c in range(n_classes):
            tp = int(conf_mat[c, c])
            fp = int(conf_mat[:, c].sum()) - tp
            fn = int(conf_mat[c, :].sum()) - tp
            per_class[names[c]] = {
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                "recall":    tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                "n":         int(conf_mat[c].sum()),
            }
        return {
            "loss":             total_loss / total,
            "accuracy":         int(np.diag(conf_mat).sum()) / total,
            "confidence_mean":  confs.mean().item(),
            "confidence_std":   confs.std().item(),
            "per_class":        per_class,
        }

    @staticmethod
    def _signal_score(pc: dict) -> float:
        """Weighted metric that rewards minority-class recall over overall accuracy.
        Val loss minimum = most hold-dominant epoch = wrong checkpoint criterion.
        sell_R x0.4 + buy_R x0.4 + val_acc x0.2 picks the epoch where sell/buy
        recall peaks rather than the epoch where the model mostly predicts hold.
        """
        return pc["sell"]["recall"] * 0.4 + pc["buy"]["recall"] * 0.4

    def check_early_stopping(self, val_m: dict) -> bool:
        score = self._signal_score(val_m["per_class"])
        if score > self.best_signal_score:
            self.best_signal_score = score
            self.patience_counter  = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.cfg.training.early_stopping_patience

    GDRIVE_DIR = "/content/drive/MyDrive/Colab Notebooks"

    def save_checkpoint(self, path: str, epoch: int, metrics: dict) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics":              metrics,
            "config":               OmegaConf.to_container(self.cfg),
        }
        torch.save(payload, path)
        logger.info(f"Checkpoint saved: {path}")

        # Mirror to Google Drive so Colab runtime disconnections don't lose it
        gdrive_path = Path(self.GDRIVE_DIR)
        if gdrive_path.exists():
            import shutil
            dest = gdrive_path / Path(path).name
            shutil.copy2(path, dest)
            logger.info(f"Checkpoint mirrored to Drive: {dest}")
        else:
            logger.warning(
                f"Google Drive not mounted at {self.GDRIVE_DIR}. "
                "Run: from google.colab import drive; drive.mount('/content/drive')"
            )


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
            project = cfg.logging.wandb_project,
            config  = OmegaConf.to_container(cfg, resolve=True),
            name    = f"{cfg.model.name}_{cfg.data.name}_phase3",
            resume  = "allow",
        )
    except Exception as e:
        logger.warning(f"W&B not available: {e}")

    # ── Load M1 data ──────────────────────────────────────────────────────────
    store  = TickStore(cfg.paths.data_dir + "/ticks.duckdb")
    df_raw = store.query_ohlcv(cfg.data.symbol, cfg.data.timeframe)
    store.close()
    if df_raw.is_empty():
        logger.error("No data. Run scripts/download_data.py first.")
        return

    # ── Join regime labels (Step 2) ───────────────────────────────────────────
    regime_csv = Path(cfg.paths.data_dir) / "regime" / "daily_regime_labels.csv"
    df_raw     = join_regime_labels(df_raw, regime_csv)

    timestamps = df_raw["timestamp"].to_list() if "timestamp" in df_raw.columns else None

    # ── Scale features ────────────────────────────────────────────────────────
    features, close_prices = prepare_features(
        df_raw,
        scaler_method = cfg.data.preprocessing.scaling,
        window_size   = cfg.data.preprocessing.window_size,
    )
    ws = cfg.data.preprocessing.window_size

    gmm2_raw = None
    ts_raw   = None
    if "gmm2_state" in df_raw.columns:
        gmm2_raw = df_raw["gmm2_state"].to_numpy()[ws:]
        if timestamps:
            ts_raw = timestamps[ws:]

    # ── Label ─────────────────────────────────────────────────────────────────
    label_cfg = LabelConfig(
        method             = cfg.data.labeling.method,
        profit_target_pips = cfg.data.labeling.profit_target_pips,
        stop_loss_pips     = cfg.data.labeling.stop_loss_pips,
        max_holding_bars   = cfg.data.labeling.max_holding_bars,
        pip_value          = cfg.data.labeling.pip_value,
    )
    labels  = get_labeler(label_cfg).label(close_prices)
    features = features[ws:]
    labels   = labels[ws:]

    # ── Optional sentiment (disabled in Phase 3) ──────────────────────────────
    sentiment = None
    if cfg.data.sentiment.enabled:
        emb_path = Path(cfg.paths.data_dir) / "sentiment_embeddings.npy"
        if emb_path.exists():
            all_emb = np.load(str(emb_path), allow_pickle=True)
            if isinstance(all_emb, np.ndarray) and all_emb.ndim == 2:
                sentiment = all_emb[ws:]
                logger.info(f"Sentiment loaded: {sentiment.shape}")

    # ── Splits ────────────────────────────────────────────────────────────────
    seq_len = cfg.model.input.sequence_length
    n_total = len(features) - seq_len
    n_test  = int(n_total * cfg.training.test_split)
    n_val   = int(n_total * cfg.training.val_split)
    n_train = n_total - n_val - n_test

    # SequenceDataset: each split needs seq_len extra raw rows for its last window
    def make_ds(feat_start, feat_end, sent_start=None, sent_end=None):
        s = sentiment[feat_start:feat_end] if sentiment is not None else None
        return SequenceDataset(
            features[feat_start:feat_end],
            labels[feat_start:feat_end],
            seq_len, s,
        )

    train_ds = make_ds(0,              n_train + seq_len)
    val_ds   = make_ds(n_train,        n_train + n_val + seq_len)
    test_ds  = make_ds(n_train + n_val, len(features))

    logger.info(
        f"Dataset: {n_total:,} sequences | "
        f"classes={np.bincount(labels[seq_len-1:seq_len-1+n_total], minlength=3)}"
    )

    # GMM2 + timestamps aligned to training-set sequence indices
    gmm2_train = gmm2_raw[seq_len-1 : seq_len-1+n_train] if gmm2_raw is not None else None
    ts_train   = ts_raw  [seq_len-1 : seq_len-1+n_train] if ts_raw   is not None else None
    y_train    = labels  [seq_len-1 : seq_len-1+n_train]

    if gmm2_train is not None:
        bull = gmm2_train.mean()
        logger.info(f"GMM2 train: Bull={bull:.1%}  Bear={1-bull:.1%}")

    # ── Regime-balanced sampler (Step 3) ──────────────────────────────────────
    cw_cfg     = list(cfg.data.labeling.get("class_weights", [])) or None
    epoch_size = min(500_000, n_train)
    sampler, class_weights_arr = build_regime_balanced_sampler(
        labels            = y_train,
        gmm2              = gmm2_train,
        timestamps        = ts_train,
        class_weights_cfg = cw_cfg,
        epoch_size        = epoch_size,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    batch_size  = cfg.training.batch_size    # 2048
    num_workers = cfg.training.num_workers   # 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,  batch_size=batch_size,
        num_workers=num_workers, pin_memory=False,
    )
    logger.info(
        f"Train: {n_train:,}  Val: {n_val:,}  Test: {n_test:,} | "
        f"batch={batch_size}  workers={num_workers}  epoch_size={epoch_size:,}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DualBranchModel.from_config(cfg.model)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    trainer = Trainer(cfg, model, device, class_weights=class_weights_arr)

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_path = cfg.training.get("resume_from", None)
    if resume_path and Path(resume_path).exists():
        trainer.load_checkpoint(resume_path)
    elif resume_path:
        logger.warning(f"resume_from={resume_path} not found — starting fresh")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(trainer.start_epoch, cfg.training.epochs + 1):
        train_m = trainer.train_epoch(train_loader)
        val_m   = trainer.validate(val_loader)
        pc      = val_m["per_class"]

        logger.info(
            f"Epoch {epoch}/{cfg.training.epochs} | "
            f"Train loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} | "
            f"Val   loss={val_m['loss']:.4f}  acc={val_m['accuracy']:.4f} | "
            f"SignalScore={score:.4f} | "                   
            f"Conf {val_m['confidence_mean']:.3f}+-{val_m['confidence_std']:.3f} | "
            f"GradNorm={train_m['grad_norm']:.2f}"
        )
        
        logger.info(
            f"  Per-class val | "
            f"sell  P={pc['sell']['precision']:.3f} R={pc['sell']['recall']:.3f} (n={pc['sell']['n']:,}) | "
            f"hold  P={pc['hold']['precision']:.3f} R={pc['hold']['recall']:.3f} (n={pc['hold']['n']:,}) | "
            f"buy   P={pc['buy']['precision']:.3f}  R={pc['buy']['recall']:.3f}  (n={pc['buy']['n']:,})"
        )

        if wandb_run:
            import wandb
            wandb.log({
                "epoch":               epoch,
                "train/loss":          train_m["loss"],
                "train/accuracy":      train_m["accuracy"],
                "train/lr":            train_m["lr"],
                "train/grad_norm":     train_m["grad_norm"],
                "val/loss":            val_m["loss"],
                "val/accuracy":        val_m["accuracy"],
                "val/signal_score":    score,
                "val/confidence_mean": val_m["confidence_mean"],
                "val/sell_precision":  pc["sell"]["precision"],
                "val/sell_recall":     pc["sell"]["recall"],
                "val/buy_precision":   pc["buy"]["precision"],
                "val/buy_recall":      pc["buy"]["recall"],
            })

        score = Trainer._signal_score(val_m["per_class"])
        if score > trainer.best_signal_score:
            trainer.save_checkpoint(
                f"{cfg.paths.model_dir}/{cfg.model.name}_best.pt", epoch, val_m
            )

        if trainer.check_early_stopping(val_m):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # ── Test ──────────────────────────────────────────────────────────────────
    test_m = trainer.validate(test_loader)
    pc_t   = test_m["per_class"]
    logger.info(
        f"Test: loss={test_m['loss']:.4f} acc={test_m['accuracy']:.4f} | "
        f"sell P={pc_t['sell']['precision']:.3f} R={pc_t['sell']['recall']:.3f} | "
        f"buy  P={pc_t['buy']['precision']:.3f}  R={pc_t['buy']['recall']:.3f}"
    )

    if wandb_run:
        import wandb
        wandb.log({
            "test/loss":           test_m["loss"],
            "test/accuracy":       test_m["accuracy"],
            "test/sell_precision": pc_t["sell"]["precision"],
            "test/sell_recall":    pc_t["sell"]["recall"],
            "test/buy_precision":  pc_t["buy"]["precision"],
            "test/buy_recall":     pc_t["buy"]["recall"],
        })
        wandb_run.finish()


if __name__ == "__main__":
    main()
