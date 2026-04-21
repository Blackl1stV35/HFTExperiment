"""Supervised training pipeline — Phase 3 regime-balanced edition.

OOM fix (v2):
    SequenceDataset slices windows on the fly — avoids pre-allocating the
    full (N, seq_len, F) array that caused the 16.4 GB RAM OOM crash.

v3 fixes applied from training review:
    1. gradient_clip raised 2.0 -> 5.0  (was hitting ceiling every batch)
    2. learning_rate lowered 2e-4 -> 1e-4
    3. post-2024 x0.5 undersample removed (test set IS post-2024 data)
    4. Checkpoint criterion: signal_score = sell_R*0.4 + buy_R*0.4
       replaces val_loss minimum (which selected the most underfitting epoch)
    5. conf_loss weight raised 0.3 -> 0.5
    6. Google Drive mirroring in save_checkpoint

Hardware settings:
    batch_size  = 2048  (config or override)
    num_workers = 0     (Colab T4)
    pin_memory  = False
    epoch_size  = 500_000
"""

from __future__ import annotations

import shutil
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
    """Sliding-window dataset — computes windows on the fly.

    Holds only (N, F) features in RAM instead of (N, seq_len, F),
    dropping peak allocation from 16.4 GB to ~0.13 GB for 5.68 M bars.
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
    """WeightedRandomSampler: per-sample weight = class_weight x regime_mult.

    Regime multipliers:
        Bear (gmm2=0):  x2.0  — underseen in Phase 2
        Bull (gmm2=1):  x1.0
        pre-2020 bars:  x1.5  — boost historical diversity
        post-2024 bars: x1.0  — FIX: removed x0.5 undersample; test set IS
                                post-2024 data, downsampling it caused 30x
                                buy-recall collapse on test (0.296 -> 0.010)
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
    regime_mult    = np.ones(n, dtype=np.float32)

    if gmm2 is not None and len(gmm2) == n:
        regime_mult[gmm2 == 0] = 2.0

    if timestamps is not None and len(timestamps) == n:
        import pandas as pd
        ts      = pd.to_datetime(timestamps)
        pre2020 = (ts < pd.Timestamp("2020-01-01")).to_numpy()
        regime_mult[pre2020] *= 1.5
        logger.info(f"Temporal reweighting: pre-2020={pre2020.sum():,} x1.5")

    final_weights     = torch.FloatTensor(sample_weights * regime_mult)
    actual_epoch_size = min(epoch_size, n)
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
    """Training loop with per-class metrics, signal-score checkpointing,
    and Google Drive mirroring."""

    GDRIVE_DIR = "/content/drive/MyDrive/Colab Notebooks"

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
            lr           = cfg.training.learning_rate,   # 1e-4 (was 2e-4)
            weight_decay = cfg.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        w = torch.FloatTensor(class_weights).to(device) if class_weights is not None else None
        self.criterion            = nn.CrossEntropyLoss(weight=w)
        self.confidence_criterion = nn.MSELoss()

        self.best_signal_score = 0.0   # FIX: checkpoint by signal quality, not val_loss
        self.best_val_loss     = float("inf")
        self.patience_counter  = 0
        self.start_epoch       = 1

    # ── Signal score (checkpoint criterion) ───────────────────────────────────

    @staticmethod
    def signal_score(per_class: dict) -> float:
        """Minority-class recall metric used for checkpointing and early stopping.

        Val loss minimum selects the most hold-dominant epoch — wrong criterion
        when 97.5% of labels are hold. This metric rewards the epoch where
        sell and buy recall peak together, which is where real signal lives.

        sell_R * 0.4 + buy_R * 0.4
        (val_acc is intentionally excluded — it's dominated by hold precision)
        """
        return (
            per_class["sell"]["recall"] * 0.4
            + per_class["buy"]["recall"]  * 0.4
        )

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def load_checkpoint(self, path: str) -> None:
        """Restore model + optimizer state; set start_epoch."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch   = ckpt.get("epoch", 0) + 1
        # Restore best signal score if present; fall back to 0 so first epoch
        # with any sell/buy recall will beat it and save immediately
        self.best_signal_score = ckpt.get("metrics", {}).get("signal_score", 0.0)
        self.best_val_loss     = ckpt.get("metrics", {}).get("loss", float("inf"))
        logger.info(
            f"Resumed from {path} | "
            f"start_epoch={self.start_epoch} | "
            f"best_signal_score={self.best_signal_score:.4f}"
        )

    def save_checkpoint(self, path: str, epoch: int, metrics: dict) -> None:
        """Save to local path and mirror to Google Drive."""
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

        # Mirror to Drive — survives Colab runtime disconnections
        gdrive = Path(self.GDRIVE_DIR)
        if gdrive.exists():
            dest = gdrive / Path(path).name
            shutil.copy2(path, dest)
            logger.info(f"Mirrored to Drive: {dest}")
        else:
            logger.warning(
                f"Drive not mounted at {self.GDRIVE_DIR} — local save only. "
                "Run: from google.colab import drive; drive.mount('/content/drive')"
            )

    # ── Batch helpers ─────────────────────────────────────────────────────────

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            return (batch[0].to(self.device),
                    batch[1].to(self.device),
                    batch[2].to(self.device))
        return batch[0].to(self.device), None, batch[-1].to(self.device)

    # ── Training epoch ────────────────────────────────────────────────────────

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

            # FIX: conf_loss weight raised 0.3 -> 0.5
            # At 0.3 the confidence head learned to output ~1.0 unconditionally
            # (always correct for hold 97.5% of the time). Raising to 0.5 forces
            # it to track actual per-sample correctness more closely.
            conf_loss = self.confidence_criterion(confidence, is_correct)
            loss = signal_loss + 0.5 * conf_loss

            loss.backward()
            # FIX: gradient_clip raised 2.0 -> 5.0 (in config.yaml)
            # GradNorm was exactly 2.0 every batch = clipped at ceiling.
            # LR also lowered 2e-4 -> 1e-4 to prevent repeated clipping.
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

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        """Returns overall metrics plus per-class precision/recall.

        Val acc ~0.98 is dominated by hold (97.5% of labels) and does not
        indicate signal quality. Per-class recall on sell/buy is the real
        indicator of whether the model is learning tradeable patterns.
        """
        self.model.eval()
        total_loss = 0.0
        total      = 0
        all_confs  = []
        n_classes  = 3
        conf_mat   = np.zeros((n_classes, n_classes), dtype=np.int64)

        for batch in loader:
            X, S, y = self._unpack_batch(batch)
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
            "loss":            total_loss / total,
            "accuracy":        int(np.diag(conf_mat).sum()) / total,
            "confidence_mean": confs.mean().item(),
            "confidence_std":  confs.std().item(),
            "per_class":       per_class,
        }

    # ── Early stopping ────────────────────────────────────────────────────────

    def check_early_stopping(self, score: float) -> bool:
        """Patience on signal_score (sell_R*0.4 + buy_R*0.4), not val_loss.

        Val loss minimum = most hold-dominant epoch = wrong checkpoint.
        Signal score peaks around ep9-10 where minority recall stabilises.
        """
        if score > self.best_signal_score:
            self.best_signal_score = score
            self.patience_counter  = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.cfg.training.early_stopping_patience


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

    # ── Join regime labels ────────────────────────────────────────────────────
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
    labels   = get_labeler(label_cfg).label(close_prices)
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

    def make_ds(f_start, f_end):
        s = sentiment[f_start:f_end] if sentiment is not None else None
        return SequenceDataset(features[f_start:f_end], labels[f_start:f_end], seq_len, s)

    train_ds = make_ds(0,              n_train + seq_len)
    val_ds   = make_ds(n_train,        n_train + n_val + seq_len)
    test_ds  = make_ds(n_train + n_val, len(features))

    # Labels aligned to sequence indices (for sampler and logging)
    y_train    = labels  [seq_len - 1 : seq_len - 1 + n_train]
    gmm2_train = gmm2_raw[seq_len - 1 : seq_len - 1 + n_train] if gmm2_raw is not None else None
    ts_train   = ts_raw  [seq_len - 1 : seq_len - 1 + n_train] if ts_raw   is not None else None

    logger.info(
        f"Dataset: {n_total:,} sequences | "
        f"classes={np.bincount(y_train, minlength=3)}"
    )
    if gmm2_train is not None:
        bull = gmm2_train.mean()
        logger.info(f"GMM2 train: Bull={bull:.1%}  Bear={1-bull:.1%}")

    # ── Regime-balanced sampler ───────────────────────────────────────────────
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
    batch_size  = cfg.training.batch_size
    num_workers = cfg.training.num_workers

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
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

        # Compute signal score FIRST — used in log, checkpoint, and early stopping
        score = Trainer.signal_score(pc)

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
            f"sell  P={pc['sell']['precision']:.3f} R={pc['sell']['recall']:.3f} "
            f"(n={pc['sell']['n']:,}) | "
            f"hold  P={pc['hold']['precision']:.3f} R={pc['hold']['recall']:.3f} "
            f"(n={pc['hold']['n']:,}) | "
            f"buy   P={pc['buy']['precision']:.3f}  R={pc['buy']['recall']:.3f}  "
            f"(n={pc['buy']['n']:,})"
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
                "val/confidence_mean": val_m["confidence_mean"],
                "val/signal_score":    score,
                "val/sell_precision":  pc["sell"]["precision"],
                "val/sell_recall":     pc["sell"]["recall"],
                "val/buy_precision":   pc["buy"]["precision"],
                "val/buy_recall":      pc["buy"]["recall"],
            })

        # Checkpoint when signal_score improves (not val_loss)
        if score > trainer.best_signal_score:
            val_m["signal_score"] = score   # persist in checkpoint metrics
            trainer.save_checkpoint(
                f"{cfg.paths.model_dir}/{cfg.model.name}_best.pt", epoch, val_m
            )

        if trainer.check_early_stopping(score):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # ── Test ──────────────────────────────────────────────────────────────────
    test_m = trainer.validate(test_loader)
    pc_t   = test_m["per_class"]
    test_score = Trainer.signal_score(pc_t)
    logger.info(
        f"Test: loss={test_m['loss']:.4f} acc={test_m['accuracy']:.4f} | "
        f"SignalScore={test_score:.4f} | "
        f"sell P={pc_t['sell']['precision']:.3f} R={pc_t['sell']['recall']:.3f} | "
        f"buy  P={pc_t['buy']['precision']:.3f}  R={pc_t['buy']['recall']:.3f}"
    )

    if wandb_run:
        import wandb
        wandb.log({
            "test/loss":           test_m["loss"],
            "test/accuracy":       test_m["accuracy"],
            "test/signal_score":   test_score,
            "test/sell_precision": pc_t["sell"]["precision"],
            "test/sell_recall":    pc_t["sell"]["recall"],
            "test/buy_precision":  pc_t["buy"]["precision"],
            "test/buy_recall":     pc_t["buy"]["recall"],
        })
        wandb_run.finish()


if __name__ == "__main__":
    main()
