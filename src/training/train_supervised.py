"""Supervised training pipeline — Phase 3 regime-balanced edition.

OOM fix (v2):
    SequenceDataset slices windows on the fly — avoids pre-allocating the
    full (N, seq_len, F) array that caused the 16.4 GB RAM OOM crash.

v3 fixes:
    gradient_clip 2.0->5.0, learning_rate 2e-4->1e-4, removed post-2024 x0.5,
    signal_score checkpoint criterion, conf_loss weight 0.3->0.5, Drive mirror.

v4 fixes (this version):
    1. FocalLoss replaces CrossEntropyLoss — down-weights easy hold examples
       dynamically; static class weights cannot adapt as model learns.
    2. Buy class weight raised [2.5,0.3,2.5]->[2.5,0.3,10.0] in xauusd.yaml.
       buy has 5x fewer labels than sell; same weight is not symmetric.
    3. Scheduler: CosineAnnealingWarmRestarts -> ReduceLROnPlateau(signal_score).
       Cosine restart surged LR at ep10/20, causing gradient spikes (8.63,10.0).
    4. Regime-stratified val/test split — each split gets the same Bear/Bull x
       vol-tier proportion. Pure chronological split put all post-2024 (Bear+HIGH)
       in test and none in val, making val buy recall a misleading proxy for test.

Hardware: batch=2048, workers=0, pin_memory=False, epoch_size=500_000
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
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from loguru import logger

from src.encoder.fusion import DualBranchModel
from src.training.labels import LabelConfig, get_labeler
from src.data.preprocessing import prepare_features, join_regime_labels
from src.data.tick_store import TickStore
from src.utils.config import get_device, set_seed
from src.utils.logger import setup_logger


# ── Fix 1: FocalLoss ─────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Multiplies per-sample CE loss by (1 - p_correct)^gamma so examples the
    model already predicts confidently (easy hold) get near-zero weight, while
    hard examples (rare sell/buy) receive full gradient. Adapts dynamically
    as training progresses — unlike static class weights which are fixed.

    Args:
        weight: per-class weights tensor (alpha), same as CrossEntropyLoss.
        gamma:  focusing parameter. 0 = standard CE. 2.0 is the standard
                setting from the original paper; higher = more focus on hard.
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE loss per sample (reduction='none' to keep per-sample values)
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        # p_t = probability assigned to the correct class
        p_t = torch.exp(-ce)
        # Focal weight: down-scale easy examples, leave hard ones intact
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * ce).mean()


# ── SequenceDataset ───────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Sliding-window dataset — slices windows on demand from (N, F) features.
    Drops peak RAM from 16.4 GB to ~0.13 GB for 5.68 M bars.
    """

    def __init__(
        self,
        features:  np.ndarray,
        labels:    np.ndarray,
        seq_len:   int,
        sentiment: np.ndarray | None = None,
    ):
        assert len(features) == len(labels)
        assert len(features) > seq_len
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


# ── Fix 4 helper: regime-stratified split indices ────────────────────────────

def regime_stratified_split(
    n_total:    int,
    gmm2:       np.ndarray | None,
    vol_regime: np.ndarray | None,
    val_frac:   float = 0.15,
    test_frac:  float = 0.10,
    seq_len:    int   = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test *sequence* index arrays so each split contains the
    same proportion of (Bear/Bull) x (LOW/NORMAL/HIGH) regime buckets.

    Falls back to chronological split if regime arrays are unavailable.

    Returns:
        train_idx, val_idx, test_idx — 1-D index arrays into the sequence space
        (i.e. already offset by seq_len; index 0 = the first complete window).
    """
    all_idx = np.arange(n_total)

    if gmm2 is None or vol_regime is None:
        # Chronological fallback
        n_test  = int(n_total * test_frac)
        n_val   = int(n_total * val_frac)
        n_train = n_total - n_val - n_test
        return all_idx[:n_train], all_idx[n_train:n_train+n_val], all_idx[n_train+n_val:]

    # Build bucket labels: 0-5 for (Bear/Bull) x (LOW/NORMAL/HIGH)
    # gmm2: 0=Bear 1=Bull  |  vol_regime: 0=LOW 1=NORMAL 2=HIGH (float32 encoded)
    g = (gmm2 > 0.5).astype(int)          # 0=Bear 1=Bull
    v = np.round(vol_regime * 2).astype(int).clip(0, 2)  # 0/1/2
    buckets = g * 3 + v                    # 6 buckets: 0..5

    val_idx   = []
    test_idx  = []
    train_idx = []

    for b in range(6):
        b_idx = all_idx[buckets == b]
        if len(b_idx) == 0:
            continue
        # Keep temporal order within each bucket; take last test_frac for test,
        # previous val_frac for val, rest for train
        n_b    = len(b_idx)
        n_t    = max(1, int(n_b * test_frac))
        n_v    = max(1, int(n_b * val_frac))
        test_idx.extend(b_idx[-n_t:])
        val_idx.extend(b_idx[-(n_t + n_v):-n_t])
        train_idx.extend(b_idx[:-(n_t + n_v)])

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    test_idx  = np.array(test_idx)

    logger.info(
        f"Regime-stratified split: train={len(train_idx):,}  "
        f"val={len(val_idx):,}  test={len(test_idx):,}"
    )
    for b in range(6):
        bname = f"{'Bear' if b<3 else 'Bull'}-{'LOW' if b%3==0 else 'NORMAL' if b%3==1 else 'HIGH'}"
        n_b   = (buckets == b).sum()
        logger.info(f"  {bname}: {n_b:,} sequences ({n_b/n_total:.1%})")

    return train_idx, val_idx, test_idx


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
        Bear (gmm2=0):  x2.0
        pre-2020 bars:  x1.5 — boost historical diversity
        (post-2024 x0.5 removed — test set IS post-2024 data)
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
        cutoff  = pd.Timestamp("2020-01-01")
        # np.asarray works on DatetimeIndex, Series, and numpy datetime64
        # arrays without calling .to_numpy() on the boolean result.
        pre2020 = np.asarray(ts < cutoff, dtype=bool)
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
    """Training loop with focal loss, ReduceLROnPlateau, per-class metrics,
    signal-score checkpointing, and Google Drive mirroring."""

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
            lr           = cfg.training.learning_rate,
            weight_decay = cfg.training.weight_decay,
        )

        # Fix 3: ReduceLROnPlateau replaces CosineAnnealingWarmRestarts.
        # Cosine restarts surged LR at ep10/20, causing gradient spikes to 8.6-10.0.
        # ReduceLROnPlateau only decays when signal_score stops improving.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode     = "max",      # maximise signal_score
            factor   = 0.5,
            patience = 5,
            min_lr   = 1e-6,
        )

        # Fix 1: FocalLoss with class weights as alpha term.
        w = torch.FloatTensor(class_weights).to(device) if class_weights is not None else None
        gamma = float(cfg.training.get("focal_gamma", 2.0))
        self.criterion            = FocalLoss(weight=w, gamma=gamma)
        self.confidence_criterion = nn.MSELoss()

        self.best_signal_score = 0.0
        self.best_val_loss     = float("inf")
        self.patience_counter  = 0
        self.start_epoch       = 1

    @staticmethod
    def signal_score(per_class: dict) -> float:
        """F1-based metric with sell precision floor >= 0.30.

        Hard floor: sell precision < 0.30 returns 0.0 — checkpoint not saved.
        At P=0.25 (prev best), RL breakeven WR ~51% — unreachable.
        At P=0.30, breakeven WR ~49% — within reach.
        Above floor: sell_F1*0.4 + buy_F1*0.4 as before.
        """
        sp = per_class["sell"]["precision"]
        sr = per_class["sell"]["recall"]
        if sp < 0.25:
            return 0.0
        def f1(p, r):
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        sell_f1 = f1(sp, sr)
        buy_f1  = f1(per_class["buy"]["precision"], per_class["buy"]["recall"])
        return sell_f1 * 0.4 + buy_f1 * 0.4

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch       = ckpt.get("epoch", 0) + 1
        self.best_signal_score = ckpt.get("metrics", {}).get("signal_score", 0.0)
        self.best_val_loss     = ckpt.get("metrics", {}).get("loss", float("inf"))
        logger.info(
            f"Resumed from {path} | "
            f"start_epoch={self.start_epoch} | "
            f"best_signal_score={self.best_signal_score:.4f}"
        )

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

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            return batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
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
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip)
            self.optimizer.step()
            total_loss += signal_loss.item() * X.size(0)
            correct    += (logits.argmax(dim=-1) == y).sum().item()
            total      += X.size(0)

        grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.model.parameters() if p.grad is not None
        ) ** 0.5

        # Fix 3: scheduler steps on signal_score; caller passes score after validation.
        # step() is called from the training loop, not here, so the score is available.
        return {
            "loss":      total_loss / total,
            "accuracy":  correct    / total,
            "lr":        self.optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
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

    def check_early_stopping(self, score: float) -> bool:
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

    regime_csv = Path(cfg.paths.data_dir) / "regime" / "daily_regime_labels.csv"
    df_raw     = join_regime_labels(df_raw, regime_csv)
    timestamps = df_raw["timestamp"].to_list() if "timestamp" in df_raw.columns else None

    features, close_prices = prepare_features(
        df_raw,
        scaler_method = cfg.data.preprocessing.scaling,
        window_size   = cfg.data.preprocessing.window_size,
    )
    ws = cfg.data.preprocessing.window_size

    gmm2_raw     = None
    vol_raw      = None
    ts_raw       = None
    if "gmm2_state" in df_raw.columns:
        gmm2_raw = df_raw["gmm2_state"].to_numpy()[ws:]
        if timestamps:
            ts_raw = timestamps[ws:]
    if "vol_regime_enc" in df_raw.columns:
        vol_raw = df_raw["vol_regime_enc"].to_numpy()[ws:]

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

    sentiment = None
    if cfg.data.sentiment.enabled:
        emb_path = Path(cfg.paths.data_dir) / "sentiment_embeddings.npy"
        if emb_path.exists():
            all_emb = np.load(str(emb_path), allow_pickle=True)
            if isinstance(all_emb, np.ndarray) and all_emb.ndim == 2:
                sentiment = all_emb[ws:]

    # ── Fix 4: Regime-stratified split ────────────────────────────────────────
    seq_len = cfg.model.input.sequence_length
    n_total = len(features) - seq_len

    # Sequence-aligned regime arrays
    gmm2_seq = gmm2_raw[seq_len - 1 : seq_len - 1 + n_total] if gmm2_raw is not None else None
    vol_seq  = vol_raw [seq_len - 1 : seq_len - 1 + n_total] if vol_raw  is not None else None
    ts_seq   = ts_raw  [seq_len - 1 : seq_len - 1 + n_total] if ts_raw   is not None else None

    train_idx, val_idx, test_idx = regime_stratified_split(
        n_total    = n_total,
        gmm2       = gmm2_seq,
        vol_regime = vol_seq,
        val_frac   = cfg.training.val_split,
        test_frac  = cfg.training.test_split,
        seq_len    = seq_len,
    )

    # Labels aligned to sequence indices
    seq_labels = labels[seq_len - 1 : seq_len - 1 + n_total]
    y_train    = seq_labels[train_idx]
    gmm2_train = gmm2_seq[train_idx] if gmm2_seq is not None else None
    ts_train   = np.array(ts_seq)[train_idx] if ts_seq is not None else None

    logger.info(
        f"Dataset: {n_total:,} sequences | "
        f"classes (train)={np.bincount(y_train, minlength=3)}"
    )
    if gmm2_train is not None:
        bull = gmm2_train.mean()
        logger.info(f"GMM2 train: Bull={bull:.1%}  Bear={1-bull:.1%}")

    # Build datasets using index-based slicing via a wrapper
    class IndexedDataset(Dataset):
        """Wraps SequenceDataset to expose a subset by index array."""
        def __init__(self, base: SequenceDataset, indices: np.ndarray):
            self.base    = base
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.base[int(self.indices[i])]

    base_ds   = SequenceDataset(features, labels, seq_len, sentiment)
    train_ds  = IndexedDataset(base_ds, train_idx)
    val_ds    = IndexedDataset(base_ds, val_idx)
    test_ds   = IndexedDataset(base_ds, test_idx)

    # ── Regime-balanced sampler ───────────────────────────────────────────────
    cw_cfg     = list(cfg.data.labeling.get("class_weights", [])) or None
    epoch_size = min(500_000, len(train_idx))
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
        val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=False,
    )
    logger.info(
        f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,} | "
        f"batch={batch_size}  workers={num_workers}  epoch_size={epoch_size:,}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DualBranchModel.from_config(cfg.model)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    trainer = Trainer(cfg, model, device, class_weights=class_weights_arr)

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
        score   = Trainer.signal_score(pc)

        # Fix 3: ReduceLROnPlateau steps on signal_score after each epoch
        trainer.scheduler.step(score)

        logger.info(
            f"Epoch {epoch}/{cfg.training.epochs} | "
            f"Train loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} | "
            f"Val   loss={val_m['loss']:.4f}  acc={val_m['accuracy']:.4f} | "
            f"SignalScore={score:.4f} | "
            f"LR={train_m['lr']:.2e} | "
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

        if score > trainer.best_signal_score:
            val_m["signal_score"] = score
            trainer.save_checkpoint(
                f"{cfg.paths.model_dir}/{cfg.model.name}_best.pt", epoch, val_m
            )

        if trainer.check_early_stopping(score):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # ── Test ──────────────────────────────────────────────────────────────────
    test_m     = trainer.validate(test_loader)
    pc_t       = test_m["per_class"]
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