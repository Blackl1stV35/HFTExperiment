"""GPU-pinned sequence dataset — Phase 4 I/O optimisation.

Problem:
    SequenceDataset.__getitem__ slices CPU numpy arrays per sample:
        features[i:i+240]  →  copy  →  pin_memory  →  PCIe  →  GPU
    At batch=4096, seq_len=240, dim=10: 37.5 MB per batch over PCIe (32 GB/s)
    = ~1.2 ms transfer + worker overhead = ~500 ms stall = 30% GPU idle.

Solution — GPUPinnedSequenceDataset:
    Transfer features (215 MB) and labels (45 MB) to GPU HBM2e ONCE at init.
    __getitem__ calls features_gpu[i:i+seq_len] — GPU internal memory at 2 TB/s.
    PCIe transfer eliminated entirely. DataLoader stall drops from ~500 ms → ~1 ms.
    Expected GPU utilisation: 40% → 85-90%.

Memory budget (40 GB A100):
    features  (N=5.68M, 10, f32): 0.21 GB
    labels    (N=5.68M, i64):     0.04 GB
    gmm2/vol  (N=5.68M, f32):     0.04 GB
    ────────────────────────────────────
    Total pinned:                  0.29 GB  (trivial vs 40 GB HBM2e)

Usage:
    ds = GPUPinnedSequenceDataset(features, labels, seq_len=240, device=device)
    loader = DataLoader(ds, batch_size=4096, num_workers=0)  # workers=0 correct
    # — no CPU work remains to parallelise; workers add overhead only

Compatibility:
    Drop-in replacement for SequenceDataset — same __getitem__ return signature:
        (x, y)         if sentiment is None
        (x, s, y)      if sentiment is provided
    x shape: (seq_len, feature_dim)   — on device
    y shape: ()                        — on device (long)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class GPUPinnedSequenceDataset(Dataset):
    """Sliding-window dataset with all data pinned in GPU memory.

    All sequence slicing happens on the GPU via integer indexing of
    a pre-transferred CUDA tensor. Zero CPU work, zero PCIe traffic
    per batch after __init__.

    Args:
        features:  (N, F) float32 array — raw features (NOT seq-expanded)
        labels:    (N,)   int64   array — per-bar labels
        seq_len:   int — sliding window length (e.g. 240)
        device:    torch.device — target GPU
        sentiment: (N, E) float32 | None — optional sentiment embeddings
        dtype:     torch dtype for features (default bfloat16 on A100 for
                   halved HBM usage; pass torch.float32 to keep precision)

    __getitem__(i) returns:
        x: (seq_len, F) tensor on device — features[i : i+seq_len]
        y: ()           tensor on device — labels[i+seq_len-1]
        s: (E,)         tensor on device — sentiment[i+seq_len-1]  (if set)
    """

    def __init__(
        self,
        features:  np.ndarray,
        labels:    np.ndarray,
        seq_len:   int,
        device:    torch.device,
        sentiment: np.ndarray | None = None,
        dtype:     torch.dtype = torch.float32,
    ):
        assert features.ndim == 2, f"features must be (N, F), got {features.shape}"
        assert len(features) == len(labels), "features/labels length mismatch"
        assert len(features) > seq_len, "dataset shorter than seq_len"

        self.seq_len = seq_len
        self.device  = device
        self._n      = len(features) - seq_len

        # Transfer to GPU once — stays resident in HBM2e for all epochs
        self._features = torch.as_tensor(
            features, dtype=dtype
        ).to(device, non_blocking=True)

        self._labels = torch.as_tensor(
            labels, dtype=torch.long
        ).to(device, non_blocking=True)

        self._sentiment = None
        if sentiment is not None:
            self._sentiment = torch.as_tensor(
                sentiment, dtype=dtype
            ).to(device, non_blocking=True)

        mb = (self._features.nbytes + self._labels.nbytes) / 1024**2
        if self._sentiment is not None:
            mb += self._sentiment.nbytes / 1024**2

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int):
        # All ops on GPU — no CPU involvement, no PCIe transfer
        x = self._features[i : i + self.seq_len]          # (seq_len, F)
        y = self._labels[i + self.seq_len - 1]            # scalar
        if self._sentiment is not None:
            s = self._sentiment[i + self.seq_len - 1]     # (E,)
            return x, s, y
        return x, y


class GPUPinnedIndexedDataset(Dataset):
    """Index-masked view of a GPUPinnedSequenceDataset.

    Equivalent to IndexedDataset but avoids going back to CPU for
    index remapping — the base dataset is already on GPU.

    Args:
        base:    GPUPinnedSequenceDataset
        indices: (M,) numpy int64 — subset indices into base
    """

    def __init__(self, base: GPUPinnedSequenceDataset, indices: np.ndarray):
        self.base    = base
        # Keep indices on CPU — they're used for Python-level __getitem__
        # which is trivial (one integer lookup, no data movement)
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[int(self.indices[i])]