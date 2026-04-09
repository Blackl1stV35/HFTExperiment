"""Configuration utilities and helpers."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf


def load_env() -> None:
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)


def get_device(preference: str = "auto") -> torch.device:
    """Resolve device preference to actual torch device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent


@dataclass
class BrokerConfig:
    """Broker connection configuration loaded from environment."""

    login: Optional[str] = None
    password: Optional[str] = None
    server: Optional[str] = None
    path: Optional[str] = None

    @classmethod
    def from_env(cls) -> "BrokerConfig":
        load_env()
        return cls(
            login=os.getenv("MT5_LOGIN"),
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
            path=os.getenv("MT5_PATH"),
        )


def print_config(cfg: DictConfig) -> None:
    """Pretty-print a Hydra config."""
    print(OmegaConf.to_yaml(cfg, resolve=True))


# Re-export for convenience
from src.utils.logger import setup_logger  # noqa: E402, F401
