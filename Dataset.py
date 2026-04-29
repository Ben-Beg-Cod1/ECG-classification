"""
dataset.py
----------
PyTorch Dataset klassen voor MIT-BIH en CinC 2017.

Gebruik:
    from data.dataset import MITBIHDataset, CinC2017Dataset, get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders("mitbih")
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional

PROCESSED_DIR = Path(__file__).parent / "processed"

# ----------------------------------------------------------------- labels ---
MITBIH_CLASSES = ["Normal (N)", "Supraventriculair (S)", "Ventriculair (V)", "Fusie (F)", "Onbekend (Q)"]
CINC2017_CLASSES = ["Normal (N)", "Atrial Fibrillation (A)", "Other rhythm (O)", "Noisy (~)"]


# ====================================================== Dataset klassen ======

class ECGDataset(Dataset):
    """Generieke ECG dataset die .npy bestanden inlaadt."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
    ):
        """
        Args:
            X: Signalen van vorm (N, lengte, 1)
            y: Labels van vorm (N,)
            augment: Schakel simpele data augmentatie in (alleen voor training)
        """
        self.X = torch.from_numpy(X).float()       # (N, L, 1)
        self.X = self.X.permute(0, 2, 1)           # → (N, 1, L)  voor Conv1d
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()

        if self.augment:
            x = self._random_noise(x)
            x = self._random_scale(x)

        return x, self.y[idx]

    # ---------------------------------------------------- augmentaties ------

    @staticmethod
    def _random_noise(x: torch.Tensor, std: float = 0.02) -> torch.Tensor:
        """Voeg Gaussische ruis toe."""
        return x + torch.randn_like(x) * std

    @staticmethod
    def _random_scale(x: torch.Tensor, low: float = 0.9, high: float = 1.1) -> torch.Tensor:
        """Schaal het signaal willekeurig."""
        factor = torch.empty(1).uniform_(low, high)
        return x * factor


class MITBIHDataset(ECGDataset):
    """MIT-BIH beat-segmenten (360 samples, 5 klassen)."""

    CLASSES = MITBIH_CLASSES
    NUM_CLASSES = 5
    SIGNAL_LENGTH = 360

    @classmethod
    def load(cls, split: str, augment: bool = False) -> "MITBIHDataset":
        return cls(*_load_npy("mitbih", split), augment=augment)


class CinC2017Dataset(ECGDataset):
    """PhysioNet CinC 2017 opnames (9000 samples, 4 klassen)."""

    CLASSES = CINC2017_CLASSES
    NUM_CLASSES = 4
    SIGNAL_LENGTH = 9000

    @classmethod
    def load(cls, split: str, augment: bool = False) -> "CinC2017Dataset":
        return cls(*_load_npy("cinc2017", split), augment=augment)


# ============================================================= helpers =======

def _load_npy(dataset: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Laadt X en y .npy bestanden voor een gegeven dataset en split."""
    base = PROCESSED_DIR / dataset
    X_path = base / f"X_{split}.npy"
    y_path = base / f"y_{split}.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Verwerkte data niet gevonden in {base}. "
            f"Voer eerst 'python data/preprocess.py --dataset {dataset}' uit."
        )

    X = np.load(X_path)
    y = np.load(y_path)
    return X, y


def get_dataloaders(
    dataset: str = "cinc2017",
    batch_size: int = 64,
    num_workers: int = 2,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Geeft train, val en test DataLoaders terug.

    Args:
        dataset:       'mitbih' of 'cinc2017'
        batch_size:    aantal samples per batch
        num_workers:   parallelle workers voor data laden
        augment_train: augmentatie aan voor trainingset

    Returns:
        (train_loader, val_loader, test_loader)

    Voorbeeld:
        train_loader, val_loader, test_loader = get_dataloaders("cinc2017")
        for X_batch, y_batch in train_loader:
            # X_batch: (batch, 1, 9000)
            # y_batch: (batch,)
            ...
    """
    dataset_cls = {"mitbih": MITBIHDataset, "cinc2017": CinC2017Dataset}.get(dataset)
    if dataset_cls is None:
        raise ValueError(f"Onbekende dataset '{dataset}'. Kies 'mitbih' of 'cinc2017'.")

    train_ds = dataset_cls.load("train", augment=augment_train)
    val_ds   = dataset_cls.load("val",   augment=False)
    test_ds  = dataset_cls.load("test",  augment=False)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"[{dataset}] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    print(f"  Signaalvorm per sample: (1, {dataset_cls.SIGNAL_LENGTH})")
    print(f"  Klassen: {dataset_cls.CLASSES}")

    return train_loader, val_loader, test_loader
