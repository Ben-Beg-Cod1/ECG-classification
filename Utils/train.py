"""
utils/train.py
--------------
Herbruikbare trainingsloop voor alle modellen in dit project.

Bevat:
  - Trainer klasse met train/eval loop
  - Early stopping
  - Model checkpoint opslaan

Gebruik:
    from utils.train import Trainer
    trainer = Trainer(model, train_loader, val_loader, num_classes=4)
    history = trainer.fit(epochs=30)
"""

import time
import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


class EarlyStopping:
    """Stop training als val_loss niet verbetert na `patience` epochs."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Traint een PyTorch model en slaat het beste checkpoint op.

    Args:
        model:          PyTorch model (nn.Module)
        train_loader:   DataLoader voor trainingdata
        val_loader:     DataLoader voor validatiedata
        num_classes:    aantal klassen (voor gewogen loss bij klasse-onbalans)
        lr:             leersnelheid (default 1e-3)
        weight_decay:   L2 regularisatie (default 1e-4)
        patience:       early stopping geduld in epochs (default 10)
        device:         'cuda', 'mps' of 'cpu' (auto-detectie als None)
        checkpoint_name naam voor het opgeslagen model bestand
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        device: Optional[str] = None,
        checkpoint_name: str = "best_model",
    ):
        self.device = device or self._auto_device()
        print(f"[Trainer] Gebruik device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=patience)

    # --------------------------------------------------------- publieke API --

    def fit(self, epochs: int = 50) -> dict:
        """
        Train het model voor maximaal `epochs` epochs.

        Returns:
            history dict met lijsten: train_loss, val_loss, train_acc, val_acc
        """
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_weights = copy.deepcopy(self.model.state_dict())
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch(train=True)
            val_loss, val_acc     = self._run_epoch(train=False)

            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"loss {train_loss:.4f}/{val_loss:.4f} | "
                f"acc {train_acc:.3f}/{val_acc:.3f} | "
                f"{elapsed:.1f}s"
            )

            # Sla beste model op
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                torch.save(best_weights, self.checkpoint_path)

            # Early stopping check
            if self.early_stopping.step(val_loss):
                print(f"[Early stopping] Gestopt na epoch {epoch}.")
                break

        # Laad beste gewichten terug
        self.model.load_state_dict(best_weights)
        print(f"[Trainer] Beste model opgeslagen in {self.checkpoint_path}")
        return history

    def evaluate(self, loader: DataLoader) -> tuple:
        """Evalueer het model op een gegeven loader. Geeft (loss, accuracy) terug."""
        return self._run_epoch(train=False, loader=loader)

    # --------------------------------------------------------- interne logic -

    def _run_epoch(self, train: bool, loader: Optional[DataLoader] = None):
        if loader is None:
            loader = self.train_loader if train else self.val_loader

        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        return total_loss / total, correct / total

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
