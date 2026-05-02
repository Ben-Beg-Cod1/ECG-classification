"""
utils/metrics.py
----------------
Evaluatiemetrics voor ECG classificatie.

Functies:
  - evaluate_model()    → accuracy, F1, rapport per klasse
  - plot_confusion_matrix()
  - plot_training_history()
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    class_names: list,
    device: Optional[str] = None,
) -> dict:
    """
    Evalueer model op een DataLoader.

    Returns dict met:
        accuracy, f1_macro, f1_weighted, report (str), y_true, y_pred
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = (y_true == y_pred).mean()
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, target_names=class_names)

    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"F1 macro:    {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")
    print(f"\nKlassificatierapport:\n{report}")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
):
    """Plot en sla optioneel een confusion matrix op."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix opgeslagen: {save_path}")
    plt.show()


def plot_training_history(
    history: dict,
    title: str = "Trainingsverloop",
    save_path: Optional[str] = None,
):
    """
    Plot loss en accuracy over epochs.

    Args:
        history: dict met lijsten 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"],   label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train accuracy")
    ax2.plot(epochs, history["val_acc"],   label="Val accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Trainingsverloop opgeslagen: {save_path}")
    plt.show()
