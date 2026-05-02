"""
models/cnn_baseline.py
----------------------
Eenvoudige 1D CNN voor ECG classificatie.

Architectuur:
    Input (1, L)
    → Conv blok x3  (Conv1d → BatchNorm → ReLU → MaxPool)
    → Global Average Pooling
    → Dropout → Linear → softmax

Gebruik:
    from models.cnn_baseline import CNN1D
    model = CNN1D(num_classes=4, signal_length=9000)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d + BatchNorm + ReLU + MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, pool_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1D(nn.Module):
    """
    Baseline 1D CNN voor ECG classificatie.

    Args:
        num_classes:    aantal uitvoerklassen (4 voor CinC2017, 5 voor MIT-BIH)
        signal_length:  lengte van het invoersignaal in samples
        dropout:        dropout kans voor regularisatie
    """

    def __init__(self, num_classes: int = 4, signal_length: int = 9000, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,   32, kernel_size=7,  pool_size=3),   # (1, L) → (32, L/3)
            ConvBlock(32,  64, kernel_size=5,  pool_size=3),   # → (64, L/9)
            ConvBlock(64, 128, kernel_size=3,  pool_size=3),   # → (128, L/27)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)         # → (128, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),                                      # → (128,)
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor van vorm (batch, 1, signal_length)
        Returns:
            logits van vorm (batch, num_classes)
        """
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x


# ----------------------------------------------------------------- quick test
if __name__ == "__main__":
    model = CNN1D(num_classes=4, signal_length=9000)
    dummy = torch.randn(8, 1, 9000)
    out = model(dummy)
    print("Output vorm:", out.shape)  # verwacht: (8, 4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Totaal parameters: {total_params:,}")
