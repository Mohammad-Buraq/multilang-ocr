# src/model.py

import torch
import torch.nn as nn

from .alphabet import ALPHABET, BLANK_INDEX

NUM_CLASSES = len(ALPHABET) + 1  # +1 for CTC blank


class CRNN(nn.Module):
    """
    Compact CRNN for line-level OCR, suitable for ONNX export and
    browser inference via onnxruntime-web.

    Input shape:  (B, 1, H=32, W<=~256)
    Output shape: (T, B, C)  where:
        T ~ W/4  (time steps),
        C = NUM_CLASSES
    """

    def __init__(self, img_h: int = 32, hidden_size: int = 128):
        super().__init__()
        self.img_h = img_h

        # CNN feature extractor
        self.cnn = nn.Sequential(
            # (B, 1, H, W) -> (B, 32, H, W)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 32, H/2, W/2)

            # (B, 32, H/2, W/2) -> (B, 64, H/2, W/2)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, H/4, W/4)

            # (B, 64, H/4, W/4) -> (B, 128, H/4, W/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # (B, 128, H/4, W/4) -> (B, 128, H/8, W/4)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # downsample height only
        )

        # After CNN: (B, 128, H', W') with H' ~= img_h / 8
        # Collapse H' by averaging -> (B, 128, W')

        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,  # expects (T, B, F)
        )

        self.fc = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, H, W)
        Returns:
            logits: (T, B, C) time-major for CTC
        """
        feats = self.cnn(x)  # (B, 128, H', W')
        b, c, h, w = feats.size()

        # Collapse height: average over H'
        feats = feats.mean(dim=2)  # (B, 128, W')

        # Prepare for RNN: (T=W', B, C=128)
        feats = feats.permute(2, 0, 1)  # (T, B, C)

        rnn_out, _ = self.rnn(feats)  # (T, B, 2*hidden)
        logits = self.fc(rnn_out)     # (T, B, NUM_CLASSES)

        return logits
