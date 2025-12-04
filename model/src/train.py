# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import CRNN, NUM_CLASSES
from .dataset import SyntheticEnglishDataset, collate_fn


def train_toy(
    num_epochs: int = 2,
    batch_size: int = 32,
    num_samples: int = 3000,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = SyntheticEnglishDataset(num_samples=num_samples, img_height=32)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = CRNN(img_h=32, hidden_size=128).to(device)
    ctc_loss = nn.CTCLoss(blank=NUM_CLASSES - 1, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for i, (images, flat_labels, input_lengths, label_lengths) in enumerate(loader):
            images = images.to(device)
            flat_labels = flat_labels.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(images)           # (T, B, C)
            log_probs = logits.log_softmax(2)

            loss = ctc_loss(log_probs, flat_labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}, step {i+1}/{len(loader)}, "
                    f"loss = {loss.item():.4f}"
                )

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1}] avg loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), "crnn_toy_english.pth")
    print("Saved model to crnn_toy_english.pth")


if __name__ == "__main__":
    train_toy()
