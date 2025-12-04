# model/src/test_model.py

import torch

from .model import CRNN, NUM_CLASSES
from .alphabet import ALPHABET, BLANK_INDEX


def main():
    print("Alphabet length:", len(ALPHABET))
    print("Blank index:", BLANK_INDEX)
    print("Num classes (including blank):", NUM_CLASSES)

    model = CRNN(img_h=32, hidden_size=128)
    model.eval()

    # Dummy batch: 2 lines sized 32x128
    dummy = torch.randn(2, 1, 32, 128)  # (B, C, H, W)

    with torch.no_grad():
        logits = model(dummy)  # (T, B, C)

    print("Logits shape:", logits.shape)  # e.g. (T, 2, NUM_CLASSES)
    T, B, C = logits.shape
    print(f"T (time steps): {T}")
    print(f"B (batch size): {B}")
    print(f"C (classes): {C}")


if __name__ == "__main__":
    main()
