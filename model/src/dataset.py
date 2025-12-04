# src/dataset.py

import random
import string
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset

from .alphabet import text_to_indices


def random_english_text(min_len=5, max_len=25) -> str:
    """Generate a random English-like string."""
    chars = string.ascii_letters + string.digits + "     "  # letters, digits, spaces
    length = random.randint(min_len, max_len)
    return "".join(random.choice(chars) for _ in range(length)).strip()


def render_text_line(text: str, height: int = 32, font_path: str | None = None) -> Image.Image:
    """Render a text line to a grayscale PIL image of fixed height."""
    if font_path is not None:
        font = ImageFont.truetype(font_path, size=32)
    else:
        font = ImageFont.load_default()

    # Rough big temp canvas, we will crop
    temp_w = max(100, len(text) * 20)
    temp_h = 64
    img = Image.new("L", (temp_w, temp_h), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, font=font, fill=0)

    bbox = img.getbbox()
    if bbox is not None:
        img = img.crop(bbox)

    w, h = img.size
    if h == 0 or w == 0:
        img = Image.new("L", (100, height), color=255)
    else:
        new_w = int(w * (height / float(h)))
        img = img.resize((max(new_w, 10), height), Image.BILINEAR)

    return img


class SyntheticEnglishDataset(Dataset):
    def __init__(self, num_samples: int = 10000, img_height: int = 32, font_path: str | None = None):
        self.num_samples = num_samples
        self.img_height = img_height
        self.font_path = font_path

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = random_english_text()
        img = render_text_line(text, height=self.img_height, font_path=self.font_path)

        # To numpy [0,1]
        img_np = np.array(img).astype("float32") / 255.0
        h, w = img_np.shape
        img_np = img_np[None, :, :]  # (1, H, W)

        image = torch.from_numpy(img_np)  # (1, H, W)
        label_indices = text_to_indices(text)
        label = torch.tensor(label_indices, dtype=torch.long)

        sample = {
            "image": image,
            "label": label,
            "text": text,
            "width": w,
        }
        return sample


def collate_fn(batch: list[dict]):
    """
    Collate function for variable-width images and labels (for CTC).
    Pads images along width.
    """
    images = [b["image"] for b in batch]
    labels = [b["label"] for b in batch]

    heights = [img.shape[1] for img in images]
    widths = [img.shape[2] for img in images]
    max_w = max(widths)
    h = heights[0]

    padded = []
    for img in images:
        c, h_i, w_i = img.shape
        pad_w = max_w - w_i
        if pad_w > 0:
            pad = torch.ones((c, h_i, pad_w), dtype=img.dtype)  # white (1.0)
            img_p = torch.cat([img, pad], dim=2)
        else:
            img_p = img
        padded.append(img_p)

    images_tensor = torch.stack(padded, dim=0)  # (B, 1, H, max_W)

    flat_labels = torch.cat(labels, dim=0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    # Approx input lengths for CTC (after CNN width downsample by ~4)
    input_lengths = torch.tensor([max_w // 4 for _ in batch], dtype=torch.long)

    return images_tensor, flat_labels, input_lengths, label_lengths
