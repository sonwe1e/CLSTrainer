from __future__ import annotations


def pair_collate(batch):
    import torch

    return {
        "images": torch.stack([sample["images"] for sample in batch]),
        "labels": torch.tensor([sample["label"] for sample in batch], dtype=torch.long),
        "meta": [sample["meta"] for sample in batch],
    }

