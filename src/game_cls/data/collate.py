from __future__ import annotations


def pair_collate(batch):
    import torch

    result = {
        "images": torch.stack([sample["images"] for sample in batch]),
        "labels": torch.tensor([sample["label"] for sample in batch], dtype=torch.long),
        "meta": [sample["meta"] for sample in batch],
    }
    for key in ("game_id", "game_label_id", "video_group_id"):
        if key in batch[0]:
            result[key] = torch.tensor(
                [sample[key] for sample in batch], dtype=torch.int64
            )
    return result
