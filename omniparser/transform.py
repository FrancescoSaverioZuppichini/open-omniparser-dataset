from typing import Union

import torch
from torchvision.transforms import v2


def get_transforms(size: Union[int, int]):
    train_transform = v2.Compose(
        [
            v2.ToImage(),
            # v2.RandomCrop(size=(720, 1152)),  # ~10% crop from 1280x800
            v2.RandomHorizontalFlip(p=0.25),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02),
            v2.Resize(size, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform
