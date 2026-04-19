from __future__ import annotations

from torchvision import transforms


def build_image_transform(train: bool):
    steps = [transforms.ToTensor()]
    if train:
        steps.insert(0, transforms.RandomHorizontalFlip(p=0.5))
    return transforms.Compose(steps)
