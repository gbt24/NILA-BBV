from __future__ import annotations

from torchvision import transforms


def build_image_transform(train: bool, dataset_name: str = "cifar10"):
    steps = [transforms.ToTensor()]
    if dataset_name == "mnist":
        steps.append(transforms.Pad(2))
        steps.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    elif train:
        steps.insert(0, transforms.RandomHorizontalFlip(p=0.5))
    return transforms.Compose(steps)
