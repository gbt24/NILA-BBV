from pathlib import Path
from types import SimpleNamespace

import torch


def test_load_dataset_returns_real_sample_interface(
    monkeypatch, tmp_path: Path
) -> None:
    from bbv.datasets.loaders import load_dataset

    class FakeCIFAR10:
        def __init__(self, root: str, train: bool, download: bool, transform) -> None:
            self.root = root
            self.train = train
            self.download = download
            self.transform = transform
            self.classes = ["airplane", "car", "bird"]

        def __len__(self) -> int:
            return 5

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            image = torch.zeros(3, 32, 32)
            return image, int(index % 3)

    monkeypatch.setattr(
        "bbv.datasets.loaders.datasets",
        SimpleNamespace(CIFAR10=FakeCIFAR10),
    )

    loaded = load_dataset(
        name="cifar10", root=tmp_path / "raw", train=True, download=False
    )

    assert loaded.dataset_name == "cifar10"
    assert loaded.split_name == "train"
    image, label = loaded.dataset[0]
    assert image.shape == (3, 32, 32)
    assert isinstance(label, int)
