from pathlib import Path
import json
from types import SimpleNamespace

import torch

from bbv.attacks import run_attack
from bbv.models import build_model


def test_pruning_attack_zeroes_fraction_of_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model_state": {"w": torch.tensor([0.1, 0.2, 0.9, 1.2])}}, checkpoint)

    result = run_attack(
        attack_name="pruning",
        checkpoint_path=checkpoint,
        output_root=tmp_path / "attacks",
        seed=0,
    )
    attacked = torch.load(result.attacked_checkpoint, map_location="cpu")["model_state"]["w"]
    assert (attacked == 0).sum().item() >= 1
    attack_log = json.loads(result.attack_log.read_text(encoding="utf-8"))
    assert "attack_config" in attack_log


def test_run_attack_persists_attack_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model_state": {"w": torch.tensor([0.1, 0.2, 0.9, 1.2])}}, checkpoint)

    result = run_attack(
        attack_name="pruning",
        checkpoint_path=checkpoint,
        output_root=tmp_path / "attacks",
        seed=0,
    )

    payload = json.loads(result.attack_log.read_text())
    assert payload["attack_config"]["attack_name"] == "pruning"


def test_run_attack_log_contains_minimal_protocol_fields(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    model = build_model("mlp", num_classes=2, input_shape=(3, 32, 32))
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": "mlp",
            "num_classes": 2,
            "input_shape": (3, 32, 32),
        },
        checkpoint,
    )

    class FakeDataset:
        def __init__(self, size: int) -> None:
            self.classes = ["0", "1"]
            self.targets = [index % 2 for index in range(size)]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.randn(3, 32, 32), int(self.targets[index])

    def fake_load_dataset(*, root: Path, train: bool, download: bool, name: str):
        del root, download, name
        return SimpleNamespace(dataset=FakeDataset(8 if train else 4), split_name="train")

    import bbv.attacks.extraction as extraction_module

    original_load_dataset = extraction_module.load_dataset if hasattr(extraction_module, "load_dataset") else None
    extraction_module.load_dataset = fake_load_dataset

    try:
        result = run_attack(
            attack_name="extraction",
            checkpoint_path=checkpoint,
            output_root=tmp_path / "attacks",
            seed=3,
        )
    finally:
        if original_load_dataset is None:
            delattr(extraction_module, "load_dataset")
        else:
            extraction_module.load_dataset = original_load_dataset

    payload = json.loads(result.attack_log.read_text())
    assert payload["attack"] == "extraction"
    assert payload["source_checkpoint"] == str(checkpoint)
    assert payload["attacked_checkpoint"] == str(result.attacked_checkpoint)
    assert payload["seed"] == 3


def test_finetune_attack_trains_for_multiple_steps(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    model = build_model("mlp", num_classes=2, input_shape=(3, 32, 32))
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": "mlp",
            "num_classes": 2,
            "input_shape": (3, 32, 32),
        },
        checkpoint,
    )

    class FakeDataset:
        def __init__(self, size: int) -> None:
            self.classes = ["0", "1"]
            self.targets = [index % 2 for index in range(size)]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.randn(3, 32, 32), int(self.targets[index])

    def fake_load_dataset(*, root: Path, train: bool, download: bool, name: str):
        del root, download, name
        return SimpleNamespace(dataset=FakeDataset(8 if train else 4))

    import bbv.attacks.finetune as finetune_module

    original_load_dataset = finetune_module.load_dataset if hasattr(finetune_module, "load_dataset") else None
    finetune_module.load_dataset = fake_load_dataset

    try:
        result = run_attack(
            attack_name="finetune",
            checkpoint_path=checkpoint,
            output_root=tmp_path / "attacks",
            seed=0,
            dataset_name="cifar10",
        )
    finally:
        if original_load_dataset is None:
            delattr(finetune_module, "load_dataset")
        else:
            finetune_module.load_dataset = original_load_dataset

    log = json.loads(result.attack_log.read_text(encoding="utf-8"))
    assert log["attack"] == "finetune"
    assert log["attack_config"]["num_optimizer_steps"] > 0


def test_distillation_attack_logs_student_training(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    model = build_model("mlp", num_classes=2, input_shape=(3, 32, 32))
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": "mlp",
            "num_classes": 2,
            "input_shape": (3, 32, 32),
        },
        checkpoint,
    )

    class FakeDataset:
        def __init__(self, size: int) -> None:
            self.classes = ["0", "1"]
            self.targets = [index % 2 for index in range(size)]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.randn(3, 32, 32), int(self.targets[index])

    def fake_load_dataset(*, root: Path, train: bool, download: bool, name: str):
        del root, download, name
        return SimpleNamespace(dataset=FakeDataset(8 if train else 4), split_name="train")

    import bbv.attacks.distillation as distillation_module

    original_load_dataset = distillation_module.load_dataset if hasattr(distillation_module, "load_dataset") else None
    distillation_module.load_dataset = fake_load_dataset

    try:
        result = run_attack(
            attack_name="distillation",
            checkpoint_path=checkpoint,
            output_root=tmp_path / "attacks",
            seed=0,
            dataset_name="cifar10",
        )
    finally:
        if original_load_dataset is None:
            delattr(distillation_module, "load_dataset")
        else:
            distillation_module.load_dataset = original_load_dataset

    payload = json.loads(result.attack_log.read_text(encoding="utf-8"))
    assert payload["attack"] == "distillation"
    assert payload["attack_config"]["num_student_steps"] > 0
    assert payload["attack_config"]["teacher_query_mode"] in {"logits", "hard-label"}


def test_extraction_attack_uses_query_budget(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    model = build_model("mlp", num_classes=2, input_shape=(3, 32, 32))
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": "mlp",
            "num_classes": 2,
            "input_shape": (3, 32, 32),
        },
        checkpoint,
    )

    class FakeDataset:
        def __init__(self, size: int) -> None:
            self.classes = ["0", "1"]
            self.targets = [index % 2 for index in range(size)]

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.randn(3, 32, 32), int(self.targets[index])

    def fake_load_dataset(*, root: Path, train: bool, download: bool, name: str):
        del root, download, name
        return SimpleNamespace(dataset=FakeDataset(16 if train else 4), split_name="train")

    import bbv.attacks.extraction as extraction_module

    original_load_dataset = extraction_module.load_dataset if hasattr(extraction_module, "load_dataset") else None
    extraction_module.load_dataset = fake_load_dataset

    try:
        result = run_attack(
            attack_name="extraction",
            checkpoint_path=checkpoint,
            output_root=tmp_path / "attacks",
            seed=0,
            dataset_name="cifar10",
            attack_config={"query_budget": 6},
        )
    finally:
        if original_load_dataset is None:
            delattr(extraction_module, "load_dataset")
        else:
            extraction_module.load_dataset = original_load_dataset

    payload = json.loads(result.attack_log.read_text(encoding="utf-8"))
    assert payload["attack"] == "extraction"
    assert payload["attack_config"]["query_budget"] == 6
    assert payload["attack_config"]["num_student_steps"] > 0
