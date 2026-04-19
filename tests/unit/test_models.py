import torch

from bbv.models import build_model


def test_build_model_resnet18_outputs_expected_shape() -> None:
    model = build_model("resnet18", num_classes=10)
    logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, 10)
