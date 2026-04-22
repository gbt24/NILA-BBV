import torch

from bbv.models import build_model


def test_build_model_resnet18_outputs_expected_shape() -> None:
    model = build_model("resnet18", num_classes=10)
    logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, 10)


def test_build_model_supports_text_classifier() -> None:
    model = build_model("text_cnn", num_classes=2, input_shape=(32,))
    logits = model(torch.randint(0, 2048, (4, 32), dtype=torch.long))
    assert logits.shape == (4, 2)
