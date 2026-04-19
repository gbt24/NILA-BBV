import torch

from bbv.models import build_model


def test_build_model_supports_mlp_and_resnet18() -> None:
    mlp = build_model("mlp", num_classes=10, input_shape=(3, 32, 32))
    resnet = build_model("resnet18", num_classes=10)

    assert mlp(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert resnet(torch.randn(2, 3, 32, 32)).shape == (2, 10)
