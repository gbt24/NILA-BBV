import torch
from torch.utils.data import DataLoader, TensorDataset

from bbv.federated.evaluate import evaluate_accuracy


class PredictFirstLogit(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack((features[:, 0], features[:, 1]), dim=1)


def test_evaluate_accuracy_consumes_full_loader() -> None:
    features = torch.tensor(
        [[4.0, 0.0], [3.0, 1.0], [0.0, 5.0], [1.0, 2.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)

    accuracy = evaluate_accuracy(PredictFirstLogit(), loader, torch.device("cpu"))

    assert accuracy == 1.0
