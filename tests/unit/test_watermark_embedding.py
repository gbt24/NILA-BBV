import torch
from torch.utils.data import TensorDataset

from bbv.federated.fedavg import FederatedClient, _train_one_client, build_server
from bbv.federated.hooks import WatermarkHook
from bbv.verification.baseline import recover_codeword_from_logits
from bbv.watermarking.losses import compute_watermark_loss


def test_watermark_loss_combines_task_and_query_batches() -> None:
    logits_main = torch.randn(4, 10, requires_grad=True)
    labels_main = torch.tensor([1, 2, 3, 4])
    logits_query = torch.randn(4, 2, requires_grad=True)
    bits = torch.tensor([0, 1, 0, 1])

    loss = compute_watermark_loss(
        logits_main=logits_main,
        labels_main=labels_main,
        logits_query=logits_query,
        bits=bits,
        task_weight=1.0,
        wm_weight=0.2,
    )
    assert loss.item() > 0.0


def test_watermark_hook_materializes_codebook_and_queries() -> None:
    hook = WatermarkHook(owner_id="owner0", code_length=8, wm_weight=0.2, seed=0)

    assert len(hook.codebook) == 8
    assert len(hook.positive_queries) == 8
    assert len(hook.negative_queries) == 8
    assert hook.wm_weight == 0.2


def test_train_one_client_accepts_non_cifar_query_shape() -> None:
    features = torch.zeros(8, 3, 32, 32)
    labels = torch.arange(8) % 2
    client = FederatedClient(
        client_id=0,
        dataset=type("ClientDatasetStub", (), {
            "client_id": 0,
            "dataset": TensorDataset(features, labels),
            "label_histogram": {"0": 4, "1": 4},
        })(),
        labels=labels,
    )
    hook = WatermarkHook(owner_id="owner0", code_length=4, wm_weight=0.2, seed=0)
    object.__setattr__(
        hook,
        "positive_queries",
        [torch.zeros(1, 28, 28) for _ in hook.positive_queries],
    )
    server = build_server(model_name="resnet18", num_classes=2, seed=0)

    _, losses = _train_one_client(
        global_model=server.model,
        client=client,
        learning_rate=0.05,
        local_epochs=1,
        batch_size=4,
        watermark_hook=hook,
    )

    assert losses["wm_loss"] >= 0.0


def test_verification_recovers_bits_from_parity_buckets() -> None:
    logits = [torch.tensor([4.9, 5.0, 4.9, 0.0])]

    recovered = recover_codeword_from_logits(logits)

    assert recovered == [0]
