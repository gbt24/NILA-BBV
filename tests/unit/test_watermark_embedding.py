import torch

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
