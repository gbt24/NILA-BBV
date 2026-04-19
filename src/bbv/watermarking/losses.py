from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_watermark_loss(
    *,
    logits_main: torch.Tensor,
    labels_main: torch.Tensor,
    logits_query: torch.Tensor,
    bits: torch.Tensor,
    task_weight: float,
    wm_weight: float,
) -> torch.Tensor:
    task_loss = F.cross_entropy(logits_main, labels_main)
    wm_loss = F.cross_entropy(logits_query, bits)
    return task_weight * task_loss + wm_weight * wm_loss
