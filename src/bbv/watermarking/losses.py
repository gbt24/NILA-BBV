from __future__ import annotations

import torch
import torch.nn.functional as F


def _project_parity_logits(logits_query: torch.Tensor) -> torch.Tensor:
    if logits_query.shape[1] == 2:
        return logits_query
    even_logits = torch.logsumexp(logits_query[:, 0::2], dim=1)
    odd_logits = torch.logsumexp(logits_query[:, 1::2], dim=1)
    return torch.stack([even_logits, odd_logits], dim=1)


def compute_loss_components(
    *,
    logits_main: torch.Tensor,
    labels_main: torch.Tensor,
    logits_query: torch.Tensor,
    bits: torch.Tensor,
    task_weight: float,
    wm_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    task_loss = F.cross_entropy(logits_main, labels_main)
    wm_loss = F.cross_entropy(_project_parity_logits(logits_query), bits)
    total_loss = task_weight * task_loss + wm_weight * wm_loss
    return total_loss, task_loss, wm_loss


def compute_watermark_loss(
    *,
    logits_main: torch.Tensor,
    labels_main: torch.Tensor,
    logits_query: torch.Tensor,
    bits: torch.Tensor,
    task_weight: float,
    wm_weight: float,
) -> torch.Tensor:
    total_loss, _, _ = compute_loss_components(
        logits_main=logits_main,
        labels_main=labels_main,
        logits_query=logits_query,
        bits=bits,
        task_weight=task_weight,
        wm_weight=wm_weight,
    )
    return total_loss
