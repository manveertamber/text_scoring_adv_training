from __future__ import annotations

import torch
import torch.nn.functional as F


def softmax_nll_loss(scores: torch.Tensor, temperature: float = 1.0, sample_weights: torch.Tensor | None = None,) -> torch.Tensor:

    scores = scores.clamp(-100, 100)
    logits = scores / temperature
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=scores.device)
    loss_vec = F.cross_entropy(logits, targets, reduction="none")

    if sample_weights is None:
        return loss_vec.mean()

    denom = sample_weights.sum().clamp_min(1e-8)
        
    return (loss_vec * sample_weights).sum() / denom


def hinge_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    margin: float = 0.0,
    squared: bool = True,
) -> torch.Tensor:

    diff = negative_scores - positive_scores + margin
    losses = torch.relu(diff)
    if squared:
        losses = losses.pow(2)
    return losses.mean()

def mse_loss(
    predicted_scores: torch.Tensor,
    target_scores: torch.Tensor,
) -> torch.Tensor:

    return F.mse_loss(predicted_scores, target_scores, reduction="mean")


def kl_divergence_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor, 
    teacher_temperature: float = 1.0,
    student_temperature: float = 1.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    s = student_scores.clamp(-100, 100) / student_temperature
    t = teacher_scores.clamp(-100, 100) / teacher_temperature
    log_p_s = F.log_softmax(s, dim=-1)
    p_t     = F.softmax(t, dim=-1)

    loss = F.kl_div(log_p_s, p_t, reduction=reduction)
    return loss