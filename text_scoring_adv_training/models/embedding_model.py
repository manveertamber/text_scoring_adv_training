from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from typing import Literal, Optional

__all__ = ["EmbeddingModel"]

def masked_mean_pool(
    hidden: torch.Tensor,
    mask:   torch.Tensor,
    eps: float = 1e-8,
    l2_normalise: bool = True,
) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(1)
    counts = mask.sum(1).clamp_min_(1.0)
    pooled = summed / counts
    return (
        torch.nn.functional.normalize(pooled, dim=-1, p=2, eps=eps)
        if l2_normalise
        else pooled
    )


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        dropout: float = 0.0,
    ):
        super().__init__()

        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        cfg.hidden_dropout_prob = dropout
        cfg.attention_probs_dropout_prob = dropout

        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            config=cfg,
            trust_remote_code=True,
        )
        
    def forward(self, 
                input_ids:      torch.Tensor | None = None, 
                attention_mask: torch.Tensor | None = None, 
                inputs_embeds:  torch.Tensor | None = None,) -> torch.Tensor:
        if attention_mask is None:
            raise ValueError("`attention_mask` must be provided.")
        
        if inputs_embeds is not None:
            out = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
        elif input_ids is not None:
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            raise ValueError("Provide either `input_ids` or `inputs_embeds`.")

        return masked_mean_pool(out.last_hidden_state, attention_mask)
