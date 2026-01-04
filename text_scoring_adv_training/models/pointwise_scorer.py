from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

__all__ = ["PointwiseScorer"]


class PointwiseScorer(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        dtype: str = "bfloat16",
        use_flash_attention: bool = False,
        gradient_checkpointing: bool = False,
        torch_compile_model: bool = True,
    ):
        super().__init__()

        if use_flash_attention:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=1,
                torch_dtype=dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=1,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if torch_compile_model:
            self.model = torch.compile(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"

        if (self.model.config.pad_token_id is None):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def forward(self,
                input_ids:      torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                inputs_embeds:  torch.Tensor | None = None,
                **extra,) -> torch.Tensor:
        if attention_mask is None:
            raise ValueError("`attention_mask` must be provided.")

        if inputs_embeds is not None:
            logits = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **extra,
            ).logits
        elif input_ids is not None:
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **extra,
            ).logits
        else:
            raise ValueError("Provide either `input_ids` or `inputs_embeds`.")

        return logits.squeeze(-1)
