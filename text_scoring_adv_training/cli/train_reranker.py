import argparse, os, sys
from dataclasses import fields
from typing import Type, TypeVar

import deepspeed

from text_scoring_adv_training.training.reranker_trainer import (
    RerankerTrainer, RerankerConfig
)


def _build_config_from_args(args: argparse.Namespace, ConfigCls):
    cfg_keys = {f.name for f in fields(ConfigCls)}
    cfg = {k: getattr(args, k) for k in cfg_keys if hasattr(args, k)}
    return ConfigCls(**cfg)

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train pointwise reranker (with optional adversarial training)."
    )

    p.add_argument("--train_file", required=True, type=str)
    p.add_argument("--dev_file",   required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--paraphrased_path", type=str, default=None)
    p.add_argument("--injected_path", type=str, default=None)

    p.add_argument("--num_epochs", type=int, default=RerankerConfig.num_epochs)
    p.add_argument("--eval_every", type=int, default=RerankerConfig.eval_every)
    p.add_argument("--save_model", action="store_true", default=RerankerConfig.save_model)

    p.add_argument("--temperature", type=float, default=RerankerConfig.temperature)
    p.add_argument("--negatives_per_query", type=int, default=RerankerConfig.negatives_per_query)

    p.add_argument("--use_paraphrased", action="store_true", default=False)
    p.add_argument("--use_injected", action="store_true", default=False)
    p.add_argument("--use_rudimentary_manipulations", action="store_true", default=False)
    p.add_argument("--use_hotflip_swaps", action="store_true", default=False)
    p.add_argument("--use_pgd", action="store_true", default=False)

    p.add_argument("--paraphrase_weight",  type=float, default=RerankerConfig.paraphrase_weight)
    p.add_argument("--injection_weight",   type=float, default=RerankerConfig.injection_weight)
    p.add_argument("--rudimentary_weight", type=float, default=RerankerConfig.rudimentary_weight)
    p.add_argument("--hotflip_weight",     type=float, default=RerankerConfig.hotflip_weight)
    p.add_argument("--pgd_weight",         type=float, default=RerankerConfig.pgd_weight)

    p.add_argument("--eps_max",           type=float, default=RerankerConfig.eps_max)
    p.add_argument("--pgd_steps",         type=int,   default=RerankerConfig.pgd_steps)
    p.add_argument("--pgd_lr_factor",     type=float, default=RerankerConfig.pgd_lr_factor)
    p.add_argument("--eps_warmup_steps",  type=int,   default=RerankerConfig.eps_warmup_steps)
    
    p.add_argument("--max_length", type=int, default=RerankerConfig.max_length)
    p.add_argument("--model_name", type=str, default=RerankerConfig.model_name)
    p.add_argument("--dropout",    type=float, default=RerankerConfig.dropout)
    p.add_argument("--gradient_checkpointing", action="store_true", default=RerankerConfig.gradient_checkpointing)

    p.add_argument("--train_num_workers", type=int, default=RerankerConfig.train_num_workers)
    p.add_argument("--dev_num_workers",   type=int, default=RerankerConfig.dev_num_workers)

    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))

    return p

def main():
    parser = get_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    cfg = _build_config_from_args(args, RerankerConfig)
    trainer = RerankerTrainer(config=cfg, deepspeed_args=args)
    trainer.train()

if __name__ == "__main__":
    main()
