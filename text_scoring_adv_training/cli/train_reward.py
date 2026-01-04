import argparse, os, sys
from dataclasses import fields

import deepspeed

from text_scoring_adv_training.training.reward_trainer import (
    RewardTrainer, RewardConfig
)

def _build_config_from_args(args: argparse.Namespace, ConfigCls):
    cfg_keys = {f.name for f in fields(ConfigCls)}
    cfg = {k: getattr(args, k) for k in cfg_keys if hasattr(args, k)}
    return ConfigCls(**cfg)

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train reward model (with optional adversarial training)."
    )

    p.add_argument("--train_file", required=True, type=str)
    p.add_argument("--dev_file",   required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--paraphrased_path", type=str, default=None)
    p.add_argument("--injected_path",    type=str, default=None)
    p.add_argument("--max_dev_samples",  type=int, default=None)

    p.add_argument("--num_epochs", type=int, default=RewardConfig.num_epochs)
    p.add_argument("--eval_every", type=int, default=RewardConfig.eval_every)
    p.add_argument("--save_model", action="store_true", default=RewardConfig.save_model)

    p.add_argument("--temperature", type=float, default=RewardConfig.temperature)

    p.add_argument("--use_paraphrased", action="store_true", default=False)
    p.add_argument("--use_injected", action="store_true", default=False)
    p.add_argument("--use_rudimentary_manipulations", action="store_true", default=False)
    p.add_argument("--use_hotflip_swaps", action="store_true", default=False)
    p.add_argument("--use_pgd", action="store_true", default=False)

    p.add_argument("--paraphrase_weight",  type=float, default=RewardConfig.paraphrase_weight)
    p.add_argument("--injection_weight",   type=float, default=RewardConfig.injection_weight)
    p.add_argument("--rudimentary_weight", type=float, default=RewardConfig.rudimentary_weight)
    p.add_argument("--hotflip_weight",     type=float, default=RewardConfig.hotflip_weight)
    p.add_argument("--pgd_weight",         type=float, default=RewardConfig.pgd_weight)

    p.add_argument("--eps_max",          type=float, default=RewardConfig.eps_max)
    p.add_argument("--pgd_steps",        type=int,   default=RewardConfig.pgd_steps)
    p.add_argument("--pgd_lr_factor",    type=float, default=RewardConfig.pgd_lr_factor)
    p.add_argument("--eps_warmup_steps", type=int,   default=RewardConfig.eps_warmup_steps)

    p.add_argument("--max_length",         type=int,  default=RewardConfig.max_length)
    p.add_argument("--pad_to_multiple_of", type=int,  default=RewardConfig.pad_to_multiple_of)
    p.add_argument("--padding_side",       type=str,  choices=["left", "right"], default=RewardConfig.padding_side)

    p.add_argument("--model_name", type=str, default=RewardConfig.model_name)
    p.add_argument("--gradient_checkpointing", action="store_true", default=RewardConfig.gradient_checkpointing)

    p.add_argument("--train_num_workers", type=int, default=RewardConfig.train_num_workers)
    p.add_argument("--dev_num_workers",   type=int, default=RewardConfig.dev_num_workers)

    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))

    return p

def main():
    parser = get_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    cfg = _build_config_from_args(args, RewardConfig)
    trainer = RewardTrainer(config=cfg, deepspeed_args=args)
    trainer.train()

if __name__ == "__main__":
    main()
