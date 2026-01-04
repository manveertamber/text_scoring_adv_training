import argparse, os, sys
from dataclasses import fields
from typing import Type, TypeVar

import deepspeed

from text_scoring_adv_training.training.retriever_trainer import (
    RetrieverTrainer, RetrieverConfig
)

def _build_config_from_args(args: argparse.Namespace, ConfigCls):
    cfg_keys = {f.name for f in fields(ConfigCls)}
    cfg = {k: getattr(args, k) for k in cfg_keys if hasattr(args, k)}
    return ConfigCls(**cfg)

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train dense retriever (with optional adversarial training)."
    )

    p.add_argument("--train_file", required=True, type=str)
    p.add_argument("--dev_file",   required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--paraphrased_path", type=str, default=None)
    p.add_argument("--injected_path", type=str, default=None)
    p.add_argument("--max_dev_samples", type=int, default=RetrieverConfig.max_dev_samples)

    p.add_argument("--num_epochs", type=int, default=RetrieverConfig.num_epochs)
    p.add_argument("--eval_every", type=int, default=RetrieverConfig.eval_every)
    p.add_argument("--save_model", action="store_true", default=RetrieverConfig.save_model)
    p.add_argument("--save_every_epoch", action="store_true", default=RetrieverConfig.save_every_epoch)
    
    p.add_argument("--temperature", type=float, default=RetrieverConfig.temperature)
    p.add_argument("--negatives_per_query", type=int, default=RetrieverConfig.negatives_per_query)

    p.add_argument("--distill_with_kl", action="store_true", default=RetrieverConfig.distill_with_kl)
    p.add_argument("--distill_teacher_temperature", type=float, default=RetrieverConfig.distill_teacher_temperature)
    p.add_argument("--distill_student_temperature", type=float, default=RetrieverConfig.distill_student_temperature)
    p.add_argument("--teacher_scores_train", type=str, default=RetrieverConfig.teacher_scores_train)
    p.add_argument("--teacher_scores_dev", type=str, default=RetrieverConfig.teacher_scores_dev)

    p.add_argument("--use_paraphrased", action="store_true", default=False)
    p.add_argument("--use_injected", action="store_true", default=False)
    p.add_argument("--use_rudimentary_manipulations", action="store_true", default=False)
    p.add_argument("--use_hotflip_swaps", action="store_true", default=False)
    p.add_argument("--use_pgd", action="store_true", default=False)

    p.add_argument("--paraphrase_weight",  type=float, default=RetrieverConfig.paraphrase_weight)
    p.add_argument("--injection_weight",   type=float, default=RetrieverConfig.injection_weight)
    p.add_argument("--rudimentary_weight", type=float, default=RetrieverConfig.rudimentary_weight)
    p.add_argument("--hotflip_weight",     type=float, default=RetrieverConfig.hotflip_weight)
    p.add_argument("--pgd_weight",         type=float, default=RetrieverConfig.pgd_weight)

    p.add_argument("--eps_max",           type=float, default=RetrieverConfig.eps_max)
    p.add_argument("--pgd_steps",         type=int,   default=RetrieverConfig.pgd_steps)
    p.add_argument("--pgd_lr_factor",     type=float, default=RetrieverConfig.pgd_lr_factor)
    p.add_argument("--eps_warmup_steps",  type=int,   default=RetrieverConfig.eps_warmup_steps)

    p.add_argument("--query_maxlen",   type=int,   default=RetrieverConfig.query_maxlen)
    p.add_argument("--passage_maxlen", type=int,   default=RetrieverConfig.passage_maxlen)
    p.add_argument("--model_name",     type=str,   default=RetrieverConfig.model_name)
    p.add_argument("--dropout",        type=float, default=RetrieverConfig.dropout)
    p.add_argument("--query_prefix",   type=str,   default=RetrieverConfig.query_prefix)
    p.add_argument("--passage_prefix", type=str,   default=RetrieverConfig.passage_prefix)

    p.add_argument("--train_num_workers", type=int, default=RetrieverConfig.train_num_workers)
    p.add_argument("--dev_num_workers",   type=int, default=RetrieverConfig.dev_num_workers)

    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))

    return p

def main():
    parser = get_parser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    cfg = _build_config_from_args(args, RetrieverConfig)
    trainer = RetrieverTrainer(config=cfg, deepspeed_args=args)
    trainer.train()

if __name__ == "__main__":
    main()
