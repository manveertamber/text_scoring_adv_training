from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import List, Set

import torch
from tqdm import tqdm

from robust_rm_utils import (
    make_reward_scorer, load_reward_bench_jsonl,
)
from text_scoring_adv_training.evaluation.robustness_tests.common import ExperimentConfig, RunningEditStats, progress_line, dataset_slug_from_file, log_result, seed_everything, override_models_from_cli, load_existing_results
from text_scoring_adv_training.evaluation.robustness_tests.common.beam import select_next_beam
from text_scoring_adv_training.evaluation.robustness_tests.common.paths import PATHS, local_eval_out_dir
from text_scoring_adv_training.evaluation.robustness_tests.common.rudimentary_edits import sample_variants


DATA_PATH = "reward_bench_consensus_filtered_shuffled_sampled.jsonl"
MAX_ITEMS = 100
EVAL_OUT_ROOT = local_eval_out_dir(__file__)


CFG = ExperimentConfig(
    models=[
        "reward_llama3b",
    ],
    models_dir=PATHS.checkpoints,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=128,
    max_steps=512,
    num_beams=16,
    test_variants_per_beam=16,
    diversity_last_n=3,
    elitism_keep=8,
    out_dir=EVAL_OUT_ROOT / "per_sample_stats",
    run_tag="test",
    script_slug="reward_rudimentary_beam",
).materialize(__file__)


@dataclass(eq=False)
class BeamItem:
    text: str
    score: float
    history_text: List[str]


def main():
    override_models_from_cli(CFG)
    dev = CFG.device
    ds_slug = dataset_slug_from_file(DATA_PATH)
    items = load_reward_bench_jsonl(DATA_PATH, max_items=MAX_ITEMS)

    for MODEL in CFG.models:
        seed_everything(CFG.seed)
        progress_line(f"\n=== {MODEL} ===")
        tok, mdl, score_fn, _, _ = make_reward_scorer(
            MODEL, device=dev, batch_size=CFG.batch_size, models_dir=CFG.models_dir, use_flash_attention=True
        )

        stats = RunningEditStats(max_steps=CFG.max_steps)
        stats_path = CFG.stats_path(MODEL, ds_slug)
        completed = load_existing_results(stats_path, stats=stats)

        for idx, it in enumerate(tqdm(items, desc="items")):
            rid = str(it.get("id", idx))
            prompt   = it["prompt"]
            chosen   = it["chosen"]
            rejected = it["rejected"][:3]

            base_pairs = [(prompt, chosen)] + [(prompt, r) for r in rejected]
            base_scores = score_fn(base_pairs).tolist()
            chosen_score = float(base_scores[0])

            for j, r_text in enumerate(rejected):
                sample_id = f"{rid}:rej{j}"
                if sample_id in completed:
                    continue
        
                j_global = 1 + j
                cur_score = float(base_scores[j_global])

                beams: List[BeamItem] = [BeamItem(text=r_text, score=cur_score, history_text=[r_text])]
                depth = 0
                success = False
                visited: Set[str] = {r_text}

                while depth < CFG.max_steps:
                    all_variants: List[str] = []
                    parents: List[BeamItem] = []

                    for b in beams:
                        variants = sample_variants(b.text, CFG.test_variants_per_beam)
                        for v in variants:
                            if v in visited:
                                continue
                            visited.add(v)
                            all_variants.append(v)
                            parents.append(b)

                    if not all_variants:
                        break

                    scores = score_fn([(prompt, v) for v in all_variants]).tolist()

                    expansions: List[BeamItem] = []
                    for v, sc, parent in zip(all_variants, scores, parents):
                        expansions.append(
                            BeamItem(text=v, score=float(sc), history_text=parent.history_text + [v])
                        )

                    if not expansions:
                        break

                    beams = select_next_beam(
                        expansions,
                        beams,
                        beam_size=CFG.num_beams,
                        last_n=int(CFG.diversity_last_n),
                        elitism_keep=int(CFG.elitism_keep or 0),
                    )

                    depth += 1

                    for b in beams:
                        visited.add(b.text)
                        
                    if beams and max(b.score for b in beams) > chosen_score:
                        success = True
                        break

                edits = depth if success else CFG.max_steps
                log_result(stats_path, sample_id, success, edits)
                stats.record(success, depth)

        for line in stats.summary_strings(MODEL):
            progress_line(line)

        del mdl, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
