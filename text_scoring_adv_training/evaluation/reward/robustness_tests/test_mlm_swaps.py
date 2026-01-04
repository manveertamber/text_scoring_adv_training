from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Dict, List, Set

import torch
from tqdm import tqdm

from robust_rm_utils import (
    make_reward_scorer, load_reward_bench_jsonl,
)
from text_scoring_adv_training.evaluation.robustness_tests.common import ExperimentConfig, RunningEditStats, progress_line, dataset_slug_from_file, log_result, seed_everything, override_models_from_cli, load_existing_results
from text_scoring_adv_training.evaluation.robustness_tests.common.beam import select_next_beam
from text_scoring_adv_training.evaluation.robustness_tests.common.paths import PATHS, local_eval_out_dir

from text_scoring_adv_training.evaluation.robustness_tests.common.mlm import (
    load_mlm,
    enumerate_swaps_dynamic,
)


DATA_PATH = "reward_bench_consensus_filtered_shuffled_sampled.jsonl"
MAX_ITEMS = 100
EVAL_OUT_ROOT = local_eval_out_dir(__file__)

MLM_MODEL = "answerdotai/ModernBERT-large"
MLM_TOP_K = 64
MLM_PROB_MIN = 1e-3
MLM_BATCH_SIZE = 256


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
    script_slug="reward_mlm_beam",
).materialize(__file__)


@dataclass(eq=False)
class BeamItem:
    ids: torch.Tensor
    score: float
    text: str
    history_text: List[str]


def main():
    override_models_from_cli(CFG)
    dev = CFG.device
    ds_slug = dataset_slug_from_file(DATA_PATH)
    items = load_reward_bench_jsonl(DATA_PATH, max_items=MAX_ITEMS)

    for MODEL in CFG.models:
        seed_everything(CFG.seed)
        progress_line(f"\n=== {MODEL} ===")

        mlm_tok, mlm_model, MASK_ID, SPECIALS = load_mlm(MLM_MODEL, device=dev)

        tok, mdl, score_fn, _, _ = make_reward_scorer(MODEL, device=dev, batch_size=CFG.batch_size, models_dir=CFG.models_dir)
        stats = RunningEditStats(max_steps=CFG.max_steps, success_phrase="beat chosen")
        stats_path = CFG.stats_path(MODEL, ds_slug)

        passage_cache: Dict[str, torch.Tensor] = {}

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

                if r_text not in passage_cache:
                    ids_orig = mlm_tok(
                        r_text,
                        return_tensors="pt",
                        truncation=True,
                    )["input_ids"][0].cpu()
                    passage_cache[r_text] = ids_orig
                else:
                    ids_orig = passage_cache[r_text]


                beams: List[BeamItem] = [BeamItem(ids=ids_orig.clone(), score=cur_score, text=r_text, history_text=[r_text])]
                visited: Set[str] = {r_text}
                depth = 0
                success = False

                while depth < CFG.max_steps:
                    expansions_texts: List[str] = []
                    mapping: List[torch.Tensor] = []
                    parents: List[int] = []

                    for bi, b in enumerate(beams):
                        variants = enumerate_swaps_dynamic(
                            mlm_tok,
                            b.ids,
                            MASK_ID,
                            SPECIALS,
                            mlm_model,
                            num_positions=CFG.test_variants_per_beam,
                            limit=CFG.test_variants_per_beam,
                            batch_size=MLM_BATCH_SIZE,
                            top_k=MLM_TOP_K,
                            prob_min=MLM_PROB_MIN,
                        )
                        for txt, ids_new in variants:
                            if txt in visited:
                                continue

                            visited.add(txt)
                            
                            expansions_texts.append(txt)
                            mapping.append(ids_new)
                            parents.append(bi)

                    if not expansions_texts:
                        break

                    scores = score_fn([(prompt, t) for t in expansions_texts]).tolist()
                    expansions: List[BeamItem] = []
                    for t, sc, ids_new, bi in zip(expansions_texts, scores, mapping, parents):
                        parent = beams[bi]
                        expansions.append(BeamItem(ids=ids_new.clone(), score=float(sc), text=t, history_text=parent.history_text + [t]))

                    beams = select_next_beam(
                        expansions,
                        beams,
                        beam_size=CFG.num_beams,
                        last_n=int(CFG.diversity_last_n),
                        elitism_keep=int(CFG.elitism_keep or 0),
                    )

                    for b in beams:
                        visited.add(b.text)

                    depth += 1

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
