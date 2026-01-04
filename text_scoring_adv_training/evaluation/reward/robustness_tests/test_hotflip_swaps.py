from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import List, Tuple, Set

import torch
from tqdm import tqdm

from robust_rm_utils import (
    make_reward_scorer, load_reward_bench_jsonl,
)
from text_scoring_adv_training.evaluation.robustness_tests.common import ExperimentConfig, RunningEditStats, progress_line, dataset_slug_from_file, log_result, seed_everything, override_models_from_cli, load_existing_results
from text_scoring_adv_training.evaluation.robustness_tests.common.beam import select_next_beam
from text_scoring_adv_training.evaluation.robustness_tests.common.paths import PATHS, local_eval_out_dir
from text_scoring_adv_training.evaluation.robustness_tests.common.hotflip import hotflip_pointwise
from text_scoring_adv_training.evaluation.robustness_tests.common import decode_span_text


DATA_PATH = "reward_bench_consensus_filtered_shuffled_sampled.jsonl"
MAX_ITEMS = 100
EVAL_OUT_ROOT = local_eval_out_dir(__file__)

N_SAMPLE_POS = 1
HOTFLIP_TOP_K_OVERALL = 256


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
    script_slug="reward_hotflip_beam",
).materialize(__file__)


@dataclass(eq=False)
class BeamItem:
    ids: torch.Tensor
    att: torch.Tensor
    span: Tuple[int, int]
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
        tok, scorer, score_fn, format_ids_fn, specials = make_reward_scorer(
            MODEL, device=dev, batch_size=CFG.batch_size, models_dir=CFG.models_dir, use_flash_attention=True
        )

        stats = RunningEditStats(max_steps=CFG.max_steps, success_phrase="beat chosen")
        stats_path = CFG.stats_path(MODEL, ds_slug)

        completed = load_existing_results(stats_path, stats=stats)

        for idx, it in enumerate(tqdm(items, desc="items")):
            rid = str(it.get("id", idx))
            prompt = it["prompt"]
            chosen = it["chosen"]
            rejected = it["rejected"][:3]

            base_pairs = [(prompt, chosen)] + [(prompt, r) for r in rejected]
            base_scores = score_fn(base_pairs).tolist()
            chosen_score = float(base_scores[0])

            for j, r_text in enumerate(rejected):
                sample_id = f"{rid}:rej{j}"
                if sample_id in completed:
                    continue

                j_global = 1 + j
                r_score = float(base_scores[j_global])

                ids, att, s, e = format_ids_fn(prompt, r_text)
                cur_text = decode_span_text(ids, s, e, tok)
                cur_score = r_score
                best_score = cur_score

                beam: List[BeamItem] = [
                    BeamItem(ids=ids, att=att, span=(s, e), score=cur_score, text=cur_text, history_text=[cur_text])
                ]
                visited: Set[str] = {cur_text}
                depth = 0
                success = False

                while depth < CFG.max_steps:
                    expansions: List[BeamItem] = []
                    for bi in beam:
                        per_beam_k = min(CFG.test_variants_per_beam, HOTFLIP_TOP_K_OVERALL)
                        variants, _, _ = hotflip_pointwise(
                            ids=bi.ids, att=bi.att, span=bi.span,
                            scorer=scorer, specials=specials,
                            n_sample_pos=N_SAMPLE_POS,
                            top_k_overall=per_beam_k, 
                            token_embeddings=scorer.model.get_input_embeddings(),
                            device=dev
                        )
                        if not variants:
                            continue

                        cands: List[str] = []
                        cands_ids: List[torch.Tensor] = []

                        for v_ids in variants:
                            t = decode_span_text(v_ids, bi.span[0], bi.span[1], tok)
                            if not t:
                                continue
                            if t in visited:
                                continue

                            visited.add(t) 
                            
                            cands.append(t)
                            cands_ids.append(v_ids)

                        if not cands:
                            continue

                        scores = score_fn([(prompt, t) for t in cands]).tolist()
                        for t, sc, v_ids in zip(cands, scores, cands_ids):
                            ids2, att2, s2, e2 = format_ids_fn(prompt, t)
                            expansions.append(
                                BeamItem(
                                    ids=ids2, att=att2, span=(s2, e2),
                                    score=float(sc), text=t,
                                    history_text=bi.history_text + [t],
                                )
                            )


                    if not expansions:
                        break

                    beam = select_next_beam(
                        expansions,
                        beam,
                        beam_size=CFG.num_beams,
                        last_n=int(CFG.diversity_last_n),
                        elitism_keep=int(CFG.elitism_keep or 0),
                    )

                    for bi in beam:
                        visited.add(bi.text)
                        best_score = max(best_score, bi.score)

                    depth += 1

                    if best_score > chosen_score:
                        success = True
                        break

                edits = depth if success else CFG.max_steps
                log_result(stats_path, sample_id, success, edits)
                stats.record(success, depth)

        for line in stats.summary_strings(MODEL):
            progress_line(line)

        del scorer, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
