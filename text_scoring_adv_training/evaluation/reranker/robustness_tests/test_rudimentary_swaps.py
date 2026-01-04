#!/usr/bin/env python
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import List, Set

import torch
from tqdm import tqdm

from robust_reranker_utils import (
    make_reranker_scorer, baseline_top_score_for_query
)

from text_scoring_adv_training.evaluation.robustness_tests.common import (
    load_corpus_jsonl, load_queries_tsv, load_qrels, load_run, dataset_files,
    annotated_nonrel_pids, choose_work_set_from_qrels, log_result,
    RunningEditStats, progress_line, ExperimentConfig, seed_everything,
    override_models_from_cli, load_existing_results
)
from text_scoring_adv_training.evaluation.robustness_tests.common.beam import select_next_beam
from text_scoring_adv_training.evaluation.robustness_tests.common.paths import PATHS, local_eval_out_dir, local_runs_dir
from text_scoring_adv_training.evaluation.robustness_tests.common.rudimentary_edits import sample_variants

DATASETS = ["dl19", "dl20"]
DATA_ROOT = PATHS.data
RUNS_DIR = local_runs_dir(__file__)
EVAL_OUT_ROOT = local_eval_out_dir(__file__)
CORPUS_PATH = PATHS.data / "corpora/msmarco_full_corpus.jsonl"
TOP_K_RUN = 1000

NONREL_MAX_REL = 0
N_PASSAGES_PER_QUERY = 3


CFG = ExperimentConfig(
    models=[
        "rlhn_reranker_qwen_base",
    ],
    models_dir=PATHS.checkpoints,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=256,
    max_steps=512,
    num_beams=16,
    test_variants_per_beam=16,
    diversity_last_n=3,
    elitism_keep=8,
    out_dir=EVAL_OUT_ROOT / "per_sample_stats",
    run_tag="test",
    script_slug="reranker_rudimentary_beam",
).materialize(__file__)


@dataclass(eq=False)
class BeamItem:
    text: str
    score: float
    history_text: List[str]


def main():
    override_models_from_cli(CFG)
    dev = CFG.device
    corpus = load_corpus_jsonl(CORPUS_PATH)

    for MODEL in CFG.models:
        seed_everything(CFG.seed)
        progress_line(f"\n=== {MODEL} ===")
        tok, mdl, score_fn, _, _ = make_reranker_scorer(
            MODEL, device=dev, batch_size=CFG.batch_size, models_dir=CFG.models_dir, use_flash_attention=True
        )

        stats = RunningEditStats(max_steps=CFG.max_steps, success_phrase="beat top score")
        
        completed_by_ds = {
            ds: load_existing_results(CFG.stats_path(MODEL, ds), stats=stats)
            for ds in DATASETS
        }

        for ds in DATASETS:
            stats_path = CFG.stats_path(MODEL, ds)
            q_path, qrels_path = dataset_files(DATA_ROOT, ds)
            run_path = RUNS_DIR / f"run.{MODEL}.{ds}.txt"

            queries = load_queries_tsv(q_path)
            qrels_all = load_qrels(qrels_path, min_rel=0)
            runs = load_run(run_path, TOP_K_RUN)

            completed = completed_by_ds.get(ds, set())

            for qid, qtxt in tqdm(queries.items(), desc=f"{ds} queries"):
                run_full = runs.get(qid, [])
                if not run_full:
                    continue

                baseline_pid, baseline_score = baseline_top_score_for_query(score_fn, qtxt, run_full, corpus)
                if baseline_pid is None:
                    continue

                nonrel = annotated_nonrel_pids(qrels_all, qid, NONREL_MAX_REL)
                work = choose_work_set_from_qrels(corpus, nonrel, N_PASSAGES_PER_QUERY)
                if not work:
                    continue

                init_scores = score_fn([(qtxt, corpus[pid]) for pid, _ in work]).tolist()

                for (pid, _), cur_sc in zip(work, init_scores):
                    sample_id = f"{ds}:{qid}:{pid}"
                    if sample_id in completed:
                        continue

                    base_text = corpus[pid]

                    beams: List[BeamItem] = [BeamItem(text=base_text, score=float(cur_sc), history_text=[base_text])]
                    visited: Set[str] = {base_text}
                    depth = 0
                    success = False

                    while depth < CFG.max_steps:
                        all_variants: List[str] = []
                        parents: List[BeamItem] = []

                        for b in beams:
                            variants_all = sample_variants(b.text, CFG.test_variants_per_beam)
                            for v in variants_all:
                                if v in visited:
                                    continue

                                visited.add(v)
                                all_variants.append(v)
                                parents.append(b)

                        if not all_variants:
                            break

                        scores = score_fn([(qtxt, v) for v in all_variants]).tolist()
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

                        for b in beams:
                            visited.add(b.text)
                            
                        depth += 1

                        if beams and max(b.score for b in beams) > baseline_score:
                            stats.record(True, depth)
                            success = True
                            break

                    edits = depth if success else CFG.max_steps
                    log_result(stats_path, sample_id, success, edits)
                    if not success:
                        stats.record(False, 0)

        for line in stats.summary_strings(MODEL):
            progress_line(line)

        del mdl, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
