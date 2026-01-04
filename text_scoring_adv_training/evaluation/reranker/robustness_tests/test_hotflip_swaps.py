from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import List, Tuple, Set

import torch
from tqdm import tqdm


from robust_reranker_utils import (
    make_reranker_scorer, baseline_top_score_for_query
)

from text_scoring_adv_training.evaluation.robustness_tests.common import (
    load_corpus_jsonl, load_queries_tsv, load_qrels, load_run, dataset_files,
    annotated_nonrel_pids, choose_work_set_from_qrels, log_result,
    RunningEditStats, progress_line, ExperimentConfig, decode_span_text, seed_everything,
    override_models_from_cli, load_existing_results
)
from text_scoring_adv_training.evaluation.robustness_tests.common.beam import select_next_beam
from text_scoring_adv_training.evaluation.robustness_tests.common.paths import PATHS, local_eval_out_dir, local_runs_dir
from text_scoring_adv_training.evaluation.robustness_tests.common.hotflip import hotflip_pointwise


DATASETS = ["dl19", "dl20"]
DATA_ROOT = PATHS.data
RUNS_DIR = local_runs_dir(__file__)
EVAL_OUT_ROOT = local_eval_out_dir(__file__)
CORPUS_PATH = PATHS.data / "corpora/msmarco_full_corpus.jsonl"
TOP_K_RUN = 1000

NONREL_MAX_REL = 0
N_PASSAGES_PER_QUERY = 3

N_SAMPLE_POS = 1
HOTFLIP_TOP_K_OVERALL = 256


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
    script_slug="reranker_hotflip_beam",
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
    corpus = load_corpus_jsonl(CORPUS_PATH)

    for MODEL in CFG.models:
        seed_everything(CFG.seed)
        progress_line(f"\n=== {MODEL} ===")
        tok, scorer, score_fn, format_ids_fn, specials = make_reranker_scorer(
            MODEL,
            device=dev,
            batch_size=CFG.batch_size,
            models_dir=CFG.models_dir,
            use_flash_attention=True,
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

                baseline_pid, baseline_score = baseline_top_score_for_query(
                    score_fn, qtxt, run_full, corpus
                )
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

                    ids, att, s, e = format_ids_fn(qtxt, corpus[pid])
                    cur_text = decode_span_text(ids, s, e, tok)
                    best_score = float(cur_sc)

                    beam: List[BeamItem] = [
                        BeamItem(ids=ids, att=att, span=(s, e), score=best_score, text=cur_text, history_text=[cur_text])
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
                                n_sample_pos=N_SAMPLE_POS, top_k_overall=per_beam_k, 
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

                            scores = score_fn([(qtxt, t) for t in cands]).tolist()
                            for t, sc in zip(cands, scores):
                                ids2, att2, s2, e2 = format_ids_fn(qtxt, t)
                                expansions.append(
                                    BeamItem(
                                        ids=ids2,
                                        att=att2,
                                        span=(s2, e2),
                                        score=float(sc),
                                        text=t,
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

                        best_bi = max(beam, key=lambda x: x.score)
                        best_score = max(best_score, best_bi.score)
                        
                        depth += 1

                        if best_score > baseline_score:
                            stats.record(True, depth)
                            success = True
                            break

                    edits = depth if success else CFG.max_steps
                    log_result(stats_path, sample_id, success, edits)
                    if not success:
                        stats.record(False, 0)

        for line in stats.summary_strings(MODEL):
            progress_line(line)

        del scorer, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
