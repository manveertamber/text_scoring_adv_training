from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Dict, List, Set

import torch
from tqdm import tqdm

from robust_retriever_utils import (
    P_PREFIX, make_embedder, baseline_top_sim_for_query,
    sims_on_prefixed
)

from text_scoring_adv_training.evaluation.robustness_tests.common import (
    load_corpus_jsonl, load_queries_tsv, load_qrels, load_run, 
    dataset_files, log_result, annotated_nonrel_pids, choose_work_set_from_qrels,
    RunningEditStats, progress_line, ExperimentConfig, seed_everything,
    override_models_from_cli, load_existing_results
)
from text_scoring_adv_training.evaluation.robustness_tests.common.beam import select_next_beam
from text_scoring_adv_training.evaluation.robustness_tests.common.paths import PATHS, local_eval_out_dir, local_runs_dir

from text_scoring_adv_training.evaluation.robustness_tests.common.mlm import (
    load_mlm,
    enumerate_swaps_dynamic,
)


DATASETS = ["dl19", "dl20"]
DATA_ROOT = PATHS.data
RUNS_DIR = local_runs_dir(__file__)
EVAL_OUT_ROOT = local_eval_out_dir(__file__)
CORPUS_PATH = PATHS.data / "corpora/msmarco_full_corpus.jsonl"
TOP_K_RUN = 1000

EMBED_BATCH = 8192
N_PASSAGES_PER_QUERY = 3
NONREL_MAX_REL = 0

MAX_STEPS = 512

MLM_MODEL = "answerdotai/ModernBERT-large"
MLM_TOP_K = 64
MLM_PROB_MIN = 1e-3
MLM_BATCH_SIZE = 256


CFG = ExperimentConfig(
    models=[
        "rlhn_retriever_e5_base_pgd_0.015625",
    ],
    models_dir=PATHS.checkpoints,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=EMBED_BATCH,
    max_steps=MAX_STEPS,
    num_beams=16,
    test_variants_per_beam=16,
    diversity_last_n=3,
    elitism_keep=8,
    out_dir=EVAL_OUT_ROOT / "per_sample_stats",
    run_tag="test",
    script_slug="retriever_mlm_beam",
).materialize(__file__)


@dataclass(eq=False)
class BeamItem:
    ids: torch.Tensor
    sim: float
    text: str
    history_text: List[str]


def main():
    override_models_from_cli(CFG)
    corpus = load_corpus_jsonl(CORPUS_PATH)

    for model_name in CFG.models:
        seed_everything(CFG.seed)
        progress_line(f"\n=== {model_name} ===")

        mlm_tok, mlm_model, MASK_ID, SPECIALS = load_mlm(MLM_MODEL, device=CFG.device)
        
        tok, mdl, embed = make_embedder(
            model_name,
            device=CFG.device,
            batch_size=EMBED_BATCH,
            models_dir=CFG.models_dir
        )

        stats = RunningEditStats(max_steps=CFG.max_steps, success_phrase="reach rank #1")

        completed_by_ds = {
            ds: load_existing_results(CFG.stats_path(model_name, ds), stats=stats)
            for ds in DATASETS
        }

        for ds in DATASETS:
            stats_path = CFG.stats_path(model_name, ds)

            q_path, qrels_path = dataset_files(DATA_ROOT, ds)
            run_path = RUNS_DIR / f"run.{model_name}.{ds}.txt"

            queries = load_queries_tsv(q_path)
            qrels_all = load_qrels(qrels_path, min_rel=0)
            runs = load_run(run_path, TOP_K_RUN)

            completed = completed_by_ds.get(ds, set()) 

            passage_cache: Dict[str, torch.Tensor] = {}

            for qid, q_text in tqdm(queries.items(), desc=f"{ds} queries"):
                run_full = runs.get(qid, [])
                if not run_full:
                    continue

                baseline_pid, baseline_top, q_emb = baseline_top_sim_for_query(embed, q_text, run_full, corpus)
                if baseline_pid is None:
                    continue
                nonrel = annotated_nonrel_pids(qrels_all, qid, NONREL_MAX_REL)
                work = choose_work_set_from_qrels(corpus, nonrel, N_PASSAGES_PER_QUERY)
                if not work:
                    continue

                p_texts_work = [P_PREFIX + corpus[pid] for pid, _ in work]
                p_sims = sims_on_prefixed(embed, q_emb, p_texts_work, batch=EMBED_BATCH).tolist()

                for (pid, _), cur_sim in zip(work, p_sims):
                    sample_id = f"{ds}:{qid}:{pid}"
                    if sample_id in completed:
                        continue

                    if pid not in passage_cache:
                        ids_orig = mlm_tok(
                            corpus[pid],
                            return_tensors="pt",
                            truncation=True,
                        )["input_ids"][0].cpu()
                        passage_cache[pid] = ids_orig
                    else:
                        ids_orig = passage_cache[pid]


                    beams: List[BeamItem] = [
                        BeamItem(ids=ids_orig.clone(), sim=float(cur_sim), text=corpus[pid], history_text=[corpus[pid]])
                    ]
                    visited: Set[str] = {corpus[pid]}
                    depth = 0
                    success = False

                    while depth < CFG.max_steps:
                        expansions: List[BeamItem] = []
                        score_texts: List[str] = []
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

                                score_texts.append(P_PREFIX + txt)
                                mapping.append(ids_new)
                                parents.append(bi)

                        if not score_texts:
                            break

                        sims_all = sims_on_prefixed(embed, q_emb, score_texts, batch=EMBED_BATCH).tolist()

                        for s_txt, sim, ids_new, bi in zip(score_texts, sims_all, mapping, parents):
                            unpref = s_txt[len(P_PREFIX):]
                            parent = beams[bi]
                            expansions.append(
                                BeamItem(ids=ids_new.clone(), sim=float(sim), text=unpref, history_text=parent.history_text + [unpref])
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

                        if beams and max(b.sim for b in beams) > baseline_top:
                            stats.record(True, depth)
                            success = True
                            break

                    edits = depth if success else CFG.max_steps
                    log_result(stats_path, sample_id, success, edits)
                    if not success:
                        stats.record(False, 0)

        for line in stats.summary_strings(model_name):
            progress_line(line)

        del mdl, tok
        gc.collect()


if __name__ == "__main__":
    main()
