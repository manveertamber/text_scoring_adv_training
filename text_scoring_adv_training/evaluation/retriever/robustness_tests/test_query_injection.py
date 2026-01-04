from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from robust_retriever_utils import P_PREFIX, make_embedder, baseline_top_sim_for_query, sims_on_prefixed

from text_scoring_adv_training.evaluation.robustness_tests.common import (
    load_corpus_jsonl, load_queries_tsv, load_qrels, load_run, 
    dataset_files, log_injection_trial, annotated_nonrel_pids, choose_work_set_from_qrels,
    progress_line, ExperimentConfig, seed_everything, override_models_from_cli
)
from text_scoring_adv_training.evaluation.robustness_tests.common.paths import PATHS, local_eval_out_dir, local_runs_dir
from text_scoring_adv_training.evaluation.robustness_tests.common.injections import load_injections_jsonl
from text_scoring_adv_training.utils.rudimentary_injections_generator import RudimentaryInjectionsGenerator


DATASETS = ["dl19", "dl20"]
DATA_ROOT = PATHS.data
RUNS_DIR = local_runs_dir(__file__)
EVAL_OUT_ROOT = local_eval_out_dir(__file__)
CORPUS_PATH = PATHS.data / "corpora/msmarco_full_corpus.jsonl"
TOP_K_RUN = 1000

EMBED_BATCH = 8192
N_PASSAGES_PER_QUERY = 16384
NONREL_MAX_REL = 0

INJECTED_JSONL_PATH = PATHS.repo_root / "data_files/ranking_data_generation/dl_query_injections.jsonl"


CFG = ExperimentConfig(
    models=[
        "rlhn_retriever_e5_base_7hn_injection_4096",
    ],
    models_dir=PATHS.checkpoints,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=EMBED_BATCH,
    out_dir=EVAL_OUT_ROOT / "per_sample_stats",
    run_tag="test",
    script_slug=f"retriever_query_injection",
).materialize(__file__)


def main():
    override_models_from_cli(CFG)
    corpus = load_corpus_jsonl(CORPUS_PATH)

    for use_generated_injected in [False, True]:
        CFG.run_tag = f"generated_{use_generated_injected}_test"  
        
        inj_by_qid = inj_by_qid_pid = None
        if use_generated_injected:
            inj_by_qid, inj_by_qid_pid = load_injections_jsonl(Path(INJECTED_JSONL_PATH))

        for model_name in CFG.models:
            seed_everything(CFG.seed)
            progress_line(f"\n=== {model_name} ===")

            adv = RudimentaryInjectionsGenerator()
            
            def _make_injections_for_passage(base_passage: str, query_text: str):
                texts = []
                for loc in ("start", "middle", "end"):
                    t = adv.inject_query_into_passage(base_passage, query_text, loc, num_injections=1)
                    texts.append(t)
                return texts

            tok, mdl, embed = make_embedder(
                model_name,
                device=CFG.device,
                batch_size=EMBED_BATCH,
                models_dir=CFG.models_dir
            )

            model_tot_successful_variants, model_tot_attempted_variants = 0, 0

            for ds in DATASETS:
                stats_path = CFG.stats_path(model_name, ds)
                q_path, qrels_path = dataset_files(DATA_ROOT, ds)
                run_path = RUNS_DIR / f"run.{model_name}.{ds}.txt"

                queries: Dict[str, str] = load_queries_tsv(q_path)
                qrels_all: Dict[str, Dict[str, int]] = load_qrels(qrels_path, min_rel=0)
                runs = load_run(run_path, TOP_K_RUN)

                for qid, qtxt in tqdm(queries.items(), desc=f"{ds} queries"):
                    run_full = runs.get(qid, [])
                    if not run_full:
                        continue

                    baseline_pid, top_sim, q_vec = baseline_top_sim_for_query(embed, qtxt, run_full, corpus)
                    if baseline_pid is None:
                        continue
                    nonrel = annotated_nonrel_pids(qrels_all, qid, NONREL_MAX_REL)
                    work = choose_work_set_from_qrels(corpus, nonrel, N_PASSAGES_PER_QUERY)
                    if not work:
                        continue

                    for pid, rank in work:
                        base_text = corpus[pid]

                        if use_generated_injected:
                            inj_texts: List[str] = []
                            if inj_by_qid_pid is not None:
                                inj_texts = inj_by_qid_pid.get((qid, pid), [])
                            if not inj_texts and inj_by_qid is not None:
                                inj_texts = inj_by_qid.get(qid, [])
                        else:
                            inj_texts = _make_injections_for_passage(base_text, qtxt)

                        if not inj_texts:
                            continue

                        sims = sims_on_prefixed(
                            embed, q_vec, [P_PREFIX + t for t in inj_texts], batch=EMBED_BATCH
                        ).numpy()

                        sample_id = f"{ds}:{qid}:{pid}"

                        for trial_id, sim in enumerate(sims):
                            success = float(sim) > top_sim

                            log_injection_trial(
                                stats_path,
                                sample_id=sample_id,
                                trial_id=trial_id,
                                success=success,
                                num_injections=1,
                            )

                            model_tot_successful_variants += int(success)
                            model_tot_attempted_variants += 1

            overall = (100.0 * model_tot_successful_variants / model_tot_attempted_variants) if model_tot_attempted_variants else 0.0
            progress_line(f"→ OVERALL for {model_name}: {model_tot_successful_variants}/{model_tot_attempted_variants} = {overall:6.2f}% success rate across all variants")

            del tok, mdl
            torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    main()
