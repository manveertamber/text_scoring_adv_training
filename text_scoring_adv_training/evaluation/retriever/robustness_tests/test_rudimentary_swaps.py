from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import List, Set

import torch
from tqdm import tqdm

from robust_retriever_utils import P_PREFIX, make_embedder, baseline_top_sim_for_query, sims_on_prefixed

from text_scoring_adv_training.evaluation.robustness_tests.common import (
    load_corpus_jsonl, load_queries_tsv, load_qrels, load_run, 
    dataset_files, log_result, annotated_nonrel_pids, choose_work_set_from_qrels,
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

EMBED_BATCH = 8192
N_PASSAGES_PER_QUERY = 3
NONREL_MAX_REL = 0

MAX_STEPS = 512


CFG = ExperimentConfig(
    models=[
        "rlhn_retriever_e5_base_rudimentary_262544",
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
    script_slug="retriever_rudimentary_beam",
).materialize(__file__)


@dataclass(eq=False)
class BeamItem:
    text: str
    sim: float
    history_text: List[str]


def main():
    override_models_from_cli(CFG)
    corpus = load_corpus_jsonl(CORPUS_PATH)

    for model_name in CFG.models:
        seed_everything(CFG.seed)
        progress_line(f"\n=== {model_name} ===")
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

                p_texts_pref = [P_PREFIX + corpus[pid] for pid, _ in work]
                sims_init = sims_on_prefixed(embed, q_vec, p_texts_pref, batch=EMBED_BATCH).tolist()

                for (pid, _), cur_sim in zip(work, sims_init):
                    sample_id = f"{ds}:{qid}:{pid}"
                    if sample_id in completed:
                        continue

                    base_text = corpus[pid]

                    beams: List[BeamItem] = [BeamItem(text=base_text, sim=float(cur_sim), history_text=[base_text])]
                    visited: Set[str] = {base_text}
                    depth = 0
                    success = False

                    while depth < CFG.max_steps:
                        score_texts: List[str] = []
                        parents: List[BeamItem] = []

                        for b in beams:
                            variants_all = sample_variants(b.text, CFG.test_variants_per_beam)
                            for v in variants_all:
                                if v in visited:
                                    continue
                                visited.add(v) 
                                
                                score_texts.append(P_PREFIX + v)
                                parents.append(b)

                        if not score_texts:
                            break

                        sims_all = sims_on_prefixed(embed, q_vec, score_texts, batch=EMBED_BATCH).tolist()
                        
                        expansions: List[BeamItem] = []
                        for i, (sc, parent) in enumerate(zip(sims_all, parents)):
                            v_text = score_texts[i][len(P_PREFIX):]
                            expansions.append(
                                BeamItem(text=v_text, sim=float(sc), history_text=parent.history_text + [v_text])
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

                        if beams and max(b.sim for b in beams) > top_sim:
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
