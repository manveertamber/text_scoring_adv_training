from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from robust_retriever_utils import Q_PREFIX, P_PREFIX, make_embedder, sims_on_prefixed

from text_scoring_adv_training.evaluation.robustness_tests.common import (
    load_corpus_jsonl, load_queries_tsv, load_qrels, dataset_files, 
    log_injection_trial, progress_line, ExperimentConfig, seed_everything,
    override_models_from_cli
)
from text_scoring_adv_training.evaluation.robustness_tests.common.paths import PATHS, local_eval_out_dir
from text_scoring_adv_training.evaluation.robustness_tests.common.injections import load_injections_jsonl
from text_scoring_adv_training.utils.rudimentary_injections_generator import RudimentaryInjectionsGenerator


DATASETS = ["dl19", "dl20"]
DATA_ROOT = PATHS.data
EVAL_OUT_ROOT = local_eval_out_dir(__file__)
CORPUS_PATH = PATHS.data / "corpora/msmarco_full_corpus.jsonl"
SENTENCES_FILE = PATHS.data / "sentences/test_sentences.txt"

EMBED_BATCH = 8192
N_PASSAGES_PER_QUERY = 16384
N_SENTENCE_SAMPLES = 33
INJECTED_JSONL_PATH = PATHS.repo_root / "data_files/ranking_data_generation/dl_sentence_injections.jsonl"

MIN_REL = 3

CFG = ExperimentConfig(
    models=[
        "rlhn_retriever_e5_base_7hn_injection_4096",
    ],
    models_dir=PATHS.checkpoints,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=EMBED_BATCH,
    out_dir=EVAL_OUT_ROOT / "per_sample_stats",
    run_tag="test",
    script_slug="retriever_sentence_injection",
).materialize(__file__)


def main():
    override_models_from_cli(CFG)
    corpus = load_corpus_jsonl(CORPUS_PATH)

    for use_generated_injected in [False, True]:
        CFG.run_tag = f"generated_{use_generated_injected}_test"    

        inj_by_qid, inj_by_qid_pid = None, None
        if use_generated_injected:
            inj_by_qid, inj_by_qid_pid = load_injections_jsonl(Path(INJECTED_JSONL_PATH))

        overall_table: Dict[str, float] = {}

        for MODEL in CFG.models:
            seed_everything(CFG.seed)
            progress_line(f"\n=== {MODEL} ===")

            injections_generator = RudimentaryInjectionsGenerator(
                injection_sentences_file=str(SENTENCES_FILE),
            )

            tok, mdl, embed = make_embedder(
                MODEL, device=CFG.device, batch_size=CFG.batch_size, models_dir=CFG.models_dir
            )

            model_attempt_all = 0
            model_success_all = 0

            for ds in DATASETS:
                stats_path = CFG.stats_path(MODEL, ds)

                q_path, qrels_path = dataset_files(DATA_ROOT, ds)

                queries = load_queries_tsv(q_path)
                qrels   = load_qrels(qrels_path, min_rel=MIN_REL)

                qids = list(queries.keys())
                q_texts = [Q_PREFIX + queries[qid] for qid in qids]
                q_mat = embed(q_texts)
                qrow = {qid: i for i, qid in enumerate(qids)}

                ds_attempt = 0
                ds_success = 0

                for qid, rel_pid_map in tqdm(qrels.items(), desc=f"{ds} queries"):
                    if qid not in queries:
                        continue

                    rel_pids = [pid for pid in rel_pid_map if pid in corpus]
                    if not rel_pids:
                        continue
                    rel_pids = rel_pids[:N_PASSAGES_PER_QUERY]

                    rel_texts_for_filter = [corpus[pid] for pid in rel_pids]

                    q_vec = q_mat[qrow[qid]]

                    for pid in rel_pid_map:
                        if pid not in corpus:
                            continue
                        orig_text = corpus[pid]

                        if use_generated_injected:
                            cand_texts = (inj_by_qid_pid or {}).get((str(qid), str(pid)), [])
                            if not cand_texts:
                                continue
                            mod_texts: List[str] = [orig_text] + cand_texts
                        else:
                            def sample_clean_sentences(n: int) -> List[str]:
                                picked: List[str] = []
                                attempts = 0
                                rel_lower = [t.lower() for t in rel_texts_for_filter]
                                while len(picked) < n:
                                    need = n - len(picked)
                                    cand = injections_generator.get_random_sentences(need * 2)
                                    fresh = [s for s in cand if not any(s.lower() in t for t in rel_lower)]
                                    picked.extend(fresh[:need])
                                    attempts += 1
                                    if attempts > 100 and len(picked) < n:
                                        raise RuntimeError(f"Couldn’t find {n} clean sentences after {attempts} tries")
                                return picked

                            mod_texts = [orig_text]
                            for loc in ("start", "middle", "end"):
                                sentences = sample_clean_sentences(N_SENTENCE_SAMPLES)
                                for s in sentences:
                                    injected = injections_generator.inject_sentences(orig_text, [s], loc)
                                    if isinstance(injected, list):
                                        mod_texts.extend(injected)
                                    else:
                                        mod_texts.append(injected)

                        all_prefixed = [P_PREFIX + t for t in mod_texts]
                        scores_all = sims_on_prefixed(embed, q_vec, all_prefixed, batch=CFG.batch_size)
                        orig_score = scores_all[0].item()
                        scores = scores_all[1:]

                        sample_id = f"{ds}:{qid}:{pid}"

                        for trial_id, sc in enumerate(scores):
                            success = float(sc.item()) >= orig_score

                            log_injection_trial(
                                stats_path,
                                sample_id=sample_id,
                                trial_id=trial_id,
                                success=success,
                                num_injections=1,
                            )

                            ds_success += int(success)
                            ds_attempt += 1

                rate = (100.0 * ds_success / ds_attempt) if ds_attempt else 0.0
                model_success_all += ds_success
                model_attempt_all += ds_attempt
                progress_line(f"[{ds}] Sentence-injection vulnerability: {ds_success}/{ds_attempt} = {rate:6.2f}%")

            overall = (model_success_all / model_attempt_all) if model_attempt_all else 0.0
            overall_table[MODEL] = overall
            progress_line(f"→ OVERALL for {MODEL}: {model_success_all}/{model_attempt_all} = {overall*100:6.2f}%")

            del tok, mdl
            torch.cuda.empty_cache(); gc.collect()

        from tabulate import tabulate
        rows = [(m, f"{v:.2%}") for m, v in sorted(overall_table.items())]
        progress_line("\nSentence-injection vulnerability (higher = worse):")
        progress_line(tabulate(rows, headers=["model", "overall vulnerability"], tablefmt="github"))



if __name__ == "__main__":
    main()
