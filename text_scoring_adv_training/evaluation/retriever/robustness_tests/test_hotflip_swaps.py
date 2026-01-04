from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import List, Set

import torch
from tqdm import tqdm

from robust_retriever_utils import (
    P_PREFIX, make_embedder, baseline_top_sim_for_query,
    sims_on_prefixed, prefix_token_len
)

from text_scoring_adv_training.evaluation.robustness_tests.common import (
    load_corpus_jsonl, load_queries_tsv, load_qrels, load_run, 
    dataset_files, log_result, annotated_nonrel_pids, choose_work_set_from_qrels,
    RunningEditStats, progress_line, ExperimentConfig, seed_everything,
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

EMBED_BATCH = 8192
N_PASSAGES_PER_QUERY = 3
NONREL_MAX_REL = 0

MAX_STEPS = 512

N_SAMPLE_POS = 1
HOTFLIP_TOP_K_OVERALL = 256


CFG = ExperimentConfig(
    models=[
        "rlhn_retriever_e5_base",
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
    script_slug="retriever_hotflip_beam",
).materialize(__file__)


@dataclass(eq=False)
class BeamItem:
    ids: torch.Tensor
    att: torch.Tensor
    sim: float
    text: str
    history_text: List[str]


class RetrieverScoreWrapper:
    def __init__(self, model, q_vec: torch.Tensor):
        self.model = model
        self.q_vec = q_vec.clone()

    def __call__(self, *, inputs_embeds, attention_mask):
        pooled = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return pooled @ self.q_vec



def main():
    override_models_from_cli(CFG)
    dev = CFG.device
    corpus = load_corpus_jsonl(CORPUS_PATH)

    for model_name in CFG.models:
        seed_everything(CFG.seed)
        progress_line(f"\n=== {model_name} ===")
        tok, mdl, embed = make_embedder(
            model_name,
            device=dev,
            batch_size=EMBED_BATCH,
            models_dir=CFG.models_dir,
        )
        
        specials = {x for x in tok.all_special_ids if x is not None}
        if tok.pad_token_id is not None:
            specials.add(tok.pad_token_id)

        pfx_len = prefix_token_len(tok, P_PREFIX)

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
                q_vec_dev = q_vec.to(dev)
                scorer = RetrieverScoreWrapper(mdl, q_vec_dev)

                for (pid, _), cur_sim in zip(work, sims_init):
                    sample_id = f"{ds}:{qid}:{pid}"
                    if sample_id in completed:
                        continue

                    base_text = corpus[pid]
                    enc0 = tok(P_PREFIX + base_text, return_tensors="pt", truncation=True, max_length=512)
                    ids_orig = enc0["input_ids"][0].cpu()
                    att_orig = enc0["attention_mask"][0].cpu()

                    beam: List[BeamItem] = [
                        BeamItem(ids=ids_orig, att=att_orig, sim=float(cur_sim), text=base_text, history_text=[base_text])
                    ]
                    visited: Set[str] = {base_text}
                    depth = 0
                    success = False

                    while depth < CFG.max_steps:
                        expansions: List[BeamItem] = []

                        for bi, b in enumerate(beam):
                            span_end = int(b.att.sum().item())
                            span = (int(pfx_len), span_end)
                            if span[0] >= span[1]:
                                continue

                            per_beam_k = min(CFG.test_variants_per_beam, HOTFLIP_TOP_K_OVERALL)
                            v_ids_list, _swaps, _meta = hotflip_pointwise(
                                ids=b.ids, att=b.att, span=span,
                                scorer=scorer, specials=specials,
                                n_sample_pos=N_SAMPLE_POS,
                                top_k_overall=per_beam_k,
                                token_embeddings=mdl.encoder.get_input_embeddings(),
                                device=dev,
                            )
                            if not v_ids_list:
                                continue

                            cand_prefixed: List[str] = []
                            decoded_unpref: List[str] = []

                            for v_ids in v_ids_list:
                                txt_unpref = tok.decode(
                                    v_ids[span[0]:span[1]],
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False,
                                )
                                if not txt_unpref:
                                    continue
                                if txt_unpref in visited:
                                    continue

                                visited.add(txt_unpref)

                                decoded_unpref.append(txt_unpref)
                                cand_prefixed.append(P_PREFIX + txt_unpref)

                            if not cand_prefixed:
                                continue

                            sims = sims_on_prefixed(embed, q_vec, cand_prefixed, batch=EMBED_BATCH).tolist()
                            for unpref, sim in zip(decoded_unpref, sims):
                                enc2 = tok(P_PREFIX + unpref, return_tensors="pt", truncation=True, max_length=512)
                                ids2 = enc2["input_ids"][0].cpu()
                                att2 = enc2["attention_mask"][0].cpu()
                                parent = beam[bi]
                                expansions.append(
                                    BeamItem(
                                        ids=ids2,
                                        att=att2,
                                        sim=float(sim),
                                        text=unpref,
                                        history_text=parent.history_text + [unpref],
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

                        for b in beam:
                            visited.add(b.text)
                            
                        depth += 1

                        if beam and max(b.sim for b in beam) > top_sim:
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
