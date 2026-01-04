from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

import torch
from tqdm import tqdm

from text_scoring_adv_training.evaluation.robustness_tests.common import (
    dataset_slug_from_file, log_result,
    progress_line, short_text, ExperimentConfig, RunningEditStats,
)
from text_scoring_adv_training.evaluation.robustness_tests.common.beam import select_next_beam
from text_scoring_adv_training.evaluation.robustness_tests.common.paths import PATHS, local_eval_out_dir
from robust_gen_llm_utils import (
    make_generative_model, load_targets_jsonl,
    score_target_logprob, generate_and_check, generate_text,
)
from text_scoring_adv_training.evaluation.robustness_tests.common.mlm import (
    load_mlm,
    enumerate_swaps_dynamic,
)


DATA_PATH = Path("targets_sample.jsonl")
EVAL_OUT_ROOT = local_eval_out_dir(__file__)

GEN_MAX_NEW_TOKENS = None
GEN_MAX_NEW_TOKENS_FULL = 1024
VERBOSE = True
PROGRESS_EVERY = 1

MLM_MODEL = "answerdotai/ModernBERT-large"
MLM_TOP_K = 2048
MLM_PROB_MIN = 1e-3
MLM_BATCH_SIZE = 256


CFG = ExperimentConfig(
    models=[
        "google/gemma-3-27b-it",
    ],
    models_dir=PATHS.checkpoints,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=64,
    max_steps=2048,
    num_beams=16,
    test_variants_per_beam=16,
    diversity_last_n=3,
    elitism_keep=8,
    out_dir=EVAL_OUT_ROOT / "per_sample_stats",
    run_tag="test",
    script_slug="gen_mlm_beam",
).materialize(__file__)


@dataclass(eq=False)
class BeamItem:
    ids: torch.Tensor
    prompt_text: str
    score: float
    history_text: List[str]

@torch.inference_mode()
def compute_prompt_editable_positions(
    prompt: str,
    prompt_frozen: str,
    *,
    mlm_tok,
    specials: Set[int],
) -> Tuple[torch.Tensor, List[int]]:

    if isinstance(prompt_frozen, str) and prompt_frozen and prompt.startswith(prompt_frozen):
        start_char = len(prompt_frozen)
    else:
        start_char = 0
    end_char = len(prompt)

    enc = mlm_tok(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=mlm_tok.model_max_length,
        return_offsets_mapping=True,
    )
    ids = enc["input_ids"][0].cpu()
    offsets = enc["offset_mapping"][0].tolist()
    ids_list = ids.tolist()

    editable_positions: List[int] = []
    for i, (tid, (a, b)) in enumerate(zip(ids_list, offsets)):
        if tid in specials:
            continue
        if b <= start_char or a >= end_char:
            continue
        editable_positions.append(i)

    return ids, editable_positions

def main():
    dev = CFG.device
    
    mlm_tok, mlm_model, MASK_ID, SPECIALS = load_mlm(
        MLM_MODEL,
        device=dev,
    )

    items = load_targets_jsonl(DATA_PATH)
    ds_slug = dataset_slug_from_file(DATA_PATH)

    for MODEL in CFG.models:
        progress_line(f"\n=== {MODEL} ===")
        gm = make_generative_model(MODEL, device=dev, models_dir=CFG.models_dir)
        tok = gm.tokenizer
        stats = RunningEditStats(max_steps=CFG.max_steps)
        stats_path = CFG.stats_path(MODEL, ds_slug)

        for idx, it in enumerate(tqdm(items, desc="items")):
            rid = str(it.get("id", idx))
            prompt = it["prompt"]
            targets = it["targets"]
            prompt_frozen = it.get("prompt_frozen") or ""

            base_score = float(
                score_target_logprob(
                    gm,
                    [(prompt, targets)],
                    batch_size=CFG.batch_size,
                )[0]
            )

            ids_orig, editable_positions = compute_prompt_editable_positions(
                prompt=prompt,
                prompt_frozen=prompt_frozen,
                mlm_tok=mlm_tok,
                specials=SPECIALS,
            )

            beam: List[BeamItem] = [
                BeamItem(
                    ids=ids_orig.clone(),
                    prompt_text=prompt,
                    score=base_score,
                    history_text=[prompt],
                )
            ]
            visited: Set[str] = {prompt}
            depth = 0
            success = False

            if VERBOSE:
                num_positions = len(editable_positions)
                preview = " || ".join(short_text(t, 40) for t in targets)
                progress_line(f"[{rid}] baseline score={base_score:.4f} | targets={preview}")
                progress_line(
                    f"[{rid}] editable token positions in suffix={num_positions}"
                )


            while depth < CFG.max_steps:
                prev_top = beam[0].score if beam else float("-inf")
                expansions: List[BeamItem] = []

                for b in beam:
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
                        allowed_positions=editable_positions,
                    )
                    made = 0
                    for txt, ids_new in variants:
                        if not txt or txt == b.prompt_text:
                            continue
                        if txt in visited:
                            continue

                        visited.add(txt)
                        
                        expansions.append(
                            BeamItem(
                                ids=ids_new.clone(),
                                prompt_text=txt,
                                score=0.0,
                                history_text=b.history_text + [txt],
                            )
                        )
                        made += 1
                        if made >= CFG.test_variants_per_beam:
                            break

                if not expansions:
                    if VERBOSE:
                        progress_line(f"[{rid}] no more expansions at step {depth}.")
                    break

                scores = score_target_logprob(
                    gm,
                    [(bi.prompt_text, targets) for bi in expansions],
                    batch_size=CFG.batch_size,
                ).tolist()

                for bi, sc in zip(expansions, scores):
                    bi.score = float(sc)

                beam = select_next_beam(
                    expansions,
                    beam,
                    beam_size=CFG.num_beams,
                    last_n=int(CFG.diversity_last_n),
                    elitism_keep=int(CFG.elitism_keep or 0),
                )

                for b in beam:
                    visited.add(b.prompt_text)

                depth += 1

                ok, out_text = generate_and_check(
                    gm,
                    beam[0].prompt_text,
                    targets,
                    max_new_tokens=GEN_MAX_NEW_TOKENS,
                    temperature=0.0,
                    match_mode="text_prefix",
                )

                if VERBOSE and (depth % max(1, int(PROGRESS_EVERY)) == 0):
                    top_score = beam[0].score
                    delta = (top_score - prev_top) if prev_top != float("-inf") else 0.0
                    progress_line(f"[{rid}] step {depth}/{CFG.max_steps} | best={top_score:.4f} (Δ{delta:+.4f})")
                    progress_line(f"[{rid}] CURRENT PROMPT:\n{beam[0].prompt_text}")
                    progress_line(f"[{rid}] MODEL OUTPUT (check pass):\n{out_text}")

                if ok:
                    success = True
                    full_out = generate_text(
                        gm,
                        beam[0].prompt_text,
                        max_new_tokens=int(GEN_MAX_NEW_TOKENS_FULL or 512),
                        temperature=0.0,
                    )
                    progress_line(f"[SUCCESS id={rid}] Prompt (full):\n{beam[0].prompt_text}")
                    progress_line(f"[SUCCESS id={rid}] Model output (FULL):\n{full_out}")
                    break

            edits = depth if success else CFG.max_steps
            log_result(stats_path, f"{rid}", success, edits)
            stats.record(success, depth)

        for line in stats.summary_strings(MODEL):
            progress_line(line)

        del gm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
