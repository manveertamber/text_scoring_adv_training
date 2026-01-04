from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

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
from text_scoring_adv_training.evaluation.robustness_tests.common.rudimentary_edits import sample_variants_with_frozen


DATA_PATH = Path("targets_sample.jsonl")
EVAL_OUT_ROOT = local_eval_out_dir(__file__)

GEN_MAX_NEW_TOKENS = None
GEN_MAX_NEW_TOKENS_FULL = 1024
VERBOSE = True
PROGRESS_EVERY = 1


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
    script_slug="gen_rudimentary_beam",
).materialize(__file__)


@dataclass
class BeamItem:
    prompt_text: str
    score: float
    history_text: List[str]


def main():
    dev = CFG.device
    items = load_targets_jsonl(DATA_PATH)
    ds_slug = dataset_slug_from_file(DATA_PATH)

    for MODEL in CFG.models:
        progress_line(f"\n=== {MODEL} ===")
        gm = make_generative_model(MODEL, device=dev, models_dir=CFG.models_dir)
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

            if VERBOSE:
                preview = " || ".join(short_text(t, 40) for t in targets)
                progress_line(f"[{rid}] baseline score={base_score:.4f} | targets={preview}")

            beam: List[BeamItem] = [
                BeamItem(prompt_text=prompt, score=base_score, history_text=[prompt])
            ]
            visited: Set[str] = {prompt}
            depth = 0
            success = False

            while depth < CFG.max_steps:
                prev_top = beam[0].score if beam else float("-inf")
                
                all_variants: List[str] = []
                parents: List[BeamItem] = []

                for bi in beam:
                    variants = sample_variants_with_frozen(
                        full_text=bi.prompt_text,
                        max_variants=CFG.test_variants_per_beam,
                        frozen_prefix=prompt_frozen
                    )
                    
                    for v in variants:
                        v_norm = v.strip()
                        if not v_norm:
                            continue
                        if v_norm in visited:
                            continue

                        visited.add(v_norm)
                        all_variants.append(v_norm)
                        parents.append(bi)

                if not all_variants:
                    if VERBOSE:
                        progress_line(f"[{rid}] no more expansions at step {depth}.")
                    break

                scores = score_target_logprob(
                    gm, 
                    [(v, targets) for v in all_variants], 
                    batch_size=CFG.batch_size
                ).tolist()

                expansions: List[BeamItem] = []
                for v, sc, parent in zip(all_variants, scores, parents):
                    expansions.append(BeamItem(
                        prompt_text=v,
                        score=float(sc),
                        history_text=parent.history_text + [v]
                    ))

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
                        gm, beam[0].prompt_text, max_new_tokens=int(GEN_MAX_NEW_TOKENS_FULL or 512),
                        temperature=0.0
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
