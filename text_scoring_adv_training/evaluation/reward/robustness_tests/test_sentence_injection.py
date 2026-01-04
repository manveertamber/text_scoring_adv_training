from __future__ import annotations

import gc, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from tqdm import tqdm

from robust_rm_utils import (
    make_reward_scorer, load_reward_bench_jsonl, load_reward_injections_jsonl,
)
from text_scoring_adv_training.evaluation.robustness_tests.common import ExperimentConfig, progress_line, dataset_slug_from_file, log_injection_trial, seed_everything, override_models_from_cli
from text_scoring_adv_training.evaluation.robustness_tests.common.paths import PATHS, local_eval_out_dir
from text_scoring_adv_training.utils.rudimentary_injections_generator import RudimentaryInjectionsGenerator


DATA_PATH = "reward_bench_consensus_filtered_shuffled_sampled.jsonl"
EVAL_OUT_ROOT = local_eval_out_dir(__file__)

INJECTED_JSONL_PATH = "reward_bench_sentence_injections.jsonl"
SENTENCES_FILE = PATHS.data / "sentences/test_sentences.txt"
N_SENTENCE_SAMPLES = 33


CFG = ExperimentConfig(
    models=[
        "reward_llama3b",
    ],
    models_dir=PATHS.checkpoints,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=128,
    out_dir=EVAL_OUT_ROOT / "per_sample_stats",
    run_tag="test",
    script_slug=f"reward_sentence_injection",
).materialize(__file__)


def main():
    override_models_from_cli(CFG)
    dev = CFG.device
    ds_slug = dataset_slug_from_file(DATA_PATH)
    items = load_reward_bench_jsonl(DATA_PATH)

    for use_generated_injected in [False, True]:
        CFG.run_tag = f"generated_{use_generated_injected}_test"  

        injected_by_id: Dict[str, Dict[str, Any]] = {}
        if use_generated_injected:
            injected_by_id = load_reward_injections_jsonl(Path(INJECTED_JSONL_PATH))
            progress_line(f"Loaded injected variants for {len(injected_by_id)} items from {INJECTED_JSONL_PATH}")

        for MODEL in CFG.models:
            seed_everything(CFG.seed)
            progress_line(f"\n=== {MODEL} ===")

            generator = RudimentaryInjectionsGenerator(
                injection_sentences_file=str(SENTENCES_FILE)
            )

            tok, mdl, score_fn, _, _ = make_reward_scorer(
                MODEL, device=dev, batch_size=CFG.batch_size, models_dir=CFG.models_dir, use_flash_attention=True
            )

            stats_path = CFG.stats_path(MODEL, ds_slug)
            total_attempts, total_success = 0, 0
            missing = 0

            for idx, it in enumerate(tqdm(items, desc="items")):
                rid = str(it.get("id", idx))
                prompt = it["prompt"]
                chosen = it["chosen"]

                base = float(score_fn([(prompt, chosen)])[0].item())
                cand_pairs: List[Tuple[str, str]] = []

                if not use_generated_injected:
                    for loc in ("start", "middle", "end"):
                        sentences = generator.get_random_sentences(N_SENTENCE_SAMPLES)
                        for s in sentences:
                            injected = generator.inject_sentences(chosen, [s], loc)
                            if isinstance(injected, list):
                                cand_pairs.extend((prompt, t) for t in injected)
                            else:
                                cand_pairs.append((prompt, injected))
                else:
                    rec = injected_by_id.get(rid)
                    if not rec or not rec.get("chosen"):
                        missing += 1
                        continue
                    for txt in rec["chosen"]:
                        cand_pairs.append((prompt, txt))

                if not cand_pairs:
                    continue

                scores = score_fn(cand_pairs).tolist()

                sample_id = f"{rid}:chosen"

                for trial_id, sc in enumerate(scores):
                    success = float(sc) >= base

                    total_attempts += 1
                    total_success  += int(success)

                    log_injection_trial(
                        stats_path,
                        sample_id=sample_id,
                        trial_id=trial_id,
                        success=success,
                        num_injections=1,
                    )

            rate = (100.0 * total_success / total_attempts) if total_attempts else 0.0
            progress_line(f"Sentence-injection vulnerability: {total_success}/{total_attempts} = {rate:6.2f}%")
            if use_generated_injected and missing:
                warnings.warn(f"{missing} item(s) had no injected CHOSEN variants; skipped.")

            del mdl, tok
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
