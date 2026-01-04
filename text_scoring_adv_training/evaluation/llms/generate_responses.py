import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
from datasets import load_dataset


def clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    clean = []
    for m in messages:
        clean.append({
            "role": m.get("role"),
            "content": m.get("content")
        })
    return clean


def run_evals(
    llm: LLM,
    sampling: SamplingParams,
    out_path: Path,
    arena_hard_path: Path,
    model_name: str,
) -> None:
    
    all_examples = []
    if arena_hard_path.exists():
        print(f"Loading Arena Hard from {arena_hard_path}...")
        with open(arena_hard_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    row = json.loads(line)
                    all_examples.append({
                        "dataset": "arena_hard",
                        "id": row.get("uid", row.get("id", "unknown")),
                        "history": [{"role": "user", "content": row["prompt"]}],
                        "checklist": None
                    })
                except json.JSONDecodeError:
                    continue
    else:
        print(f"Warning: Arena Hard file not found at {arena_hard_path}")

    print("Loading allenai/WildBench (v2, test)...")
    try:
        wb_ds = load_dataset("allenai/WildBench", "v2", split="test")
        for row in wb_ds:
            raw_input = row.get("conversation_input", [])
            history = clean_messages(raw_input)
            
            all_examples.append({
                "dataset": "wildbench",
                "id": row.get("id"),
                "history": history,
                "checklist": row.get("checklist")
            })
    except Exception as e:
        print(f"Error loading WildBench: {e}")

    print(f"Total examples to run: {len(all_examples)}")

    if not all_examples:
        return

    prompts_batch = [ex["history"] for ex in all_examples]
    
    outs = llm.chat(messages=prompts_batch, sampling_params=sampling)

    with out_path.open("w", encoding="utf-8") as f:
        for ex, out in zip(all_examples, outs):
            response_text = out.outputs[0].text
            
            output_row = {
                "dataset": ex["dataset"],
                "id": ex["id"],
                "model": model_name,
                "history": ex["history"],
                "checklist": ex["checklist"],
                "response": response_text
            }
            f.write(json.dumps(output_row, ensure_ascii=False) + "\n")


def slugify_model_name(name: str) -> str:
    return name.replace("/", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo or local path")
    ap.add_argument("--out-dir", default="runs", help="Directory to write outputs")
    ap.add_argument("--arena-hard-file", default="arena_hard.jsonl", help="Path to arena_hard.jsonl")
    
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--frequency-penalty", type=float, default=0.0)
    ap.add_argument("--dtype", default="auto")
    
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    arena_path = Path(args.arena_hard_file) if args.arena_hard_file else (script_dir / "arena_hard.jsonl")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_slug = slugify_model_name(args.model)
    out_file = out_dir / f"{model_slug}_evals.jsonl"

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        frequency_penalty=args.frequency_penalty,
    )

    print(f"Writing all results to -> {out_file}")
    
    run_evals(
        llm=llm,
        sampling=sampling,
        out_path=out_file,
        arena_hard_path=arena_path,
        model_name=args.model,
    )

    print("done.")


if __name__ == "__main__":
    main()