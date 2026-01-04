import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Set, Optional
from tqdm import tqdm

from openai import OpenAI
from google import genai
from google.genai import types

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def key_for_row(row: Dict[str, Any]) -> Tuple[str, str]:
    ds = row.get("dataset", "unknown")
    ex_id = str(row.get("id", ""))
    return (ds, ex_id)

def to_map(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    m = {}
    for r in rows:
        m[key_for_row(r)] = r
    return m

def already_judged_keys(out_path: str) -> Set[Tuple[str, str, str]]:
    s: Set[Tuple[str, str, str]] = set()
    p = Path(out_path)
    if not p.exists():
        return s
    for row in load_jsonl(out_path):
        ds = row.get("dataset")
        ex_id = row.get("id")
        order = row.get("order", {})
        model_a = order.get("A")
        if ds and ex_id and model_a:
            s.add((str(ds), str(ex_id), str(model_a)))
    return s

def format_history(history: List[Dict[str, str]]) -> str:
    lines = []
    for msg in history:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)

def format_checklist(checklist: Any) -> str:
    if not checklist:
        return ""
    out = "Judge Checklist / Constraints:\n"
    if isinstance(checklist, list):
        for item in checklist:
            out += f"- {item}\n"
    elif isinstance(checklist, str):
        out += f"{checklist}\n"
    return out

SCORE_REGEX = re.compile(r"(?<!\d)(-2|-1|0|\+?1|\+?2)(?!\d)")

def parse_score(text: str) -> int:
    if not text: return 0
    t = text.strip()
    m = SCORE_REGEX.search(t)
    if m:
        try: return int(m.group(1))
        except: pass
    for tok in re.findall(r"[-+]?\d+", t):
        try:
            v = int(tok)
            if v in (-2, -1, 0, 1, 2): return v
        except: continue
    return 0

def score_to_which(score: int) -> str:
    if score < 0: return "a"
    if score > 0: return "b"
    return "tie"

def get_system_prompt(dataset_name: str) -> str:
    base = (
        "Judge two candidate LLM responses (A and B) to the same user input based on correctness, helpfulness, clarity, style/structure, safety, and overall response quality. "
    )
    
    if dataset_name == "wildbench":
        base += "Additionally, use the provided evaluation checklist as guidance for evaluating the responses. "
        
    base += (
        "\nAssign a single integer score that best compares the two candidates using this scale:\n"
        "-2: Candidate A is better\n"
        "-1: Candidate A is slightly better\n"
        "0: Candidates are equally good or bad\n"
        "1: Candidate B is slightly better\n"
        "2: Candidate B is better\n\n"
        "Output exactly one of these values: -2, -1, 0, 1, or 2."
    )
    return base



class JudgeResult(BaseModel):
    score: int = Field(description="Single integer in {-2,-1,0,1,2}. Negative favors A, positive favors B, 0 is tie.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="JSONL from model A")
    ap.add_argument("--run-b", required=True, help="JSONL from model B")
    ap.add_argument("--out", required=True, help="Where to write judgments JSONL")
    ap.add_argument("--judge-model", default="gpt-5.2-2025-12-11", help="Judge model ID")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    
    is_gemini = "gemini" in args.judge_model.lower()

    if is_gemini:
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError("Please set GOOGLE_API_KEY.")
        client = genai.Client()
    else: 
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Please set OPENAI_API_KEY.")
        client = OpenAI()

    a_rows = load_jsonl(args.run_a)
    b_rows = load_jsonl(args.run_b)
    a_map = to_map(a_rows)
    b_map = to_map(b_rows)

    all_keys = sorted(set(a_map.keys()) & set(b_map.keys()))
    judged_signatures = already_judged_keys(args.out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "a", encoding="utf-8")

    for (dataset_name, ex_id) in tqdm(all_keys, desc="Judging pairs"):
        ra = a_map[(dataset_name, ex_id)]
        rb = b_map[(dataset_name, ex_id)]

        model_a_name = ra.get("model", "model_A")
        model_b_name = rb.get("model", "model_B")
        
        checklist = ra.get("checklist") or rb.get("checklist")
        
        current_system_prompt = get_system_prompt(dataset_name)

        permutations = [
            {"label_to_row": {"A": ra, "B": rb}, "label_to_model": {"A": model_a_name, "B": model_b_name}},
            {"label_to_row": {"A": rb, "B": ra}, "label_to_model": {"A": model_b_name, "B": model_a_name}},
        ]

        for perm in permutations:
            label_to_row = perm["label_to_row"]
            label_to_model = perm["label_to_model"]
            
            signature = (str(dataset_name), str(ex_id), str(label_to_model["A"]))
            if signature in judged_signatures:
                continue

            history = label_to_row["A"].get("history", [])
            convo_text = format_history(history)
            
            ans_A = label_to_row["A"].get("response", "")
            ans_B = label_to_row["B"].get("response", "")

            user_msg_parts = []
            
            if dataset_name == "wildbench":
                user_msg_parts.append(f"### Context / Conversation History:\n{convo_text}")
                if checklist:
                    user_msg_parts.append(f"### {format_checklist(checklist)}")
                user_msg_parts.append(f"### Candidate A Response:\n{ans_A}")
                user_msg_parts.append(f"### Candidate B Response:\n{ans_B}")
            
            else:
                user_msg_parts.append(f"### User Query:\n{convo_text}")
                user_msg_parts.append(f"### Candidate A Response:\n{ans_A}")
                user_msg_parts.append(f"### Candidate B Response:\n{ans_B}")

            user_msg_parts.append("\nGive a single integer score: -2, -1, 0, 1, or 2.")
            user_msg = "\n\n".join(user_msg_parts)

            try:
                text = ""
                if is_gemini:
                    response = client.models.generate_content(
                        model=args.judge_model,
                        contents=user_msg,
                        config=types.GenerateContentConfig(
                            system_instruction=current_system_prompt,
                            thinking_config=types.ThinkingConfig(thinking_level="medium"),
                            response_mime_type="application/json",
                            response_schema=JudgeResult,
                        ),
                    )
                    text = (response.text or "").strip()
                else:
                    resp = client.responses.create(
                        model=args.judge_model,
                        instructions=current_system_prompt,
                        input=user_msg,
                        reasoning={"effort": "medium"},
                        text={"verbosity": "low"},
                    )
                    text = (resp.output_text or "").strip()
            except Exception as e:
                print(f"[warn] API error on ({dataset_name}, {ex_id}): {e}")
                continue

            try:
                jr = JudgeResult.model_validate_json(text)
                score = int(jr.score)
            except Exception:
                score = parse_score(text)
            
            judgment = {
                "dataset": dataset_name,
                "id": ex_id,
                "model_a": model_a_name,
                "model_b": model_b_name,
                "judge_model": args.judge_model,
                "seed": args.seed,
                "order": {"A": label_to_model["A"], "B": label_to_model["B"]},
                "score": score,
                "which": score_to_which(score),
                "raw_response": text
            }

            out_f.write(json.dumps(judgment, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

    out_f.close()
    print(f"Judgments written to {args.out}")

if __name__ == "__main__":
    main()