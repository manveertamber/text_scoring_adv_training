import argparse
import json
import math
import sys
from collections import defaultdict, Counter
from typing import Dict, Tuple, Any, Iterable, List, Optional

def load_jsonl(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows

def pct(n: float) -> str:
    if n != n or math.isinf(n):
        return "n/a"
    return f"{100.0*n:.2f}%"

def norm_which(w: str) -> str:
    if not w:
        return "tie"
    ch = w.strip().lower()[:1]
    if ch == "a": return "a"
    if ch == "b": return "b"
    return "tie"

def get_label_to_model(row: Dict[str, Any]) -> Dict[str, str]:
    order = row.get("order") or {}
    order_norm = {str(k).upper(): v for k, v in order.items() if v}

    model_a = row.get("model_a") or "model_A"
    model_b = row.get("model_b") or "model_B"

    return {
        "A": order_norm.get("A", model_a),
        "B": order_norm.get("B", model_b),
    }

def get_score(row: Dict[str, Any]) -> Optional[int]:

    if "score" not in row:
        return None
    try:
        s = int(row["score"])
        if s in (-2, -1, 0, 1, 2):
            return s
    except Exception:
        pass
    return None

def outcome_from_row(row: Dict[str, Any]) -> Tuple[int, str]:
    s = get_score(row)
    if s is not None:
        if s < 0: return (-1, "score")
        if s > 0: return ( 1, "score")
        return (0, "score")

    w = norm_which(row.get("which", "tie"))
    if w == "a": return (-1, "which")
    if w == "b": return ( 1, "which")
    return (0, "which")

def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Counter]:
    stats: Dict[str, Counter] = defaultdict(Counter)

    for r in rows:
        label_to_model = get_label_to_model(r)
        model_A = label_to_model["A"]
        model_B = label_to_model["B"]

        sign, _src = outcome_from_row(r)

        if sign == 0:
            for m in {model_A, model_B}:
                stats[m]["ties"] += 1
                stats[m]["total"] += 1
        elif sign < 0:
            stats[model_A]["wins"] += 1
            stats[model_A]["total"] += 1
            stats[model_B]["losses"] += 1
            stats[model_B]["total"] += 1
        else:
            stats[model_B]["wins"] += 1
            stats[model_B]["total"] += 1
            stats[model_A]["losses"] += 1
            stats[model_A]["total"] += 1

        s = get_score(r)
        if s is not None:
            stats[model_A]["score_sum"] += (-s)
            stats[model_A]["score_n"] += 1
            stats[model_B]["score_sum"] += ( s)
            stats[model_B]["score_n"] += 1

    return stats

def main():
    ap = argparse.ArgumentParser(description="Summarize pairwise judge outputs into per-model win rates.")
    ap.add_argument("inputs", nargs="+", help="One or more judgments JSONL files to aggregate.")
    ap.add_argument("--sort", choices=["elo", "strict", "ties", "score"], default="ties",
                    help="Sort by: strict (wins/(wins+losses)) or ties ((wins+0.5*ties)/total) or score (avg_score). 'elo' is an alias of 'ties'.")
    ap.add_argument("--csv", help="Optional path to write a CSV of the table.")
    args = ap.parse_args()

    rows = load_jsonl(args.inputs)
    if not rows:
        print("No rows found.", file=sys.stderr)
        sys.exit(1)

    stats = summarize(rows)

    table = []
    for m, c in stats.items():
        wins = c.get("wins", 0)
        losses = c.get("losses", 0)
        ties = c.get("ties", 0)
        total = c.get("total", 0)

        denom_strict = wins + losses
        wr_strict = (wins / denom_strict) if denom_strict > 0 else float("nan")
        wr_ties = ((wins + 0.5 * ties) / total) if total > 0 else float("nan")

        score_n = c.get("score_n", 0)
        score_sum = c.get("score_sum", 0)
        avg_score = (score_sum / score_n) if score_n > 0 else float("nan")

        table.append({
            "model": m,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "total": total,
            "win_rate_strict": wr_strict,
            "win_rate_ties_as_half": wr_ties,
            "avg_score": avg_score,
            "score_n": score_n,
        })

    if args.sort == "score":
        key = lambda r: r["avg_score"]
    elif args.sort in ("ties", "elo"):
        key = lambda r: r["win_rate_ties_as_half"]
    else:
        key = lambda r: r["win_rate_strict"]

    table.sort(key=key, reverse=True)

    name_w = max(5, max(len(r["model"]) for r in table))
    header = (
        f'{"Model".ljust(name_w)}  '
        f'{"Wins":>6}  {"Losses":>6}  {"Ties":>6}  {"Total":>6}  '
        f'{"Win% (strict)":>14}  {"Win% (ties=0.5)":>17}  '
        f'{"AvgScore":>9}  {"ScoreN":>6}'
    )
    print(header)
    print("-" * len(header))
    for r in table:
        avg_score_str = "n/a" if (r["avg_score"] != r["avg_score"]) else f'{r["avg_score"]: .3f}'
        print(
            f'{r["model"].ljust(name_w)}  '
            f'{r["wins"]:6d}  {r["losses"]:6d}  {r["ties"]:6d}  {r["total"]:6d}  '
            f'{pct(r["win_rate_strict"]).rjust(14)}  {pct(r["win_rate_ties_as_half"]).rjust(17)}  '
            f'{avg_score_str.rjust(9)}  {r["score_n"]:6d}'
        )

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "model","wins","losses","ties","total",
                "win_rate_strict","win_rate_ties_as_half",
                "avg_score","score_n"
            ])
            for r in table:
                w.writerow([
                    r["model"], r["wins"], r["losses"], r["ties"], r["total"],
                    f"{r['win_rate_strict']:.6f}" if r["win_rate_strict"]==r["win_rate_strict"] else "",
                    f"{r['win_rate_ties_as_half']:.6f}" if r["win_rate_ties_as_half"]==r["win_rate_ties_as_half"] else "",
                    f"{r['avg_score']:.6f}" if r["avg_score"]==r["avg_score"] else "",
                    r["score_n"],
                ])

if __name__ == "__main__":
    main()
