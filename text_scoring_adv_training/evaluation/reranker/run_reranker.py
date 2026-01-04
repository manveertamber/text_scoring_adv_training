import argparse
import os
import math
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt



class RerankerModel(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        text_maxlength: int = 1024,
        batch_size: int = 512,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.text_maxlength = text_maxlength

        lname = model_name.lower()

        if ("qwen" in lname) and ("rlhn" in lname):
            self.model_type = "qwen3_sequence_cls"

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.tokenizer.padding_side = "left"
            
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True,
                )
                .to(self.device)
                .eval()
            )

            self.model = torch.compile(self.model)
            
            self._body = lambda q, d: (
                "How relevant is the following document to the query?\n\n"
                f"Query: {q}\n\nDocument: {d}"
            )

        elif "qwen3-reranker" in lname:
            self.model_type = "qwen3_reranker"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            tp = torch.cuda.device_count() if torch.cuda.is_available() else 1
            self.model = LLM(
                model=model_name,
                tensor_parallel_size=tp,
                max_model_len=8192,
                enable_prefix_caching=True,
                enforce_eager=True,
            )

            self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
            self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
            self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
            self.sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
                allowed_token_ids=[self.true_token, self.false_token],
            )
            self.default_instruction = (
                "Given a web search query, retrieve relevant passages that answer the query"
            )

        else:
            raise ValueError(
                f"Model '{model_name}' not recognized as a Qwen reranker.\n"
                "Expected name containing one of:\n"
                "  - 'qwen3-reranker' (generative, vLLM)\n"
                "  - 'qwen3sequenceclassifier' or 'qwen3-seqcls' (sequence-classification)"
            )

    def score_pairs(self, pairs, instruction: str | None = None):
        if self.model_type == "qwen3_sequence_cls":
            scores = []
            for i in tqdm(range(0, len(pairs), self.batch_size)):
                batch_pairs = pairs[i : i + self.batch_size]
                prompts = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": self._body(q, d)}],
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                    for q, d in batch_pairs
                ]
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.text_maxlength,
                    truncation=True,
                    add_special_tokens=False,
                ).to(self.device)

                with torch.no_grad(), torch.amp.autocast(self.device, dtype=torch.bfloat16):
                    logits = self.model(**inputs).logits.squeeze(-1)
                    if logits.dim() == 1:
                        scores.extend(logits.tolist())
                    else:
                        probs = torch.softmax(logits, dim=-1)[:, -1]
                        scores.extend(probs.tolist())
            return scores

        elif self.model_type == "qwen3_reranker":
            task_instruction = instruction or self.default_instruction
            scores = []
            for i in tqdm(range(0, len(pairs), self.batch_size)):
                batch_pairs = pairs[i : i + self.batch_size]
                msgs = [
                    self._format_qwen3_instruction(task_instruction, q, d)
                    for q, d in batch_pairs
                ]
                tokens = [
                    self.tokenizer.apply_chat_template(
                        conv, tokenize=True, add_generation_prompt=False, enable_thinking=False
                    )
                    for conv in msgs
                ]

                tokens = [
                    t[: self.text_maxlength - len(self.suffix_tokens)] + self.suffix_tokens
                    for t in tokens
                ]
                prompts = [TokensPrompt(prompt_token_ids=t) for t in tokens]
                outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=False)

                for out in outputs:
                    lp = out.outputs[0].logprobs[-1]
                    t_log = lp.get(self.true_token, None)
                    f_log = lp.get(self.false_token, None)
                    t_lp = -10 if t_log is None else t_log.logprob
                    f_lp = -10 if f_log is None else f_log.logprob
                    t_p, f_p = math.exp(t_lp), math.exp(f_lp)
                    scores.append(t_p / (t_p + f_p))
            return scores

        else:
            raise RuntimeError("Unsupported model_type in score_pairs().")

    def _format_qwen3_instruction(self, inst, query, doc):
        return [
            {
                "role": "system",
                "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".',
            },
            {
                "role": "user",
                "content": f"<Instruct>: {inst}\n\n<Query>: {query}\n\n<Document>: {doc}",
            },
        ]

    def __del__(self):
        if getattr(self, "model_type", None) == "qwen3_reranker":
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except Exception:
                pass

DL_DATASETS = {"dl19", "dl20"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def norm_text(s: str, limit: int = 2048) -> str:
    return str(s).replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()[:limit]


def load_corpus(dataset: str, cache: dict) -> dict:
    beir_key = "msmarco" if dataset in DL_DATASETS else dataset
    if ("corpus", beir_key) in cache:
        return cache[("corpus", beir_key)]
    ds = load_dataset(f"BeIR/{beir_key}", "corpus")["corpus"]
    id_key = "_id"
    passages = {
        str(row[id_key]): norm_text(row["text"])
        for row in tqdm(ds, desc=f"Loading corpus: {beir_key}")
    }
    cache[("corpus", beir_key)] = passages
    return passages


def load_queries(dataset: str, queries_dir: str, cache: dict) -> dict:
    if dataset in DL_DATASETS:
        key = ("queries-tsv", dataset)
        if key in cache:
            return cache[key]
        tsv = os.path.join(queries_dir, f"{dataset}-queries.tsv")
        q = {}
        with open(tsv, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                qid, query = line.split("\t", 1)
                q[qid.strip()] = norm_text(query)
        cache[key] = q
        return q

    key = ("queries-hf", dataset)
    if key in cache:
        return cache[key]
    ds = load_dataset(f"BeIR/{dataset}", "queries")["queries"]
    id_key = "_id"
    q = {
        str(row[id_key]): norm_text(row["text"])
        for row in tqdm(ds, desc=f"Loading queries: {dataset}")
    }
    cache[key] = q
    return q


def read_retrieval(run_path: str, topk: int = 100):
    qids, pids = [], []
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, pid, rank, *_ = parts
            if int(rank) <= topk:
                qids.append(qid.strip())
                pids.append(pid.strip())
    return qids, pids


def score_and_write(
    model: RerankerModel,
    model_name: str,
    dataset: str,
    qids: list[str],
    pids: list[str],
    queries: dict,
    corpus: dict,
    out_dir: str,
    instruction: str | None = None,
):
    pairs, valid_idx = [], []
    missing_q = missing_p = 0
    for i, (qid, pid) in enumerate(zip(qids, pids)):
        q = queries.get(qid)
        p = corpus.get(pid)
        if q is None:
            missing_q += 1
            continue
        if p is None:
            missing_p += 1
            continue
        pairs.append([q, p])
        valid_idx.append(i)

    if missing_q or missing_p:
        print(
            f"[{dataset}] Skipped {missing_q} missing queries and {missing_p} missing passages."
        )
    if not pairs:
        print(f"[{dataset}] No valid pairs; skipping.")
        return

    with torch.no_grad():
        scores = model.score_pairs(pairs, instruction=instruction)

    results = list(zip([qids[i] for i in valid_idx], [pids[i] for i in valid_idx], scores))

    grouped = defaultdict(list)
    for qid, pid, score in results:
        grouped[qid].append((pid, float(score)))
    sorted_results = {qid: sorted(ps, key=lambda x: x[1], reverse=True) for qid, ps in grouped.items()}

    ensure_dir(out_dir)
    formatted = model_name.replace("/", "_")
    out_path = os.path.join(out_dir, f"run.{formatted}.{dataset}.txt")
    with open(out_path, "w", encoding="utf-8") as out:
        for qid in sorted(sorted_results.keys()):
            for rank, (pid, score) in enumerate(sorted_results[qid], start=1):
                out.write(f"{qid} Q0 {pid} {rank} {score} reranked\n")
    print(f"[{dataset}] Wrote: {out_path}")


def parse_models(model_specs: list[str]) -> list[dict]:
    parsed = []
    for spec in model_specs:
        if ":" in spec:
            name, bs = spec.split(":", 1)
            parsed.append({"name": name.strip(), "batch_size": int(bs)})
        else:
            parsed.append({"name": spec.strip(), "batch_size": 512})
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Qwen reranker runner (generative + seq-cls)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "../../../checkpoints/rlhn_reranker_qwen_base_combined_pgd_rudi_hotflip_inject:128",
        ],
        help="Model names as NAME[:BATCH_SIZE]. Supports 'qwen3-reranker*' and 'qwen3-seqcls*'.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["dl19", "dl20", "climate-fever", "dbpedia-entity", "fever", "fiqa", "hotpotqa", "nfcorpus", "nq", "scifact", "trec-covid", "webis-touche2020"],
        help="Datasets to process. dl19/dl20 use MS MARCO corpus + local TSV queries.",
    )
    parser.add_argument("--retrieval-dir", default="retrieval_runs", help="Directory with initial run files.")
    parser.add_argument(
        "--retrieval-prefix",
        default="run.rlhn_retriever_e5_base_15hn",
        help="Prefix before .<dataset>.txt in retrieval dir.",
    )
    parser.add_argument("--queries-dir", default="../../../data_files/queries/", help="Directory containing <dataset>-queries.tsv (dl19/dl20).")
    parser.add_argument("--out-dir", default="runs", help="Output directory for TREC run files.")
    parser.add_argument("--device", default="cuda", help="Device (cuda, cuda:0, cpu).")
    parser.add_argument("--text-maxlength", type=int, default=8192, help="Max tokens for input to the model.")
    parser.add_argument("--topk", type=int, default=100, help="Rerank top-K from initial retrieval.")
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Optional task instruction for generative Qwen rerankers. If omitted, uses the model default.",
    )
    args = parser.parse_args()

    models = parse_models(args.models)
    cache = {}

    for m in models:
        name = m["name"]
        bs = m["batch_size"]
        print(f"\n=== Model: {name} (batch_size={bs}) ===")
        model = RerankerModel(
            name,
            text_maxlength=args.text_maxlength,
            batch_size=bs,
            device=args.device,
        ).eval()

        for dataset in args.datasets:
            print(f"\n--- Dataset: {dataset} ---")
            corpus = load_corpus(dataset, cache)
            queries = load_queries(dataset, args.queries_dir, cache)

            run_path = os.path.join(args.retrieval_dir, f"{args.retrieval_prefix}.{dataset}.txt")
            if not os.path.exists(run_path):
                print(f"[WARN] Retrieval run not found: {run_path} — skipping {dataset}.")
                continue

            qids, pids = read_retrieval(run_path, topk=args.topk)
            if not qids:
                print(f"[WARN] No candidates in {run_path} (topk={args.topk}). Skipping {dataset}.")
                continue

            score_and_write(
                model=model,
                model_name=name,
                dataset=dataset,
                qids=qids,
                pids=pids,
                queries=queries,
                corpus=corpus,
                out_dir=args.out_dir,
                instruction=args.instruction,
            )
        del model
        
        print(f"\n=== Finished: {name} ===")


if __name__ == "__main__":
    main()
