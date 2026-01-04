import json, os, argparse
from transformers import AutoModel, AutoTokenizer
import torch, numpy as np, faiss
from tqdm import tqdm
from datasets import load_dataset
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class EmbeddingModelWrapper(torch.nn.Module):
    def __init__(self, model_name, normalize=True, pooling="cls"):
        super().__init__()
        self.model   = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.normalize, self.pooling = normalize, pooling

    def forward(self, ids, mask):
        out = self.model(ids, mask)
        if   self.pooling == "mean": embs = self.average_pool(out.last_hidden_state, mask)
        elif self.pooling == "LLM":  embs = self.last_token_pool(out.last_hidden_state, mask)
        else:                        embs = out[0][:, 0]              # CLS

        return torch.nn.functional.normalize(embs, p=2, dim=1) if self.normalize else embs

    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class PassageDataset(torch.utils.data.Dataset):
    def __init__(self, passages, prefix):
        self.passages, self.prefix = passages, prefix

    def __len__(self):  return len(self.passages)

    def __getitem__(self, idx):
        item  = self.passages[idx]
        return {"docid": str(item["docid"]),
                "text":  f"{self.prefix}{item['text']}"}

class Collator:
    def __init__(self, tokenizer, text_maxlength):  self.tok, self.maxlen = tokenizer, text_maxlength
    def __call__(self, batch):
        texts   = [b["text"] for b in batch]
        docids  = [b["docid"] for b in batch]
        toks    = self.tok(texts, max_length=self.maxlen, padding=True, truncation=True, return_tensors="pt")
        return toks["input_ids"], toks["attention_mask"].bool(), docids

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name",     required=True)
    ap.add_argument("--dataset",        required=True, help="Name of corpus to encode")
    ap.add_argument("--normalize",      action="store_true")
    ap.add_argument("--pooling",        choices=["cls", "mean", "LLM"], default="cls")
    ap.add_argument("--batch_size",     type=int, default=2000)
    ap.add_argument("--dim",            type=int, default=768)
    ap.add_argument("--max_len",        type=int, default=512)
    ap.add_argument("--query_prefix",   type=str, default="query: ")
    ap.add_argument("--passage_prefix", type=str, default="passage: ")
    args = ap.parse_args()

    ds_name = args.dataset.lower()
    prefix  = args.query_prefix if ds_name == "quora" or "cqadupstack" in ds_name else args.passage_prefix
    print(f"Using prefix '{prefix.strip()}' for dataset '{args.dataset}'")

    tokenizer  = get_tokenizer(args.model_name)
    model      = EmbeddingModelWrapper(args.model_name, args.normalize, args.pooling).cuda().eval()

    corpus = load_dataset(f"BeIR/{args.dataset}", "corpus")["corpus"]
    passages = [{"docid": str(row["_id"]),
                 "text":  str(row["text"]).replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()}
                for row in tqdm(corpus, desc="Reading corpus")]

    dataset   = PassageDataset(passages, prefix)
    collate   = Collator(tokenizer, text_maxlength=args.max_len)
    loader    = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            num_workers=8, collate_fn=collate,
                                            shuffle=False, drop_last=False)

    index  = faiss.IndexFlatIP(args.dim)
    outdir = f"indices/{os.path.basename(args.model_name)}_{args.dataset}_index"
    os.makedirs(outdir, exist_ok=True)

    with torch.no_grad(), open(f"{outdir}/docid", "w", encoding="utf-8") as did_f:
        for ids, mask, docids in tqdm(loader, total=len(loader), desc="Encoding"):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                embs = model(ids.cuda(), mask.cuda()).cpu().numpy().astype(np.float32)
            index.add(embs)
            did_f.writelines(f"{d}\n" for d in docids)

    faiss.write_index(index, f"{outdir}/index")
    print("✓ Finished indexing")

if __name__ == "__main__":
    main()