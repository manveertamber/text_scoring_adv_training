import random
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union

from tqdm import tqdm

import torch
from torch.utils.data import Dataset


__all__ = ["RankingDataset", "PreferenceDataset"]


class JsonlOffsetIndex:
    def __init__(self, path: str):
        self.path = path
        self.offsets: Dict[Tuple[str, str], int] = self._build_index(path)
        self._fp = None

    @staticmethod
    def _to_key(d: Dict[str, Any]) -> Tuple[str, str]:
        return (str(d.get("sample_id")), str(d.get("docid")))

    def _build_index(self, path: str) -> Dict[Tuple[str, str], int]:
        idx: Dict[Tuple[str, str], int] = {}
        with open(path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                idx[self._to_key(data)] = pos
        return idx

    def _ensure_open(self):
        if self._fp is None:
            self._fp = open(self.path, "rb")

    def get_raw_line(self, key: Tuple[str, str]) -> Optional[Dict[str, Any]]:
        off = self.offsets.get(key)
        if off is None:
            return None
        self._ensure_open()
        self._fp.seek(off)
        line = self._fp.readline()
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def get_text_and_type(self, key: Tuple[str, str]) -> Tuple[Optional[str], Optional[str]]:
        rec = self.get_raw_line(key)
        if rec is None:
            return None, None
        return rec.get("text"), rec.get("injection_type")

    def get_score(self, key: Tuple[str, str]) -> Optional[float]:
        rec = self.get_raw_line(key)
        if rec is None:
            return None
        try:
            return float(rec.get("score"))
        except (TypeError, ValueError):
            return None

    def close(self):
        if self._fp is not None:
            try:
                self._fp.close()
            finally:
                self._fp = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None
        return state

    def __del__(self):
        self.close()



class RankingDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        paraphrased_path: Optional[str] = None,
        injected_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        teacher_scores_path: Optional[str] = None,
    ):
        self.samples: List[Dict[str, Any]] = []

        self.fetch_paraphrased = paraphrased_path is not None
        self.fetch_injected   = injected_path is not None
        self.fetch_teacher_scores    = teacher_scores_path is not None

        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f)):
                if max_samples is not None and i >= max_samples:
                    break

                data = json.loads(line)
                sample_id = str(data["sample_id"])
                
                def _create_passage_dict(p_data: Dict) -> Dict[str, Any]:
                    return {
                        "docid": str(p_data["docid"]),
                        "original": p_data["text"],
                        "paraphrased": None,
                        "injected": None,
                        "injection_type": None,
                        "teacher_score": None,
                    }

                pos_passage_dict = _create_passage_dict(data["positive_passage"])

                neg_passage_dicts = [_create_passage_dict(neg) for neg in data["negative_passages"]]


                self.samples.append({
                    "query": data["query"],
                    "sample_id": sample_id,
                    "positive_passage": pos_passage_dict,
                    "negative_passages": neg_passage_dicts,
                })

        rng = random.Random(42)
        rng.shuffle(self.samples)

       
        self._paraphrased_index: Optional[JsonlOffsetIndex] = (
            JsonlOffsetIndex(paraphrased_path) if self.fetch_paraphrased else None
        )
        self._injection_index: Optional[JsonlOffsetIndex] = (
            JsonlOffsetIndex(injected_path) if self.fetch_injected else None
        )
        self._teacher_index: Optional[JsonlOffsetIndex] = (
            JsonlOffsetIndex(teacher_scores_path) if self.fetch_teacher_scores else None
        )

    def _fill_aug_fields(self, sample_id: str, passage: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(passage)
        key = (sample_id, str(passage["docid"]))

        if self.fetch_paraphrased:
            text, _ = self._paraphrased_index.get_text_and_type(key)
            out["paraphrased"] = text

        if self.fetch_injected:
            text, inj_type = self._injection_index.get_text_and_type(key)
            out["injected"] = text
            out["injection_type"] = inj_type

        if self.fetch_teacher_scores and self._teacher_index is not None:
            out["teacher_score"] = self._teacher_index.get_score(key)

        return out


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, Dict[str, Optional[str]], List[Dict[str, Optional[str]]]]:
        sample = self.samples[idx]
        
        sid = sample["sample_id"]

        pos = self._fill_aug_fields(sid, sample["positive_passage"])
        negs = [self._fill_aug_fields(sid, nd) for nd in sample["negative_passages"]]

        return sample["query"], pos, negs

    @classmethod
    def build_train_dev(
        cls,
        train_jsonl_path: str,
        dev_jsonl_path: str,
        paraphrased_path: Optional[str] = None,
        injected_path: Optional[str] = None,
        max_train_samples: Optional[int] = None,
        max_dev_samples: Optional[int] = None,
        teacher_train_scores_path: Optional[str] = None,
        teacher_dev_scores_path: Optional[str] = None,
    ) -> Tuple["RankingDataset", "RankingDataset"]:

        shared_par_idx = JsonlOffsetIndex(paraphrased_path) if paraphrased_path else None
        shared_inj_idx = JsonlOffsetIndex(injected_path) if injected_path else None

        train_ds = cls(
            jsonl_path=train_jsonl_path,
            paraphrased_path=None,
            injected_path=None,
            max_samples=max_train_samples,
            teacher_scores_path=teacher_train_scores_path,
        )
        dev_ds = cls(
            jsonl_path=dev_jsonl_path,
            paraphrased_path=None,
            injected_path=None,
            max_samples=max_dev_samples,
            teacher_scores_path=teacher_dev_scores_path,
        )

        for ds in (train_ds, dev_ds):
            ds._paraphrased_index = shared_par_idx
            ds._injection_index = shared_inj_idx
            ds.fetch_paraphrased = shared_par_idx is not None
            ds.fetch_injected = shared_inj_idx is not None

        return train_ds, dev_ds


class PreferenceJsonlOffsetIndex(JsonlOffsetIndex):
    @staticmethod
    def _to_key(d: Dict[str, Any]) -> Tuple[str, str]:
        return (str(d.get("sample_id")), str(d.get("docid")))

    def get_text_and_type(self, key: Tuple[str, str]) -> Tuple[Optional[str], Optional[str]]:
        rec = self.get_raw_line(key)
        if rec is None:
            return None, None
        return rec.get("text"), rec.get("injection_type")


class PreferenceDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        paraphrased_path: Optional[str] = None,
        injected_path: Optional[str] = None,
        max_samples: Optional[int] = None
    ):
        self.samples: List[Dict[str, Any]] = []
        self.fetch_paraphrased = paraphrased_path is not None
        self.fetch_injected = injected_path is not None

        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc="Loading preference data")):
                if max_samples is not None and i >= max_samples:
                    break
                try:
                    data = json.loads(line)
                    if "sample_id" not in data or "chosen" not in data or "rejected" not in data:
                        continue
                    self.samples.append({
                        "sample_id": str(data["sample_id"]),
                        "chosen": data["chosen"]["messages"],
                        "rejected": data["rejected"]["messages"],
                        "weight": float(data["weight"]),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
        
        self._paraphrased_index = PreferenceJsonlOffsetIndex(paraphrased_path) if self.fetch_paraphrased else None
        self._injection_index = PreferenceJsonlOffsetIndex(injected_path) if self.fetch_injected else None

    def _get_full_response(self, sample_id: str, docid: str, original_messages: List[Dict]) -> Dict[str, Any]:

        key = (sample_id, docid)
        response_data = {
            "original": original_messages,
            "paraphrased": None,
            "injected": None,
            "injection_type": None,
        }

        if self.fetch_paraphrased and self._paraphrased_index:
            response_data["paraphrased"], _ = self._paraphrased_index.get_text_and_type(key)
        if self.fetch_injected and self._injection_index:
            content, inj_type = self._injection_index.get_text_and_type(key)
            response_data["injected"] = content
            response_data["injection_type"] = inj_type
        return response_data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any], Union[float, int]]:
        sample = self.samples[idx]
        sid = sample["sample_id"]
        chosen_data = self._get_full_response(sid, "chosen", sample["chosen"])
        rejected_data = self._get_full_response(sid, "rejected", sample["rejected"])
        weight = float(sample.get("weight", 1.0))
        return chosen_data, rejected_data, weight

