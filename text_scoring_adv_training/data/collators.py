"""
Collator classes for batching and tokenization.
"""

import random
import string
from typing import List, Tuple, Dict, Any, Optional

import torch
from transformers import AutoTokenizer

__all__ = ["RetrieverCollator", "RerankerCollator", "PreferenceCollator"]

LETTER_CHARS = string.ascii_letters
DIGIT_CHARS = string.digits
PUNC_CHARS = string.punctuation

def _apply_rudimentary_manipulations(text: str) -> str:
    if not text.strip(): return text
    
    if random.choice([True, False]):
        words = text.split(' ')
        if not words: 
            return text
        i = random.randint(0, len(words)-1)
        ops = ["repeat"]
        if len(words) > 1: ops.append("delete")
        if i < len(words)-1 and words[i] != words[i+1]: ops.append("swap")
        op = random.choice(ops)
        if   op == "delete": words.pop(i)
        elif op == "swap":   words[i], words[i+1] = words[i+1], words[i]
        else:                words.insert(i+1, words[i])
        return " ".join(words)
    
    t = list(text)
    if not t: return text
    i = random.randint(0, len(t)-1)
    pool = LETTER_CHARS + DIGIT_CHARS + PUNC_CHARS
    ops = ["sub","ins"]
    if len(t)>1: ops.append("del")
    if i < len(t)-1 and t[i] != t[i+1]: ops.append("swap")
    op = random.choice(ops)
    if   op == "sub":
        c = random.choice(pool)
        while c == t[i]: c = random.choice(pool)
        t[i] = c
    elif op == "ins": t.insert(i, random.choice(pool))
    elif op == "del": t.pop(i)
    else:             t[i], t[i+1] = t[i+1], t[i]
    return "".join(t)

def _get_adversarial_passage(passage_dict, choices):
    for choice in random.sample(choices, k=len(choices)) + ["identity"]:
        if choice == "paraphrase" and passage_dict.get("paraphrased"):
            return passage_dict["paraphrased"], "paraphrase"
        if choice == "injection" and passage_dict.get("injected"):
            t = passage_dict.get("injection_type", "")
            return passage_dict["injected"], f"injection_{'query' if 'query' in str(t).lower() else 'sentence'}"
        if choice == "rudimentary":
            return _apply_rudimentary_manipulations(passage_dict["original"]), "rudimentary"
        if choice == "hotflip":
            return passage_dict["original"], "hotflip"
        if choice == "identity":
            return passage_dict["original"], "none"
        

class RetrieverCollator:
    """
    Collator for retrieval models that handles tokenization and batching.
    
    It tokenizes original and adversarial texts and returns them in separate,
    keyed dictionary entries for direct use. Adversarial manipulations
    are chosen on a per-passage basis.
    
    Parameters are the same as the original script.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        query_max_length: int = 512,
        passage_max_length: int = 512,
        negatives_per_query: int = 10,
        sample_negatives: bool = True,
        use_paraphrased: bool = False,
        use_injected: bool = False,
        use_rudimentary_manipulations: bool = False,
        use_hotflip_swaps: bool = False,
        emit_teacher_scores: bool = False,
    ):
        self.tokenizer = tokenizer
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.negatives_per_query = negatives_per_query
        self.sample_negatives = sample_negatives
        self.use_paraphrased = use_paraphrased
        self.use_injected = use_injected
        self.use_rudimentary_manipulations = use_rudimentary_manipulations
        self.use_hotflip_swaps = use_hotflip_swaps
        self.emit_teacher_scores = emit_teacher_scores

    def _enabled_choices(self) -> List[str]:
        choices = []
        if self.use_paraphrased:
            choices.append("paraphrase")
        if self.use_injected:
            choices.append("injection")
        if self.use_rudimentary_manipulations:
            choices.append("rudimentary")
        if self.use_hotflip_swaps:
            choices.append("hotflip")
        return choices
    
    def _encode(self, texts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )

    def __call__(self, batch: List[Tuple]) -> Dict[str, Any]:

        possible_manipulations = self._enabled_choices()
        adv_passage_enabled = bool(possible_manipulations)
        
        queries, pos_passage_dicts, neg_passage_dict_lists = zip(*batch)
        
        original_queries = [self.query_prefix + q for q in queries]
        original_pos_passages = [self.passage_prefix + p['original'] for p in pos_passage_dicts]
        
        flat_original_neg_passages = []
        sampled_neg_dicts_per_sample = []
        for neg_list in neg_passage_dict_lists:
            if self.sample_negatives and len(neg_list) > self.negatives_per_query:
                chosen_negs = random.sample(neg_list, self.negatives_per_query)
            else:
                chosen_negs = neg_list[:self.negatives_per_query]
            sampled_neg_dicts_per_sample.append(chosen_negs)
            flat_original_neg_passages.extend([self.passage_prefix + d['original'] for d in chosen_negs])

        adv_pos_passages, adv_neg_passages = [], []
        manipulation_info = []

        teacher_rows: List[List[float]] = []

        for i in range(len(queries)):
            sample_info = {"query": None, "pos": None, "negs": []}
            row_scores: List[Optional[float]] = []

            pos_dict = pos_passage_dicts[i]

            if self.emit_teacher_scores:
                row_scores.append(pos_dict.get("teacher_score", None))

            if adv_passage_enabled:
                adv_pos_text, adv_pos_type = _get_adversarial_passage(pos_dict, possible_manipulations)
                adv_pos_passages.append(self.passage_prefix + adv_pos_text)
                sample_info["pos"] = adv_pos_type
            else:
                sample_info["pos"] = "none"

            for neg_dict in sampled_neg_dicts_per_sample[i]:
                if adv_passage_enabled:
                    adv_neg_text, adv_neg_type = _get_adversarial_passage(neg_dict, possible_manipulations)
                    adv_neg_passages.append(self.passage_prefix + adv_neg_text)
                    sample_info["negs"].append(adv_neg_type)
                else:
                    sample_info["negs"].append("none")

                if self.emit_teacher_scores:
                    row_scores.append(neg_dict.get("teacher_score", None))

            manipulation_info.append(sample_info)

            if self.emit_teacher_scores:
                teacher_rows.append([float(s) if s is not None else 0.0 for s in row_scores])

        all_texts, boundaries = [], {}
        def add_to_batch(key, texts_to_add):
            start = len(all_texts)
            if texts_to_add: all_texts.extend(texts_to_add)
            boundaries[key] = (start, len(all_texts))

        add_to_batch('query', original_queries)
        add_to_batch('pos_passage', original_pos_passages)
        add_to_batch('neg_passages', flat_original_neg_passages)
        add_to_batch('adversarial_pos_passage', adv_pos_passages)
        add_to_batch('adversarial_negative_passages', adv_neg_passages)

        if not all_texts:
            return {key: None for key in boundaries} | {'manipulation_info': []}

        max_len = max(self.query_max_length, self.passage_max_length)
        enc = self._encode(all_texts, max_len)

        result_batch = {}
        for key, (start, end) in boundaries.items():
            result_batch[key] = {'input_ids': enc['input_ids'][start:end], 'attention_mask': enc['attention_mask'][start:end]} if start < end else None
        
        result_batch['manipulation_info'] = manipulation_info

        if self.emit_teacher_scores and teacher_rows:
            result_batch['teacher_scores'] = torch.tensor(teacher_rows, dtype=torch.float32)

        return result_batch
    

class RerankerCollator:
    """
    Collator for reranking models that creates query-document pairs.
    Adversarial manipulations are chosen on a per-passage basis.
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        negatives_per_query: int = 10,
        sample_negatives: bool = True,
        padding_side: str = "left",
        use_paraphrased: bool = False,
        use_injected: bool = False,
        use_rudimentary_manipulations: bool = False,
        use_hotflip_swaps: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.negatives_per_query = negatives_per_query
        self.sample_negatives = sample_negatives
        self.use_paraphrased = use_paraphrased
        self.use_injected = use_injected
        self.use_rudimentary_manipulations = use_rudimentary_manipulations
        self.use_hotflip_swaps = use_hotflip_swaps

        self.tokenizer.padding_side = padding_side

    def _create_prompt(self, query: str, document: str) -> Tuple[str, int]:
        prefix = f"How relevant is the following document to the query?\n\nQuery: {query}\n\nDocument: "

        full = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prefix + document}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

        prefix_only = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prefix}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

        start_ids = self.tokenizer(prefix_only, add_special_tokens=False,
                                truncation=True, max_length=self.max_length)["input_ids"]
        return full, len(start_ids) - 1

    def _enabled_choices(self) -> List[str]:
        choices = []
        if self.use_paraphrased:
            choices.append("paraphrase")
        if self.use_injected:
            choices.append("injection")
        if self.use_rudimentary_manipulations:
            choices.append("rudimentary")
        if self.use_hotflip_swaps:
            choices.append("hotflip")
        return choices
    
    def __call__(self, batch: List[Tuple]) -> Dict[str, Any]:
        possible_manipulations = self._enabled_choices()
        adv_passage_enabled = bool(possible_manipulations)

        queries, pos_passage_dicts, neg_passage_dict_lists = zip(*batch)
        
        pos_prompts, neg_prompts, adv_pos_prompts, adv_neg_prompts = [], [], [], []
        pos_doc_start_indices, neg_doc_start_indices = [], []
        manipulation_info = []

        for i, query in enumerate(queries):
            sample_info = {"pos": "none", "negs": []}

            pos_dict = pos_passage_dicts[i]
            prompt, start_idx = self._create_prompt(query, pos_dict['original'])
            pos_prompts.append(prompt)
            pos_doc_start_indices.append(start_idx)
            
            if adv_passage_enabled:
                adv_passage, adv_tag = _get_adversarial_passage(pos_dict, possible_manipulations)
                adv_prompt, adv_start_idx = self._create_prompt(query, adv_passage)
                adv_pos_prompts.append(adv_prompt)
                sample_info["pos"] = adv_tag

            neg_dict_list = neg_passage_dict_lists[i]
            assert len(neg_dict_list) >= self.negatives_per_query, f"Need at least {self.negatives_per_query} negs, got {len(neg_dict_list)}"
            chosen_negs = random.sample(neg_dict_list, self.negatives_per_query) if self.sample_negatives and len(neg_dict_list) > self.negatives_per_query else neg_dict_list[:self.negatives_per_query]
            
            for neg_dict in chosen_negs:
                neg_prompt, neg_start_idx = self._create_prompt(query, neg_dict['original'])
                neg_prompts.append(neg_prompt)
                neg_doc_start_indices.append(neg_start_idx)

                if adv_passage_enabled:
                    adv_neg_passage, adv_neg_tag = _get_adversarial_passage(neg_dict, possible_manipulations)
                    adv_prompt, _ = self._create_prompt(query, adv_neg_passage)
                    adv_neg_prompts.append(adv_prompt)
                    sample_info["negs"].append(adv_neg_tag)
                else:
                    sample_info["negs"].append("none")

            manipulation_info.append(sample_info)

        all_prompts, boundaries = [], {}
        def add_to_batch(key, prompts_to_add):
            start = len(all_prompts)
            if prompts_to_add: all_prompts.extend(prompts_to_add)
            boundaries[key] = (start, len(all_prompts))
        
        add_to_batch('positive_pairs', pos_prompts)
        add_to_batch('negative_pairs', neg_prompts)
        add_to_batch('adversarial_positive_pairs', adv_pos_prompts)
        add_to_batch('adversarial_negative_pairs', adv_neg_prompts)

        if not all_prompts:
            return {key: None for key in boundaries} | {'manipulation_info': []}
            
        enc = self.tokenizer(all_prompts, padding=True, pad_to_multiple_of=8, truncation=True, max_length=self.max_length, return_tensors="pt", add_special_tokens=False,)
        
        result_batch = {}
        for key, (start, end) in boundaries.items():
            result_batch[key] = {'input_ids': enc['input_ids'][start:end], 'attention_mask': enc['attention_mask'][start:end]} if start < end else None

        result_batch['positive_doc_start_indices'] = pos_doc_start_indices[:len(pos_prompts)]
        result_batch['negative_doc_start_indices'] = neg_doc_start_indices[:len(neg_prompts)]
        
        result_batch['manipulation_info'] = manipulation_info
        return result_batch


class PreferenceCollator:
    """
    Collator for preference reward model training
    For each sample, it renders the 'chosen' and 'rejected' conversations with the
    tokenizer's chat template and (optionally) produces adversarial variants by
    replacing the final assistant message with a paraphrase/injection or applying
    rudimentary character/word edits.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 4096,
        padding_side: str = "left",
        pad_to_multiple_of: int = 8,
        use_paraphrased: bool = False,
        use_injected: bool = False,
        use_rudimentary_manipulations: bool = False,
        use_hotflip_swaps: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

        self.use_paraphrased = use_paraphrased
        self.use_injected = use_injected
        self.use_rudimentary_manipulations = use_rudimentary_manipulations
        self.use_hotflip_swaps = use_hotflip_swaps

        self.tokenizer.padding_side = padding_side


    def _enabled_choices(self) -> List[str]:
        choices = []
        if self.use_paraphrased:
            choices.append("paraphrase")
        if self.use_injected:
            choices.append("injection")
        if self.use_rudimentary_manipulations:
            choices.append("rudimentary")
        if self.use_hotflip_swaps:
            choices.append("hotflip")
        return choices

    def _render_and_get_idx(self, messages: List[Dict[str, Any]]) -> Tuple[str, int]:
        ctx_msgs = messages[:-1]
        full = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        ctx = self.tokenizer.apply_chat_template(
            ctx_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        start_ids = self.tokenizer(ctx, add_special_tokens=False,
                                truncation=True, max_length=self.max_length)["input_ids"]
        return full, len(start_ids) - 1
        

    @staticmethod
    def _copy_with_last_assistant(messages: List[Dict[str, Any]], new_text: str) -> List[Dict[str, Any]]:
        msgs = [dict(m) for m in messages]
        
        assert msgs[-1].get("role") == "assistant"
        msgs[-1]["content"] = new_text
        return msgs

    def _adv_for_response(self, resp_data: Dict[str, Any], choices: List[str]) -> Tuple[str, str, int]:
        messages = resp_data["original"]

        for choice in random.sample(choices, k=len(choices)) + ["identity"]:
            if choice == "paraphrase" and resp_data.get("paraphrased"):
                msgs = self._copy_with_last_assistant(messages, resp_data["paraphrased"])
                text, idx = self._render_and_get_idx(msgs)
                return text, "paraphrase", idx

            if choice == "injection" and resp_data.get("injected"):
                inj_type = str(resp_data.get("injection_type", "")).lower()
                tag = f"injection_{'query' if 'query' in inj_type else 'sentence'}"
                msgs = self._copy_with_last_assistant(messages, resp_data["injected"])
                text, idx = self._render_and_get_idx(msgs)
                return text, tag, idx

            if choice == "rudimentary":
                assert messages[-1].get("role") == "assistant"
                base_text = messages[-1].get("content", "")
                mutated = _apply_rudimentary_manipulations(base_text)
                msgs = self._copy_with_last_assistant(messages, mutated)
                text, idx = self._render_and_get_idx(msgs)
                return text, "rudimentary", idx

            if choice == "hotflip":
                text, idx = self._render_and_get_idx(messages)
                return text, "hotflip", idx

            if choice == "identity":
                text, idx = self._render_and_get_idx(messages)
                return text, "none", idx

        text, idx = self._render_and_get_idx(messages)
        return text, "none", idx


    def __call__(self, batch: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
        possible = self._enabled_choices()
        adv_enabled = bool(possible)

        chosen_texts, rejected_texts = [], []
        adv_chosen_texts, adv_rejected_texts = [], []
        chosen_indices, rejected_indices, adv_chosen_indices, adv_rejected_indices = [], [], [], []
        manipulation_info = []
        weights: List[float] = []

        for sample in batch:
            chosen_data, rejected_data, w = sample
            w = float(w)
            
            weights.append(w)
            c_text, c_idx = self._render_and_get_idx(chosen_data["original"])
            r_text, r_idx = self._render_and_get_idx(rejected_data["original"])
            chosen_texts.append(c_text)
            chosen_indices.append(c_idx)
            rejected_texts.append(r_text)
            rejected_indices.append(r_idx)

            info = {"chosen": "none", "rejected": "none"}

            if adv_enabled:
                adv_c, tag_c, adv_c_idx = self._adv_for_response(chosen_data, possible)
                adv_r, tag_r, adv_r_idx = self._adv_for_response(rejected_data, possible)
                adv_chosen_texts.append(adv_c)
                adv_chosen_indices.append(adv_c_idx)
                adv_rejected_texts.append(adv_r)
                adv_rejected_indices.append(adv_r_idx)
                info["chosen"] = tag_c
                info["rejected"] = tag_r

            manipulation_info.append(info)

        all_texts, boundaries = [], {}

        def add_block(key: str, texts: List[str]):
            start = len(all_texts)
            if texts:
                all_texts.extend(texts)
            boundaries[key] = (start, len(all_texts))

        add_block("chosen", chosen_texts)
        add_block("rejected", rejected_texts)
        add_block("adversarial_chosen", adv_chosen_texts)
        add_block("adversarial_rejected", adv_rejected_texts)

        if not all_texts:
            return {k: None for k in boundaries} | {"manipulation_info": [], "weights": torch.tensor([], dtype=torch.float32)}

        enc = self.tokenizer(
            all_texts,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        out: Dict[str, Any] = {}
        for key, (s, e) in boundaries.items():
            out[key] = (
                {"input_ids": enc["input_ids"][s:e], "attention_mask": enc["attention_mask"][s:e]}
                if s < e
                else None
            )

        if chosen_indices:
            out["chosen_start_indices"] = chosen_indices
        if rejected_indices:
            out["rejected_start_indices"] = rejected_indices
        if adv_chosen_indices:
            out["adversarial_chosen_start_indices"] = adv_chosen_indices
        if adv_rejected_indices:
            out["adversarial_rejected_start_indices"] = adv_rejected_indices

        out["manipulation_info"] = manipulation_info
        out["weights"] = torch.tensor(weights, dtype=torch.float32)
        
        return out