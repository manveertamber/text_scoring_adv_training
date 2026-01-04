import random
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from typing import List, Optional, Dict, Sequence
import csv
import string
from pathlib import Path
import json

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
STOPWORDS = set(stopwords.words('english'))

class RudimentaryInjectionsGenerator:
    def __init__(
        self,
        passages_file: str = None,
        injection_sentences_file: str = None,
        replacement_map_file: str     = None,
    ):

        self.passages = []
        self.injection_sentences = []

        if passages_file:
            with open(passages_file, 'r', encoding='utf-8') as f:
                tsv_reader = csv.DictReader(f, delimiter='\t')
                for row in tsv_reader:
                    passage_id = row['id']
                    passage_text = row['passage'].strip()
                    self.passages.append(passage_text)

        if injection_sentences_file:
            with open(injection_sentences_file, 'r', encoding='utf-8') as f:
                self.injection_sentences = [line.strip() for line in f]

        self.replacement_maps: Dict[str, Dict[str, Sequence[str]]] = {}
        if replacement_map_file:           
            p = Path(replacement_map_file)
            with p.open(encoding="utf-8") as fh:
                raw = json.load(fh)
            self.replacement_maps = {
                pid: {k.lower(): v for k, v in repls.items()}
                for pid, repls in raw.items()
            }

        self.LETTER_CHARS = string.ascii_letters
        self.DIGIT_CHARS  = string.digits
        self.PUNC_CHARS   = string.punctuation

    def sample_random_passage(self, counter_query=None) -> str:

        if not self.passages:
            raise ValueError("No passages available to sample from.")
        
        if counter_query:
            keywords = self.extract_keywords(counter_query)
            while True:        
                chosen_passage = random.choice(self.passages)
                for keyword in keywords:
                    if keyword.lower() not in chosen_passage.lower():
                        return chosen_passage

        else:
            return random.choice(self.passages)

    def add_typos(
        self,
        passage: str,
        *,
        n_typos: int = 1,
    ) -> str:
        t = list(passage)
        pos = random.sample(range(len(t)), k=min(n_typos, len(t)))
        pos.sort(reverse=True)
        for i in pos:
            pool, ops = self.LETTER_CHARS + self.DIGIT_CHARS + self.PUNC_CHARS, []
            if pool: ops += ["sub", "ins"]
            if len(t) > 1:
                ops.append("del")
                if i < len(t) - 1 and t[i] != t[i + 1]:
                    ops.append("swap")
            if not ops: continue
            op = random.choice(ops)
            if op == "sub":
                c = t[i]
                while c == t[i]:
                    c = random.choice(pool)
                t[i] = c
            elif op == "ins":
                t.insert(i, random.choice(pool))
            elif op == "del":
                t.pop(i)
            elif op == "swap":
                t[i], t[i + 1] = t[i + 1], t[i]
        return "".join(t)
    
    
    def swap_tokens(
        self,
        passage: str,
        *,
        pid: str,
        n_swaps: int = 1,
    ) -> str:
        repl_map = self.replacement_maps.get(pid)
        if not repl_map:
            return passage

        tokens = [(m.group(0), m.span()) for m in re.finditer(r"\b\w+\b", passage)]
        eligible = [(tok, span) for tok, span in tokens if tok.lower() in repl_map]
        if not eligible:
            return passage
        random.shuffle(eligible)

        chars = list(passage)
        made  = 0
        for tok, (s, e) in eligible:
            if made >= n_swaps:
                break
            new_tok = random.choice(repl_map[tok.lower()])
            if tok.istitle():   new_tok = new_tok.capitalize()
            elif tok.isupper(): new_tok = new_tok.upper()
            chars[s:e] = list(new_tok)
            delta = len(new_tok) - (e - s)
            if delta:
                for j in range(len(eligible)):
                    t, (ss, ee) = eligible[j]
                    if ss > s:
                        eligible[j] = (t, (ss + delta, ee + delta))
            made += 1
        return "".join(chars)

    def extract_keywords(self, text: str) -> List[str]:
        words = word_tokenize(text.lower())
        keywords = [word for word in words if (word not in STOPWORDS) and (word.isalpha())]
        return keywords

    def get_random_sentences(self, num_sentences: int = 1) -> List[str]:
        return random.sample(self.injection_sentences, k=num_sentences)

    def form_text_from_random_words(self, num_words: int = 100) -> str:
        text_words = []
        while len(text_words) < num_words:
            sentence = random.choice(self.injection_sentences)
            words = word_tokenize(sentence)
            if words:
                word = random.choice(words)
                text_words.append(word)
        return ' '.join(text_words[:num_words])
    
    def prepend_query_to_passage(self, query: str, passage: str) -> str:
        if not query:
            raise ValueError("No query given. Please provide a query.")
        return f"Query: {query}\n{passage}"

    def inject_sentences(
        self, text: str, inject_sentences: List[str], location: Optional[str] = 'random', allow_modification=False
    ) -> str:

        valid_locations = {'start', 'middle', 'end', 'random'}
        if location not in valid_locations:
            raise ValueError("Invalid location. Choose from 'start', 'middle', 'end', or 'random'.")

        if location == 'random':
            location = random.choice(['start', 'middle', 'end'])
        
        def modify_sentence(sentence):
            """Randomly remove period or uncapitalize the first word."""
            if random.random() < 0.5:
                sentence = sentence.rstrip('.')
            
            if random.random() < 0.5:
                words = sentence.split()
                if words:
                    first_word = words[0]
                    if first_word[0].isupper() and first_word[1:].islower():
                        words[0] = first_word[0].lower() + first_word[1:]
                        sentence = ' '.join(words)
            return sentence

        if allow_modification:
            inject_sentences = [modify_sentence(sentence) for sentence in inject_sentences]

        if location == 'start':
            modified_text = ' '.join(inject_sentences) + ' ' + text
        elif location == 'end':
            modified_text = text + ' ' + ' '.join(inject_sentences)
        elif location == 'middle':
            words = text.split()
            for sentence in inject_sentences:
                if len(words) > 1:
                    insert_index = random.randint(1, len(words) - 1)
                else:
                    insert_index = 0
                words.insert(insert_index, sentence)
            modified_text = ' '.join(words)

        return modified_text

    def remove_sentence(self, text: str) -> str:
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return text
        if random.randint(0, 1):
            del sentences[0]
        else:
            del sentences[-1]
        return ' '.join(sentences)

    def inject_query_keywords_into_passage(
        self, passage: str, query: str, location: Optional[str] = 'random', num_injections: int = 1, shorten_passage: bool = False,
    ) -> str:
        if shorten_passage:
            passage = self.remove_sentence(passage)

        query_keywords = self.extract_keywords(query)
        
        injected_passage = passage
        for _ in range(num_injections):
            injected_passage = self.inject_sentences(injected_passage, query_keywords, location)
        
        return injected_passage

    def inject_query_into_passage(
        self, passage: str, query: str, location: Optional[str] = 'random', num_injections: int = 1, shorten_passage: bool = False,
    ) -> str:
        if shorten_passage:
            passage = self.remove_sentence(passage)
        injected_passage = passage
        for _ in range(num_injections):
            injected_passage = self.inject_sentences(injected_passage, [query], location)

        return injected_passage

    def inject_random_sentences_into_text(
        self, text: str, num_sentences: int = 1, location: Optional[str] = 'random', shorten_passage: bool = False,
    ) -> str:
        if shorten_passage:
            text = self.remove_sentence(text)

        while True:
            inject_sentences = self.get_random_sentences(num_sentences)
           
            text_lower = text.lower()
            if all(sent.lower().rstrip('.') not in text_lower for sent in inject_sentences):
                break

        return self.inject_sentences(text, inject_sentences, location, allow_modification=True)

    def inject_random_sentences_into_passage(
        self, passage: str, num_sentences: int = 1, location: Optional[str] = 'random', shorten_passage: bool = False,
    ) -> str:
        return self.inject_random_sentences_into_text(passage, num_sentences, location, shorten_passage)
