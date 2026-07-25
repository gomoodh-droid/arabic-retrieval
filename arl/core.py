"""نواة مختبر الاسترجاع العربي — Arabic Retrieval Lab Core"""
from __future__ import annotations
import math
import os
import re
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- التطبيع والتقطيع
ALL_NORM_OPS = [
    "nfkc",
    "strip_diacritics",
    "alef",
    "yaa",
    "taa",
    "waw",
    "digits",
    "punct",
    "collapse_ws",
    "stem_light"
]
CHUNK_MODES = ["chars", "sentences", "discourse"]


DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")

PUNCT = re.compile(r"[^\w\s\u0600-\u06FF]")
DIGITS = str.maketrans("0123456789٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

DISCOURSE = [
    " ولذلك ", " وبالتالي ", " ولكن ", " غير أن ", " أما ", " ثم ", " إضافة إلى ",
    " علاوة على ", " بناء على ", " من جهة ", " في حين ", " الأمر الذي "
]

STATS = {"built": 0, "reused": 0}

def stem_word(w: str) -> str:
    if len(w) <= 3:
        return w
    # strip prefixes
    for p in ["بال", "فال", "كال", "وال", "ال"]:
        if w.startswith(p) and len(w) - len(p) >= 3:
            w = w[len(p):]
            break
    for p in ["و", "ب", "ف", "ل"]:
        if w.startswith(p) and len(w) - len(p) >= 3:
            w = w[len(p):]
            break
    # strip suffixes
    for s in ["ها", "هم", "هن", "كم", "كن", "نا", "ين", "ون", "ات"]:
        if w.endswith(s) and len(w) - len(s) >= 3:
            w = w[:-len(s)]
            break
    for s in ["ه", "ة", "ي"]:
        if w.endswith(s) and len(w) - len(s) >= 3:
            w = w[:-len(s)]
            break
    return w

def normalize(text: str, ops: list[str]) -> str:
    s = text
    for op in ops:
        if op == "nfkc":
            import unicodedata
            s = unicodedata.normalize("NFKC", s)
        elif op == "strip_diacritics":
            s = DIACRITICS.sub("", s)
        elif op == "alef":
            s = re.sub(r"[إأآء]", "ا", s)
        elif op == "yaa":
            s = re.sub(r"ى", "ي", s)
        elif op == "taa":
            s = re.sub(r"ة", "ه", s)
        elif op == "waw":
            s = re.sub(r"ؤ", "و", s)
        elif op == "digits":
            s = s.translate(DIGITS)
        elif op == "punct":
            s = PUNCT.sub(" ", s)
        elif op == "collapse_ws":
            s = re.sub(r"\s+", " ", s).strip()
        elif op == "stem_light":
            words = s.split()
            s = " ".join(stem_word(w) for w in words)
    return s

def chunk(text: str, mode: str = "chars", size: int = 200, overlap: int = 40) -> list[str]:
    if not text:
        return []
    if mode == "chars":
        step = max(1, size - overlap)
        return [text[i:i + size] for i in range(0, len(text), step)]
    elif mode == "sentences":
        sentences = [s.strip() for s in re.split(r"[.!\?\n]+", text) if s.strip()]
        if not sentences:
            return [text]
        out, cur = [], ""
        for s in sentences:
            if len(cur) + len(s) + 1 <= size:
                cur = (cur + " " + s).strip()
            else:
                if cur:
                    out.append(cur)
                cur = s
        if cur:
            out.append(cur)
        if overlap > 0 and len(out) > 1:
            # add overlap text
            ov_out = []
            for i in range(len(out)):
                if i > 0:
                    ov = out[i-1][-overlap:] + " " + out[i]
                    ov_out.append(ov)
                else:
                    ov_out.append(out[i])
            return ov_out
        return out
    elif mode == "discourse":
        parts = [text]
        for conn in DISCOURSE:
            new_parts = []
            for p in parts:
                if len(p) > size:
                    splits = p.split(conn)
                    accum = ""
                    for idx, sp in enumerate(splits):
                        token = (conn if idx > 0 else "") + sp
                        if len(accum) + len(token) <= size:
                            accum += token
                        else:
                            if accum:
                                new_parts.append(accum)
                            accum = token
                    if accum:
                        new_parts.append(accum)
                else:
                    new_parts.append(p)
            parts = new_parts
        return [p.strip() for p in parts if p.strip()]
    return [text]

def tokens(text: str, ngram: int = 0) -> list[str]:
    words = text.split()
    if ngram == 0:
        return words
    res = list(words)
    for w in words:
        if len(w) >= ngram:
            for i in range(len(w) - ngram + 1):
                res.append(w[i:i + ngram])
    return res

# ---------------------------------------------------------------- فضاء الاستراتيجيات
@dataclass
class Strategy:
    name: str = "default"
    norm_ops: list[str] = field(default_factory=lambda: ["strip_diacritics", "alef", "yaa", "taa", "collapse_ws"])
    chunk_mode: str = "discourse"
    chunk_size: int = 250
    overlap: int = 0
    ngram: int = 3

    def key(self) -> str:
        ops = "-".join(sorted(self.norm_ops))
        return f"{self.name}_{ops}_{self.chunk_mode}_{self.chunk_size}_{self.overlap}_{self.ngram}"

    def text_layer_key(self) -> str:
        ops = "-".join(sorted(self.norm_ops))
        return f"{ops}_{self.chunk_mode}_{self.chunk_size}_{self.overlap}"

@dataclass
class TextLayer:
    chunks: list[str]
    doc_ids: list[str]
    reused: bool = False

def build_text_layer(corpus: list[dict], s: Strategy, use_cache: bool = True) -> TextLayer:
    key = s.text_layer_key()
    cache_dir = Path(".arl_cache")
    cache_path = cache_dir / f"{key}.json"
    
    if use_cache and cache_path.exists():
        try:
            d = json.loads(cache_path.read_text("utf-8"))
            STATS["reused"] += 1
            return TextLayer(chunks=d["chunks"], doc_ids=d["doc_ids"], reused=True)
        except Exception:
            pass
            
    chunks, doc_ids = [], []
    for doc in corpus:
        text = normalize(doc.get("text", ""), s.norm_ops)
        parts = chunk(text, s.chunk_mode, s.chunk_size, s.overlap)
        for p in parts:
            chunks.append(p)
            doc_ids.append(doc["id"])
            
    STATS["built"] += 1
    if use_cache:
        cache_dir.mkdir(exist_ok=True)
        cache_path.write_text(json.dumps({"chunks": chunks, "doc_ids": doc_ids}, ensure_ascii=False), "utf-8")
        
    return TextLayer(chunks=chunks, doc_ids=doc_ids, reused=False)

# ---------------------------------------------------------------- BM25
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.N = len(docs)
        self.avg = sum(len(d) for d in docs) / max(1, self.N)
        self.tf = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, q: list[str], i: int) -> float:
        s, dl = 0.0, len(self.docs[i])
        for t in q:
            f = self.tf[i].get(t, 0)
            if not f:
                continue
            s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return s

    def scores(self, q: list[str]) -> dict[int, float]:
        return {i: self.score(q, i) for i in range(self.N)}

# ---------------------------------------------------------------- المسطرة الحتمية
@dataclass
class Result:
    score: float
    recall_at_k: float
    mrr: float
    k: int
    failures: list[dict] = field(default_factory=list)
    per_question: list[dict] = field(default_factory=list)
    n_chunks: int = 0
    layer_reused: bool = False
    strategy: Strategy | None = None

    def label(self) -> str:
        return f"recall@{self.k}"

    def hits(self) -> list[float]:
        return [q["hit"] for q in self.per_question]

    def rrs(self) -> list[float]:
        return [q["rr"] for q in self.per_question]

def load_data(data_dir: str = None) -> tuple[list[dict], list[dict]]:
    if not data_dir:
        data_dir = str(Path(__file__).parent / "data")
    c_raw = json.loads(Path(data_dir, "corpus.json").read_text("utf-8"))
    q_raw = json.loads(Path(data_dir, "questions.json").read_text("utf-8"))
    corpus = c_raw.get("passages", c_raw) if isinstance(c_raw, dict) else c_raw
    questions = q_raw.get("questions", q_raw) if isinstance(q_raw, dict) else q_raw
    return corpus, questions


def evaluate(s: Strategy, corpus: list[dict] = None, questions: list[dict] = None, k: int = 3, use_cache: bool = True) -> Result:
    if corpus is None or questions is None:
        corpus, questions = load_data()

    layer = build_text_layer(corpus, s, use_cache=use_cache)
    docs = [tokens(c, s.ngram) for c in layer.chunks]
    bm = BM25(docs)

    hits, rrs = [], []
    failures = []
    per_question = []

    for idx, q in enumerate(questions):
        qt = tokens(normalize(q["q"], s.norm_ops), s.ngram)
        sc = [bm.score(qt, i) for i in range(len(docs))]

        # Tying breaker: sort by score desc, then index asc
        ranked_indices = sorted(range(len(docs)), key=lambda i: (sc[i], -i), reverse=True)
        top = [layer.doc_ids[i] for i in ranked_indices[:k]]

        gold_set = set(q.get("gold", [q.get("target_id")]))
        hit = 1.0 if any(g in top for g in gold_set) else 0.0
        hits.append(hit)

        rr = 0.0
        for rank, doc_id in enumerate(top, 1):
            if doc_id in gold_set:
                rr = 1.0 / rank
                break
        rrs.append(rr)

        q_id = q.get("id", f"q_{idx}")
        per_question.append({"id": q_id, "hit": hit, "rr": rr, "top": top, "gold": list(gold_set)})

        if not hit:
            failures.append({"id": q_id, "q": q["q"], "gold": list(gold_set), "got": top})

    score = sum(hits) / max(1, len(hits))
    mrr = sum(rrs) / max(1, len(rrs))
    return Result(
        score=score,
        recall_at_k=score,
        mrr=mrr,
        k=k,
        failures=failures,
        per_question=per_question,
        n_chunks=len(layer.chunks),
        layer_reused=layer.reused,
        strategy=s
    )

