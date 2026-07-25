"""بصمة المتن وتوصيف الخصائص — Corpus Feature Extraction & Profiling"""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass

@dataclass
class CorpusFeatures:
    n_docs: int
    total_chars: int
    avg_doc_chars: float
    median_doc_chars: float
    vocab: int
    tokens: int
    type_token_ratio: float
    hapax_ratio: float
    avg_word_len: float
    diacritics_ratio: float
    latin_ratio: float
    digit_ratio: float
    al_prefix_ratio: float

def extract(corpus: list[dict]) -> CorpusFeatures:
    n_docs = len(corpus)
    if n_docs == 0:
        return CorpusFeatures(0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    lens = [len(doc.get("text", "")) for doc in corpus]
    total_chars = sum(lens)
    lens_sorted = sorted(lens)
    median_chars = lens_sorted[n_docs // 2] if n_docs % 2 != 0 else (lens_sorted[n_docs // 2 - 1] + lens_sorted[n_docs // 2]) / 2.0

    all_text = " ".join(doc.get("text", "") for doc in corpus)
    words = all_text.split()
    n_tokens = len(words)
    counts = Counter(words)
    vocab = len(counts)
    ttr = vocab / max(1, n_tokens)
    hapax = sum(1 for c in counts.values() if c == 1) / max(1, vocab)

    avg_w_len = sum(len(w) for w in words) / max(1, n_tokens)
    diacritics = len(re.findall(r"[\u064B-\u065F\u0670]", all_text)) / max(1, total_chars)
    latin = len(re.findall(r"[a-zA-Z]", all_text)) / max(1, total_chars)
    digits = len(re.findall(r"[0-9٠١٢٣٤٥٦٧٨٩]", all_text)) / max(1, total_chars)
    al_prefix = sum(1 for w in words if w.startswith("ال")) / max(1, n_tokens)

    return CorpusFeatures(
        n_docs=n_docs,
        total_chars=total_chars,
        avg_doc_chars=total_chars / n_docs,
        median_doc_chars=median_chars,
        vocab=vocab,
        tokens=n_tokens,
        type_token_ratio=ttr,
        hapax_ratio=hapax,
        avg_word_len=avg_w_len,
        diacritics_ratio=diacritics,
        latin_ratio=latin,
        digit_ratio=digits,
        al_prefix_ratio=al_prefix,
    )

def compare_to_reference(f: CorpusFeatures) -> str:
    notes = []
    if f.n_docs > 100_000:
        notes.append("متن أكبر بكثير من المرجعي")
    if f.latin_ratio > 0.1:
        notes.append("نسبة نص لاتيني مرتفعة")
    if f.diacritics_ratio > 0.1:
        notes.append("نسبة تشكيل عالية جداً")
    if not notes:
        return "المتن يطابق الخصائص المرجعية للمقالات الصحفية."
    return "تنبيه: " + "، و".join(notes) + "."
