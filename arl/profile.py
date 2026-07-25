"""
قياس التشغيل: الجودة وحدها لا تكفي لاختيار أداة.
من يضعها في مشروعه يحتاج أن يعرف: كم تستغرق الفهرسة؟ وكم يستهلك الفهرس؟
وكم يستغرق السؤال الواحد؟ وكم ذاكرة تأكل؟
"""
from __future__ import annotations
import gc, statistics, time, tracemalloc
from dataclasses import asdict, dataclass

from .core import BM25, Strategy, build_text_layer, evaluate, normalize, tokens


@dataclass
class Profile:
    strategy_name: str
    score: float
    recall_at_k: float
    k: int
    n_docs: int
    n_chunks: int
    vocab: int
    build_ms: float          # زمن بناء الفهرس
    query_p50_ms: float      # وسيط زمن الاستعلام
    query_p95_ms: float
    index_mb: float          # ذروة الذاكرة أثناء البناء
    postings: int            # حجم الفهرس المعكوس (عدد المداخل)

    def row(self) -> str:
        return (f"{self.strategy_name:<12} score={self.score:<7} "
                f"بناء={self.build_ms:>7.0f}ms  استعلام(p50/p95)={self.query_p50_ms:>6.1f}/"
                f"{self.query_p95_ms:>6.1f}ms  ذاكرة={self.index_mb:>6.1f}MB  "
                f"مفردات={self.vocab:<7} مقاطع={self.n_chunks}")


def profile(s: Strategy, corpus: list[dict], questions: list[dict],
            k: int = 3, sample_queries: int = 200) -> Profile:
    gc.collect()
    tracemalloc.start()

    t0 = time.perf_counter()
    layer = build_text_layer(corpus, s, use_cache=False)
    docs = [tokens(c, s.ngram) for c in layer.chunks]
    bm = BM25(docs)
    build_ms = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lat = []
    for q in questions[:sample_queries]:
        t1 = time.perf_counter()
        qt = tokens(normalize(q["q"], s.norm_ops), s.ngram)
        acc = bm.scores(qt)
        sorted(acc, key=acc.get, reverse=True)[:k]
        lat.append((time.perf_counter() - t1) * 1000)

    r = evaluate(s, corpus, questions, k=k, use_cache=False)
    lat.sort()
    return Profile(
        strategy_name=s.name, score=r.score, recall_at_k=r.recall_at_k, k=k,
        n_docs=len(corpus), n_chunks=len(layer.chunks), vocab=len(bm.idf),
        build_ms=round(build_ms, 1),
        query_p50_ms=round(statistics.median(lat), 3),
        query_p95_ms=round(lat[int(len(lat) * 0.95)] if len(lat) > 1 else lat[0], 3),
        index_mb=round(peak / 1024 / 1024, 2),
        postings=sum(len(d) for d in bm.docs),
    )



def compare(strategies: list[Strategy], corpus, questions, k: int = 3) -> list[Profile]:
    out = []
    for s in strategies:
        p = profile(s, corpus, questions, k)
        out.append(p)
        print("  " + p.row(), flush=True)
    return out


def as_markdown(profiles: list[Profile]) -> str:
    h = ("| التهيئة | score | بناء (ms) | استعلام p50 (ms) | استعلام p95 (ms) "
         "| ذاكرة (MB) | مقاطع | مفردات |\n|---|---|---|---|---|---|---|---|\n")
    return h + "\n".join(
        f"| {p.strategy_name} | {p.score} | {p.build_ms:.0f} | {p.query_p50_ms:.2f} "
        f"| {p.query_p95_ms:.2f} | {p.index_mb:.1f} | {p.n_chunks} | {p.vocab} |"
        for p in profiles)
