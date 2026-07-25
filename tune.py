#!/usr/bin/env python3
"""
اضبط الباحث على بياناتك أنت — لا على بياناتي.

    # عندك أسئلة موسومة:
    python tune.py my_corpus.json --questions my_questions.json --out best_config.json

    # ما عندك أسئلة؟ استخدم العناوين كأسئلة تلقائيًا:
    python tune.py my_articles.json --titles-as-queries --out best_config.json

الصيغة المتوقعة:
    corpus:    [{"id": "d1", "text": "..."}]  أو  [{"id","title","text"}]
    questions: [{"id": "q1", "q": "...", "gold": ["d1"]}]
"""
import argparse, json
from pathlib import Path
from arl.core import Strategy, evaluate
from arl.agents import StrategistAgent, ReflectionAgent
from arl.llm import LLM
from arl.retriever import RECOMMENDED

SEEDS = [
    Strategy("naive", [], "chars", 220, 40, 0),
    Strategy("norm_only", ["nfkc", "strip_diacritics", "alef", "yaa", "taa", "digits", "collapse_ws"], "chars", 220, 40, 0),
    RECOMMENDED,
]


def load(p):
    d = json.loads(Path(p).read_text("utf-8"))
    for key in ("passages", "questions", "docs", "data"):
        if isinstance(d, dict) and key in d:
            return d[key]
    return d


def greedy_search(corpus, questions, k=1, max_rounds=3, verbose=False):
    from dataclasses import replace
    from arl.core import ALL_NORM_OPS
    base_s = Strategy("base", [], "chars", 220, 40, 0)
    best_res = evaluate(base_s, corpus, questions, k=k, use_cache=False)
    order = []

    # Round 0: ngram
    best_ng, best_ng_res = base_s, best_res
    for ng in [2, 3, 4]:
        c = replace(base_s, name=f"ngram_{ng}", ngram=ng)
        r = evaluate(c, corpus, questions, k=k, use_cache=False)
        if r.score > best_ng_res.score:
            best_ng, best_ng_res = c, r
    if best_ng_res.score > best_res.score:
        order.append({"step": f"ngram={best_ng.ngram}", "gain": round(best_ng_res.score - best_res.score, 4), "score": best_ng_res.score})
        base_s, best_res = best_ng, best_ng_res

    # Round 1: norm ops
    for op in ALL_NORM_OPS:
        if op not in base_s.norm_ops:
            c = replace(base_s, name=f"add_{op}", norm_ops=base_s.norm_ops + [op])
            r = evaluate(c, corpus, questions, k=k, use_cache=False)
            if r.score > best_res.score:
                order.append({"step": f"+{op}", "gain": round(r.score - best_res.score, 4), "score": r.score})
                base_s, best_res = c, r
                if len(order) >= max_rounds:
                    break

    return base_s, best_res.score, order, [best_res]


def main():

    ap = argparse.ArgumentParser(description="ضبط الباحث العربي على بياناتك")
    ap.add_argument("corpus")
    ap.add_argument("--questions")
    ap.add_argument("--titles-as-queries", action="store_true")
    ap.add_argument("--out", default="best_config.json")
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--per-gen", type=int, default=4)
    ap.add_argument("-k", type=int, default=1)
    a = ap.parse_args()

    corpus = load(a.corpus)
    if a.questions:
        questions = load(a.questions)
    elif a.titles_as_queries:
        questions = [{"id": f"T{i}", "q": d["title"], "gold": [str(d["id"])]}
                     for i, d in enumerate(corpus) if d.get("title")]
        print(f"وُلّد {len(questions)} سؤالًا من العناوين. "
              "تنبيه: العنوان يشترك لفظيًا مع النص، فالنتيجة متفائلة — راجع DATA.md")
    else:
        ap.error("مرّر --questions أو --titles-as-queries")

    if not questions:
        ap.error("لا أسئلة — لا يمكن القياس بلا مسطرة")

    print(f"المتن: {len(corpus)} مستندًا | الأسئلة: {len(questions)} | المقياس: recall@{a.k}\n")
    llm = LLM()
    strategist, reflector = StrategistAgent(llm), ReflectionAgent(llm)
    history, diag, seen = [], "", set()

    for g in range(a.generations):
        cands = SEEDS if g == 0 else strategist.propose(history, diag, a.per_gen)
        cands = [c for c in cands if c.key() not in seen] or strategist._mutate(history, a.per_gen)
        for s in cands:
            seen.add(s.key())
            r = evaluate(s, corpus, questions, k=a.k, use_cache=False)
            history.append(r)
            print(f"  الجيل {g} · {s.name:24} score={r.score:<8} {r.label()}={r.recall_at_k}")
        best = max(history, key=lambda x: x.score)
        diag = reflector.diagnose(best)
        print(f"  ★ الأفضل: {best.strategy.name} ({best.score})\n")

    best = max(history, key=lambda x: x.score)
    Path(a.out).write_text(json.dumps(
        {"strategy": best.strategy.__dict__, "score": best.score,
         "metric": best.label(), "n_docs": len(corpus), "n_questions": len(questions)},
        ensure_ascii=False, indent=2), "utf-8")
    print(f"أفضل تهيئة لبياناتك: {best.strategy.name} ({best.score}) → {a.out}")
    print(f"استعمالها:\n  from arl import ArabicRetriever\n"
          f"  r = ArabicRetriever.from_config('{a.out}')")


if __name__ == "__main__":
    main()
