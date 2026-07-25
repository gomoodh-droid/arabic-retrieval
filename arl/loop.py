"""
الحلقة: اقترح ← قِس ← انعكس ← اقترح أفضل.
أوزان النموذج مثبّتة؛ الذي يتطوّر هو الاستراتيجيات — وهذا ما يجعلها قابلة للتفسير والتكرار.
"""
from __future__ import annotations
import json, time
from dataclasses import asdict
from pathlib import Path
from .core import STATS, Strategy, evaluate, load_data
from .llm import LLM
from .agents import StrategistAgent, ReflectionAgent, CorpusAgent, ReporterAgent

RUNS = Path(__file__).parent.parent / "runs"

SEEDS = [
    Strategy("baseline_naive", [], "chars", 220, 40, 0),
    Strategy("baseline_ws", ["collapse_ws"], "chars", 220, 40, 0),
]


def run(generations: int = 4, per_gen: int = 4, k: int = 3,
        expand_corpus: bool = False, quiet: bool = False):
    llm = LLM()
    strategist, reflector = StrategistAgent(llm), ReflectionAgent(llm)
    reporter, corpus_agent = ReporterAgent(llm), CorpusAgent(llm)

    passages, questions = load_data()
    mode = f"نموذج: {llm.provider}" + (f"/{llm.model}" if llm.model else "") if llm.online else "وضع استكشافي (بلا نموذج)"
    say = (lambda *a: None) if quiet else print
    say(f"\n=== مختبر الاسترجاع العربي — {mode} ===")
    say(f"المتن: {len(passages)} مقطعًا | الأسئلة: {len(questions)}")

    if expand_corpus:
        extra = corpus_agent.expand(passages)
        if extra:
            questions += extra
            say(f"وكيل المتن أضاف {len(extra)} سؤالًا → الإجمالي {len(questions)}")

    history, best_by_gen, mean_by_gen, seen, diagnosis = [], [], [], set(), ""

    for gen in range(generations):
        cands = SEEDS if gen == 0 else strategist.propose(history, diagnosis, per_gen)
        cands = [c for c in cands if c.key() not in seen] or strategist._mutate(history, per_gen)
        say(f"\n— الجيل {gen}: {len(cands)} تهيئة مقترحة")

        for s in cands:
            seen.add(s.key())
            r = evaluate(s, passages, questions, k=k)
            history.append(r)
            say(f"   {s.name:26} score={r.score:<8} {r.label()}={r.recall_at_k:<8} "
                f"mrr={r.mrr:<8} chunks={r.n_chunks:<4} fails={len(r.failures):<3}"
                f"{'  [طبقة نص مُعاد استخدامها]' if r.layer_reused else ''}")

        gen_results = history[-len(cands):]
        mean_by_gen.append(round(sum(r.score for r in gen_results) / len(gen_results), 4))
        best = max(history, key=lambda r: r.score)
        best_by_gen.append(best.score)
        diagnosis = reflector.diagnose(best)
        say(f"   ★ الأفضل حتى الآن: {best.strategy.name} ({best.score})")
        say(f"   التشخيص: {diagnosis[:300]}")

    best = max(history, key=lambda r: r.score)
    summary = reporter.summarize(history)

    say("\n=== منحنى التحسّن ===")
    say("  (█ الأفضل حتى الآن — لا ينزل بالتصميم | ░ متوسط الجيل — يُظهر الاستكشاف الحقيقي)")
    for i, (bs, ms) in enumerate(zip(best_by_gen, mean_by_gen)):
        say(f"  الجيل {i}  أفضل {bs:<7} {'█' * int(bs * 36)}")
        say(f"           متوسط {ms:<7} {'░' * int(ms * 36)}")
    say(f"\n{summary}\n")

    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "mode": mode, "llm_calls": llm.calls,
        "n_passages": len(passages), "n_questions": len(questions),
        "k": k, "curve_best": best_by_gen, "curve_mean": mean_by_gen,
        "text_layer": dict(STATS), "summary": summary,
        "best": {"strategy": asdict(best.strategy), "score": best.score,
                 "recall_at_k": best.recall_at_k, "mrr": best.mrr,
                 "remaining_failures": [f["id"] for f in best.failures]},
        "all": [{"strategy": asdict(r.strategy), "score": r.score,
                 "recall_at_k": r.recall_at_k, "mrr": r.mrr} for r in history],
    }, ensure_ascii=False, indent=2), "utf-8")
    say(f"طبقة النص: بُنيت {STATS['built']} مرة، وأُعيد استخدامها {STATS['reused']} مرة "
        f"(الفصل بين النص والتمثيل يوفّر إعادة التقطيع)")
    say(f"سُجّلت النتائج في {out.relative_to(RUNS.parent)}")
    return {"curve": best_by_gen, "curve_mean": mean_by_gen, "best": best,
            "summary": summary, "path": out, "text_layer": dict(STATS)}
