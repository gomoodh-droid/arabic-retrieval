"""
الوكلاء الأربعة. كل وكيل له مسارٌ بالنموذج ومسارٌ استكشافي بديل،
فالحلقة تدور في الحالتين، وتصير أذكى عند توفّر نموذج.
"""
from __future__ import annotations
import random
from dataclasses import asdict
from .core import ALL_NORM_OPS, CHUNK_MODES, Strategy, Result
from .llm import LLM, parse_json



class StrategistAgent:
    """يقترح استراتيجيات جديدة اعتمادًا على تاريخ النتائج وتشخيص الانعكاس."""
    SYS = ("أنت مهندس استرجاع عربي. مهمتك اقتراح تهيئات جديدة لتحسين استرجاع نص عربي. "
           "أجب بمصفوفة JSON فقط، كل عنصر: "
           '{"name":str,"norm_ops":[str],"chunk_mode":"chars|sentences|discourse",'
           '"chunk_size":int,"overlap":int,"ngram":int}. '
           f"عمليات التطبيع المتاحة فقط: {ALL_NORM_OPS}. ngram بين 0 و4.")

    def __init__(self, llm: LLM):
        self.llm = llm

    def propose(self, history: list[Result], diagnosis: str, n: int = 4) -> list[Strategy]:
        if self.llm.online:
            hist = "\n".join(f"- {r.strategy.name}: score={r.score} {r.label()}={r.recall_at_k} "
                             f"chunks={r.n_chunks} ops={r.strategy.norm_ops} "
                             f"mode={r.strategy.chunk_mode}/{r.strategy.chunk_size} ngram={r.strategy.ngram}"
                             for r in sorted(history, key=lambda x: -x.score)[:8]) or "لا تاريخ بعد"
            out = parse_json(self.llm.ask(self.SYS,
                f"النتائج السابقة:\n{hist}\n\nتشخيص آخر دورة:\n{diagnosis or 'لا يوجد'}\n\n"
                f"اقترح {n} تهيئات جديدة مختلفة عن السابق، وفسّر الاسم بما يعالجه."), [])
            strats = []
            for d in out[:n] if isinstance(out, list) else []:
                try:
                    strats.append(Strategy(
                        name=str(d.get("name", "llm"))[:40],
                        norm_ops=[o for o in d.get("norm_ops", []) if o in ALL_NORM_OPS],
                        chunk_mode=d["chunk_mode"] if d.get("chunk_mode") in CHUNK_MODES else "discourse",
                        chunk_size=max(60, min(600, int(d.get("chunk_size", 180)))),
                        overlap=max(0, min(200, int(d.get("overlap", 0)))),
                        ngram=max(0, min(4, int(d.get("ngram", 0))))))
                except Exception:  # noqa: BLE001
                    continue
            if strats:
                return strats
        return self._mutate(history, n)

    @staticmethod
    def _mutate(history: list[Result], n: int) -> list[Strategy]:
        """بديل استكشافي: تطوير من أفضل ما وُجد، مع استكشاف عشوائي محكوم."""
        rng = random.Random(len(history) * 7 + 13)
        base = sorted(history, key=lambda x: -x.score)[0].strategy if history else Strategy("seed", [])
        out = []
        # مرشّح واحد يغيّر التمثيل فقط ويُبقي طبقة النص كما هي —
        # فحص رخيص لأثر ngram وحده، ويستفيد من طبقة نص مخزَّنة سابقًا.
        rep_only = [g for g in (0, 2, 3, 4) if g != base.ngram]
        out.append(Strategy(f"rep{len(history)}_ngram{rep_only[0]}", list(base.norm_ops),
                            base.chunk_mode, base.chunk_size, base.overlap, rng.choice(rep_only)))
        for i in range(max(0, n - 1)):
            ops = list(base.norm_ops)
            cand = rng.choice(ALL_NORM_OPS)
            ops = [o for o in ops if o != cand] if cand in ops and rng.random() < .4 else sorted(set(ops + [cand]))
            out.append(Strategy(
                name=f"mut{len(history)}_{i}",
                norm_ops=ops,
                chunk_mode=rng.choice(CHUNK_MODES) if rng.random() < .5 else base.chunk_mode,
                chunk_size=max(60, min(600, base.chunk_size + rng.choice([-80, -40, 0, 40, 80]))),
                overlap=max(0, min(120, base.overlap + rng.choice([-40, 0, 40]))),
                ngram=rng.choice([0, 0, 2, 3, 4]) if rng.random() < .6 else base.ngram))
        return out


class ReflectionAgent:
    """يقرأ حالات الفشل ويكتب تشخيصًا موجّهًا للجيل القادم."""
    SYS = ("أنت محلّل أخطاء استرجاع عربي. اقرأ حالات الفشل واكتب تشخيصًا موجزًا (٣ أسطر) "
           "يحدّد النمط اللغوي المشترك بين الأخطاء، ثم توصية واحدة محدّدة للتهيئة القادمة.")

    def __init__(self, llm: LLM):
        self.llm = llm

    def diagnose(self, best: Result) -> str:
        if not best.failures:
            return "لا فشل في هذي الدورة — جرّب مقياسًا أصعب (k=1) أو أسئلة أطول."
        if self.llm.online:
            cases = "\n".join(f"- «{f['q']}» المتوقع {f['gold']} والمسترجع {f['got']}" for f in best.failures[:12])
            txt = self.llm.ask(self.SYS,
                f"التهيئة: ops={best.strategy.norm_ops} mode={best.strategy.chunk_mode}/"
                f"{best.strategy.chunk_size} ngram={best.strategy.ngram}\nالفشل:\n{cases}")
            if txt.strip():
                return txt.strip()
        return self._heuristic(best)

    @staticmethod
    def _heuristic(best: Result) -> str:
        import re
        qs = " ".join(f["q"] for f in best.failures)
        notes = []
        if re.search(r"[ىئؤإأآة]", qs):
            notes.append("الأسئلة الفاشلة تحمل همزات/تاءً مربوطة غير موحّدة → فعّل alef وyaa وtaa.")
        if re.search(r"\b(وش|ايش|ليش|كيف|متى)\b", qs):
            notes.append("صيغ عامية لا تتقاطع لفظيًا مع المتن → جرّب ngram=3 لتقريب الجذوع.")
        if best.strategy.chunk_mode == "chars":
            notes.append("التقطيع بالأحرف يقطع الجمل → جرّب discourse.")
        if best.strategy.ngram == 0:
            notes.append("لا مقاطع حرفية → التطابق سطحي أمام الاختلاف الصرفي.")
        return " ".join(notes) or "لا نمط واضح — وسّع الاستكشاف."


class CorpusAgent:
    """يوسّع مجموعة الاختبار بأسئلة جديدة من المقاطع نفسها."""
    SYS = ('أنت معلّم عربي. لكل مقطع اكتب سؤالين: أحدهما بالفصحى والآخر بالعامية الخليجية، '
           'بلا نسخ ألفاظ المقطع حرفيًا. أجب بمصفوفة JSON: [{"q":str,"gold":["Pn"]}]')

    def __init__(self, llm: LLM):
        self.llm = llm

    def expand(self, passages: list[dict], limit: int = 6) -> list[dict]:
        if not self.llm.online:
            return []
        src = "\n".join(f'{p["id"]}: {p["text"][:220]}' for p in passages[:limit])
        out = parse_json(self.llm.ask(self.SYS, src), [])
        return [{"id": f"G{i}", "q": d["q"], "gold": d["gold"]}
                for i, d in enumerate(out) if isinstance(d, dict) and d.get("q") and d.get("gold")]


class ReporterAgent:
    """يكتب ملخص الدورة."""
    SYS = "أنت كاتب تقارير تقنية. اكتب ملخصًا من أربعة أسطر: ما تحسّن، وبفضل أي تغيير، وما بقي."

    def __init__(self, llm: LLM):
        self.llm = llm

    def summarize(self, history: list[Result]) -> str:
        rows = "\n".join(f"- gen{i}: {r.strategy.name} score={r.score}" for i, r in enumerate(history))
        if self.llm.online:
            t = self.llm.ask(self.SYS, rows)
            if t.strip():
                return t.strip()
        best = max(history, key=lambda r: r.score)
        first = history[0]
        return (f"أفضل تهيئة: {best.strategy.name} بدرجة {best.score} "
                f"(البداية {first.score}، التحسّن {round(best.score - first.score, 4)}). "
                f"العناصر الفاعلة: ops={best.strategy.norm_ops}، "
                f"تقطيع={best.strategy.chunk_mode}/{best.strategy.chunk_size}، ngram={best.strategy.ngram}. "
                f"بقي {len(best.failures)} سؤالًا فاشلًا.")
