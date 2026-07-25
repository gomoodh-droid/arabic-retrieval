"""
تحليل المساهمة (Ablation): كم يضيف كل قرارٍ وحده؟

هذا هو مخرَج المعرفة في المشروع. الجداول تقول «التهيئة الفلانية 0.37»،
وهذا لا يعلّم أحدًا شيئًا. أمّا هذا الملف فيقول:
«توحيد الهمزات يضيف كذا، والمقاطع الحرفية تضيف كذا، والتقطيع الدلالي لا يضيف شيئًا يُذكر»
— وهذي جملة ينتفع بها من لن يشغّل الأداة أبدًا.

منظوران، لأن أحدهما وحده يخدع:
  • الإضافة المفردة  (add-one-in)  : من الصفر، ماذا يضيف هذا القرار وحده؟
  • الحذف المفرد     (leave-one-out): من التهيئة الكاملة، كم نخسر بحذفه؟
الفرق بينهما يكشف التداخل: قرارٌ يبدو نافعًا منفردًا وقد يكون زائدًا عن الحاجة داخل المجموعة.

    python ablate.py --corpus arl/data/corpus.json --questions arl/data/questions.json -k 1
"""
from __future__ import annotations
import argparse, json
from dataclasses import replace
from pathlib import Path

from arl.core import ALL_NORM_OPS, Strategy, evaluate, load_data

BASE = Strategy("base", [], "chars", 220, 40, 0)
FULL_OPS = ["nfkc", "strip_diacritics", "alef", "yaa", "taa", "digits", "punct", "collapse_ws"]
FULL = Strategy("full", FULL_OPS, "discourse", 300, 0, 3)

STRUCTURAL = [
    ("تقطيع: جُمل", {"chunk_mode": "sentences"}),
    ("تقطيع: روابط خطاب", {"chunk_mode": "discourse"}),
    ("مقاطع حرفية 2", {"ngram": 2}),
    ("مقاطع حرفية 3", {"ngram": 3}),
    ("مقاطع حرفية 4", {"ngram": 4}),
    ("مقطع أكبر (300)", {"chunk_size": 300}),
    ("بلا تداخل", {"overlap": 0}),
]


def run(corpus, questions, k: int):
    sc = lambda s: evaluate(s, corpus, questions, k=k, use_cache=False).score  # noqa: E731

    base = sc(BASE)
    full = sc(FULL)
    print(f"الأساس (بلا شيء): {base}   |   الكامل: {full}   |   الفارق: {round(full - base, 4)}\n")

    print("=== الإضافة المفردة: من الأساس، ماذا يضيف كل قرار وحده؟ ===")
    add = {}
    for op in ALL_NORM_OPS:
        add[f"تطبيع: {op}"] = round(sc(replace(BASE, norm_ops=[op])) - base, 4)
    for label, kw in STRUCTURAL:
        add[label] = round(sc(replace(BASE, **kw)) - base, 4)
    for name, d in sorted(add.items(), key=lambda x: -x[1]):
        bar = "█" * max(0, int(d * 120))
        print(f"  {name:<24} {d:+.4f}  {bar}")

    print("\n=== الحذف المفرد: من التهيئة الكاملة، كم نخسر بحذف كل قرار؟ ===")
    drop = {}
    for op in FULL.norm_ops:
        drop[f"تطبيع: {op}"] = round(sc(replace(FULL, norm_ops=[o for o in FULL.norm_ops if o != op])) - full, 4)
    drop["المقاطع الحرفية"] = round(sc(replace(FULL, ngram=0)) - full, 4)
    drop["التقطيع الدلالي"] = round(sc(replace(FULL, chunk_mode="chars")) - full, 4)
    for name, d in sorted(drop.items(), key=lambda x: x[1]):
        bar = "▓" * max(0, int(-d * 120))
        print(f"  {name:<24} {d:+.4f}  {bar}")

    print("\n=== القراءة ===")
    top_add = max(add, key=add.get)
    worst_drop = min(drop, key=drop.get)
    useless = [n for n, d in add.items() if abs(d) < 0.005 and n.startswith("تطبيع")]
    print(f"  • أكبر مساهم منفردًا: {top_add} ({add[top_add]:+.4f})")
    print(f"  • أكثر ما يُخسر بحذفه: {worst_drop} ({drop[worst_drop]:+.4f})")
    if useless:
        print(f"  • بلا أثر يُذكر على هذا المتن: {'، '.join(u.split(': ')[1] for u in useless)}")
    print("  • تنبيه: هذي مساهمات على *هذا المتن*. أعِد التحليل على متنك.")

    return {"base": base, "full": full, "add_one_in": add, "leave_one_out": drop, "k": k}


def load(p):
    d = json.loads(Path(p).read_text("utf-8"))
    for key in ("passages", "questions"):
        if isinstance(d, dict) and key in d:
            return d[key]
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="تحليل مساهمة كل قرار في خط الاسترجاع")
    ap.add_argument("--corpus")
    ap.add_argument("--questions")
    ap.add_argument("-k", type=int, default=1)
    ap.add_argument("--out", default="runs/ablation.json")
    a = ap.parse_args()

    corpus, questions = (load(a.corpus), load(a.questions)) if a.corpus else load_data()
    print(f"المتن: {len(corpus)} | الأسئلة: {len(questions)} | المقياس: recall@{a.k}\n")
    res = run(corpus, questions, a.k)
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), "utf-8")
    print(f"\nسُجّل في {a.out}")
