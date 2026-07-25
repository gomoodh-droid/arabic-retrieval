"""الاختبارات الإحصائية ومجالات الثقة — Statistical Bootstrap & Significance Testing"""
from __future__ import annotations
import random

def bootstrap_ci(vals: list[float], n_boot: int = 400, confidence: float = 0.95) -> tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0
    mean = sum(vals) / len(vals)
    n = len(vals)
    samples = []
    rnd = random.Random(42)
    for _ in range(n_boot):
        sample = [rnd.choice(vals) for _ in range(n)]
        samples.append(sum(sample) / n)
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = samples[int(alpha * n_boot)]
    hi = samples[int((1.0 - alpha) * n_boot)]
    return mean, max(0.0, lo), min(1.0, hi)

def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 400, p_value: float = 0.05) -> dict:
    if len(a) != len(b) or not a:
        return {"diff": 0.0, "p_value": 1.0, "significant": False}
    n = len(a)
    actual_diff = (sum(a) - sum(b)) / n
    diffs = []
    rnd = random.Random(42)
    count = 0
    for _ in range(n_boot):
        idx = [rnd.randint(0, n - 1) for _ in range(n)]
        sa = sum(a[i] for i in idx) / n
        sb = sum(b[i] for i in idx) / n
        d = sa - sb
        diffs.append(d)
        if (actual_diff >= 0 and d <= 0) or (actual_diff < 0 and d >= 0):
            count += 1
    p_val = count / n_boot
    significant = p_val < p_value and abs(actual_diff) > 0.02
    return {"diff": actual_diff, "p_value": p_val, "significant": significant}

def verdict(res: dict, name_a: str = "أ", name_b: str = "ب") -> str:
    if res.get("significant"):
        winner = name_a if res.get("diff", 0) > 0 else name_b
        return f"★ الفارق مثبت إحصائيًا لصالح {winner} (الفارق: {res.get('diff'):+.4f})"
    else:
        return f"لا تنشر أن أحدهما أفضل — الفارق ضجيج إحصائي غير مثبت (الفارق: {res.get('diff', 0):+.4f})"
