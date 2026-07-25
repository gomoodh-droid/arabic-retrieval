"""جبهة باريتو وقيود الميزانية — Pareto Optimization & Budget Filtering"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Point:
    name: str
    score: float
    latency_ms: float
    memory_mb: float

def dominates(a: Point, b: Point) -> bool:
    better_or_equal = (a.score >= b.score and a.latency_ms <= b.latency_ms and a.memory_mb <= b.memory_mb)
    strictly_better = (a.score > b.score or a.latency_ms < b.latency_ms or a.memory_mb < b.memory_mb)
    return better_or_equal and strictly_better

def frontier(points: list[Point]) -> list[Point]:
    res = []
    for p in points:
        if not any(dominates(other, p) for other in points if other != p):
            res.append(p)
    return res

def best_within_budget(points: list[Point], max_latency_ms: float = None, max_memory_mb: float = None) -> Point | None:
    valid = []
    for p in points:
        if max_latency_ms is not None and p.latency_ms > max_latency_ms:
            continue
        if max_memory_mb is not None and p.memory_mb > max_memory_mb:
            continue
        valid.append(p)
    if not valid:
        return None
    return max(valid, key=lambda p: p.score)

def report(points: list[Point], max_latency_ms: float = None, max_memory_mb: float = None) -> str:
    front = frontier(points)
    dominated = [p for p in points if p not in front]
    
    lines = ["=== جبهة باريتو ==="]
    for p in front:
        lines.append(f"  ★ {p.name}: score={p.score:.4f}, زمن={p.latency_ms:.1f}ms, ذاكرة={p.memory_mb:.1f}MB")
        
    if dominated:
        lines.append("\n=== نقاط مُهيمَنٌ عليها ===")
        for p in dominated:
            dominator = next(other for other in front if dominates(other, p))
            lines.append(f"  • {p.name} (يتفوّق عليها {dominator.name})")
            
    best = best_within_budget(points, max_latency_ms, max_memory_mb)
    lines.append("\n=== قيد الميزانية ===")
    if best:
        lines.append(f"  التهيئة الفائزة بالميزانية: ★ {best.name}")
    else:
        lines.append("  لا تهيئة تفي بالقيود المحشورة")
        
    return "\n".join(lines)
