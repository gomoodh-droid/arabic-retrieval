"""
الواجهة العملية: باحث عربي جاهز للاستعمال في مشروعك.

    from arl import ArabicRetriever

    r = ArabicRetriever()                  # بالإعدادات الموصى بها (مقيسة لا مخمّنة)
    r.add([{"id": "d1", "text": "..."}])
    r.search("سؤالي", k=3)

ولضبطه على بياناتك أنت:
    python tune.py corpus.json --questions questions.json --out best_config.json
    r = ArabicRetriever.from_config("best_config.json")
"""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path

from .core import BM25, Strategy, chunk, normalize, tokens

# الإعدادات الموصى بها — مستخرَجة من قياس على ألف مقال عربي حقيقي
# تحت ضبط التسريب (انظر DATA.md). ليست تخمينًا، ولا تصلح لكل متن:
# اضبطها على بياناتك بـ tune.py متى استطعت.
RECOMMENDED = Strategy(
    name="recommended_2026_07",
    norm_ops=["nfkc", "strip_diacritics", "alef", "yaa", "taa", "digits", "punct", "collapse_ws"],
    chunk_mode="discourse",
    chunk_size=300,
    overlap=0,
    ngram=3,
)


class ArabicRetriever:
    """باحث لفظي عربي: تطبيع + تقطيع + BM25 بفهرس معكوس. بلا اعتماديات ولا نماذج."""

    def __init__(self, strategy: Strategy | None = None):
        self.strategy = strategy or RECOMMENDED
        self._chunks: list[str] = []
        self._owners: list[str] = []
        self._meta: dict[str, dict] = {}
        self._bm: BM25 | None = None

    # ------------------------------------------------------------ الإعداد
    @classmethod
    def from_config(cls, path: str | Path) -> "ArabicRetriever":
        d = json.loads(Path(path).read_text("utf-8"))
        d = d.get("strategy", d)
        return cls(Strategy(**{k: v for k, v in d.items() if k in Strategy.__dataclass_fields__}))

    def save_config(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"strategy": asdict(self.strategy)},
                                         ensure_ascii=False, indent=2), "utf-8")

    # ------------------------------------------------------------ الفهرسة
    def add(self, docs: list[dict]) -> "ArabicRetriever":
        """docs: [{"id": str, "text": str, ...أي حقول إضافية}]"""
        s = self.strategy
        for d in docs:
            did = str(d["id"])
            self._meta[did] = {k: v for k, v in d.items() if k != "text"}
            for c in chunk(normalize(d["text"], s.norm_ops), s.chunk_mode, s.chunk_size, s.overlap):
                self._chunks.append(c)
                self._owners.append(did)
        self._bm = None                       # الفهرس يُعاد بناؤه عند أول بحث
        return self

    def _ensure_index(self) -> None:
        if self._bm is None:
            self._bm = BM25([tokens(c, self.strategy.ngram) for c in self._chunks])

    # ------------------------------------------------------------ البحث
    def search(self, query: str, k: int = 3, per_doc: bool = True) -> list[dict]:
        """يرجع: [{"id", "score", "chunk", ...حقول المستند}] مرتبة تنازليًا."""
        if not self._chunks:
            return []
        self._ensure_index()
        qt = tokens(normalize(query, self.strategy.norm_ops), self.strategy.ngram)
        acc = self._bm.scores(qt)
        if not acc:
            return []

        out, seen = [], set()
        for i in sorted(acc, key=acc.get, reverse=True):
            if acc[i] <= 0:
                break
            did = self._owners[i]

            if per_doc and did in seen:
                continue
            seen.add(did)
            out.append({"id": did, "score": round(acc[i], 4),
                        "chunk": self._chunks[i], **self._meta.get(did, {})})
            if len(out) >= k:
                break
        return out

    def __len__(self) -> int:
        return len(self._meta)

    def __repr__(self) -> str:
        return (f"ArabicRetriever(docs={len(self)}, chunks={len(self._chunks)}, "
                f"strategy={self.strategy.name})")
