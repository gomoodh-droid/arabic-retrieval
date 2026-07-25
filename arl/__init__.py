"""مختبر الاسترجاع العربي — Arabic Retrieval Lab"""
from .retriever import ArabicRetriever, RECOMMENDED
from .core import Strategy, evaluate, BM25, normalize, chunk, tokens, stem_word, load_data
from .stats import bootstrap_ci, paired_bootstrap, verdict
from .pareto import Point, frontier, best_within_budget, report
from .features import CorpusFeatures, extract, compare_to_reference
from .deploy import to_analyzer, rationale, export, TARGETS

__all__ = [
    "ArabicRetriever", "RECOMMENDED", "Strategy", "evaluate", "BM25", "normalize", "chunk", "tokens",
    "stem_word", "load_data", "bootstrap_ci", "paired_bootstrap", "verdict", "Point", "frontier",
    "best_within_budget", "report", "CorpusFeatures", "extract", "compare_to_reference",
    "to_analyzer", "rationale", "export", "TARGETS"
]
