"""التصدير لمحركات البحث الإنتاجية — Exporting Configurations to Elasticsearch / OpenSearch"""
from __future__ import annotations
import json
from pathlib import Path
from .core import Strategy

TARGETS = ["elasticsearch", "opensearch"]

def to_analyzer(s: Strategy, target: str = "elasticsearch") -> dict:
    if target.lower() not in TARGETS:
        raise ValueError(f"Target '{target}' is not supported. Choose from {TARGETS}")

    filters = ["lowercase"]
    if "alef" in s.norm_ops or "yaa" in s.norm_ops or "taa" in s.norm_ops:
        filters.append("arabic_normalization")
    if "stem_light" in s.norm_ops:
        filters.append("arabic_stem")
    if s.ngram > 0:
        filters.append(f"arabic_ngram_{s.ngram}")

    filter_defs = {}
    if s.ngram > 0:
        filter_defs[f"arabic_ngram_{s.ngram}"] = {
            "type": "ngram",
            "min_gram": s.ngram,
            "max_gram": s.ngram
        }

    return {
        "settings": {
            "analysis": {
                "filter": filter_defs,
                "analyzer": {
                    "arabic_tuned": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": filters
                    }
                }
            }
        }
    }

def rationale(s: Strategy, stats: dict) -> str:
    lines = [
        "# تقرير مبرّرات التهيئة النشرية",
        f"• الدقة المقيسة: {stats.get('score', 0.0)} ({stats.get('metric', 'recall@1')}) على {stats.get('n_docs', 0)} مستند.",
        f"• خيار التقطيع: {s.chunk_mode} بحجم {s.chunk_size} حرفاً.",
        f"• خيار التطبيع: {', '.join(s.norm_ops)}.",
    ]
    if s.ngram > 0:
        lines.append("• تنبيه: فائدة المقاطع الحرفية (ngram) تتلاشى عند التوسع لملايين المستندات، أسقطها وقِس عند التشغيل الحقيقي.")

    return "\n".join(lines)

def export(config_path: str, target: str = "elasticsearch", out_dir: str = "out") -> tuple[str, str]:
    cfg = json.loads(Path(config_path).read_text("utf-8"))
    st_dict = cfg.get("strategy", cfg)
    s = Strategy(
        name=st_dict.get("name", "tuned"),
        norm_ops=st_dict.get("norm_ops", []),
        chunk_mode=st_dict.get("chunk_mode", "discourse"),
        chunk_size=st_dict.get("chunk_size", 250),
        overlap=st_dict.get("overlap", 0),
        ngram=st_dict.get("ngram", 0)
    )

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    analyzer_body = to_analyzer(s, target=target)
    settings_file = out_p / f"deploy_{target}.json"
    settings_file.write_text(json.dumps(analyzer_body, ensure_ascii=False, indent=2), "utf-8")

    rat_txt = rationale(s, cfg)
    rationale_file = out_p / "deploy_rationale.md"
    rationale_file.write_text(rat_txt, "utf-8")

    return str(settings_file), str(rationale_file)
