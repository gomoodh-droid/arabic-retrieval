"""اختبارات النواة — شغّلها قبل أن تثق برقم واحد: python -m pytest -q"""
from arl.core import STATS, Strategy, build_text_layer, chunk, evaluate, load_data, normalize, tokens


def test_normalization_is_optional_and_ordered():
    src = "الإدارةُ الأولــى ٢٠٢٦"
    assert normalize(src, []) == src                      # لا شيء افتراضي
    n = normalize(src, ["strip_diacritics", "alef", "taa", "digits", "collapse_ws"])
    assert "ُ" not in n and "إ" not in n and "ة" not in n and "2026" in n


def test_normalization_is_deterministic():
    ops = ["nfkc", "strip_diacritics", "alef", "yaa", "digits", "collapse_ws"]
    s = "الأسئلةُ عن الطاقةِ ٥٠٪"
    assert normalize(s, ops) == normalize(s, ops)


def test_discourse_chunking_respects_connectors():
    t = "هذا نصٌّ أول طويل بما يكفي ليصير قطعة مستقلة. ولذلك نتوقع حدًّا دلاليًّا هنا تمامًا."
    parts = chunk(t, "discourse", 40, 0)
    assert len(parts) >= 2
    assert any(p.startswith("ولذلك") for p in parts)


def test_char_chunking_covers_whole_text():
    t = "أ" * 500
    parts = chunk(t, "chars", 100, 0)
    assert len(parts) == 5
    assert "".join(parts) == t          # بلا تداخل، الضمّ يعيد النص كما هو


def test_ngrams_expand_tokens():
    assert len(tokens("المدرسة", 3)) > len(tokens("المدرسة", 0))


def test_metric_bounds_and_reproducibility():
    s = Strategy("t", ["collapse_ws"], "discourse", 180, 0, 3)
    a, b = evaluate(s), evaluate(s)
    assert a.score == b.score                              # المسطرة لا ترتجف
    assert 0.0 <= a.recall_at_k <= 1.0 and 0.0 <= a.mrr <= 1.0


def test_metric_label_matches_k():
    assert evaluate(Strategy("t", ["collapse_ws"]), k=1).label() == "recall@1"
    assert evaluate(Strategy("t", ["collapse_ws"]), k=3).label() == "recall@3"


def test_text_layer_is_independent_of_representation():
    """جوهر المعمار: تغيير ngram لا يُبطل طبقة النص."""
    a = Strategy("a", ["collapse_ws"], "discourse", 180, 0, 0)
    b = Strategy("b", ["collapse_ws"], "discourse", 180, 0, 4)
    assert a.text_layer_key() == b.text_layer_key()
    c = Strategy("c", ["collapse_ws", "alef"], "discourse", 180, 0, 0)
    assert a.text_layer_key() != c.text_layer_key()


def test_text_layer_is_reused_from_cache():
    corpus, _ = load_data()
    s0 = Strategy("x", ["collapse_ws", "taa"], "sentences", 150, 0, 0)
    before = dict(STATS)
    l1 = build_text_layer(corpus, s0)
    l2 = build_text_layer(corpus, Strategy("y", ["collapse_ws", "taa"], "sentences", 150, 0, 3))
    assert l1.chunks == l2.chunks and l2.reused
    assert STATS["reused"] > before["reused"]


def test_overlap_applies_to_sentence_modes():
    t = "جملةٌ أولى طويلة بما يكفي لتصير قطعةً مستقلةً هنا. وجملةٌ ثانية طويلة أيضًا بما يكفي تمامًا."
    no_ov = chunk(t, "sentences", 45, 0)
    with_ov = chunk(t, "sentences", 45, 20)
    assert len("".join(with_ov)) > len("".join(no_ov))     # التداخل يزيد التغطية فعلًا


def test_normalization_improves_over_naive():
    naive = evaluate(Strategy("naive", [], "chars", 220, 40, 0))
    tuned = evaluate(Strategy("tuned",
                              ["nfkc", "strip_diacritics", "alef", "yaa", "taa", "digits", "punct", "collapse_ws"],
                              "discourse", 160, 0, 3))
    assert tuned.score > naive.score


def test_loop_curve_is_monotonic_non_decreasing():
    from arl.loop import run
    out = run(generations=2, per_gen=3, quiet=True)
    c, m = out["curve"], out["curve_mean"]
    assert len(c) == 2 and all(c[i + 1] >= c[i] for i in range(len(c) - 1))
    assert len(m) == 2 and all(0.0 <= x <= 1.0 for x in m)


# ---------------------------------------------------------------- الواجهة العملية
def test_retriever_finds_the_right_document():
    from arl import ArabicRetriever
    r = ArabicRetriever().add([
        {"id": "d1", "text": "يخفّض ترسّب الغبار إنتاج الألواح الشمسية بنسبٍ كبيرة حسب الموقع."},
        {"id": "d2", "text": "أصعب قرار في مساعد المجموعات معرفة متى يصمت لا كيف يجيب."},
    ])
    assert len(r) == 2
    assert r.search("وش تاثير الغبار على الالواح؟", k=1)[0]["id"] == "d1"
    assert r.search("متى يسكت المساعد", k=1)[0]["id"] == "d2"


def test_retriever_survives_dialect_and_spelling_variants():
    """المقاطع الحرفية هي ما يجعل هذا يعمل — وهذا سبب وجودها في الإعداد الموصى به."""
    from arl import ArabicRetriever
    r = ArabicRetriever().add([{"id": "x", "text": "ارتفعت قدرة إنتاج المياه المحلاة في المملكة."}])
    for q in ["المياه المحلاه", "انتاج الميـاه", "قدره الانتاج"]:
        assert r.search(q, k=1)[0]["id"] == "x"


def test_retriever_returns_empty_without_docs_or_matches():
    from arl import ArabicRetriever
    assert ArabicRetriever().search("أي سؤال") == []
    r = ArabicRetriever().add([{"id": "a", "text": "نصٌّ عن الزراعة المحمية."}])
    assert r.search("zzz qqq") == []


def test_config_roundtrip_preserves_strategy(tmp_path):
    from arl import ArabicRetriever
    from arl.core import Strategy
    p = tmp_path / "cfg.json"
    ArabicRetriever(Strategy("custom", ["alef"], "sentences", 111, 7, 2)).save_config(p)
    s = ArabicRetriever.from_config(p).strategy
    assert (s.name, s.chunk_mode, s.chunk_size, s.overlap, s.ngram) == ("custom", "sentences", 111, 7, 2)


def test_extra_document_fields_are_returned():
    from arl import ArabicRetriever
    r = ArabicRetriever().add([{"id": "d", "text": "التحلية كثيفة الطاقة.", "url": "http://x", "year": 2026}])
    hit = r.search("الطاقة والتحلية", k=1)[0]
    assert hit["url"] == "http://x" and hit["year"] == 2026


# ---------------------------------------------------------------- ملاحظات المراجعة
def test_discourse_chunking_does_not_emit_tiny_fragments():
    """رُفع ادّعاء بأن «ثم» تُنتج مقاطع ٢٠ حرفًا. الكود يراكم حتى بلوغ الحجم — هذا يثبته."""
    from arl.core import chunk
    t = ("نزل المطر. ثم توقف. وبعد ذلك خرج الناس إلى الشوارع يتفقدون ما حدث "
         "في الليلة الماضية من أضرار، ولكن الأضرار كانت أقل مما توقعه الجميع.")
    for size in (100, 200, 300):
        parts = chunk(t, "discourse", size, 0)
        assert all(len(p) >= size * 0.4 for p in parts[:-1]), \
            f"مقطع أقصر من اللازم عند size={size}: {[len(p) for p in parts]}"


def test_profile_reports_operational_numbers():
    from arl.core import Strategy, load_data
    from arl.profile import profile
    c, q = load_data()
    p = profile(Strategy("t", ["collapse_ws"], "discourse", 200, 0, 3), c, q, k=3, sample_queries=10)
    assert p.build_ms > 0 and p.query_p50_ms >= 0 and p.index_mb > 0
    assert p.vocab > 0 and p.postings >= p.vocab


def test_greedy_search_is_deterministic_and_improves():
    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("tune", Path(__file__).parent.parent / "tune.py")
    tune = importlib.util.module_from_spec(spec); sys.modules["tune"] = tune
    spec.loader.exec_module(tune)
    from arl.core import load_data
    c, q = load_data()
    s1, sc1, order, _ = tune.greedy_search(c, q, k=3, max_rounds=2, verbose=False)
    s2, sc2, _, _ = tune.greedy_search(c, q, k=3, max_rounds=2, verbose=False)
    assert sc1 == sc2 and s1.key() == s2.key()          # حتمي
    assert all(o["gain"] > 0 for o in order)            # لا خطوة بلا فائدة


# ---------------------------------------------------------------- التجذيع والإحصاء
def test_light_stemmer_strips_articles_not_root_letters():
    """«ك» في «كتابها» ليست سابقة — هذا الاختبار يحرس ضد خطأ وقعنا فيه فعلًا."""
    from arl.core import stem_word
    assert stem_word("والمدرسة") == "مدرس"
    assert stem_word("بالكتاب") == "كتاب"
    assert stem_word("كتابها") == "كتاب"        # لا «تاب»
    assert stem_word("في") == "في"              # القصيرة لا تُمسّ
    assert len(stem_word("كتب")) == 3


def test_stemming_is_applied_after_other_normalization():
    from arl.core import normalize
    out = normalize("الأولــى", ["strip_diacritics", "alef", "stem_light"])
    assert "ـ" not in out and "أ" not in out


def test_bootstrap_ci_brackets_the_mean():
    from arl.stats import bootstrap_ci
    vals = [1.0] * 30 + [0.0] * 70
    m, lo, hi = bootstrap_ci(vals, n_boot=400)
    assert abs(m - 0.30) < 1e-9 and lo < m < hi and 0.0 <= lo and hi <= 1.0


def test_paired_bootstrap_detects_real_and_fake_differences():
    from arl.stats import paired_bootstrap
    a = [1.0] * 80 + [0.0] * 20
    b = [0.0] * 80 + [1.0] * 20
    assert paired_bootstrap(a, b, n_boot=400)["significant"]          # فرق كبير
    c = [1.0] * 50 + [0.0] * 50
    d = c[:49] + [1.0] + c[50:]                                       # فرق سؤال واحد
    assert not paired_bootstrap(c, d, n_boot=400)["significant"]      # ضجيج


def test_evaluate_exposes_per_question_results():
    from arl.core import Strategy, evaluate
    r = evaluate(Strategy("t", ["collapse_ws"], "discourse", 200, 0, 3), k=3)
    assert r.per_question and len(r.hits()) == len(r.rrs())
    assert abs(sum(r.hits()) / len(r.hits()) - r.recall_at_k) < 1e-4


# ---------------------------------------------------------------- باريتو والنشر والخصائص
def test_pareto_excludes_dominated_points():
    from arl.pareto import Point, frontier, best_within_budget
    pts = [Point("a", 0.10, 10, 100), Point("b", 0.20, 5, 50), Point("c", 0.15, 20, 200)]
    names = {p.name for p in frontier(pts)}
    assert "b" in names and "a" not in names and "c" not in names   # b يهيمن على الاثنين
    assert best_within_budget(pts, max_latency_ms=4) is None          # لا شيء يفي
    assert best_within_budget(pts, max_latency_ms=6).name == "b"


def test_pareto_keeps_tradeoffs():
    """نقطتان تتبادلان التفوّق يجب أن تبقيا كلتاهما."""
    from arl.pareto import Point, frontier
    pts = [Point("دقيق-بطيء", 0.30, 50, 900), Point("سريع-أقل", 0.25, 3, 100)]
    assert len(frontier(pts)) == 2


def test_deploy_maps_decisions_to_engine_filters():
    from arl.core import Strategy
    from arl.deploy import to_analyzer
    s = Strategy("x", ["alef", "stem_light"], "discourse", 300, 0, 3)
    body = to_analyzer(s)
    flt = body["settings"]["analysis"]["analyzer"]["arabic_tuned"]["filter"]
    assert "arabic_normalization" in flt and "arabic_stem" in flt
    assert any("ngram" in f for f in flt)
    plain = to_analyzer(Strategy("y", ["alef"], "discourse", 300, 0, 0))
    assert "arabic_stem" not in plain["settings"]["analysis"]["analyzer"]["arabic_tuned"]["filter"]


def test_deploy_rationale_warns_about_ngram_at_scale():
    from arl.core import Strategy
    from arl.deploy import rationale
    txt = rationale(Strategy("x", ["stem_light"], "discourse", 300, 0, 3), {"score": 0.3})
    assert "تتلاشى" in txt and "أسقطها" in txt


def test_corpus_features_are_computed():
    from arl.core import load_data
    from arl.features import extract, compare_to_reference
    c, _ = load_data()
    f = extract(c)
    assert f.n_docs == len(c) and f.vocab > 0 and 0 < f.type_token_ratio <= 1
    assert isinstance(compare_to_reference(f), str)


# ---------------------------------------------------------------- مسارات يراها المستخدم
def test_pareto_report_names_the_dominator():
    """تقرير باريتو هو ما يقرأه المستخدم فعلًا — لا الدالة."""
    from arl.pareto import Point, report
    txt = report([Point("جيد", 0.30, 5, 100), Point("رديء", 0.10, 50, 900)],
                 max_latency_ms=10, max_memory_mb=200)
    assert "جبهة باريتو" in txt and "مُهيمَنٌ عليها" in txt
    assert "يتفوّق عليها جيد" in txt          # يسمّي من هزمه
    assert "★ جيد" in txt


def test_pareto_report_says_so_when_budget_is_impossible():
    from arl.pareto import Point, report
    txt = report([Point("بطيء", 0.9, 500, 5000)], max_latency_ms=1)
    assert "لا تهيئة تفي بالقيود" in txt      # نتيجة لا خطأ


def test_deploy_export_writes_settings_and_rationale(tmp_path):
    import json
    from arl.deploy import export
    cfg = tmp_path / "best.json"
    cfg.write_text(json.dumps({"strategy": {"name": "r", "norm_ops": ["alef", "stem_light"],
                                            "chunk_mode": "discourse", "chunk_size": 300,
                                            "overlap": 0, "ngram": 3},
                               "score": 0.357, "metric": "recall@1", "n_docs": 1500}),
                   encoding="utf-8")
    s_path, r_path = export(str(cfg), "opensearch", str(tmp_path / "out"))
    body = json.loads(open(s_path, encoding="utf-8").read())
    assert "arabic_stem" in body["settings"]["analysis"]["analyzer"]["arabic_tuned"]["filter"]
    txt = open(r_path, encoding="utf-8").read()
    assert "0.357" in txt and "recall@1" in txt and "1500" in txt   # الدليل مرافق للقرار


def test_deploy_rejects_unknown_target():
    import pytest
    from arl.core import Strategy
    from arl.deploy import to_analyzer
    with pytest.raises(ValueError):
        to_analyzer(Strategy("x"), target="solr")


def test_llm_offline_never_raises_and_returns_empty():
    """الوضع الاستكشافي هو الافتراضي — يجب ألا ينهار أبدًا."""
    from arl.llm import LLM
    llm = LLM()
    assert not llm.online and llm.ask("s", "u") == "" and llm.calls == 0


def test_parse_json_survives_fenced_and_chatty_output():
    """النماذج تُغلّف JSON بأسوار وشرح — هذا أكثر مصدر أعطال في هذي الأدوات."""
    from arl.llm import parse_json
    assert parse_json('```json\n[{"a":1}]\n```', []) == [{"a": 1}]
    assert parse_json('طبعًا! إليك النتيجة:\n[{"a":2}]\nأتمنى أن تفيدك.', []) == [{"a": 2}]
    assert parse_json("نصٌّ بلا JSON إطلاقًا", "افتراضي") == "افتراضي"
    assert parse_json("", None) is None


def test_strategist_falls_back_to_mutation_without_a_model():
    from arl.agents import StrategistAgent
    from arl.llm import LLM
    props = StrategistAgent(LLM()).propose([], "", n=4)
    assert len(props) == 4
    assert props[0].ngram != 0                     # مرشّح التمثيل وحده حاضر دائمًا
    assert len({p.key() for p in props}) == 4      # لا تكرار


def test_reflection_diagnoses_from_failures_without_a_model():
    from arl.agents import ReflectionAgent
    from arl.core import Strategy, evaluate
    from arl.llm import LLM
    r = evaluate(Strategy("weak", [], "chars", 220, 40, 0), k=1)
    txt = ReflectionAgent(LLM()).diagnose(r)
    assert isinstance(txt, str) and len(txt) > 10


def test_features_flag_a_divergent_corpus():
    from arl.features import CorpusFeatures, compare_to_reference
    weird = CorpusFeatures(n_docs=500_000, total_chars=1, avg_doc_chars=1, median_doc_chars=1,
                           vocab=1, tokens=1, type_token_ratio=0.9, hapax_ratio=0.5,
                           avg_word_len=4, diacritics_ratio=0.4, latin_ratio=0.5,
                           digit_ratio=0.01, al_prefix_ratio=0.01)
    note = compare_to_reference(weird)
    assert "أكبر" in note and "لاتيني" in note and "تشكيل" in note


def test_profile_markdown_table_is_wellformed():
    from arl.core import Strategy, load_data
    from arl.profile import as_markdown, profile
    c, q = load_data()
    md = as_markdown([profile(Strategy("t", ["collapse_ws"], "discourse", 200, 0, 2),
                              c, q, k=3, sample_queries=5)])
    lines = md.strip().split("\n")
    assert lines[0].count("|") == lines[2].count("|")     # الرأس والصف متطابقان


def test_verdict_refuses_to_declare_a_winner_on_noise():
    from arl.stats import paired_bootstrap, verdict
    same = [1.0] * 50 + [0.0] * 50
    txt = verdict(paired_bootstrap(same, list(same), n_boot=300), "أ", "ب")
    assert "لا تنشر أن أحدهما أفضل" in txt
