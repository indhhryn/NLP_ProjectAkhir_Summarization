import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kredivo_project import topics

class FakeTopicModel:
    def fit_transform(self, docs):
        return [i % 3 for i in range(len(docs))], None
    def transform(self, docs):
        return [i % 3 for i in range(len(docs))], None
    def get_topic_info(self):
        return pd.DataFrame({'Topic':[0,1,2,-1], 'Count':[50,50,50,2], 'Name':['0_login_otp','1_limit_approval','2_payment','-1_outlier']})
    def get_topic(self, tid):
        return [(f'term{tid}',0.1),('aplikasi',0.05)]

def test_fit_assign_mapping_and_generic_exclusion(monkeypatch, tmp_path):
    rows=[]
    for i in range(300):
        rows.append({'doc_id':f'd{i}','normalized_text':f'review kendala transaksi aplikasi kata {i}',
                     'content':f'review kendala transaksi aplikasi kata {i}',
                     'source_review_ids':f'["r{i}"]','duplicate_count':1,'source_review_count':1,
                     'is_generic_praise':False})
    for i in range(5):
        rows.append({'doc_id':f'g{i}','normalized_text':'bagus','content':'bagus',
                     'source_review_ids':f'["g{i}"]','duplicate_count':1,'source_review_count':1,
                     'is_generic_praise':True})
    monkeypatch.setattr(topics, '_build_model', lambda cfg: FakeTopicModel())
    cfg={'min_topic_size':10,'test_size':0.1,'validation_size':0.1,'random_state':42,'max_fit_docs':60,
         'transform_batch_size':40,'n_keywords':5,'save_topic_model':False}
    assigned, model, topic_table, kw = topics.fit_and_assign_topics(pd.DataFrame(rows),cfg,tmp_path)
    assert len(assigned)==305
    assert (assigned.loc[assigned['is_generic_praise'],'topic_id']==-2).all()
    assert (assigned.loc[~assigned['is_generic_praise'],'topic_id']>=0).all()
    assert set(topic_table['topic_id'])=={0,1,2}
    assert len(kw)==3


def test_reference_template_pins_exact_evaluation_documents(tmp_path):
    import json
    import pandas as pd
    from kredivo_project.evaluation import create_reference_templates

    assignments = pd.DataFrame([
        {"doc_id": f"d{i}", "topic_id": 2, "topic_split": "test", "content": f"Ulasan uji nomor {i} tentang pembayaran digital.",
         "source_review_ids": json.dumps([f"review-{i}"]), "duplicate_count": 1}
        for i in range(20)
    ] + [
        {"doc_id": f"train{i}", "topic_id": 2, "topic_split": "train", "content": "Data train tidak masuk referensi uji.",
         "source_review_ids": json.dumps([f"train-review-{i}"]), "duplicate_count": 1}
        for i in range(10)
    ])
    topics = pd.DataFrame([{"topic_id": 2, "topic_label": "pembayaran | digital"}])
    cfg = {"reference_topics": 1, "reference_examples_per_topic": 7, "random_state": 42}
    create_reference_templates(assignments, topics, tmp_path, cfg)

    refs = pd.read_csv(tmp_path / "reference_summaries_template.csv", keep_default_na=False)
    examples = pd.read_csv(tmp_path / "reference_examples.csv", keep_default_na=False)
    pinned_ids = set(json.loads(refs.loc[0, "evaluation_doc_ids"]))
    assert len(pinned_ids) == 7
    assert pinned_ids == set(examples["doc_id"].astype(str))
    assert all(x.startswith("d") for x in pinned_ids)


def test_evaluation_uses_same_pinned_sample_as_reference(tmp_path, monkeypatch):
    import json
    import sys
    import types
    import pandas as pd
    from kredivo_project.evaluation import create_reference_templates, run_evaluation

    assignments = pd.DataFrame([
        {"doc_id": f"d{i}", "topic_id": 2, "topic_split": "test", "content": f"Keluhan pembayaran aplikasi bermasalah nomor {i}.",
         "normalized_text": f"keluhan pembayaran aplikasi bermasalah nomor {i}",
         "source_review_ids": json.dumps([f"review-{i}"]), "source_review_count": 1, "duplicate_count": 1}
        for i in range(20)
    ] + [
        {"doc_id": f"train{i}", "topic_id": 2, "topic_split": "train", "content": "Data train tidak boleh masuk evaluasi.",
         "normalized_text": f"data train tidak boleh masuk evaluasi {i}",
         "source_review_ids": json.dumps([f"train-review-{i}"]), "source_review_count": 1, "duplicate_count": 1}
        for i in range(10)
    ])
    topics = pd.DataFrame([{"topic_id": 2, "topic_label": "pembayaran"}])
    keywords = pd.DataFrame([{"topic_id": 2, "keywords": "pembayaran | aplikasi"}])
    cfg = {"reference_topics": 1, "reference_examples_per_topic": 7, "random_state": 42,
           "summary_sentences": 2, "mmr_lambda": 0.72}
    create_reference_templates(assignments, topics, tmp_path, cfg)
    refs_path = tmp_path / "reference_summaries_template.csv"
    refs = pd.read_csv(refs_path, keep_default_na=False)
    refs.loc[0, "reference_summary"] = "Ulasan membahas kendala pembayaran pada aplikasi."
    refs.to_csv(refs_path, index=False)

    class FakeScorer:
        def __init__(self, *args, **kwargs): pass
        def score(self, reference, prediction):
            item = types.SimpleNamespace(fmeasure=0.5)
            return {"rouge1": item, "rouge2": item, "rougeL": item}
    pkg = types.ModuleType("rouge_score")
    pkg.rouge_scorer = types.SimpleNamespace(RougeScorer=FakeScorer)
    monkeypatch.setitem(sys.modules, "rouge_score", pkg)

    result = run_evaluation(assignments, topics, keywords, tmp_path, cfg)
    pred = pd.read_csv(tmp_path / "evaluation_predictions.csv")
    assert result["status"] == "completed"
    assert set(pred["test_unique_review_count"]) == {7}
    assert pred["source_review_count"].max() == 7


def test_reference_summary_is_cleared_when_evaluation_documents_change(tmp_path):
    import json
    import pandas as pd
    from kredivo_project.evaluation import create_reference_templates

    def data(prefix):
        assignments = pd.DataFrame([
            {"doc_id": f"{prefix}{i}", "topic_id": 3, "topic_split": "test",
             "content": f"Review {prefix}{i} membahas kendala transaksi aplikasi.",
             "source_review_ids": json.dumps([f"review-{prefix}{i}"]), "duplicate_count": 1}
            for i in range(10)
        ])
        return assignments

    topics = pd.DataFrame([{"topic_id": 3, "topic_label": "transaksi | aplikasi"}])
    cfg = {"reference_topics": 1, "reference_examples_per_topic": 3, "random_state": 42}
    create_reference_templates(data("a"), topics, tmp_path, cfg)
    refs_path = tmp_path / "reference_summaries_template.csv"
    refs = pd.read_csv(refs_path, keep_default_na=False)
    refs.loc[0, "reference_summary"] = "Ringkasan acuan yang ditulis manusia."
    refs.to_csv(refs_path, index=False)

    create_reference_templates(data("b"), topics, tmp_path, cfg)
    refs_after = pd.read_csv(refs_path, keep_default_na=False)
    assert refs_after.loc[0, "reference_summary"] == ""


def test_topic_quality_template_uses_validation_only(tmp_path):
    import json
    import pandas as pd
    from kredivo_project.evaluation import create_topic_quality_template

    assignments = pd.DataFrame([
        {"doc_id": "v1", "topic_id": 4, "topic_split": "validation", "content": "Login gagal pada aplikasi.", "source_review_ids": json.dumps(["rv1"])},
        {"doc_id": "v2", "topic_id": 4, "topic_split": "validation", "content": "Kode OTP tidak diterima.", "source_review_ids": json.dumps(["rv2"])},
        {"doc_id": "t1", "topic_id": 4, "topic_split": "test", "content": "JANGAN MASUK KE TEMPLATE TEST.", "source_review_ids": json.dumps(["rt1"])},
    ])
    topics = pd.DataFrame([{"topic_id": 4, "topic_label": "login | otp", "keywords": "login | otp"}])
    create_topic_quality_template(assignments, topics, tmp_path, {"reference_topics": 5, "random_state": 42})
    result = pd.read_csv(tmp_path / "topic_quality_review_template.csv", keep_default_na=False)
    assert len(result) == 1
    assert "rv1" in result.loc[0, "sample_review_ids"] or "rv2" in result.loc[0, "sample_review_ids"]
    assert "rt1" not in result.loc[0, "sample_review_ids"]
    assert "JANGAN MASUK" not in result.loc[0, "sample_reviews"]


def test_evaluation_fails_closed_when_reference_doc_ids_are_empty(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import pytest
    from kredivo_project import evaluation

    assignments = pd.DataFrame([
        {"doc_id": "d1", "topic_id": 2, "topic_split": "test", "content": "Review kendala pembayaran aplikasi.",
         "source_review_ids": json.dumps(["r1"]), "duplicate_count": 1, "source_review_count": 1,
         "source_variants": json.dumps([{"content": "Review kendala pembayaran aplikasi.", "source_review_ids": ["r1"], "occurrence_count": 1}])}
    ])
    refs = pd.DataFrame([{
        "topic_id": 2, "topic_label": "pembayaran", "reference_summary": "Ringkasan acuan manusia.",
        "evaluation_doc_ids": ""
    }])
    refs.to_csv(tmp_path / "reference_summaries_template.csv", index=False)
    topics = pd.DataFrame([{"topic_id": 2, "keywords": "pembayaran"}])
    keywords = pd.DataFrame([{"topic_id": 2, "keywords": "pembayaran"}])
    monkeypatch.setattr(evaluation, "_rouge_scores", lambda reference, prediction: {"rouge1": 0.1, "rouge2": 0.1, "rougeL": 0.1})
    with pytest.raises(ValueError, match="evaluation_doc_ids kosong"):
        evaluation.run_evaluation(assignments, topics, keywords, tmp_path, {"summary_sentences": 2, "mmr_lambda": 0.7})


def test_human_eval_sheet_contains_prediction_and_invalidates_stale_scores(tmp_path, monkeypatch):
    import json
    import sys
    import types
    import pandas as pd
    from kredivo_project.evaluation import create_reference_templates, run_evaluation

    assignments = pd.DataFrame([
        {"doc_id": f"d{i}", "topic_id": 2, "topic_split": "test",
         "content": f"Keluhan pembayaran aplikasi bermasalah dengan proses nomor {i}.",
         "source_review_ids": json.dumps([f"review-{i}"]), "source_review_count": 1, "duplicate_count": 1}
        for i in range(8)
    ])
    topics = pd.DataFrame([{"topic_id": 2, "topic_label": "pembayaran", "keywords": "pembayaran | aplikasi"}])
    keywords = pd.DataFrame([{"topic_id": 2, "keywords": "pembayaran | aplikasi"}])
    cfg = {"reference_topics": 1, "reference_examples_per_topic": 5, "random_state": 42,
           "summary_sentences": 2, "mmr_lambda": 0.72}
    create_reference_templates(assignments, topics, tmp_path, cfg)
    refs_path = tmp_path / "reference_summaries_template.csv"
    refs = pd.read_csv(refs_path, keep_default_na=False)
    refs.loc[0, "reference_summary"] = "Review menyampaikan kendala pembayaran pada aplikasi."
    refs.to_csv(refs_path, index=False)

    class FakeScorer:
        def __init__(self, *args, **kwargs): pass
        def score(self, reference, prediction):
            item = types.SimpleNamespace(fmeasure=0.5)
            return {"rouge1": item, "rouge2": item, "rougeL": item}
    pkg = types.ModuleType("rouge_score")
    pkg.rouge_scorer = types.SimpleNamespace(RougeScorer=FakeScorer)
    monkeypatch.setitem(sys.modules, "rouge_score", pkg)

    run_evaluation(assignments, topics, keywords, tmp_path, cfg)
    human_path = tmp_path / "human_evaluation_template.csv"
    human = pd.read_csv(human_path, keep_default_na=False)
    assert human["prediction"].str.len().gt(0).all()
    assert human["reference_summary"].str.len().gt(0).all()
    assert human["output_fingerprint"].str.len().eq(64).all()
    human.loc[0, "groundedness_1_to_5"] = 5
    human.loc[0, "reviewer"] = "R1"
    human.to_csv(human_path, index=False)

    # Identical rerun keeps the rating because the exact prediction is unchanged.
    run_evaluation(assignments, topics, keywords, tmp_path, cfg)
    same = pd.read_csv(human_path, keep_default_na=False)
    assert str(same.loc[0, "groundedness_1_to_5"]) == "5"

    # Changed source content changes the fingerprint and resets old scores.
    changed = assignments.copy()
    changed["content"] = changed["content"].str.replace("Keluhan pembayaran", "Transaksi gagal", regex=False)
    run_evaluation(changed, topics, keywords, tmp_path, cfg)
    updated = pd.read_csv(human_path, keep_default_na=False)
    assert updated.loc[0, "groundedness_1_to_5"] == ""
