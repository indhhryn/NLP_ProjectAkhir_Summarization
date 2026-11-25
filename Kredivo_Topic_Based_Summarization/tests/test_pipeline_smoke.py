import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kredivo_project import pipeline


def test_pipeline_writes_expected_artifacts_with_deterministic_topic_stub(tmp_path, monkeypatch):
    rows = []
    for i in range(60):
        if i < 30:
            content = f"Pembayaran tagihan aplikasi berhasil diproses dengan cepat nomor {i}."
            category = "payment_reconciliation"
            severity = "high"
        else:
            content = f"Persetujuan limit kredit aplikasi perlu ditingkatkan nomor {i}."
            category = "limit_approval"
            severity = "medium"
        rows.append({
            "reviewId": f"review-{i}", "content": content,
            "score": 3, "at": "2026-01-01", "issue_category": category,
            "severity_level": severity, "sentiment": "neutral", "evidence_span": "aplikasi",
        })
    for i in range(4):
        rows.append({
            "reviewId": f"praise-{i}", "content": "bagus", "score": 5,
            "at": "2026-01-01", "issue_category": "general_satisfaction",
            "severity_level": "none", "sentiment": "positive", "evidence_span": "bagus",
        })
    input_path = tmp_path / "reviews.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    output_dir = tmp_path / "outputs"

    cfg = {
        "dataset_path": str(input_path), "text_column": "content", "id_column": "reviewId",
        "test_size": 0.1, "validation_size": 0.1, "random_state": 42, "max_fit_docs": 50,
        "transform_batch_size": 32, "embedding_model": "unused-in-test", "device": "cpu",
        "min_topic_size": 5, "umap_n_neighbors": 5, "umap_n_components": 2,
        "n_keywords": 5, "summary_sentences": 2, "mmr_lambda": 0.72,
        "textrank_neighbors": 5, "reference_topics": 2,
        "reference_examples_per_topic": 5, "audit_sample_size": 12,
        "audit_min_agreement": 0.8, "stopword_removal": True, "save_topic_model": False,
    }
    monkeypatch.setattr(pipeline, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(pipeline, "resolve_path", lambda value: input_path)
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", output_dir)

    def fake_fit_and_assign_topics(docs, config, out):
        assigned = docs.copy()
        assigned["topic_id"] = -2
        assigned["topic_split"] = "excluded_generic"
        assigned["keywords"] = ""
        for idx in assigned.index[~assigned["is_generic_praise"].astype(bool)]:
            row = assigned.loc[idx]
            tid = 0 if "pembayaran" in row["normalized_text"] else 1
            assigned.loc[idx, "topic_id"] = tid
            assigned.loc[idx, "topic_split"] = "test" if idx % 5 == 0 else ("validation" if idx % 5 == 1 else "train")
            assigned.loc[idx, "keywords"] = "pembayaran | aplikasi" if tid == 0 else "limit | kredit"
        topic_rows = []
        keyword_rows = []
        for tid, group in assigned[assigned["topic_id"] >= 0].groupby("topic_id"):
            topic_rows.append({
                "topic_id": int(tid), "unique_documents": int(group["doc_id"].nunique()),
                "total_source_reviews": int(group["duplicate_count"].sum()),
                "test_documents": int((group["topic_split"] == "test").sum()),
                "validation_documents": int((group["topic_split"] == "validation").sum()),
                "train_documents": int((group["topic_split"] == "train").sum()),
                "topic_label": "pembayaran | aplikasi" if int(tid) == 0 else "limit | kredit",
                "keywords": "pembayaran | aplikasi" if int(tid) == 0 else "limit | kredit",
            })
            keyword_rows.append({"topic_id": int(tid), "keywords": "pembayaran | aplikasi" if int(tid) == 0 else "limit | kredit"})
        return assigned, object(), pd.DataFrame(topic_rows), pd.DataFrame(keyword_rows)

    monkeypatch.setattr(pipeline, "fit_and_assign_topics", fake_fit_and_assign_topics)
    status = pipeline.run_pipeline()

    assert status["pipeline_status"] == "completed"
    assert status["topics_created"] == 2
    for name in [
        "data_quality_report.json", "unique_documents.csv", "document_topic_assignments.csv",
        "topics.csv", "topic_keywords_ctfidf.csv", "reviews_assigned.csv",
        "summary_sources.csv", "summaries.csv", "reference_summaries_template.csv",
        "reference_examples.csv", "human_evaluation_template.csv", "label_audit_sample.csv",
        "label_audit_status.json", "topic_quality_review_template.csv", "run_status.json",
    ]:
        assert (output_dir / name).is_file(), name
    refs = pd.read_csv(output_dir / "reference_summaries_template.csv", keep_default_na=False)
    assert refs["evaluation_doc_ids"].map(lambda raw: len(json.loads(raw)) > 0).all()
