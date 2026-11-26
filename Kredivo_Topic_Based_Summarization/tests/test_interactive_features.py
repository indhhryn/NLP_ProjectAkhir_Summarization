import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kredivo_project.explorer import (
    compare_summarizers_on_docs,
    filter_reviews,
    get_available_filter_options,
    get_evaluation_status_summary,
    parse_keyword_scores,
)
from kredivo_project.inference import (
    analyze_new_reviews,
    parse_manual_reviews,
    validate_and_prepare_uploaded_csv,
)
from kredivo_project.summarization import MODEL_NAMES


class _StubTopicModel:
    """Deterministic BERTopic stub simulating BaseCluster cosine-similarity inference."""

    def __init__(self):
        self.hdbscan_model = type("BaseCluster", (), {})()

    def transform(self, docs):
        topics, sims = [], []
        for text in docs:
            low = str(text).casefold()
            if "login" in low or "otp" in low:
                topics.append(4)
                sims.append(0.78)
            elif "limit" in low:
                topics.append(2)
                sims.append(0.74)
            else:
                topics.append(0)
                sims.append(0.29)  # below default 0.42 -> outlier
        return np.array(topics), np.array(sims)

    def get_topics(self):
        return {
            -1: [("outlier", 0.1)],
            2: [("limit", 0.6), ("naik", 0.4)],
            4: [("login", 0.7), ("otp", 0.5)],
        }

    def get_topic(self, tid):
        return self.get_topics().get(int(tid), [])


def _stub_bundle():
    return {
        "topic_model": _StubTopicModel(),
        "embedding_model": None,
        "uses_cosine_fallback": True,
    }


def _stub_topics_and_keywords():
    topics_df = pd.DataFrame([
        {"topic_id": 2, "topic_label": "limit | naik"},
        {"topic_id": 4, "topic_label": "login | otp"},
    ])
    keywords_df = pd.DataFrame([
        {"topic_id": 2, "keywords": "limit | naik"},
        {"topic_id": 4, "keywords": "login | otp"},
    ])
    return topics_df, keywords_df


def test_analyze_single_valid_review_does_not_force_multidoc_summary():
    topics_df, kw_df = _stub_topics_and_keywords()
    df = parse_manual_reviews("Aplikasi gagal login terus dan kode OTP tidak masuk.")
    res = analyze_new_reviews(df, _stub_bundle(), topics_df, kw_df, cfg={"inference_min_similarity": 0.42})

    assert res["status"] == "completed"
    assert res["counts"]["total_input"] == 1
    assert res["counts"]["valid_reviews"] == 1
    assert res["counts"]["assigned_reviews"] == 1
    assert res["topic_summary_status"][4]["can_summarize"] is False
    assert res["topic_summary_status"][4]["status"] == "insufficient_documents"
    assert res["summaries"].empty


def test_analyze_multiple_reviews_handles_duplicates_outliers_generic_and_special_chars():
    topics_df, kw_df = _stub_topics_and_keywords()
    raw = pd.DataFrame([
        {"reviewId": "u1", "content": "Gagal login di aplikasi, kode OTP tidak masuk!! \ufffd", "score": 1},
        {"reviewId": "u2", "content": "Gagal login di aplikasi, kode OTP tidak masuk!!", "score": 1},  # duplicate after clean
        {"reviewId": "u3", "content": "Saya tidak bisa login karena OTP bermasalah sejak update.", "score": 2},
        {"reviewId": "u4", "content": "   ", "score": 3},  # empty/invalid
        {"reviewId": "u5", "content": "!!! ???", "score": 3},  # no tokens
        {"reviewId": "u6", "content": "bagus 😊", "score": 5},  # generic praise (-2)
        {"reviewId": "u7", "content": "Resep masakan rendang pedas mantap", "score": 4},  # outlier (-1)
    ])
    res = analyze_new_reviews(raw, _stub_bundle(), topics_df, kw_df, cfg={"inference_min_similarity": 0.42, "summary_sentences": 2})

    c = res["counts"]
    assert c["total_input"] == 7
    assert c["valid_reviews"] == 5
    assert c["invalid_reviews"] == 2
    assert c["duplicate_extra_rows"] == 1
    assert c["generic_praise_reviews"] == 1
    assert c["outlier_reviews"] == 1
    assert c["assigned_reviews"] == 3
    assert res["topic_summary_status"][4]["can_summarize"] is True
    assert set(res["summaries"]["model"]) == set(MODEL_NAMES)

    # Traceability: every selected sentence maps to real input reviewId(s)
    valid_ids = {"u1", "u2", "u3"}
    for raw_ids in res["summary_sources"]["source_review_ids"]:
        ids = set(json.loads(raw_ids))
        assert ids and ids.issubset(valid_ids)


def test_analyze_all_empty_reviews_returns_no_valid_reviews():
    topics_df, kw_df = _stub_topics_and_keywords()
    df = parse_manual_reviews("   \n  ")
    res = analyze_new_reviews(df, _stub_bundle(), topics_df, kw_df)
    assert res["status"] == "no_valid_reviews"
    assert res["counts"]["valid_reviews"] == 0
    assert res["counts"]["invalid_reviews"] == 1


def test_uploaded_csv_validation_and_optional_columns():
    # Missing content column must fail clearly
    bad_csv = io.BytesIO(b"id,ulasan\n1,bagus\n")
    with pytest.raises(ValueError, match="content"):
        validate_and_prepare_uploaded_csv(bad_csv)

    # CSV without reviewId generates deterministic session IDs and preserves optional columns
    good_csv = io.BytesIO("content,score,sentiment,issue_category\nLimit turun terus,1,negative,limit_approval\n".encode("utf-8"))
    prepared, meta = validate_and_prepare_uploaded_csv(good_csv)
    assert meta["generated_temp_ids"] is True
    assert meta["optional_columns_found"] == ["score", "sentiment", "issue_category"]
    assert prepared.loc[0, "reviewId"] == "upload-0001"
    assert prepared.loc[0, "content"] == "Limit turun terus"


def test_compare_summarizers_on_same_docs_and_missing_rouge_status(tmp_path):
    docs = pd.DataFrame([
        {"doc_id": "d1", "content": "Gagal login aplikasi karena kode OTP tidak masuk.", "source_review_ids": json.dumps(["r1"]), "duplicate_count": 1},
        {"doc_id": "d2", "content": "Kode OTP tidak diterima saat login.", "source_review_ids": json.dumps(["r2"]), "duplicate_count": 1},
        {"doc_id": "d3", "content": "Aplikasi error ketika verifikasi login.", "source_review_ids": json.dumps(["r3"]), "duplicate_count": 1},
    ])
    cmp_res = compare_summarizers_on_docs(docs, keywords=["login", "otp"], n_sentences=2, topic_id=4)
    assert cmp_res["can_compare"] is True
    assert set(cmp_res["comparison_table"]["model"]) == set(MODEL_NAMES)
    assert (cmp_res["comparison_table"]["source_traceability_rate"] == 1.0).all()
    assert not cmp_res["sentence_overlap"].empty

    # Empty reference summaries must never yield fabricated ROUGE
    pd.DataFrame([{"topic_id": 4, "reference_summary": ""}]).to_csv(tmp_path / "reference_summaries_template.csv", index=False)
    eval_status = get_evaluation_status_summary(tmp_path)
    assert eval_status["rouge_available"] is False
    assert eval_status["comparison_df"].empty


def test_review_filters_respect_audit_gate_and_preserve_traceability():
    reviews = pd.DataFrame([
        {
            "reviewId": "r1", "content": "Gagal login karena OTP tidak masuk", "score": 1,
            "sentiment": "negative", "issue_category": "login_biometric", "severity_level": "high",
            "doc_id": "d1", "topic_id": 4, "topic_split": "train", "topic_keywords": "login | otp",
        },
        {
            "reviewId": "r2", "content": "Limit naik cepat sekali", "score": 5,
            "sentiment": "positive", "issue_category": "limit_approval", "severity_level": "none",
            "doc_id": "d2", "topic_id": 2, "topic_split": "test", "topic_keywords": "limit | naik",
        },
        {
            "reviewId": "r3", "content": "Resep masakan", "score": 3,
            "sentiment": "neutral", "issue_category": "general_satisfaction", "severity_level": "none",
            "doc_id": "d3", "topic_id": -1, "topic_split": "validation", "topic_keywords": "",
        },
    ])
    topics = pd.DataFrame([{"topic_id": 2, "topic_label": "limit | naik"}, {"topic_id": 4, "topic_label": "login | otp"}])

    opts_unverified = get_available_filter_options(reviews, topics, {"passed": False})
    assert opts_unverified["severity_audit_passed"] is False
    assert opts_unverified["severities"] == []

    # Severity filter must not activate when audit failed
    res_unverified = filter_reviews(reviews, severities=["high"], severity_audit_passed=False)
    assert len(res_unverified) == 3

    # Text search + topic filter preserves reviewId link
    res_filtered = filter_reviews(reviews, search_query="OTP", topic_ids=[4])
    assert list(res_filtered["reviewId"]) == ["r1"]

    # Verified audit unlocks severity filter
    opts_verified = get_available_filter_options(reviews, topics, {"passed": True})
    assert opts_verified["severity_audit_passed"] is True
    assert set(opts_verified["severities"]) == {"high", "none"}
    res_sev = filter_reviews(reviews, severities=["high"], severity_audit_passed=True)
    assert list(res_sev["reviewId"]) == ["r1"]


def test_parse_keyword_scores():
    df = parse_keyword_scores(json.dumps([["limit", 0.6], ["naik", 0.45]]))
    assert list(df["keyword"]) == ["limit", "naik"]
    assert df["ctfidf_score"].tolist() == [0.6, 0.45]
    assert parse_keyword_scores("").empty


def test_real_saved_bertopic_artifact_inference_if_present():
    model_dir = ROOT / "outputs" / "bertopic_model"
    if not model_dir.exists():
        pytest.skip("Artifact outputs/bertopic_model belum tersedia.")

    from kredivo_project.config import OUTPUT_DIR
    from kredivo_project.inference import load_inference_bundle

    bundle = load_inference_bundle(model_dir)
    topics_df = pd.read_csv(OUTPUT_DIR / "topics.csv", keep_default_na=False)
    kw_df = pd.read_csv(OUTPUT_DIR / "topic_keywords_ctfidf.csv", keep_default_na=False)

    sample = pd.DataFrame([
        {"reviewId": "real-1", "content": "Limit saya tidak bisa dipakai padahal sudah lunas semua tagihannya."},
        {"reviewId": "real-2", "content": "Limit kredivo tiba-tiba turun padahal tidak pernah telat bayar."},
        {"reviewId": "real-3", "content": "Bunga terlalu tinggi dan mencekik nasabah."},
        {"reviewId": "real-4", "content": "bagus"},
        {"reviewId": "real-5", "content": "Resep nasi goreng enak pedas manis di jakarta"},
    ])
    res = analyze_new_reviews(sample, bundle, topics_df, kw_df)
    assert res["status"] == "completed"
    assigned = res["reviews_assigned"].set_index("reviewId")
    assert int(assigned.loc["real-4", "topic_id"]) == -2
    assert int(assigned.loc["real-5", "topic_id"]) == -1
    assert int(assigned.loc["real-1", "topic_id"]) >= 0
    assert int(assigned.loc["real-3", "topic_id"]) >= 0

