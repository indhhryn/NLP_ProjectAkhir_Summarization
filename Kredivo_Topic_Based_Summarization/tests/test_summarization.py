import json
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kredivo_project.summarization import summarize_topic, build_summary_texts, MODEL_NAMES

def sample_docs():
    rows = [
        {"doc_id":"a", "content":"Aplikasi sering gagal saat login. Kode OTP tidak masuk.", "source_review_ids":json.dumps(["r1"]), "duplicate_count":1},
        {"doc_id":"b", "content":"Saya tidak bisa login karena OTP tidak diterima.", "source_review_ids":json.dumps(["r2"]), "duplicate_count":1},
        {"doc_id":"c", "content":"Login aplikasi gagal setelah pembaruan.", "source_review_ids":json.dumps(["r3"]), "duplicate_count":1},
        {"doc_id":"d", "content":"Aplikasi lambat ketika membuka halaman pembayaran.", "source_review_ids":json.dumps(["r4"]), "duplicate_count":1},
    ]
    return pd.DataFrame(rows)

def test_three_extractors_return_source_traceable_sentences():
    outputs, metrics = summarize_topic(sample_docs(), n_sentences=2, keywords=["login", "otp"], topic_id=4)
    assert set(outputs["model"]) == set(MODEL_NAMES)
    assert len(outputs) == 6
    assert outputs["source_review_ids"].map(lambda x: len(json.loads(x)) > 0).all()
    assert set(metrics) == set(MODEL_NAMES)
    assert all(0 <= m["keyword_coverage"] <= 1 for m in metrics.values())
    assert all(0 <= m["source_traceability_rate"] <= 1 for m in metrics.values())
    summaries = build_summary_texts(outputs)
    assert len(summaries) == 3
    assert summaries["summary"].str.len().gt(0).all()


def test_source_ids_are_not_shared_between_normalized_but_different_raw_sentences():
    from kredivo_project.data import make_unique_documents
    from kredivo_project.summarization import build_sentence_pool
    raw = pd.DataFrame([
        {"reviewId": "r-slang", "content": "Gak bisa login", "score": 1, "issue_category": "login_biometric", "severity_level": "high"},
        {"reviewId": "r-formal", "content": "Tidak bisa login", "score": 1, "issue_category": "login_biometric", "severity_level": "high"},
    ])
    from kredivo_project.preprocessing import normalize_text, is_generic_praise
    raw["normalized_text"] = raw["content"].map(normalize_text)
    docs = make_unique_documents(raw)
    assert len(docs) == 1
    pool = build_sentence_pool(docs)
    mapping = {row.sentence: set(json.loads(row.source_review_ids)) for row in pool.itertuples(index=False)}
    assert mapping["Gak bisa login"] == {"r-slang"}
    assert mapping["Tidak bisa login"] == {"r-formal"}


def test_vectorizer_relaxes_max_df_for_homogeneous_short_texts():
    # Common tokens can all be pruned at max_df=0.98 in tiny topic clusters.
    docs = pd.DataFrame([
        {"doc_id": "u1", "content": "sangat membantu sekali", "source_review_ids": json.dumps(["r1"]), "duplicate_count": 1},
        {"doc_id": "u2", "content": "sangat membantu sekali ..", "source_review_ids": json.dumps(["r2"]), "duplicate_count": 1},
        {"doc_id": "u3", "content": "sangat membantu sekali !!!", "source_review_ids": json.dumps(["r3"]), "duplicate_count": 1},
    ])
    outputs, metrics = summarize_topic(docs, n_sentences=2, keywords=["membantu"], topic_id=9)
    assert set(outputs["model"]) == set(MODEL_NAMES)
    assert len(outputs) > 0
    assert all(0 <= value["redundancy"] <= 1 for value in metrics.values())
