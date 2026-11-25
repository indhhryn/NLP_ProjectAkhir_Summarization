from __future__ import annotations
import json
import logging
from pathlib import Path
import pandas as pd
from .config import ROOT, OUTPUT_DIR, load_config, resolve_path
from .data import load_reviews, make_unique_documents, build_quality_report
from .topics import fit_and_assign_topics
from .summarization import summarize_topic, build_summary_texts
from .audit import create_audit_sample, evaluate_label_audit, save_audit_status
from .evaluation import create_reference_templates, create_topic_quality_template, run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("kredivo_pipeline")


def run_pipeline(config_path: Path | None = None) -> dict:
    cfg = load_config(config_path)
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    dataset_path = resolve_path(cfg["dataset_path"])
    LOGGER.info("Membaca dataset: %s", dataset_path)
    reviews = load_reviews(dataset_path)
    LOGGER.info("Jumlah review valid: %s", len(reviews))
    docs = make_unique_documents(reviews)
    quality = build_quality_report(reviews, docs)
    (out / "data_quality_report.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "data_quality_report.md").write_text(_quality_markdown(quality), encoding="utf-8")
    docs.to_csv(out / "unique_documents.csv", index=False, encoding="utf-8-sig")

    LOGGER.info("Menjalankan BERTopic pada dokumen unik informatif...")
    assignments, topic_model, topics, keywords = fit_and_assign_topics(docs, cfg, out)
    assignments.to_csv(out / "document_topic_assignments.csv", index=False, encoding="utf-8-sig")
    topics.to_csv(out / "topics.csv", index=False, encoding="utf-8-sig")
    keywords.to_csv(out / "topic_keywords_ctfidf.csv", index=False, encoding="utf-8-sig")

    # Kembalikan label topic ke level ulasan asli agar setiap reviewId dapat dilacak.
    doc_map = assignments.set_index("doc_id")[["topic_id", "topic_split", "keywords"]].to_dict("index")
    reviews_out = reviews.copy()
    reviews_out["doc_id"] = reviews_out["normalized_text"].map(lambda x: _hash_text(x))
    reviews_out["topic_id"] = reviews_out["doc_id"].map(lambda x: doc_map.get(x, {}).get("topic_id", -2))
    reviews_out["topic_split"] = reviews_out["doc_id"].map(lambda x: doc_map.get(x, {}).get("topic_split", "excluded"))
    reviews_out["topic_keywords"] = reviews_out["doc_id"].map(lambda x: doc_map.get(x, {}).get("keywords", ""))
    keep = [c for c in ["reviewId", "content", "score", "at", "sentiment", "issue_category", "severity_level", "doc_id", "topic_id", "topic_split", "topic_keywords"] if c in reviews_out]
    reviews_out[keep].to_csv(out / "reviews_assigned.csv", index=False, encoding="utf-8-sig")

    LOGGER.info("Membuat ringkasan untuk setiap topik...")
    source_frames = []
    for idx, row in topics[topics["topic_id"] >= 0].iterrows():
        tid = int(row["topic_id"])
        topic_docs = assignments[assignments["topic_id"] == tid]
        kw = [x.strip() for x in str(row.get("keywords", "")).split("|") if x.strip()]
        selected, _ = summarize_topic(topic_docs, n_sentences=int(cfg["summary_sentences"]),
                                      mmr_lambda=float(cfg["mmr_lambda"]), keywords=kw, topic_id=tid,
                                      textrank_neighbors=int(cfg["textrank_neighbors"]))
        if not selected.empty:
            source_frames.append(selected)
        if (len(source_frames) % 10) == 0:
            LOGGER.info("Topik selesai: %d", len(source_frames))
    sources = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame()
    summaries = build_summary_texts(sources)
    sources.to_csv(out / "summary_sources.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(out / "summaries.csv", index=False, encoding="utf-8-sig")

    # Sampling audit label tidak mengubah label sistem. Severity hanya aktif setelah audit manusia lulus.
    audit_sample_path = out / "label_audit_sample.csv"
    create_audit_sample(reviews, audit_sample_path, int(cfg["audit_sample_size"]), int(cfg["random_state"]))
    audit_status = evaluate_label_audit(
        audit_sample_path,
        float(cfg["audit_min_agreement"]),
        float(cfg.get("audit_min_class_agreement", 0.70)),
        int(cfg.get("audit_min_class_support", 10)),
    )
    save_audit_status(out / "label_audit_status.json", audit_status)

    create_reference_templates(assignments, topics, out, cfg)
    create_topic_quality_template(assignments, topics, out, cfg)
    eval_status = run_evaluation(assignments, topics, keywords, out, cfg)
    run_status = {
        "pipeline_status": "completed",
        "dataset_path": str(dataset_path),
        "raw_reviews": int(len(reviews)),
        "unique_documents": int(len(docs)),
        "informative_documents": int((~docs["is_generic_praise"]).sum()),
        "topics_created": int(topics.loc[topics["topic_id"] >= 0, "topic_id"].nunique()),
        "outlier_documents": int((assignments["topic_id"] == -1).sum()),
        "generic_praise_documents_excluded": int((assignments["topic_id"] == -2).sum()),
        "train_documents": int((assignments["topic_split"] == "train").sum()),
        "validation_documents": int((assignments["topic_split"] == "validation").sum()),
        "test_documents": int((assignments["topic_split"] == "test").sum()),
        "evaluation": eval_status,
        "label_audit": audit_status,
    }
    (out / "run_status.json").write_text(json.dumps(run_status, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Pipeline selesai. Hasil tersimpan di: %s", out)
    return run_status


def _hash_text(normalized: str) -> str:
    import hashlib
    return hashlib.sha1(str(normalized).encode("utf-8")).hexdigest()[:16]


def _quality_markdown(q: dict) -> str:
    lines = [
        "# Data Quality Audit — Kredivo Reviews",
        "",
        f"- Baris CSV awal: **{q.get('raw_csv_rows', q['raw_review_count']):,}**",
        f"- Baris kosong/tidak valid yang dilewati: **{q.get('dropped_invalid_rows', 0):,}**",
        f"- Jumlah review valid: **{q['raw_review_count']:,}**",
        f"- Dokumen unik setelah normalisasi: **{q['unique_normalized_reviews']:,}**",
        f"- Baris duplikat tambahan: **{q['duplicate_extra_rows']:,}** ({q['duplicate_extra_share']:.1%})",
        f"- Jenis teks pujian generik yang dikecualikan: **{q['generic_praise_unique_docs']:,}** ({q.get('generic_praise_review_count', 0):,} review asli)",
        f"- Dokumen unik informatif untuk clustering: **{q['informative_unique_docs']:,}** ({q.get('informative_review_count', 0):,} review asli)",
        "",
        "## Distribusi label (belum dianggap tervalidasi)",
    ]
    for key in ["sentiment_distribution", "issue_category_distribution", "severity_level_distribution"]:
        if key in q:
            lines.append(f"### {key}")
            lines.extend([f"- {k}: {v:,}" for k,v in q[key].items()])
    for label in ["sentiment", "issue_category", "severity_level"]:
        conflict_key = f"{label}_conflicting_duplicate_groups"
        if conflict_key in q:
            lines.append(f"- Kelompok teks duplikat dengan label {label} yang saling berbeda: **{q[conflict_key]:,}**")
    if "evidence_span_literal_match_rate" in q:
        lines.extend(["", "## Pemeriksaan evidence_span", f"- Substring literal cocok: {q['evidence_span_literal_match_count']:,}/{q['evidence_span_nonempty_count']:,} ({q['evidence_span_literal_match_rate']:.1%})", f"- Catatan: {q['evidence_span_note']}"])
    lines.extend(["", "> Label tambahan issue_category, sentiment, severity_level dan evidence_span tidak digunakan untuk membentuk topik atau ringkasan. Lakukan audit manual sebelum menggunakan severity filter."])
    return "\n".join(lines)
