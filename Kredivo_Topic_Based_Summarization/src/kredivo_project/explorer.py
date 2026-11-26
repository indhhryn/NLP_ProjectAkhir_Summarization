from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .summarization import MODEL_NAMES, build_sentence_pool, build_summary_texts, summarize_topic


def compare_summarizers_on_docs(
    topic_docs: pd.DataFrame,
    keywords: list[str] | None = None,
    n_sentences: int = 5,
    mmr_lambda: float = 0.72,
    textrank_neighbors: int = 15,
    topic_id: int = 0,
    min_docs_for_summary: int = 2,
) -> dict:
    """Run all three extractive summarizers on the exact same documents for fair comparison."""
    if topic_docs is None or topic_docs.empty:
        return {
            "can_compare": False,
            "reason": "Tidak ada dokumen pada topik yang dipilih.",
            "sources": pd.DataFrame(),
            "summaries": pd.DataFrame(),
            "metrics": {},
            "comparison_table": pd.DataFrame(),
            "sentence_overlap": pd.DataFrame(),
            "candidate_sentence_count": 0,
            "unique_document_count": 0,
        }

    pool = build_sentence_pool(topic_docs)
    unique_doc_count = int(len(topic_docs))
    candidate_count = int(len(pool))

    if unique_doc_count < int(min_docs_for_summary) or candidate_count < 2:
        return {
            "can_compare": False,
            "reason": (
                f"Dokumen unik ({unique_doc_count}) atau kalimat kandidat ({candidate_count}) belum cukup "
                f"untuk perbandingan multi-document summarization (minimal {min_docs_for_summary} dokumen dan 2 kalimat)."
            ),
            "sources": pd.DataFrame(),
            "summaries": pd.DataFrame(),
            "metrics": {},
            "comparison_table": pd.DataFrame(),
            "sentence_overlap": pd.DataFrame(),
            "candidate_sentence_count": candidate_count,
            "unique_document_count": unique_doc_count,
        }

    target_n = min(max(1, int(n_sentences)), candidate_count)
    sources, metrics = summarize_topic(
        topic_docs,
        n_sentences=target_n,
        mmr_lambda=float(mmr_lambda),
        keywords=keywords or [],
        topic_id=int(topic_id),
        textrank_neighbors=int(textrank_neighbors),
    )
    summaries = build_summary_texts(sources)

    table_rows = []
    for model_name in MODEL_NAMES:
        m = metrics.get(model_name, {})
        summary_row = summaries[summaries["model"] == model_name]
        selected_count = int(summary_row.iloc[0]["selected_sentence_count"]) if not summary_row.empty else 0
        table_rows.append({
            "model": model_name,
            "selected_sentences": selected_count,
            "keyword_coverage": float(m.get("keyword_coverage", 0.0)),
            "redundancy": float(m.get("redundancy", 0.0)),
            "source_traceability_rate": float(m.get("source_traceability_rate", 0.0)),
            "processing_seconds": float(m.get("processing_seconds", 0.0)),
        })

    # Sentence selection comparison matrix across the 3 models
    overlap_rows = []
    if not sources.empty:
        by_sid: dict[str, dict] = {}
        for row in sources.itertuples(index=False):
            sid = str(row.sentence_id)
            if sid not in by_sid:
                by_sid[sid] = {
                    "sentence_id": sid,
                    "sentence": str(row.sentence),
                    "source_review_count": int(row.source_review_count),
                    "source_occurrences": int(row.source_occurrences),
                    "TF-IDF Sentence Ranking": "-",
                    "TextRank": "-",
                    "TextRank + MMR": "-",
                    "selected_by_count": 0,
                }
            by_sid[sid][str(row.model)] = f"Rank #{int(row.rank)} ({float(row.method_score):.4f})"
            by_sid[sid]["selected_by_count"] += 1
        overlap_rows = sorted(by_sid.values(), key=lambda r: (-r["selected_by_count"], r["sentence"]))

    return {
        "can_compare": True,
        "reason": "",
        "sources": sources,
        "summaries": summaries,
        "metrics": metrics,
        "comparison_table": pd.DataFrame(table_rows),
        "sentence_overlap": pd.DataFrame(overlap_rows),
        "candidate_sentence_count": candidate_count,
        "unique_document_count": unique_doc_count,
    }


def get_evaluation_status_summary(output_dir: Path) -> dict:
    """Inspect official ROUGE, ablation, human evaluation, and error analysis artifacts
    without ever fabricating placeholder scores.
    """
    refs_path = output_dir / "reference_summaries_template.csv"
    comp_path = output_dir / "evaluation_comparison.csv"
    ablation_path = output_dir / "summary_length_ablation_comparison.csv"
    error_path = output_dir / "error_analysis_cases.csv"
    human_path = output_dir / "human_evaluation_template.csv"

    filled_ref_count = 0
    total_ref_rows = 0
    if refs_path.exists() and refs_path.stat().st_size > 0:
        refs_df = pd.read_csv(refs_path, keep_default_na=False)
        total_ref_rows = int(len(refs_df))
        if "reference_summary" in refs_df:
            filled_ref_count = int(refs_df["reference_summary"].astype(str).str.strip().ne("").sum())

    comparison_df = pd.DataFrame()
    if comp_path.exists() and comp_path.stat().st_size > 0:
        temp = pd.read_csv(comp_path, keep_default_na=False)
        if not temp.empty and "ROUGE_1" in temp.columns:
            comparison_df = temp

    ablation_df = pd.DataFrame()
    if ablation_path.exists() and ablation_path.stat().st_size > 0:
        ablation_df = pd.read_csv(ablation_path, keep_default_na=False)

    errors_df = pd.DataFrame()
    if error_path.exists() and error_path.stat().st_size > 0:
        errors_df = pd.read_csv(error_path, keep_default_na=False)

    human_rated_count = 0
    if human_path.exists() and human_path.stat().st_size > 0:
        hdf = pd.read_csv(human_path, keep_default_na=False)
        score_cols = [c for c in ["groundedness_1_to_5", "relevance_1_to_5", "coherence_1_to_5", "non_redundancy_1_to_5"] if c in hdf]
        if score_cols:
            human_rated_count = int(hdf[score_cols].apply(lambda s: s.astype(str).str.strip().ne("")).any(axis=1).sum())

    rouge_available = (not comparison_df.empty) and (filled_ref_count > 0)
    return {
        "rouge_available": rouge_available,
        "filled_reference_count": filled_ref_count,
        "total_reference_templates": total_ref_rows,
        "human_rated_rows": human_rated_count,
        "comparison_df": comparison_df if rouge_available else pd.DataFrame(),
        "ablation_df": ablation_df if rouge_available else pd.DataFrame(),
        "errors_df": errors_df if rouge_available else pd.DataFrame(),
    }


def get_available_filter_options(
    reviews_df: pd.DataFrame,
    topics_df: pd.DataFrame,
    audit_status: dict,
) -> dict:
    """Discover only filter dimensions supported by existing columns and audit gates."""
    if reviews_df is None or reviews_df.empty:
        return {
            "topics": [],
            "scores": [],
            "sentiments": [],
            "issue_categories": [],
            "severities": [],
            "severity_audit_passed": False,
            "has_score": False,
            "has_sentiment": False,
            "has_issue_category": False,
            "has_severity": False,
            "has_keywords": False,
        }

    topic_label_map: dict[int, str] = {}
    if topics_df is not None and not topics_df.empty and "topic_id" in topics_df:
        for row in topics_df.itertuples(index=False):
            topic_label_map[int(row.topic_id)] = str(getattr(row, "topic_label", ""))

    topic_label_map.setdefault(-1, "Outlier / belum terkelompok")
    topic_label_map.setdefault(-2, "Pujian generik (dikecualikan)")

    topic_ids = sorted(
        {int(x) for x in reviews_df["topic_id"].dropna().tolist()} if "topic_id" in reviews_df else set(),
        key=lambda t: (0 if t >= 0 else 1, t),
    )
    topic_choices = [(tid, f"{tid} — {topic_label_map.get(tid, f'Topik {tid}')}") for tid in topic_ids]

    def _clean_values(col: str) -> list[str]:
        if col not in reviews_df:
            return []
        vals = [str(v).strip() for v in reviews_df[col].dropna().unique().tolist()]
        return sorted(v for v in vals if v and v.lower() != "nan")

    scores = []
    has_score = "score" in reviews_df.columns
    if has_score:
        numeric_scores = pd.to_numeric(reviews_df["score"], errors="coerce").dropna().unique().tolist()
        scores = sorted(int(s) if float(s).is_integer() else float(s) for s in numeric_scores)

    severity_passed = bool((audit_status or {}).get("passed", False))
    has_severity = "severity_level" in reviews_df.columns

    return {
        "topics": topic_choices,
        "scores": scores,
        "sentiments": _clean_values("sentiment"),
        "issue_categories": _clean_values("issue_category"),
        "severities": _clean_values("severity_level") if (has_severity and severity_passed) else [],
        "severity_audit_passed": severity_passed,
        "has_score": has_score and bool(scores),
        "has_sentiment": "sentiment" in reviews_df.columns,
        "has_issue_category": "issue_category" in reviews_df.columns,
        "has_severity": has_severity,
        "has_keywords": "topic_keywords" in reviews_df.columns,
    }


def filter_reviews(
    reviews_df: pd.DataFrame,
    search_query: str = "",
    topic_ids: list[int] | None = None,
    keyword_filter: str = "",
    sentiments: list[str] | None = None,
    issue_categories: list[str] | None = None,
    severities: list[str] | None = None,
    score_values: list | None = None,
    severity_audit_passed: bool = False,
) -> pd.DataFrame:
    """Filter reviews without mutating source data or breaking reviewId traceability."""
    if reviews_df is None or reviews_df.empty:
        return pd.DataFrame()

    mask = pd.Series(True, index=reviews_df.index)

    if topic_ids:
        allowed_topics = {int(t) for t in topic_ids}
        mask &= reviews_df["topic_id"].astype(int).isin(allowed_topics)

    if keyword_filter and keyword_filter.strip():
        kw = keyword_filter.strip().casefold()
        kw_in_topic = (
            reviews_df["topic_keywords"].fillna("").astype(str).str.casefold().str.contains(kw, regex=False)
            if "topic_keywords" in reviews_df.columns
            else pd.Series(False, index=reviews_df.index)
        )
        kw_in_text = reviews_df["content"].fillna("").astype(str).str.casefold().str.contains(kw, regex=False)
        mask &= (kw_in_topic | kw_in_text)

    if sentiments and "sentiment" in reviews_df.columns:
        mask &= reviews_df["sentiment"].astype(str).isin([str(s) for s in sentiments])

    if issue_categories and "issue_category" in reviews_df.columns:
        mask &= reviews_df["issue_category"].astype(str).isin([str(c) for c in issue_categories])

    if severities and severity_audit_passed and "severity_level" in reviews_df.columns:
        mask &= reviews_df["severity_level"].astype(str).isin([str(s) for s in severities])

    if score_values and "score" in reviews_df.columns:
        parsed_scores = pd.to_numeric(reviews_df["score"], errors="coerce")
        allowed_scores = {float(v) for v in score_values}
        mask &= parsed_scores.isin(allowed_scores)

    if search_query and search_query.strip():
        q = search_query.strip().casefold()
        in_content = reviews_df["content"].fillna("").astype(str).str.casefold().str.contains(q, regex=False)
        in_id = reviews_df["reviewId"].fillna("").astype(str).str.casefold().str.contains(q, regex=False)
        mask &= (in_content | in_id)

    return reviews_df.loc[mask].copy()


def parse_keyword_scores(keyword_scores_raw: str) -> pd.DataFrame:
    """Parse stored c-TF-IDF keyword_scores JSON into a DataFrame for visualization."""
    if not keyword_scores_raw:
        return pd.DataFrame(columns=["keyword", "ctfidf_score"])
    try:
        items = json.loads(keyword_scores_raw)
    except Exception:
        return pd.DataFrame(columns=["keyword", "ctfidf_score"])
    rows = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            rows.append({"keyword": str(item[0]), "ctfidf_score": float(item[1])})
    return pd.DataFrame(rows, columns=["keyword", "ctfidf_score"])
