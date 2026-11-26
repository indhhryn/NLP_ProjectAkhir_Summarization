from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, load_config
from .data import make_unique_documents
from .preprocessing import normalize_text, tokenize_id
from .summarization import MODEL_NAMES, build_sentence_pool, build_summary_texts, summarize_topic

OPTIONAL_REVIEW_COLUMNS = ["score", "at", "sentiment", "issue_category", "severity_level"]


def _clean_raw_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).replace("\ufffd", " ").replace("\x00", " ")
    return text.strip()


def parse_manual_reviews(
    bulk_text: str = "",
    individual_items: list[str] | None = None,
) -> pd.DataFrame:
    """Parse manual user inputs into a normalized session DataFrame."""
    entries: list[str] = []
    if individual_items:
        entries.extend("" if item is None else str(item) for item in individual_items)

    if bulk_text is not None and str(bulk_text) != "":
        lines = str(bulk_text).splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            entries.extend(non_empty_lines)
        elif not individual_items:
            # Preserve single blank entry so validation reports 1 invalid/empty input.
            entries.append(str(bulk_text))

    rows = [
        {
            "reviewId": f"manual-{idx:04d}",
            "content": text,
        }
        for idx, text in enumerate(entries, start=1)
    ]
    return pd.DataFrame(rows, columns=["reviewId", "content"])


def _read_csv_flexible(uploaded_file: Any) -> pd.DataFrame:
    if isinstance(uploaded_file, pd.DataFrame):
        return uploaded_file.copy()
    if isinstance(uploaded_file, (str, Path)):
        raw_bytes = Path(uploaded_file).read_bytes()
    elif hasattr(uploaded_file, "getvalue"):
        raw_bytes = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        raw_bytes = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
    else:
        raise ValueError("Format sumber CSV tidak dikenali.")

    if not raw_bytes or not str(raw_bytes).strip():
        raise ValueError("File CSV kosong.")

    last_exc: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw_bytes.decode(encoding) if encoding != "latin-1" else raw_bytes.decode(encoding, errors="replace")
            return pd.read_csv(io.StringIO(text), low_memory=False)
        except Exception as exc:
            last_exc = exc
    raise ValueError(f"Gagal membaca CSV: {last_exc}")


def validate_and_prepare_uploaded_csv(
    uploaded_file: Any,
    text_col: str = "content",
    id_col: str = "reviewId",
) -> tuple[pd.DataFrame, dict]:
    """Validate uploaded CSV structure and build consistent session IDs if missing."""
    df = _read_csv_flexible(uploaded_file)
    if df.empty and text_col not in df.columns:
        raise ValueError(f"CSV kosong atau tidak memiliki kolom '{text_col}'.")
    if text_col not in df.columns:
        raise ValueError(
            f"Kolom teks wajib '{text_col}' tidak ditemukan pada CSV. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    prepared = pd.DataFrame(index=df.index)
    generated_temp_ids = False
    if id_col in df.columns:
        raw_ids = df[id_col].fillna("").astype(str).str.strip()
        missing_mask = raw_ids.eq("") | raw_ids.str.casefold().isin({"nan", "none", "null"})
        if missing_mask.any():
            generated_temp_ids = True
            fallback_ids = [f"upload-{i + 1:04d}" for i in range(len(df))]
            raw_ids = raw_ids.where(~missing_mask, pd.Series(fallback_ids, index=df.index))
        prepared["reviewId"] = raw_ids
    else:
        generated_temp_ids = True
        prepared["reviewId"] = [f"upload-{i + 1:04d}" for i in range(len(df))]

    prepared["content"] = df[text_col].map(lambda x: "" if pd.isna(x) else str(x))

    preserved_optional = []
    for col in OPTIONAL_REVIEW_COLUMNS:
        if col in df.columns:
            prepared[col] = df[col]
            preserved_optional.append(col)

    meta = {
        "total_csv_rows": int(len(df)),
        "generated_temp_ids": generated_temp_ids,
        "optional_columns_found": preserved_optional,
        "source_columns": list(df.columns),
    }
    return prepared.reset_index(drop=True), meta


def load_inference_bundle(
    model_dir: Path | None = None,
    cfg: dict | None = None,
) -> dict:
    """Load saved BERTopic model and SentenceTransformer without retraining."""
    cfg = cfg or load_config()
    model_path = Path(model_dir) if model_dir is not None else (OUTPUT_DIR / "bertopic_model")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Artifact model BERTopic belum ditemukan di {model_path}. "
            "Jalankan pipeline utama terlebih dahulu (`python scripts/run_pipeline.py`) untuk melatih dan menyimpan model dari train split."
        )

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Dependensi BERTopic/SentenceTransformer belum terpasang di virtual environment."
        ) from exc

    model_name = cfg["embedding_model"]
    device = cfg.get("device", "cpu")
    try:
        embedding_model = SentenceTransformer(model_name, device=device, local_files_only=True)
    except Exception:
        embedding_model = SentenceTransformer(model_name, device=device)

    topic_model = BERTopic.load(str(model_path), embedding_model=embedding_model)
    uses_cosine_fallback = type(getattr(topic_model, "hdbscan_model", None)).__name__ == "BaseCluster"
    return {
        "topic_model": topic_model,
        "embedding_model": embedding_model,
        "model_dir": str(model_path),
        "embedding_model_name": model_name,
        "uses_cosine_fallback": uses_cosine_fallback,
    }


def _extract_topic_metadata(
    topic_model: Any,
    topics_df: pd.DataFrame | None,
    keywords_df: pd.DataFrame | None,
    n_keywords: int,
) -> tuple[dict[int, str], dict[int, str]]:
    label_map: dict[int, str] = {}
    kw_map: dict[int, str] = {}

    if topics_df is not None and not topics_df.empty and "topic_id" in topics_df:
        for row in topics_df.itertuples(index=False):
            tid = int(row.topic_id)
            label_map[tid] = str(getattr(row, "topic_label", ""))
            if hasattr(row, "keywords") and str(getattr(row, "keywords", "")):
                kw_map[tid] = str(getattr(row, "keywords", ""))

    if keywords_df is not None and not keywords_df.empty and "topic_id" in keywords_df:
        for row in keywords_df.itertuples(index=False):
            tid = int(row.topic_id)
            kw_text = str(getattr(row, "keywords", ""))
            if kw_text:
                kw_map[tid] = kw_text
                if tid not in label_map or not label_map[tid]:
                    label_map[tid] = kw_text

    # Fallback directly to c-TF-IDF terms inside the loaded BERTopic model
    if hasattr(topic_model, "get_topics"):
        for tid in topic_model.get_topics().keys():
            tid = int(tid)
            if tid < 0 or tid in kw_map:
                continue
            terms = topic_model.get_topic(tid) or []
            kw_text = " | ".join(str(term) for term, _ in terms[:n_keywords])
            if kw_text:
                kw_map[tid] = kw_text
                label_map.setdefault(tid, kw_text)

    label_map[-1] = "Outlier / topik tidak teridentifikasi"
    label_map[-2] = "Pujian generik (dikecualikan dari clustering)"
    return label_map, kw_map


def analyze_new_reviews(
    raw_df: pd.DataFrame,
    model_bundle: dict,
    topics_df: pd.DataFrame | None = None,
    keywords_df: pd.DataFrame | None = None,
    cfg: dict | None = None,
    min_similarity: float | None = None,
    min_docs_for_summary: int = 2,
    summary_sentences: int | None = None,
    mmr_lambda: float | None = None,
) -> dict:
    """Run validation, preprocessing, duplicate-aware grouping, BERTopic inference,
    multi-document extractive summarization, and source traceability on new reviews.
    """
    cfg = cfg or load_config()
    if raw_df is None or raw_df.empty:
        raise ValueError("Tidak ada ulasan yang dimasukkan untuk dianalisis.")
    if "content" not in raw_df.columns:
        raise ValueError("Data input wajib memiliki kolom 'content'.")

    df = raw_df.copy().reset_index(drop=True)
    if "reviewId" not in df.columns:
        df["reviewId"] = [f"session-{i + 1:04d}" for i in range(len(df))]
    else:
        raw_ids = df["reviewId"].fillna("").astype(str).str.strip()
        missing_id_mask = raw_ids.eq("")
        if missing_id_mask.any():
            fallback = pd.Series([f"session-{i + 1:04d}" for i in range(len(df))], index=df.index)
            raw_ids = raw_ids.where(~missing_id_mask, fallback)
        df["reviewId"] = raw_ids

    df["content"] = df["content"].map(_clean_raw_text)
    df["normalized_text"] = df["content"].map(normalize_text)
    df["token_count"] = df["normalized_text"].map(lambda t: len(tokenize_id(t, remove_stopwords=False)))

    valid_mask = df["content"].ne("") & df["normalized_text"].ne("") & df["token_count"].gt(0)
    valid_df = df[valid_mask].copy().reset_index(drop=True)
    invalid_df = df[~valid_mask].copy().reset_index(drop=True)

    if not invalid_df.empty:
        invalid_df["invalid_reason"] = np.where(
            invalid_df["content"].eq(""),
            "Teks ulasan kosong",
            "Teks tidak mengandung kata/token yang dapat diproses setelah normalisasi",
        )

    total_input = int(len(df))
    valid_count = int(len(valid_df))
    invalid_count = int(len(invalid_df))

    if valid_df.empty:
        return {
            "status": "no_valid_reviews",
            "counts": {
                "total_input": total_input,
                "valid_reviews": 0,
                "invalid_reviews": invalid_count,
                "unique_documents": 0,
                "duplicate_extra_rows": 0,
                "generic_praise_reviews": 0,
                "informative_reviews": 0,
                "assigned_reviews": 0,
                "outlier_reviews": 0,
                "detected_topics": 0,
            },
            "invalid_reviews": invalid_df[[c for c in ["reviewId", "content", "invalid_reason"] if c in invalid_df]],
            "reviews_assigned": pd.DataFrame(),
            "unique_documents": pd.DataFrame(),
            "topic_overview": pd.DataFrame(),
            "summaries": pd.DataFrame(),
            "summary_sources": pd.DataFrame(),
            "metrics_by_topic": {},
            "topic_summary_status": {},
            "similarity_threshold": float(cfg.get("inference_min_similarity", 0.42) if min_similarity is None else min_similarity),
            "uses_cosine_fallback": bool(model_bundle.get("uses_cosine_fallback", False)),
        }

    docs = make_unique_documents(valid_df)
    duplicate_extra_rows = int(valid_count - len(docs))

    informative_mask = ~docs["is_generic_praise"].astype(bool)
    informative = docs.loc[informative_mask].copy().reset_index(drop=True)
    generic = docs.loc[~informative_mask].copy().reset_index(drop=True)

    topic_model = model_bundle["topic_model"]
    uses_cosine_fallback = bool(model_bundle.get("uses_cosine_fallback", False))
    threshold = float(cfg.get("inference_min_similarity", 0.42) if min_similarity is None else min_similarity)

    if not informative.empty:
        pred_topics, pred_probs = topic_model.transform(informative["normalized_text"].tolist())
        pred_topics_list = [int(t) for t in pred_topics]
        if pred_probs is None:
            sim_scores = [None] * len(pred_topics_list)
        else:
            probs_arr = np.asarray(pred_probs)
            if probs_arr.ndim == 2:
                sim_scores = [float(np.max(row)) for row in probs_arr]
            else:
                sim_scores = [float(x) for x in probs_arr.ravel()]

        final_topics: list[int] = []
        final_statuses: list[str] = []
        for tid, sim in zip(pred_topics_list, sim_scores):
            if tid < 0:
                final_topics.append(-1)
                final_statuses.append("outlier")
            elif uses_cosine_fallback and sim is not None and sim < threshold:
                final_topics.append(-1)
                final_statuses.append("outlier_low_similarity")
            else:
                final_topics.append(int(tid))
                final_statuses.append("assigned")

        informative["topic_id"] = final_topics
        informative["topic_similarity"] = sim_scores
        informative["topic_status"] = final_statuses
    else:
        informative["topic_id"] = pd.Series(dtype="int64")
        informative["topic_similarity"] = pd.Series(dtype="float64")
        informative["topic_status"] = pd.Series(dtype="object")

    if not generic.empty:
        generic["topic_id"] = -2
        generic["topic_similarity"] = np.nan
        generic["topic_status"] = "generic_praise"

    assigned_docs = pd.concat([informative, generic], ignore_index=True)
    assigned_docs["topic_id"] = assigned_docs["topic_id"].astype(int)

    label_map, kw_map = _extract_topic_metadata(
        topic_model,
        topics_df,
        keywords_df,
        int(cfg.get("n_keywords", 12)),
    )
    assigned_docs["topic_keywords"] = assigned_docs["topic_id"].map(lambda tid: kw_map.get(int(tid), ""))
    assigned_docs["topic_label"] = assigned_docs["topic_id"].map(
        lambda tid: label_map.get(int(tid), f"Topik {tid}")
    )

    doc_lookup = assigned_docs.set_index("doc_id")[
        ["topic_id", "topic_label", "topic_keywords", "topic_similarity", "topic_status", "is_generic_praise"]
    ].to_dict("index")

    reviews_assigned = valid_df.copy()
    reviews_assigned["doc_id"] = reviews_assigned["normalized_text"].map(
        lambda norm: hashlib.sha1(str(norm).encode("utf-8")).hexdigest()[:16]
    )
    reviews_assigned["topic_id"] = reviews_assigned["doc_id"].map(lambda d: int(doc_lookup[d]["topic_id"]))
    reviews_assigned["topic_label"] = reviews_assigned["doc_id"].map(lambda d: doc_lookup[d]["topic_label"])
    reviews_assigned["topic_keywords"] = reviews_assigned["doc_id"].map(lambda d: doc_lookup[d]["topic_keywords"])
    reviews_assigned["topic_similarity"] = reviews_assigned["doc_id"].map(lambda d: doc_lookup[d]["topic_similarity"])
    reviews_assigned["topic_status"] = reviews_assigned["doc_id"].map(lambda d: doc_lookup[d]["topic_status"])
    reviews_assigned["is_generic_praise"] = reviews_assigned["doc_id"].map(
        lambda d: bool(doc_lookup[d]["is_generic_praise"])
    )

    # Summarization per detected topic (>= 0) with sufficiency guard
    n_sent_cfg = int(summary_sentences if summary_sentences is not None else cfg.get("summary_sentences", 5))
    lambda_cfg = float(mmr_lambda if mmr_lambda is not None else cfg.get("mmr_lambda", 0.72))
    neighbors_cfg = int(cfg.get("textrank_neighbors", 15))

    source_frames: list[pd.DataFrame] = []
    metrics_by_topic: dict[int, dict] = {}
    topic_summary_status: dict[int, dict] = {}
    overview_rows: list[dict] = []

    detected_topic_ids = sorted(tid for tid in assigned_docs["topic_id"].unique().tolist() if int(tid) >= 0)
    for tid in detected_topic_ids:
        topic_docs = assigned_docs[assigned_docs["topic_id"] == tid].copy()
        kw_list = [k.strip() for k in str(kw_map.get(tid, "")).split("|") if k.strip()]
        pool = build_sentence_pool(topic_docs)
        unique_doc_count = int(len(topic_docs))
        review_count = int(topic_docs["duplicate_count"].sum())
        sentence_count = int(len(pool))

        if unique_doc_count < int(min_docs_for_summary) or sentence_count < 2:
            reason = (
                f"Jumlah dokumen unik ({unique_doc_count}) atau kalimat kandidat ({sentence_count}) "
                f"pada topik ini belum mencukupi untuk multi-document extractive summarization "
                f"(minimal {min_docs_for_summary} dokumen unik dan 2 kalimat kandidat)."
            )
            topic_summary_status[tid] = {
                "can_summarize": False,
                "status": "insufficient_documents",
                "reason": reason,
                "unique_documents": unique_doc_count,
                "review_count": review_count,
                "candidate_sentences": sentence_count,
            }
        else:
            n_target = min(n_sent_cfg, sentence_count)
            selected, metrics = summarize_topic(
                topic_docs,
                n_sentences=n_target,
                mmr_lambda=lambda_cfg,
                keywords=kw_list,
                topic_id=tid,
                textrank_neighbors=neighbors_cfg,
            )
            if not selected.empty:
                source_frames.append(selected)
            metrics_by_topic[tid] = metrics
            topic_summary_status[tid] = {
                "can_summarize": True,
                "status": "summarized",
                "reason": "",
                "unique_documents": unique_doc_count,
                "review_count": review_count,
                "candidate_sentences": sentence_count,
            }

        overview_rows.append({
            "topic_id": tid,
            "topic_label": label_map.get(tid, f"Topik {tid}"),
            "keywords": kw_map.get(tid, ""),
            "review_count": review_count,
            "unique_documents": unique_doc_count,
            "candidate_sentences": sentence_count,
            "summarization_status": topic_summary_status[tid]["status"],
        })

    # Also summarize special groups (-1 outlier, -2 generic praise) in overview
    for special_tid, special_label in [
        (-1, "Outlier / topik tidak teridentifikasi"),
        (-2, "Pujian generik (dikecualikan dari clustering)"),
    ]:
        grp = assigned_docs[assigned_docs["topic_id"] == special_tid]
        if not grp.empty:
            overview_rows.append({
                "topic_id": special_tid,
                "topic_label": special_label,
                "keywords": "",
                "review_count": int(grp["duplicate_count"].sum()),
                "unique_documents": int(len(grp)),
                "candidate_sentences": 0,
                "summarization_status": "excluded_from_topic_summary",
            })

    summary_sources_df = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame(
        columns=[
            "topic_id", "model", "rank", "sentence_id", "sentence",
            "source_review_ids", "source_review_count", "source_occurrences", "method_score",
        ]
    )
    summaries_df = build_summary_texts(summary_sources_df)
    topic_overview_df = pd.DataFrame(overview_rows)
    if not topic_overview_df.empty:
        pos = topic_overview_df[topic_overview_df["topic_id"] >= 0].sort_values("review_count", ascending=False)
        neg = topic_overview_df[topic_overview_df["topic_id"] < 0].sort_values("topic_id", ascending=False)
        topic_overview_df = pd.concat([pos, neg], ignore_index=True)

    counts = {
        "total_input": total_input,
        "valid_reviews": valid_count,
        "invalid_reviews": invalid_count,
        "unique_documents": int(len(docs)),
        "duplicate_extra_rows": duplicate_extra_rows,
        "generic_praise_reviews": int(reviews_assigned["is_generic_praise"].sum()),
        "informative_reviews": int((~reviews_assigned["is_generic_praise"]).sum()),
        "assigned_reviews": int((reviews_assigned["topic_id"] >= 0).sum()),
        "outlier_reviews": int((reviews_assigned["topic_id"] == -1).sum()),
        "detected_topics": len(detected_topic_ids),
    }

    return {
        "status": "completed",
        "counts": counts,
        "invalid_reviews": invalid_df[[c for c in ["reviewId", "content", "invalid_reason"] if c in invalid_df]],
        "reviews_assigned": reviews_assigned,
        "unique_documents": assigned_docs,
        "topic_overview": topic_overview_df,
        "summaries": summaries_df,
        "summary_sources": summary_sources_df,
        "metrics_by_topic": metrics_by_topic,
        "topic_summary_status": topic_summary_status,
        "similarity_threshold": threshold,
        "uses_cosine_fallback": uses_cosine_fallback,
    }
