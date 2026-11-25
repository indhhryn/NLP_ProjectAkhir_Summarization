from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
from .preprocessing import normalize_text, is_generic_praise, corpus_profile

REQUIRED_COLUMNS = {"reviewId", "content"}


def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path, low_memory=False)
    raw_row_count = int(len(df))
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ada: {sorted(missing)}. Kolom tersedia: {list(df.columns)}")
    df["reviewId"] = df["reviewId"].fillna("").astype(str)
    df["content"] = df["content"].fillna("").astype(str)
    valid_mask = df["reviewId"].str.len().gt(0) & df["content"].str.strip().ne("")
    df = df[valid_mask].copy()
    df["normalized_text"] = df["content"].map(normalize_text)
    df = df[df["normalized_text"].str.len().gt(0)].copy()
    df.attrs["raw_csv_rows"] = raw_row_count
    df.attrs["dropped_invalid_rows"] = int(raw_row_count - len(df))
    return df


def make_unique_documents(reviews: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for norm, group in reviews.groupby("normalized_text", sort=False, dropna=False):
        ids = group["reviewId"].astype(str).drop_duplicates().tolist()
        digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]
        # Keep raw-text variants separate so a selected sentence is linked only
        # to review IDs that contain that exact source wording.
        variants = []
        for raw_content, variant_group in group.groupby("content", sort=False, dropna=False):
            variant_ids = variant_group["reviewId"].astype(str).drop_duplicates().tolist()
            variants.append({
                "content": str(raw_content),
                "source_review_ids": variant_ids,
                "occurrence_count": int(len(variant_group)),
            })
        rows.append({
            "doc_id": digest,
            "normalized_text": norm,
            "content": group["content"].iloc[0],
            "source_review_ids": json.dumps(ids, ensure_ascii=False),
            "source_variants": json.dumps(variants, ensure_ascii=False),
            "duplicate_count": int(len(group)),
            "source_review_count": int(len(ids)),
            "is_generic_praise": bool(is_generic_praise(norm)),
            "issue_category_mode": group["issue_category"].mode().iloc[0] if "issue_category" in group and not group["issue_category"].mode().empty else "unknown",
            "severity_mode": group["severity_level"].mode().iloc[0] if "severity_level" in group and not group["severity_level"].mode().empty else "unknown",
            "sentiment_mode": group["sentiment"].mode().iloc[0] if "sentiment" in group and not group["sentiment"].mode().empty else "unknown",
        })
    return pd.DataFrame(rows)


def build_quality_report(reviews: pd.DataFrame, docs: pd.DataFrame) -> dict:
    report = {
        "raw_csv_rows": int(reviews.attrs.get("raw_csv_rows", len(reviews))),
        "dropped_invalid_rows": int(reviews.attrs.get("dropped_invalid_rows", 0)),
        "raw_review_count": int(len(reviews)),
        "unique_normalized_reviews": int(len(docs)),
        "duplicate_extra_rows": int(len(reviews) - len(docs)),
        "duplicate_extra_share": float((len(reviews) - len(docs)) / max(1, len(reviews))),
        "generic_praise_unique_docs": int(docs["is_generic_praise"].sum()),
        "generic_praise_review_count": int(docs.loc[docs["is_generic_praise"], "duplicate_count"].sum()),
        "informative_unique_docs": int((~docs["is_generic_praise"]).sum()),
        "informative_review_count": int(docs.loc[~docs["is_generic_praise"], "duplicate_count"].sum()),
        "review_columns": list(reviews.columns),
        "text_profile": corpus_profile(reviews["content"].tolist()),
    }
    for col in ["sentiment", "issue_category", "severity_level"]:
        if col in reviews:
            report[f"{col}_distribution"] = {str(k): int(v) for k,v in reviews[col].fillna("missing").value_counts().items()}
            if "normalized_text" in reviews:
                label_counts = reviews.groupby("normalized_text")[col].nunique(dropna=False)
                report[f"{col}_conflicting_duplicate_groups"] = int((label_counts > 1).sum())
    if "evidence_span" in reviews:
        spans = reviews["evidence_span"].fillna("").astype(str)
        contents = reviews["content"].fillna("").astype(str)
        nonempty = spans.str.strip().ne("")
        matches = [str(s).casefold() in str(c).casefold() for s,c in zip(spans[nonempty], contents[nonempty])]
        report["evidence_span_nonempty_count"] = int(nonempty.sum())
        report["evidence_span_literal_match_count"] = int(sum(matches))
        report["evidence_span_literal_match_rate"] = float(sum(matches) / max(1,len(matches)))
        report["evidence_span_note"] = "Pencocokan substring literal hanya pemeriksaan awal, bukan validasi semantik." 
    return report
