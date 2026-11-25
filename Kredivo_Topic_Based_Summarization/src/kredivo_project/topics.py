from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def _build_model(cfg: dict):
    try:
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from .preprocessing import STOPWORDS_ID, NEGATION_AND_ISSUE_WORDS
        from umap import UMAP
        from hdbscan import HDBSCAN
    except ImportError as exc:
        raise RuntimeError(
            "Komponen BERTopic belum terpasang. Aktifkan virtual environment lalu jalankan "
            "pip install -r requirements.txt. Pastikan Python 3.10–3.12."
        ) from exc

    # Bahasa Indonesia ditangani dengan multilingual sentence embeddings.
    embedding_model = SentenceTransformer(cfg["embedding_model"], device=cfg.get("device", "cpu"))
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        stop_words=sorted(STOPWORDS_ID - NEGATION_AND_ISSUE_WORDS),
        token_pattern=r"(?u)\b[\w]+\b",
        lowercase=True,
    )
    umap_model = UMAP(
        n_neighbors=int(cfg["umap_n_neighbors"]),
        n_components=int(cfg["umap_n_components"]),
        min_dist=0.0,
        metric="cosine",
        random_state=int(cfg["random_state"]),
        low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=int(cfg["min_topic_size"]),
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    return BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        nr_topics="auto",
        verbose=True,
    )


def _transform_in_batches(model, docs: list[str], batch_size: int) -> list[int]:
    predictions: list[int] = []
    for start in range(0, len(docs), batch_size):
        batch_topics, _ = model.transform(docs[start:start + batch_size])
        predictions.extend(int(x) for x in batch_topics)
    return predictions


def fit_and_assign_topics(docs: pd.DataFrame, cfg: dict, output_dir: Path) -> tuple[pd.DataFrame, object, pd.DataFrame, pd.DataFrame]:
    """Fit BERTopic hanya pada train split; validation/test ditetapkan via transform().

    Dokumen pujian generik diberi topic_id=-2 dan tidak memengaruhi representasi topik.
    Dokumen outlier BERTopic tetap menggunakan topic_id=-1.
    """
    informative = docs.loc[~docs["is_generic_praise"].astype(bool)].copy().reset_index(drop=True)
    generic = docs.loc[docs["is_generic_praise"].astype(bool)].copy()
    if len(informative) < max(100, int(cfg["min_topic_size"]) * 3):
        raise ValueError(
            f"Dokumen informatif hanya {len(informative)}. Jumlah ini terlalu kecil untuk konfigurasi "
            f"min_topic_size={cfg['min_topic_size']}. Periksa filter pujian atau turunkan min_topic_size."
        )

    rng = np.random.default_rng(int(cfg["random_state"]))
    order = rng.permutation(len(informative))
    test_fraction = float(cfg.get("test_size", 0.10))
    validation_fraction = float(cfg.get("validation_size", 0.10))
    if test_fraction <= 0 or validation_fraction < 0 or test_fraction + validation_fraction >= 1:
        raise ValueError("Konfigurasi split harus memenuhi test_size > 0, validation_size >= 0, dan total keduanya < 1.")
    test_n = max(1, int(round(len(informative) * test_fraction)))
    validation_n = int(round(len(informative) * validation_fraction)) if validation_fraction > 0 else 0
    if test_n + validation_n >= len(informative):
        raise ValueError("Ukuran validation/test menyisakan terlalu sedikit data train.")
    test_idx = set(order[:test_n].tolist())
    validation_idx = set(order[test_n:test_n + validation_n].tolist())
    informative["split"] = [
        "test" if i in test_idx else ("validation" if i in validation_idx else "train")
        for i in range(len(informative))
    ]
    train = informative[informative["split"] == "train"].copy()
    validation = informative[informative["split"] == "validation"].copy()
    test = informative[informative["split"] == "test"].copy()

    max_fit = int(cfg.get("max_fit_docs", 0))
    if max_fit > 0 and len(train) > max_fit:
        chosen = rng.choice(train.index.to_numpy(), size=max_fit, replace=False)
        fit = train.loc[chosen].copy()
    else:
        fit = train.copy()
    fit = fit.reset_index(drop=True)

    if len(fit) < int(cfg["min_topic_size"]) * 3:
        raise ValueError("Dokumen untuk fitting BERTopic terlalu sedikit dibanding min_topic_size.")

    model = _build_model(cfg)
    fit_topics, _ = model.fit_transform(fit["normalized_text"].tolist())
    assigned = pd.Series(index=informative.index, dtype="int64")
    fit_topic_map = dict(zip(fit["doc_id"].astype(str), [int(x) for x in fit_topics]))
    informative["topic_id"] = informative["doc_id"].astype(str).map(fit_topic_map)
    assigned.loc[informative["topic_id"].notna()] = informative.loc[informative["topic_id"].notna(), "topic_id"].astype(int)

    # Dokumen train yang tidak masuk sampel fit dan seluruh test diproyeksikan ke topik hasil fit.
    fit_doc_ids = set(fit["doc_id"].astype(str))
    remaining = informative.loc[~informative["doc_id"].astype(str).isin(fit_doc_ids)].copy()
    if len(remaining):
        predicted = _transform_in_batches(model, remaining["normalized_text"].tolist(), int(cfg["transform_batch_size"]))
        assigned.loc[remaining.index] = predicted

    informative["topic_id"] = assigned.astype(int).to_numpy()
    informative["topic_split"] = informative["split"]
    generic["topic_id"] = -2
    generic["topic_split"] = "excluded_generic"
    result = pd.concat([informative, generic], ignore_index=True)
    result["topic_id"] = result["topic_id"].astype(int)

    # Topik dari BERTopic dan kata kunci c-TF-IDF.
    info = model.get_topic_info().copy()
    keyword_rows = []
    for tid in sorted(set(informative["topic_id"].tolist())):
        if tid == -1:
            continue
        terms = model.get_topic(int(tid)) or []
        keyword_rows.append({
            "topic_id": int(tid),
            "keywords": " | ".join(str(term) for term, _ in terms[:int(cfg["n_keywords"])]),
            "keyword_scores": json.dumps([[str(t), float(s)] for t,s in terms[:int(cfg["n_keywords"])]], ensure_ascii=False),
        })
    keywords = pd.DataFrame(keyword_rows, columns=["topic_id", "keywords", "keyword_scores"])
    result = result.merge(keywords[["topic_id", "keywords"]], on="topic_id", how="left")

    info = info.rename(columns={"Name": "bertopic_name", "Count": "fit_count"})
    info = info[[c for c in ["Topic", "bertopic_name", "fit_count"] if c in info]].rename(columns={"Topic":"topic_id"})
    counts = result[result["topic_id"] >= -1].groupby("topic_id", as_index=False).agg(
        unique_documents=("doc_id", "nunique"),
        total_source_reviews=("duplicate_count", "sum"),
        test_documents=("topic_split", lambda s: int((s == "test").sum())),
        validation_documents=("topic_split", lambda s: int((s == "validation").sum())),
        train_documents=("topic_split", lambda s: int((s == "train").sum())),
    )
    topics = counts.merge(info, on="topic_id", how="left").merge(keywords[["topic_id", "keywords"]], on="topic_id", how="left")
    topics["topic_label"] = topics["keywords"].fillna("").replace("", "Topik tanpa kata kunci")
    topics.loc[topics["topic_id"] == -1, "topic_label"] = "Outlier / belum terkelompok"
    topics = topics.sort_values(["total_source_reviews", "unique_documents"], ascending=False)

    if bool(cfg.get("save_topic_model", True)):
        model_path = output_dir / "bertopic_model"
        try:
            model.save(str(model_path), serialization="safetensors", save_ctfidf=True, save_embedding_model=False)
        except Exception as exc:
            (output_dir / "bertopic_model_save_warning.txt").write_text(str(exc), encoding="utf-8")
    return result, model, topics, keywords
