from __future__ import annotations
import hashlib
import json
import time
from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from .preprocessing import normalize_text, tokenize_id, split_sentences

MODEL_NAMES = ["TF-IDF Sentence Ranking", "TextRank", "TextRank + MMR"]


def build_sentence_pool(topic_docs: pd.DataFrame) -> pd.DataFrame:
    """Split source variants into exact sentence candidates with precise evidence IDs.

    Normalized-equivalent review texts can have different original wording (e.g.
    slang vs formal spelling). We deduplicate documents for topic modeling but do
    not merge their raw sentence evidence, preventing a paraphrased variant's
    reviewId from being attached to a sentence it does not literally contain.
    """
    grouped: dict[str, dict] = {}
    for row in topic_docs.itertuples(index=False):
        raw_variants = getattr(row, "source_variants", "")
        variants = []
        if raw_variants:
            try:
                parsed = json.loads(raw_variants)
                if isinstance(parsed, list):
                    variants = parsed
            except Exception:
                variants = []
        if not variants:
            # Backward-compatible fallback for test/legacy rows without variants.
            ids = []
            try:
                ids = json.loads(getattr(row, "source_review_ids", "[]"))
            except Exception:
                pass
            variants = [{
                "content": str(getattr(row, "content", "")),
                "source_review_ids": ids or [str(getattr(row, "doc_id", "unknown"))],
                "occurrence_count": int(getattr(row, "duplicate_count", 1)),
            }]

        for variant in variants:
            raw = str(variant.get("content", ""))
            ids = [str(x) for x in variant.get("source_review_ids", [])]
            occurrence_count = int(variant.get("occurrence_count", 1))
            for sentence in split_sentences(raw):
                sentence = sentence.strip()
                norm = normalize_text(sentence)
                if len(tokenize_id(norm)) == 0:
                    continue
                # Exact source sentence is the key; normalized text is only a feature.
                key = sentence
                if key not in grouped:
                    grouped[key] = {
                        "sentence": sentence,
                        "normalized_sentence": norm,
                        "source_ids": set(),
                        "source_occurrences": 0,
                    }
                grouped[key]["source_ids"].update(ids)
                grouped[key]["source_occurrences"] += occurrence_count

    rows = []
    for exact_sentence, item in grouped.items():
        sid = hashlib.sha1(exact_sentence.encode("utf-8")).hexdigest()[:16]
        ids = sorted(item["source_ids"])
        rows.append({
            "sentence_id": sid,
            "sentence": item["sentence"],
            "normalized_sentence": item["normalized_sentence"],
            "source_review_ids": json.dumps(ids, ensure_ascii=False),
            "source_review_count": len(ids),
            "source_occurrences": int(item["source_occurrences"]),
        })
    return pd.DataFrame(rows)


def _vectorize(sentences: list[str]):
    """Vectorize sentence candidates, relaxing max_df for small homogeneous topics.

    With a small candidate pool, max_df=0.98 can prune every token (for example,
    when a term appears in all selected sentences). In that case retain common
    terms rather than stopping the complete pipeline.
    """
    common = {
        "tokenizer": tokenize_id,
        "token_pattern": None,
        "lowercase": False,
        "ngram_range": (1, 2),
        "min_df": 1,
        "sublinear_tf": True,
    }
    try:
        return TfidfVectorizer(max_df=0.98, **common).fit_transform(sentences)
    except ValueError as exc:
        if "no terms remain" not in str(exc).casefold() and "empty vocabulary" not in str(exc).casefold():
            raise
        return TfidfVectorizer(max_df=1.0, **common).fit_transform(sentences)


def _rank_tfidf(matrix) -> np.ndarray:
    # Sesuai modul praktikum: jumlah bobot TF-IDF setiap kalimat.
    return np.asarray(matrix.sum(axis=1)).ravel().astype(float)


def _rank_textrank(matrix, n_neighbors: int = 15) -> np.ndarray:
    """PageRank pada graph k-nearest-neighbor berbobot cosine similarity.

    Graph sparse menjaga biaya memori tetap terukur untuk topik dengan banyak review.
    Setiap node terhubung ke kalimat paling mirip, bukan membentuk matriks NxN penuh.
    """
    n = matrix.shape[0]
    if n <= 1:
        return np.ones(n, dtype=float)
    k = min(max(1, int(n_neighbors)), n - 1)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(matrix)
    distances = nn.kneighbors_graph(X=None, n_neighbors=k, mode="distance")
    similarities = distances.copy()
    similarities.data = np.clip(1.0 - similarities.data, 0.0, 1.0)
    similarities.eliminate_zeros()
    # Simetriskan graph: hubungan A→B atau B→A tetap menjadi edge untuk PageRank.
    similarities = similarities.maximum(similarities.T).tocsr()
    graph = nx.from_scipy_sparse_array(similarities, edge_attribute="weight")
    if graph.number_of_edges() == 0:
        return np.full(n, 1.0 / n)
    scores_dict = nx.pagerank(graph, weight="weight", max_iter=300, tol=1e-8)
    return np.array([scores_dict[i] for i in range(n)], dtype=float)


def _select_mmr(relevance: np.ndarray, matrix, n: int, lambda_mmr: float) -> list[int]:
    n = min(max(1, n), len(relevance))
    rel = relevance.astype(float)
    if rel.max() > rel.min():
        rel = (rel - rel.min()) / (rel.max() - rel.min())
    selected: list[int] = []
    candidates = set(range(len(rel)))
    while candidates and len(selected) < n:
        if selected:
            # Hitung similarity hanya terhadap kalimat yang sudah dipilih, bukan NxN penuh.
            similarities = cosine_similarity(matrix, matrix[selected])
        else:
            similarities = None
        best_i, best_score = None, -float("inf")
        for i in sorted(candidates):
            redundancy = float(similarities[i].max()) if similarities is not None else 0.0
            score = lambda_mmr * float(rel[i]) - (1.0 - lambda_mmr) * redundancy
            if score > best_score:
                best_i, best_score = i, score
        selected.append(int(best_i))
        candidates.remove(best_i)
    return selected


def summarize_topic(topic_docs: pd.DataFrame, n_sentences: int = 5, mmr_lambda: float = 0.72,
                    keywords: list[str] | None = None, topic_id: int | None = None,
                    textrank_neighbors: int = 15) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    pool = build_sentence_pool(topic_docs)
    empty_metrics = {"keyword_coverage": 0.0, "redundancy": 0.0, "source_traceability_rate": 0.0, "processing_seconds": 0.0}
    if pool.empty:
        return pool, empty_metrics
    sentences = pool["sentence"].astype(str).tolist()
    vector_started = time.perf_counter()
    try:
        matrix = _vectorize(sentences)
    except ValueError:
        pool["tfidf_score"] = 0.0
        pool["textrank_score"] = 1.0 / len(pool)
        matrix = None
    vectorization_seconds = time.perf_counter() - vector_started
    if matrix is not None:
        tfidf_started = time.perf_counter()
        tfidf_scores = _rank_tfidf(matrix)
        tfidf_rank_seconds = time.perf_counter() - tfidf_started
        textrank_started = time.perf_counter()
        textrank_scores = _rank_textrank(matrix, n_neighbors=textrank_neighbors)
        textrank_rank_seconds = time.perf_counter() - textrank_started
    else:
        tfidf_scores = np.ones(len(pool), dtype=float)
        textrank_scores = np.ones(len(pool), dtype=float) / len(pool)
        sim = None
        tfidf_rank_seconds = 0.0
        textrank_rank_seconds = 0.0
    pool["tfidf_score"] = tfidf_scores
    pool["textrank_score"] = textrank_scores

    n = min(max(1, int(n_sentences)), len(pool))
    tfidf_idx = sorted(np.argsort(-tfidf_scores, kind="stable")[:n].tolist())
    textrank_idx = sorted(np.argsort(-textrank_scores, kind="stable")[:n].tolist())
    mmr_started = time.perf_counter()
    # Pertahankan urutan greedy MMR: kalimat pertama adalah pilihan relevansi-diversity
    # terbaik, kalimat berikutnya dipilih setelah mempertimbangkan redundansi sebelumnya.
    mmr_idx = _select_mmr(textrank_scores, matrix, n, float(mmr_lambda)) if matrix is not None else textrank_idx
    mmr_rank_seconds = time.perf_counter() - mmr_started

    selections = {
        "TF-IDF Sentence Ranking": (tfidf_idx, tfidf_scores, False),
        "TextRank": (textrank_idx, textrank_scores, False),
        "TextRank + MMR": (mmr_idx, textrank_scores, True),
    }
    outputs = []
    for model_name, (idxs, scores, preserve_selection_order) in selections.items():
        chosen = pool.iloc[idxs].copy()
        # MMR mempertahankan urutan greedy; dua baseline diurutkan berdasarkan skor aslinya.
        chosen = chosen.assign(method_score=[float(scores[i]) for i in idxs])
        if not preserve_selection_order:
            chosen = chosen.sort_values("method_score", ascending=False, kind="stable")
        chosen = chosen.reset_index(drop=True)
        for rank, row in enumerate(chosen.itertuples(index=False), start=1):
            outputs.append({
                "topic_id": int(topic_id) if topic_id is not None else -999,
                "model": model_name,
                "rank": rank,
                "sentence_id": row.sentence_id,
                "sentence": row.sentence,
                "source_review_ids": row.source_review_ids,
                "source_review_count": int(row.source_review_count),
                "source_occurrences": int(row.source_occurrences),
                "method_score": float(row.method_score),
            })
    output = pd.DataFrame(outputs)
    all_metrics = {}
    runtime_by_model = {
        "TF-IDF Sentence Ranking": vectorization_seconds + tfidf_rank_seconds,
        "TextRank": vectorization_seconds + textrank_rank_seconds,
        "TextRank + MMR": vectorization_seconds + textrank_rank_seconds + mmr_rank_seconds,
    }
    for model_name in MODEL_NAMES:
        selected = output[output["model"] == model_name]
        selected_sents = selected["sentence"].astype(str).tolist()
        if len(selected_sents) >= 2:
            sm = _vectorize(selected_sents)
            pair_sim = cosine_similarity(sm)
            vals = [pair_sim[i,j] for i,j in combinations(range(len(selected_sents)), 2)]
            redundancy = float(np.clip(np.mean(vals), 0.0, 1.0)) if vals else 0.0
        else:
            redundancy = 0.0
        if keywords:
            summary_tokens = tokenize_id(" ".join(selected_sents))
            kw_tokens = [tokenize_id(k) for k in keywords]
            kw_tokens = [tokens for tokens in kw_tokens if tokens]
            matched = 0
            for phrase in kw_tokens:
                width = len(phrase)
                found = any(summary_tokens[i:i + width] == phrase for i in range(len(summary_tokens) - width + 1))
                matched += int(found)
            keyword_coverage = matched / max(1, len(kw_tokens))
        else:
            keyword_coverage = 0.0
        # Ini hanya mengukur apakah setiap kalimat terpilih memiliki reviewId sumber.
        # Bukan pengukuran faithfulness semantik; groundedness dinilai manusia.
        traceable = 0
        for raw in selected["source_review_ids"].astype(str):
            try:
                ids = json.loads(raw)
                traceable += int(bool(ids))
            except Exception:
                pass
        source_traceability_rate = traceable / max(1, len(selected))
        all_metrics[model_name] = {
            "keyword_coverage": float(keyword_coverage),
            "redundancy": float(redundancy),
            "source_traceability_rate": float(source_traceability_rate),
            "processing_seconds": float(runtime_by_model[model_name]),
        }
    return output, all_metrics


def build_summary_texts(source_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if source_rows.empty:
        return pd.DataFrame(columns=["topic_id", "model", "summary", "selected_sentence_count"])
    for (tid, model), group in source_rows.groupby(["topic_id", "model"], sort=False):
        group = group.sort_values("rank")
        rows.append({
            "topic_id": int(tid),
            "model": model,
            "summary": " ".join(group["sentence"].astype(str).tolist()),
            "selected_sentence_count": int(len(group)),
        })
    return pd.DataFrame(rows)
