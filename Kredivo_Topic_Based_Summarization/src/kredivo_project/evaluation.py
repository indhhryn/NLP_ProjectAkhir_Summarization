from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .summarization import summarize_topic, build_summary_texts, MODEL_NAMES


def create_reference_templates(assignments: pd.DataFrame, topics: pd.DataFrame, output_dir: Path, cfg: dict) -> None:
    """Buat lembar kerja anotasi manusia; kolom reference_summary sengaja dikosongkan."""
    refs_path = output_dir / "reference_summaries_template.csv"
    examples_path = output_dir / "reference_examples.csv"
    test_docs = assignments[(assignments["topic_split"] == "test") & (assignments["topic_id"] >= 0)].copy()
    test_sizes = test_docs.groupby("topic_id").size().sort_values(ascending=False)
    chosen_topics = test_sizes.head(int(cfg["reference_topics"])).index.tolist()
    label_map = topics.set_index("topic_id")["topic_label"].to_dict() if not topics.empty else {}
    existing = pd.read_csv(refs_path, keep_default_na=False) if refs_path.exists() else pd.DataFrame()
    existing_map = {int(r.topic_id): r._asdict() for r in existing.itertuples(index=False)} if not existing.empty else {}
    rows, example_rows = [], []
    for tid in chosen_topics:
        group = test_docs[test_docs["topic_id"] == tid].copy()
        # Sampel deterministik dan bervariasi; tidak membuat ringkasan acuan otomatis.
        take = min(int(cfg["reference_examples_per_topic"]), len(group))
        examples = group.sample(n=take, random_state=int(cfg["random_state"]) + int(tid) + 100).sort_values("doc_id")
        ref_ids = []
        for row in examples.itertuples(index=False):
            try: ids = json.loads(row.source_review_ids)
            except Exception: ids = []
            if ids:
                rid = str(ids[0])
                ref_ids.append(rid)
                example_rows.append({"topic_id": int(tid), "doc_id": str(row.doc_id), "reviewId": rid, "content": row.content, "duplicate_count": int(row.duplicate_count)})
        old = existing_map.get(int(tid), {})
        evaluation_doc_ids = json.dumps(examples["doc_id"].astype(str).tolist(), ensure_ascii=False)
        same_evaluation_sample = str(old.get("evaluation_doc_ids", "")) == evaluation_doc_ids
        rows.append({
            "topic_id": int(tid),
            "topic_label": str(label_map.get(int(tid), "")),
            "test_unique_review_count": int(len(group)),
            "representative_review_ids": " | ".join(ref_ids),
            # Evaluasi memakai dokumen contoh yang sama persis dengan yang dibaca anotator.
            "evaluation_doc_ids": evaluation_doc_ids,
            "reference_summary": str(old.get("reference_summary", "")) if same_evaluation_sample else "",
            "annotator": str(old.get("annotator", "")) if same_evaluation_sample else "",
            "adjudicator": str(old.get("adjudicator", "")) if same_evaluation_sample else "",
            "annotation_notes": str(old.get("annotation_notes", "")) if same_evaluation_sample else "",
        })
    new_refs = pd.DataFrame(rows)
    # Pertahankan ringkasan manual yang telah diisi saat pipeline dijalankan ulang.
    new_refs.to_csv(refs_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(example_rows).to_csv(examples_path, index=False, encoding="utf-8-sig")

    # Template penilaian manusia untuk kualitas semantik dan faithfulness.
    human_path = output_dir / "human_evaluation_template.csv"
    human_rows = []
    for tid in chosen_topics:
        for model_name in MODEL_NAMES:
            human_rows.append({
                "topic_id": int(tid), "model": model_name,
                "summary_sentences": int(cfg.get("summary_sentences", 5)),
                "evaluation_doc_ids": "", "reference_summary": "", "prediction": "",
                "output_fingerprint": "",
                "groundedness_1_to_5": "", "relevance_1_to_5": "",
                "coherence_1_to_5": "", "non_redundancy_1_to_5": "",
                "reviewer": "", "notes": ""
            })
    if not human_path.exists():
        pd.DataFrame(human_rows).to_csv(human_path, index=False, encoding="utf-8-sig")



def create_topic_quality_template(assignments: pd.DataFrame, topics: pd.DataFrame, output_dir: Path, cfg: dict) -> None:
    """Prepare a human inspection sheet from validation documents only.

    The examples help reviewers judge whether each cluster is coherent and whether
    its c-TF-IDF keywords represent the reviews. This template is not an automatic
    topic-quality score and must be completed by humans.
    """
    path = output_dir / "topic_quality_review_template.csv"
    validation = assignments[(assignments["topic_split"] == "validation") & (assignments["topic_id"] >= 0)].copy()
    if validation.empty:
        return
    ranked = topics[topics["topic_id"] >= 0].copy()
    validation_counts = validation.groupby("topic_id").size()
    ranked["_validation_count"] = ranked["topic_id"].map(validation_counts).fillna(0).astype(int)
    ranked = ranked[ranked["_validation_count"] > 0].sort_values("_validation_count", ascending=False)
    topic_ids = ranked.head(int(cfg.get("reference_topics", 30)))["topic_id"].astype(int).tolist()
    labels = topics.set_index("topic_id")["topic_label"].to_dict() if "topic_label" in topics else {}
    keywords = topics.set_index("topic_id")["keywords"].to_dict() if "keywords" in topics else {}
    old = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame()
    old_map = {int(r.topic_id): r._asdict() for r in old.itertuples(index=False)} if not old.empty else {}
    rows = []
    for tid in topic_ids:
        group = validation[validation["topic_id"] == tid]
        if group.empty:
            continue
        take = min(5, len(group))
        examples = group.sample(n=take, random_state=int(cfg["random_state"]) + tid + 500).sort_values("doc_id")
        snippets = []
        review_ids = []
        for row in examples.itertuples(index=False):
            try:
                ids = json.loads(row.source_review_ids)
            except Exception:
                ids = []
            review_id = str(ids[0]) if ids else str(row.doc_id)
            review_ids.append(review_id)
            snippets.append(f"[{review_id}] {row.content}")
        previous = old_map.get(tid, {})
        current_review_ids = " | ".join(review_ids)
        current_label = str(labels.get(tid, ""))
        current_keywords = str(keywords.get(tid, ""))
        import hashlib
        fingerprint = hashlib.sha256("|".join([current_label, current_keywords, current_review_ids]).encode("utf-8")).hexdigest()
        same_topic_sample = str(previous.get("topic_fingerprint", "")) == fingerprint
        rows.append({
            "topic_id": int(tid),
            "topic_label_auto": current_label,
            "ctfidf_keywords": current_keywords,
            "validation_document_count": int(len(group)),
            "sample_review_ids": current_review_ids,
            "sample_reviews": "\n---\n".join(snippets),
            "topic_fingerprint": fingerprint,
            "human_topic_coherence_1_to_5": str(previous.get("human_topic_coherence_1_to_5", "")) if same_topic_sample else "",
            "human_keyword_relevance_1_to_5": str(previous.get("human_keyword_relevance_1_to_5", "")) if same_topic_sample else "",
            "is_mixed_topic": str(previous.get("is_mixed_topic", "")) if same_topic_sample else "",
            "reviewer": str(previous.get("reviewer", "")) if same_topic_sample else "",
            "notes": str(previous.get("notes", "")) if same_topic_sample else "",
        })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _rouge_scores(reference: str, prediction: str) -> dict:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError("Library rouge-score belum tersedia. Jalankan pip install -r requirements.txt.") from exc
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(str(reference), str(prediction))
    return {name: float(scores[name].fmeasure) for name in ["rouge1", "rouge2", "rougeL"]}


def _update_human_evaluation_template(predictions: pd.DataFrame, refs: pd.DataFrame,
                                      output_dir: Path) -> None:
    """Attach the exact held-out prediction to the manual scoring sheet.

    Existing human scores are preserved only when the prediction fingerprint is
    unchanged. Re-running with different model output clears stale ratings.
    """
    import hashlib
    path = output_dir / "human_evaluation_template.csv"
    old = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame()
    old_map = {}
    if not old.empty and {"topic_id", "model"}.issubset(old.columns):
        old_map = {(int(row.topic_id), str(row.model)): row._asdict() for row in old.itertuples(index=False)}
    ref_ids = {}
    for row in refs.itertuples(index=False):
        ref_ids[int(row.topic_id)] = str(getattr(row, "evaluation_doc_ids", ""))

    score_columns = ["groundedness_1_to_5", "relevance_1_to_5", "coherence_1_to_5", "non_redundancy_1_to_5"]
    rows = []
    for row in predictions.itertuples(index=False):
        tid, model = int(row.topic_id), str(row.model)
        evaluation_doc_ids = ref_ids.get(tid, "")
        fingerprint_payload = "|".join([
            str(tid), model, str(int(row.summary_sentences)), evaluation_doc_ids,
            str(row.prediction), str(row.reference),
        ])
        fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
        previous = old_map.get((tid, model), {})
        same_output = str(previous.get("output_fingerprint", "")) == fingerprint
        record = {
            "topic_id": tid, "model": model,
            "summary_sentences": int(row.summary_sentences),
            "evaluation_doc_ids": evaluation_doc_ids,
            "reference_summary": str(row.reference),
            "prediction": str(row.prediction),
            "output_fingerprint": fingerprint,
        }
        for column in score_columns + ["reviewer", "notes"]:
            record[column] = str(previous.get(column, "")) if same_output else ""
        rows.append(record)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _human_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, keep_default_na=False)
    score_cols = ["groundedness_1_to_5", "relevance_1_to_5", "coherence_1_to_5", "non_redundancy_1_to_5"]
    for c in score_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def run_evaluation(assignments: pd.DataFrame, topics: pd.DataFrame, keywords: pd.DataFrame,
                   output_dir: Path, cfg: dict) -> dict:
    refs_path = output_dir / "reference_summaries_template.csv"
    if not refs_path.exists():
        return {"status": "awaiting_reference_summaries", "evaluated_topics": 0}
    refs = pd.read_csv(refs_path, keep_default_na=False)
    refs["reference_summary"] = refs["reference_summary"].astype(str).str.strip()
    refs = refs[refs["reference_summary"].ne("")].copy()
    if refs.empty:
        # Simpan keluaran evaluasi kosong dan bersihkan penilaian manusia yang sudah
        # tidak terikat pada reference/test sample yang aktif.
        pd.DataFrame(columns=["model", "rouge1", "rouge2", "rougeL", "keyword_coverage", "redundancy", "source_traceability_rate", "processing_seconds_mean"]).to_csv(output_dir / "evaluation_comparison.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "evaluation_predictions.csv", index=False)
        human_path = output_dir / "human_evaluation_template.csv"
        if human_path.exists():
            human = pd.read_csv(human_path, keep_default_na=False)
            for column in ["evaluation_doc_ids", "reference_summary", "prediction", "output_fingerprint", "groundedness_1_to_5", "relevance_1_to_5", "coherence_1_to_5", "non_redundancy_1_to_5", "reviewer", "notes"]:
                if column in human:
                    human[column] = ""
            human.to_csv(human_path, index=False, encoding="utf-8-sig")
        return {"status": "awaiting_reference_summaries", "evaluated_topics": 0,
                "message": "Isi reference_summary dengan ringkasan yang ditulis manusia; jangan gunakan keluaran model sebagai referensi."}

    test_docs = assignments[(assignments["topic_split"] == "test") & (assignments["topic_id"] >= 0)].copy()
    keyword_map = keywords.set_index("topic_id")["keywords"].to_dict() if not keywords.empty else {}
    pred_rows, ablation_rows = [], []
    base_budget = int(cfg.get("summary_sentences", 5))
    for ref in refs.itertuples(index=False):
        tid = int(ref.topic_id)
        topic_docs = test_docs[test_docs["topic_id"] == tid].copy()
        raw_eval_ids = str(getattr(ref, "evaluation_doc_ids", "")).strip()
        if not raw_eval_ids:
            raise ValueError(f"evaluation_doc_ids kosong untuk topic_id={tid}; evaluasi dihentikan agar tidak diam-diam memakai seluruh test topic.")
        try:
            parsed_ids = json.loads(raw_eval_ids)
            if not isinstance(parsed_ids, list) or not parsed_ids:
                raise ValueError("harus berupa JSON list yang tidak kosong")
            eval_doc_ids = set(str(x) for x in parsed_ids)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"evaluation_doc_ids tidak valid untuk topic_id={tid}: {raw_eval_ids}") from exc
        available_ids = set(topic_docs["doc_id"].astype(str))
        missing_ids = eval_doc_ids - available_ids
        if missing_ids:
            raise ValueError(f"evaluation_doc_ids untuk topic_id={tid} memuat {len(missing_ids)} dokumen yang bukan bagian test topic ini.")
        topic_docs = topic_docs[topic_docs["doc_id"].astype(str).isin(eval_doc_ids)].copy()
        if topic_docs.empty:
            raise ValueError(f"Tidak ada dokumen uji yang cocok untuk topic_id={tid}.")
        kw = [x.strip() for x in str(keyword_map.get(tid, "")).split("|") if x.strip()]
        for budget in sorted(set([1, 2, 3, base_budget])):
            sources, metrics = summarize_topic(topic_docs, n_sentences=budget, mmr_lambda=float(cfg["mmr_lambda"]), keywords=kw, topic_id=tid)
            summaries = build_summary_texts(sources)
            for row in summaries.itertuples(index=False):
                rouge = _rouge_scores(str(ref.reference_summary), str(row.summary))
                met = metrics.get(row.model, {})
                record = {
                    "topic_id": tid, "topic_label": str(getattr(ref, "topic_label", "")),
                    "model": row.model, "summary_sentences": int(budget),
                    "prediction": str(row.summary), "reference": str(ref.reference_summary),
                    "test_unique_review_count": int(len(topic_docs)),
                    "source_review_count": int(topic_docs["source_review_count"].sum()),
                    "keyword_coverage": float(met.get("keyword_coverage", 0.0)),
                    "redundancy": float(met.get("redundancy", 0.0)),
                    "source_traceability_rate": float(met.get("source_traceability_rate", 0.0)),
                    "processing_seconds": float(met.get("processing_seconds", 0.0)),
                    **rouge,
                }
                if budget == base_budget:
                    pred_rows.append(record)
                ablation_rows.append(record)

    predictions = pd.DataFrame(pred_rows)
    ablation = pd.DataFrame(ablation_rows)
    if not predictions.empty:
        _update_human_evaluation_template(predictions, refs, output_dir)
    predictions.to_csv(output_dir / "evaluation_predictions.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(output_dir / "summary_length_ablation.csv", index=False, encoding="utf-8-sig")
    if predictions.empty:
        return {"status": "no_evaluable_topics", "evaluated_topics": 0}

    human = _human_scores(output_dir / "human_evaluation_template.csv")
    if not human.empty:
        human_cols = [c for c in ["groundedness_1_to_5", "relevance_1_to_5", "coherence_1_to_5", "non_redundancy_1_to_5"] if c in human]
        has_manual_scores = bool(human[human_cols].notna().any(axis=None)) if human_cols else False
        if human_cols and has_manual_scores:
            human_agg = human.groupby(["topic_id", "model"], as_index=False)[human_cols].mean()
            predictions = predictions.merge(human_agg, on=["topic_id", "model"], how="left")
            predictions.to_csv(output_dir / "evaluation_predictions.csv", index=False, encoding="utf-8-sig")

    comparison = predictions.groupby("model", as_index=False).agg(
        evaluated_topics=("topic_id", "nunique"),
        ROUGE_1=("rouge1", "mean"), ROUGE_2=("rouge2", "mean"), ROUGE_L=("rougeL", "mean"),
        keyword_coverage=("keyword_coverage", "mean"), redundancy=("redundancy", "mean"),
        source_traceability_rate=("source_traceability_rate", "mean"), processing_seconds_mean=("processing_seconds", "mean"),
    )
    if not human.empty:
        for c in ["groundedness_1_to_5", "relevance_1_to_5", "coherence_1_to_5", "non_redundancy_1_to_5"]:
            if c in predictions and predictions[c].notna().any():
                comparison[c] = predictions.groupby("model")[c].mean().reindex(comparison["model"]).to_numpy()
    comparison.to_csv(output_dir / "evaluation_comparison.csv", index=False, encoding="utf-8-sig")

    if not ablation.empty:
        ablation_summary = ablation.groupby(["model", "summary_sentences"], as_index=False).agg(
            ROUGE_1=("rouge1", "mean"), ROUGE_2=("rouge2", "mean"), ROUGE_L=("rougeL", "mean"),
            keyword_coverage=("keyword_coverage", "mean"), redundancy=("redundancy", "mean"),
            processing_seconds_mean=("processing_seconds", "mean"),
        )
        ablation_summary.to_csv(output_dir / "summary_length_ablation_comparison.csv", index=False, encoding="utf-8-sig")

    # 3 kasus terbaik dan terburuk per metode untuk dibahas pada error analysis.
    cases = []
    for model_name, group in predictions.groupby("model"):
        for label, selected in [("worst", group.nsmallest(3, "rougeL")), ("best", group.nlargest(3, "rougeL"))]:
            for row in selected.itertuples(index=False):
                cases.append({**row._asdict(), "case_type": label, "error_reason": "", "reviewer_notes": ""})
    pd.DataFrame(cases).to_csv(output_dir / "error_analysis_cases.csv", index=False, encoding="utf-8-sig")

    return {
        "status": "completed",
        "evaluated_topics": int(predictions["topic_id"].nunique()),
        "reference_rows": int(len(predictions)),
        "summary_sentences": base_budget,
    }
