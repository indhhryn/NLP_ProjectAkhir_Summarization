from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st
import plotly.express as px
from kredivo_project.config import OUTPUT_DIR, load_config
from kredivo_project.pipeline import run_pipeline

st.set_page_config(page_title="Kredivo Topic Summarizer", page_icon="🧭", layout="wide")
st.title("Topic-Based, Evidence-Grounded Multi-Document Extractive Summarization")
st.caption("Analisis ulasan Kredivo dengan BERTopic, c-TF-IDF, TF-IDF Ranking, TextRank, dan TextRank + MMR")


def read_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False, low_memory=False)
    except Exception:
        return pd.DataFrame()


def read_json(name: str, fallback=None):
    path = OUTPUT_DIR / name
    if not path.exists():
        return fallback if fallback is not None else {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback if fallback is not None else {}


with st.sidebar:
    st.header("Kontrol proyek")
    st.caption("Jalankan pipeline satu kali dari terminal sebelum eksplorasi dashboard.")
    if st.button("Jalankan pipeline NLP", type="primary", use_container_width=True):
        try:
            with st.spinner("Menjalankan audit data, BERTopic, summarization, dan evaluasi..."):
                status = run_pipeline()
            st.success("Pipeline selesai.")
            st.json(status)
            st.rerun()
        except Exception as exc:
            st.error(f"Pipeline gagal: {exc}")
            st.exception(exc)
    st.divider()
    st.caption("Output disimpan di folder outputs/.")

status = read_json("run_status.json")
topics = read_csv("topics.csv")
keywords = read_csv("topic_keywords_ctfidf.csv")
sources = read_csv("summary_sources.csv")
summaries = read_csv("summaries.csv")
reviews = read_csv("reviews_assigned.csv")
comparison = read_csv("evaluation_comparison.csv")
errors = read_csv("error_analysis_cases.csv")
length_ablation = read_csv("summary_length_ablation_comparison.csv")
quality = read_json("data_quality_report.json")
audit = read_json("label_audit_status.json", {"passed": False, "status": "not_audited", "reason": "Audit belum dijalankan."})

if topics.empty or summaries.empty:
    st.info("Belum ada hasil pipeline. Pilih tombol **Jalankan pipeline NLP** di panel kiri atau jalankan `run_windows.bat` dari folder proyek.")
    st.markdown("Setelah pertama kali dijalankan, model multilingual akan diunduh satu kali dari Hugging Face. Koneksi internet diperlukan untuk unduhan awal ini.")
    st.stop()

# Metric overview
m1, m2, m3, m4 = st.columns(4)
m1.metric("Review asli", f"{quality.get('raw_review_count', 0):,}")
m2.metric("Teks unik", f"{quality.get('unique_normalized_reviews', 0):,}")
m3.metric("Topik ditemukan", f"{status.get('topics_created', 0):,}")
m4.metric("Outlier BERTopic", f"{status.get('outlier_documents', 0):,}")

page = st.sidebar.radio("Halaman", ["Ringkasan Proyek", "Eksplorasi Topik", "Perbandingan Model", "Audit Label & Data"])

if page == "Ringkasan Proyek":
    st.subheader("Topik dominan")
    display_topics = topics[topics["topic_id"].astype(int) >= 0].head(15).copy()
    if not display_topics.empty:
        fig = px.bar(display_topics.sort_values("total_source_reviews"), x="total_source_reviews", y="topic_label", orientation="h", title="Jumlah ulasan sumber per topik")
        fig.update_layout(yaxis_title="Topik", xaxis_title="Jumlah ulasan", height=520)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Ringkasan dari tiga metode")
    st.caption("Ringkasan bersifat extractive: setiap kalimat diambil dari review sumber dan dapat ditelusuri melalui reviewId.")
    summary_preview = summaries.merge(topics[["topic_id", "topic_label", "total_source_reviews"]], on="topic_id", how="left")
    st.dataframe(summary_preview[["topic_id", "topic_label", "model", "summary", "total_source_reviews"]], use_container_width=True, hide_index=True)
    st.subheader("Catatan metodologi")
    st.markdown("- BERTopic mengelompokkan review berdasarkan sentence embeddings multilingual; c-TF-IDF memberi kata kunci topik.\n- TF-IDF Sentence Ranking dan TextRank adalah baseline sesuai modul praktikum; TextRank + MMR adalah varian untuk mengurangi pengulangan.\n- Review identik digabung saat pemodelan agar duplikasi tidak mendominasi, tetapi jumlah kemunculan dan semua reviewId tetap disimpan.\n- Pujian generik yang berdiri sendiri dikecualikan dari clustering agar topik lebih informatif; review pendek yang mengandung keluhan tetap dipertahankan.")

elif page == "Eksplorasi Topik":
    st.subheader("Pilih topik untuk ditelusuri")
    topic_options = topics[topics["topic_id"].astype(int) >= 0].copy()
    topic_options = topic_options.sort_values("total_source_reviews", ascending=False)
    label_by_id = {int(r.topic_id): f"{int(r.topic_id)} — {r.topic_label} ({int(r.total_source_reviews):,} review)" for r in topic_options.itertuples(index=False)}
    selected_id = st.selectbox("Topik", list(label_by_id), format_func=lambda x: label_by_id[x])
    meta = topic_options[topic_options["topic_id"].astype(int) == int(selected_id)].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Review sumber", f"{int(meta['total_source_reviews']):,}")
    c2.metric("Dokumen unik", f"{int(meta['unique_documents']):,}")
    c3.metric("Validation docs", f"{int(meta.get('validation_documents', 0)):,}")
    c4.metric("Test docs", f"{int(meta['test_documents']):,}")
    kw_row = keywords[keywords["topic_id"].astype(int) == int(selected_id)]
    if not kw_row.empty:
        st.markdown("**Kata kunci c-TF-IDF**")
        st.write(kw_row.iloc[0]["keywords"])

    st.divider()
    methods = ["TF-IDF Sentence Ranking", "TextRank", "TextRank + MMR"]
    model_choice = st.selectbox("Metode yang ditampilkan", methods, index=2)
    topic_sources = sources[(sources["topic_id"].astype(int) == int(selected_id)) & (sources["model"] == model_choice)].sort_values("rank")
    summary_text = summaries[(summaries["topic_id"].astype(int) == int(selected_id)) & (summaries["model"] == model_choice)]
    st.markdown("**Hasil ringkasan**")
    if not summary_text.empty:
        st.write(summary_text.iloc[0]["summary"])
    st.markdown("**Kalimat terpilih dan bukti sumber**")
    severity_filter = "Semua severity"
    if bool(audit.get("passed", False)) and not reviews.empty and "severity_level" in reviews:
        values = ["Semua severity"] + sorted([x for x in reviews["severity_level"].astype(str).unique() if x and x != "nan"])
        severity_filter = st.selectbox("Filter sumber berdasarkan severity (audit lulus)", values)
    else:
        st.caption("Severity filter dinonaktifkan karena label issue/severity belum lolos audit manual.")
    for row in topic_sources.itertuples(index=False):
        st.markdown(f"**{int(row.rank)}.** {row.sentence}")
        try:
            ids = json.loads(row.source_review_ids)
        except Exception:
            ids = []
        evidence = reviews[reviews["reviewId"].astype(str).isin([str(x) for x in ids])].copy() if ids else pd.DataFrame()
        if severity_filter != "Semua severity" and not evidence.empty:
            evidence = evidence[evidence["severity_level"].astype(str) == severity_filter]
        st.caption(f"{int(row.source_review_count)} reviewId sumber · {len(evidence)} bukti tampil · kemunculan identik: {int(row.source_occurrences)}")
        if not evidence.empty:
            with st.expander("Lihat review asli / reviewId"):
                for ev in evidence.head(30).itertuples(index=False):
                    st.write(f"**reviewId:** `{ev.reviewId}` · score: {getattr(ev, 'score', '-')}")
                    st.write(ev.content)
                    st.divider()
        elif ids:
            st.caption("Tidak ada bukti yang cocok dengan filter severity yang dipilih.")
    if severity_filter != "Semua severity":
        st.info("Filter severity di sini menyaring daftar bukti sumber yang ditampilkan. Teks ringkasan tetap dihitung dari seluruh dokumen topik; filter tidak menulis ulang ringkasan.")

elif page == "Perbandingan Model":
    st.subheader("Evaluasi otomatis")
    if comparison.empty:
        st.warning("ROUGE belum dihitung karena dataset review tidak memiliki reference_summary. Isi kolom `reference_summary` pada `outputs/reference_summaries_template.csv` dengan ringkasan acuan yang ditulis manusia, lalu jalankan ulang pipeline.")
        st.markdown("**Cara membuat reference summary:** buka `reference_examples.csv`, baca contoh review untuk setiap topic_id, lalu tulis ringkasan singkat yang merangkum tema tanpa menambahkan fakta di luar review. Jangan menyalin hasil dari salah satu model sebagai referensi.")
    else:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        rouge_plot = comparison.melt(id_vars="model", value_vars=["ROUGE_1", "ROUGE_2", "ROUGE_L"], var_name="Metrik", value_name="F1")
        fig = px.bar(rouge_plot, x="model", y="F1", color="Metrik", barmode="group", title="Perbandingan ROUGE F1")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ROUGE mengukur overlap leksikal terhadap ringkasan referensi, bukan kesamaan makna sepenuhnya.")
        st.markdown("**Source traceability rate** mengukur apakah kalimat terpilih memiliki `reviewId` sumber. Ini bukan skor faithfulness semantik; groundedness perlu dinilai manusia pada template penilaian.")
    if not length_ablation.empty:
        st.subheader("Eksperimen panjang ringkasan")
        fig2 = px.line(length_ablation, x="summary_sentences", y="ROUGE_L", color="model", markers=True, title="Dampak jumlah kalimat terhadap ROUGE-L")
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(length_ablation, use_container_width=True, hide_index=True)
    if not errors.empty:
        st.subheader("Contoh prediksi terbaik dan terburuk untuk error analysis")
        st.dataframe(errors[[c for c in ["topic_id", "model", "case_type", "ROUGE_L", "prediction", "reference", "error_reason", "reviewer_notes"] if c in errors]], use_container_width=True, hide_index=True)
    st.info("Penilaian manual tersedia pada `outputs/human_evaluation_template.csv` (groundedness, relevance, coherence, non-redundancy skala 1–5). Isi setelah menilai keluaran, kemudian jalankan ulang pipeline untuk merangkum nilainya.")

elif page == "Audit Label & Data":
    st.subheader("Data quality audit")
    if quality:
        q1, q2, q3 = st.columns(3)
        q1.metric("Duplikasi tambahan", f"{quality.get('duplicate_extra_rows',0):,}")
        q2.metric("Unique review", f"{quality.get('unique_normalized_reviews',0):,}")
        q3.metric("Pujian generik dikecualikan", f"{quality.get('generic_praise_unique_docs',0):,}")
        if "sentiment_distribution" in quality:
            st.markdown("**Distribusi sentimen pada CSV (belum diverifikasi)**")
            st.dataframe(pd.DataFrame(quality["sentiment_distribution"].items(), columns=["sentiment", "jumlah"]), hide_index=True)
        st.download_button("Unduh laporan audit data", (OUTPUT_DIR / "data_quality_report.md").read_bytes(), file_name="data_quality_report.md")
    st.divider()
    st.subheader("Pemeriksaan kualitas cluster BERTopic")
    topic_audit_path = OUTPUT_DIR / "topic_quality_review_template.csv"
    if topic_audit_path.exists():
        st.caption("Template ini memakai contoh dokumen validation, bukan test. Isi skor koherensi topik dan relevansi kata kunci setelah membaca contoh review.")
        st.download_button("Unduh template audit kualitas topik", topic_audit_path.read_bytes(), file_name="topic_quality_review_template.csv")
    st.divider()
    st.subheader("Status validasi label issue_category dan severity_level")
    if audit.get("passed"):
        st.success(f"Audit label lulus. Issue category: {audit.get('agreement_issue_category', 0):.1%} · Severity: {audit.get('agreement_severity_level', 0):.1%}")
    else:
        st.warning(audit.get("reason", "Label belum divalidasi oleh manusia."))
    if audit:
        st.json(audit)
    st.markdown("Unduh `outputs/label_audit_sample.csv`, isi `manual_issue_category` dan `manual_severity_level`, pastikan setiap baris `reviewed=true`, simpan kembali, lalu jalankan ulang pipeline. Severity filter baru aktif saat agreement keseluruhan dan per kelas memenuhi ambang serta setiap kelas memiliki sampel minimum.")
    if (OUTPUT_DIR / "label_audit_sample.csv").exists():
        st.download_button("Unduh sampel audit label", (OUTPUT_DIR / "label_audit_sample.csv").read_bytes(), file_name="label_audit_sample.csv")
    st.caption("Label tambahan pada dataset tidak digunakan sebagai ground truth untuk clustering atau summarization. Nama pengguna dan gambar profil tidak dipakai dalam pipeline/dashboard.")
