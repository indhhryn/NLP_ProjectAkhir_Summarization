from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

import kredivo_project  # noqa: F401 - ensures NUMBA_CACHE_DIR is configured early
from kredivo_project.config import OUTPUT_DIR, load_config
from kredivo_project.explorer import (
    compare_summarizers_on_docs,
    filter_reviews,
    get_available_filter_options,
    get_evaluation_status_summary,
    parse_keyword_scores,
)
from kredivo_project.inference import (
    analyze_new_reviews,
    load_inference_bundle,
    parse_manual_reviews,
    validate_and_prepare_uploaded_csv,
)
from kredivo_project.pipeline import run_pipeline
from kredivo_project.summarization import MODEL_NAMES

st.set_page_config(
    page_title="Kredivo Topic Summarizer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    .method-pill {
        display: inline-block; padding: 0.25rem 0.65rem; border-radius: 999px;
        background-color: #eef2ff; color: #1e3a8a; font-size: 0.82rem; font-weight: 600;
        margin-right: 0.4rem; margin-bottom: 0.3rem; border: 1px solid #c7d2fe;
    }
    .summary-card {
        padding: 0.9rem 1rem; border-radius: 0.6rem; border: 1px solid #e2e8f0;
        background-color: #f8fafc; margin-bottom: 0.75rem; line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _load_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False, low_memory=False)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_json_cached(path_str: str, mtime: float) -> dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    return _load_csv_cached(str(path), _file_mtime(path))


def read_json(name: str, fallback=None):
    path = OUTPUT_DIR / name
    data = _load_json_cached(str(path), _file_mtime(path))
    if not data and fallback is not None:
        return fallback
    return data


@st.cache_resource(show_spinner="Memuat model SentenceTransformer & BERTopic...")
def get_cached_inference_bundle(model_dir_str: str, embedding_model: str, device: str) -> dict:
    cfg = load_config()
    cfg["embedding_model"] = embedding_model
    cfg["device"] = device
    return load_inference_bundle(Path(model_dir_str), cfg)


def _render_source_evidence_list(
    sentences_df: pd.DataFrame,
    reviews_lookup_df: pd.DataFrame,
    severity_filter: str = "Semua severity",
    key_prefix: str = "ev",
) -> None:
    for row in sentences_df.itertuples(index=False):
        st.markdown(f"**{int(row.rank)}.** {row.sentence}")
        try:
            ids = [str(x) for x in json.loads(row.source_review_ids)]
        except Exception:
            ids = []
        evidence = (
            reviews_lookup_df[reviews_lookup_df["reviewId"].astype(str).isin(ids)].copy()
            if ids and not reviews_lookup_df.empty
            else pd.DataFrame()
        )
        if severity_filter != "Semua severity" and not evidence.empty and "severity_level" in evidence:
            evidence = evidence[evidence["severity_level"].astype(str) == severity_filter]

        score_val = getattr(row, "method_score", None)
        score_text = f" · skor algoritma: `{float(score_val):.4f}`" if score_val is not None else ""
        st.caption(
            f"{int(row.source_review_count)} reviewId sumber · {len(evidence)} bukti tampil · "
            f"kemunculan identik: {int(row.source_occurrences)}{score_text}"
        )
        if not evidence.empty:
            with st.expander(f"Telusuri review sumber ({len(evidence)} bukti) — Kalimat #{int(row.rank)}", expanded=False):
                for ev in evidence.head(30).itertuples(index=False):
                    score_info = f" · rating CSV: {ev.score}" if hasattr(ev, "score") and str(ev.score) != "" else ""
                    st.markdown(f"**reviewId:** `{ev.reviewId}`{score_info}")
                    st.write(ev.content)
                    st.divider()
        elif ids:
            st.caption("Tidak ada bukti yang cocok dengan filter severity yang dipilih.")


def render_overview(
    status: dict,
    quality: dict,
    topics: pd.DataFrame,
    summaries: pd.DataFrame,
) -> None:
    st.markdown(
        '<span class="method-pill">Data Audit & Duplicate-Aware</span>'
        '<span class="method-pill">BERTopic + c-TF-IDF</span>'
        '<span class="method-pill">TF-IDF Ranking</span>'
        '<span class="method-pill">TextRank</span>'
        '<span class="method-pill">TextRank + MMR</span>'
        '<span class="method-pill">Source Traceability</span>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Review Valid", f"{quality.get('raw_review_count', 0):,}")
    m2.metric("Dokumen Unik", f"{quality.get('unique_normalized_reviews', 0):,}")
    m3.metric("Topik BERTopic", f"{status.get('topics_created', 0):,}")
    m4.metric("Outlier (-1)", f"{status.get('outlier_documents', 0):,}")
    m5.metric("Pujian Generik (-2)", f"{quality.get('generic_praise_review_count', 0):,}")

    s1, s2, s3 = st.columns(3)
    s1.metric("Train Split (Fit & Proyeksi)", f"{status.get('train_documents', 0):,}")
    s2.metric("Validation Split (Audit Topik)", f"{status.get('validation_documents', 0):,}")
    s3.metric("Test Split (Evaluasi ROUGE)", f"{status.get('test_documents', 0):,}")

    st.subheader("Distribusi Topik Dominan (Dataset Utama)")
    regular_topics = topics[topics["topic_id"].astype(int) >= 0].copy()
    if not regular_topics.empty:
        display_topics = regular_topics.head(15).copy()
        display_topics["topic_name_short"] = display_topics.apply(
            lambda r: f"Topik {int(r['topic_id'])}: {str(r['topic_label'])[:55]}", axis=1
        )
        fig = px.bar(
            display_topics.sort_values("total_source_reviews"),
            x="total_source_reviews",
            y="topic_name_short",
            orientation="h",
            text="total_source_reviews",
            title="Jumlah Ulasan Sumber per Topik (Top 15 Topik Terbesar)",
        )
        fig.update_layout(yaxis_title="Topik BERTopic", xaxis_title="Jumlah ulasan sumber", height=520)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    outlier_row = topics[topics["topic_id"].astype(int) == -1]
    if not outlier_row.empty:
        out_reviews = int(outlier_row.iloc[0]["total_source_reviews"])
        out_docs = int(outlier_row.iloc[0]["unique_documents"])
        st.info(
            f"**Catatan Outlier BERTopic (`topic_id = -1`):** Terdapat **{out_docs:,} dokumen unik** "
            f"(mewakili **{out_reviews:,} ulasan**) yang tidak dipaksakan masuk ke cluster topik tertentu. "
            "Outlier tetap dilacak secara eksplisit dan tidak dimasukkan ke dalam ringkasan per topik."
        )

    st.subheader("Pratinjau Ringkasan Ekstraktif per Topik dan Metode")
    st.caption(
        "Setiap kalimat ringkasan diambil langsung dari ulasan asli (extractive summarization) dan terhubung ke `reviewId` sumber."
    )
    if not summaries.empty and not topics.empty:
        summary_preview = summaries.merge(
            topics[["topic_id", "topic_label", "total_source_reviews"]], on="topic_id", how="left"
        )
        model_filter = st.multiselect(
            "Filter metode pada pratinjau",
            MODEL_NAMES,
            default=MODEL_NAMES,
            key="overview_model_filter",
        )
        if model_filter:
            summary_preview = summary_preview[summary_preview["model"].isin(model_filter)]
        st.dataframe(
            summary_preview[["topic_id", "topic_label", "model", "summary", "total_source_reviews"]],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Alur & Catatan Metodologi Penelitian", expanded=False):
        st.markdown(
            "- **Data Quality Audit & Duplicate-Aware Processing:** Teks identik setelah normalisasi digabung menjadi satu dokumen unik untuk menghindari dominasi duplikasi, sementara seluruh `reviewId` dan jumlah kemunculan tetap dipertahankan.\n"
            "- **Pemisahan Pujian Generik (`topic_id = -2`):** Pujian singkat yang berdiri sendiri (misalnya *bagus*, *mantap*) dikecualikan dari clustering agar cluster topik lebih informatif; keluhan singkat tetap dipertahankan.\n"
            "- **BERTopic & c-TF-IDF:** BERTopic di-fit hanya pada train split (`max_fit_docs=12.000`), lalu validation/test diproyeksikan via `transform()`. Kata kunci tiap topik diekstrak dengan c-TF-IDF.\n"
            "- **Tiga Metode Extractive Summarization:** TF-IDF Sentence Ranking (baseline 1), TextRank (baseline 2), dan TextRank + MMR (varian pengurang redundansi)."
        )


def render_topic_explorer(
    topics: pd.DataFrame,
    keywords: pd.DataFrame,
    summaries: pd.DataFrame,
    sources: pd.DataFrame,
    reviews: pd.DataFrame,
    audit: dict,
) -> None:
    st.header("Topic Explorer Interaktif")
    st.caption("Jelajahi struktur topik BERTopic, bobot kata kunci c-TF-IDF, distribusi ulasan, ringkasan 3 metode, serta bukti sumber asli.")

    if topics.empty:
        st.warning("Data topik belum tersedia. Jalankan pipeline terlebih dahulu.")
        return

    topic_options = topics.copy()
    topic_options["_sort_group"] = topic_options["topic_id"].astype(int).map(lambda x: 0 if x >= 0 else 1)
    topic_options = topic_options.sort_values(["_sort_group", "total_source_reviews"], ascending=[True, False])

    label_by_id = {}
    for r in topic_options.itertuples(index=False):
        tid = int(r.topic_id)
        prefix = "⚠️ Outlier (-1)" if tid == -1 else f"Topik {tid}"
        label_by_id[tid] = f"{prefix} — {r.topic_label} ({int(r.total_source_reviews):,} review)"

    selected_id = st.selectbox(
        "Pilih Topik (termasuk Outlier -1)",
        list(label_by_id),
        format_func=lambda x: label_by_id[x],
        key="topic_explorer_select",
    )

    meta = topic_options[topic_options["topic_id"].astype(int) == int(selected_id)].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Review", f"{int(meta['total_source_reviews']):,}")
    c2.metric("Dokumen Unik", f"{int(meta['unique_documents']):,}")
    c3.metric("Train Docs", f"{int(meta.get('train_documents', 0)):,}")
    c4.metric("Validation Docs", f"{int(meta.get('validation_documents', 0)):,}")
    c5.metric("Test Docs", f"{int(meta.get('test_documents', 0)):,}")

    topic_reviews = (
        reviews[reviews["topic_id"].astype(int) == int(selected_id)].copy()
        if not reviews.empty and "topic_id" in reviews
        else pd.DataFrame()
    )

    if int(selected_id) == -1:
        st.warning(
            "**Ini adalah kelompok Outlier BERTopic (`topic_id = -1`).** Dokumen di kelompok ini tidak memiliki kepadatan semantik yang cukup untuk masuk ke cluster topik stabil. "
            "Sesuai metodologi penelitian, outlier tidak digabung paksa dan tidak dibuatkan ringkasan topik."
        )
    else:
        col_kw, col_dist = st.columns([1.15, 1.0])
        with col_kw:
            st.subheader("Kata Kunci Utama c-TF-IDF")
            kw_row = keywords[keywords["topic_id"].astype(int) == int(selected_id)]
            if not kw_row.empty:
                st.write(kw_row.iloc[0]["keywords"])
                scores_df = parse_keyword_scores(str(kw_row.iloc[0].get("keyword_scores", "")))
                if not scores_df.empty:
                    fig_kw = px.bar(
                        scores_df.sort_values("ctfidf_score"),
                        x="ctfidf_score",
                        y="keyword",
                        orientation="h",
                        title=f"Skor c-TF-IDF Kata Kunci — Topik {selected_id}",
                    )
                    fig_kw.update_layout(yaxis_title="Kata Kunci", xaxis_title="Skor c-TF-IDF", height=360)
                    st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.info("Kata kunci c-TF-IDF belum tersedia untuk topik ini.")

        with col_dist:
            st.subheader("Distribusi Ulasan pada Topik")
            if not topic_reviews.empty:
                if "score" in topic_reviews.columns:
                    score_counts = (
                        topic_reviews["score"].astype(str).value_counts().reset_index()
                    )
                    score_counts.columns = ["score", "jumlah"]
                    score_counts = score_counts.sort_values("score")
                    fig_score = px.bar(
                        score_counts,
                        x="score",
                        y="jumlah",
                        title="Distribusi Rating Bintang (CSV)",
                        text="jumlah",
                    )
                    fig_score.update_layout(xaxis_title="Rating (score)", yaxis_title="Jumlah Review", height=220)
                    st.plotly_chart(fig_score, use_container_width=True)

                if "sentiment" in topic_reviews.columns:
                    sent_counts = topic_reviews["sentiment"].astype(str).value_counts().reset_index()
                    sent_counts.columns = ["sentiment", "jumlah"]
                    st.caption("Distribusi kolom `sentiment` dari CSV (label awal — belum tervalidasi manual):")
                    st.dataframe(sent_counts, use_container_width=True, hide_index=True)
            else:
                st.info("Detail ulasan untuk topik ini belum tersedia.")

        st.divider()
        st.subheader("Ringkasan Ekstraktif & Source Traceability (Ketiga Metode)")

        severity_filter = "Semua severity"
        if bool(audit.get("passed", False)) and not reviews.empty and "severity_level" in reviews:
            values = ["Semua severity"] + sorted(
                [x for x in reviews["severity_level"].astype(str).unique() if x and x != "nan"]
            )
            severity_filter = st.selectbox(
                "Filter bukti sumber berdasarkan severity (audit lulus)",
                values,
                key="topic_explorer_severity",
            )
        else:
            st.caption("Severity filter dinonaktifkan karena label issue/severity belum lolos audit manual.")

        tabs = st.tabs(MODEL_NAMES)
        for tab, model_name in zip(tabs, MODEL_NAMES):
            with tab:
                summary_row = summaries[
                    (summaries["topic_id"].astype(int) == int(selected_id)) & (summaries["model"] == model_name)
                ]
                if not summary_row.empty:
                    st.markdown(
                        f'<div class="summary-card">{summary_row.iloc[0]["summary"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Ringkasan belum tersedia untuk metode ini.")

                topic_sources = sources[
                    (sources["topic_id"].astype(int) == int(selected_id)) & (sources["model"] == model_name)
                ].sort_values("rank")
                if not topic_sources.empty:
                    st.markdown("**Kalimat terpilih & penelusuran `reviewId` sumber:**")
                    _render_source_evidence_list(
                        topic_sources,
                        topic_reviews if not topic_reviews.empty else reviews,
                        severity_filter=severity_filter,
                        key_prefix=f"te_{selected_id}_{model_name}",
                    )

        if severity_filter != "Semua severity":
            st.info(
                "Filter severity hanya menyaring daftar bukti review yang ditampilkan; teks ringkasan tetap dihitung dari seluruh dokumen topik."
            )

    st.divider()
    st.subheader("Daftar & Contoh Review dalam Topik")
    if not topic_reviews.empty:
        col_search, col_limit = st.columns([2, 1])
        local_q = col_search.text_input(
            "Cari di dalam topik ini", placeholder="Ketik kata kunci atau reviewId...", key="te_local_search"
        )
        max_rows = col_limit.selectbox("Jumlah baris ditampilkan", [50, 100, 250, 500], index=1, key="te_limit")
        filtered_topic_reviews = filter_reviews(topic_reviews, search_query=local_q)
        st.caption(f"Menampilkan {min(len(filtered_topic_reviews), max_rows):,} dari {len(filtered_topic_reviews):,} review cocok.")
        show_cols = [
            c
            for c in ["reviewId", "content", "score", "topic_split", "sentiment", "issue_category", "at"]
            if c in filtered_topic_reviews.columns
        ]
        st.dataframe(
            filtered_topic_reviews[show_cols].head(int(max_rows)),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Tidak ada data review pada topik ini.")


def render_analyze_new_reviews(topics: pd.DataFrame, keywords: pd.DataFrame, cfg: dict) -> None:
    st.header("Analyze New Reviews")
    st.caption(
        "Uji pipeline NLP secara langsung pada review baru tanpa melatih ulang model penelitian: "
        "Validasi → Preprocessing → Embedding & BERTopic Inference → Keyword c-TF-IDF → Summarization (3 Metode) → Source Traceability."
    )

    model_dir = OUTPUT_DIR / "bertopic_model"
    if not model_dir.exists():
        st.error(
            f"Artifact model BERTopic belum ditemukan di `{model_dir}`. "
            "Jalankan pipeline utama terlebih dahulu agar model dari train split tersimpan sebelum melakukan inferensi review baru."
        )
        return

    try:
        model_bundle = get_cached_inference_bundle(
            str(model_dir),
            str(cfg["embedding_model"]),
            str(cfg.get("device", "cpu")),
        )
    except Exception as exc:
        st.error(f"Gagal memuat model inference: {exc}")
        return

    st.success(
        f"Model inference siap: `{cfg['embedding_model']}` + BERTopic artifact (`outputs/bertopic_model`). "
        "Data baru diproses pada session tanpa mengubah dataset utama atau melatih ulang BERTopic."
    )

    if "manual_review_items" not in st.session_state:
        st.session_state["manual_review_items"] = []

    with st.expander("Pengaturan Inferensi & Summarization (Opsional)", expanded=False):
        c1, c2, c3 = st.columns(3)
        default_sim = float(cfg.get("inference_min_similarity", 0.42))
        sim_threshold = c1.slider(
            "Ambang similarity minimum (Outlier)",
            min_value=0.10,
            max_value=0.80,
            value=default_sim,
            step=0.02,
            help="Karena artifact safetensors melakukan proyeksi berbasis cosine similarity ke centroid topik, dokumen dengan similarity di bawah ambang ini ditandai sebagai Outlier (-1).",
        )
        new_n_sentences = c2.slider("Jumlah kalimat ringkasan", 2, 7, int(cfg.get("summary_sentences", 5)))
        new_mmr_lambda = c3.slider("MMR lambda (relevansi vs variasi)", 0.1, 1.0, float(cfg.get("mmr_lambda", 0.72)), 0.02)

    input_tab_manual, input_tab_csv = st.tabs(["A. Input Manual", "B. Upload CSV"])
    input_df: pd.DataFrame | None = None
    input_ready_label = ""

    with input_tab_manual:
        st.markdown("Masukkan satu atau beberapa ulasan (satu baris per ulasan pada kotak teks, dan/atau tambahkan ulasan multi-kalimat satu per satu).")
        col_bulk, col_single = st.columns([1.3, 1.0])

        with col_bulk:
            if "manual_bulk_text" not in st.session_state:
                st.session_state["manual_bulk_text"] = ""
            bulk_text = st.text_area(
                "Daftar review (1 baris = 1 review)",
                height=190,
                placeholder="Contoh:\nAplikasi gagal login terus setelah update, OTP tidak masuk\nKode OTP belum masuk padahal jaringan normal\nLimit saya turun drastis padahal bayar selalu tepat waktu\nResep masakan rendang enak sekali",
                key="manual_bulk_text",
            )
            b_demo, b_clear = st.columns(2)
            if b_demo.button("Isi Contoh Demo", key="btn_fill_demo"):
                st.session_state["manual_bulk_text"] = (
                    "Aplikasi sering gagal login dan kode OTP tidak masuk.\n"
                    "Sudah coba login ulang tapi kode OTP tetap tidak diterima.\n"
                    "Limit saya tidak bisa dipakai padahal tagihan sudah lunas.\n"
                    "Limit kredivo tiba-tiba turun tanpa alasan jelas.\n"
                    "Bunganya terasa ringan dan proses pencairannya cepat.\n"
                    "bagus\n"
                    "Resep nasi goreng pedas manis enak sekali"
                )
                st.rerun()
            if b_clear.button("Bersihkan Input", key="btn_clear_manual"):
                st.session_state["manual_bulk_text"] = ""
                st.session_state["manual_review_items"] = []
                st.rerun()

        with col_single:
            single_text = st.text_area(
                "Tambah 1 review utuh (boleh multi-kalimat)",
                height=110,
                placeholder="Ketik satu review lengkap di sini lalu klik Tambahkan...",
                key="single_review_draft",
            )
            if st.button("Tambahkan ke Daftar Review", key="btn_add_single"):
                st.session_state["manual_review_items"].append(single_text)
                st.success(f"Review ditambahkan. Total dalam daftar terpisah: {len(st.session_state['manual_review_items'])}")

            if st.session_state["manual_review_items"]:
                st.caption(f"Daftar review terpisah ({len(st.session_state['manual_review_items'])} item):")
                st.dataframe(
                    pd.DataFrame(
                        [{"no": i + 1, "content": txt} for i, txt in enumerate(st.session_state["manual_review_items"])]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=130,
                )

        if st.button("Analyze Manual Reviews", type="primary", use_container_width=True, key="btn_analyze_manual"):
            parsed = parse_manual_reviews(bulk_text, st.session_state["manual_review_items"])
            if parsed.empty:
                st.warning("Masukkan minimal satu ulasan terlebih dahulu sebelum menekan Analyze.")
            else:
                with st.spinner("Menjalankan preprocessing, BERTopic inference, c-TF-IDF, dan 3 metode summarization..."):
                    result = analyze_new_reviews(
                        parsed,
                        model_bundle,
                        topics_df=topics,
                        keywords_df=keywords,
                        cfg=cfg,
                        min_similarity=sim_threshold,
                        summary_sentences=new_n_sentences,
                        mmr_lambda=new_mmr_lambda,
                    )
                st.session_state["new_review_analysis"] = result
                st.session_state["new_review_source_label"] = "Input Manual"

    with input_tab_csv:
        st.markdown(
            "Unggah file CSV dengan kolom wajib **`content`**. "
            "Kolom **`reviewId`** opsional (dibuatkan ID sesi otomatis jika tidak ada). "
            "Kolom tambahan (`score`, `at`, `sentiment`, `issue_category`) dipertahankan jika tersedia."
        )
        uploaded_file = st.file_uploader("Pilih file CSV ulasan", type=["csv"], key="new_reviews_csv_uploader")
        if uploaded_file is not None:
            try:
                prepared_df, upload_meta = validate_and_prepare_uploaded_csv(uploaded_file)
                st.success(
                    f"CSV valid: **{upload_meta['total_csv_rows']:,} baris** terbaca. "
                    + (
                        "ID sementara dibuat otomatis karena kolom `reviewId` tidak lengkap/tidak ada."
                        if upload_meta["generated_temp_ids"]
                        else "Kolom `reviewId` ditemukan dan digunakan."
                    )
                )
                if upload_meta["optional_columns_found"]:
                    st.caption(f"Kolom opsional terdeteksi: {', '.join(upload_meta['optional_columns_found'])}")
                st.dataframe(prepared_df.head(10), use_container_width=True, hide_index=True)

                if st.button("Analyze Uploaded CSV", type="primary", use_container_width=True, key="btn_analyze_csv"):
                    with st.spinner("Memproses seluruh CSV melalui BERTopic & 3 metode extractive summarization..."):
                        result = analyze_new_reviews(
                            prepared_df,
                            model_bundle,
                            topics_df=topics,
                            keywords_df=keywords,
                            cfg=cfg,
                            min_similarity=sim_threshold,
                            summary_sentences=new_n_sentences,
                            mmr_lambda=new_mmr_lambda,
                        )
                    st.session_state["new_review_analysis"] = result
                    st.session_state["new_review_source_label"] = f"Upload CSV ({uploaded_file.name})"
            except Exception as exc:
                st.error(f"Struktur atau isi CSV tidak sesuai: {exc}")

    # Render latest session analysis result
    analysis = st.session_state.get("new_review_analysis")
    if not analysis:
        st.info("Belum ada hasil analisis ulasan baru pada sesi ini. Masukkan ulasan atau unggah CSV lalu klik tombol **Analyze**.")
        return

    st.divider()
    st.subheader(f"Hasil Analisis NLP — {st.session_state.get('new_review_source_label', 'Review Baru')}")
    counts = analysis["counts"]
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Input", f"{counts['total_input']:,}")
    k2.metric("Review Valid", f"{counts['valid_reviews']:,}")
    k3.metric("Tidak Valid", f"{counts['invalid_reviews']:,}")
    k4.metric("Duplikat Digabung", f"{counts['duplicate_extra_rows']:,}")
    k5.metric("Topik Terdeteksi", f"{counts['detected_topics']:,}")
    k6.metric("Outlier (-1)", f"{counts['outlier_reviews']:,}")

    if counts["generic_praise_reviews"] > 0:
        st.caption(
            f"Sebanyak **{counts['generic_praise_reviews']} review** terdeteksi sebagai pujian generik berdiri sendiri (`topic_id = -2`) dan dikecualikan dari clustering topik sesuai metodologi pipeline utama."
        )

    invalid_df = analysis.get("invalid_reviews", pd.DataFrame())
    if not invalid_df.empty:
        with st.expander(f"Review Tidak Valid / Kosong ({len(invalid_df)} baris)", expanded=False):
            st.dataframe(invalid_df, use_container_width=True, hide_index=True)

    if analysis["status"] == "no_valid_reviews":
        st.warning("Seluruh input tidak valid atau kosong setelah preprocessing, sehingga tidak dapat diproses lebih lanjut.")
        return

    overview_df = analysis["topic_overview"]
    if not overview_df.empty:
        c_tbl, c_chart = st.columns([1.2, 1.0])
        with c_tbl:
            st.markdown("**Ringkasan Distribusi Topik pada Input Baru**")
            st.dataframe(overview_df, use_container_width=True, hide_index=True)
        with c_chart:
            fig_new = px.bar(
                overview_df.sort_values("review_count"),
                x="review_count",
                y="topic_label",
                orientation="h",
                text="review_count",
                title="Jumlah Review per Topik (Input Baru)",
            )
            fig_new.update_layout(yaxis_title="Topik", xaxis_title="Jumlah Review", height=320)
            st.plotly_chart(fig_new, use_container_width=True)

    # Per-topic extractive summaries & source traceability
    st.subheader("Ringkasan Ekstraktif & Source Traceability per Topik")
    reviews_assigned = analysis["reviews_assigned"]
    summaries_df = analysis["summaries"]
    sources_df = analysis["summary_sources"]
    topic_status = analysis["topic_summary_status"]

    positive_topics = overview_df[overview_df["topic_id"] >= 0] if not overview_df.empty else pd.DataFrame()
    if positive_topics.empty:
        st.info("Tidak ada review yang masuk ke topik reguler (`topic_id >= 0`). Periksa bagian Outlier atau Pujian Generik di bawah.")
    else:
        for trow in positive_topics.itertuples(index=False):
            tid = int(trow.topic_id)
            tstat = topic_status.get(tid, {})
            with st.expander(
                f"Topik {tid} — {trow.topic_label} ({int(trow.review_count)} review / {int(trow.unique_documents)} dokumen unik)",
                expanded=True,
            ):
                if trow.keywords:
                    st.markdown(f"**Kata kunci c-TF-IDF (model BERTopic):** `{trow.keywords}`")

                topic_reviews = reviews_assigned[reviews_assigned["topic_id"] == tid]
                if not tstat.get("can_summarize", False):
                    st.info(tstat.get("reason", "Jumlah dokumen belum mencukupi untuk multi-document extractive summarization."))
                    st.markdown("**Review yang terpetakan ke topik ini:**")
                    st.dataframe(
                        topic_reviews[[c for c in ["reviewId", "content", "topic_similarity"] if c in topic_reviews]],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    method_tabs = st.tabs(MODEL_NAMES)
                    topic_metrics = analysis["metrics_by_topic"].get(tid, {})
                    for tab, m_name in zip(method_tabs, MODEL_NAMES):
                        with tab:
                            srow = summaries_df[(summaries_df["topic_id"] == tid) & (summaries_df["model"] == m_name)]
                            if not srow.empty:
                                st.markdown(f'<div class="summary-card">{srow.iloc[0]["summary"]}</div>', unsafe_allow_html=True)
                            met = topic_metrics.get(m_name, {})
                            if met:
                                mc1, mc2, mc3, mc4 = st.columns(4)
                                mc1.metric("Keyword Coverage", f"{met.get('keyword_coverage', 0):.2%}")
                                mc2.metric("Redundancy", f"{met.get('redundancy', 0):.3f}")
                                mc3.metric("Source Traceability", f"{met.get('source_traceability_rate', 0):.0%}")
                                mc4.metric("Waktu Proses", f"{met.get('processing_seconds', 0):.4f}s")

                            tsrc = sources_df[(sources_df["topic_id"] == tid) & (sources_df["model"] == m_name)].sort_values("rank")
                            if not tsrc.empty:
                                _render_source_evidence_list(tsrc, topic_reviews, key_prefix=f"new_{tid}_{m_name}")

    # Outliers and generic praise breakdown
    excluded_reviews = reviews_assigned[reviews_assigned["topic_id"] < 0]
    if not excluded_reviews.empty:
        with st.expander(f"Review Tidak Masuk Topik: Outlier (-1) & Pujian Generik (-2) ({len(excluded_reviews)} review)", expanded=True):
            st.caption(
                "Review Outlier (-1) tidak cocok dengan topik mana pun di atas ambang similarity, sedangkan Pujian Generik (-2) dikecualikan karena tidak membawa informasi topikal."
            )
            st.dataframe(
                excluded_reviews[[c for c in ["reviewId", "content", "topic_id", "topic_label", "topic_similarity", "topic_status"] if c in excluded_reviews]],
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Tabel Lengkap Hasil Pemrosesan per reviewId", expanded=False):
        display_cols = [
            c
            for c in [
                "reviewId", "content", "topic_id", "topic_label", "topic_similarity",
                "topic_status", "topic_keywords", "score", "sentiment", "issue_category", "at",
            ]
            if c in reviews_assigned.columns
        ]
        st.dataframe(reviews_assigned[display_cols], use_container_width=True, hide_index=True)


def render_compare_models(
    topics: pd.DataFrame,
    keywords: pd.DataFrame,
    reviews: pd.DataFrame,
    cfg: dict,
) -> None:
    st.header("Compare Summarization Models")
    st.caption(
        "Perbandingan langsung antara **TF-IDF Sentence Ranking**, **TextRank**, dan **TextRank + MMR** pada kumpulan dokumen dan parameter yang sama persis."
    )

    new_analysis = st.session_state.get("new_review_analysis")
    has_new_summarizable = False
    if new_analysis and new_analysis.get("topic_summary_status"):
        has_new_summarizable = any(v.get("can_summarize", False) for v in new_analysis["topic_summary_status"].values())

    source_choices = ["Dataset Utama (Pipeline Kredivo)"]
    if has_new_summarizable:
        source_choices.append("Hasil Analisis Review Baru (Session)")

    source_mode = st.radio("Sumber dokumen yang dibandingkan", source_choices, horizontal=True, key="cmp_source_mode")

    selected_docs = pd.DataFrame()
    selected_reviews_lookup = pd.DataFrame()
    selected_kw_list: list[str] = []
    selected_topic_id = 0

    if source_mode == "Dataset Utama (Pipeline Kredivo)":
        assignments = read_csv("document_topic_assignments.csv")
        regular_topics = topics[topics["topic_id"].astype(int) >= 0].sort_values("total_source_reviews", ascending=False)
        if regular_topics.empty or assignments.empty:
            st.warning("Data dokumen topik belum tersedia. Jalankan pipeline terlebih dahulu.")
            return
        label_map = {
            int(r.topic_id): f"Topik {int(r.topic_id)} — {r.topic_label} ({int(r.total_source_reviews):,} review)"
            for r in regular_topics.itertuples(index=False)
        }
        selected_topic_id = st.selectbox("Pilih topik", list(label_map), format_func=lambda x: label_map[x], key="cmp_main_topic")
        selected_docs = assignments[assignments["topic_id"].astype(int) == int(selected_topic_id)].copy()
        selected_reviews_lookup = reviews[reviews["topic_id"].astype(int) == int(selected_topic_id)].copy() if not reviews.empty else pd.DataFrame()
        kw_row = keywords[keywords["topic_id"].astype(int) == int(selected_topic_id)]
        if not kw_row.empty:
            selected_kw_list = [k.strip() for k in str(kw_row.iloc[0]["keywords"]).split("|") if k.strip()]
    else:
        overview = new_analysis["topic_overview"]
        summarizable_ids = [tid for tid, s in new_analysis["topic_summary_status"].items() if s.get("can_summarize")]
        sub_topics = overview[overview["topic_id"].isin(summarizable_ids)]
        label_map = {
            int(r.topic_id): f"Topik {int(r.topic_id)} — {r.topic_label} ({int(r.review_count)} review baru)"
            for r in sub_topics.itertuples(index=False)
        }
        selected_topic_id = st.selectbox("Pilih topik dari review baru", list(label_map), format_func=lambda x: label_map[x], key="cmp_new_topic")
        u_docs = new_analysis["unique_documents"]
        selected_docs = u_docs[u_docs["topic_id"].astype(int) == int(selected_topic_id)].copy()
        r_new = new_analysis["reviews_assigned"]
        selected_reviews_lookup = r_new[r_new["topic_id"].astype(int) == int(selected_topic_id)].copy()
        kw_str = str(sub_topics[sub_topics["topic_id"] == selected_topic_id].iloc[0]["keywords"])
        selected_kw_list = [k.strip() for k in kw_str.split("|") if k.strip()]

    c1, c2, c3 = st.columns([1.6, 1.0, 1.0])
    chosen_models = c1.multiselect("Metode yang dibandingkan", MODEL_NAMES, default=MODEL_NAMES, key="cmp_models")
    n_sent = c2.slider("Jumlah kalimat ringkasan", 1, 8, int(cfg.get("summary_sentences", 5)), key="cmp_nsent")
    mmr_lam = c3.slider("MMR lambda (λ)", 0.10, 1.00, float(cfg.get("mmr_lambda", 0.72)), 0.02, key="cmp_lambda")

    if not chosen_models:
        st.warning("Pilih minimal satu metode untuk ditampilkan.")
        return

    with st.spinner("Menjalankan perbandingan ketiga metode pada dokumen topik yang sama..."):
        cmp_result = compare_summarizers_on_docs(
            selected_docs,
            keywords=selected_kw_list,
            n_sentences=n_sent,
            mmr_lambda=mmr_lam,
            textrank_neighbors=int(cfg.get("textrank_neighbors", 15)),
            topic_id=selected_topic_id,
        )

    if not cmp_result["can_compare"]:
        st.warning(cmp_result["reason"])
        return

    st.caption(
        f"Perbandingan adil dijalankan pada **{cmp_result['unique_document_count']:,} dokumen unik** "
        f"(**{cmp_result['candidate_sentence_count']:,} kalimat kandidat**) dengan budget panjang yang identik ({n_sent} kalimat)."
    )

    comp_table = cmp_result["comparison_table"]
    comp_filtered = comp_table[comp_table["model"].isin(chosen_models)].copy()

    st.subheader("1. Perbandingan Teks Ringkasan")
    cols = st.columns(len(chosen_models))
    summaries_df = cmp_result["summaries"]
    sources_df = cmp_result["sources"]

    for col, model_name in zip(cols, chosen_models):
        with col:
            st.markdown(f"**{model_name}**")
            srow = summaries_df[summaries_df["model"] == model_name]
            m_info = cmp_result["metrics"].get(model_name, {})
            if not srow.empty:
                st.markdown(f'<div class="summary-card">{srow.iloc[0]["summary"]}</div>', unsafe_allow_html=True)
            st.caption(
                f"Coverage: `{m_info.get('keyword_coverage', 0):.2%}` · "
                f"Redundancy: `{m_info.get('redundancy', 0):.3f}` · "
                f"Traceability: `{m_info.get('source_traceability_rate', 0):.0%}` · "
                f"Waktu: `{m_info.get('processing_seconds', 0):.4f}s`"
            )

    st.subheader("2. Metrik Otomatis pada Topik Terpilih")
    st.dataframe(comp_filtered, use_container_width=True, hide_index=True)

    if len(comp_filtered) >= 2:
        ch1, ch2 = st.columns(2)
        with ch1:
            fig_cov = px.bar(
                comp_filtered,
                x="model",
                y=["keyword_coverage", "redundancy"],
                barmode="group",
                title="Keyword Coverage vs Redundancy (Topik Terpilih)",
            )
            fig_cov.update_layout(yaxis_title="Skor (0–1)", xaxis_title="Metode", height=330)
            st.plotly_chart(fig_cov, use_container_width=True)
        with ch2:
            fig_time = px.bar(
                comp_filtered,
                x="model",
                y="processing_seconds",
                title="Waktu Pemrosesan per Metode (detik)",
                text="processing_seconds",
            )
            fig_time.update_traces(texttemplate="%{text:.4f}s", textposition="outside")
            fig_time.update_layout(yaxis_title="Detik", xaxis_title="Metode", height=330)
            st.plotly_chart(fig_time, use_container_width=True)

    st.subheader("3. Perbedaan Kalimat Terpilih & Source Traceability")
    overlap_df = cmp_result["sentence_overlap"]
    if not overlap_df.empty:
        keep_cols = ["sentence", "source_review_count", "source_occurrences"] + [c for c in chosen_models if c in overlap_df]
        st.dataframe(overlap_df[keep_cols], use_container_width=True, hide_index=True)

    detail_tabs = st.tabs(chosen_models)
    for tab, model_name in zip(detail_tabs, chosen_models):
        with tab:
            msrc = sources_df[sources_df["model"] == model_name].sort_values("rank")
            _render_source_evidence_list(msrc, selected_reviews_lookup, key_prefix=f"cmp_{selected_topic_id}_{model_name}")

    st.divider()
    st.subheader("4. Evaluasi Ilmiah Eksperimen (ROUGE, Ablation, Human Eval & Error Analysis)")
    eval_summary = get_evaluation_status_summary(OUTPUT_DIR)

    if not eval_summary["rouge_available"]:
        st.warning(
            "**Evaluasi ROUGE-1, ROUGE-2, dan ROUGE-L belum dapat dilakukan.** "
            f"Saat ini terdapat **{eval_summary['filled_reference_count']}** dari **{eval_summary['total_reference_templates']}** "
            "topik yang memiliki `reference_summary` buatan manusia pada `outputs/reference_summaries_template.csv`."
        )
        st.markdown(
            "Dataset Kredivo berisi ulasan tanpa ringkasan referensi emas (*gold reference*), sehingga proyek **tidak menampilkan angka ROUGE dummy** atau menggunakan output model sebagai referensi. "
            "Untuk menghitung ROUGE secara sah: baca contoh review pada `outputs/reference_examples.csv`, isi `reference_summary` secara independen pada `outputs/reference_summaries_template.csv`, lalu jalankan ulang `python scripts/run_pipeline.py`."
        )
    else:
        comp_eval = eval_summary["comparison_df"]
        st.dataframe(comp_eval, use_container_width=True, hide_index=True)
        rouge_plot = comp_eval.melt(
            id_vars="model", value_vars=["ROUGE_1", "ROUGE_2", "ROUGE_L"], var_name="Metrik", value_name="F1"
        )
        fig_r = px.bar(rouge_plot, x="model", y="F1", color="Metrik", barmode="group", title="Perbandingan ROUGE F1 (Test Split)")
        st.plotly_chart(fig_r, use_container_width=True)

        if not eval_summary["ablation_df"].empty:
            st.markdown("**Ablation Panjang Ringkasan (1, 2, 3, dan konfigurasi utama kalimat)**")
            fig_ab = px.line(
                eval_summary["ablation_df"],
                x="summary_sentences",
                y="ROUGE_L",
                color="model",
                markers=True,
                title="Pengaruh Jumlah Kalimat terhadap ROUGE-L",
            )
            st.plotly_chart(fig_ab, use_container_width=True)
            st.dataframe(eval_summary["ablation_df"], use_container_width=True, hide_index=True)

        if not eval_summary["errors_df"].empty:
            st.markdown("**Kasus Terbaik & Terburuk (Error Analysis)**")
            err_cols = [
                c
                for c in ["topic_id", "model", "case_type", "ROUGE_L", "prediction", "reference", "error_reason", "reviewer_notes"]
                if c in eval_summary["errors_df"]
            ]
            st.dataframe(eval_summary["errors_df"][err_cols], use_container_width=True, hide_index=True)

    with st.expander("Definisi Singkat Metrik Evaluasi (Untuk Presentasi)", expanded=False):
        st.markdown(
            "- **ROUGE-1 / ROUGE-2 / ROUGE-L:** Mengukur overlap unigram, bigram, dan *longest common subsequence* antara ringkasan sistem dan *reference summary* manusia.\n"
            "- **Keyword Coverage:** Proporsi kata kunci c-TF-IDF topik yang tercakup dalam kalimat ringkasan terpilih.\n"
            "- **Redundancy:** Rata-rata *cosine similarity* antarkalimat terpilih (nilai lebih rendah menandakan kalimat lebih beragam/tidak berulang).\n"
            "- **Source Traceability Rate:** Proporsi kalimat terpilih yang memiliki minimal satu `reviewId` sumber asli.\n"
            "- **Groundedness / Faithfulness:** Dinilai manusia (skala 1–5) pada `outputs/human_evaluation_template.csv` untuk memastikan kalimat sesuai konteks review asli.\n"
            "- **Processing Time:** Waktu komputasi ekstraksi kalimat per metode pada perangkat lokal."
        )


def _reset_review_explorer_filters() -> None:
    for key in [
        "re_search",
        "re_topics",
        "re_keyword",
        "re_scores",
        "re_sentiments",
        "re_issues",
        "re_severities",
    ]:
        if key in st.session_state:
            if key in {"re_search", "re_keyword"}:
                st.session_state[key] = ""
            else:
                st.session_state[key] = []


def render_review_explorer_and_audit(
    reviews: pd.DataFrame,
    topics: pd.DataFrame,
    quality: dict,
    audit: dict,
) -> None:
    st.header("Review Explorer & Data Audit")

    tab_search, tab_audit = st.tabs(["Pencarian & Filter Review", "Data Quality & Audit Label"])

    with tab_search:
        if reviews.empty:
            st.warning("Data review hasil pemetaan (`outputs/reviews_assigned.csv`) belum tersedia.")
        else:
            st.info(
                "**Transparansi Sumber Label:** `topic_id` dan `topic_keywords` adalah **hasil prediksi model BERTopic + c-TF-IDF**. "
                "Kolom `score`, `sentiment`, `issue_category`, dan `severity_level` adalah **atribut bawaan CSV** yang tidak dijadikan ground truth penelitian."
            )
            opts = get_available_filter_options(reviews, topics, audit)

            col_s1, col_s2, col_btn = st.columns([2.0, 1.5, 0.7])
            search_query = col_s1.text_input(
                "Cari teks review atau reviewId",
                placeholder="Contoh: otp, limit, gagal login, atau potongan reviewId...",
                key="re_search",
            )
            keyword_filter = col_s2.text_input(
                "Filter kata kunci topik / teks",
                placeholder="Contoh: bunga, cair, spam...",
                key="re_keyword",
            )
            col_btn.markdown("<div style='height: 1.7rem'></div>", unsafe_allow_html=True)
            col_btn.button("Reset Filter", on_click=_reset_review_explorer_filters, use_container_width=True, key="btn_reset_re")

            f1, f2, f3, f4, f5 = st.columns(5)
            topic_map = {tid: lbl for tid, lbl in opts["topics"]}
            selected_topics = f1.multiselect(
                "Topik (Prediksi Model)",
                list(topic_map.keys()),
                format_func=lambda tid: topic_map[tid],
                key="re_topics",
            )
            selected_scores = (
                f2.multiselect("Rating Score (CSV)", opts["scores"], key="re_scores")
                if opts["has_score"]
                else []
            )
            selected_sentiments = (
                f3.multiselect("Sentiment (CSV - Unverified)", opts["sentiments"], key="re_sentiments")
                if opts["has_sentiment"]
                else []
            )
            selected_issues = (
                f4.multiselect("Issue Category (CSV - Unverified)", opts["issue_categories"], key="re_issues")
                if opts["has_issue_category"]
                else []
            )
            if opts["has_severity"]:
                if opts["severity_audit_passed"]:
                    selected_severities = f5.multiselect("Severity Level (Tervalidasi)", opts["severities"], key="re_severities")
                else:
                    f5.multiselect(
                        "Severity Level (Belum Tervalidasi)",
                        ["Terkunci — Audit belum lulus"],
                        disabled=True,
                        key="re_severities_locked",
                        help="Filter severity baru aktif setelah 200 sampel pada outputs/label_audit_sample.csv selesai diaudit dan memenuhi ambang agreement.",
                    )
                    selected_severities = []
            else:
                selected_severities = []

            filtered_df = filter_reviews(
                reviews,
                search_query=search_query,
                topic_ids=selected_topics,
                keyword_filter=keyword_filter,
                sentiments=selected_sentiments,
                issue_categories=selected_issues,
                severities=selected_severities,
                score_values=selected_scores,
                severity_audit_passed=opts["severity_audit_passed"],
            )

            st.metric("Jumlah Review Sesuai Filter", f"{len(filtered_df):,} dari {len(reviews):,} review")

            if filtered_df.empty:
                st.warning("Tidak ada review yang cocok dengan kombinasi filter saat ini. Klik **Reset Filter** untuk mengembalikan semua data.")
            else:
                max_display = st.selectbox("Batas baris pada tabel", [50, 100, 250, 500], index=1, key="re_table_limit")
                preview_df = filtered_df.head(int(max_display)).copy()
                cols_to_show = [
                    c
                    for c in [
                        "reviewId", "content", "topic_id", "topic_keywords", "topic_split",
                        "score", "sentiment", "issue_category", "severity_level", "at",
                    ]
                    if c in preview_df.columns
                ]
                st.dataframe(preview_df[cols_to_show], use_container_width=True, hide_index=True)

                st.subheader("Detail Review Terpilih")
                review_choices = preview_df["reviewId"].astype(str).tolist()
                chosen_rid = st.selectbox(
                    "Pilih reviewId dari hasil filter untuk melihat detail lengkap",
                    review_choices,
                    key="re_detail_select",
                )
                detail_row = preview_df[preview_df["reviewId"].astype(str) == chosen_rid].iloc[0]
                d1, d2 = st.columns([1.4, 1.0])
                with d1:
                    st.markdown(f"**reviewId:** `{detail_row['reviewId']}`")
                    st.markdown(f"**doc_id (kelompok unik):** `{detail_row.get('doc_id', '-')}`")
                    st.markdown("**Teks Review Asli:**")
                    st.info(detail_row["content"])
                with d2:
                    st.markdown("**Hasil Model NLP:**")
                    tid_val = int(detail_row.get("topic_id", -2))
                    st.write(f"- Topic ID: `{tid_val}` ({topic_map.get(tid_val, '-')})")
                    st.write(f"- Split: `{detail_row.get('topic_split', '-')}`")
                    st.write(f"- Keyword Topik (c-TF-IDF): `{detail_row.get('topic_keywords', '-') or '-'}`")
                    st.markdown("**Atribut Bawaan CSV (bukan ground truth):**")
                    st.write(f"- Score: `{detail_row.get('score', '-')}` · Waktu: `{detail_row.get('at', '-')}`")
                    st.write(f"- Sentiment (CSV): `{detail_row.get('sentiment', '-')}`")
                    st.write(f"- Issue Category (CSV): `{detail_row.get('issue_category', '-')}`")
                    sev_badge = "Tervalidasi" if opts["severity_audit_passed"] else "Belum tervalidasi"
                    st.write(f"- Severity (CSV — {sev_badge}): `{detail_row.get('severity_level', '-')}`")

    with tab_audit:
        st.subheader("Data Quality Audit")
        if quality:
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Baris CSV Awal", f"{quality.get('raw_csv_rows', 0):,}")
            q2.metric("Duplikasi Tambahan", f"{quality.get('duplicate_extra_rows', 0):,}")
            q3.metric("Unique Review", f"{quality.get('unique_normalized_reviews', 0):,}")
            q4.metric("Pujian Generik Unik", f"{quality.get('generic_praise_unique_docs', 0):,}")

            if "sentiment_distribution" in quality:
                st.markdown("**Distribusi sentimen pada CSV (belum diverifikasi)**")
                st.dataframe(
                    pd.DataFrame(quality["sentiment_distribution"].items(), columns=["sentiment", "jumlah"]),
                    hide_index=True,
                )
            if (OUTPUT_DIR / "data_quality_report.md").exists():
                st.download_button(
                    "Unduh laporan audit data (.md)",
                    (OUTPUT_DIR / "data_quality_report.md").read_bytes(),
                    file_name="data_quality_report.md",
                )

        st.divider()
        st.subheader("Pemeriksaan Kualitas Cluster BERTopic (Validation Split)")
        topic_audit_path = OUTPUT_DIR / "topic_quality_review_template.csv"
        if topic_audit_path.exists():
            st.caption(
                "Template ini memakai sampel dokumen validation, bukan test. Isi skor koherensi topik dan relevansi kata kunci setelah membaca contoh review."
            )
            st.download_button(
                "Unduh template audit kualitas topik",
                topic_audit_path.read_bytes(),
                file_name="topic_quality_review_template.csv",
            )

        st.divider()
        st.subheader("Status Validasi Label `issue_category` dan `severity_level`")
        if audit.get("passed"):
            st.success(
                f"Audit label lulus. Issue category: {audit.get('agreement_issue_category', 0):.1%} · Severity: {audit.get('agreement_severity_level', 0):.1%}"
            )
        else:
            st.warning(audit.get("reason", "Label belum divalidasi oleh manusia."))
        if audit:
            st.json(audit)
        st.markdown(
            "Unduh `outputs/label_audit_sample.csv`, isi `manual_issue_category` dan `manual_severity_level`, "
            "pastikan setiap baris `reviewed=true`, simpan kembali, lalu jalankan ulang pipeline. "
            "Severity filter baru aktif saat agreement keseluruhan dan per kelas memenuhi ambang serta setiap kelas memiliki sampel minimum."
        )
        if (OUTPUT_DIR / "label_audit_sample.csv").exists():
            st.download_button(
                "Unduh sampel audit label",
                (OUTPUT_DIR / "label_audit_sample.csv").read_bytes(),
                file_name="label_audit_sample.csv",
            )
        st.caption(
            "Label tambahan pada dataset tidak digunakan sebagai ground truth untuk clustering atau summarization. "
            "Nama pengguna dan gambar profil tidak dipakai dalam pipeline/dashboard."
        )


# ---------------------------------------------------------------------------
# App entry layout & navigation
# ---------------------------------------------------------------------------
st.title("Topic-Based, Evidence-Grounded Multi-Document Extractive Summarization")
st.caption("Analisis ulasan Kredivo dengan BERTopic, c-TF-IDF, TF-IDF Sentence Ranking, TextRank, dan TextRank + MMR")

cfg = load_config()
status = read_json("run_status.json")
topics = read_csv("topics.csv")
keywords = read_csv("topic_keywords_ctfidf.csv")
sources = read_csv("summary_sources.csv")
summaries = read_csv("summaries.csv")
reviews = read_csv("reviews_assigned.csv")
quality = read_json("data_quality_report.json")
audit = read_json(
    "label_audit_status.json",
    {"passed": False, "status": "not_audited", "reason": "Audit belum dijalankan."},
)

PAGES = [
    "Overview",
    "Topic Explorer",
    "Analyze New Reviews",
    "Compare Summarization Models",
    "Review Explorer & Data Audit",
]

with st.sidebar:
    st.header("Navigasi")
    page = st.radio("Pilih Halaman", PAGES, key="nav_page")
    st.divider()
    st.header("Kontrol Pipeline")
    st.caption("Jalankan pipeline utama untuk memperbarui model, topik, dan ringkasan dari `data/kredivo_reviews_50k.csv`.")
    if st.button("Jalankan Pipeline NLP", type="primary", use_container_width=True):
        try:
            with st.spinner("Menjalankan audit data, BERTopic, summarization, dan evaluasi..."):
                run_res = run_pipeline()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Pipeline selesai.")
            st.json(run_res)
            st.rerun()
        except Exception as exc:
            st.error(f"Pipeline gagal: {exc}")
            st.exception(exc)

    st.divider()
    model_ready = (OUTPUT_DIR / "bertopic_model").exists()
    st.caption(f"Artifact BERTopic: {'Tersedia' if model_ready else 'Belum tersedia'}")
    st.caption(f"Audit Severity: {'Lulus' if audit.get('passed') else 'Belum lulus'}")
    eval_info = get_evaluation_status_summary(OUTPUT_DIR)
    st.caption(f"Evaluasi ROUGE: {'Tersedia' if eval_info['rouge_available'] else 'Menunggu reference summary'}")

if (topics.empty or summaries.empty) and page != "Analyze New Reviews":
    st.info(
        "Belum ada hasil pipeline. Pilih tombol **Jalankan Pipeline NLP** di panel kiri atau jalankan `python scripts/run_pipeline.py` dari folder proyek."
    )
    st.stop()

if page == "Overview":
    render_overview(status, quality, topics, summaries)
elif page == "Topic Explorer":
    render_topic_explorer(topics, keywords, summaries, sources, reviews, audit)
elif page == "Analyze New Reviews":
    render_analyze_new_reviews(topics, keywords, cfg)
elif page == "Compare Summarization Models":
    render_compare_models(topics, keywords, reviews, cfg)
elif page == "Review Explorer & Data Audit":
    render_review_explorer_and_audit(reviews, topics, quality, audit)
