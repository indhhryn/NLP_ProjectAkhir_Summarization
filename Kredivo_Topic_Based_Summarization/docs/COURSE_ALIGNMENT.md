# Kesesuaian dengan Materi dan Instruksi NLP

Dokumen ini memetakan komponen implementasi ke istilah dan tujuan pada RPS serta Modul Praktikum Text Summarization yang digunakan di kelas. Pemetaan ini bukan pengganti instruksi terbaru dosen atau rubrik khusus yang mungkin dibagikan terpisah.

| Materi / instruksi | Penerapan dalam proyek | Bukti keluaran |
|---|---|---|
| Formulasi masalah, relevansi corpus, preprocessing | Multi-document summarization untuk kumpulan review pendek; audit kualitas, normalisasi slang secara konservatif, deduplikasi sadar frekuensi, dan pemisahan pujian generik | `data_quality_report.*`, `unique_documents.csv` |
| Representasi teks: TF-IDF dan cosine similarity | TF-IDF sentence ranking; cosine similarity untuk TextRank dan redundancy | `summary_sources.csv`, `summaries.csv` |
| Validasi kualitas topic clustering | Template manual untuk memeriksa koherensi cluster dan relevansi kata kunci menggunakan validation reviews, tanpa memakai test untuk tuning | `topic_quality_review_template.csv` |
| Sentence/document embeddings dan clustering | Multilingual sentence embeddings sebagai masukan BERTopic; UMAP dan HDBSCAN; c-TF-IDF untuk representasi topik/kata kunci | `topics.csv`, `topic_keywords_ctfidf.csv`, `document_topic_assignments.csv` |
| Baseline 1 pada Modul Praktikum | TF-IDF Sentence Ranking: jumlah bobot TF-IDF per kalimat | `summary_sources.csv` dan kolom model TF-IDF |
| Baseline 2 pada Modul Praktikum | TextRank: graph cosine similarity dan PageRank | `summary_sources.csv` dan kolom model TextRank |
| Pengembangan/eksperimen tambahan | TextRank + MMR untuk mengurangi redundansi; parameter `mmr_lambda` | keluaran model TextRank + MMR, metrik redundancy |
| Evaluasi ROUGE-1, ROUGE-2, ROUGE-L | F1-score dibandingkan ringkasan acuan manusia yang independen | `evaluation_comparison.csv`, `evaluation_predictions.csv` |
| Ubah `n_sentences` 1 menjadi 2 | Ablation panjang ringkasan 1, 2, 3, dan konfigurasi utama | `summary_length_ablation_comparison.csv` |
| Analisis prediksi terbaik/terburuk dan error analysis | Tiga kasus ROUGE-L tertinggi dan terendah per metode disiapkan untuk analisis manusia | `error_analysis_cases.csv` |
| Evaluasi makna di luar ROUGE | Form penilaian manusia untuk groundedness, relevance, coherence, non-redundancy; ROUGE tidak disamakan dengan penilaian semantik | `human_evaluation_template.csv` |
| Train-validation-test dan reproducibility | Split deterministik 80/10/10 pada dokumen unik informatif; BERTopic fit pada train, validation untuk inspeksi/penyesuaian, test untuk evaluasi akhir; seed/config/script dicatat | `document_topic_assignments.csv`, `config.json`, `requirements.txt`, `scripts/`, README |
| Responsible NLP | Menghindari nama/foto pengguna, memeriksa label sebelum severity filter, mencatat keterbatasan dan bias, serta tidak mengarang referensi/hasil | `label_audit_sample.csv`, `label_audit_status.json`, laporan kualitas |

## Batas implementasi

1. **Reference summary tidak tersedia di CSV sumber.** Proyek membuat template kosong dan contoh review untuk anotator. ROUGE baru dihitung setelah ringkasan acuan diisi manusia. Ringkasan model tidak boleh dijadikan ground truth.
2. **Split data 80/10/10.** BERTopic fit pada train; gunakan validation untuk inspeksi cluster/parameter dan jangan gunakan test untuk memilih parameter. Test dipakai untuk evaluasi akhir.
3. **Evaluasi memakai subset test yang sama.** `evaluation_doc_ids` mencatat dokumen yang ditunjukkan ke anotator; pipeline mengevaluasi prediksi pada dokumen test yang sama, bukan pada seluruh topik atau data train.
4. **Faithfulness tidak direduksi menjadi satu angka palsu.** Source traceability hanya memeriksa ketersediaan reviewId sumber; kebenaran konteks dan kecukupan cakupan perlu penilaian groundedness oleh manusia.
5. **Abstractive Seq2Seq/Transformer tidak diimplementasikan sebagai model perbandingan.** Modul menyebutnya sebagai usulan pengembangan. Proyek ini mempertahankan fokus pada dua baseline extractive wajib dan menambahkan MMR sebagai eksperimen.
6. **Tidak ada klaim bahwa metode atau proyek pasti menang.** Kesimpulan harus dibuat dari eksperimen yang benar-benar dijalankan, contoh keluaran, keterbatasan dan analisis kesalahan.

## Panduan menjawab poin analisis pada modul praktikum

1. **Metode dengan ROUGE-L tertinggi:** ambil dari hasil eksperimen `evaluation_comparison.csv`; jangan menyalin angka contoh yang ada di modul karena contoh itu berasal dari dataset sintetis dan eksekusi praktikum yang berbeda.
2. **Mengapa pilihan kalimat berbeda:** TF-IDF memberi skor berdasarkan jumlah bobot kata; TextRank memilih kalimat yang sentral dalam graph similarity; MMR menambahkan penalti kemiripan terhadap kalimat yang telah dipilih.
3. **Apakah ROUGE tinggi selalu berarti makna lebih baik:** tidak. ROUGE mengukur overlap kata/urutan. Gunakan skor groundedness, relevance, coherence dan contoh keluaran untuk menilai kualitas makna.
4. **Minimal tiga kesalahan:** pilih sekurangnya tiga baris dengan ROUGE-L rendah dari `error_analysis_cases.csv`; jelaskan apakah penyebabnya kalimat tidak mewakili topik, lexical mismatch, referensi terlalu abstraktif, kalimat terlalu panjang, atau informasi terpecah.
5. **Keterbatasan dataset:** dataset contoh di modul bersifat sintetis dan relatif kecil. Dataset Kredivo lebih realistis, tetapi ulasannya sering pendek, banyak duplikasi dan pujian generik, variasi bahasa informal, serta label tambahan belum diketahui provenance/validitasnya. Keterbatasan ini perlu dibahas berdasarkan audit aktual.
6. **Satu kalimat dibanding dua kalimat:** bandingkan baris `summary_sentences=1` dan `summary_sentences=2` di `summary_length_ablation_comparison.csv`, lalu lihat trade-off keyword coverage, redundancy dan ROUGE.
7. **Usulan abstractive sebagai pekerjaan lanjutan:** kembangkan pembanding Seq2Seq/Transformer encoder-decoder yang menghasilkan kalimat baru, gunakan fine-tuning pada pasangan review-cluster dan ringkasan acuan manusia, lalu evaluasi ROUGE sekaligus groundedness/hallucination secara manual. Ini adalah usulan, bukan komponen yang sudah diimplementasikan pada paket ini.
