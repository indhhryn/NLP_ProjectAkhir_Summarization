# Kredivo Topic Summarizer

**Topic-Based, Evidence-Grounded Multi-Document Extractive Summarization**

Paket proyek NLP Bahasa Indonesia untuk mengelompokkan ulasan aplikasi Kredivo berdasarkan topik, mengekstrak kata kunci, menghasilkan ringkasan extractive dari kumpulan review, membandingkan tiga metode, dan menelusuri setiap kalimat ringkasan ke review sumbernya.

## 1. Fitur dan alur NLP

```text
CSV review Kredivo
   │
   ├── Data quality audit + normalisasi teks
   ├── Duplicate-aware processing (teks identik digabung, ID dan jumlah tetap disimpan)
   ├── Pisahkan data informatif dari pujian generik yang berdiri sendiri
   │
   ├── Split dokumen unik: train 80% / validation 10% / test 10% (seed tetap)
   │
   ├── BERTopic (multilingual sentence embedding → UMAP → HDBSCAN)
   │       └── c-TF-IDF → kata kunci dan label topik
   │
   ├── Multi-document extractive summarization per topik
   │       ├── TF-IDF Sentence Ranking (baseline 1)
   │       ├── TextRank (baseline 2)
   │       └── TextRank + MMR (varian pengurang redundansi)
   │
   ├── Evidence traceability: kalimat → reviewId → teks review asli
   ├── Human reference templates + ROUGE-1/2/L + ablation panjang ringkasan
   └── Streamlit dashboard
```

**Pembagian data:** dokumen unik informatif dipisah secara deterministik menjadi 80% train, 10% validation, dan 10% test. BERTopic di-fit hanya pada train; dokumen validation/test serta dokumen train yang tidak masuk sampel fitting ditetapkan dengan `transform()`. Validation dapat digunakan untuk inspeksi kualitas cluster dan penyesuaian parameter; test disimpan untuk evaluasi akhir. Algoritma extractive tidak dilatih secara supervised. Ringkasan acuan manusia hanya digunakan ketika evaluasi, tidak pernah menjadi input ketiga metode.

### Metode ringkasan

- **TF-IDF Sentence Ranking:** setiap kalimat review menjadi unit kandidat. TF-IDF dihitung pada korpus kalimat dalam satu topik; skor kalimat adalah jumlah bobot TF-IDF-nya. Kalimat dengan skor tertinggi dipilih.
- **TextRank:** kemiripan antarkalimat dihitung menggunakan cosine similarity atas vektor TF-IDF. Nilai similarity membentuk graph k-nearest-neighbor (default 15 tetangga terdekat per kalimat), lalu PageRank mengukur sentralitas kalimat. Graph jarang ini dipakai agar penggunaan memori tetap terkendali untuk topik besar; parameter dapat diubah lewat `textrank_neighbors`.
- **TextRank + MMR:** memakai relevansi TextRank, lalu Maximum Marginal Relevance menyeimbangkan relevansi dengan perbedaan terhadap kalimat yang sudah dipilih agar ringkasan tidak terlalu berulang.

Dua baseline pertama mengikuti alur dan definisi metode pada **Modul Praktikum Text Summarization Bahasa Indonesia**. MMR merupakan pengembangan eksperimen proyek, bukan pengganti baseline.

## 2. Dataset yang disertakan

File: `data/kredivo_reviews_50k.csv`.

Pemeriksaan yang sudah dijalankan pada file yang disertakan menunjukkan **50.148 baris CSV** dan 17 kolom. Setelah membuang 100 baris tanpa ID/teks review yang dapat digunakan, tersisa **50.048 review valid**. Normalisasi yang juga mempertahankan emoji umum sebagai token menghasilkan **27.312 dokumen teks unik** dan **22.736 baris duplikat tambahan**. Sebanyak 138 bentuk teks pujian generik atau reaksi emoji tanpa konteks (13.698 baris review) dikecualikan dari clustering; 27.174 dokumen unik informatif yang mewakili 36.350 baris review dipertahankan. Angka ini adalah profil file yang dibundel dan dihitung ulang oleh pipeline saat dijalankan. Dataset memuat kolom teks asli `content`, ID `reviewId`, skor dan waktu review, serta kolom tambahan seperti `sentiment`, `issue_category`, `severity_level`, `evidence_span`, dan `confidence_score`. Dalam 50.048 review valid, 119 kelompok teks duplikat-normalisasi memiliki label sentiment berbeda, 3 kelompok memiliki issue_category berbeda, dan 70 kelompok memiliki severity_level berbeda. Karena itu, atribut label disimpan per reviewId dan tidak digeneralisasi dari satu baris duplikat ke baris lainnya.

Hal yang perlu diperhatikan:

1. Kolom `content` adalah sumber teks untuk clustering dan summarization.
2. Kolom tambahan `sentiment`, `issue_category`, `severity_level`, `evidence_span`, `cot_rationale`, dan `confidence_score` **tidak dipakai sebagai label pelatihan atau fitur untuk BERTopic/summarization**. Provenance dan ketepatan label tambahan harus diperiksa dulu.
3. `evidence_span` diperiksa secara awal dengan pencocokan substring literal. Pada file ini sekitar **69,4%** span nonkosong cocok secara literal. Ini hanya pemeriksaan awal; cocok secara literal tidak otomatis membuktikan validitas semantik, dan tidak cocok secara literal tidak otomatis berarti span salah.
4. `userName` dan `userImage` tidak digunakan atau ditampilkan. Source traceability memakai `reviewId` dan teks ulasan saja.
5. Duplikasi tidak dibuang begitu saja. Teks identik diwakili satu dokumen untuk mencegah dominasi duplikasi, sementara seluruh `reviewId` dan jumlah kemunculan tetap disimpan.
6. Stemming/lemmatization tidak diterapkan pada default pipeline karena review pendek, slang, dan bentuk informal berpotensi berubah maknanya bila dipaksakan; normalisasi slang konservatif digunakan. Tokenisasi dan stopword removal dijalankan untuk representasi sparse, dengan negasi seperti "tidak", "belum", dan kata isu penting tetap dipertahankan. Emoji umum dipetakan ke token sederhana (misalnya `emoji_senang`, `emoji_marah`, `emoji_jempol`) agar sinyalnya tidak hilang seluruhnya; teks review asli tetap disimpan.
7. Pujian generik yang berdiri sendiri (misalnya "bagus", "mantap", "ok") dikecualikan dari clustering karena minim informasi topikal. Kalimat singkat yang mengandung kendala (misalnya "tidak bisa login") tetap dipertahankan. Daftar frasa generik dapat disesuaikan di `src/kredivo_project/preprocessing.py`.

## 3. Persyaratan dan instalasi

Disarankan:

- Windows 10/11 atau Linux/macOS.
- Python **3.11 atau 3.12**.
- RAM 8 GB minimum; 16 GB lebih nyaman.
- Ruang kosong beberapa GB untuk library, cache model multilingual, dan hasil.
- Koneksi internet pada eksekusi pertama untuk mengunduh package dan model sentence-transformer. Setelah model ada di cache lokal, proses model dapat dijalankan kembali secara offline.
- CPU didukung. GPU CUDA bersifat opsional.

### Cara termudah di Windows

1. Ekstrak seluruh ZIP ke folder biasa (jangan menjalankan langsung dari dalam ZIP).
2. Pastikan Python 3.11 atau 3.12 terpasang dan perintah `py` tersedia.
3. Klik dua kali `run_windows.bat`.
4. Script membuat virtual environment `.venv`, memasang dependensi, menjalankan pipeline, lalu membuka Streamlit.
5. Buka alamat lokal yang ditampilkan di terminal (biasanya `http://localhost:8501`). Jangan tutup jendela terminal selama dashboard dipakai.

### Cara manual (Windows PowerShell)

```powershell
cd Kredivo_Topic_Based_Summarization
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\run_pipeline.py
streamlit run app.py
```

Jika PowerShell memblokir aktivasi environment, jalankan perintah berikut untuk sesi terminal saat ini:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
chmod +x run_unix.sh
./run_unix.sh
```

## 4. Hasil pipeline

Semua artefak tersimpan di `outputs/`.

| File | Isi |
|---|---|
| `data_quality_report.json` / `.md` | Profil dataset, duplikasi, panjang teks, distribusi label, pemeriksaan awal evidence span |
| `unique_documents.csv` | Dokumen hasil penggabungan teks identik, beserta daftar source ID dan duplicate count |
| `document_topic_assignments.csv` | Penetapan topic_id dan split train/validation/test untuk tiap dokumen unik |
| `topics.csv` | Jumlah dokumen/review, ukuran train-test, kata kunci, dan nama topik |
| `topic_keywords_ctfidf.csv` | Kata kunci dan skor c-TF-IDF per topik |
| `reviews_assigned.csv` | Tabel level review yang menghubungkan reviewId, teks, label tambahan, dan topic_id |
| `summary_sources.csv` | Kalimat yang dipilih tiap model, peringkat, skor, dan reviewId sumber |
| `summaries.csv` | Teks ringkasan per topik dan metode |
| `reference_summaries_template.csv` | Template ringkasan acuan manusia untuk topik test; kolom ringkasan sengaja kosong |
| `reference_examples.csv` | Contoh review sumber untuk membantu penulisan reference summary |
| `topic_quality_review_template.csv` | Template audit manusia untuk koherensi cluster BERTopic dan relevansi kata kunci menggunakan validation reviews |
| `human_evaluation_template.csv` | Lembar penilaian manusia untuk groundedness, relevance, coherence, dan non-redundancy |
| `evaluation_comparison.csv` | Perbandingan ROUGE dan metrik pendamping setelah referensi diisi |
| `summary_length_ablation_comparison.csv` | Eksperimen jumlah kalimat 1, 2, 3, dan konfigurasi utama |
| `error_analysis_cases.csv` | Tiga prediksi terbaik dan terburuk per metode untuk dianalisis |
| `label_audit_sample.csv` | Sampel label untuk pemeriksaan manusia sebelum severity filter digunakan |
| `label_audit_status.json` | Status kelulusan audit label |

## 5. Evaluasi: langkah yang wajib dilakukan

**ROUGE belum dapat dihitung secara sah sebelum reference summary tersedia.** File Kaggle yang dibundel berisi review, bukan pasangan review–ringkasan referensi. Karena itu, proyek tidak mengarang nilai ROUGE atau membuat model-generated summary sebagai ground truth.

1. Jalankan pipeline pertama kali.
2. Periksa `outputs/topic_quality_review_template.csv` yang memakai validation split. Baca sampel review per topik, beri skor koherensi cluster dan relevansi kata kunci, lalu catat cluster campuran. Jika parameter BERTopic perlu diubah, ubah berdasarkan validation dan jalankan pipeline ulang.
3. Setelah struktur topik dianggap stabil, buka `outputs/reference_examples.csv`. Untuk setiap `topic_id`, baca contoh review test yang disediakan.
4. Isi `reference_summary` pada `outputs/reference_summaries_template.csv` dengan ringkasan singkat buatan manusia, hanya berdasarkan informasi dari review. Jangan menyalin keluaran salah satu model. Upayakan 20–30 topik representatif dan panjang referensi relatif seragam. `evaluation_doc_ids` memastikan prediksi dan referensi berasal dari kumpulan dokumen test yang sama. Catat annotator; idealnya dua anggota meninjau sebagian topik dan menyelesaikan perbedaan melalui adjudikasi.
5. Jalankan ulang `python scripts/run_pipeline.py`. Evaluasi akan menghitung ROUGE-1, ROUGE-2, ROUGE-L (F1), keyword coverage, redundancy cosine similarity, source traceability rate, waktu proses, penilaian groundedness manusia, dan error analysis.
6. Setelah evaluasi menghasilkan `outputs/human_evaluation_template.csv`, file ini sudah memuat prediction dan reference yang persis dievaluasi. Isi skor 1–5 untuk groundedness, relevance, coherence, dan non-redundancy pada baris `topic_id` dan metode yang sesuai. Jalankan ulang pipeline untuk menggabungkan skor manual ke tabel evaluasi. Skor lama otomatis dikosongkan jika fingerprint prediction berubah.
7. Baca `error_analysis_cases.csv`. Isi `error_reason` dan `reviewer_notes` untuk kasus terbaik/terburuk. Analisis sekurangnya tiga kasus ROUGE-L rendah dan jelaskan penyebabnya.

### Arti metrik dan batasannya

- **ROUGE-1:** overlap unigram/kata tunggal dengan referensi.
- **ROUGE-2:** overlap bigram/dua kata berurutan.
- **ROUGE-L:** overlap berdasarkan longest common subsequence.
- **Keyword coverage:** proporsi kata kunci c-TF-IDF topik yang muncul pada ringkasan. Ini mengukur cakupan kata kunci, bukan keseluruhan informasi semantik.
- **Redundancy:** rata-rata cosine similarity antarkalimat terpilih; semakin kecil biasanya semakin tidak repetitif, tetapi perlu dibaca bersama relevansi.
- **Source traceability rate:** proporsi kalimat terpilih yang memiliki satu atau lebih `reviewId` sumber. Ini memeriksa kelengkapan tautan bukti, bukan kualitas semantik dan kemungkinan nilainya memang mendekati 100% karena sistem extractive.
- **Faithfulness/groundedness semantik:** tidak dapat dibuktikan hanya dari ekstraksi kalimat. Nilai melalui penilaian manusia pada template terpisah; periksa apakah kalimat mewakili konteks review, bukan hanya apakah kalimat dapat ditelusuri.
- **Processing time:** waktu pemrosesan yang terukur saat eksperimen dijalankan; hasil dipengaruhi perangkat dan kondisi runtime.

### Eksperimen panjang ringkasan

Konfigurasi utama memakai `summary_sentences=5` untuk merangkum beberapa review dalam satu topik. Pipeline juga menghitung ablation untuk jumlah kalimat 1, 2, 3 dan 5 saat reference summary telah diisi, sehingga instruksi praktikum untuk membandingkan satu dan dua kalimat tetap dapat diuji. Pastikan semua model dibandingkan dengan budget panjang yang sama.

## 6. Audit label dan severity filter

`outputs/label_audit_sample.csv` berisi sampel audit yang menargetkan minimal 20 contoh per issue category dan minimal 10 contoh per severity class, lalu mengisi sisa sampel secara acak hingga target 200 (dapat lebih besar bila kuota minimum memerlukan). Isi kolom `manual_issue_category`, `manual_severity_level`, set `reviewed=true` untuk setiap baris yang telah diperiksa, dan tulis catatan bila perlu. Sentiment tidak diaudit dalam paket ini karena bukan komponen metode.

Severity filter dashboard baru aktif jika semua sampel telah diperiksa, agreement keseluruhan untuk `issue_category` dan `severity_level` masing-masing mencapai ambang 0,80, macro-agreement per kelas dan agreement setiap kelas mencapai minimal 0,70, dan setiap kelas memiliki dukungan sampel minimal 10. Sampel ini sengaja meng-overrepresentasi kelas minoritas sehingga bukan estimasi akurasi populasi berbobot prevalensi. Ini adalah gerbang kehati-hatian operasional, bukan bukti bahwa label seluruh dataset pasti valid. Ambang ini adalah aturan operasional proyek, bukan standar universal. Jika agreement tidak lulus, periksa pedoman anotasi dan revisi label sebelum mengaktifkan filter. Jangan gunakan `cot_rationale` atau `confidence_score` sebagai bukti validasi.

Di halaman Eksplorasi Topik, severity filter hanya menyaring bukti review yang ditampilkan. Ringkasan default tetap dihitung dari seluruh dokumen dalam topik dan dashboard memberi pemberitahuan agar hasil tidak disalahartikan sebagai ringkasan yang sudah difilter.

## 7. Parameter yang dapat diubah

Edit `config.json`:

- `max_fit_docs`: batas dokumen train untuk fitting BERTopic. Default 12.000 mengendalikan waktu dan memori. Set `0` untuk menggunakan seluruh train split (lebih berat).
- `min_topic_size`: ukuran minimum cluster HDBSCAN. Default 30; sesuaikan berdasarkan inspeksi hasil topik.
- `embedding_model`: model SentenceTransformer multilingual.
- `validation_size` dan `test_size`: proporsi dokumen untuk validation dan test; default masing-masing 0,10 sehingga train menjadi 0,80.
- `summary_sentences`: banyak kalimat review yang dipilih per topik.
- `mmr_lambda`: keseimbangan relevansi dan keberagaman, 0–1. Angka besar lebih menekankan relevansi.
- `textrank_neighbors`: jumlah tetangga cosine terdekat untuk graph TextRank yang sparse. Default 15; nilai terlalu kecil dapat mengurangi konektivitas, terlalu besar meningkatkan biaya komputasi.
- `reference_topics`: jumlah topik test yang disertakan ke template evaluasi.
- `audit_min_agreement`: ambang agreement keseluruhan untuk label issue/severity.
- `audit_min_class_agreement` dan `audit_min_class_support`: ambang agreement minimum dan jumlah sampel minimum per kelas.

- `inference_min_similarity`: ambang *cosine similarity* minimum untuk inferensi review baru saat model dimuat dari artifact `safetensors`. Dokumen di bawah ambang ini ditandai sebagai outlier (`topic_id=-1`). Default `0.42`.

Untuk menjalankan ulang dengan konfigurasi berbeda, jalankan `python scripts/run_pipeline.py`. Template acuan yang sudah diisi dipertahankan untuk topic_id yang sama, tetapi periksa ulang bila konfigurasi clustering berubah karena nomor topik dapat bergeser.

## 7A. Antarmuka Web Interaktif (Streamlit Dashboard)

Dashboard (`app.py`) menyediakan lima halaman utama untuk mendemonstrasikan hasil penelitian secara interaktif:

1. **Overview:** Ringkasan profil dataset, pembagian train/validation/test, grafik topik dominan, jumlah outlier BERTopic, dan pratinjau ringkasan ekstraktif per topik.
2. **Topic Explorer:** Eksplorasi per topik (`topic_id >= 0` maupun kelompok outlier `-1`), meliputi grafik bobot kata kunci c-TF-IDF, distribusi skor/sentimen CSV, ringkasan dari ketiga metode, penelusuran `reviewId` sumber untuk setiap kalimat, serta daftar ulasan dalam topik.
3. **Analyze New Reviews:** Menjalankan alur NLP lengkap pada review baru tanpa melatih ulang BERTopic dan tanpa mengubah dataset utama:
   - **Input Manual:** Ketik beberapa review (satu baris per review) atau tambahkan review multi-kalimat satu per satu.
   - **Upload CSV:** Unggah CSV dengan kolom wajib **`content`**. Kolom **`reviewId`** bersifat opsional (jika tidak ada, sistem membuat ID sementara konsisten untuk sesi berjalan). Kolom tambahan seperti `score`, `at`, `sentiment`, dan `issue_category` dipertahankan bila ada tetapi tidak diwajibkan.
   - **Alur Pemrosesan:** Validasi input → preprocessing (`normalize_text`, deteksi pujian generik) → duplicate-aware grouping → embedding & proyeksi menggunakan artifact `outputs/bertopic_model` + `paraphrase-multilingual-MiniLM-L12-v2` → penandaan outlier (`-1`) & pujian generik (`-2`) → ekstraksi keyword c-TF-IDF → multi-document extractive summarization (TF-IDF Ranking, TextRank, TextRank + MMR) jika jumlah dokumen topik mencukupi (`>= 2` dokumen unik & `>= 2` kalimat) → source traceability ke `reviewId` input.
4. **Compare Summarization Models:** Perbandingan langsung TF-IDF Sentence Ranking, TextRank, dan TextRank + MMR pada kumpulan dokumen dan parameter (`summary_sentences`, `mmr_lambda`) yang sama persis—baik untuk topik dari dataset utama maupun hasil analisis review baru pada sesi berjalan. Menyajikan teks ringkasan, matriks perbedaan kalimat terpilih, bukti sumber, keyword coverage, redundancy, source traceability rate, waktu proses, serta status evaluasi ROUGE resmi.
5. **Review Explorer & Data Audit:** Pencarian teks (`content`/`reviewId`) dan filter berdasarkan topik prediksi model, kata kunci, rating CSV, sentimen CSV, dan issue category CSV, lengkap dengan tombol reset filter dan panel detail review. Filter `severity_level` tetap terkunci hingga audit manual pada `outputs/label_audit_sample.csv` dinyatakan lulus.

### Perbedaan Hasil Dataset Utama, Inference Review Baru, dan Evaluasi Eksperimen

| Aspek | Hasil Analisis Dataset Utama | Inference Review Baru (`Analyze New Reviews`) | Evaluasi Eksperimen (`Compare` & `evaluation_comparison.csv`) |
|---|---|---|---|
| **Sumber Data** | `data/kredivo_reviews_50k.csv` (50.048 review valid) | Input manual atau CSV unggahan pengguna (diproses di memory session) | Subset `test` split terpin (`evaluation_doc_ids`) untuk topik evaluasi |
| **Mekanisme Model** | Fit BERTopic pada sampel `train` split (`max_fit_docs`), lalu `transform()` ke seluruh dokumen unik informatif | Memuat artifact `outputs/bertopic_model` (`safetensors`) + `SentenceTransformer` dan memproyeksikan input baru via `transform()` tanpa retraining | Menjalankan ketiga summarizer pada subset dokumen uji yang sama persis |
| **Fungsi Utama** | Menemukan struktur topik, kata kunci c-TF-IDF, dan ringkasan korpus penelitian | Mendemonstrasikan generalisasi model pada ulasan baru dan menelusuri buktinya | Mengukur ROUGE-1/2/L terhadap ringkasan manusia, ablation panjang, human evaluation, dan error analysis |
| **Batasan Penting** | Tidak memakai label CSV sebagai ground truth | Artifact `safetensors` tidak menyimpan UMAP/HDBSCAN sehingga proyeksi memakai *cosine similarity* ke embedding topik + ambang `inference_min_similarity` | ROUGE tidak dihitung sebelum `reference_summary` diisi oleh manusia |

## 8. Catatan metodologis dan keterbatasan

1. Review aplikasi sangat pendek dan sebagian besar satu kalimat. Sistem merangkum kumpulan review per topik, bukan mencoba membuat abstraksi dari satu review pendek.
2. Kalimat extractive yang dipilih adalah potongan asli. Sistem tidak menulis ulang atau menggabungkan kata dari beberapa sumber menjadi kalimat baru.
3. BERTopic dapat menghasilkan outlier (`topic_id=-1`). Dokumen tersebut belum tentu tidak penting; model belum dapat menempatkannya pada cluster yang stabil. Outlier dihitung dan dilaporkan, tetapi tidak dimasukkan ke ringkasan per topik agar kumpulan yang tidak koheren tidak digabung paksa.
4. Hasil clustering bergantung pada model embedding, parameter UMAP/HDBSCAN, dan karakter korpus. Label topik otomatis perlu diperiksa manusia.
5. TF-IDF berbasis kata memiliki keterbatasan semantik; TextRank bergantung pada kemiripan antarkalimat; MMR dapat membuang kalimat yang mirip meskipun mengandung detail berbeda.
6. ROUGE hanya overlap leksikal. Nilai tinggi tidak selalu berarti ringkasan paling bermakna; nilai rendah dapat terjadi akibat lexical mismatch.
7. Karena itu, laporan akhir perlu menyertakan hasil aktual, contoh terbaik/terburuk, error analysis, review manual, konfigurasi, seed, dan informasi runtime. Jangan mengklaim suatu model paling baik sebelum menjalankan evaluasi.

## 9. Troubleshooting

**`No module named ...`**
Aktifkan `.venv` di folder proyek lalu jalankan `pip install -r requirements.txt` kembali.

**Unduhan model gagal / Hugging Face tidak terjangkau**
Pastikan internet tersedia, cek ruang disk dan coba jalankan pipeline ulang. Model tidak dibundel dalam ZIP karena ukuran model dan dependensi PyTorch besar. Setelah unduhan pertama selesai, model tersimpan di cache Hugging Face.

**RAM rendah / proses lama**
Ubah `max_fit_docs` menjadi 6.000–8.000, gunakan CPU, dan kurangi `reference_topics`. Jangan ubah metode baseline. Catat perubahan konfigurasi pada laporan eksperimen.

**Tidak cukup topik / terlalu banyak outlier**
Eksplorasi `min_topic_size`, `umap_n_neighbors`, dan `max_fit_docs` secara terkontrol. Jangan memilih parameter hanya karena menghasilkan tampilan dashboard yang terlihat bagus; simpan catatan dan bandingkan keterbacaan topik serta keluaran ringkasan.

**PowerShell memblokir Activate.ps1**
Jalankan `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, lalu aktifkan ulang environment.

**ROUGE belum muncul**
Pastikan kolom `reference_summary` pada CSV template sudah terisi. Pipeline tidak membuat referensi otomatis karena itu akan menyebabkan evaluasi tidak independen.

## 10. Struktur proyek

```text
Kredivo_Topic_Based_Summarization/
├── app.py
├── config.json
├── requirements.txt
├── run_windows.bat
├── run_dashboard_windows.bat
├── run_unix.sh
├── data/
│   └── kredivo_reviews_50k.csv
├── src/kredivo_project/
│   ├── data.py
│   ├── preprocessing.py
│   ├── topics.py
│   ├── summarization.py
│   ├── audit.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── explorer.py
│   └── pipeline.py
├── scripts/run_pipeline.py
├── tests/
│   ├── test_audit.py
│   ├── test_interactive_features.py
│   ├── test_pipeline_smoke.py
│   ├── test_preprocessing.py
│   ├── test_summarization.py
│   └── test_topics_integration.py
├── docs/
│   ├── ANNOTATION_GUIDE.md
│   ├── COURSE_ALIGNMENT.md
│   ├── EXPERIMENT_LOG_TEMPLATE.md
│   └── VALIDATION_STATUS.md
└── outputs/
```

## 11. Reproducibility dan Responsible NLP

- Seed split dan sampling diatur melalui `random_state` di `config.json`; catat perubahan konfigurasi dan versi package dalam laporan eksperimen.
- Dataset review dapat memuat opini dan informasi pribadi yang ditulis pengguna. Proyek tidak menggunakan `userName` atau `userImage`, tidak mencoba mengidentifikasi pengguna, dan sebaiknya hanya membagikan keluaran agregat atau kutipan seperlunya.
- Label tambahan dari dataset diperlakukan sebagai label belum tervalidasi; kelompok teks duplikat dapat memiliki label berbeda, sehingga label dipertahankan per reviewId dan tidak dipukul rata untuk setiap bukti; kategori kecil dan kelompok tertentu dapat kurang terwakili. Lakukan audit manual serta laporkan keterbatasan representasi.
- Pujian generik dikecualikan hanya jika berdiri sendiri. Jangan menganggap seluruh review positif atau seluruh kategori `general_satisfaction` tidak relevan.
- MMR, filter, dan parameter topic modeling dapat memengaruhi isu yang ditonjolkan. Dokumentasikan keputusan tersebut, cek contoh minoritas/outlier, dan jangan menyembunyikan hasil buruk.
- Model multilingual dan dependensi dijalankan lokal setelah diunduh. Pertimbangkan RAM, waktu CPU, ukuran cache, dan sumber daya saat menentukan eksperimen.

## 12. Referensi metode

- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*. arXiv:2203.05794.
- Modul Praktikum NLP — *Text Summarization Bahasa Indonesia: TF-IDF Sentence Ranking vs TextRank*, Universitas Negeri Malang (file materi yang digunakan pada mata kuliah).
- RPS NLP Pertemuan 1: target pembelajaran dan kriteria proyek yang meliputi problem formulation, relevansi dataset, preprocessing, pemilihan metode, eksperimen, evaluasi, analisis, keterbatasan, reproducibility, dokumentasi, dan presentasi.

Hasil `evaluation_comparison.csv` merupakan **macro-average per topik** (setiap topik yang dievaluasi memiliki bobot sama), bukan rata-rata berbobot jumlah review. Laporkan jumlah topik evaluasi, jumlah contoh review per topik, dan konfigurasi anotasi saat mengutip hasil tersebut.
