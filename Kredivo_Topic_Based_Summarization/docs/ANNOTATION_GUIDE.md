# Panduan anotasi proyek

## A. Reference summary per topik

1. Buka `outputs/reference_examples.csv` dan kelompokkan contoh berdasarkan `topic_id`.
2. Baca seluruh contoh review yang tersedia untuk topik tersebut sebelum menulis.
3. Tulis 1–3 kalimat yang merangkum masalah/pengalaman utama. Sasaran praktis: 25–60 kata; usahakan relatif seragam antar-topik.
4. Jangan menambahkan penyebab, fitur, kebijakan, atau kesimpulan yang tidak didukung isi review.
5. Jangan menyalin ringkasan dari TF-IDF, TextRank, atau TextRank + MMR.
6. Jika satu cluster tampak bercampur, tulis ringkasan untuk pola yang paling dominan dan catat masalah cluster di `annotation_notes`.
7. Isi nama/inisial annotator sesuai ketentuan tim dan simpan CSV dengan encoding UTF-8.

## B. Audit label

Untuk setiap baris dalam `label_audit_sample.csv`, bandingkan review dengan label sistem. Isi `manual_issue_category` dan `manual_severity_level` menurut isi review, bukan menurut dugaan sistem. Gunakan ejaan label kanonis berikut:

- `issue_category`: `general_satisfaction`, `limit_approval`, `interest_penalty`, `ui_ux`, `payment_reconciliation`, `login_biometric`, `debt_collection_dispute`, `app_technical_bug`.
- `severity_level`: `none`, `low`, `medium`, `high`, `critical`.

Kolom sentiment tidak perlu diaudit karena bukan bagian metode proyek. Set `reviewed=true` hanya setelah kedua label diperiksa. Jika kategori tidak jelas, catat alasan di `reviewer_notes` dan diskusikan dengan anotator kedua. Sampel sengaja menambah proporsi kelas minoritas; agreement keseluruhan pada sampel ini bukan estimasi akurasi dataset yang dibobot menurut prevalensi.

## C. Skor kualitas ringkasan manual

Setelah reference summary diisi dan pipeline dijalankan ulang, buka `human_evaluation_template.csv`. File ini memuat **prediction dan reference yang persis dievaluasi** untuk setiap `topic_id` dan metode (menggunakan panjang ringkasan utama). Bandingkan dua kolom tersebut, lalu isi skor di baris yang sama. Pipeline menyimpan fingerprint prediksi; jika konfigurasi atau input mengubah prediksi, skor lama pada baris tersebut otomatis dikosongkan agar tidak menempel ke hasil yang berbeda.

Gunakan skala 1–5:

- 1 = buruk/tidak terpenuhi; 2 = lemah; 3 = cukup; 4 = baik; 5 = sangat baik.
- groundedness: semua klaim ringkasan didukung review yang ditautkan.
- relevance: ringkasan mewakili isu dominan pada topik.
- coherence: urutan potongan dapat dipahami sebagai satu ringkasan.
- non-redundancy: ringkasan tidak mengulang informasi yang sama tanpa kebutuhan.

Nilai ketiga metode untuk topik yang sama. Jangan memberi skor hanya berdasarkan panjang atau gaya bahasa.

## D. Menjaga kesetaraan data evaluasi

`reference_summaries_template.csv` menyimpan daftar `evaluation_doc_ids`. Daftar itu menunjuk ke dokumen test yang juga ditampilkan di `reference_examples.csv`. Tulis referensi berdasarkan contoh yang terdaftar tersebut; jangan menambahkan review train atau review lain yang tidak tercantum. Pipeline akan membatasi prediksi evaluasi ke daftar dokumen yang sama.

Untuk penilaian dua anotator, simpan nilai masing-masing reviewer pada salinan lembar kerja terpisah, lalu sepakati nilai final. Agreement audit label di `label_audit_sample.csv` membandingkan label dataset dengan label manual reviewer; angka ini **bukan** inter-annotator agreement dan bukan bukti validitas untuk semua 49 ribu review.


## E. Audit kualitas cluster BERTopic

Gunakan `topic_quality_review_template.csv` yang dibuat dari validation split. Untuk setiap topik, baca contoh review dan kata kunci c-TF-IDF. Beri skor koherensi topik dan relevansi kata kunci (1–5), tandai apakah cluster tampak campuran, dan tulis alasan. Fingerprint topik digunakan untuk menghapus skor lama jika contoh/representasi topik berubah. Jangan gunakan test split untuk mengubah parameter clustering; simpan test untuk evaluasi akhir.
