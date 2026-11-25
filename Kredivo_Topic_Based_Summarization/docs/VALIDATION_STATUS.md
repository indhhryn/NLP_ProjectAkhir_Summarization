# Status validasi paket

## Yang sudah diverifikasi pada paket ini

- Arsip ZIP dapat dibaca tanpa kesalahan CRC.
- Semua file Python dikompilasi.
- Unit test preprocessing, summarization, audit label, dan alur pipeline dengan stub topik berjalan.
- Hasil pengujian terakhir: **18 test lulus**.
- Pipeline diuji dengan dataset sintetis terkontrol untuk memeriksa penulisan artefak, pemetaan topic_id, pemisahan test, template anotasi, dan keluaran ringkasan.
- Tiga summarizer juga dijalankan pada sampel 80 dokumen unik informatif dari CSV Kredivo asli (99 kandidat kalimat); 19 relasi reviewId yang dikeluarkan diperiksa terhadap teks mentah dan tidak ditemukan tautan sumber yang salah.
- Sampel audit label pada paket telah dibuat ulang dengan minimum 20 contoh per issue category dan 10 contoh per severity class.
- Uji regresi memastikan reviewId tidak dibagikan ke variasi teks yang hanya sama setelah normalisasi.
- Uji regresi memastikan vektorisasi tidak gagal pada kumpulan kalimat yang sangat homogen.
- Evaluasi ditolak jika `evaluation_doc_ids` kosong atau merujuk dokumen di luar subset test yang dipakai anotator.
- Template penilaian manusia menampilkan prediction/reference aktual dan menghapus skor lama bila fingerprint prediksi berubah.
- Template audit koherensi topik menghapus skor lama bila kata kunci atau sampel validation berubah.

## Yang belum dapat diklaim sudah diuji penuh

- BERTopic aktual dengan model sentence-transformer multilingual belum dijalankan end-to-end pada lingkungan pembuat paket karena dependensi BERTopic, sentence-transformers, UMAP, HDBSCAN, dan Streamlit tidak tersedia di lingkungan pengujian ini.
- Dashboard Streamlit belum diuji secara interaktif dengan hasil BERTopic aktual.
- ROUGE belum dihitung karena dataset Kredivo tidak menyediakan reference summary. Angka ROUGE baru sah setelah tim mengisi template dengan ringkasan acuan manusia yang independen.
- Label issue_category/severity_level belum divalidasi manusia. Severity filter tetap nonaktif sampai seluruh sampel audit dilengkapi dan semua ambang audit terpenuhi.

## Implikasi praktis

Jalankan `run_windows.bat` pada Windows dengan Python 3.11 atau 3.12 dan internet. Jalankan model pada CPU terlebih dahulu; waktu dan memori bergantung pada perangkat. Catat error, waktu proses, jumlah topik, ukuran outlier, dan konfigurasi aktual. Jangan menulis klaim hasil eksperimen sebelum pipeline berhasil dijalankan dan hasilnya diperiksa.
