# Template Log Eksperimen

Salin bagian ini untuk setiap eksperimen yang dibandingkan.

## Identitas eksperimen
- Experiment ID:
- Tanggal/waktu:
- Operator:
- Commit/version kode (jika memakai Git):
- Sistem operasi:
- Versi Python:
- CPU / RAM / GPU:

## Dataset dan preprocessing
- File dataset:
- Jumlah baris mentah:
- Review valid setelah audit:
- Dokumen unik setelah normalisasi:
- Dokumen informatif:
- Jumlah review duplikat tambahan:
- Strategi normalisasi/slang:
- Konfirmasi negasi dipertahankan:
- Seed split:

## Konfigurasi model
- Embedding model:
- Maksimum dokumen fitting BERTopic:
- Min topic size:
- Parameter UMAP:
- Parameter TextRank k-nearest neighbors:
- Jumlah kalimat per ringkasan:
- MMR lambda:
- Versi `requirements.txt`:

## Validasi topik
- Jumlah topik yang diperiksa pada validation split:
- Skor rata-rata koherensi topik (1–5):
- Skor relevansi kata kunci (1–5):
- Topik campuran / outlier yang perlu dicatat:

## Evaluasi
- Jumlah topik evaluasi:
- Jumlah review contoh per topik:
- Annotator/reference summary:
- ROUGE-1 F1:
- ROUGE-2 F1:
- ROUGE-L F1:
- Keyword coverage:
- Redundancy:
- Source traceability rate:
- Manual groundedness / faithfulness (1–5):
- Manual relevance (1–5):
- Manual coherence (1–5):
- Manual non-redundancy (1–5):
- Runtime:

## Observasi dan error analysis
- Contoh terbaik:
- Contoh terburuk:
- Error utama:
- Perubahan dari eksperimen sebelumnya:
- Dampak perubahan:
- Keterbatasan / risiko bias:
- Kesimpulan (hindari klaim di luar bukti eksperimen):
