# Eksperimen SML Yesaya

Folder ini berisi eksperimen dataset dan pipeline preprocessing untuk proyek klasifikasi performa belajar siswa berbasis Open University Learning Analytics Dataset (OULAD).

## Struktur

- `open_UL_analysis_raw/`: catatan sumber raw dataset.
- `preprocessing/eksperimen_yesaya.ipynb`: notebook eksplorasi dan validasi.
- `preprocessing/automate_yesaya.py`: script preprocessing otomatis.
- `open_UL_analysis_preprocessing/student_cleaned.csv`: output dataset bersih.
- `.github/workflows/preprocessing.yml`: workflow otomatis untuk menjalankan preprocessing.

## Target Klasifikasi

Target `final_result` dari OULAD dipetakan menjadi target binary:

- `Pass` dan `Distinction` menjadi `Good`
- `Fail` dan `Withdrawn` menjadi `At_Risk`

Pendekatan ini sesuai untuk sistem early warning/adaptive learning karena model memprediksi apakah siswa berada dalam kondisi performa baik atau berisiko.

## Cara Menjalankan

Jalankan dari root project:

```bash
python Eksperimen_SML_Yesaya/preprocessing/automate_yesaya.py
```

Output akan tersimpan di:

```text
Eksperimen_SML_Yesaya/open_UL_analysis_preprocessing/student_cleaned.csv
```
