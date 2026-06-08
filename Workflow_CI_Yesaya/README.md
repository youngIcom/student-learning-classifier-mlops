# Workflow CI Yesaya

Folder ini berisi pipeline training, MLflow tracking, CI/CD, serving, dan monitoring untuk sistem klasifikasi performa belajar siswa.

## Entry Point

Training lokal:

```bash
python modelling.py --data-path ../Eksperimen_SML_Yesaya/open_UL_analysis_preprocessing/student_cleaned.csv
```

MLflow project:

```bash
mlflow run .
```

Hyperparameter tuning:

```bash
python modelling_tuning.py --data-path ../Eksperimen_SML_Yesaya/open_UL_analysis_preprocessing/student_cleaned.csv
```

Serving API:

```bash
uvicorn inference:app --host 0.0.0.0 --port 8000
```

Prometheus exporter:

```bash
python prometheus_exporter.py
```

## Output Training

Artefak lokal disimpan di `artifacts/`:

- `model.keras`
- `scaler.joblib`
- `feature_columns.json`
- `metric_info.json`
- `confusion_matrix.png`
- `mlflow_model/`

## Integrasi Eksternal

- MLflow tracking dapat diarahkan ke DagsHub lewat `MLFLOW_TRACKING_URI`.
- Docker image dibangun lewat workflow CI menggunakan `mlflow models build-docker`.
- Secrets tidak boleh disimpan di repository.
