from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
import shutil

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ARTIFACT_DIR = Path("artifacts")
LABEL_MAPPING = {"At_Risk": 0, "Good": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_training_frame(data_path: str | Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {path}")

    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError("Cleaned dataset must contain a 'target' column.")

    drop_cols = [col for col in ["target", "target_label"] if col in df.columns]
    feature_frame = df.drop(columns=drop_cols)
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce").fillna(0)
    target = df["target"].astype(int)
    return feature_frame, target, list(feature_frame.columns)


def build_model(input_dim: int, learning_rate: float = 0.001, hidden_units: int = 64, dropout: float = 0.2) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_units, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(max(hidden_units // 2, 8), activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
        yticklabels=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def configure_mlflow(experiment_name: str) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def train(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, feature_columns = load_training_frame(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    configure_mlflow(args.experiment_name)
    mlflow.keras.autolog(log_models=False)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "data_path": str(args.data_path),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "hidden_units": args.hidden_units,
                "dropout": args.dropout,
                "test_size": args.test_size,
                "seed": args.seed,
            }
        )

        model = build_model(
            input_dim=X_train_scaled.shape[1],
            learning_rate=args.learning_rate,
            hidden_units=args.hidden_units,
            dropout=args.dropout,
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        model.fit(
            X_train_scaled,
            y_train,
            validation_split=0.2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early_stopping],
            verbose=args.verbose,
        )

        probabilities = model.predict(X_test_scaled, verbose=0).ravel()
        predictions = (probabilities >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        }
        mlflow.log_metrics(metrics)

        metric_path = ARTIFACT_DIR / "metric_info.json"
        metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        cm_path = ARTIFACT_DIR / "confusion_matrix.png"
        save_confusion_matrix(y_test.to_numpy(), predictions, cm_path)

        model_path = ARTIFACT_DIR / "model.keras"
        scaler_path = ARTIFACT_DIR / "scaler.joblib"
        feature_path = ARTIFACT_DIR / "feature_columns.json"
        label_path = ARTIFACT_DIR / "label_mapping.json"
        mlflow_model_path = ARTIFACT_DIR / "mlflow_model"

        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        feature_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
        label_path.write_text(json.dumps(LABEL_MAPPING, indent=2), encoding="utf-8")

        signature = infer_signature(X_test_scaled, probabilities)
        mlflow.keras.log_model(model, artifact_path="model", signature=signature)
        if mlflow_model_path.exists():
            shutil.rmtree(mlflow_model_path)
        mlflow.keras.save_model(model, path=str(mlflow_model_path), signature=signature)
        mlflow.log_artifacts(str(ARTIFACT_DIR))

        print(json.dumps(metrics, indent=2))
        return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OULAD student performance classifier.")
    parser.add_argument("--data-path", default="../Eksperimen_SML_Yesaya/open_UL_analysis_preprocessing/student_cleaned.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default="student-learning-classifier")
    parser.add_argument("--run-name", default="baseline-neural-network")
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
