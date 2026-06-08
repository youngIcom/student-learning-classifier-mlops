from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_DIRNAME = "open+university+learning+analytics+dataset"
DEFAULT_RAW_DIR = EXPERIMENT_ROOT / "dataset" / RAW_DATASET_DIRNAME
LEGACY_RAW_DIR = EXPERIMENT_ROOT.parent / "dataset" / RAW_DATASET_DIRNAME
_raw_dir_env = os.getenv("OULAD_RAW_DIR")
RAW_DIR = Path(_raw_dir_env) if _raw_dir_env else DEFAULT_RAW_DIR
if not RAW_DIR.exists() and LEGACY_RAW_DIR.exists():
    RAW_DIR = LEGACY_RAW_DIR
OUTPUT_DIR = EXPERIMENT_ROOT / "open_UL_analysis_preprocessing"
OUTPUT_FILE = OUTPUT_DIR / "student_cleaned.csv"
METADATA_FILE = OUTPUT_DIR / "preprocessing_metadata.json"

KEY_COLS = ["code_module", "code_presentation", "id_student"]
TARGET_MAP = {
    "Pass": "Good", 
    "Distinction": "Good",
    "Fail": "At_Risk",
    "Withdrawn": "At_Risk",
}
LABEL_TO_ID = {"At_Risk": 0, "Good": 1}


def _read_csv(name: str, **kwargs) -> pd.DataFrame:
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required raw dataset file not found: {path}")
    return pd.read_csv(path, na_values=["?"], **kwargs)


def _aggregate_assessments() -> pd.DataFrame:
    assessments = _read_csv("assessments.csv")
    student_assessment = _read_csv("studentAssessment.csv")

    assessments["date"] = pd.to_numeric(assessments["date"], errors="coerce")
    assessments["weight"] = pd.to_numeric(assessments["weight"], errors="coerce").fillna(0)

    student_assessment["date_submitted"] = pd.to_numeric(
        student_assessment["date_submitted"], errors="coerce"
    )
    student_assessment["score"] = pd.to_numeric(student_assessment["score"], errors="coerce")
    student_assessment["is_banked"] = pd.to_numeric(
        student_assessment["is_banked"], errors="coerce"
    ).fillna(0)

    merged = student_assessment.merge(assessments, on="id_assessment", how="left")
    merged["late_submission"] = (
        merged["date"].notna()
        & merged["date_submitted"].notna()
        & (merged["date_submitted"] > merged["date"])
    ).astype(int)
    merged["weighted_score_part"] = merged["score"].fillna(0) * merged["weight"]
    merged["score_weight"] = np.where(merged["score"].notna(), merged["weight"], 0)

    grouped = (
        merged.groupby(KEY_COLS, as_index=False)
        .agg(
            assessment_submitted_count=("id_assessment", "count"),
            assessment_mean_score=("score", "mean"),
            assessment_min_score=("score", "min"),
            assessment_max_score=("score", "max"),
            assessment_late_count=("late_submission", "sum"),
            assessment_banked_count=("is_banked", "sum"),
            weighted_score_sum=("weighted_score_part", "sum"),
            score_weight_sum=("score_weight", "sum"),
        )
    )
    grouped["assessment_weighted_score"] = (
        grouped["weighted_score_sum"] / grouped["score_weight_sum"].replace(0, np.nan)
    )
    return grouped.drop(columns=["weighted_score_sum", "score_weight_sum"])


def _aggregate_vle(chunksize: int = 500_000) -> pd.DataFrame:
    path = RAW_DIR / "studentVle.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required raw dataset file not found: {path}")

    summary_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk["date"] = pd.to_numeric(chunk["date"], errors="coerce")
        chunk["sum_click"] = pd.to_numeric(chunk["sum_click"], errors="coerce").fillna(0)
        chunk["clicks_before_start"] = np.where(chunk["date"] < 0, chunk["sum_click"], 0)
        chunk["clicks_first_30_days"] = np.where(
            chunk["date"].between(0, 30, inclusive="both"), chunk["sum_click"], 0
        )
        chunk["clicks_first_60_days"] = np.where(
            chunk["date"].between(0, 60, inclusive="both"), chunk["sum_click"], 0
        )

        summary = (
            chunk.groupby(KEY_COLS, as_index=False)
            .agg(
                vle_total_clicks=("sum_click", "sum"),
                vle_interaction_count=("sum_click", "count"),
                vle_first_click_day=("date", "min"),
                vle_last_click_day=("date", "max"),
                vle_clicks_before_start=("clicks_before_start", "sum"),
                vle_clicks_first_30_days=("clicks_first_30_days", "sum"),
                vle_clicks_first_60_days=("clicks_first_60_days", "sum"),
            )
        )
        summary_parts.append(summary)
        daily_parts.append(chunk[KEY_COLS + ["date"]].dropna().drop_duplicates())

    summary_all = (
        pd.concat(summary_parts, ignore_index=True)
        .groupby(KEY_COLS, as_index=False)
        .agg(
            vle_total_clicks=("vle_total_clicks", "sum"),
            vle_interaction_count=("vle_interaction_count", "sum"),
            vle_first_click_day=("vle_first_click_day", "min"),
            vle_last_click_day=("vle_last_click_day", "max"),
            vle_clicks_before_start=("vle_clicks_before_start", "sum"),
            vle_clicks_first_30_days=("vle_clicks_first_30_days", "sum"),
            vle_clicks_first_60_days=("vle_clicks_first_60_days", "sum"),
        )
    )

    active_days = (
        pd.concat(daily_parts, ignore_index=True)
        .drop_duplicates()
        .groupby(KEY_COLS, as_index=False)
        .size()
        .rename(columns={"size": "vle_active_days"})
    )

    result = summary_all.merge(active_days, on=KEY_COLS, how="left")
    result["vle_active_days"] = result["vle_active_days"].fillna(0)
    result["vle_avg_clicks_per_active_day"] = (
        result["vle_total_clicks"] / result["vle_active_days"].replace(0, np.nan)
    ).fillna(0)
    return result


def build_preprocessed_dataset() -> pd.DataFrame:
    student_info = _read_csv("studentInfo.csv")
    courses = _read_csv("courses.csv")
    registration = _read_csv("studentRegistration.csv")

    student_info["target_label"] = student_info["final_result"].map(TARGET_MAP)
    if student_info["target_label"].isna().any():
        unknown = sorted(student_info.loc[student_info["target_label"].isna(), "final_result"].unique())
        raise ValueError(f"Unknown final_result labels: {unknown}")

    registration["date_registration"] = pd.to_numeric(
        registration["date_registration"], errors="coerce"
    )
    registration = registration[KEY_COLS + ["date_registration"]]

    data = (
        student_info.drop(columns=["final_result"])
        .merge(courses, on=["code_module", "code_presentation"], how="left")
        .merge(registration, on=KEY_COLS, how="left")
        .merge(_aggregate_assessments(), on=KEY_COLS, how="left")
        .merge(_aggregate_vle(), on=KEY_COLS, how="left")
    )

    data = data.drop(columns=["id_student"])
    categorical_cols = [
        "code_module",
        "code_presentation",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ]
    numeric_cols = [col for col in data.columns if col not in categorical_cols + ["target_label"]]

    for col in categorical_cols:
        data[col] = data[col].fillna("Unknown").astype(str)

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        fill_value = data[col].median()
        if pd.isna(fill_value):
            fill_value = 0
        data[col] = data[col].fillna(fill_value)

    data["target"] = data["target_label"].map(LABEL_TO_ID).astype(int)
    encoded = pd.get_dummies(data, columns=categorical_cols, dtype=int)

    target_label = encoded.pop("target_label")
    target = encoded.pop("target")
    encoded["target_label"] = target_label
    encoded["target"] = target
    return encoded


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_preprocessed_dataset()
    dataset.to_csv(OUTPUT_FILE, index=False)

    feature_columns = [col for col in dataset.columns if col not in {"target", "target_label"}]
    metadata = {
        "source": "Open University Learning Analytics Dataset",
        "rows": int(dataset.shape[0]),
        "columns": int(dataset.shape[1]),
        "feature_count": len(feature_columns),
        "target_column": "target",
        "target_label_column": "target_label",
        "label_to_id": LABEL_TO_ID,
        "target_distribution": dataset["target_label"].value_counts().to_dict(),
        "feature_columns": feature_columns,
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved cleaned dataset: {OUTPUT_FILE}")
    print(f"Saved metadata: {METADATA_FILE}")
    print(f"Shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns")
    print("Target distribution:")
    print(dataset["target_label"].value_counts())


if __name__ == "__main__":
    main()
