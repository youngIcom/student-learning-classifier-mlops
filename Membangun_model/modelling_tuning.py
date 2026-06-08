from __future__ import annotations

import argparse
from itertools import product

from Workflow_CI_Yesaya.MLProject.modelling import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight hyperparameter tuning.")
    parser.add_argument("--data-path", default="../Eksperimen_SML_Yesaya/open_UL_analysis_preprocessing/student_cleaned.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    base_args = parse_args()
    search_space = {
        "learning_rate": [0.001, 0.0005],
        "hidden_units": [64, 128],
        "dropout": [0.2, 0.3],
    }

    best_run: tuple[float, str] | None = None
    for learning_rate, hidden_units, dropout in product(
        search_space["learning_rate"], search_space["hidden_units"], search_space["dropout"]
    ):
        run_name = f"tuning-lr{learning_rate}-hu{hidden_units}-do{dropout}"
        args = argparse.Namespace(
            data_path=base_args.data_path,
            epochs=base_args.epochs,
            batch_size=base_args.batch_size,
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            dropout=dropout,
            test_size=base_args.test_size,
            seed=base_args.seed,
            experiment_name="student-learning-classifier-tuning",
            run_name=run_name,
            verbose=base_args.verbose,
        )
        metrics = train(args)
        score = metrics["f1_score"]
        if best_run is None or score > best_run[0]:
            best_run = (score, run_name)

    if best_run:
        print(f"Best tuning run: {best_run[1]} with f1_score={best_run[0]:.4f}")


if __name__ == "__main__":
    main()
