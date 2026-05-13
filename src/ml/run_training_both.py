"""Train models to predict all scores in a single unified model.

Dataset: one row per (student, assessment) for both exams and assignments
         → up to (181 × 3) + (181 × 6) = 1629 rows.

One target column (normalized_score) covers both exam and assignment scores.
Two extra features help the model distinguish contexts:
    is_exam        — 1 for exam rows, 0 for assignment rows
    assess_*       — one-hot (or ordinal) assessment identity

Grade features (a1–a7) are present on exam rows only.
Assignment rows have grade features set to NaN (circular leakage guard).

Output is written to a timestamped subdirectory of --output:
    results/both_YYYYMMDD_HHMMSS/
        dataset.csv        — features + target passed to the model
        results.csv        — per-model RMSE across outer folds
        run_summary.json   — full run metadata
        final_model.pkl    — best model trained on the full dataset

Arguments
---------
    --encoding      one_hot | ordinal | none (default: none)
                    How to encode assessment identity (e1–e3, a2–a7).
    --drop-no-dialogue
                    Drop rows where the student had zero LLM activity for
                    that specific assessment period.
    --features COL [COL ...]
                    Exact column names to pass to the model.
                    Structural columns (assessment_code, is_exam) are always kept.
                    Omit to use all available columns.
    --models MODEL [MODEL ...]
                    Subset of models to evaluate.
                    Available: ridge, elastic_net, random_forest,
                               gradient_boosting, svr
                    Default: all models.
    --outer K       Number of outer GroupKFold splits (default: 5).
    --inner K       Number of inner GroupKFold splits for hyperparameter
                    search (default: 5).
    --output PATH   Base directory for run subdirectories (default: results/).

Usage
-----
    # Minimal — all defaults
    python -m src.ml.run_training_both

    # Full args example
    python -m src.ml.run_training_both \
        --encoding one_hot \
        --drop-no-dialogue \
        --features dlg_n_turns dlg_n_chats dlg_avg_prompt_chars \
        --models ridge random_forest \
        --outer 10 --inner 5 \
        --output results/
"""
import argparse
from pathlib import Path

from src.ml import config as cfg
from src.ml import data_loader, data_preprocessing, feature_engineering
from src.ml.config import (
    AssessmentEncoding,
    ASSIGNMENT_ASSESSMENT_CODES,
    EXAM_ASSESSMENT_CODES,
)
from src.ml.db import load_engine
from src.ml.model_training import (
    append_run_log,
    make_run_dir,
    print_summary,
    run_nested_cv,
    save_dataset,
    save_model,
    save_results,
    train_final_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested CV across all assessments (exam + assignment) in one model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--encoding",
        default=AssessmentEncoding.NONE.value,
        choices=[e.value for e in AssessmentEncoding],
        help="How to encode assessment identity (default: none)",
    )
    parser.add_argument(
        "--drop-no-dialogue",
        action="store_true", dest="drop_no_dialogue",
        help="Drop rows where the student had zero dialogue activity for that assessment",
    )
    parser.add_argument(
        "--features",
        nargs="+", default=None, metavar="COL",
        help=(
            "Exact feature columns to pass to the model "
            "(e.g. dlg_n_turns dlg_avg_prompt_chars). "
            "Structural columns (assessment_code, is_exam) are always kept. "
            "If omitted, all available columns are used."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+", default=None, metavar="MODEL",
        help=f"Subset of models to evaluate. Available: {list(cfg.MODEL_REGISTRY)}",
    )
    parser.add_argument(
        "--outer", type=int, default=cfg.OUTER_CV_SPLITS, metavar="K",
        help="Outer CV folds (default: %(default)s)",
    )
    parser.add_argument(
        "--inner", type=int, default=cfg.INNER_CV_SPLITS, metavar="K",
        help="Inner CV folds for hyperparameter search (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results"),
        help="Base directory for run subfolders (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = load_engine()
    run_dir = make_run_dir(args.output, pipeline_type="both")

    all_codes = EXAM_ASSESSMENT_CODES + ASSIGNMENT_ASSESSMENT_CODES

    base = data_loader.load_base_frame(engine, pipeline_type="both")

    X, y, groups = data_preprocessing.build_dataset(
        engine=engine,
        base=base,
        pipeline_type="both",
        drop_no_dialogue=args.drop_no_dialogue,
        select_features=args.features,
    )

    X = feature_engineering.engineer_features(
        X, encoding=args.encoding, expected_codes=all_codes,
    )

    save_dataset(X, y, run_dir)

    results_df = run_nested_cv(
        X=X, y=y, groups=groups,
        model_names=args.models,
        outer_splits=args.outer,
        inner_splits=args.inner,
    )

    print_summary(results_df, label="both  (exam + assignment)")

    best_model_name = results_df["mean_rmse"].idxmin()
    fitted = train_final_model(X, y, groups, best_model_name, args.inner)
    save_model(fitted, run_dir / "final_model.pkl")

    save_results(
        results_df=results_df,
        output_dir=run_dir,
        pipeline_type="both",
        assessment_codes=all_codes,
        outer_splits=args.outer,
        inner_splits=args.inner,
        n_samples=len(X),
        n_features=X.shape[1],
        n_students=int(groups.nunique()),
        feature_columns=list(X.columns),
        assessment_encoding=args.encoding,
        best_model_params=fitted.best_params_,
    )
    append_run_log(
        base_output=args.output,
        run_dir=run_dir,
        results_df=results_df,
        pipeline_type="both",
        assessment_codes=all_codes,
        outer_splits=args.outer,
        inner_splits=args.inner,
        n_samples=len(X),
        n_features=X.shape[1],
        n_students=int(groups.nunique()),
        assessment_encoding=args.encoding,
        best_model_params=fitted.best_params_,
    )

    print(f"\nRun directory: {run_dir}")


if __name__ == "__main__":
    main()
