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
    save_correlation_matrix,
    save_dataset,
    save_feature_importance,
    save_model,
    save_prediction_error_plot,
    save_predictions,
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
        "--drop-zero-score",
        action="store_true", dest="drop_zero_score",
        help="Drop rows where the student received a score of exactly 0 (non-submission)",
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
        drop_zero_score=args.drop_zero_score,
        select_features=args.features,
    )

    X = feature_engineering.engineer_features(
        X, encoding=args.encoding, expected_codes=all_codes,
    )

    save_dataset(X, y, run_dir)
    save_correlation_matrix(X, y, run_dir)

    results_df, predictions_df = run_nested_cv(
        X=X, y=y, groups=groups,
        model_names=args.models,
        outer_splits=args.outer,
        inner_splits=args.inner,
    )

    print_summary(results_df, label="both  (exam + assignment)")
    save_predictions(predictions_df, run_dir)
    save_prediction_error_plot(predictions_df, run_dir)

    best_model_name = results_df["mean_rmse"].idxmin()
    save_prediction_error_plot(predictions_df, run_dir, model_name=best_model_name)
    fitted = train_final_model(X, y, groups, best_model_name, args.inner)
    save_model(fitted, run_dir / "final_model.pkl")
    save_feature_importance(fitted, list(X.columns), run_dir)

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
