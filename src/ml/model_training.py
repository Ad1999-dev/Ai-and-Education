import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for subprocess runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml import config


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    """Median imputation followed by z-score scaling for all numeric columns."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_cols)],
        remainder="drop",
    )


def build_model_pipeline(model, numeric_cols: list[str]) -> Pipeline:
    """Wrap preprocessor + model in a single sklearn Pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor(numeric_cols)),
        ("model",        model),
    ])


# ---------------------------------------------------------------------------
# Inner fold training
# ---------------------------------------------------------------------------

def train_fold(
    model,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    inner_splits: int,
) -> GridSearchCV:
    """Run the inner hyperparameter search for one outer fold.

    Returns a fitted GridSearchCV (call .predict() for test-fold scoring).
    """
    pipe = build_model_pipeline(model, list(X_train.columns))
    inner_cv = GroupKFold(n_splits=inner_splits)
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=inner_cv,
        scoring=config.SCORING,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups_train)
    return search


# ---------------------------------------------------------------------------
# Nested CV — outer loop + RMSE collection
# ---------------------------------------------------------------------------

def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_names: list[str] | None = None,
    outer_splits: int = config.OUTER_CV_SPLITS,
    inner_splits: int = config.INNER_CV_SPLITS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute nested GroupKFold cross-validation for all (or selected) models.

    Parameters
    ----------
    X, y, groups  : Feature matrix, targets, and group labels (user_id).
    model_names   : Subset of config.MODEL_REGISTRY keys; None = all.
    outer_splits  : Outer GroupKFold folds (unbiased performance estimate).
    inner_splits  : Inner GroupKFold folds (hyperparameter selection).

    Returns
    -------
    results_df : DataFrame indexed by model name, sorted ascending by mean_rmse,
        with columns: mean_rmse, std_rmse, min_rmse, max_rmse, fold_1_rmse, ...
    predictions_df : Long-format DataFrame of out-of-fold predictions with columns:
        user_id, assessment_id, model, fold, y_true, y_pred
    """
    registry = config.MODEL_REGISTRY
    if model_names:
        unknown = set(model_names) - set(registry)
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}. Available: {list(registry)}")
        registry = {k: v for k, v in registry.items() if k in model_names}

    y_arr      = y.values
    groups_arr = groups.values
    outer_cv   = GroupKFold(n_splits=outer_splits)
    records: list[dict] = []
    pred_records: list[dict] = []

    for name, (model, param_grid) in registry.items():
        print(f"\n[{name}] Running nested CV ({outer_splits}×{inner_splits}) ...")
        fold_rmses: list[float] = []

        for fold_num, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, y_arr, groups_arr), start=1
        ):
            fitted  = train_fold(model, param_grid,
                                 X.iloc[train_idx], y_arr[train_idx],
                                 groups_arr[train_idx], inner_splits)
            y_pred  = fitted.predict(X.iloc[test_idx])
            fold_rmses.append(root_mean_squared_error(y_arr[test_idx], y_pred))

            # Collect per-row predictions for this fold
            test_index = X.index[test_idx]
            for idx, y_t, y_p in zip(test_index, y_arr[test_idx], y_pred):
                if isinstance(idx, tuple):
                    user_id, assessment_id = idx[0], idx[1]
                else:
                    user_id, assessment_id = idx, None
                pred_records.append({
                    "model":         name,
                    "fold":          fold_num,
                    "user_id":       user_id,
                    "assessment_id": assessment_id,
                    "y_true":        float(y_t),
                    "y_pred":        float(y_p),
                })

        scores = np.array(fold_rmses)
        record: dict = {
            "model":     name,
            "mean_rmse": scores.mean(),
            "std_rmse":  scores.std(),
            "min_rmse":  scores.min(),
            "max_rmse":  scores.max(),
        }
        for i, s in enumerate(scores):
            record[f"fold_{i+1}_rmse"] = s
        records.append(record)
        print(f"  RMSE = {scores.mean():.4f} ± {scores.std():.4f}  "
              f"[{scores.min():.4f} – {scores.max():.4f}]")

    results_df = (
        pd.DataFrame(records)
        .set_index("model")
        .sort_values("mean_rmse")
    )
    predictions_df = pd.DataFrame(
        pred_records,
        columns=["model", "fold", "user_id", "assessment_id", "y_true", "y_pred"],
    )
    return results_df, predictions_df


# ---------------------------------------------------------------------------
# Results display and persistence
# ---------------------------------------------------------------------------

def make_run_dir(base_output: Path, pipeline_type: str) -> Path:
    """Create and return a timestamped run subdirectory.

    Pattern: <base_output>/<pipeline_type>_YYYYMMDD_HHMMSS/
    Example: results/exam_20260405_143022/
    """
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_output) / f"{pipeline_type}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def print_summary(results_df: pd.DataFrame, label: str) -> None:
    """Print a formatted RMSE summary table with best model highlighted."""
    print(f"\n{'='*60}")
    print(f"Nested CV results  |  {label}")
    print(f"{'='*60}")
    summary_cols = ["mean_rmse", "std_rmse", "min_rmse", "max_rmse"]
    print(results_df[summary_cols].to_string(float_format="%.4f"))
    best = results_df["mean_rmse"].idxmin()
    print(f"\nBest model: {best}  (RMSE = {results_df.loc[best, 'mean_rmse']:.4f})")


def save_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    *,
    pipeline_type: str,
    assessment_codes: list[str],
    outer_splits: int,
    inner_splits: int,
    n_samples: int,
    n_features: int,
    n_students: int,
    feature_columns: list[str],
    assessment_encoding: str,
    best_model_params: dict | None = None,
) -> tuple[Path, Path]:
    """Write results.csv and run_summary.json to output_dir.

    Parameters
    ----------
    results_df          : DataFrame from run_nested_cv() (index=model name).
    output_dir          : Timestamped run directory (from make_run_dir).
    pipeline_type       : "exam", "assignment", or "both".
    assessment_codes    : Which assessment codes were included in the dataset.
    feature_columns     : Actual column names in X after feature engineering.
    best_model_params   : Best hyperparameters from the final model (optional).

    Returns (csv_path, json_path).
    """
    from datetime import datetime

    output_dir = Path(output_dir)
    csv_path  = output_dir / "results.csv"
    json_path = output_dir / "run_summary.json"

    results_df.to_csv(csv_path)

    best_name  = results_df["mean_rmse"].idxmin()
    best_rmse  = float(results_df.loc[best_name, "mean_rmse"])
    best_std   = float(results_df.loc[best_name, "std_rmse"])

    payload = {
        "run_at":    datetime.now().isoformat(timespec="seconds"),
        "pipeline":  pipeline_type,
        "target":    "normalized_score (0–1, per student × assessment)",
        "assessment_codes": assessment_codes,
        "dataset": {
            "n_samples":  n_samples,
            "n_features": n_features,
            "n_students": n_students,
        },
        "cv": {
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "group_by":     "user_id",
            "scoring":      "RMSE",
        },
        "strategies": {
            "assessment_encoding": assessment_encoding,
        },
        "feature_columns": feature_columns,
        "best_model": {
            "name":      best_name,
            "mean_rmse": best_rmse,
            "std_rmse":  best_std,
            "params":    best_model_params or {},
        },
        "model_results": results_df.reset_index().to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"Results saved → {csv_path}")
    print(f"Run summary  → {json_path}")
    return csv_path, json_path


# ---------------------------------------------------------------------------
# Final model training (full dataset, no outer held-out fold)
# ---------------------------------------------------------------------------

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_name: str,
    inner_splits: int = config.INNER_CV_SPLITS,
) -> GridSearchCV:
    """Fit the chosen model on the entire dataset with inner-CV hyperparam search.

    Called after run_nested_cv() has identified the best model name.
    Returns a fitted GridSearchCV (has .best_estimator_ and .best_params_).
    """
    if model_name not in config.MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(config.MODEL_REGISTRY)}"
        )
    model, param_grid = config.MODEL_REGISTRY[model_name]
    print(f"[train_final_model] Training '{model_name}' on {len(X)} rows ...")
    fitted = train_fold(model, param_grid, X, y.values, groups.values, inner_splits)
    print(f"  Best params: {fitted.best_params_}")
    return fitted


def save_dataset(X: pd.DataFrame, y: pd.Series, output_dir: Path) -> Path:
    """Save the training dataset (features + target) to <output_dir>/dataset.csv.

    The file has one row per (user_id, assessment_id) — the exact data passed
    to the model — with all feature columns followed by a `target` column.
    Useful for inspecting what went into a particular run.
    """
    output_dir = Path(output_dir)
    path = output_dir / "dataset.csv"
    df = X.copy()
    df["target"] = y
    df.to_csv(path)
    print(f"Dataset      → {path}  ({len(df)} rows × {X.shape[1]} features)")
    return path


def save_predictions(predictions_df: pd.DataFrame, output_dir: Path) -> Path:
    """Save out-of-fold predictions to <output_dir>/predictions.csv.

    Columns: model, fold, user_id, assessment_id, y_true, y_pred.
    One row per (model, fold, sample) — all models interleaved in one file.
    """
    output_dir = Path(output_dir)
    path = output_dir / "predictions.csv"
    predictions_df.to_csv(path, index=False)
    print(f"Predictions  → {path}  ({len(predictions_df)} rows)")
    return path


def _draw_prediction_error_ax(
    ax: plt.Axes,
    sub: pd.DataFrame,
    title: str,
    fold_colors: dict,
) -> None:
    """Draw a single prediction-error panel onto ax."""
    rmse = np.sqrt(((sub["y_true"] - sub["y_pred"]) ** 2).mean())
    for fold, group in sub.groupby("fold"):
        ax.scatter(
            group["y_true"], group["y_pred"],
            color=fold_colors[fold], s=18, alpha=0.7,
            label=f"fold {fold}",
        )
    lim_min = min(sub["y_true"].min(), sub["y_pred"].min()) - 0.02
    lim_max = max(sub["y_true"].max(), sub["y_pred"].max()) + 0.02
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color="black", linewidth=1, linestyle="--", label="perfect")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Actual score")
    ax.set_ylabel("Predicted score")
    ax.set_title(title)
    ax.text(
        0.04, 0.95, f"RMSE = {rmse:.4f}",
        transform=ax.transAxes, fontsize=8,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
    )
    ax.legend(fontsize=6, loc="lower right")


def save_prediction_error_plot(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    model_name: str | None = None,
) -> Path:
    """Plot actual vs predicted (prediction error plot).

    Uses out-of-fold predictions from run_nested_cv() — no train-set optimism.

    Parameters
    ----------
    model_name : If provided, plot only that model in a single large panel
                 and save as prediction_error_best.png.
                 If None, plot all models in a grid and save as prediction_error.png.
    """
    output_dir = Path(output_dir)

    folds = sorted(predictions_df["fold"].unique())
    cmap = plt.cm.get_cmap("tab10", len(folds))
    fold_colors = {f: cmap(i) for i, f in enumerate(folds)}

    if model_name is not None:
        sub = predictions_df[predictions_df["model"] == model_name]
        fig, ax = plt.subplots(figsize=(6, 5))
        _draw_prediction_error_ax(ax, sub, model_name, fold_colors)
        fig.suptitle("Prediction error plot  (best model, out-of-fold)", fontsize=11)
        fig.tight_layout()
        png_path = output_dir / "prediction_error_best.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Pred. error (best) → {png_path}")
        return png_path

    models = predictions_df["model"].unique()
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    for ax, name in zip(axes.flat, models):
        sub = predictions_df[predictions_df["model"] == name]
        _draw_prediction_error_ax(ax, sub, name, fold_colors)
    for ax in axes.flat[n_models:]:
        ax.set_visible(False)
    fig.suptitle("Prediction error plot  (out-of-fold)", fontsize=11, y=1.01)
    fig.tight_layout()
    png_path = output_dir / "prediction_error.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pred. error  → {png_path}")
    return png_path


def save_model(fitted_model, path: Path) -> Path:
    """Serialize a fitted model to disk with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_model, path)
    print(f"Final model  → {path}")
    return path


def load_model(path: Path):
    """Load a previously saved model."""
    return joblib.load(Path(path))


def save_correlation_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> Path:
    """Compute and save the Pearson correlation matrix of features + target.

    Files written:
      <output_dir>/correlation_matrix.csv
      <output_dir>/correlation_heatmap.png

    Only numeric columns in X are included (assessment_code is dropped).
    The target column is appended as 'target' so feature-to-grade correlations
    are visible in the same matrix.

    Returns the CSV path.
    """
    output_dir = Path(output_dir)

    numeric_cols = [c for c in X.columns if c != "assessment_code"]
    df = X[numeric_cols].copy()
    df["target"] = y.values

    corr = df.corr()

    csv_path = output_dir / "correlation_matrix.csv"
    corr.to_csv(csv_path)
    print(f"Corr. matrix → {csv_path}")

    n = len(corr)
    fig_size = max(8, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    # Lower-triangle mask: upper triangle + diagonal set to NaN
    plot_values = corr.values.astype(float).copy()
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=0)
    plot_values[upper_mask] = np.nan

    # RdBu: negative → red, positive → blue
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(plot_values, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)

    # Annotate every lower-triangle cell with its value
    font_size = max(5, 8 - n // 5)
    for i in range(n):
        for j in range(i):  # strictly lower triangle
            val = corr.iloc[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", fontsize=font_size,
                color="white" if abs(val) > 0.7 else "black",
            )

    ax.set_title("Pearson correlation matrix  (features + target)")
    fig.tight_layout()
    png_path = output_dir / "correlation_heatmap.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Corr. heatmap → {png_path}")

    return csv_path


def save_feature_importance(
    fitted_search: GridSearchCV,
    feature_names: list[str],
    output_dir: Path,
) -> Path | None:
    """Extract feature importance from the best estimator and save CSV + bar chart.

    Supports:
      - Tree models (RandomForest, GradientBoosting): feature_importances_
      - Linear models (Ridge, ElasticNet): abs(coef_), labelled as |coefficient|
      - SVR with linear kernel: abs(coef_[0])
      - SVR with rbf kernel: not supported — skipped gracefully

    Files written:
      <output_dir>/feature_importance.csv
      <output_dir>/feature_importance.png

    Returns the CSV path, or None if the model does not support importance extraction.
    """
    output_dir = Path(output_dir)
    best_pipe = fitted_search.best_estimator_
    model_step = best_pipe.named_steps["model"]

    # Extract raw importance values
    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
        importance_label = "importance"
    elif hasattr(model_step, "coef_"):
        coef = model_step.coef_
        importances = np.abs(coef.ravel())
        importance_label = "|coefficient|"
    else:
        print("Feature importance: not available for this model (skipped).")
        return None

    if len(importances) != len(feature_names):
        print(
            f"Feature importance: shape mismatch "
            f"({len(importances)} importances vs {len(feature_names)} features — skipped)."
        )
        return None

    # Build and save DataFrame sorted descending
    fi_df = (
        pd.DataFrame({"feature": feature_names, importance_label: importances})
        .sort_values(importance_label, ascending=False)
        .reset_index(drop=True)
    )
    csv_path = output_dir / "feature_importance.csv"
    fi_df.to_csv(csv_path, index=False)
    print(f"Feature imp. → {csv_path}")

    # Horizontal bar chart (sorted ascending so most important is at the top)
    plot_df = fi_df.sort_values(importance_label, ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.35)))
    ax.barh(plot_df["feature"], plot_df[importance_label], color="steelblue")
    ax.set_xlabel(importance_label)
    ax.set_title(f"Feature importance  ({type(model_step).__name__})")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    png_path = output_dir / "feature_importance.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Feature plot → {png_path}")

    return csv_path


# ---------------------------------------------------------------------------
# Persistent run log
# ---------------------------------------------------------------------------

def append_run_log(
    base_output: Path,
    run_dir: Path,
    results_df: pd.DataFrame,
    *,
    pipeline_type: str,
    assessment_codes: list[str],
    outer_splits: int,
    inner_splits: int,
    n_samples: int,
    n_features: int,
    n_students: int,
    assessment_encoding: str,
    best_model_params: dict,
) -> Path:
    """Append one summary row to <base_output>/runs_log.csv.

    The log is append-only: each run adds exactly one row.  All models that
    were evaluated get their own `rmse_<name>` and `std_<name>` columns;
    models not run in a given experiment are left as NaN.

    Returns the path of the log file.
    """
    from datetime import datetime

    log_path = Path(base_output) / "runs_log.csv"

    best_name = results_df["mean_rmse"].idxmin()
    best_row  = results_df.loc[best_name]

    row: dict = {
        "run_at":               datetime.now().isoformat(timespec="seconds"),
        "run_dir":              str(run_dir),
        "pipeline":             pipeline_type,
        "assessment_codes":     ",".join(assessment_codes),
        "n_samples":            n_samples,
        "n_students":           n_students,
        "n_features":           n_features,
        "outer_folds":          outer_splits,
        "inner_folds":          inner_splits,
        "assessment_encoding":  assessment_encoding,
        "best_model":           best_name,
        "best_rmse":            round(float(best_row["mean_rmse"]), 6),
        "best_rmse_std":        round(float(best_row["std_rmse"]),  6),
        "best_params":          json.dumps(best_model_params, default=str),
    }

    # One column per model for easy comparison across runs
    for model_name, model_row in results_df.iterrows():
        row[f"rmse_{model_name}"] = round(float(model_row["mean_rmse"]), 6)
        row[f"std_{model_name}"]  = round(float(model_row["std_rmse"]),  6)

    new_row_df = pd.DataFrame([row])

    if log_path.exists():
        existing = pd.read_csv(log_path)
        # Align columns: add any new model columns that didn't exist before
        updated = pd.concat([existing, new_row_df], ignore_index=True, sort=False)
    else:
        updated = new_row_df

    updated.to_csv(log_path, index=False)
    print(f"Run log      → {log_path}  ({len(updated)} total runs)")
    return log_path
