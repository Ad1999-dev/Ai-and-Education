"""Assemble the design matrix X, target vector y, and group labels.

The base frame (from data_loader) is the spine: one row per (student, assessment).
All feature blocks are always built and joined; use select_features to restrict
which columns reach the model.

NaN semantics
-------------
- dlg_* / sub_* columns : filled with 0 after all joins (zero activity, not missing).

Grade scores (a1–a7, e1–e3) are the prediction targets only and must never
appear as input features.
"""
import pandas as pd
from sqlalchemy.engine import Engine

from src.ml.features import dialogue, submission

_ZERO_FILL_PREFIXES = ("dlg_", "sub_")


def build_dataset(
    engine: Engine,
    base: pd.DataFrame,
    pipeline_type: str,
    drop_no_dialogue: bool = False,
    drop_zero_score: bool = False,
    select_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Merge all feature blocks onto the base frame.

    Parameters
    ----------
    engine           : SQLAlchemy engine.
    base             : Output of data_loader.load_base_frame(); index=(user_id, assessment_id).
    pipeline_type    : "exam", "assignment", or "both".
    drop_no_dialogue : If True, remove rows where the student had zero dialogue
                       activity during that assessment period (all dlg_* == 0).
    drop_zero_score  : If True, remove rows where the student's score is exactly 0
                       (i.e. did not submit / received no credit).  Distinct from
                       NULL rows, which are already excluded by data_loader.
    select_features  : If provided, keep only these columns from X (plus the
                       structural columns assessment_code / is_exam which are
                       always kept).  Raises ValueError for unknown column names.
                       Pass None to keep all columns (default).

    Returns
    -------
    X      : feature DataFrame, index=(user_id, assessment_id).
             Contains 'assessment_code' column (consumed by feature_engineering).
    y      : target Series (normalized_score 0–1), same index.
    groups : Series of user_id strings for GroupKFold.
    """
    # ------------------------------------------------------------------
    # Build and join all feature blocks (all indexed by (user_id, assessment_id))
    # ------------------------------------------------------------------
    X: pd.DataFrame = base[[]].copy()

    dlg_cl = dialogue.build_counts_lengths(engine)
    X = X.join(dlg_cl, how="left")

    dlg_cat = dialogue.build_categories(engine)
    if not dlg_cat.empty:
        X = X.join(dlg_cat, how="left")

    sub_df = submission.build(engine)
    X = X.join(sub_df, how="left")

    # Debugging - save as CSV for manual inspection
    X.to_csv("debug_X_pre_feature_engineering.csv")

    # ------------------------------------------------------------------
    # Zero-fill dialogue and submission columns (zero = no activity)
    # ------------------------------------------------------------------
    zero_cols = [
        c for c in X.columns
        if any(c.startswith(p) for p in _ZERO_FILL_PREFIXES)
    ]
    X[zero_cols] = X[zero_cols].fillna(0)

    # Debugging - check for any remaining NaNs
    X.to_csv("debug_X_after_zero_fill.csv")

    # ------------------------------------------------------------------
    # Optional: drop rows where the student received a score of exactly 0
    # (non-submission recorded as 0, not NULL)
    # ------------------------------------------------------------------
    if drop_zero_score:
        nonzero_mask = base["target"] != 0
        n_dropped = int((~nonzero_mask).sum())
        X = X[nonzero_mask]
        base = base.loc[nonzero_mask]
        print(f"[data_preprocessing] drop_zero_score: removed {n_dropped} rows with score == 0")

    # ------------------------------------------------------------------
    # Optional: drop rows with no dialogue activity during the assessment
    # ------------------------------------------------------------------
    if drop_no_dialogue:
        dlg_cols = [c for c in X.columns if c.startswith("dlg_")]
        if dlg_cols:
            active_mask = (X[dlg_cols] != 0).any(axis=1)
            n_dropped = int((~active_mask).sum())
            X = X[active_mask]
            base = base.loc[active_mask]
            print(f"[data_preprocessing] drop_no_dialogue: removed {n_dropped} rows with zero dialogue activity")

    # ------------------------------------------------------------------
    # Carry assessment_code forward (consumed by feature_engineering)
    # ------------------------------------------------------------------
    X["assessment_code"] = base["assessment_code"]

    # For "both" pipeline, add a binary is_exam indicator
    if pipeline_type == "both":
        X["is_exam"] = (base["assessment_kind"] == "exam").astype(float)

    # Convert numeric columns to float; assessment_code stays as str
    numeric_cols = [c for c in X.columns if c != "assessment_code"]
    X[numeric_cols] = X[numeric_cols].astype(float)

    # ------------------------------------------------------------------
    # Optional: keep only a specific subset of feature columns
    # ------------------------------------------------------------------
    if select_features is not None:
        _STRUCTURAL = {"assessment_code", "is_exam"}
        unknown = set(select_features) - set(X.columns)
        if unknown:
            raise ValueError(
                f"[data_preprocessing] select_features: unknown column(s) {sorted(unknown)}.\n"
                f"Available: {sorted(X.columns)}"
            )
        keep = [c for c in X.columns if c in _STRUCTURAL] + [
            c for c in select_features if c not in _STRUCTURAL
        ]
        X = X[keep]
        print(f"[data_preprocessing] select_features: keeping {len(select_features)} column(s): {select_features}")

    y = base["target"].astype(float)
    groups = pd.Series(
        X.index.get_level_values("user_id"),
        index=X.index,
        name="user_id",
    )

    print(
        f"[data_preprocessing] pipeline='{pipeline_type}'  "
        f"rows={len(X)}  features={X.shape[1]}  "
        f"students={groups.nunique()}"
    )
    return X, y, groups
