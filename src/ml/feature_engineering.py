"""Feature engineering: encode assessment identity and finalise the feature matrix.

The `assessment_code` column (carried from data_preprocessing) is consumed here
and never exposed to the sklearn preprocessor.  All other columns pass through.

Encoding strategies
-------------------
ONE_HOT  : pd.get_dummies → binary column per assessment code.
           Columns are sorted and padded with zeros so that the set is stable
           across CV folds even if a fold is missing one assessment code.
ORDINAL  : single integer column `assess_ordinal` (lexicographic order within
           the set of codes present in X).
NONE     : assessment identity is dropped; model has no explicit knowledge of
           which exam/assignment it is predicting.
"""
import pandas as pd

from src.ml.config import AssessmentEncoding


def engineer_features(
    X: pd.DataFrame,
    encoding: AssessmentEncoding | str = AssessmentEncoding.ONE_HOT,
    expected_codes: list[str] | None = None,
) -> pd.DataFrame:
    """Encode the 'assessment_code' column and return the finalised feature matrix.

    Parameters
    ----------
    X             : Feature DataFrame containing 'assessment_code' (str) column.
    encoding      : How to represent assessment identity.
    expected_codes: Full list of assessment codes that may appear across all splits
                    (e.g. ["e1","e2","e3"] for exam, or all codes for "both").
                    Required for ONE_HOT to guarantee stable column sets across folds.
                    If None, inferred from the codes present in X.

    Returns
    -------
    X_out : float64 DataFrame with 'assessment_code' removed and encoding applied.
    """
    enc = AssessmentEncoding(encoding)
    X = X.copy()

    if enc == AssessmentEncoding.ONE_HOT:
        codes = expected_codes or sorted(X["assessment_code"].unique())
        dummies = pd.get_dummies(X["assessment_code"], prefix="assess", dtype=float)
        # Ensure all expected columns are present (pad missing ones with 0)
        for code in codes:
            col = f"assess_{code}"
            if col not in dummies.columns:
                dummies[col] = 0.0
        dummies = dummies[[f"assess_{c}" for c in sorted(codes)]]
        X = X.drop(columns=["assessment_code"])
        X = pd.concat([X, dummies], axis=1)

    elif enc == AssessmentEncoding.ORDINAL:
        codes = expected_codes or sorted(X["assessment_code"].unique())
        ordinal_map = {c: i for i, c in enumerate(codes)}
        X["assess_ordinal"] = X["assessment_code"].map(ordinal_map).astype(float)
        X = X.drop(columns=["assessment_code"])

    else:  # NONE
        X = X.drop(columns=["assessment_code"])

    return X.astype(float)
