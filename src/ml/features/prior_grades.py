"""Prior grade feature: running average of assessment scores preceding each row.

For each (user_id, assessment_id) row the feature is:

    grade_prior_avg = mean of the student's normalized_score on all assessments
                      that are temporally earlier than this one, within the same
                      semester.

The temporal order is defined in config.ASSESSMENT_ORDER / PRIOR_ASSESSMENTS.

NaN semantics
-------------
- a1 has no prior assessments  → grade_prior_avg = NaN.
- A prior assessment where the student has no recorded score (NULL in DB) is
  excluded from the average (only available scores are averaged).
- NaN values are left as-is and handled by SimpleImputer(strategy="median")
  inside the sklearn Pipeline — never zero-filled.
"""
import pandas as pd
from sqlalchemy.engine import Engine

from src.ml.config import PRIOR_ASSESSMENTS


def build(engine: Engine) -> pd.DataFrame:
    """Compute the running prior-score average for every (user_id, assessment_id).

    Returns
    -------
    DataFrame indexed by (user_id, assessment_id) with one column:
        grade_prior_avg : float in [0, 1], or NaN if no prior scores exist.
    """
    query = """
        SELECT
            uas.user_id,
            uas.assessment_id,
            a.semester_code,
            a.assessment_code,
            uas.normalized_score
        FROM user_assessment_scores uas
        JOIN assessments a ON uas.assessment_id = a.assessment_id
        WHERE uas.normalized_score IS NOT NULL
    """
    all_scores = pd.read_sql(query, engine)

    # Fast lookup: (user_id, semester_code, assessment_code) → normalized_score
    score_lookup: dict[tuple, float] = (
        all_scores
        .set_index(["user_id", "semester_code", "assessment_code"])["normalized_score"]
        .to_dict()
    )

    records = []
    for _, row in all_scores.iterrows():
        prior_codes = PRIOR_ASSESSMENTS.get(row["assessment_code"], [])
        prior_scores = [
            score_lookup.get((row["user_id"], row["semester_code"], code))
            for code in prior_codes
        ]
        prior_scores = [s for s in prior_scores if s is not None]
        prior_avg = float(sum(prior_scores) / len(prior_scores)) if prior_scores else None
        records.append({
            "user_id":        row["user_id"],
            "assessment_id":  row["assessment_id"],
            "grade_prior_avg": prior_avg,
        })

    print(
        f"[prior_grades] built grade_prior_avg for {len(records)} rows  "
        f"({sum(1 for r in records if r['grade_prior_avg'] is None)} NaN — no prior scores)"
    )

    return pd.DataFrame(records).set_index(["user_id", "assessment_id"])
