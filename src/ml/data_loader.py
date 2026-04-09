"""Load the long-format base frame from the database.

Each row is one (student, assessment) pair.  The target is the raw
normalized_score for that specific assessment — not an aggregate.

Pipeline sizes (approximate, assuming 181 students):
    exam        : e1, e2, e3       → 181 × 3 = 543 rows
    assignment  : a2–a7            → 181 × 6 = 1086 rows  (a1 excluded)
    both        : exam + assignment → 1629 rows
"""
import pandas as pd
from sqlalchemy.engine import Engine

from src.ml.config import EXAM_ASSESSMENT_CODES, ASSIGNMENT_ASSESSMENT_CODES

PIPELINE_CODES: dict[str, list[str]] = {
    "exam":       EXAM_ASSESSMENT_CODES,
    "assignment": ASSIGNMENT_ASSESSMENT_CODES,
    "both":       EXAM_ASSESSMENT_CODES + ASSIGNMENT_ASSESSMENT_CODES,
}


def load_base_frame(engine: Engine, pipeline_type: str) -> pd.DataFrame:
    """Return the long-format base DataFrame.

    Parameters
    ----------
    engine        : SQLAlchemy engine connected to the studychat database.
    pipeline_type : "exam", "assignment", or "both".

    Returns
    -------
    DataFrame indexed by (user_id, assessment_id) with columns:
        semester_code      — metadata; not used as a feature
        assessment_code    — e.g. "e1", "a3"; consumed by feature_engineering
        assessment_kind    — "exam" or "assignment"
        linked_assignment_id — FK to assignments (NULL for exams)
        target             — normalized_score (0–1), never NaN

    Rows where normalized_score IS NULL are dropped (student not assessed).
    """
    if pipeline_type not in PIPELINE_CODES:
        raise ValueError(
            f"pipeline_type must be one of {list(PIPELINE_CODES)}, got '{pipeline_type}'"
        )

    codes = PIPELINE_CODES[pipeline_type]
    placeholders = ", ".join(f"'{c}'" for c in codes)

    query = f"""
        SELECT
            uas.user_id,
            uas.assessment_id,
            a.semester_code,
            a.assessment_code,
            a.assessment_kind,
            a.assignment_id        AS linked_assignment_id,
            uas.normalized_score   AS target
        FROM user_assessment_scores uas
        JOIN assessments a ON uas.assessment_id = a.assessment_id
        WHERE a.assessment_code IN ({placeholders})
          AND uas.normalized_score IS NOT NULL
        ORDER BY uas.user_id, a.assessment_code
    """
    df = pd.read_sql(query, engine)
    df = df.set_index(["user_id", "assessment_id"])

    n_students = df.index.get_level_values("user_id").nunique()
    print(
        f"[data_loader] pipeline='{pipeline_type}'  "
        f"rows={len(df)}  students={n_students}  "
        f"assessments={df['assessment_code'].nunique()}"
    )
    return df
