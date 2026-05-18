import pandas as pd
from sqlalchemy.engine import Engine


def build(engine: Engine) -> pd.DataFrame:
    """Return per-assessment submission index (no feature columns currently used).

    The full query is kept for future use; for now only the index is returned
    with a placeholder column (sub_has_submission=1) so the DataFrame is joinable.

    One row per (user_id, assessment_id) via assessments.assignment_id link.
    Exam assessments have no linked assignment_id → NaN after join → 0-filled.
    """
    # fmt: off
    # Future features (not currently used):
    #   s.total_lines_of_code AS loc, s.file_count AS n_files,
    #   CASE WHEN s.has_notebook THEN 1 ELSE 0 END AS is_notebook,
    #   CASE WHEN s.has_python   THEN 1 ELSE 0 END AS is_python,
    #   CASE WHEN s.submitted_artifact_type = 'empty' THEN 1 ELSE 0 END AS is_empty
    # fmt: on

    query = """
        SELECT s.user_id, asmt.assessment_id
        FROM v_submission_overview s
        JOIN assessments asmt ON asmt.assignment_id = s.assignment_id
    """
    df = pd.read_sql(query, engine).set_index(["user_id", "assessment_id"])
    # df["sub_has_submission"] = 1
    return df



def build_submission_text_features(engine: Engine, user_ids: list) -> pd.DataFrame:

    ids_str = "', '".join(user_ids)
    
    query = f"""
        SELECT user_id, assignment_id, extracted_text AS submission_content
        FROM public.submission_files
        WHERE user_id IN ('{ids_str}')
          AND extracted_text IS NOT NULL;
    """
    df = pd.read_sql(query, engine)
    
    if not df.empty:
        # Strip prefix for internal alignment
        df['user_id'] = df['user_id'].str.replace('user_', '', regex=False)
        # Group and merge files
        df = df.groupby(["user_id", "assignment_id"])["submission_content"].apply(
            lambda x: "\n".join(x)
        ).to_frame()
    return df