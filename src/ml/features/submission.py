"""Submission feature block.

Stats for the single submission linked to each assessment's assignment_id.
Returned DataFrame is indexed by (user_id, assessment_id).
Exam assessments have no linked assignment_id → NaN after join → 0-filled.
"""
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


import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

def build_submission_text_features(engine: Engine) -> pd.DataFrame:
    """
    MVP: Uses the exact hardcoded query verified in pgAdmin.
    """
    # The literal ID for test
    raw_id = 'user_011bb520-7041-704f-b3e7-ab5c43dd3950'
    clean_id = '011bb520-7041-704f-b3e7-ab5c43dd3950'
    
    query = text(f"""
        SELECT 
            user_id, 
            assignment_id, 
            extracted_text AS submission_content 
        FROM public.submission_files
        WHERE user_id = '{raw_id}'
          AND extracted_text IS NOT NULL;
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    if not df.empty:
        # Normalize the ID in Python so it matches the dialogue DataFrame index
        df['user_id'] = clean_id
        
        # Combine multiple files 
        df = df.groupby(["user_id", "assignment_id"])["submission_content"].apply(
            lambda x: "\n".join(x)
        ).to_frame()
    
    return df