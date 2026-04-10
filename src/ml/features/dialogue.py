"""Dialogue feature block.

Aggregates dialogue turns per (user_id, assessment_id):

  Assignment rows  — turns where dialogue_turns.assignment_id links to
                     the assessment's own assignment_id (via assessments table).
  Exam e1/e2 rows  — turns where dialogue_turns.exam_id equals the assessment_id.
  Exam e3 rows     — ALL assignment turns for the student in that semester
                     (a1–a7 combined), linked to the e3 assessment via
                     assessments.assessment_code = 'e3'.

Returned DataFrames are indexed by (user_id, assessment_id).
Assessment rows with no matching turns receive NaN → filled with 0 in
data_preprocessing.
"""
import pandas as pd
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# Counts / lengths
# ---------------------------------------------------------------------------

def build_counts_lengths(engine: Engine) -> pd.DataFrame:
    """Per-assessment dialogue counts and character lengths.

    Three branches:
      - Assignment assessments: turns linked via dialogue_turns.assignment_id.
      - Exam e1/e2: turns linked via dialogue_turns.exam_id.
      - Exam e3: all assignment turns for the student's semester.

    Indexed by (user_id, assessment_id).
    """
    query = """
        -- Assignment assessments: turns whose assignment_id links to this assessment
        SELECT
            c.user_id,
            asmt.assessment_id,
            COUNT(*)                                             AS n_turns,
            COUNT(DISTINCT dt.chat_id)                           AS n_chats,
            SUM(CHAR_LENGTH(dt.prompt))                          AS total_prompt_chars,
            AVG(CHAR_LENGTH(dt.prompt))::NUMERIC(10,2)           AS avg_prompt_chars,
            SUM(CHAR_LENGTH(COALESCE(dt.response, '')))          AS total_response_chars,
            AVG(CHAR_LENGTH(COALESCE(dt.response, '')))::NUMERIC(10,2)
                                                                 AS avg_response_chars
        FROM dialogue_turns dt
        JOIN chats c          ON dt.chat_id = c.chat_id
        JOIN assessments asmt ON asmt.assignment_id = dt.assignment_id
        GROUP BY c.user_id, asmt.assessment_id

        UNION ALL

        -- Exam e1/e2: turns linked via dialogue_turns.exam_id
        SELECT
            c.user_id,
            dt.exam_id                                           AS assessment_id,
            COUNT(*)                                             AS n_turns,
            COUNT(DISTINCT dt.chat_id)                           AS n_chats,
            SUM(CHAR_LENGTH(dt.prompt))                          AS total_prompt_chars,
            AVG(CHAR_LENGTH(dt.prompt))::NUMERIC(10,2)           AS avg_prompt_chars,
            SUM(CHAR_LENGTH(COALESCE(dt.response, '')))          AS total_response_chars,
            AVG(CHAR_LENGTH(COALESCE(dt.response, '')))::NUMERIC(10,2)
                                                                 AS avg_response_chars
        FROM dialogue_turns dt
        JOIN chats c ON dt.chat_id = c.chat_id
        WHERE dt.exam_id IS NOT NULL
        GROUP BY c.user_id, dt.exam_id

        UNION ALL

        -- Exam e3: all assignment turns for the student's semester (a1–a7)
        SELECT
            c.user_id,
            e3.assessment_id,
            COUNT(*)                                             AS n_turns,
            COUNT(DISTINCT dt.chat_id)                           AS n_chats,
            SUM(CHAR_LENGTH(dt.prompt))                          AS total_prompt_chars,
            AVG(CHAR_LENGTH(dt.prompt))::NUMERIC(10,2)           AS avg_prompt_chars,
            SUM(CHAR_LENGTH(COALESCE(dt.response, '')))          AS total_response_chars,
            AVG(CHAR_LENGTH(COALESCE(dt.response, '')))::NUMERIC(10,2)
                                                                 AS avg_response_chars
        FROM dialogue_turns dt
        JOIN chats c ON dt.chat_id = c.chat_id
        JOIN assessments e3 ON e3.semester_code = c.semester_code
                            AND e3.assessment_code = 'e3'
        WHERE dt.assignment_id IS NOT NULL
        GROUP BY c.user_id, e3.assessment_id
    """
    df = pd.read_sql(query, engine).set_index(["user_id", "assessment_id"])
    df.columns = [f"dlg_{c}" for c in df.columns]

    # print("Dialogue counts/lengths (per-assessment) - sample:")
    # print(df.columns.tolist())
    # print(df.head())

    return df


# ---------------------------------------------------------------------------
# Category distribution (llm_label counts + proportions)
# ---------------------------------------------------------------------------

def build_categories(engine: Engine) -> pd.DataFrame:
    """Per-assessment llm_label counts and proportions.

    Three branches (same logic as build_counts_lengths):
      - Assignment assessments via assignment_id.
      - Exam e1/e2 via exam_id.
      - Exam e3: all assignment turns for the student's semester.

    Returns columns:
      dlg_cat_<label>_count  — raw turn count per label
      dlg_cat_<label>_pct   — proportion of that label among all turns

    Indexed by (user_id, assessment_id).
    """
    query = """
        -- Assignment assessments
        SELECT
            c.user_id,
            asmt.assessment_id,
            dt.llm_label,
            COUNT(*) AS cnt
        FROM dialogue_turns dt
        JOIN chats c          ON dt.chat_id = c.chat_id
        JOIN assessments asmt ON asmt.assignment_id = dt.assignment_id
        WHERE dt.llm_label IS NOT NULL
        GROUP BY c.user_id, asmt.assessment_id, dt.llm_label

        UNION ALL

        -- Exam e1/e2
        SELECT
            c.user_id,
            dt.exam_id AS assessment_id,
            dt.llm_label,
            COUNT(*) AS cnt
        FROM dialogue_turns dt
        JOIN chats c ON dt.chat_id = c.chat_id
        WHERE dt.exam_id IS NOT NULL
          AND dt.llm_label IS NOT NULL
        GROUP BY c.user_id, dt.exam_id, dt.llm_label

        UNION ALL

        -- Exam e3: all assignment turns for the student's semester (a1–a7)
        SELECT
            c.user_id,
            e3.assessment_id,
            dt.llm_label,
            COUNT(*) AS cnt
        FROM dialogue_turns dt
        JOIN chats c ON dt.chat_id = c.chat_id
        JOIN assessments e3 ON e3.semester_code = c.semester_code
                            AND e3.assessment_code = 'e3'
        WHERE dt.assignment_id IS NOT NULL
          AND dt.llm_label IS NOT NULL
        GROUP BY c.user_id, e3.assessment_id, dt.llm_label
    """
    df = pd.read_sql(query, engine)

    # print("Raw dialogue category counts (per-assessment) - sample:")
    # print(df.head())

    if df.empty:
        return pd.DataFrame()

    # Sum counts across UNION branches (same user+assessment+label can appear in multiple legs)
    df = df.groupby(["user_id", "assessment_id", "llm_label"], as_index=False)["cnt"].sum()

    # Pivot to wide format: one column per label
    counts = df.pivot_table(
        index=["user_id", "assessment_id"],
        columns="llm_label",
        values="cnt",
        aggfunc="sum",
        fill_value=0,
    )

    # Compute proportions from the counts pivot
    row_totals = counts.sum(axis=1)
    pcts = counts.div(row_totals, axis=0)

    def _label(col: str) -> str:
        return str(col).lower().replace(" ", "_").replace("/", "_")

    counts.columns = [f"dlg_cat_{_label(c)}_count" for c in counts.columns]
    pcts.columns   = [f"dlg_cat_{_label(c)}_pct"   for c in pcts.columns]

    result = counts.join(pcts)
    result.index.names = ["user_id", "assessment_id"]

    # print("Dialogue categories (per-assessment) - sample:")
    # print(result.columns.tolist())
    # print(result.head())

    return result



def build_assignment_embedding_pairs(engine: Engine) -> pd.DataFrame:
    """
    Retrieves concatenated (Prompt + Response) pairs for the specific student.
    Matches the UUID format: '011bb520-7041-704f-b3e7-ab5c43dd3950'
    """
    # The literal ID for test
    target_user_uuid = '011bb520-7041-704f-b3e7-ab5c43dd3950'
    
    query = f"""
        SELECT
            c.user_id,
            dt.assignment_id,
            ARRAY_AGG(dt.prompt || ' ' || COALESCE(dt.response, '') ORDER BY dt.turn_timestamp) AS dialogue_pairs
        FROM dialogue_turns dt
        JOIN chats c ON dt.chat_id = c.chat_id
        WHERE c.user_id = '{target_user_uuid}'
          AND dt.assignment_id IS NOT NULL
        GROUP BY c.user_id, dt.assignment_id;
    """
    return pd.read_sql(query, engine).set_index(["user_id", "assignment_id"])