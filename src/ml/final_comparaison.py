import pandas as pd
from sqlalchemy.engine import Engine

from sqlalchemy.engine import Engine
from db import load_engine

def load_exam_scores(engine: Engine) -> pd.DataFrame:
    """
    Load exam scores from the database.

    Returns one row per student with columns:
        student_id
        semester
        e1
        e2
        e3
    """
    
    query = """
        SELECT
            uas.user_id,
            a.semester_code,
            a.assessment_code,
            uas.normalized_score AS target
        FROM user_assessment_scores uas
        JOIN assessments a
            ON uas.assessment_id = a.assessment_id
        WHERE a.assessment_code IN ('e1', 'e2', 'e3')
          AND uas.normalized_score IS NOT NULL
    """
    
    # ------------------------------------------------------------------
    # LOAD LONG FORMAT
    # ------------------------------------------------------------------

    df = pd.read_sql(query, engine)

    # ------------------------------------------------------------------
    # CONVERT TO WIDE FORMAT
    # ------------------------------------------------------------------

    df = (
        df.pivot(
            index=["user_id", "semester_code"],
            columns="assessment_code",
            values="target"
        )
        .reset_index()
        .rename(columns={
            "user_id": "student_id",
            "semester_code": "semester"
        })
    )

    # Optional: remove pandas column index name
    #df.columns.name = None

    return df


if __name__ == "__main__":

    engine = load_engine()

    scores_df = load_exam_scores(engine)

    #scores_df = pd.read_csv("exam_scores.csv")
    # Similarity dataframe
    # Columns expected:
    # user_id | assignment_id | max_similarity_score

    sim_df = pd.read_csv(
        "final_similarity_results.csv",
        header=None,
        names=["user_id", "assignment_id", "max_similarity_score"]
    )

    sim_df["max_similarity_score"] = pd.to_numeric(
        sim_df["max_similarity_score"],
    errors="coerce")

    # ------------------------------------------------------------------
    # EXTRACT ASSIGNMENT CODE
    # ------------------------------------------------------------------

    # Example:
    # f24_a2 -> a2
    # s25_a4 -> a4

    sim_df["assignment_code"] = sim_df["assignment_id"].str.extract(r"(a\d+)")

    # ------------------------------------------------------------------
    # COMPUTE FEATURES
    # ------------------------------------------------------------------

    features = []

    for user_id, group in sim_df.groupby("user_id"):

        row = {"student_id": user_id}

        # --------------------------------------------------------------
        # E1 -> a2 only
        # --------------------------------------------------------------

        a2 = group[group["assignment_code"] == "a2"]

        row["e1_similarity"] = (
            a2["max_similarity_score"].iloc[0]
            if not a2.empty
            else None
        )

        # --------------------------------------------------------------
        # E2 -> a3 + a4
        # --------------------------------------------------------------

        e2_group = group[group["assignment_code"].isin(["a3", "a4"])]

        row["e2_mean_similarity"] = (
            e2_group["max_similarity_score"].mean()
        )

        row["e2_participation_ratio"] = (
            e2_group["assignment_code"].nunique() / 2
        )

        # --------------------------------------------------------------
        # E3 -> a2 to a7
        # --------------------------------------------------------------

        e3_assignments = ["a2", "a3", "a4", "a5", "a6", "a7"]

        e3_group = group[group["assignment_code"].isin(e3_assignments)]

        row["e3_mean_similarity"] = (
            e3_group["max_similarity_score"].mean()
        )

        row["e3_participation_ratio"] = (
            e3_group["assignment_code"].nunique() / 6
        )

        features.append(row)

    # ------------------------------------------------------------------
    # CREATE FEATURE DATAFRAME
    # ------------------------------------------------------------------

    features_df = pd.DataFrame(features)

    # ------------------------------------------------------------------
    # MERGE WITH EXAM SCORES
    # ------------------------------------------------------------------

    final_df = scores_df.merge(
        features_df,
        on="student_id",
        how="inner"   # keep only LLM users
    )

    # ------------------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------------------

    print(final_df.head())

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    final_df.to_csv("final_exam_similarity_dataset.csv", index=False)
