from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

SIM_CSV = Path(__file__).resolve().parents[2] / "data" / "final_similarity_results.csv"

_EXAM_TO_ASSIGNMENTS: dict[str, list[str]] = {
    "e1": ["a1", "a2"],
    "e2": ["a3", "a4"],
    "e3": ["a1", "a2", "a3", "a4", "a5", "a6", "a7"],
}


def build(engine: Engine) -> pd.DataFrame:
    """Return per-(user, assessment) similarity feature.

    Returns
    -------
    DataFrame indexed by (user_id, assessment_id) with one column:
        sim_max : float in [0, 1], or NaN when no source similarity exists.
    """
    if not SIM_CSV.exists():
        raise FileNotFoundError(f"Similarity CSV not found: {SIM_CSV}")

    sim = pd.read_csv(SIM_CSV)
    # Defensive: strip an upstream 'user_' prefix if present.
    sim["user_id"] = sim["user_id"].astype(str).str.replace("user_", "", regex=False)

    sim_lookup: dict[tuple[str, str], float] = (
        sim.set_index(["user_id", "assignment_id"])["max_similarity_score"].to_dict()
    )

    query = """
        SELECT
            uas.user_id,
            uas.assessment_id,
            a.semester_code,
            a.assessment_code,
            a.assessment_kind,
            a.assignment_id AS linked_assignment_id
        FROM user_assessment_scores uas
        JOIN assessments a ON uas.assessment_id = a.assessment_id
    """
    rows = pd.read_sql(query, engine)

    records: list[dict] = []
    for _, r in rows.iterrows():
        user_id = r["user_id"]
        if r["assessment_kind"] == "assignment":
            sim_val = sim_lookup.get((user_id, r["linked_assignment_id"]))
        else:
            assignments = _EXAM_TO_ASSIGNMENTS.get(r["assessment_code"], [])
            sims = []
            for code in assignments:
                key = (user_id, f"{r['semester_code']}_{code}")
                v = sim_lookup.get(key)
                if v is not None and not pd.isna(v):
                    sims.append(float(v))
            sim_val = sum(sims) / len(sims) if sims else None

        records.append({
            "user_id": user_id,
            "assessment_id": r["assessment_id"],
            "sim_max": sim_val,
        })

    df = pd.DataFrame(records).set_index(["user_id", "assessment_id"])
    n_nan = int(df["sim_max"].isna().sum())
    print(
        f"[similarity] built sim_max for {len(df)} rows  "
        f"({n_nan} NaN → zero-filled in data_preprocessing)"
    )
    return df
