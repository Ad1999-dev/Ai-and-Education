import argparse
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "grades" / "s25_grades_released_normalized.csv"


def load_engine():
    load_dotenv(ENV_PATH)
    import os

    db_user = os.getenv("DB_USER", "studychat_user")
    db_password = os.getenv("DB_PASSWORD", "studychat_password")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "studychat")

    url = f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(url, future=True, pool_pre_ping=True)


def chunked(items: List[Dict], size: int) -> Iterable[List[Dict]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def safe_null(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def parse_arguments():
    parser = argparse.ArgumentParser(description="Ingest normalized Spring 2025 StudyChat grades.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_arguments()
    csv_path = args.csv_path
    batch_size = args.batch_size

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {
        "userId",
        "directory_name",
        "a1", "a2", "a3", "a4", "a5", "a6", "a7",
        "e1", "e2", "e3",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    score_columns = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "e1", "e2", "e3"]
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    engine = load_engine()

    semester_sql = text(
        """
        INSERT INTO semesters (semester_code, semester_name)
        VALUES ('s25', 'Spring 2025')
        ON CONFLICT (semester_code) DO NOTHING
        """
    )

    users_sql = text(
        """
        INSERT INTO users (user_id)
        VALUES (:user_id)
        ON CONFLICT (user_id) DO NOTHING
        """
    )

    assignments_sql = text(
        """
        INSERT INTO assignments (assignment_id, semester_code, assignment_code, title, description, folder_path)
        VALUES (:assignment_id, 's25', :assignment_code, NULL, NULL, NULL)
        ON CONFLICT (assignment_id) DO NOTHING
        """
    )

    assessments_sql = text(
        """
        INSERT INTO assessments (
            assessment_id,
            semester_code,
            assessment_code,
            assessment_kind,
            max_points,
            assignment_id
        )
        VALUES (
            :assessment_id,
            's25',
            :assessment_code,
            :assessment_kind,
            :max_points,
            :assignment_id
        )
        ON CONFLICT (assessment_id) DO UPDATE
        SET
            assessment_code = EXCLUDED.assessment_code,
            assessment_kind = EXCLUDED.assessment_kind,
            max_points = EXCLUDED.max_points,
            assignment_id = EXCLUDED.assignment_id
        """
    )

    grade_profile_sql = text(
        """
        INSERT INTO user_grade_profiles (user_id, semester_code, directory_name)
        VALUES (:user_id, 's25', :directory_name)
        ON CONFLICT (user_id, semester_code) DO UPDATE
        SET directory_name = EXCLUDED.directory_name
        """
    )

    score_sql = text(
        """
        INSERT INTO user_assessment_scores (user_id, assessment_id, normalized_score)
        VALUES (:user_id, :assessment_id, :normalized_score)
        ON CONFLICT (user_id, assessment_id) DO UPDATE
        SET normalized_score = EXCLUDED.normalized_score
        """
    )

    assessment_specs = {
        "a1": {"kind": "assignment", "max_points": 31, "assignment_id": "s25_a1"},
        "a2": {"kind": "assignment", "max_points": 100, "assignment_id": "s25_a2"},
        "a3": {"kind": "assignment", "max_points": 55, "assignment_id": "s25_a3"},
        "a4": {"kind": "assignment", "max_points": 90, "assignment_id": "s25_a4"},
        "a5": {"kind": "assignment", "max_points": 105, "assignment_id": "s25_a5"},
        "a6": {"kind": "assignment", "max_points": 113, "assignment_id": "s25_a6"},
        "a7": {"kind": "assignment", "max_points": 135, "assignment_id": "s25_a7"},
        "e1": {"kind": "exam", "max_points": 100, "assignment_id": None},
        "e2": {"kind": "exam", "max_points": 100, "assignment_id": None},
        "e3": {"kind": "exam", "max_points": 100, "assignment_id": None},
    }

    user_records = []
    grade_profiles = []
    score_records = []

    for _, row in df.iterrows():
        user_id = safe_null(row["userId"])
        if user_id is None:
            continue

        user_id = str(user_id)
        user_records.append({"user_id": user_id})

        grade_profiles.append(
            {
                "user_id": user_id,
                "directory_name": safe_null(row["directory_name"]),
            }
        )

        for code in score_columns:
            score_value = safe_null(row[code])
            score_records.append(
                {
                    "user_id": user_id,
                    "assessment_id": f"s25_{code}",
                    "normalized_score": score_value,
                }
            )

    assignment_records = []
    for code, spec in assessment_specs.items():
        if spec["assignment_id"] is not None:
            assignment_records.append(
                {
                    "assignment_id": spec["assignment_id"],
                    "assignment_code": code,
                }
            )

    assessment_records = []
    for code, spec in assessment_specs.items():
        assessment_records.append(
            {
                "assessment_id": f"s25_{code}",
                "assessment_code": code,
                "assessment_kind": spec["kind"],
                "max_points": spec["max_points"],
                "assignment_id": spec["assignment_id"],
            }
        )

    with engine.begin() as conn:
        conn.execute(semester_sql)

        for batch in chunked(user_records, batch_size):
            conn.execute(users_sql, batch)

        for batch in chunked(assignment_records, batch_size):
            conn.execute(assignments_sql, batch)

        for batch in chunked(assessment_records, batch_size):
            conn.execute(assessments_sql, batch)

        for batch in chunked(grade_profiles, batch_size):
            conn.execute(grade_profile_sql, batch)

        for batch in chunked(score_records, batch_size):
            conn.execute(score_sql, batch)

    print(f"Done. Inserted/updated {len(grade_profiles)} grade profiles and {len(score_records)} score rows.")


if __name__ == "__main__":
    main()