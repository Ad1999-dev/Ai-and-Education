import argparse
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_GRADES_DIR = PROJECT_ROOT / "data" / "raw" / "grades"
GRADES_GLOB = "*_grades_released_normalized.csv"

SEMESTER_NAMES = {
    "f24": "Fall 2024",
    "s25": "Spring 2025",
}

# max_points per semester — update f24 values if they differ from s25
ASSESSMENT_SPECS: dict[str, dict[str, dict]] = {
    "f24": {
        "a1": {"kind": "assignment", "max_points": 31},
        "a2": {"kind": "assignment", "max_points": 100},
        "a3": {"kind": "assignment", "max_points": 55},
        "a4": {"kind": "assignment", "max_points": 90},
        "a5": {"kind": "assignment", "max_points": 105},
        "a6": {"kind": "assignment", "max_points": 113},
        "a7": {"kind": "assignment", "max_points": 135},
        "e1": {"kind": "exam", "max_points": 100},
        "e2": {"kind": "exam", "max_points": 100},
        "e3": {"kind": "exam", "max_points": 100},
    },
    "s25": {
        "a1": {"kind": "assignment", "max_points": 31},
        "a2": {"kind": "assignment", "max_points": 100},
        "a3": {"kind": "assignment", "max_points": 55},
        "a4": {"kind": "assignment", "max_points": 90},
        "a5": {"kind": "assignment", "max_points": 105},
        "a6": {"kind": "assignment", "max_points": 113},
        "a7": {"kind": "assignment", "max_points": 135},
        "e1": {"kind": "exam", "max_points": 100},
        "e2": {"kind": "exam", "max_points": 100},
        "e3": {"kind": "exam", "max_points": 100},
    },
}


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


def infer_semester_code(csv_path: Path) -> str:
    """Derive semester code from filename prefix (e.g. 'f24_grades...' → 'f24')."""
    stem = csv_path.stem  # e.g. 'f24_grades_released_normalized'
    code = stem.split("_")[0]
    if code not in SEMESTER_NAMES:
        raise ValueError(
            f"Cannot infer semester code from filename '{csv_path.name}'. "
            f"Use --semester-code to specify one of: {sorted(SEMESTER_NAMES)}"
        )
    return code


def parse_arguments():
    parser = argparse.ArgumentParser(description="Ingest normalized StudyChat grades.")
    parser.add_argument(
        "--grades-dir", type=Path, default=DEFAULT_GRADES_DIR,
        help=f"Directory to scan for grade CSVs (default: {DEFAULT_GRADES_DIR}).",
    )
    parser.add_argument(
        "--csv-path", type=Path, default=None,
        help="Ingest a single CSV file instead of scanning the grades directory.",
    )
    parser.add_argument(
        "--semester-code", type=str, default=None, choices=sorted(SEMESTER_NAMES),
        help="Semester code — only used together with --csv-path.",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def ingest_file(csv_path: Path, semester_code: str, batch_size: int, engine) -> None:
    semester_name = SEMESTER_NAMES[semester_code]
    print(f"  Ingesting {csv_path.name} ({semester_name}) ...")

    df = pd.read_csv(csv_path)

    required = {"userId", "directory_name", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "e1", "e2", "e3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing)}")

    score_columns = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "e1", "e2", "e3"]
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    semester_sql = text(
        """
        INSERT INTO semesters (semester_code, semester_name)
        VALUES (:semester_code, :semester_name)
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
        VALUES (:assignment_id, :semester_code, :assignment_code, NULL, NULL, NULL)
        ON CONFLICT (assignment_id) DO NOTHING
        """
    )
    assessments_sql = text(
        """
        INSERT INTO assessments (
            assessment_id, semester_code, assessment_code, assessment_kind, max_points, assignment_id
        )
        VALUES (
            :assessment_id, :semester_code, :assessment_code, :assessment_kind, :max_points, :assignment_id
        )
        ON CONFLICT (assessment_id) DO UPDATE
        SET
            assessment_code = EXCLUDED.assessment_code,
            assessment_kind = EXCLUDED.assessment_kind,
            max_points      = EXCLUDED.max_points,
            assignment_id   = EXCLUDED.assignment_id
        """
    )
    grade_profile_sql = text(
        """
        INSERT INTO user_grade_profiles (user_id, semester_code, directory_name)
        VALUES (:user_id, :semester_code, :directory_name)
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

    assessment_specs = ASSESSMENT_SPECS[semester_code]

    user_records: List[Dict] = []
    grade_profiles: List[Dict] = []
    score_records: List[Dict] = []

    for _, row in df.iterrows():
        user_id = safe_null(row["userId"])
        if user_id is None:
            continue
        user_id = str(user_id)
        user_records.append({"user_id": user_id})
        grade_profiles.append({
            "user_id": user_id,
            "semester_code": semester_code,
            "directory_name": safe_null(row["directory_name"]),
        })
        for code in score_columns:
            score_records.append({
                "user_id": user_id,
                "assessment_id": f"{semester_code}_{code}",
                "normalized_score": safe_null(row[code]),
            })

    assignment_records = [
        {"assignment_id": f"{semester_code}_{code}", "assignment_code": code, "semester_code": semester_code}
        for code, spec in assessment_specs.items()
        if spec["kind"] == "assignment"
    ]
    assessment_records = [
        {
            "assessment_id": f"{semester_code}_{code}",
            "assessment_code": code,
            "assessment_kind": spec["kind"],
            "max_points": spec["max_points"],
            "assignment_id": f"{semester_code}_{code}" if spec["kind"] == "assignment" else None,
            "semester_code": semester_code,
        }
        for code, spec in assessment_specs.items()
    ]

    with engine.begin() as conn:
        conn.execute(semester_sql, {"semester_code": semester_code, "semester_name": semester_name})
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

    print(f"    → {len(grade_profiles)} grade profiles, {len(score_records)} score rows inserted/updated.")


def main():
    args = parse_arguments()
    batch_size = args.batch_size

    if args.csv_path is not None:
        csv_files = [args.csv_path]
    else:
        csv_files = sorted(args.grades_dir.glob(GRADES_GLOB))
        if not csv_files:
            raise FileNotFoundError(f"No grade CSVs found in {args.grades_dir} matching '{GRADES_GLOB}'.")

    engine = load_engine()

    for csv_path in csv_files:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        semester_code = args.semester_code or infer_semester_code(csv_path)
        ingest_file(csv_path, semester_code, batch_size, engine)

    print(f"Done. Processed {len(csv_files)} file(s).")


if __name__ == "__main__":
    main()