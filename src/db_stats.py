from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

TABLES = [
    "semesters",
    "users",
    "assignments",
    "assessments",
    "user_grade_profiles",
    "user_assessment_scores",
    "chats",
    "dialogue_turns",
    "dialogue_messages",
    "assignment_files",
    "submissions",
    "submission_files",
]


def load_engine():
    load_dotenv(ENV_PATH)
    import os

    url = (
        f"postgresql+psycopg://{os.getenv('DB_USER', 'studychat_user')}"
        f":{os.getenv('DB_PASSWORD', 'studychat_password')}"
        f"@{os.getenv('DB_HOST', 'localhost')}"
        f":{os.getenv('DB_PORT', '5432')}"
        f"/{os.getenv('DB_NAME', 'studychat')}"
    )
    return create_engine(url, future=True, pool_pre_ping=True)


def main():
    engine = load_engine()
    with engine.connect() as conn:
        print(f"{'table':<28} {'rows':>8}")
        print("-" * 38)
        total = 0
        for table in TABLES:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"{table:<28} {n:>8,}")
            total += n
        print("-" * 38)
        print(f"{'TOTAL':<28} {total:>8,}")


if __name__ == "__main__":
    main()
