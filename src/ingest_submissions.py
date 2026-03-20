import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Dict, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_SUBMISSIONS_DIR = PROJECT_ROOT / "data" / "raw" / "submissions"

TEXT_EXTENSIONS = {
    ".md", ".txt", ".py", ".csv", ".json", ".yaml", ".yml",
    ".html", ".htm", ".css", ".js", ".sql", ".tex", ".r", ".java", ".c", ".cpp", ".ipynb"
}
MAX_TEXT_FILE_SIZE = 2_000_000
MAX_EXTRACTED_CHARS = 50_000


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


def extract_user_id(directory_name: str) -> str:
    return directory_name


def parse_assignment_code(folder_name: str) -> Optional[str]:
    folder_name = folder_name.lower()
    match = re.search(r"assignment[-_]?(\d+)", folder_name)
    if match:
        return f"a{int(match.group(1))}"
    if re.fullmatch(r"a\d+", folder_name):
        return folder_name
    return None


def safe_text_read(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return None
    if path.stat().st_size > MAX_TEXT_FILE_SIZE:
        return None

    if path.suffix.lower() == ".ipynb":
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            parts = []
            for cell in notebook.get("cells", []):
                source = cell.get("source", [])
                if isinstance(source, list):
                    parts.append("".join(source))
                elif isinstance(source, str):
                    parts.append(source)
            return "\n\n".join(parts)[:MAX_EXTRACTED_CHARS]
        except Exception:
            return None

    for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return path.read_text(encoding=encoding)[:MAX_EXTRACTED_CHARS]
        except Exception:
            continue
    return None


def count_lines_of_code(path: Path) -> Optional[int]:
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return None

    if path.suffix.lower() == ".ipynb":
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            total = 0
            for cell in notebook.get("cells", []):
                source = cell.get("source", [])
                if isinstance(source, list):
                    total += len(source)
                elif isinstance(source, str):
                    total += len(source.splitlines())
            return total
        except Exception:
            return None

    try:
        return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    except Exception:
        return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Ingest normalized StudyChat submission folders.")
    parser.add_argument("--submissions-dir", type=Path, default=DEFAULT_SUBMISSIONS_DIR)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_arguments()
    submissions_dir = args.submissions_dir
    batch_size = args.batch_size

    if not submissions_dir.exists():
        raise FileNotFoundError(f"Submissions directory not found: {submissions_dir}")

    engine = load_engine()

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

    submissions_sql = text(
        """
        INSERT INTO submissions (
            user_id,
            assignment_id,
            directory_name,
            folder_path
        )
        VALUES (
            :user_id,
            :assignment_id,
            :directory_name,
            :folder_path
        )
        ON CONFLICT (user_id, assignment_id) DO UPDATE
        SET
            directory_name = EXCLUDED.directory_name,
            folder_path = EXCLUDED.folder_path
        """
    )

    submission_file_sql = text(
        """
        INSERT INTO submission_files (
            user_id,
            assignment_id,
            relative_path,
            file_name,
            file_type,
            extracted_text,
            file_size_bytes,
            lines_of_code
        )
        VALUES (
            :user_id,
            :assignment_id,
            :relative_path,
            :file_name,
            :file_type,
            :extracted_text,
            :file_size_bytes,
            :lines_of_code
        )
        ON CONFLICT (user_id, assignment_id, relative_path) DO UPDATE
        SET
            file_name = EXCLUDED.file_name,
            file_type = EXCLUDED.file_type,
            extracted_text = EXCLUDED.extracted_text,
            file_size_bytes = EXCLUDED.file_size_bytes,
            lines_of_code = EXCLUDED.lines_of_code
        """
    )

    total_submissions = 0
    total_files = 0

    for semester_dir in sorted([p for p in submissions_dir.iterdir() if p.is_dir()]):
        semester_code = semester_dir.name.lower()
        if semester_code not in {"f24", "s25"}:
            continue

        semester_name = "Fall 2024" if semester_code == "f24" else "Spring 2025"

        with engine.begin() as conn:
            conn.execute(
                semester_sql,
                {
                    "semester_code": semester_code,
                    "semester_name": semester_name,
                },
            )

        assignment_dirs = sorted([p for p in semester_dir.iterdir() if p.is_dir()])
        for assignment_dir in assignment_dirs:
            assignment_code = parse_assignment_code(assignment_dir.name)
            if assignment_code is None:
                continue

            assignment_id = f"{semester_code}_{assignment_code}"

            with engine.begin() as conn:
                conn.execute(
                    assignments_sql,
                    {
                        "assignment_id": assignment_id,
                        "semester_code": semester_code,
                        "assignment_code": assignment_code,
                    },
                )

            user_dirs = sorted([p for p in assignment_dir.iterdir() if p.is_dir()])
            for user_dir in user_dirs:
                if user_dir.name == ".git":
                    continue

                user_id = extract_user_id(user_dir.name)
                folder_path = str(user_dir.relative_to(PROJECT_ROOT))

                files = [
                    p for p in user_dir.rglob("*")
                    if p.is_file() and ".git" not in p.parts
                ]

                submission_record = {
                    "user_id": user_id,
                    "assignment_id": assignment_id,
                    "directory_name": user_dir.name,
                    "folder_path": folder_path,
                }

                file_records = []
                for file_path in files:
                    relative_path = str(file_path.relative_to(user_dir))
                    file_type = file_path.suffix.lower().lstrip(".") or "no_extension"
                    extracted_text = safe_text_read(file_path)
                    file_size_bytes = file_path.stat().st_size
                    lines_of_code = count_lines_of_code(file_path)

                    file_records.append(
                        {
                            "user_id": user_id,
                            "assignment_id": assignment_id,
                            "relative_path": relative_path,
                            "file_name": file_path.name,
                            "file_type": file_type,
                            "extracted_text": extracted_text,
                            "file_size_bytes": file_size_bytes,
                            "lines_of_code": lines_of_code,
                        }
                    )

                with engine.begin() as conn:
                    conn.execute(users_sql, {"user_id": user_id})
                    conn.execute(submissions_sql, submission_record)

                    for batch in chunked(file_records, batch_size):
                        conn.execute(submission_file_sql, batch)

                total_submissions += 1
                total_files += len(file_records)
                print(f"Ingested {assignment_id} / {user_id}: {len(file_records)} files")

    print(f"Done. Ingested/updated {total_submissions} submissions and {total_files} submission files.")


if __name__ == "__main__":
    main()