import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Tuple, Union

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_ASSIGNMENTS_DIR = PROJECT_ROOT / "data" / "raw" / "assignment_text"

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


def extract_title_and_description(assignment_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    readme_candidates = [
        assignment_dir / "README.md",
        assignment_dir / "readme.md",
        assignment_dir / "README.txt",
        assignment_dir / "README",
    ]

    for readme in readme_candidates:
        if readme.exists():
            content = safe_text_read(readme)
            if not content:
                continue

            title = None
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("#"):
                    title = line.lstrip("#").strip()
                    break

            if title is None:
                title = assignment_dir.name

            return title, content[:MAX_EXTRACTED_CHARS]

    return assignment_dir.name, None


def parse_semester_folder(folder_name: str) -> Tuple[Optional[str], Optional[str]]:
    folder_lower = folder_name.lower()
    if "f24" in folder_lower:
        return "f24", "Fall 2024"
    if "s25" in folder_lower:
        return "s25", "Spring 2025"
    return None, None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Ingest normalized StudyChat assignment folders.")
    parser.add_argument("--assignments-dir", type=Path, default=DEFAULT_ASSIGNMENTS_DIR)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_arguments()
    assignments_dir = args.assignments_dir
    batch_size = args.batch_size

    if not assignments_dir.exists():
        raise FileNotFoundError(f"Assignments directory not found: {assignments_dir}")

    semester_roots = [p for p in assignments_dir.iterdir() if p.is_dir()]
    if not semester_roots:
        raise FileNotFoundError(f"No semester assignment folders found in: {assignments_dir}")

    engine = load_engine()

    semester_sql = text(
        """
        INSERT INTO semesters (semester_code, semester_name)
        VALUES (:semester_code, :semester_name)
        ON CONFLICT (semester_code) DO NOTHING
        """
    )

    assignment_sql = text(
        """
        INSERT INTO assignments (
            assignment_id,
            semester_code,
            assignment_code,
            title,
            description,
            folder_path
        )
        VALUES (
            :assignment_id,
            :semester_code,
            :assignment_code,
            :title,
            :description,
            :folder_path
        )
        ON CONFLICT (assignment_id) DO UPDATE
        SET
            semester_code = EXCLUDED.semester_code,
            assignment_code = EXCLUDED.assignment_code,
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            folder_path = EXCLUDED.folder_path
        """
    )

    assignment_file_sql = text(
        """
        INSERT INTO assignment_files (
            assignment_id,
            relative_path,
            file_name,
            file_type,
            extracted_text
        )
        VALUES (
            :assignment_id,
            :relative_path,
            :file_name,
            :file_type,
            :extracted_text
        )
        ON CONFLICT (assignment_id, relative_path) DO UPDATE
        SET
            file_name = EXCLUDED.file_name,
            file_type = EXCLUDED.file_type,
            extracted_text = EXCLUDED.extracted_text
        """
    )

    total_assignments = 0
    total_files = 0

    for semester_root in semester_roots:
        semester_code, semester_name = parse_semester_folder(semester_root.name)
        if semester_code is None:
            continue

        with engine.begin() as conn:
            conn.execute(
                semester_sql,
                {
                    "semester_code": semester_code,
                    "semester_name": semester_name,
                },
            )

        assignment_dirs = sorted([
            p for p in semester_root.iterdir()
            if p.is_dir() and re.fullmatch(r"a\d+", p.name.lower())
        ])

        for assignment_dir in assignment_dirs:
            assignment_code = assignment_dir.name.lower()
            assignment_id = f"{semester_code}_{assignment_code}"
            title, description = extract_title_and_description(assignment_dir)
            folder_path = str(assignment_dir.relative_to(PROJECT_ROOT))

            assignment_record = {
                "assignment_id": assignment_id,
                "semester_code": semester_code,
                "assignment_code": assignment_code,
                "title": title,
                "description": description,
                "folder_path": folder_path,
            }

            file_records = []
            for file_path in sorted(assignment_dir.rglob("*")):
                if not file_path.is_file():
                    continue

                relative_path = str(file_path.relative_to(assignment_dir))
                file_type = file_path.suffix.lower().lstrip(".") or "no_extension"
                extracted_text = safe_text_read(file_path)

                file_records.append(
                    {
                        "assignment_id": assignment_id,
                        "relative_path": relative_path,
                        "file_name": file_path.name,
                        "file_type": file_type,
                        "extracted_text": extracted_text,
                    }
                )

            with engine.begin() as conn:
                conn.execute(assignment_sql, assignment_record)
                for batch in chunked(file_records, batch_size):
                    conn.execute(assignment_file_sql, batch)

            total_assignments += 1
            total_files += len(file_records)
            print(f"Ingested {assignment_id}: {len(file_records)} assignment files")

    print(f"Done. Ingested/updated {total_assignments} assignments and {total_files} assignment files.")


if __name__ == "__main__":
    main()