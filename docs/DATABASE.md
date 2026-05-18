# Database Setup

## Prerequisites

- Docker + Docker Compose
- Python 3.11+ with [uv](https://docs.astral.sh/uv/)

## 1. Environment

```bash
cp .envexample .env   # fill in DB_USER, DB_PASSWORD, DB_NAME, PGADMIN_* credentials
```

## 2. Start the database

```bash
docker compose up -d
```

This starts:
- **PostgreSQL** on port `5432` — schema is applied automatically on first start
- **pgAdmin** on `http://localhost:8081`

## 3. Install Python dependencies

```bash
uv sync
```

## 4. Run ingestion (in order)

```bash
uv run python src/ingestion/ingest_parquet.py      # StudyChat dialogue logs
uv run python src/ingestion/ingest_grades.py       # grade CSVs (f24 + s25 auto-detected)
uv run python src/ingestion/ingest_assignments.py  # assignment text files
uv run python src/ingestion/ingest_submissions.py  # student submission files
```

Each script is idempotent — re-running is safe (all inserts use `ON CONFLICT DO UPDATE`).

## 5. Verify

```bash
uv run python src/db_stats.py
```

Expected output after full ingestion:

```
table                            rows
--------------------------------------
semesters                           2
users                             365
assignments                        14
assessments                        20
user_grade_profiles               181
user_assessment_scores          1,810
chats                           2,214
dialogue_turns                 16,851
dialogue_messages             342,088
assignment_files                   65
submissions                       921
submission_files                3,761
```

> `dialogue_messages` is populated from the `messages` column in the parquet files.
> Requires pyarrow <18 — the project pins `pyarrow>=15.0,<18.0` in `pyproject.toml`.

## Reset

```bash
docker compose down -v && docker compose up -d   # wipes all data and re-applies schema
```

## Schema overview

| Table | Description |
|---|---|
| `semesters` | `f24` (Fall 2024) and `s25` (Spring 2025) |
| `users` | one row per student UUID |
| `assignments` | per-semester assignment stubs (`s25_a1` … `f24_a7`) |
| `assessments` | assignments + exams with `max_points` and `assessment_kind` |
| `user_grade_profiles` | links a user to a semester directory |
| `user_assessment_scores` | normalized score (0–1) per user × assessment |
| `chats` | StudyChat conversation sessions |
| `dialogue_turns` | one row per prompt/response pair, with `llm_label` / `llm_sublabel` |
| `dialogue_messages` | raw message list (role + content) for each turn |
| `assignment_files` | text-extracted files from assignment folders |
| `submissions` | one row per student × assignment |
| `submission_files` | individual files within each submission |
