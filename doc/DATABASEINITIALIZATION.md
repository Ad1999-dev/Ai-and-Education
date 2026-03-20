# Database Initialization

## 1. Project files

### Root files
- `docker-compose.yml`: Starts the PostgreSQL and pgAdmin containers.

- `.env`: Stores database and pgAdmin configuration variables such as database name, user, password, and port.

- `requirements.txt`: Lists the Python packages required for ingestion scripts.

## 2. SQL files

- `sql/schema.sql`: Creates the normalized database schema, including tables, keys, constraints, and indexes.

- `sql/views.sql`: Creates derived views for analysis, such as user activity, grade aggregates, and submission summaries.

## 3. ETL / ingestion scripts

- `src/ingest_parquet.py`: Loads StudyChat dialogue data from parquet files into:
  - `users`
  - `chats`
  - `dialogue_turns`
  - `dialogue_messages`

- `src/ingest_grades.py`: Loads the normalized Spring 2025 grades from CSV into:
  - `assessments`
  - `user_grade_profiles`
  - `user_assessment_scores`

- `src/ingest_assignments.py`: Loads assignment metadata and files into:
  - `assignments`
  - `assignment_files`

- `src/ingest_submissions.py`: Loads submission metadata and files into:
  - `submissions`
  - `submission_files`


## 4. Raw data folders

Place the raw files in these locations:
- `data/raw/parquet/`
  - `0000.parquet`
  - `0001.parquet`

- `data/raw/grades/`
  - `s25_grades_released_normalized.csv`

- `data/raw/assignment_text/`
  - assignment folders such as `studychat_f24_assignments/` and `studychat_s25_assignments/`

- `data/raw/submissions/`
  - semester folders such as `f24/` and `s25/`


## 5. Initialization steps
Run:

```bash
docker compose down -v
docker compose up -d
```

This recreates the PostgreSQL database and automatically runs:

* `sql/schema.sql`
* `sql/views.sql`

**pgAdmin**

Log in using the credentials from `.env`.

```text
http://localhost:8081
```

**Run ingestion scripts**

Run:

```bash
pip install -r requirements.txt
```

Run in this order:

```bash
python src/ingest_parquet.py
python src/ingest_grades.py
python src/ingest_assignments.py
python src/ingest_submissions.py
```
