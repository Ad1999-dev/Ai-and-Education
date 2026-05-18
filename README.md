# AI and Education
PROJ0021 Data science project

**Research question**: do students who use an LLM assistant on assignments perform differently on exams (no LLM)?

---

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
cp .envexample .env          # fill in DB credentials
uv sync                      # install Python dependencies
docker compose up -d         # start postgres + pgAdmin (schema auto-applied)
```

pgAdmin is served at `http://localhost:8081` (credentials from `.env`).

To fully reset the database:

```bash
docker compose down -v && docker compose up -d
```

---

## Data ingestion

Run the four scripts in order. Each is idempotent (`ON CONFLICT DO UPDATE`) and accepts `--batch-size N`.

```bash
python src/ingestion/ingest_parquet.py       # StudyChat dialogues → chats, dialogue_turns
python src/ingestion/ingest_grades.py        # grades CSV          → assessments, scores
python src/ingestion/ingest_assignments.py   # assignment text     → assignments, assignment_files
python src/ingestion/ingest_submissions.py   # student submissions → submissions, submission_files
```

Sanity check after ingest:

```bash
python src/db_stats.py                       # row counts per table
docker exec -it studychat_postgres psql -U studychat_user -d studychat
```

---

## Modeling

Three nested-CV pipelines for grade prediction — one entry point each:

| Script | Target | Rows |
|---|---|---|
| `run_training_exam.py` | exam scores (e1, e2, e3) | ~543 |
| `run_training_assignment.py` | assignment scores (a2–a7) | ~1086 |
| `run_training_both.py` | pooled, with `is_exam` flag | ~1629 |

```bash
python -m src.ml.run_training_exam
python -m src.ml.run_training_assignment --models ridge random_forest
python -m src.ml.run_training_both --outer 10 --inner 5 --output results/
```

Common CLI flags (all three scripts):

- `--features col1 col2 …` — restrict to specific feature columns
- `--drop-no-dialogue` — drop rows with zero dialogue activity
- `--drop-zero-score` — drop rows where score == 0 (non-submission, distinct from NULL)
- `--assessments code1 …` — restrict to a subset of assessment codes
- `--encoding one_hot|ordinal|none` — how to encode `assessment_id`
- `--models name1 …` — subset of `MODEL_REGISTRY` (default: all)
- `--outer K` / `--inner K` — CV fold counts

### Batch experiments

`run_experiments.py` sweeps a powerset of three feature blocks {volume, DA proportions, similarity} anchored to the prior-grade baseline = 8 models × 4 cleaning combos = 32 variants:

```bash
python -m src.ml.run_experiments
```

Results land in `results/<pipeline>_YYYYMMDD_HHMMSS/` (one dir per run) plus a cumulative `run_log.csv`.

### Similarity feature (optional, requires GPU)

The `sim_max` feature reads precomputed BGE-M3 cosine similarities from `src/data/final_similarity_results.csv`. To regenerate that file (needs CUDA + `FlagEmbedding`):

```bash
python -m src.ml.features.code_similarity
```

---

## Repo structure

```
.
├── data/raw/        # student data (gitignored): parquet, grades, assignments, submissions
├── docs/            # DATABASE.md, ML_PIPELINE.md, FEATURE_DICTIONARY.md, RESULTS.md
├── sql/             # schema.sql + views.sql
├── src/
│   ├── ingestion/   # 4 idempotent ingest scripts
│   ├── ml/          # modeling pipeline (nested CV, training entry points)
│   │   └── features/  # prior_grades, dialogue, submission, similarity
│   └── db_stats.py
├── notebooks/       # EDA, sanity checks
├── docker-compose.yml
└── pyproject.toml
```


---

## How we work (GitFlow)

* `main`: milestone-ready only
* `develop`: integration branch
* `feature/*`: development branches

---

## Contributors

* Mohamed khalil Ankri
* Antoine DECKERS
* Hoang Linh BUI
* Duy Vu DINH
