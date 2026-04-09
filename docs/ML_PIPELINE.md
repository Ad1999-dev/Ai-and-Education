# ML Pipeline

Predicts student performance (exam or assignment scores) from LLM-usage features
using nested cross-validation.

## Quick start

```bash
# Minimal — all defaults (all features, all students, all models)
python -m src.ml.run_training_exam
python -m src.ml.run_training_assignment
python -m src.ml.run_training_both

# Select specific features
python -m src.ml.run_training_exam --features dlg_n_turns dlg_avg_prompt_chars

# Drop rows with zero dialogue activity for that assessment
python -m src.ml.run_training_both --drop-no-dialogue

# Change encoding or restrict models
python -m src.ml.run_training_assignment --encoding ordinal --models ridge random_forest

# Full args — exam pipeline
python -m src.ml.run_training_exam \
    --encoding one_hot \
    --drop-no-dialogue \
    --features dlg_n_turns dlg_n_chats dlg_avg_prompt_chars \
    --models ridge random_forest \
    --outer 10 --inner 5 \
    --output results/

# Full args — assignment pipeline
python -m src.ml.run_training_assignment \
    --encoding one_hot \
    --drop-no-dialogue \
    --features dlg_n_turns dlg_n_chats dlg_avg_prompt_chars \
    --models ridge random_forest \
    --outer 10 --inner 5 \
    --output results/

# Full args — both pipeline
python -m src.ml.run_training_both \
    --encoding one_hot \
    --drop-no-dialogue \
    --features dlg_n_turns dlg_n_chats dlg_avg_prompt_chars \
    --models ridge random_forest \
    --outer 10 --inner 5 \
    --output results/
```

Results are written to `results/<pipeline>_YYYYMMDD_HHMMSS/`.

## Batch experiment runner

`run_experiments.py` sweeps over predefined feature groups × `drop_no_dialogue` on/off,
calling the appropriate `run_training_*.py` as a subprocess per variant.

```bash
# Edit PIPELINE_TYPE, ASSESSMENT_ENCODING, _ENABLED at the top of the file, then:
python -m src.ml.run_experiments
python -m src.ml.run_experiments --outer 5 --inner 5 --output results/
```

**Feature groups** (each runs with `all_students` and `active_only`):

| Group name | Columns included |
|---|---|
| `all` | All available columns (no `--features` flag) |
| `counts` | `dlg_n_turns`, `dlg_n_chats` |
| `prompt_length` | `dlg_total_prompt_chars`, `dlg_avg_prompt_chars` |
| `response_length` | `dlg_total_response_chars`, `dlg_avg_response_chars` |
| `lengths` | All prompt + response char columns |
| `counts_lengths` | Counts + all lengths |
| `cat_pct` | All `dlg_cat_*_pct` columns |
| `cat_count` | All `dlg_cat_*_count` columns |
| `cat_all` | All category count + pct columns |
| `counts_cat_pct` | Counts + category proportions |
| `counts_cat_all` | Counts + all category columns |
| `counts_lengths_cat_pct` | Counts + lengths + category proportions |
| `counts_lengths_cat_all` | Counts + lengths + all category columns |
| `all_with_sub` | All columns including `sub_has_submission` |
| `counts_sub` | Counts + `sub_has_submission` |
| `counts_lengths_sub` | Counts + lengths + `sub_has_submission` |

Toggle individual groups with `_ENABLED` dict; variant names follow the pattern `<group>__all_students` / `<group>__active_only`.

## Arguments (all three run_training scripts)

| Argument | Values | Default | Description |
|---|---|---|---|
| `--encoding` | `none` \| `one_hot` \| `ordinal` | `none` | Assessment identity encoding |
| `--drop-no-dialogue` | flag | off | Remove rows with zero dialogue activity |
| `--features COL ...` | column names | *(all)* | Exact columns passed to the model |
| `--models MODEL ...` | `ridge` `elastic_net` `random_forest` `gradient_boosting` `svr` | *(all)* | Models to evaluate |
| `--outer K` | int | `5` | Outer GroupKFold splits |
| `--inner K` | int | `5` | Inner GroupKFold splits (hyperparameter search) |
| `--output PATH` | path | `results/` | Base directory for run subdirectories |

## Pipelines

| Script | Dataset | Target | Notes |
|---|---|---|---|
| `run_training_exam.py` | ~181×3 = 543 rows | `normalized_score` for e1/e2/e3 | All features available |
| `run_training_assignment.py` | ~181×6 = 1086 rows | `normalized_score` for a2–a7 | a1 excluded |
| `run_training_both.py` | ~1629 rows | same, pooled | adds `is_exam` flag |

## Design

**Unit of analysis**: one row per `(user_id, assessment_id)`.

**Groups for CV**: `user_id` — the same student never appears in both train and test folds.

**No-leakage guarantee**: the outer CV loop is implemented manually so that
`groups_train` (subsetted to each outer fold) is passed explicitly to
`inner_search.fit()`. Using `cross_val_score` with a `GridSearchCV(GroupKFold)`
estimator would silently pass `groups=None` to the inner split.

**Preprocessing** (fitted inside each fold, never on the full dataset):
1. `SimpleImputer(strategy="median")` — handles any remaining NaNs
2. `StandardScaler` — z-score normalisation

## Feature blocks

All blocks are always built. Use `--features` to restrict which columns reach the model.

### Dialogue counts / lengths  (`dlg_*`)
Aggregated per `(user_id, assessment_id)`:

- **Assignment rows**: turns where `dialogue_turns.assignment_id` links to this assessment.
- **Exam e1/e2 rows**: turns where `dialogue_turns.exam_id` matches.
- **Exam e3 rows**: all assignment turns for the student's semester (a1–a7 combined).

| Column | Description |
|---|---|
| `dlg_n_turns` | Total dialogue turns |
| `dlg_n_chats` | Distinct chat sessions |
| `dlg_total_prompt_chars` | Total characters in user prompts |
| `dlg_avg_prompt_chars` | Average characters per prompt |
| `dlg_total_response_chars` | Total characters in LLM responses |
| `dlg_avg_response_chars` | Average characters per LLM response |

### Dialogue categories  (`dlg_cat_*`)
`llm_label` proportions and raw counts per `(user_id, assessment_id)`.  
Same UNION logic as counts/lengths (assignment via `assignment_id`, e1/e2 via `exam_id`, e3 = all assignments).

| Column pattern | Description |
|---|---|
| `dlg_cat_<label>_count` | Raw turn count for that label |
| `dlg_cat_<label>_pct` | Proportion (0–1) of turns with that label |

Label names are derived from `llm_label` values in the DB.

### Submission  (`sub_*`)
Currently a presence flag only (full submission stats are preserved in the query for future use).

| Column | Description |
|---|---|
| `sub_has_submission` | 1 if a submission exists for this `(user, assignment)`, else NaN → 0 |

### Assessment identity  (`assess_*`)
Controlled by `--encoding`:

| Value | Columns | Description |
|---|---|---|
| `none` *(default)* | — | No assessment identity |
| `one_hot` | `assess_a2` … `assess_e3` | Binary column per assessment code |
| `ordinal` | `assess_ordinal` | Single integer (lexicographic order) |

### `is_exam` flag  (`both` pipeline only)
`1` for exam rows, `0` for assignment rows.

## Cleaning options

| Flag | Effect |
|---|---|
| *(none)* | All rows kept, including students with zero dialogue activity |
| `--drop-no-dialogue` | Removes rows where all `dlg_*` columns are 0 |

## Module layout (`src/ml/`)

| File | Role |
|---|---|
| `config.py` | `MODEL_REGISTRY`, CV params, `AssessmentEncoding` enum |
| `db.py` | Shared `load_engine()` |
| `data_loader.py` | Long-format base frame — one row per `(user_id, assessment_id)` |
| `data_preprocessing.py` | Joins feature blocks; zero-fill; `drop_no_dialogue`; `select_features` |
| `feature_engineering.py` | Assessment encoding (one-hot / ordinal / none) |
| `model_training.py` | sklearn `Pipeline` builder, nested CV loop, result saving |
| `run_training_exam.py` | CLI entry point — exam pipeline |
| `run_training_assignment.py` | CLI entry point — assignment pipeline |
| `run_training_both.py` | CLI entry point — combined pipeline |
| `run_experiments.py` | Batch runner — calls the above via subprocess |
| `features/dialogue.py` | `build_counts_lengths()`, `build_categories()` |
| `features/submission.py` | `build()` — presence flag; full stats commented for future use |

## Models

Defined in `MODEL_REGISTRY` in `config.py`:

| Key | Estimator | Tuned hyperparameters |
|---|---|---|
| `ridge` | `Ridge` | `alpha` |
| `elastic_net` | `ElasticNet` | `alpha`, `l1_ratio` |
| `random_forest` | `RandomForestRegressor` | `n_estimators`, `max_depth`, `min_samples_leaf` |
| `gradient_boosting` | `GradientBoostingRegressor` | `n_estimators`, `learning_rate`, `max_depth` |
| `svr` | `SVR` | `C`, `epsilon`, `kernel` |

To add a model: add an entry to `MODEL_REGISTRY` — no other files change.

## Extending the pipeline

**New feature block:**
1. Create `src/ml/features/<name>.py` with `build(engine: Engine) -> pd.DataFrame`
   indexed by `["user_id", "assessment_id"]`, columns prefixed with a unique tag.
2. Join the result in `data_preprocessing.py` after the existing blocks.
3. Add the prefix to `_ZERO_FILL_PREFIXES` if zero-fill applies.

**New model:** add an entry to `MODEL_REGISTRY` in `config.py`.

## Output files

### Per-run directory  (`results/<pipeline>_YYYYMMDD_HHMMSS/`)

| File | Description |
|---|---|
| `dataset.csv` | Exact features + target passed to the model |
| `results.csv` | Per-model RMSE across outer folds (mean, std, min, max, per-fold) |
| `run_summary.json` | Full metadata: pipeline, CV config, encoding, feature columns, best model |
| `final_model.pkl` | Best model trained on the full dataset (joblib) |

### Persistent log  (`results/runs_log.csv`)

One row per run, appended automatically.  Key columns:

| Column | Description |
|---|---|
| `run_at` | ISO-8601 datetime |
| `pipeline` | `exam`, `assignment`, or `both` |
| `n_samples` / `n_features` / `n_students` | Dataset shape |
| `assessment_encoding` | `one_hot`, `ordinal`, or `none` |
| `best_model` / `best_rmse` / `best_rmse_std` | Best model summary |
| `rmse_<model>` / `std_<model>` | Per-model results (NaN if not evaluated) |
