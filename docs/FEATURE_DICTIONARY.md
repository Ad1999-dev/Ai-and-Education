# Feature Dictionary

All features available to the ML pipeline (`src/ml/`).

**Unit of analysis**: one row per `(user_id, assessment_id)`.  
**Target**: `normalized_score` (0–1) for that specific assessment — never an input feature.  
**Index**: all feature DataFrames are indexed by `(user_id, assessment_id)` and joined directly on the base frame. All blocks are always built; use `--features col1 col2 ...` to restrict which columns reach the model.

---

## Complete feature pool

Copy any subset of these into `--features` or a variant's `"features"` list.

### Dialogue counts / lengths

| Column | Description |
|---|---|
| `dlg_n_turns` | Total dialogue turns |
| `dlg_n_chats` | Distinct chat sessions |
| `dlg_total_prompt_chars` | Total characters in all user prompts |
| `dlg_avg_prompt_chars` | Average characters per user prompt |
| `dlg_total_response_chars` | Total characters in all LLM responses |
| `dlg_avg_response_chars` | Average characters per LLM response |

### Dialogue category counts  (raw turn counts per label)

| Column | Label |
|---|---|
| `dlg_cat_conceptual_questions_count` | Conceptual questions |
| `dlg_cat_contextual_questions_count` | Contextual questions |
| `dlg_cat_editing_request_count` | Editing requests |
| `dlg_cat_misc_count` | Miscellaneous |
| `dlg_cat_off_topic_count` | Off-topic |
| `dlg_cat_provide_context_count` | Providing context |
| `dlg_cat_verification_count` | Verification / checking |
| `dlg_cat_writing_request_count` | Writing requests |

### Dialogue category proportions  (fraction of total turns per row, sums to 1)

| Column | Label |
|---|---|
| `dlg_cat_conceptual_questions_pct` | Conceptual questions |
| `dlg_cat_contextual_questions_pct` | Contextual questions |
| `dlg_cat_editing_request_pct` | Editing requests |
| `dlg_cat_misc_pct` | Miscellaneous |
| `dlg_cat_off_topic_pct` | Off-topic |
| `dlg_cat_provide_context_pct` | Providing context |
| `dlg_cat_verification_pct` | Verification / checking |
| `dlg_cat_writing_request_pct` | Writing requests |

### Submission

| Column | Description |
|---|---|
| `sub_has_submission` | `1` if a submission exists for this assessment's assignment, else `0` |

### Structural  (always kept — not selectable via `--features`)

| Column | Description | Pipeline |
|---|---|---|
| `assessment_code` | Assessment code string (e.g. `e1`, `a3`); consumed by `--encoding`, dropped before model | all |
| `is_exam` | `1.0` for exam rows, `0.0` for assignment rows | `both` only |

### Assessment identity  (produced by `--encoding`, after `build_dataset`)

| `--encoding` value | Columns | Description |
|---|---|---|
| `none` *(default)* | — | No identity encoding |
| `one_hot` | `assess_a2`, `assess_a3`, `assess_a4`, `assess_a5`, `assess_a6`, `assess_a7`, `assess_e1`, `assess_e2`, `assess_e3` | Binary per code |
| `ordinal` | `assess_ordinal` | Integer index (lexicographic) |

> **Note**: `assess_*` / `assess_ordinal` columns are produced by `feature_engineering.engineer_features` *after* `build_dataset` returns, so they cannot be specified via `--features`.

---

## Quick reference — all selectable columns

```
# Counts / lengths
dlg_n_turns
dlg_n_chats
dlg_total_prompt_chars
dlg_avg_prompt_chars
dlg_total_response_chars
dlg_avg_response_chars

# Category counts
dlg_cat_conceptual_questions_count
dlg_cat_contextual_questions_count
dlg_cat_editing_request_count
dlg_cat_misc_count
dlg_cat_off_topic_count
dlg_cat_provide_context_count
dlg_cat_verification_count
dlg_cat_writing_request_count

# Category proportions
dlg_cat_conceptual_questions_pct
dlg_cat_contextual_questions_pct
dlg_cat_editing_request_pct
dlg_cat_misc_pct
dlg_cat_off_topic_pct
dlg_cat_provide_context_pct
dlg_cat_verification_pct
dlg_cat_writing_request_pct

# Submission
sub_has_submission
```

---

## Aggregation logic

All dialogue features are aggregated per `(user_id, assessment_id)` using three UNION branches:

| Row type | Turns included |
|---|---|
| Assignment (a1–a7) | Turns where `dialogue_turns.assignment_id` links to this assessment |
| Exam e1 / e2 | Turns where `dialogue_turns.exam_id` matches this exam |
| Exam e3 | **All** assignment turns for the student's semester (a1–a7 combined) |

A turn on assignment a1 appears in **both** the `a1` row and the `e1` row — intentional, since that activity is predictive of both.

---

## NaN semantics

| Source | Fill | Reason |
|---|---|---|
| `dlg_*` after left-join | `0` | Zero activity is a real, meaningful value |
| `sub_*` after left-join | `0` | No submission = no activity |
| Any remaining NaN in the model | median-imputed per fold | `SimpleImputer` inside the sklearn `Pipeline` |

---

## Feature availability by pipeline

| Column group | exam | assignment | both |
|---|---|---|---|
| `dlg_n_*`, `dlg_*_chars` | ✓ | ✓ | ✓ |
| `dlg_cat_*_count` / `*_pct` | ✓ | ✓ | ✓ |
| `sub_has_submission` | ✓ (0 for exam rows) | ✓ | ✓ (0 for exam rows) |
| `assess_*` / `assess_ordinal` | ✓ | ✓ | ✓ |
| `is_exam` | ✗ | ✗ | ✓ |
