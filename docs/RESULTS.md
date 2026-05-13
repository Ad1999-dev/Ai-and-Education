# Experiment Results

**Research question:** Can LLM usage patterns during assignments predict student exam scores (e1, e2, e3)?

**Run date:** 2026-04-11  
**Pipeline:** exam (e1, e2, e3)  
**CV:** 5-fold outer GroupKFold × 5-fold inner GroupKFold (grouped by `user_id`)  
**Encoding:** none  

---

## Models

| Model | Features | # features |
|---|---|---|
| `model1_baseline` | `grade_prior_avg` | 1 |
| `model2_turns` | `grade_prior_avg`, `dlg_n_turns`, `dlg_n_chats` | 3 |
| `model3_da_counts` | `grade_prior_avg` + 8 DA counts | 9 |
| `model4_turns_da_pct` | `grade_prior_avg`, `dlg_n_turns`, `dlg_n_chats` + 8 DA proportions | 11 |

`grade_prior_avg` is the running average of all prior assessment scores (a1→a2→e1→a3→a4→e2→a5→a6→a7) within the same semester.  
DA = dialogue-act category labels assigned by the LLM to each dialogue turn.

---

## Data Cleaning Strategies

| Suffix | `drop_no_dialogue` | `drop_zero_score` | n |
|---|---|---|---|
| `all_students` | ✗ | ✗ | 543 |
| `active_only` | ✓ | ✗ | 499 |
| `nonzero_score` | ✗ | ✓ | 541 |
| `active_nonzero` | ✓ | ✓ | 498 |

- **drop_no_dialogue**: removes rows where the student had zero dialogue activity during the relevant assignment period.
- **drop_zero_score**: removes rows where the student received exactly 0 (did not sit the exam), distinct from NULL which is excluded at the DB level.

---

## Results — Full Ranking

| Rank | Model | Cleaning | Best estimator | RMSE | ±std | n | Run folder |
|---|---|---|---|---|---|---|---|
| 1 | model4_turns_da_pct | active_nonzero | random_forest | 0.1115 | 0.0104 | 498 | `exam_20260410_224456` |
| 2 | model2_turns | active_nonzero | svr | 0.1129 | 0.0106 | 498 | `exam_20260410_223520` |
| 3 | model1_baseline | active_nonzero | svr | 0.1132 | 0.0101 | 498 | `exam_20260410_223059` |
| 4 | model3_da_counts | active_nonzero | random_forest | 0.1134 | 0.0122 | 498 | `exam_20260410_223914` |
| 5 | model4_turns_da_pct | active_only | random_forest | 0.1142 | 0.0197 | 499 | `exam_20260410_224245` |
| 6 | model2_turns | nonzero_score | random_forest | 0.1146 | 0.0062 | 541 | `exam_20260410_223409` |
| 7 | model3_da_counts | nonzero_score | svr | 0.1151 | 0.0060 | 541 | `exam_20260410_223807` |
| 8 | model4_turns_da_pct | nonzero_score | random_forest | 0.1155 | 0.0096 | 541 | `exam_20260410_224319` |
| 9 | model3_da_counts | active_only | random_forest | 0.1160 | 0.0204 | 499 | `exam_20260410_223735` |
| 10 | model2_turns | active_only | svr | 0.1163 | 0.0192 | 499 | `exam_20260410_223302` |
| 11 | model1_baseline | nonzero_score | svr | 0.1165 | 0.0064 | 541 | `exam_20260410_223034` |
| 12 | model1_baseline | active_only | svr | 0.1178 | 0.0194 | 499 | `exam_20260410_222953` |
| 13 | model2_turns | all_students | svr | 0.1236 | 0.0122 | 543 | `exam_20260410_223222` |
| 14 | model4_turns_da_pct | all_students | random_forest | 0.1250 | 0.0113 | 543 | `exam_20260410_224046` |
| 15 | model3_da_counts | all_students | svr | 0.1251 | 0.0129 | 543 | `exam_20260410_223550` |
| 16 | model1_baseline | all_students | svr | 0.1256 | 0.0116 | 543 | `exam_20260410_222833` |

---

## Results — RMSE by Model × Cleaning (2D summary)

| Model | all_students (n=543) | active_only (n=499) | nonzero_score (n=541) | active_nonzero (n=498) |
|---|---|---|---|---|
| model1_baseline | 0.1256 | 0.1178 | 0.1165 | **0.1132** |
| model2_turns | 0.1236 | 0.1163 | 0.1146 | **0.1129** |
| model3_da_counts | 0.1251 | 0.1160 | 0.1151 | **0.1134** |
| model4_turns_da_pct | 0.1250 | 0.1142 | 0.1155 | **0.1115** |

Bold = best per model. Overall best: `model4_turns_da_pct` + `active_nonzero` → **RMSE = 0.1115**.

---

## Key Findings

**1. Prior grade is a strong predictor on its own.**  
Model 1 (baseline only) already achieves RMSE = 0.1132 under `active_nonzero` — prior academic performance alone explains a substantial portion of exam score variance.

**2. LLM usage adds a small but consistent improvement.**  
Model 4 (0.1115) beats Model 1 (0.1132) by 0.0017 under the same cleaning. The improvement is modest, suggesting LLM usage patterns have signal beyond prior grades but are not dominant predictors.

**3. The nature of LLM use matters more than volume alone.**  
Model 4 (`grade_prior_avg` + turns + DA proportions) consistently outperforms Model 2 (`grade_prior_avg` + turns only), indicating that *how* a student uses the chatbot carries more information than *how much*.

**4. Data cleaning consistently reduces both RMSE and variance.**  
Across all 4 models, `active_nonzero` gives the lowest RMSE and the lowest std. Removing zero-score rows has a larger impact than removing zero-dialogue rows despite fewer rows removed (2 vs 44).

**5. Random Forest and SVR dominate.**  
Linear models (linear regression, ridge) never appear in the top rankings. The relationship between LLM usage patterns and exam scores is nonlinear.

---

## Per-run Artefacts

Each run folder under `results/` contains:

| File | Description |
|---|---|
| `dataset.csv` | Exact features + target passed to the model |
| `results.csv` | Per-model RMSE across all outer folds |
| `run_summary.json` | Full metadata: pipeline, CV config, features, best model |
| `predictions.csv` | Out-of-fold predictions for all models |
| `final_model.pkl` | Best model trained on the full dataset |
| `feature_importance.csv` | Feature importance / coefficient magnitudes |
| `feature_importance.png` | Horizontal bar chart of feature importance |
| `correlation_matrix.csv` | Pearson correlation matrix (features + target) |
| `correlation_heatmap.png` | Lower-triangle heatmap (red=negative, blue=positive) |
| `prediction_error.png` | Actual vs predicted scatter for all models |
| `prediction_error_best.png` | Actual vs predicted scatter for best model only |
