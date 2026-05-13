"""Central configuration for the ML pipeline.

Edit this file to change the default behaviour.  All settings can be
overridden via CLI flags in the run_training_*.py entry points.
"""
from enum import Enum

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# ---------------------------------------------------------------------------
# Assessment codes
# ---------------------------------------------------------------------------
EXAM_ASSESSMENT_CODES: list[str] = ["e1", "e2", "e3"]
# a1 excluded from assignment pipeline: it has no prior scores so grade_prior_avg
# is always NaN — it would contribute no signal to Model 1 / 2 / 3.
ASSIGNMENT_ASSESSMENT_CODES: list[str] = ["a2", "a3", "a4", "a5", "a6", "a7"]

# ---------------------------------------------------------------------------
# Temporal ordering of assessments (used to compute grade_prior_avg)
# Based on course schedule: a1→a2→e1→a3→a4→e2→a5→a6→a7→e3
# ---------------------------------------------------------------------------
ASSESSMENT_ORDER: list[str] = ["a1", "a2", "e1", "a3", "a4", "e2", "a5", "a6", "a7", "e3"]

# For each assessment code, the list of codes that come strictly before it.
# grade_prior_avg for a given row = mean of the student's scores on these codes
# within the same semester.  a1 has no prior → grade_prior_avg = NaN.
PRIOR_ASSESSMENTS: dict[str, list[str]] = {
    code: ASSESSMENT_ORDER[:i]
    for i, code in enumerate(ASSESSMENT_ORDER)
}


# ---------------------------------------------------------------------------
# Strategy enums — select how each feature dimension is computed
# ---------------------------------------------------------------------------
class AssessmentEncoding(str, Enum):
    ONE_HOT  = "one_hot"   # binary column per assessment code
    ORDINAL  = "ordinal"   # single integer column (lexicographic order)
    NONE     = "none"      # drop assessment identity entirely


class DialogueStrategy(str, Enum):
    PER_ASSESSMENT = "per_assessment"  # per assessment: assignment turns via assignment_id,
                                       # exam turns via exam_id (a1/a2→e1, a3/a4→e2)


class SubmissionStrategy(str, Enum):
    PER_ASSIGNMENT = "per_assignment"  # single row for the linked assignment_id


# ---------------------------------------------------------------------------
# Feature blocks — set False to exclude a block entirely
# ---------------------------------------------------------------------------
FEATURE_CONFIG: dict[str, bool] = {
    "dialogue_counts":     True,   # n_turns, n_chats, n_assignments_with_dialogue
    "dialogue_lengths":    True,   # avg/total prompt & response character lengths
    "dialogue_categories": True,   # llm_label proportions
    "submission_features": True,   # lines of code, file counts, submission types
}

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
OUTER_CV_SPLITS: int = 5
INNER_CV_SPLITS: int = 5
RANDOM_STATE:    int = 42
SCORING:         str = "neg_root_mean_squared_error"

# ---------------------------------------------------------------------------
# Model registry
# Keys in param_grid must use the "model__" prefix (sklearn Pipeline convention).
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, tuple] = {
    "linear_regression": (
        LinearRegression(),
        {},
    ),
    "ridge": (
        Ridge(),
        {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ),
    "decision_tree": (
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_leaf": [1, 3, 5],
        },
    ),
    "random_forest": (
        RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        {
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 3, 5],
        },
    ),
    "gradient_boosting": (
        GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
        {
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 5],
        },
    ),
    "svr": (
        SVR(),
        {
            "model__C": [0.1, 1.0, 10.0],
            "model__epsilon": [0.05, 0.1, 0.2],
            "model__kernel": ["rbf", "linear"],
        },
    ),
    "knn": (
        KNeighborsRegressor(),
        {
            "model__n_neighbors": [3, 5, 7, 10],
            "model__weights": ["uniform", "distance"],
        },
    ),
}
