"""Batch experiment runner.

Calls run_training_{pipeline}.py as a subprocess once per variant.

Three models are compared (matching the paper structure):

    Model 1 — Baseline
        Input:  grade_prior_avg  (running average of prior assessment scores)
        Output: current assessment score

    Model 2 — Baseline + total chatbot usage
        Input:  grade_prior_avg, dlg_n_turns
        Output: current assessment score

    Model 3 — Baseline + 8 dialogue-act counts + proportions
        Input:  grade_prior_avg, 8 DA counts, 8 DA proportions
        Output: current assessment score

Each model is run across 4 data-cleaning variants:
    all_students   — no cleaning
    active_only    — drop rows with zero dialogue activity
    nonzero_score  — drop rows where score == 0
    active_nonzero — both cleaning steps

To configure a run:
  1. Set PIPELINE_TYPE, ASSESSMENT_ENCODING, OUTER_SPLITS, INNER_SPLITS at the top.
  2. Run:
        python -m src.ml.run_experiments
        python -m src.ml.run_experiments --outer 5 --inner 5 --output results/
"""
# ===========================================================================
# TOP-LEVEL CONFIGURATION
# ===========================================================================

# "exam" | "assignment" | "both"
PIPELINE_TYPE: str = "exam"

# "none" | "one_hot" | "ordinal"
ASSESSMENT_ENCODING: str = "none"

# CV folds (overridable via CLI)
OUTER_SPLITS: int = 5
INNER_SPLITS: int = 5

# Models to evaluate — None means all registered models
MODELS: list[str] | None = None


# ---------------------------------------------------------------------------
# Feature groups — one entry per model (1, 2, 3)
# ---------------------------------------------------------------------------

_PRIOR    = ["grade_prior_avg"]

_CAT_COUNT = [
    "dlg_cat_conceptual_questions_count",
    "dlg_cat_contextual_questions_count",
    "dlg_cat_editing_request_count",
    "dlg_cat_misc_count",
    "dlg_cat_off_topic_count",
    "dlg_cat_provide_context_count",
    "dlg_cat_verification_count",
    "dlg_cat_writing_request_count",
]
_CAT_PCT = [
    "dlg_cat_conceptual_questions_pct",
    "dlg_cat_contextual_questions_pct",
    "dlg_cat_editing_request_pct",
    "dlg_cat_misc_pct",
    "dlg_cat_off_topic_pct",
    "dlg_cat_provide_context_pct",
    "dlg_cat_verification_pct",
    "dlg_cat_writing_request_pct",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    # Model 1 — baseline: running average of prior scores only
    "model1_baseline":    _PRIOR,

    # Model 2 — baseline + total chatbot usage volume
    "model2_turns":       _PRIOR + ["dlg_n_turns", "dlg_n_chats"],

    # Model 3 — baseline + 8 dialogue-act raw counts
    "model3_da_counts":   _PRIOR + _CAT_COUNT,

    # Model 4 — baseline + volume + 8 dialogue-act proportions
    "model4_turns_da_pct": _PRIOR + ["dlg_n_turns", "dlg_n_chats"] + _CAT_PCT,
}

# ===========================================================================
# Auto-generate variants: each model × 4 cleaning combos
# ===========================================================================

_CLEANING_COMBOS = [
    # (suffix,              drop_no_dialogue, drop_zero_score)
    ("all_students",        False,            False),
    ("active_only",         True,             False),
    ("nonzero_score",       False,            True),
    ("active_nonzero",      True,             True),
]

VARIANTS: list[dict] = [
    {
        "name":             f"{group}__{suffix}",
        "drop_no_dialogue": drop_dlg,
        "drop_zero_score":  drop_zero,
        "features":         FEATURE_GROUPS[group],
    }
    for group in FEATURE_GROUPS
    for suffix, drop_dlg, drop_zero in _CLEANING_COMBOS
]

# ===========================================================================
# Runner — no need to edit below this line
# ===========================================================================
import argparse
import subprocess
import sys
from pathlib import Path

from src.ml import config as cfg
from src.ml.config import ASSIGNMENT_ASSESSMENT_CODES, EXAM_ASSESSMENT_CODES
from src.ml.db import load_engine
from src.ml.model_training import make_run_dir

_MODULE = {
    "exam":       "src.ml.run_training_exam",
    "assignment": "src.ml.run_training_assignment",
    "both":       "src.ml.run_training_both",
}

_PIPELINE_CODES = {
    "exam":       EXAM_ASSESSMENT_CODES,
    "assignment": ASSIGNMENT_ASSESSMENT_CODES,
    "both":       EXAM_ASSESSMENT_CODES + ASSIGNMENT_ASSESSMENT_CODES,
}


def build_cmd(variant: dict, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-m", _MODULE[PIPELINE_TYPE]]
    cmd += ["--encoding", ASSESSMENT_ENCODING]
    if variant["drop_no_dialogue"]:
        cmd += ["--drop-no-dialogue"]
    if variant["drop_zero_score"]:
        cmd += ["--drop-zero-score"]
    if variant["features"] is not None:
        cmd += ["--features"] + variant["features"]
    if MODELS:
        cmd += ["--models"] + MODELS
    cmd += ["--outer", str(args.outer)]
    cmd += ["--inner", str(args.inner)]
    cmd += ["--output", str(args.output)]
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Batch runner → {_MODULE[PIPELINE_TYPE]}  ({len(VARIANTS)} variants)",
    )
    parser.add_argument("--outer",  type=int,  default=OUTER_SPLITS, metavar="K")
    parser.add_argument("--inner",  type=int,  default=INNER_SPLITS, metavar="K")
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


def print_best_variant(run_dirs: list[Path], variant_names: list[str]) -> None:
    """Print a ranked summary of all completed variants for this experiment run."""
    import json

    rows = []
    for run_dir, variant_name in zip(run_dirs, variant_names):
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        d = json.loads(summary_path.read_text())
        best = d["best_model"]
        rows.append({
            "variant":    variant_name,
            "model":      best["name"],
            "mean_rmse":  best["mean_rmse"],
            "std_rmse":   best["std_rmse"],
            "n_samples":  d["dataset"]["n_samples"],
        })

    if not rows:
        print("No completed variants to summarise.")
        return

    rows.sort(key=lambda r: r["mean_rmse"])
    width = max(len(r["variant"]) for r in rows)

    print(f"\n{'='*70}")
    print(f"Experiment summary  |  pipeline={PIPELINE_TYPE}  encoding={ASSESSMENT_ENCODING}")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Variant':<{width}}  {'Model':<20}  {'RMSE':>8}  {'±std':>7}  {'n':>5}")
    print(f"{'-'*5} {'-'*width}  {'-'*20}  {'-'*8}  {'-'*7}  {'-'*5}")
    for rank, r in enumerate(rows, 1):
        print(
            f"{rank:<5} {r['variant']:<{width}}  {r['model']:<20}  "
            f"{r['mean_rmse']:>8.4f}  {r['std_rmse']:>7.4f}  {r['n_samples']:>5}"
        )

    best = rows[0]
    print(f"\nBest: [{best['variant']}]  model={best['model']}  "
          f"RMSE={best['mean_rmse']:.4f} ± {best['std_rmse']:.4f}  n={best['n_samples']}")


def main() -> None:
    args = parse_args()

    print(f"Pipeline   : {PIPELINE_TYPE}  ({_MODULE[PIPELINE_TYPE]})")
    print(f"Encoding   : {ASSESSMENT_ENCODING}")
    print(f"CV         : outer={args.outer}  inner={args.inner}")
    print(f"Output     : {args.output}")
    print(f"Variants   : {len(VARIANTS)}")
    print()

    n_ok = n_fail = 0
    completed_dirs: list[Path] = []
    completed_names: list[str] = []

    for i, variant in enumerate(VARIANTS, 1):
        cmd = build_cmd(variant, args)
        print(f"[{i:02d}/{len(VARIANTS)}] {variant['name']}")
        print("  " + " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})\n")
            n_fail += 1
        else:
            print(f"  OK\n")
            n_ok += 1
            # Identify the run_dir that was just created (newest matching dir)
            run_dirs = sorted(
                args.output.glob(f"{PIPELINE_TYPE}_*"),
                key=lambda p: p.stat().st_mtime,
            )
            if run_dirs:
                completed_dirs.append(run_dirs[-1])
                completed_names.append(variant["name"])

    print(f"Done — {n_ok} succeeded, {n_fail} failed.")
    print_best_variant(completed_dirs, completed_names)

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
