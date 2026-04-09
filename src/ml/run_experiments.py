"""Batch experiment runner.

Calls run_training_{pipeline}.py as a subprocess once per variant.
All logic stays in the run_training scripts — this file is a shortcut for
sweeping over feature combinations and the drop-no-dialogue cleaning option.

To configure a run:
  1. Set PIPELINE_TYPE, ASSESSMENT_ENCODING, OUTER_SPLITS, INNER_SPLITS at the top.
  2. Toggle groups on/off in FEATURE_GROUPS (True = include in sweep).
  3. Run:
        python -m src.ml.run_experiments
        python -m src.ml.run_experiments --outer 10 --inner 5 --output results/

Each variant name is  <group_name>__all_students  or  <group_name>__active_only.
Results land in  results/exp_<pipeline>_YYYYMMDD_HHMMSS/<variant_name>/.
"""
# ===========================================================================
# TOP-LEVEL CONFIGURATION
# ===========================================================================

# "exam" | "assignment" | "both"
PIPELINE_TYPE: str = "both"

# "none" | "one_hot" | "ordinal"
ASSESSMENT_ENCODING: str = "none"

# CV folds (overridable via CLI)
OUTER_SPLITS: int = 5
INNER_SPLITS: int = 5

# Models to evaluate — None means all registered models
MODELS: list[str] | None = None

# ---------------------------------------------------------------------------
# Feature groups.
# Set a group's value to True to include it in the sweep, False to skip.
#
# Full selectable column pool:
#   dlg_n_turns, dlg_n_chats
#   dlg_total_prompt_chars, dlg_avg_prompt_chars
#   dlg_total_response_chars, dlg_avg_response_chars
#   dlg_cat_conceptual_questions_count/pct
#   dlg_cat_contextual_questions_count/pct
#   dlg_cat_editing_request_count/pct
#   dlg_cat_misc_count/pct
#   dlg_cat_off_topic_count/pct
#   dlg_cat_provide_context_count/pct
#   dlg_cat_verification_count/pct
#   dlg_cat_writing_request_count/pct
#   sub_has_submission
# ---------------------------------------------------------------------------

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
_COUNTS   = ["dlg_n_turns", "dlg_n_chats"]
_PROMPT   = ["dlg_total_prompt_chars", "dlg_avg_prompt_chars"]
_RESPONSE = ["dlg_total_response_chars", "dlg_avg_response_chars"]
_LENGTHS  = _PROMPT + _RESPONSE
_SUB      = ["sub_has_submission"]

FEATURE_GROUPS: dict[str, list[str] | None] = {
    # None = keep all available columns (no --features flag passed)
    "all":                    None,

    # --- Volume of usage ---
    "counts":                 _COUNTS,
    "prompt_length":          _PROMPT,
    "response_length":        _RESPONSE,
    "lengths":                _LENGTHS,
    "counts_lengths":         _COUNTS + _LENGTHS,

    # --- Nature of usage (label proportions) ---
    "cat_pct":                _CAT_PCT,
    "cat_count":              _CAT_COUNT,
    "cat_all":                _CAT_COUNT + _CAT_PCT,

    # --- Combined ---
    "counts_cat_pct":         _COUNTS + _CAT_PCT,
    "counts_cat_all":         _COUNTS + _CAT_COUNT + _CAT_PCT,
    "counts_lengths_cat_pct": _COUNTS + _LENGTHS + _CAT_PCT,
    "counts_lengths_cat_all": _COUNTS + _LENGTHS + _CAT_COUNT + _CAT_PCT,

    # --- With submission flag ---
    "all_with_sub":           None,   # None → all columns including sub_has_submission
    "counts_sub":             _COUNTS + _SUB,
    "counts_lengths_sub":     _COUNTS + _LENGTHS + _SUB,
}

# Toggle groups: set False to skip without deleting the entry
_ENABLED: dict[str, bool] = {
    "all":                    True,
    "counts":                 True,
    "prompt_length":          True,
    "response_length":        True,
    "lengths":                True,
    "counts_lengths":         True,
    "cat_pct":                True,
    "cat_count":              True,
    "cat_all":                True,
    "counts_cat_pct":         True,
    "counts_cat_all":         True,
    "counts_lengths_cat_pct": True,
    "counts_lengths_cat_all": True,
    "all_with_sub":           True,
    "counts_sub":             True,
    "counts_lengths_sub":     True,
}

# ===========================================================================
# Auto-generate variants: each enabled group × {all_students, active_only}
# ===========================================================================

VARIANTS: list[dict] = [
    {
        "name":             f"{group}__{suffix}",
        "drop_no_dialogue": drop,
        "features":         FEATURE_GROUPS[group],
    }
    for group in FEATURE_GROUPS
    if _ENABLED.get(group, True)
    for suffix, drop in [("all_students", False), ("active_only", True)]
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


def main() -> None:
    args = parse_args()

    print(f"Pipeline   : {PIPELINE_TYPE}  ({_MODULE[PIPELINE_TYPE]})")
    print(f"Encoding   : {ASSESSMENT_ENCODING}")
    print(f"CV         : outer={args.outer}  inner={args.inner}")
    print(f"Output     : {args.output}")
    print(f"Variants   : {len(VARIANTS)}")
    print()

    n_ok = n_fail = 0
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

    print(f"Done — {n_ok} succeeded, {n_fail} failed.")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
