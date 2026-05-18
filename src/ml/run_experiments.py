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

_PRIOR  = ["grade_prior_avg"]
_VOLUME = ["dlg_n_turns", "dlg_n_chats"]
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
_SIM = ["sim_max"]

# Powerset of {volume, da_pct, sim}, each anchored to the prior-grade baseline.
# 2^3 = 8 models. Each runs across 4 cleaning combos → 32 variants total.
FEATURE_GROUPS: dict[str, list[str]] = {
    "m1_prior":          _PRIOR,
    "m2_volume":         _PRIOR + _VOLUME,
    "m3_da":             _PRIOR + _CAT_PCT,
    "m4_sim":            _PRIOR + _SIM,
    "m5_volume_da":      _PRIOR + _VOLUME + _CAT_PCT,
    "m6_volume_sim":     _PRIOR + _VOLUME + _SIM,
    "m7_da_sim":         _PRIOR + _CAT_PCT + _SIM,
    "m8_all":            _PRIOR + _VOLUME + _CAT_PCT + _SIM,
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
