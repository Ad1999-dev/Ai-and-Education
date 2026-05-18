from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = PROJECT_ROOT / "results"
DEFAULT_IMAGES = PROJECT_ROOT / "report" / "images" / "modeling"


# ---------------------------------------------------------------------------
# Variant identification
# ---------------------------------------------------------------------------

MODEL_FAMILY_ORDER: list[str] = [
    "m1_prior",
    "m2_volume",
    "m3_da",
    "m4_sim",
    "m5_volume_da",
    "m6_volume_sim",
    "m7_da_sim",
    "m8_all",
]

CLEANING_BY_N: dict[int, str] = {
    543: "all_students",
    499: "active_only",
    541: "nonzero_score",
    498: "active_nonzero",
}
CLEANING_ORDER: list[str] = [
    "all_students",
    "active_only",
    "nonzero_score",
    "active_nonzero",
]

_FAMILY_BY_FLAGS: dict[tuple[bool, bool, bool], str] = {
    (False, False, False): "m1_prior",
    (True,  False, False): "m2_volume",
    (False, True,  False): "m3_da",
    (False, False, True ): "m4_sim",
    (True,  True,  False): "m5_volume_da",
    (True,  False, True ): "m6_volume_sim",
    (False, True,  True ): "m7_da_sim",
    (True,  True,  True ): "m8_all",
}


def identify_family(feature_columns: list[str]) -> str:
    cols = set(feature_columns)
    has_volume = "dlg_n_turns" in cols
    has_da = any(c.startswith("dlg_cat_") and c.endswith("_pct") for c in cols)
    has_sim = "sim_max" in cols
    return _FAMILY_BY_FLAGS[(has_volume, has_da, has_sim)]


def identify_cleaning(n_samples: int) -> str | None:
    return CLEANING_BY_N.get(n_samples)


def load_variants(results_dir: Path) -> dict[tuple[str, str], dict]:
    """Return dict mapping `(family, cleaning)` to the latest run_summary."""
    variants: dict[tuple[str, str], dict] = {}
    for run_dir in sorted(results_dir.glob("exam_*")):
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            print(f"  WARNING: could not parse {summary_path}")
            continue
        feat = summary.get("feature_columns", [])
        n = summary.get("dataset", {}).get("n_samples")
        if n not in CLEANING_BY_N:
            continue
        try:
            family = identify_family(feat)
        except KeyError:
            continue
        cleaning = identify_cleaning(n)
        if cleaning is None:
            continue
        # Keep the most recent run for each (family, cleaning) key.
        existing = variants.get((family, cleaning))
        if existing is None or summary.get("run_at", "") >= existing.get("run_at", ""):
            summary["_run_dir"] = run_dir
            variants[(family, cleaning)] = summary
    return variants


# ---------------------------------------------------------------------------
# Figure 2: ablation heatmap
# ---------------------------------------------------------------------------

def make_heatmap(variants: dict, output_path: Path) -> None:
    grid = np.full((len(MODEL_FAMILY_ORDER), len(CLEANING_ORDER)), np.nan)
    for i, fam in enumerate(MODEL_FAMILY_ORDER):
        for j, clean in enumerate(CLEANING_ORDER):
            v = variants.get((fam, clean))
            if v is None:
                continue
            grid[i, j] = v["best_model"]["mean_rmse"]

    if np.all(np.isnan(grid)):
        print(f"  WARNING: no variants found; skipping {output_path.name}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean RMSE")

    ax.set_xticks(range(len(CLEANING_ORDER)))
    ax.set_xticklabels(CLEANING_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(len(MODEL_FAMILY_ORDER)))
    ax.set_yticklabels(MODEL_FAMILY_ORDER)

    best_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
    mid = np.nanmean(grid)

    for i in range(len(MODEL_FAMILY_ORDER)):
        for j in range(len(CLEANING_ORDER)):
            val = grid[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center",
                        color="gray", fontsize=9)
                continue
            label = f"{val:.4f}"
            if (i, j) == best_idx:
                label = "★ " + label
            color = "white" if val > mid else "black"
            ax.text(j, i, label, ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    ax.set_xlabel("Cleaning regime")
    ax.set_ylabel("Model family")
    ax.set_title("Ablation RMSE — 8 model families × 4 cleaning regimes")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: marginal-contribution chart
# ---------------------------------------------------------------------------

def make_marginal_chart(variants: dict, output_path: Path) -> None:
    baseline_rmse: dict[str, float] = {}
    for clean in CLEANING_ORDER:
        v = variants.get(("m1_prior", clean))
        if v is not None:
            baseline_rmse[clean] = v["best_model"]["mean_rmse"]

    if not baseline_rmse:
        print(f"  WARNING: no m1_prior runs; skipping {output_path.name}")
        return

    families = [f for f in MODEL_FAMILY_ORDER if f != "m1_prior"]
    deltas: dict[str, dict[str, float]] = {}
    for fam in families:
        per_cleaning: dict[str, float] = {}
        for clean in baseline_rmse:
            v = variants.get((fam, clean))
            if v is None:
                continue
            per_cleaning[clean] = (
                v["best_model"]["mean_rmse"] - baseline_rmse[clean]
            )
        if per_cleaning:
            deltas[fam] = per_cleaning

    families_present = [f for f in families if f in deltas]
    if not families_present:
        print(f"  WARNING: no non-baseline variants; skipping {output_path.name}")
        return

    avgs = np.array(
        [np.mean(list(deltas[f].values())) for f in families_present]
    )
    mins = np.array([min(deltas[f].values()) for f in families_present])
    maxs = np.array([max(deltas[f].values()) for f in families_present])

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(families_present))
    colors = ["tab:green" if v < 0 else "tab:red" for v in avgs]
    ax.barh(
        y, avgs, color=colors,
        xerr=[avgs - mins, maxs - avgs],
        error_kw=dict(ecolor="black", lw=1, capsize=3),
    )
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(families_present)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE delta vs m1_prior  (negative = improvement)")
    ax.set_title(
        "Marginal contribution of each model family\n"
        "(average across cleanings; whiskers = min/max range)"
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


# ---------------------------------------------------------------------------
# Figure 4: estimator-selection histogram
# ---------------------------------------------------------------------------

def make_estimator_selection_chart(variants: dict, output_path: Path) -> None:
    counts: Counter[str] = Counter()
    for v in variants.values():
        counts[v["best_model"]["name"]] += 1

    if not counts:
        print(f"  WARNING: no variants found; skipping {output_path.name}")
        return

    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in ordered]
    n_wins = [v for _, v in ordered]

    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(names))
    bars = ax.barh(y, n_wins, color="steelblue")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(f"Number of variants (out of {sum(n_wins)})")
    ax.set_title("Estimator family selected by inner CV")
    for i, b in enumerate(bars):
        ax.text(
            b.get_width() + 0.3,
            b.get_y() + b.get_height() / 2,
            str(n_wins[i]), va="center", fontsize=9,
        )
    ax.set_xlim(0, max(n_wins) + 2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


# ---------------------------------------------------------------------------
# Figures 1, 5, 6: copies of per-run plots
# ---------------------------------------------------------------------------

def copy_run_plot(
    variants: dict,
    family: str,
    cleaning: str,
    src_name: str,
    dst_path: Path,
) -> None:
    v = variants.get((family, cleaning))
    if v is None:
        print(f"  WARNING: no run for ({family}, {cleaning}); skipping {dst_path.name}")
        return
    src = v["_run_dir"] / src_name
    if not src.exists():
        print(f"  WARNING: {src} does not exist; skipping {dst_path.name}")
        return
    shutil.copyfile(src, dst_path)
    print(f"  → {dst_path}  (copied from {src.relative_to(PROJECT_ROOT)})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS,
        help="Directory containing `exam_*` run subdirectories",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_IMAGES,
        help="Where to write the report-ready PNGs",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        raise SystemExit(f"results dir not found: {args.results_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading runs from {args.results_dir}")
    variants = load_variants(args.results_dir)
    if not variants:
        raise SystemExit("No usable runs found.")

    print(f"Identified {len(variants)} unique (family, cleaning) variants:")
    for key in sorted(variants):
        fam, clean = key
        rmse = variants[key]["best_model"]["mean_rmse"]
        est = variants[key]["best_model"]["name"]
        print(f"  {fam:18s} {clean:18s}  RMSE={rmse:.4f}  estimator={est}")

    missing = [
        (fam, clean)
        for fam in MODEL_FAMILY_ORDER
        for clean in CLEANING_ORDER
        if (fam, clean) not in variants
    ]
    if missing:
        print(f"\nMissing cells ({len(missing)} / 32):")
        for fam, clean in missing:
            print(f"  {fam:18s} {clean}")

    print(f"\nGenerating figures in {args.output_dir}")
    make_heatmap(variants, args.output_dir / "ablation_heatmap.png")
    make_marginal_chart(variants, args.output_dir / "marginal_contribution.png")
    make_estimator_selection_chart(
        variants, args.output_dir / "estimator_selection.png"
    )
    copy_run_plot(
        variants, "m1_prior", "active_nonzero",
        "prediction_error_best.png",
        args.output_dir / "baseline_pred_error.png",
    )
    copy_run_plot(
        variants, "m5_volume_da", "active_nonzero",
        "feature_importance.png",
        args.output_dir / "feature_importance.png",
    )
    copy_run_plot(
        variants, "m5_volume_da", "active_nonzero",
        "prediction_error_best.png",
        args.output_dir / "headline_pred_error.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
