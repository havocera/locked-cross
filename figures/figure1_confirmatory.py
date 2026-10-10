"""Confirmatory transfer gaps and screen-dependent model rankings."""

from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from figstyle import PALETTE, WIDE, figure_grid, panel_labels, save, use_style  # noqa: E402


def main() -> None:
    use_style()
    gap = pd.read_csv(ROOT / "results" / "holdout_confirmation" / "matched_gap.csv")
    summary = pd.read_csv(ROOT / "results" / "holdout_confirmation" / "holdout_summary.csv")

    gap_order = [
        ("GDSCv1", "internal_random_overlap", "GDSCv1 shared"),
        ("GDSCv1", "internal_random_exact_pair", "GDSCv1 exact"),
        ("NCI60", "internal_random_overlap", "NCI60 shared"),
        ("NCI60", "internal_random_exact_pair", "NCI60 exact"),
    ]
    gap_rows = []
    for study, regime, label in gap_order:
        row = gap.loc[(gap.test_study == study) & (gap.internal_regime == regime)].iloc[0]
        gap_rows.append((label, row.gap_internal_minus_external, row.ci_low, row.ci_high))

    models = ["selected_inductive_mf", "itens_k16", "drug_knn", "lightgbm", "ridge_bilinear"]
    labels = {
        "selected_inductive_mf": "Inductive MF",
        "itens_k16": "ITENS",
        "drug_knn": "Drug kNN",
        "lightgbm": "LightGBM",
        "ridge_bilinear": "Ridge",
    }

    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.48)
    ax_gap, ax_models = axes
    panel_labels(axes)

    y = np.arange(len(gap_rows))[::-1]
    values = np.asarray([row[1] for row in gap_rows])
    low = np.asarray([row[2] for row in gap_rows])
    high = np.asarray([row[3] for row in gap_rows])
    colors = [PALETTE["blue"], PALETTE["blue"], PALETTE["orange"], PALETTE["orange"]]
    for value, yy, lo, hi, color in zip(values, y, low, high, colors):
        ax_gap.errorbar(
            value, yy, xerr=[[value - lo], [hi - value]], fmt="o",
            color=color, elinewidth=1.2, capsize=2, markersize=4,
        )
    ax_gap.axvline(0.10, color="#7F7F7F", linestyle="--", linewidth=1)
    ax_gap.set_yticks(y, [row[0] for row in gap_rows])
    ax_gap.set_xlabel("Internal minus external macro Spearman")
    ax_gap.set_ylabel("Matched evaluation domain")
    ax_gap.set_xlim(0, 0.62)
    ax_gap.grid(axis="x")
    ax_gap.grid(axis="y", visible=False)
    ax_gap.text(
        0.105, 0.04, "0.10 threshold", transform=ax_gap.get_xaxis_transform(),
        fontsize=6, color="#5F5F5F", rotation=90, va="bottom", ha="left",
        backgroundcolor="white",
    )

    y_models = np.arange(len(models))[::-1]
    offsets = {"GDSCv1": 0.10, "NCI60": -0.10}
    screen_colors = {"GDSCv1": PALETTE["blue"], "NCI60": PALETTE["orange"]}
    for study in ("GDSCv1", "NCI60"):
        rows = []
        for model in models:
            row = summary.loc[
                (summary.test_study == study)
                & (summary.regime == "cross_study_overlap")
                & (summary.model == model)
            ].iloc[0]
            rows.append(row)
        value = np.asarray([row.macro_spearman for row in rows])
        lo = np.asarray([row.ci_low for row in rows])
        hi = np.asarray([row.ci_high for row in rows])
        yy = y_models + offsets[study]
        ax_models.errorbar(
            value, yy, xerr=np.vstack([value - lo, hi - value]), fmt="o",
            color=screen_colors[study], label=study, capsize=2, markersize=3.5,
        )
    ax_models.axvline(0, color="#7F7F7F", linewidth=0.8)
    ax_models.set_yticks(y_models, [labels[model] for model in models])
    ax_models.set_xlabel("External macro per-drug Spearman")
    ax_models.set_ylabel("Model")
    ax_models.set_xlim(-0.08, 0.30)
    ax_models.legend(loc="upper left")
    ax_models.grid(axis="x")
    ax_models.grid(axis="y", visible=False)

    save(fig, str(HERE / "figure1_confirmatory"), close=False)
    preview = os.environ.get("FIGURE_PREVIEW_PATH")
    if preview:
        fig.savefig(preview, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
