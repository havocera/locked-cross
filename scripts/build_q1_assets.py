"""Build Q1-level manuscript tables and data-backed figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures_q1"
TABLES = ROOT / "tables_q1"
sys.path.insert(0, str(ROOT / "figures"))
from figstyle import PALETTE, WIDE, figure_grid, panel_labels, save, use_style  # noqa: E402


MODEL_ORDER = [
    "additive_matched", "ridge_bilinear", "histgb", "lightgbm",
    "inductive_mf", "cell_knn", "drug_knn", "product_knn", "itens_k16",
]
MODEL_LABEL = {
    "additive_matched": "Additive",
    "ridge_bilinear": "Ridge",
    "histgb": "HistGB",
    "lightgbm": "LightGBM",
    "inductive_mf": "Inductive MF",
    "selected_inductive_mf": "Locked MF",
    "cell_knn": "Cell kNN",
    "drug_knn": "Drug kNN",
    "product_knn": "Product kNN",
    "itens_k16": "ITENS",
}
COLORS = {
    "Additive": "#7F7F7F", "Ridge": PALETTE["black"], "HistGB": PALETTE["green"],
    "LightGBM": PALETTE["cyan"], "Inductive MF": PALETTE["blue"],
    "Locked MF": PALETTE["blue"], "Cell kNN": PALETTE["yellow"],
    "Drug kNN": PALETTE["orange"], "Product kNN": PALETTE["red"],
    "ITENS": PALETTE["purple"],
}


def write_table(frame: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / f"{name}.csv", index=False)
    values = frame.fillna("NA").astype(str)
    lines = [
        "| " + " | ".join(values.columns) + " |",
        "| " + " | ".join(["---"] * len(values.columns)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in values.to_numpy())
    (TABLES / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


def load_feature_entities(directory: Path, name: str) -> set[str]:
    return set(pd.read_csv(directory / f"{name}_entities.csv", dtype=str).entity_id)


def build_tables() -> dict[str, pd.DataFrame]:
    development = pd.read_csv(ROOT / "data" / "confirmatory" / "responses.csv", dtype={"cell_id": str, "drug_id": str})
    holdout = pd.read_csv(ROOT / "data" / "holdout_processed" / "responses.csv", dtype={"cell_id": str, "drug_id": str})
    development_flow = pd.read_csv(ROOT / "data" / "confirmatory" / "identifier_filter_flow.csv")
    holdout_flow = pd.read_csv(ROOT / "data" / "holdout_processed" / "identifier_filter_flow.csv")
    holdout_coverage = pd.read_csv(ROOT / "data" / "holdout_features" / "holdout_feature_coverage.csv")
    dev_cells = load_feature_entities(ROOT / "data" / "processed_multistudy" / "features", "cell_expression")
    dev_drugs = load_feature_entities(ROOT / "data" / "processed_multistudy" / "features", "drug_descriptors")

    rows = []
    for row in development_flow.itertuples(index=False):
        frame = development.loc[development.study == row.study]
        complete = frame.loc[frame.cell_id.isin(dev_cells) & frame.drug_id.isin(dev_drugs)]
        rows.append({
            "Study": row.study, "Role": "Development", "Time (h)": "Study-defined",
            "Input fitted pairs": row.input_rows, "Stable-ID pairs": row.retained_rows,
            "Feature-complete pairs": len(complete), "Cells": complete.cell_id.nunique(),
            "Drugs": complete.drug_id.nunique(),
        })
    for row in holdout_flow.itertuples(index=False):
        coverage = holdout_coverage.loc[holdout_coverage.study == row.study].iloc[0]
        rows.append({
            "Study": row.study, "Role": "Untouched confirmation",
            "Time (h)": int(row.selected_time), "Input fitted pairs": row.input_rows,
            "Stable-ID pairs": row.retained_rows,
            "Feature-complete pairs": int(coverage.feature_complete_pairs),
            "Cells": int(coverage.feature_complete_cells), "Drugs": int(coverage.feature_complete_drugs),
        })
    table1 = pd.DataFrame(rows)
    write_table(table1, "table1_dataset_flow")

    table2 = pd.DataFrame([
        ["Additive bias", "Entity IDs", "No", "Yes", "No", "No", "Global fallback for unseen drugs; no drug-specific effect"],
        ["Ridge", "Expression + descriptors + bilinear terms", "Yes", "Yes", "Yes", "No", "Linear extrapolation"],
        ["HistGradientBoosting", "Same pair features", "Yes", "Yes", "Yes", "No", "Tree ensemble"],
        ["LightGBM", "Same pair features", "Yes", "Yes", "Yes", "No", "Boosted trees"],
        ["Cell kNN", "Expression neighbors + response edges", "Yes", "No", "No", "Exact weights", "Undefined for unseen drugs"],
        ["Drug kNN", "Descriptor neighbors + response edges", "No", "Yes", "No", "Exact weights", "Undefined for unseen cells"],
        ["Product kNN", "Cell x drug neighborhoods", "Yes", "Yes", "Yes", "Exact weights", "Sparse support"],
        ["Inductive MF", "Response matrix + feature projections", "Yes", "Yes", "Yes", "No", "ALS + ridge projection"],
        ["ITENS", "Three training-edge neighborhood experts", "Yes", "Yes", "Yes", "Exact signed edges", "ESS shrinkage"],
    ], columns=["Model", "Information used", "Cold cell", "Cold drug", "Cold both", "Exact decomposition", "Primary limitation"])
    write_table(table2, "table2_model_capabilities")

    selection = pd.read_csv(ROOT / "results" / "internal_seeds" / "selection_summary.csv")
    all_metrics = pd.read_csv(ROOT / "results" / "internal_seeds" / "all_metrics.csv")
    cold = all_metrics.loc[
        (all_metrics.metric == "spearman_drug_macro_mean")
        & all_metrics.regime.isin(["cold_cell", "cold_drug", "cold_both"])
    ]
    regime = cold.groupby(["model", "regime"], as_index=False).value.mean().pivot(index="model", columns="regime", values="value")
    table3 = selection.merge(regime, left_on="model", right_index=True, how="left")
    table3["Model"] = table3.model.map(MODEL_LABEL)
    table3 = table3[["Model", "cold_cell", "cold_drug", "cold_both", "mean_selection_score", "seed_sd", "n_seeds"]]
    table3.columns = ["Model", "Cold cell", "Cold drug", "Cold both", "Three-regime mean", "Seed SD", "Seeds"]
    table3 = table3.sort_values("Three-regime mean", ascending=False, na_position="last")
    write_table(table3, "table3_source_selection")

    gap = pd.read_csv(ROOT / "results" / "repair_historical_population" / "gaps.csv")
    gap["external_regime"] = gap["domain"].map({"overlap": "cross_study_overlap", "exact_pair": "cross_study_exact_pair"})
    table4 = gap[[
        "test_study", "external_regime", "n_drugs", "mean_internal_spearman",
        "mean_external_spearman", "gap_internal_minus_external", "ci_low", "ci_high", "supports_C1",
    ]].copy()
    table4.columns = ["Screen", "Matched domain", "Drugs", "Internal random", "External", "Gap", "95% CI low", "95% CI high", "C1 supported"]
    table4["Matched domain"] = table4["Matched domain"].map({"cross_study_overlap": "Shared entities", "cross_study_exact_pair": "Exact pairs"})
    write_table(table4, "table4_confirmatory_gap")

    holdout_summary = pd.read_csv(ROOT / "results" / "holdout_confirmation_seeded" / "seeded_holdout_summary.csv")
    full = holdout_summary.loc[
        holdout_summary.regime == "cross_study",
        ["test_study", "model", "n_eligible_drugs", "macro_spearman", "ci_low", "ci_high"],
    ].copy()
    total_drugs = {"GDSCv1": 290, "NCI60": 54896}
    full["coverage_fraction"] = full.apply(lambda row: row.n_eligible_drugs / total_drugs[row.test_study], axis=1)
    full["Model"] = full.model.map(MODEL_LABEL)
    table5 = full[["test_study", "Model", "n_eligible_drugs", "coverage_fraction", "macro_spearman", "ci_low", "ci_high"]]
    table5.columns = ["Screen", "Model", "Eligible drugs", "Coverage fraction", "Macro Spearman", "95% CI low", "95% CI high"]
    write_table(table5, "table5_full_domain_models")

    comparisons = pd.read_csv(ROOT / "results" / "holdout_confirmation_seeded" / "seeded_comparisons.csv")
    write_table(comparisons, "supp_table1_hypothesis_tests")
    reciprocal = pd.read_csv(ROOT / "results" / "reciprocal" / "reciprocal_long.csv")
    write_table(reciprocal, "supp_table2_reciprocal_transfer")
    return {"table1": table1, "table2": table2, "table3": table3, "table4": table4, "table5": table5}


def save_source(frame: pd.DataFrame, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FIGURES / f"{name}_source.csv", index=False)


def figure1_data_flow(table1: pd.DataFrame) -> None:
    use_style()
    frame = table1.copy()
    order = list(frame.Study)
    y = np.arange(len(order))[::-1]
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.48)
    panel_labels(axes, pad=17)
    ax1, ax2 = axes
    stages = ["Input fitted pairs", "Stable-ID pairs", "Feature-complete pairs"]
    stage_colors = ["#BDBDBD", PALETTE["cyan"], PALETTE["blue"]]
    for stage, color, marker in zip(stages, stage_colors, ["s", "o", "D"]):
        ax1.scatter(frame[stage], y, color=color, marker=marker, s=22, label=stage)
        for yy, start, end in zip(y, frame["Input fitted pairs"], frame[stage]):
            if stage == "Feature-complete pairs":
                ax1.plot([end, start], [yy, yy], color="#D0D0D0", linewidth=0.7, zorder=0)
    ax1.set_xscale("log")
    ax1.set_yticks(y, order)
    ax1.set_xlabel("Response pairs (log scale)")
    ax1.set_ylabel("Screen")
    ax1.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1,
        fontsize=5.6, frameon=True, borderpad=0.25, labelspacing=0.2,
    )
    ax1.grid(axis="x")
    ax1.grid(axis="y", visible=False)

    width = 0.36
    ax2.barh(y + width / 2, frame.Cells, height=width, color=PALETTE["blue"], label="Cells")
    ax2.barh(y - width / 2, frame.Drugs, height=width, color=PALETTE["orange"], label="Drugs")
    ax2.set_xscale("log")
    ax2.set_yticks(y, order)
    ax2.set_xlabel("Feature-complete entities (log scale)")
    ax2.set_ylabel("Screen")
    ax2.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
        fontsize=5.8, frameon=True, borderpad=0.25, columnspacing=0.6,
    )
    ax2.grid(axis="x")
    ax2.grid(axis="y", visible=False)
    source = frame.copy()
    save_source(source, "figure1_data_flow")
    save(fig, str(FIGURES / "figure1_data_flow"), formats=("pdf", "svg", "png"))


def figure2_internal_selection() -> None:
    use_style()
    per_seed = pd.read_csv(ROOT / "results" / "internal_seeds" / "selection_per_seed.csv")
    all_metrics = pd.read_csv(ROOT / "results" / "internal_seeds" / "all_metrics.csv")
    eligible = per_seed.loc[per_seed.valid_regimes == 3].copy()
    plot_label = {**MODEL_LABEL, "product_knn": "Prod. kNN"}
    eligible["label"] = eligible.model.map(plot_label)
    labels = [plot_label[m] for m in ["inductive_mf", "itens_k16", "lightgbm", "product_knn", "histgb", "ridge_bilinear"]]
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.48)
    panel_labels(axes)
    ax1, ax2 = axes
    positions = np.arange(len(labels))
    for index, label in enumerate(labels):
        values = eligible.loc[eligible.label == label, "selection_score"].to_numpy()
        color_key = "Product kNN" if label == "Prod. kNN" else label
        ax1.scatter(np.full(len(values), index), values, color=COLORS[color_key], s=14, alpha=0.8)
        ax1.plot([index - 0.22, index + 0.22], [values.mean(), values.mean()], color="black", linewidth=1.2)
    ax1.set_xticks(positions, labels, rotation=55, ha="right", fontsize=6)
    ax1.set_ylabel("Mean macro Spearman across cold regimes")
    ax1.set_xlabel("Model")
    ax1.grid(axis="y")
    ax1.grid(axis="x", visible=False)

    cold = all_metrics.loc[
        (all_metrics.metric == "spearman_drug_macro_mean")
        & all_metrics.regime.isin(["cold_cell", "cold_drug", "cold_both"])
    ]
    matrix = cold.groupby(["model", "regime"]).value.mean().unstack()
    model_keys = ["additive_matched", "ridge_bilinear", "histgb", "lightgbm", "inductive_mf", "cell_knn", "drug_knn", "product_knn", "itens_k16"]
    regime_keys = ["cold_cell", "cold_drug", "cold_both"]
    values = matrix.reindex(model_keys)[regime_keys].to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("#D9D9D9")
    image = ax2.imshow(masked, vmin=0, vmax=0.5, cmap=cmap, aspect="auto")
    ax2.set_xticks(range(3), ["Cold cell", "Cold drug", "Cold both"], rotation=25, ha="right")
    ax2.set_yticks(range(len(model_keys)), [MODEL_LABEL[key] for key in model_keys])
    ax2.set_xlabel("Evaluation regime")
    ax2.set_ylabel("Model")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = "NA" if not np.isfinite(values[i, j]) else f"{values[i, j]:.2f}"
            color = "black" if not np.isfinite(values[i, j]) or values[i, j] < 0.32 else "white"
            ax2.text(j, i, text, ha="center", va="center", fontsize=6, color=color)
    fig.colorbar(image, ax=ax2, label="Macro Spearman", fraction=0.05, pad=0.03)
    save_source(eligible, "figure2_internal_selection")
    save_source(cold, "figure2_cold_regimes")
    save(fig, str(FIGURES / "figure2_internal_selection"), formats=("pdf", "svg", "png"))


def figure3_directional_transfer() -> None:
    use_style()
    reciprocal = pd.read_csv(ROOT / "results" / "reciprocal" / "reciprocal_long.csv")
    studies = ["CTRPv2", "PRISM", "GDSCv2", "gCSI"]
    regimes = [("cross_study", "Full target domain"), ("cross_study_exact_pair", "Exact shared pairs")]
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.48)
    panel_labels(axes)
    for ax, (regime, panel_name) in zip(axes, regimes):
        subset = reciprocal.loc[(reciprocal.model == "inductive_mf") & (reciprocal.regime == regime)]
        matrix = subset.pivot(index="train_study", columns="test_study", values="value").reindex(index=studies, columns=studies)
        values = matrix.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(values)
        cmap = mpl.colormaps["RdBu_r"].copy()
        cmap.set_bad("#F2F2F2")
        image = ax.imshow(masked, vmin=-0.1, vmax=0.45, cmap=cmap)
        ax.set_xticks(range(4), studies, rotation=35, ha="right")
        ax.set_yticks(range(4), studies)
        ax.set_xlabel("Target screen")
        ax.set_ylabel("Source screen")
        for i in range(4):
            for j in range(4):
                if np.isfinite(values[i, j]):
                    color = "white" if values[i, j] < 0.02 or values[i, j] > 0.34 else "black"
                    ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)
                elif i == j:
                    ax.text(j, i, "--", ha="center", va="center", fontsize=7, color="#777777")
        fig.colorbar(image, ax=ax, label="Macro per-drug Spearman", fraction=0.05, pad=0.03)
    save_source(reciprocal.loc[reciprocal.model == "inductive_mf"], "figure3_directional_transfer")
    save(fig, str(FIGURES / "figure3_directional_transfer"), formats=("pdf", "svg", "png"))


def figure4_confirmatory() -> None:
    use_style()
    gap = pd.read_csv(ROOT / "results" / "repair_historical_population" / "gaps.csv")
    gap["external_regime"] = gap["domain"].map({"overlap": "cross_study_overlap", "exact_pair": "cross_study_exact_pair"})
    summary = pd.read_csv(ROOT / "results" / "holdout_confirmation_seeded" / "seeded_holdout_summary.csv")
    gap["label"] = gap.test_study + " " + gap.external_regime.map({
        "cross_study_overlap": "shared entities", "cross_study_exact_pair": "exact pairs",
    })
    gap = gap.sort_values(["test_study", "external_regime"], ascending=[True, False]).reset_index(drop=True)
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.48)
    panel_labels(axes)
    ax1, ax2 = axes
    y = np.arange(len(gap))[::-1]
    colors = [PALETTE["blue"] if study == "GDSCv1" else PALETTE["orange"] for study in gap.test_study]
    for position, color, row in zip(y, colors, gap.itertuples(index=False)):
        ax1.errorbar(
            row.gap_internal_minus_external, position,
            xerr=[[row.gap_internal_minus_external - row.ci_low], [row.ci_high - row.gap_internal_minus_external]],
            fmt="o", color=color, ecolor=color, markersize=4, markeredgecolor="black",
            markeredgewidth=0.3, elinewidth=1.2, capsize=2,
        )
    ax1.axvline(0.10, color="#777777", linestyle="--", linewidth=0.9)
    ax1.set_yticks(y, gap.label)
    ax1.set_xlabel("Internal minus external macro Spearman")
    ax1.set_ylabel("Matched confirmation domain")
    ax1.grid(axis="x")
    ax1.grid(axis="y", visible=False)

    models = ["ridge_bilinear", "lightgbm", "itens_k16", "selected_inductive_mf"]
    subset = summary.loc[
        (summary.regime == "cross_study_overlap") & summary.model.isin(models)
    ].copy()
    subset["label"] = subset.model.map(MODEL_LABEL)
    subset.loc[subset.model == "selected_inductive_mf", "label"] = "MF (expanded)"
    positions = np.arange(len(models))[::-1]
    offsets = {"GDSCv1": 0.10, "NCI60": -0.10}
    for study, color, marker in (("GDSCv1", PALETTE["blue"], "o"), ("NCI60", PALETTE["orange"], "s")):
        study_rows = subset.loc[subset.test_study == study].set_index("model").loc[models]
        yy = positions + offsets[study]
        ax2.errorbar(
            study_rows.macro_spearman, yy,
            xerr=np.vstack([study_rows.macro_spearman - study_rows.ci_low, study_rows.ci_high - study_rows.macro_spearman]),
            fmt=marker, color=color, ecolor=color, markersize=4, elinewidth=1.0, capsize=2, label=study,
        )
    ax2.axvline(0, color="#777777", linewidth=0.8)
    model_labels = subset.drop_duplicates("model").set_index("model").loc[models, "label"].tolist()
    ax2.set_yticks(positions, model_labels)
    ax2.set_xlabel("Post-lock expanded-source macro Spearman")
    ax2.set_ylabel("Model")
    ax2.legend(loc="upper right", fontsize=6)
    ax2.grid(axis="x")
    ax2.grid(axis="y", visible=False)
    save_source(gap, "figure4_confirmatory_gap")
    save_source(subset, "figure4_confirmatory_models")
    save(fig, str(FIGURES / "figure4_confirmatory"), formats=("pdf", "svg", "png"))


def figure5_coverage_failure() -> None:
    use_style()
    summary = pd.read_csv(ROOT / "results" / "holdout_confirmation_seeded" / "seeded_holdout_summary.csv")
    metrics = pd.read_csv(ROOT / "results" / "holdout_confirmation" / "metrics.csv")
    seeded_metrics = pd.read_csv(ROOT / "results" / "holdout_mf_seeds" / "seed_variability.csv")
    seeded_metrics = seeded_metrics.rename(columns={"mean": "value"})
    seeded_metrics["model"] = "selected_inductive_mf"
    metrics = pd.concat([
        metrics.loc[metrics.model != "selected_inductive_mf"],
        seeded_metrics[["test_study", "regime", "model", "metric", "value"]],
    ], ignore_index=True)
    full = summary.loc[summary.regime == "cross_study"].copy()
    totals = {"GDSCv1": 290, "NCI60": 54896}
    full["coverage"] = full.apply(lambda row: row.n_eligible_drugs / totals[row.test_study], axis=1)
    full["label"] = full.model.map(MODEL_LABEL)
    selected = ["ridge_bilinear", "histgb", "lightgbm", "selected_inductive_mf", "product_knn", "itens_k16", "cell_knn"]
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.58)
    panel_labels(axes)
    ax1, ax2 = axes
    marker = {"GDSCv1": "o", "NCI60": "s"}
    study_color = {"GDSCv1": PALETTE["blue"], "NCI60": PALETTE["orange"]}
    selected_rows = full.loc[full.model.isin(selected)]
    # Connect paired estimates for the same model; this makes the screen shift
    # visible without assigning a different, unexplained color to every model.
    for model in selected:
        pair = selected_rows.loc[selected_rows.model == model].set_index("test_study")
        if {"GDSCv1", "NCI60"}.issubset(pair.index):
            ax1.plot(
                [pair.loc["GDSCv1", "coverage"], pair.loc["NCI60", "coverage"]],
                [pair.loc["GDSCv1", "macro_spearman"], pair.loc["NCI60", "macro_spearman"]],
                color="#C9C9C9", linewidth=0.65, zorder=0,
            )
    for study in ("GDSCv1", "NCI60"):
        subset = selected_rows.loc[selected_rows.test_study == study]
        ax1.scatter(
            subset.coverage, subset.macro_spearman, marker=marker[study],
            color=study_color[study], s=28, edgecolor="black", linewidth=0.35,
            zorder=3, label=study,
        )
    # Label the low-coverage Cell kNN and high-RMSE Ridge points without
    # crowding the cluster of other models at full coverage.
    annotation_xy = {
        "ridge_bilinear": (0.66, 0.19),
        "cell_knn": (0.009, 0.225),
    }
    for row in selected_rows.loc[
        (selected_rows.test_study == "NCI60")
        & selected_rows.model.isin(annotation_xy)
    ].itertuples(index=False):
        ax1.annotate(
            row.label, (row.coverage, row.macro_spearman),
            xytext=annotation_xy[row.model], textcoords="data", xycoords="data",
            ha="center", va="center", fontsize=5.8, color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.45},
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.45, "shrinkA": 1, "shrinkB": 2},
            zorder=4,
        )
    ax1.set_xscale("log")
    ax1.set_xlabel("Eligible-drug coverage fraction (log scale)")
    ax1.set_ylabel("Full-domain macro Spearman")
    ax1.grid(True)
    ax1.set_ylim(-0.05, 0.25)
    ax1.legend(loc="lower left", fontsize=5.8, title="Screen", title_fontsize=5.8)

    rmse = metrics.loc[
        (metrics.regime == "cross_study") & (metrics.metric == "rmse")
        & metrics.model.isin(["ridge_bilinear", "histgb", "lightgbm", "selected_inductive_mf", "product_knn", "itens_k16"])
    ].copy()
    rmse["label"] = rmse.model.map(MODEL_LABEL)
    rmse_model_keys = ["ridge_bilinear", "histgb", "lightgbm", "selected_inductive_mf", "product_knn", "itens_k16"]
    model_labels = ["Ridge", "HistGB", "LightGBM", "Locked MF", "Prod. kNN", "ITENS"]
    x = np.arange(len(model_labels))
    width = 0.34
    for offset, study, color in [(-width / 2, "GDSCv1", PALETTE["blue"]), (width / 2, "NCI60", PALETTE["orange"])]:
        values = [rmse.loc[(rmse.test_study == study) & (rmse.model == key), "value"].iloc[0] for key in rmse_model_keys]
        ax2.bar(x + offset, values, width=width, color=color, label=study)
    ax2.set_yscale("log")
    ax2.set_xticks(x, model_labels, rotation=58, ha="right", fontsize=6)
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Full-domain RMSE (log scale)")
    ax2.legend(loc="upper right", fontsize=6)
    ax2.grid(axis="y")
    ax2.grid(axis="x", visible=False)
    save_source(full, "figure5_coverage")
    save_source(rmse, "figure5_rmse")
    save(fig, str(FIGURES / "figure5_coverage_failure"), formats=("pdf", "svg", "png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.root.resolve() != ROOT:
        raise RuntimeError(f"Script is bound to project root {ROOT}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    tables = build_tables()
    figure1_data_flow(tables["table1"])
    figure2_internal_selection()
    figure3_directional_transfer()
    figure4_confirmatory()
    figure5_coverage_failure()
    print(f"WROTE {len(list(TABLES.glob('*.csv')))} CSV tables")
    print(f"WROTE {len(list(FIGURES.glob('*.pdf')))} PDF figures")


if __name__ == "__main__":
    main()

