"""
compare_mice_results.py

Collect per-mouse Stage 5 outputs from the HCR-Tasic matching pipeline and
produce cross-mouse comparison figures in a single output directory.

Expected input layout (produced by run_hcr_tasic_matching.py):

    {results_dir}/stage5/
        hcr_assignments_leiden_named.csv
        clustering_quality_summary.csv
        mouse_{id}/
            cluster_correlations.csv
            clustering_r2_per_gene.csv
            clustering_r2_per_cluster.csv
            clustering_silhouette.csv
            residual_gene_instability.csv
            residual_cell_norms.csv
            mean_expression_centroids.csv

Output:
    {results_dir}/stage5/cross_mouse/
        01_assignment_distribution.png
        02_cluster_representation.png
        03_global_quality_metrics.png
        04_per_gene_r2_comparison.png
        05_cluster_correlation_comparison.png
        06_gene_instability_comparison.png
        07_residual_norm_distributions.png
        08_silhouette_comparison.png
        cross_mouse_summary.csv

Usage:
    python compare_mice_results.py
    python compare_mice_results.py --results-dir /root/capsule/results/hcr_tasic_matching
    python compare_mice_results.py --results-dir /root/capsule/results/hcr_tasic_matching --out-dir /root/capsule/results/cross_mouse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_RESULTS_DIR = Path("/root/capsule/results/hcr_tasic_matching")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_per_mouse(stage5_dir: Path, filename: str) -> dict[str, pd.DataFrame]:
    """Load one CSV file from every mouse_{id} subdirectory that has it."""
    data = {}
    for mouse_dir in sorted(stage5_dir.glob("mouse_*")):
        path = mouse_dir / filename
        if path.exists():
            data[mouse_dir.name.removeprefix("mouse_")] = pd.read_csv(path)
        else:
            print(f"  [skip] {path} not found")
    return data


def _mouse_colors(mice: list[str]) -> dict[str, tuple]:
    cmap = plt.cm.get_cmap("tab10", len(mice))
    return {m: cmap(i) for i, m in enumerate(mice)}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Assignment distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_assignment_distribution(
    assignments: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Stacked bar: how many cells each mouse assigns to each cluster, as
    a fraction of that mouse's total.
    """
    mice = sorted(assignments["mouse_id"].unique())
    all_clusters = sorted(assignments["assignment"].unique())
    colors = _mouse_colors(mice)

    # Fraction table: rows=mice, cols=clusters
    frac_table = pd.DataFrame(index=mice, columns=all_clusters, dtype=float).fillna(0.0)
    count_table = pd.DataFrame(index=mice, columns=all_clusters, dtype=int).fillna(0)
    for mouse in mice:
        sub = assignments[assignments["mouse_id"] == mouse]
        counts = sub["assignment"].value_counts()
        for cl in all_clusters:
            n = counts.get(cl, 0)
            count_table.loc[mouse, cl] = n
            frac_table.loc[mouse, cl] = n / len(sub) if len(sub) > 0 else 0.0

    # Sort clusters by mean fraction descending
    cluster_order = frac_table.mean(axis=0).sort_values(ascending=False).index.tolist()
    frac_table = frac_table[cluster_order]
    count_table = count_table[cluster_order]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(cluster_order) * 0.6), 6))

    # Left: stacked bar by cluster, grouped by mouse
    ax = axes[0]
    x = np.arange(len(cluster_order))
    width = 0.8 / len(mice)
    for i, mouse in enumerate(mice):
        ax.bar(
            x + i * width - 0.4 + width / 2,
            frac_table.loc[mouse].values,
            width=width,
            label=f"Mouse {mouse}",
            color=colors[mouse],
            edgecolor="black",
            linewidth=0.3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of mouse cells", fontsize=9)
    ax.set_title("Assignment distribution per mouse\n(fraction)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Right: heatmap of fractions (mice × clusters)
    ax = axes[1]
    sns.heatmap(
        frac_table.astype(float),
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"label": "fraction", "shrink": 0.7},
        linewidths=0.4,
    )
    ax.set_title("Assignment fraction heatmap\n(mice × clusters)", fontsize=10, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.suptitle("Cross-mouse: Cell assignment distributions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "01_assignment_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 01_assignment_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cluster representation (count absolute)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_representation(
    assignments: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Absolute cell counts per (mouse, cluster) as a grouped heatmap."""
    mice = sorted(assignments["mouse_id"].unique())
    all_clusters = sorted(assignments["assignment"].unique())

    count_table = pd.DataFrame(index=mice, columns=all_clusters, dtype=int).fillna(0)
    for mouse in mice:
        sub = assignments[assignments["mouse_id"] == mouse]
        counts = sub["assignment"].value_counts()
        for cl in all_clusters:
            count_table.loc[mouse, cl] = counts.get(cl, 0)

    cluster_order = count_table.sum(axis=0).sort_values(ascending=False).index.tolist()
    count_table = count_table[cluster_order]

    fig, ax = plt.subplots(figsize=(max(10, len(cluster_order) * 0.55), len(mice) * 0.7 + 1.5))
    sns.heatmap(
        count_table.astype(int),
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt="d",
        annot_kws={"size": 8},
        cbar_kws={"label": "cell count", "shrink": 0.6},
        linewidths=0.4,
    )
    ax.set_title("Cross-mouse: Absolute cell counts per cluster", fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"Mouse {m}" for m in mice], rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "02_cluster_representation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 02_cluster_representation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Global quality metrics summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_global_quality_metrics(
    quality_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Side-by-side bar charts: global R² and mean silhouette per mouse.
    """
    mice = quality_summary["mouse_id"].astype(str).tolist()
    colors = list(_mouse_colors(mice).values())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # R²
    ax = axes[0]
    r2_vals = pd.to_numeric(quality_summary["global_r2"], errors="coerce")
    bars = ax.bar(mice, r2_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylabel("Global R²", fontsize=9)
    ax.set_title("Variance explained\nby cluster labels", fontsize=10, fontweight="bold")
    ax.set_xticklabels([f"Mouse {m}" for m in mice], rotation=15, ha="right", fontsize=8)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Mean silhouette
    ax = axes[1]
    sil_vals = pd.to_numeric(quality_summary["mean_silhouette"], errors="coerce")
    bars = ax.bar(mice, sil_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.5, color="green", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylabel("Mean silhouette score", fontsize=9)
    ax.set_title("Mean cluster separation\n(silhouette)", fontsize=10, fontweight="bold")
    ax.set_xticklabels([f"Mouse {m}" for m in mice], rotation=15, ha="right", fontsize=8)
    for bar, val in zip(bars, sil_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # n_cells
    ax = axes[2]
    n_cells = quality_summary["n_cells"].astype(int)
    bars = ax.bar(mice, n_cells, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Assigned cells", fontsize=9)
    ax.set_title("Cells assigned\n(Approach C)", fontsize=10, fontweight="bold")
    ax.set_xticklabels([f"Mouse {m}" for m in mice], rotation=15, ha="right", fontsize=8)
    for bar, val in zip(bars, n_cells):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, str(val),
                ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Cross-mouse: Global clustering quality", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "03_global_quality_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 03_global_quality_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-gene R² comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_gene_r2_comparison(
    r2_per_gene_by_mouse: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Heatmap (mice × genes) of per-gene R², plus grouped barplot.
    """
    if not r2_per_gene_by_mouse:
        return

    mice = sorted(r2_per_gene_by_mouse)
    # Union of all genes, ordered by mean R² descending
    all_genes = sorted(
        {g for df in r2_per_gene_by_mouse.values() for g in df["gene"]}
    )

    r2_table = pd.DataFrame(index=mice, columns=all_genes, dtype=float).fillna(np.nan)
    for mouse, df in r2_per_gene_by_mouse.items():
        for _, row in df.iterrows():
            r2_table.loc[mouse, row["gene"]] = row["r2"]

    gene_order = r2_table.mean(axis=0).sort_values(ascending=False).index.tolist()
    r2_table = r2_table[gene_order]

    colors = _mouse_colors(mice)

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(gene_order) * 0.55), 9),
                             gridspec_kw={"height_ratios": [1, 1.4]})

    # Top: grouped barplot
    ax = axes[0]
    x = np.arange(len(gene_order))
    width = 0.8 / len(mice)
    for i, mouse in enumerate(mice):
        vals = r2_table.loc[mouse, gene_order].values.astype(float)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width,
               label=f"Mouse {mouse}", color=colors[mouse],
               edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(gene_order, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylabel("R² (variance explained)", fontsize=9)
    ax.set_title("Per-gene R²: grouped by mouse", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Bottom: heatmap
    ax = axes[1]
    sns.heatmap(
        r2_table.astype(float),
        ax=ax,
        cmap="RdYlGn",
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        annot_kws={"size": 8},
        cbar_kws={"label": "R²", "shrink": 0.6},
        linewidths=0.4,
    )
    ax.set_title("Per-gene R² heatmap (mice × genes)", fontsize=10, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([f"Mouse {m}" for m in mice], rotation=0, fontsize=9)

    plt.suptitle("Cross-mouse: Per-gene variance explained by cluster labels",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "04_per_gene_r2_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 04_per_gene_r2_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cluster correlation vs Tasic reference
# ─────────────────────────────────────────────────────────────────────────────

def plot_cluster_correlation_comparison(
    corr_by_mouse: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    For each cluster present in any mouse, show Pearson r per mouse as a
    grouped barplot + heatmap.
    """
    if not corr_by_mouse:
        return

    mice = sorted(corr_by_mouse)
    all_clusters = sorted(
        {cl for df in corr_by_mouse.values() for cl in df["cluster"]}
    )

    corr_table = pd.DataFrame(index=mice, columns=all_clusters, dtype=float).fillna(np.nan)
    for mouse, df in corr_by_mouse.items():
        for _, row in df.iterrows():
            corr_table.loc[mouse, row["cluster"]] = row["pearson_r"]

    cluster_order = corr_table.mean(axis=0).sort_values(ascending=False).index.tolist()
    corr_table = corr_table[cluster_order]

    colors = _mouse_colors(mice)
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(cluster_order) * 0.5), 9),
                             gridspec_kw={"height_ratios": [1, 1.4]})

    # Top: grouped barplot
    ax = axes[0]
    x = np.arange(len(cluster_order))
    width = 0.8 / len(mice)
    for i, mouse in enumerate(mice):
        vals = corr_table.loc[mouse, cluster_order].values.astype(float)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width,
               label=f"Mouse {mouse}", color=colors[mouse],
               edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(-1, 1)
    ax.axhline(0.8, color="green", linestyle="--", linewidth=1, alpha=0.6, label="r=0.8")
    ax.axhline(0.6, color="orange", linestyle="--", linewidth=1, alpha=0.6, label="r=0.6")
    ax.set_ylabel("Pearson r (vs Tasic centroid)", fontsize=9)
    ax.set_title("Per-cluster correlation with Tasic reference", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # Bottom: heatmap
    ax = axes[1]
    sns.heatmap(
        corr_table.astype(float),
        ax=ax,
        cmap="RdYlGn",
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"label": "Pearson r", "shrink": 0.6},
        linewidths=0.4,
    )
    ax.set_title("Pearson r heatmap (mice × clusters)", fontsize=10, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"Mouse {m}" for m in mice], rotation=0, fontsize=9)

    plt.suptitle("Cross-mouse: Cluster correlation with Tasic reference",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "05_cluster_correlation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 05_cluster_correlation_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Gene instability comparison (residual std)
# ─────────────────────────────────────────────────────────────────────────────

def plot_gene_instability_comparison(
    instability_by_mouse: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Grouped barplot + heatmap of mean residual std per gene across mice.
    """
    if not instability_by_mouse:
        return

    mice = sorted(instability_by_mouse)
    all_genes = sorted(
        {g for df in instability_by_mouse.values() for g in df["gene"]}
    )

    instab_table = pd.DataFrame(index=mice, columns=all_genes, dtype=float).fillna(np.nan)
    for mouse, df in instability_by_mouse.items():
        for _, row in df.iterrows():
            instab_table.loc[mouse, row["gene"]] = row["mean_residual_std"]

    # Order by mean instability descending (most problematic gene first)
    gene_order = instab_table.mean(axis=0).sort_values(ascending=False).index.tolist()
    instab_table = instab_table[gene_order]
    colors = _mouse_colors(mice)

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(gene_order) * 0.55), 9),
                             gridspec_kw={"height_ratios": [1, 1.2]})

    # Top: grouped barplot
    ax = axes[0]
    x = np.arange(len(gene_order))
    width = 0.8 / len(mice)
    for i, mouse in enumerate(mice):
        vals = instab_table.loc[mouse, gene_order].values.astype(float)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width,
               label=f"Mouse {mouse}", color=colors[mouse],
               edgecolor="black", linewidth=0.3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(gene_order, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean residual std (z-score units)", fontsize=9)
    ax.set_title("Per-gene instability: spread of cells around centroid",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Bottom: heatmap
    ax = axes[1]
    sns.heatmap(
        instab_table.astype(float),
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        annot=True, fmt=".2f",
        annot_kws={"size": 8},
        cbar_kws={"label": "Mean residual std", "shrink": 0.6},
        linewidths=0.4,
    )
    ax.set_title("Gene instability heatmap (mice × genes)", fontsize=10, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([f"Mouse {m}" for m in mice], rotation=0, fontsize=9)

    plt.suptitle("Cross-mouse: Gene instability (residual spread around assigned centroid)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "06_gene_instability_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 06_gene_instability_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Per-cell residual norm distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_residual_norm_distributions(
    norms_by_mouse: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Overlapping KDE + boxplot of ||cell - centroid||₂ distributions per mouse.
    """
    if not norms_by_mouse:
        return

    mice = sorted(norms_by_mouse)
    colors = _mouse_colors(mice)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: KDE overlay
    ax = axes[0]
    for mouse in mice:
        vals = norms_by_mouse[mouse]["residual_norm"].dropna().values
        ax.hist(vals, bins=40, alpha=0.45, color=colors[mouse],
                density=True, label=f"Mouse {mouse}", edgecolor="none")
        # KDE line
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals)
        xg = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(xg, kde(xg), color=colors[mouse], linewidth=2)
    ax.set_xlabel("||cell − centroid||₂", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Residual norm distributions per mouse", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: box + strip plot
    ax = axes[1]
    all_data = []
    for mouse in mice:
        df = norms_by_mouse[mouse][["residual_norm", "assignment"]].copy()
        df["mouse_id"] = mouse
        all_data.append(df)
    combined = pd.concat(all_data, ignore_index=True)

    bp = ax.boxplot(
        [norms_by_mouse[m]["residual_norm"].dropna().values for m in mice],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        showfliers=False,
        vert=True,
    )
    for patch, mouse in zip(bp["boxes"], mice):
        patch.set_facecolor(colors[mouse])
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(mice) + 1))
    ax.set_xticklabels([f"Mouse {m}" for m in mice], fontsize=9)
    ax.set_ylabel("||cell − centroid||₂", fontsize=9)
    ax.set_title("Residual norm boxplot per mouse", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Annotate with median and 90th pct
    for i, mouse in enumerate(mice):
        vals = norms_by_mouse[mouse]["residual_norm"].dropna().values
        med = np.median(vals)
        p90 = np.percentile(vals, 90)
        ax.text(i + 1, p90 * 1.02, f"p90={p90:.2f}", ha="center", fontsize=7, color="dimgrey")

    plt.suptitle("Cross-mouse: Per-cell residual norm distributions",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "07_residual_norm_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 07_residual_norm_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Silhouette score comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_silhouette_comparison(
    sil_by_mouse: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Per-cluster silhouette medians across mice, as a heatmap and strip/box plot.
    """
    if not sil_by_mouse:
        return

    mice = sorted(sil_by_mouse)
    colors = _mouse_colors(mice)
    all_clusters = sorted(
        {cl for df in sil_by_mouse.values() for cl in df["assignment"]}
    )

    # Median silhouette per (mouse, cluster)
    med_table = pd.DataFrame(index=mice, columns=all_clusters, dtype=float).fillna(np.nan)
    for mouse, df in sil_by_mouse.items():
        for cl in all_clusters:
            sub = df.loc[df["assignment"] == cl, "silhouette"]
            if len(sub) > 0:
                med_table.loc[mouse, cl] = sub.median()

    cluster_order = med_table.mean(axis=0).sort_values(ascending=False).index.tolist()
    med_table = med_table[cluster_order]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(cluster_order) * 0.55), 5))

    # Left: heatmap
    ax = axes[0]
    sns.heatmap(
        med_table.astype(float),
        ax=ax,
        cmap="RdYlGn",
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"label": "Median silhouette", "shrink": 0.6},
        linewidths=0.4,
    )
    ax.set_title("Median silhouette per (mouse, cluster)", fontsize=10, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"Mouse {m}" for m in mice], rotation=0, fontsize=9)

    # Right: overall violin per mouse
    ax = axes[1]
    all_sil_data = [sil_by_mouse[m]["silhouette"].dropna().values for m in mice]
    parts = ax.violinplot(all_sil_data, vert=True, showmedians=True, showextrema=True)
    for i, (pc, mouse) in enumerate(zip(parts["bodies"], mice)):
        pc.set_facecolor(colors[mouse])
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(mice) + 1))
    ax.set_xticklabels([f"Mouse {m}" for m in mice], fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.5, color="green", linewidth=1, linestyle=":", alpha=0.7)
    ax.set_ylabel("Silhouette score", fontsize=9)
    ax.set_ylim(-1, 1)
    ax.set_title("Silhouette distribution per mouse", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, (mouse, vals) in enumerate(zip(mice, all_sil_data)):
        med = np.median(vals)
        ax.text(i + 1, med + 0.05, f"{med:.2f}", ha="center", fontsize=8)

    plt.suptitle("Cross-mouse: Silhouette score comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "08_silhouette_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 08_silhouette_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_cross_mouse_summary(
    assignments: pd.DataFrame,
    quality_summary: pd.DataFrame | None,
    corr_by_mouse: dict[str, pd.DataFrame],
    instability_by_mouse: dict[str, pd.DataFrame],
    norms_by_mouse: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    mice = sorted(assignments["mouse_id"].astype(str).unique())
    rows = []
    for mouse in mice:
        sub = assignments[assignments["mouse_id"].astype(str) == mouse]
        row = {
            "mouse_id": mouse,
            "n_assigned_cells": len(sub),
            "n_clusters": sub["assignment"].nunique(),
            "mean_confidence": sub["confidence"].mean() if "confidence" in sub.columns else np.nan,
        }
        if quality_summary is not None:
            q = quality_summary[quality_summary["mouse_id"].astype(str) == mouse]
            if len(q) > 0:
                row["global_r2"] = q.iloc[0]["global_r2"]
                row["mean_silhouette"] = q.iloc[0]["mean_silhouette"]
        if mouse in corr_by_mouse:
            row["mean_cluster_pearson_r"] = corr_by_mouse[mouse]["pearson_r"].mean()
            row["min_cluster_pearson_r"] = corr_by_mouse[mouse]["pearson_r"].min()
        if mouse in instability_by_mouse:
            row["mean_gene_instability"] = instability_by_mouse[mouse]["mean_residual_std"].mean()
            row["max_gene_instability"] = instability_by_mouse[mouse]["mean_residual_std"].max()
        if mouse in norms_by_mouse:
            norms = norms_by_mouse[mouse]["residual_norm"]
            row["median_cell_norm"] = norms.median()
            row["p90_cell_norm"] = norms.quantile(0.9)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = out_dir / "cross_mouse_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n  Cross-mouse summary ({len(rows)} mice) saved to {out_path}")
    print(summary_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(results_dir: Path, out_dir: Path) -> None:
    stage5_dir = results_dir / "stage5"
    if not stage5_dir.exists():
        raise FileNotFoundError(f"Stage 5 output directory not found: {stage5_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCollecting outputs from: {stage5_dir}")
    print(f"Writing cross-mouse figures to: {out_dir}\n")

    # ── Load shared data ────────────────────────────────────────────────────
    assignments_path = stage5_dir / "hcr_assignments_leiden_named.csv"
    if not assignments_path.exists():
        raise FileNotFoundError(f"Assignments file not found: {assignments_path}")
    assignments = pd.read_csv(assignments_path)
    assignments["mouse_id"] = assignments["mouse_id"].astype(str)

    quality_path = stage5_dir / "clustering_quality_summary.csv"
    quality_summary = pd.read_csv(quality_path) if quality_path.exists() else None
    if quality_summary is not None:
        quality_summary["mouse_id"] = quality_summary["mouse_id"].astype(str)

    # ── Load per-mouse CSVs ─────────────────────────────────────────────────
    corr_by_mouse = _load_per_mouse(stage5_dir, "cluster_correlations.csv")
    r2_by_mouse = _load_per_mouse(stage5_dir, "clustering_r2_per_gene.csv")
    instability_by_mouse = _load_per_mouse(stage5_dir, "residual_gene_instability.csv")
    norms_by_mouse = _load_per_mouse(stage5_dir, "residual_cell_norms.csv")
    sil_by_mouse = _load_per_mouse(stage5_dir, "clustering_silhouette.csv")

    mice = sorted(assignments["mouse_id"].unique())
    print(f"Mice found: {mice}\n")

    # ── Plots ───────────────────────────────────────────────────────────────
    print("[1/8] Assignment distributions...")
    plot_assignment_distribution(assignments, out_dir)

    print("[2/8] Cluster representation...")
    plot_cluster_representation(assignments, out_dir)

    print("[3/8] Global quality metrics...")
    if quality_summary is not None:
        plot_global_quality_metrics(quality_summary, out_dir)
    else:
        print("  Skipped (clustering_quality_summary.csv not found)")

    print("[4/8] Per-gene R² comparison...")
    plot_per_gene_r2_comparison(r2_by_mouse, out_dir)

    print("[5/8] Cluster correlation comparison...")
    plot_cluster_correlation_comparison(corr_by_mouse, out_dir)

    print("[6/8] Gene instability comparison...")
    plot_gene_instability_comparison(instability_by_mouse, out_dir)

    print("[7/8] Residual norm distributions...")
    plot_residual_norm_distributions(norms_by_mouse, out_dir)

    print("[8/8] Silhouette comparison...")
    plot_silhouette_comparison(sil_by_mouse, out_dir)

    # ── Summary CSV ─────────────────────────────────────────────────────────
    save_cross_mouse_summary(
        assignments, quality_summary, corr_by_mouse,
        instability_by_mouse, norms_by_mouse, out_dir,
    )

    print(f"\nDone. All outputs in {out_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect per-mouse HCR-Tasic pipeline outputs and plot cross-mouse comparisons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root results directory containing the stage5/ subfolder.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for cross-mouse figures. "
             "Defaults to {results_dir}/stage5/cross_mouse/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out_dir = args.out_dir or (args.results_dir / "stage5" / "cross_mouse")
    run(args.results_dir, out_dir)
