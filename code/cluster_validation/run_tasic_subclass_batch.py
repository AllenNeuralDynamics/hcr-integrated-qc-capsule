from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from sklearn.metrics import adjusted_rand_score

from aind_hcr_data_loader import get_hcr_dataset_pairwise
import aind_hcr_qc.viz as viz

import cluster_validation_utils


def _safe_close(fig):
    if fig is not None:
        plt.close(fig)


def _save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def _cluster_label_table_from_markers(
    high_conf_markers: pd.DataFrame,
    cluster_col: str,
    n_label_genes: int = 3,
    label_sep: str = " | ",
) -> pd.DataFrame:
    enriched_for_labels = (
        high_conf_markers[
            (high_conf_markers["direction"] == "enriched")
            & (high_conf_markers["rank"] <= n_label_genes)
        ]
        .sort_values(["cluster", "rank"])
        .copy()
    )

    if enriched_for_labels.empty:
        return pd.DataFrame(
            columns=[
                "cluster",
                "cluster_label_string",
                "marker_count",
                "mean_label_stability_pct",
                "min_label_stability_pct",
            ]
        )

    cluster_label_table = (
        enriched_for_labels.groupby("cluster", as_index=False)
        .agg(
            cluster_label_string=("gene", lambda s: label_sep.join(s.astype(str).tolist())),
            marker_count=("gene", "count"),
            mean_label_stability_pct=("stability_pct", "mean"),
            min_label_stability_pct=("stability_pct", "min"),
        )
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    label_detail_wide = (
        enriched_for_labels.pivot(index="cluster", columns="rank", values="gene")
        .rename(columns=lambda c: f"label_gene_{c}")
    )
    stability_detail_wide = (
        enriched_for_labels.pivot(index="cluster", columns="rank", values="stability_pct")
        .rename(columns=lambda c: f"label_gene_{c}_stability_pct")
    )

    cluster_label_detailed = (
        cluster_label_table.merge(label_detail_wide.reset_index(), on="cluster", how="left")
        .merge(stability_detail_wide.reset_index(), on="cluster", how="left")
    )
    return cluster_label_detailed


def _run_leiden_sweep_on_adata(
    adata_in,
    reference_cluster_col: str = "cluster",
    n_neighbors_range: list[int] | None = None,
    resolution_range: list[float] | None = None,
    compute_umap: bool = True,
):
    """Run Leiden parameter sweep without any gene-based filtering."""
    if n_neighbors_range is None:
        n_neighbors_range = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    if resolution_range is None:
        resolution_range = [0.3, 0.5, 0.8, 1.0, 1.5]

    adata = adata_in.copy()
    sweep_rows = []
    for n_neighbors in n_neighbors_range:
        for resolution in resolution_range:
            sc.pp.neighbors(adata, use_rep="X", n_neighbors=n_neighbors)
            sc.tl.leiden(adata, resolution=resolution, key_added="leiden_test", flavor="igraph")
            ari = adjusted_rand_score(
                adata.obs[reference_cluster_col].astype(str),
                adata.obs["leiden_test"].astype(str),
            )
            sweep_rows.append(
                {
                    "n_neighbors": n_neighbors,
                    "resolution": resolution,
                    "ari": ari,
                }
            )

    sweep_df = pd.DataFrame(sweep_rows).sort_values("ari", ascending=False).reset_index(drop=True)
    best = sweep_df.iloc[0]

    best_n = int(best["n_neighbors"])
    best_r = float(best["resolution"])
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=best_n)
    sc.tl.leiden(adata, resolution=best_r, flavor="igraph")
    if compute_umap:
        sc.tl.umap(adata)

    return adata, {
        "sweep_df": sweep_df,
        "best_params": {
            "best_n_neighbors": best_n,
            "best_resolution": best_r,
            "best_ari": float(best["ari"]),
        },
    }


def _build_subclass_sorted_labels(
    adata_sub,
    cluster_col: str,
    cluster_label_df: pd.DataFrame | None,
) -> tuple[pd.Series, dict[str, str], dict[str, str]]:
    """Create ordered labels and text-color map using dominant subclass-marker expression."""
    cluster_ids = adata_sub.obs[cluster_col].astype(str)

    cxg = pd.DataFrame(
        adata_sub.X if not hasattr(adata_sub.X, "toarray") else adata_sub.X.toarray(),
        columns=adata_sub.var_names,
        index=adata_sub.obs_names,
    )
    marker_cols = [g for g in ["Lamp5", "Vip", "Sst", "Pvalb"] if g in cxg.columns]
    if not marker_cols:
        marker_cols = [c for c in cxg.columns[:1]]

    mean_markers = (
        cxg[marker_cols]
        .assign(_cluster=cluster_ids.values)
        .groupby("_cluster")
        .mean()
    )
    dominant = mean_markers.idxmax(axis=1).to_dict()
    dom_score = mean_markers.max(axis=1).to_dict()

    priority = {"Lamp5": 0, "Vip": 1, "Sst": 2, "Pvalb": 3}
    sorted_clusters = sorted(
        mean_markers.index.astype(str).tolist(),
        key=lambda c: (priority.get(str(dominant.get(c, "Other")), 9), -float(dom_score.get(c, 0.0)), c),
    )

    label_map = {}
    if cluster_label_df is not None and not cluster_label_df.empty and {"cluster", "cluster_label_string"}.issubset(set(cluster_label_df.columns)):
        label_map = dict(
            zip(
                cluster_label_df["cluster"].astype(str),
                cluster_label_df["cluster_label_string"].astype(str),
            )
        )

    subclass_color_map = {
        "Lamp5": "#4DAF4A",
        "Vip": "#984EA3",
        "Sst": "#E41A1C",
        "Pvalb": "#377EB8",
        "Other": "#4d4d4d",
    }

    # Build prefixed labels so viz natural-sorting follows our desired order.
    cluster_display_map: dict[str, str] = {}
    cluster_text_color_map: dict[str, str] = {}
    for rank, cl in enumerate(sorted_clusters):
        sub = str(dominant.get(cl, "Other"))
        sub = sub if sub in subclass_color_map else "Other"
        base = f"{cl}: {label_map.get(cl, cl)}"
        prefixed = f"{rank:02d}__{sub}__{base}"
        cluster_display_map[cl] = prefixed
        cluster_text_color_map[prefixed] = subclass_color_map[sub]

    labels_for_plot = cluster_ids.map(cluster_display_map)
    return labels_for_plot, cluster_display_map, cluster_text_color_map


def _save_cellxgene_and_cluster_mean_plots(
    adata_sub,
    out_dir: Path,
    title_prefix: str,
    cluster_col: str = "leiden",
    cluster_label_df: pd.DataFrame | None = None,
    clip_range: tuple[float, float] = (0, 30),
) -> None:
    """Save two notebook-style plots: labeled cell x gene and cluster x gene mean heatmap."""
    labels_for_plot, cluster_display_map, cluster_text_color_map = _build_subclass_sorted_labels(
        adata_sub=adata_sub,
        cluster_col=cluster_col,
        cluster_label_df=cluster_label_df,
    )

    cxg_sub = pd.DataFrame(
        adata_sub.X if not hasattr(adata_sub.X, "toarray") else adata_sub.X.toarray(),
        columns=adata_sub.var_names,
        index=adata_sub.obs_names,
    )

    # Plot 1: cell x gene in viz style.
    fig1, _, _ = viz.plot_cell_x_gene_labeled(
        cxg_sub,
        labels=labels_for_plot,
        clip_range=clip_range,
        fig_size=(8, 10),
        label_fontsize=8,
        title=f"{title_prefix} (ordered by Leiden clusters)",
    )
    ax1 = fig1.axes[0]
    # Remove x-label text and recolor/relabel cluster annotations.
    ax1.set_xlabel("")
    for txt in ax1.texts:
        s = txt.get_text()
        if "__" in s:
            parts = s.split("__", 2)
            if len(parts) == 3:
                pretty = parts[2]
                txt.set_text(pretty)
        txt.set_color(cluster_text_color_map.get(s, txt.get_color()))
    _save_fig(fig1, out_dir / "13_cell_x_gene_labeled.png")
    _safe_close(fig1)

    # Plot 2: cluster x gene mean-expression heatmap with square cells.
    mean_expr_by_cluster = (
        cxg_sub.assign(_cluster=adata_sub.obs[cluster_col].astype(str).values)
        .groupby("_cluster")
        .mean()
    )
    mean_expr_plot = mean_expr_by_cluster.copy()
    # Reuse same sorted order and pretty display text as the labeled cell plot.
    inv_display = {v: k for k, v in cluster_display_map.items()}
    sort_keys = []
    for cl in mean_expr_plot.index.astype(str):
        disp = cluster_display_map.get(cl)
        sort_keys.append(disp if disp is not None else f"99__Other__{cl}: {cl}")
    mean_expr_plot = mean_expr_plot.assign(_sort_key=sort_keys).sort_values("_sort_key").drop(columns=["_sort_key"])
    mean_expr_plot.index = [k.split("__", 2)[2] if "__" in k else str(k) for k in mean_expr_plot.index]

    n_rows, n_cols = mean_expr_plot.shape
    fig2_w = max(8, 0.45 * n_cols)
    fig2_h = max(4, 0.45 * n_rows)
    fig2, ax2 = plt.subplots(figsize=(fig2_w, fig2_h))
    sns.heatmap(
        mean_expr_plot,
        ax=ax2,
        cmap="magma",
        square=True,
        cbar_kws={"label": "Mean expression", "shrink": 0.85},
    )
    ax2.set_title("Cluster x gene mean expression")
    ax2.set_xlabel("gene")
    ax2.set_ylabel(cluster_col)
    ax2.tick_params(axis="x", labelrotation=90)
    ax2.tick_params(axis="y", labelrotation=0)
    plt.tight_layout()
    _save_fig(fig2, out_dir / "14_cluster_x_gene_mean_expression.png")
    _safe_close(fig2)

    mean_expr_by_cluster.to_csv(out_dir / "cluster_x_gene_mean_expression.csv")


def run_subclass_analysis(
    adata_log_inh,
    gene: str,
    expression_threshold: float,
    out_dir: Path,
    min_cells_per_reference_cluster: int = 10,
    old_label_col: str = "cluster",
    new_label_col: str = "leiden",
    top_n_genes: int = 3,
    bootstrap_iterations: int = 100,
    stability_threshold: float = 80.0,
    top_n_pairs: int = 3,
    pair_min_nonzero_fraction: float = 0.05,
) -> dict[str, Any]:
    print(f"\n=== Running subclass analysis: {gene}+ (threshold>{expression_threshold}) ===")
    subclass_out = out_dir / gene.lower()
    subclass_out.mkdir(parents=True, exist_ok=True)

    adata_sub, subclass_results = cluster_validation_utils.cluster_gene_positive_cells(
        adata_log_inh,
        gene=gene,
        expression_threshold=expression_threshold,
        reference_cluster_col=old_label_col,
        min_cells_per_reference_cluster=min_cells_per_reference_cluster,
        compute_umap=True,
    )

    # 1) Sweep heatmap
    fig = cluster_validation_utils.plot_clustering_sweep_heatmap(
        subclass_results["sweep_df"],
        metric_col="ari",
        title=f"ARI of Leiden clustering vs Tasic {gene}+ clusters",
        figsize=(5, 5),
    )
    _save_fig(fig, subclass_out / "01_clustering_sweep_heatmap.png")
    _safe_close(fig)

    # 2) Composition stacked bar
    fig = cluster_validation_utils.plot_new_cluster_composition_stacked(
        adata_sub,
        old_label_col=old_label_col,
        new_label_col=new_label_col,
        normalize=True,
        min_fraction=0.02,
        figsize=(12, 5),
        title=f"Original cluster composition within each Leiden cluster ({gene}+)",
    )
    _save_fig(fig, subclass_out / "02_new_cluster_composition_stacked.png")
    _safe_close(fig)

    # 3) Overlap + enrichment heatmaps
    fig = cluster_validation_utils.plot_old_new_overlap_heatmap_hierarchical(
        adata_sub,
        old_label_col=old_label_col,
        new_label_col=new_label_col,
        normalize="old",
        figsize=(7, 5),
        title=f"Old vs new overlap (row-normalized, hierarchical) ({gene}+)",
    )
    _save_fig(fig, subclass_out / "03_old_new_overlap_hierarchical.png")
    _safe_close(fig)

    fig = cluster_validation_utils.plot_old_new_log2_enrichment_heatmap(
        adata_sub,
        old_label_col=old_label_col,
        new_label_col=new_label_col,
        vmax_abs=3.0,
        figsize=(7, 5),
        title=f"Old vs new enrichment (log2 observed/expected) ({gene}+)",
    )
    _save_fig(fig, subclass_out / "04_old_new_log2_enrichment.png")
    _safe_close(fig)

    # 4) Accuracy metrics subplots
    fig, results = cluster_validation_utils.plot_accuracy_metrics_subplots(
        adata_sub,
        old_label_col=old_label_col,
        new_label_col=new_label_col,
        title_prefix=f"Small gene panel accuracy ({gene}+ cells)",
    )
    _save_fig(fig, subclass_out / "05_accuracy_metrics_subplots.png")
    _safe_close(fig)

    # Save summary metrics
    auc_series = pd.Series(results["auc_per_old_label"]).sort_values(ascending=False)
    purity_series = pd.Series(results["purity_metrics"]["purity_by_new_label"]).sort_values(ascending=False)
    metrics_df = pd.DataFrame(
        {
            "metric": [
                "auc_mean", "auc_median", "auc_std", "auc_high_gt_0.8_count", "auc_n_labels",
                "purity_mean", "purity_median", "purity_std", "purity_high_gt_0.8_count", "purity_n_clusters",
            ],
            "value": [
                auc_series.mean(),
                auc_series.median(),
                auc_series.std(),
                float((auc_series > 0.8).sum()),
                float(len(auc_series)),
                purity_series.mean(),
                purity_series.median(),
                purity_series.std(),
                float((purity_series > 0.8).sum()),
                float(len(purity_series)),
            ],
        }
    )
    metrics_df.to_csv(subclass_out / "metrics_summary.csv", index=False)

    # 5) k-NN confidence old/new labels
    fig_old, soft_old = cluster_validation_utils.plot_knn_confidence_subplots(
        adata_sub,
        label_col=old_label_col,
        n_neighbors=15,
        title=f"Panel confidence - recovering original '{old_label_col}' labels ({gene}+)"
    )
    _save_fig(fig_old, subclass_out / "06_knn_confidence_old_labels.png")
    _safe_close(fig_old)

    fig_new, soft_new = cluster_validation_utils.plot_knn_confidence_subplots(
        adata_sub,
        label_col=new_label_col,
        n_neighbors=15,
        title=f"Panel confidence - new '{new_label_col}' cluster assignments ({gene}+)"
    )
    _save_fig(fig_new, subclass_out / "07_knn_confidence_new_labels.png")
    _safe_close(fig_new)

    knn_summary = pd.DataFrame(
        {
            "analysis": ["recover_old", "new_clusters"],
            "mean_confidence": [soft_old["_confidence"].mean(), soft_new["_confidence"].mean()],
            "median_confidence": [soft_old["_confidence"].median(), soft_new["_confidence"].median()],
            "mean_margin": [soft_old["_margin"].mean(), soft_new["_margin"].mean()],
            "median_margin": [soft_old["_margin"].median(), soft_new["_margin"].median()],
            "frac_conf_gt_0_8": [
                float((soft_old["_confidence"] > 0.8).mean()),
                float((soft_new["_confidence"] > 0.8).mean()),
            ],
            "frac_conf_lt_0_5": [
                float((soft_old["_confidence"] < 0.5).mean()),
                float((soft_new["_confidence"] < 0.5).mean()),
            ],
        }
    )
    knn_summary.to_csv(subclass_out / "knn_confidence_summary.csv", index=False)

    # 6) Discriminable genes + stability
    markers_df = cluster_validation_utils.top_discriminable_genes_per_cluster(
        adata_sub,
        cluster_col=new_label_col,
        top_n=top_n_genes,
        exclude_genes=("Gad2", "Sst"),
        use_abs_effect_size=False,
        include_depleted=True,
        bootstrap_iterations=bootstrap_iterations,
        random_state=0,
    )
    markers_df.to_csv(subclass_out / "markers_enriched_depleted_long.csv", index=False)

    # Enriched/depleted compact tables
    if not markers_df.empty:
        enriched_table = (
            markers_df[markers_df["direction"] == "enriched"]
            .pivot(index="cluster", columns="rank", values="gene")
            .rename(columns=lambda c: f"top_enriched_{c}")
            .reset_index()
        )
        depleted_table = (
            markers_df[markers_df["direction"] == "depleted"]
            .pivot(index="cluster", columns="rank", values="gene")
            .rename(columns=lambda c: f"top_depleted_{c}")
            .reset_index()
        )
        enriched_table.to_csv(subclass_out / "markers_enriched_compact.csv", index=False)
        depleted_table.to_csv(subclass_out / "markers_depleted_compact.csv", index=False)

        plot_df = markers_df.copy()
        plot_df["label"] = plot_df["direction"] + "_r" + plot_df["rank"].astype(str)

        import seaborn as sns
        fig = plt.figure(figsize=(11, max(5, 0.45 * len(plot_df["cluster"].unique()) * 2)))
        ax = sns.barplot(
            data=plot_df,
            x="effect_size",
            y="cluster",
            hue="label",
            orient="h",
            palette="viridis",
        )
        ax.set_title(
            f"Top {top_n_genes} enriched/depleted genes per {new_label_col} cluster\n"
            f"(bootstrap stability computed over {bootstrap_iterations} resamples) ({gene}+)"
        )
        ax.set_xlabel("One-vs-rest effect size (z-scored mean difference)")
        ax.set_ylabel(new_label_col)
        ax.axvline(0, color="black", linewidth=1, alpha=0.7)
        plt.tight_layout()
        _save_fig(fig, subclass_out / "08_discriminable_genes_effect_sizes.png")
        _safe_close(fig)

    # 7) High-confidence marker table and final label CSV
    high_conf_markers = markers_df[markers_df["stability_pct"] >= stability_threshold].copy()
    high_conf_markers.to_csv(subclass_out / "high_conf_markers_long.csv", index=False)

    cluster_label_detailed = _cluster_label_table_from_markers(
        high_conf_markers,
        cluster_col=new_label_col,
        n_label_genes=3,
        label_sep=" | ",
    )
    cluster_label_detailed.to_csv(subclass_out / "final_cluster_labels.csv", index=False)

    # 7b) Notebook-style cell x gene and cluster-mean plots
    _save_cellxgene_and_cluster_mean_plots(
        adata_sub=adata_sub,
        out_dir=subclass_out,
        title_prefix=f"{gene}+ cells",
        cluster_col=new_label_col,
        cluster_label_df=cluster_label_detailed,
    )

    # 8) Signed gene pairs
    pairs_df = cluster_validation_utils.top_discriminable_gene_pairs_per_cluster(
        adata_sub,
        cluster_col=new_label_col,
        top_n=top_n_pairs,
        exclude_genes=("Gad2", "Sst"),
        min_nonzero_fraction=pair_min_nonzero_fraction,
    )
    pairs_df.to_csv(subclass_out / "signed_gene_pairs_long.csv", index=False)

    if not pairs_df.empty:
        import seaborn as sns
        fig = plt.figure(figsize=(10, max(5, 0.5 * len(pairs_df["cluster"].unique()) * top_n_pairs)))
        ax = sns.barplot(
            data=pairs_df,
            x="effect_size",
            y="cluster",
            hue="rank",
            orient="h",
            palette="magma",
        )
        for bar, (_, row) in zip(ax.patches, pairs_df.iterrows()):
            x_end = bar.get_x() + bar.get_width()
            y_mid = bar.get_y() + bar.get_height() / 2
            ax.text(
                x_end + 0.04,
                y_mid,
                row["pair_label"],
                va="center",
                ha="left",
                fontsize=7.5,
            )

        ax.set_title(
            f"Top {top_n_pairs} signed gene pairs per {new_label_col} cluster\n"
            f"(gene_A+ / gene_B+/-; one-vs-rest effect size; min_nonzero_frac={pair_min_nonzero_fraction}) ({gene}+)"
        )
        ax.set_xlabel("Combined effect size (z-scored)")
        ax.set_ylabel(new_label_col)
        ax.axvline(0, color="black", linewidth=1, alpha=0.7)
        plt.tight_layout()
        _save_fig(fig, subclass_out / "09_signed_gene_pairs_effect_sizes.png")
        _safe_close(fig)

    # 9) UMAP export for quick visual check
    if "X_umap" in adata_sub.obsm:
        fig = sc.pl.umap(adata_sub, color=new_label_col, title=f"{gene}+ cells ({new_label_col})", show=False, return_fig=True)
        _save_fig(fig, subclass_out / "10_umap_leiden.png")
        _safe_close(fig)

        fig = sc.pl.umap(adata_sub, color="subclass", title=f"{gene}+ cells (subclass)", show=False, return_fig=True)
        _save_fig(fig, subclass_out / "11_umap_subclass.png")
        _safe_close(fig)

        fig = sc.pl.umap(adata_sub, color=old_label_col, title=f"{gene}+ cells ({old_label_col})", show=False, return_fig=True)
        _save_fig(fig, subclass_out / "12_umap_old_cluster.png")
        _safe_close(fig)

    # Save small run manifest
    manifest = pd.DataFrame(
        {
            "gene": [gene],
            "expression_threshold": [expression_threshold],
            "n_cells_after_filters": [adata_sub.n_obs],
            "n_leiden_clusters": [adata_sub.obs[new_label_col].nunique()],
            "best_n_neighbors": [subclass_results["best_params"]["best_n_neighbors"]],
            "best_resolution": [subclass_results["best_params"]["best_resolution"]],
            "best_ari": [subclass_results["best_params"]["best_ari"]],
        }
    )
    manifest.to_csv(subclass_out / "run_manifest.csv", index=False)

    return {
        "gene": gene,
        "n_cells": adata_sub.n_obs,
        "n_clusters": int(adata_sub.obs[new_label_col].nunique()),
        "best_ari": float(subclass_results["best_params"]["best_ari"]),
        "out_dir": str(subclass_out),
    }


def run_all_inhibitory_analysis(
    adata_log_inh,
    out_dir: Path,
    old_label_col: str = "cluster",
    new_label_col: str = "leiden",
) -> dict[str, Any]:
    """Run a no-subclass-filter analysis on all inhibitory cells."""
    name = "all_inhibitory"
    run_out = out_dir / name
    run_out.mkdir(parents=True, exist_ok=True)

    print("\n=== Running all inhibitory analysis (no subclass filtering) ===")
    adata_sub = adata_log_inh.copy()

    # Use the same sweep strategy as subclass runs for apples-to-apples comparison.
    adata_sub, sweep_results = _run_leiden_sweep_on_adata(
        adata_sub,
        reference_cluster_col=old_label_col,
        compute_umap=True,
    )

    fig = cluster_validation_utils.plot_clustering_sweep_heatmap(
        sweep_results["sweep_df"],
        metric_col="ari",
        title="ARI of Leiden clustering vs Tasic all inhibitory clusters",
        figsize=(5, 5),
    )
    _save_fig(fig, run_out / "01_clustering_sweep_heatmap.png")
    _safe_close(fig)
    sweep_results["sweep_df"].to_csv(run_out / "sweep_results.csv", index=False)

    # Marker-based labels for consistency with subclass runs.
    markers_df = cluster_validation_utils.top_discriminable_genes_per_cluster(
        adata_sub,
        cluster_col=new_label_col,
        top_n=3,
        exclude_genes=("Gad2", "Sst"),
        use_abs_effect_size=False,
        include_depleted=True,
        bootstrap_iterations=100,
        random_state=0,
    )
    high_conf = markers_df[markers_df["stability_pct"] >= 80.0].copy()
    cluster_label_detailed = _cluster_label_table_from_markers(
        high_conf,
        cluster_col=new_label_col,
        n_label_genes=3,
        label_sep=" | ",
    )
    cluster_label_detailed.to_csv(run_out / "final_cluster_labels.csv", index=False)

    # Requested plots.
    _save_cellxgene_and_cluster_mean_plots(
        adata_sub=adata_sub,
        out_dir=run_out,
        title_prefix="All inhibitory cells",
        cluster_col=new_label_col,
        cluster_label_df=cluster_label_detailed,
    )

    # A compact manifest for this run.
    pd.DataFrame(
        {
            "run": [name],
            "n_cells": [adata_sub.n_obs],
            "n_leiden_clusters": [adata_sub.obs[new_label_col].nunique()],
            "best_n_neighbors": [sweep_results["best_params"]["best_n_neighbors"]],
            "best_resolution": [sweep_results["best_params"]["best_resolution"]],
            "best_ari": [sweep_results["best_params"]["best_ari"]],
        }
    ).to_csv(run_out / "run_manifest.csv", index=False)

    return {
        "gene": name,
        "n_cells": adata_sub.n_obs,
        "n_clusters": int(adata_sub.obs[new_label_col].nunique()),
        "best_ari": np.nan,
        "out_dir": str(run_out),
    }


def main() -> None:
    data_dir = Path("/root/capsule/data")
    ss_path = "/root/capsule/scratch/mouse_VISp_gene_expression_matrices_2018-06-14"
    out_root = Path("/root/capsule/results/tasic_subclass_batch")
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading pairwise dataset...")
    dataset, pw_ds, _ = get_hcr_dataset_pairwise(
        mouse_id="790322",
        data_dir=data_dir,
        load_spots=False,
        return_removed=False,
        coreg_cells_only=False,
    )

    _ = dataset
    cxg_inh_ad = pw_ds.load_inhibitory_cells(unmixed=True, all_spots=False, as_anndata=True)

    print("Loading Smart-seq reference and preparing filtered inhibitory view...")
    hcr_genes = cxg_inh_ad.var_names
    hcr_genes = hcr_genes[hcr_genes != "GFP"]
    smartseq_data = cluster_validation_utils.load_visp_expression(ss_path, genes=hcr_genes, layer="exon")

    smartseq_data_cpm = smartseq_data.copy()
    sc.pp.normalize_total(smartseq_data_cpm, target_sum=1e6)

    smartseq_data_log = smartseq_data_cpm.copy()
    sc.pp.log1p(smartseq_data_log, base=2)

    filtered_log = cluster_validation_utils.make_filtered_views_for_smartseq(smartseq_data_log)
    adata_log_inh = filtered_log["inhibitory"]

    # Match current notebook defaults.
    subclass_specs = [
        {"gene": "Vip", "expression_threshold": 15},
        {"gene": "Sst", "expression_threshold": 15},
        {"gene": "Pvalb", "expression_threshold": 15},
        {"gene": "Lamp5", "expression_threshold": 15},
    ]

    run_rows = []
    for spec in subclass_specs:
        run_rows.append(
            run_subclass_analysis(
                adata_log_inh=adata_log_inh,
                gene=spec["gene"],
                expression_threshold=spec["expression_threshold"],
                out_dir=out_root,
            )
        )

    run_rows.append(
        run_all_inhibitory_analysis(
            adata_log_inh=adata_log_inh,
            out_dir=out_root,
        )
    )

    summary_df = pd.DataFrame(run_rows).sort_values("gene").reset_index(drop=True)
    summary_df.to_csv(out_root / "batch_summary.csv", index=False)

    print("\nBatch completed. Summary:")
    print(summary_df.to_string(index=False))
    print(f"\nOutputs written to: {out_root}")


if __name__ == "__main__":
    main()
