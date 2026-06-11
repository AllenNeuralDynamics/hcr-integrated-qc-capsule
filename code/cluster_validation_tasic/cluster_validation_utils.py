
from typing import Literal

from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import seaborn as sns


def load_visp_expression(
    data_dir: str | Path,
    genes: list[str] | None = None,
    layer: Literal["sum", "exon", "intron"] = "sum",
) -> ad.AnnData:
    """
    Load Allen Brain Atlas VISp Smart-seq gene expression data as AnnData.

    Parameters
    ----------
    data_dir : path to the extracted dataset folder
    genes : list of gene symbols to load (e.g. ['Nos1', 'Vip']).
            If None, loads all genes (warning: large).
    layer : which counts to use for X — 'exon', 'intron', or 'sum' (default).

    Returns
    -------
    AnnData with:
      - X        : counts per `layer` (cells × genes, float32)
      - obs      : sample_name (index), class, subclass, cluster, brain_region, brain_subregion
      - var      : gene_symbol (index), gene_id, chromosome, gene_entrez_id, gene_name
    Only Glutamatergic and GABAergic cells are included.
    """
    if layer not in ("sum", "exon", "intron"):
        raise ValueError(f"layer must be 'sum', 'exon', or 'intron'; got {layer!r}")

    data_dir = Path(data_dir)
    prefix = "mouse_VISp_2018-06-14"

    # --- gene metadata ---
    genes_df = pl.read_csv(data_dir / f"{prefix}_genes-rows.csv")
    # columns: gene_symbol, gene_id, chromosome, gene_entrez_id, gene_name

    if genes is not None:
        genes_df = genes_df.filter(pl.col("gene_symbol").is_in(genes))
        if genes_df.is_empty():
            raise ValueError(f"None of the requested genes were found: {genes}")

    target_entrez = set(genes_df["gene_entrez_id"].cast(pl.Utf8).to_list())

    # --- cell metadata ---
    meta_df = (
        pl.read_csv(
            data_dir / f"{prefix}_samples-columns.csv",
            null_values=["NA"],
        )
        .select(["sample_name", "class", "subclass", "cluster", "brain_region", "brain_subregion"])
        .filter(pl.col("class").is_in(["Glutamatergic", "GABAergic"]))
    )
    neuronal_cell_ids = set(meta_df["sample_name"].to_list())

    # --- expression matrix: rows=genes, cols=cells ---
    need_exon = layer in ("sum", "exon")
    need_intron = layer in ("sum", "intron")

    row_id_col = None

    if need_exon:
        exon_full = pl.read_csv(data_dir / f"{prefix}_exon-matrix.csv", infer_schema_length=0)
        row_id_col = exon_full.columns[0]
        exon_sub = exon_full.filter(pl.col(row_id_col).is_in(target_entrez))
    if need_intron:
        intron_full = pl.read_csv(data_dir / f"{prefix}_intron-matrix.csv", infer_schema_length=0)
        row_id_col = intron_full.columns[0]
        intron_sub = intron_full.filter(pl.col(row_id_col).is_in(target_entrez))

    entrez_to_symbol = dict(
        zip(
            genes_df["gene_entrez_id"].cast(pl.Utf8).to_list(),
            genes_df["gene_symbol"].to_list(),
        )
    )

    ref_full = exon_full if need_exon else intron_full
    all_cell_ids = ref_full.columns[1:]
    keep_col_idx = [i for i, cid in enumerate(all_cell_ids) if cid in neuronal_cell_ids]
    keep_cell_ids = [all_cell_ids[i] for i in keep_col_idx]

    gene_symbols = []
    count_rows = []

    if layer == "sum":
        rows = zip(exon_sub.iter_rows(), intron_sub.iter_rows())
        for row_e, row_i in rows:
            entrez_id = row_e[0]
            all_counts = [int(e) + int(i) for e, i in zip(row_e[1:], row_i[1:])]
            count_rows.append([all_counts[i] for i in keep_col_idx])
            gene_symbols.append(entrez_to_symbol[entrez_id])
    elif layer == "exon":
        for row in exon_sub.iter_rows():
            entrez_id = row[0]
            all_counts = [int(v) for v in row[1:]]
            count_rows.append([all_counts[i] for i in keep_col_idx])
            gene_symbols.append(entrez_to_symbol[entrez_id])
    else:  # intron
        for row in intron_sub.iter_rows():
            entrez_id = row[0]
            all_counts = [int(v) for v in row[1:]]
            count_rows.append([all_counts[i] for i in keep_col_idx])
            gene_symbols.append(entrez_to_symbol[entrez_id])

    # Build X as (n_cells, n_genes)
    X = np.array(count_rows, dtype=np.float32).T

    obs = (
        meta_df.to_pandas()
        .set_index("sample_name")
        .loc[keep_cell_ids]
    )
    var = (
        genes_df.to_pandas()
        .set_index("gene_symbol")
        .loc[gene_symbols]
    )

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["layer"] = layer
    return adata


def plot_cell_count_by_subregion(adata: ad.AnnData) -> plt.Figure:
    sns.set_context("notebook", font_scale=1.2)

    counts = (
        adata.obs
        .groupby(["brain_subregion", "class"])
        .size()
        .reset_index(name="cell_count")
        .sort_values("brain_subregion")
    )

    subregion_order = (
        counts.groupby("brain_subregion")["cell_count"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    class_colors = {"Glutamatergic": "#E07B39", "GABAergic": "#4878CF"}

    fig, ax = plt.subplots(figsize=(6, max(4, len(subregion_order) * 0.4)))

    lefts = {sr: 0 for sr in subregion_order}
    for cls, color in class_colors.items():
        cls_counts = counts[counts["class"] == cls].set_index("brain_subregion")["cell_count"]
        values = [cls_counts.get(sr, 0) for sr in subregion_order]
        ax.barh(subregion_order, values, left=[lefts[sr] for sr in subregion_order],
                color=color, label=cls)
        for sr, v in zip(subregion_order, values):
            lefts[sr] += v

    ax.set_xlabel("Cell count")
    ax.set_title("Cell count per brain subregion")
    ax.legend(title="Class")
    plt.tight_layout()
    return fig


def filter_subclass(
    adata: ad.AnnData,
    exclude_subclasses: set[str] | None = None,
    exclude_brain_regions: set[str] | None = None,
) -> ad.AnnData:
    obs = adata.obs
    mask = obs["subclass"].notna().values
    if exclude_subclasses:
        mask &= ~obs["subclass"].isin(exclude_subclasses).values
    if exclude_brain_regions:
        mask &= ~obs["brain_subregion"].isin(exclude_brain_regions).values
    return adata[mask].copy()


def plot_subclass_heatmap(
    adata: ad.AnnData,
    title: str,
) -> plt.Figure:
    gene_cols = adata.var_names.tolist()
    subclasses = sorted(adata.obs["subclass"].dropna().unique())

    heatmap_data = np.zeros((len(subclasses), len(gene_cols)))
    for i, sc_name in enumerate(subclasses):
        mask = adata.obs["subclass"].values == sc_name
        heatmap_data[i] = adata.X[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(len(gene_cols) * .5, len(subclasses) * 0.35 + 1))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="magma")

    ax.set_xticks(range(len(gene_cols)))
    ax.set_xticklabels(gene_cols, rotation=90, ha="right")
    ax.set_yticks(range(len(subclasses)))
    ax.set_yticklabels(subclasses)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Subclass")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Mean expression")
    plt.tight_layout()
    return fig


def plot_cell_gene_with_cluster_means(
    adata: ad.AnnData,
    cluster_col: str = "leiden",
    gene_order: list[str] | None = None,
    cluster_order: list[str] | None = None,
    cluster_label_map: dict[str, str] | None = None,
    cluster_label_df=None,
    cluster_label_col: str = "cluster_label_string",
    clip_range: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (14, 6),
    cmap: str = "magma",
    title: str | None = None,
) -> tuple[plt.Figure, "pd.DataFrame"]:
    """
    Plot two aligned heatmaps:
      1) Cell x gene matrix ordered by cluster
      2) Mean gene expression per cluster

    Parameters
    ----------
    adata : AnnData
        Expression matrix (cells x genes).
    cluster_col : str
        Column in adata.obs containing cluster labels (default: "leiden").
    gene_order : list[str] | None
        Optional ordered subset of genes to plot. If None, uses adata.var_names order.
    cluster_order : list[str] | None
        Optional ordered subset/order of clusters. If None, uses sorted unique labels.
    cluster_label_map : dict[str, str] | None
        Optional mapping from cluster id -> display label.
    cluster_label_df : pd.DataFrame | None
        Optional table with columns [cluster_col, cluster_label_col] used to build
        cluster_label_map.
    cluster_label_col : str
        Label column name in cluster_label_df (default: "cluster_label_string").
    clip_range : tuple[float, float] | None
        Optional clipping range for expression values before plotting.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap for both heatmaps.
    title : str | None
        Optional figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    mean_df : pd.DataFrame
        Cluster x gene mean expression table used for the right subplot.
    """
    import pandas as pd

    if cluster_col not in adata.obs.columns:
        raise ValueError(f"cluster_col '{cluster_col}' not found in adata.obs")

    if gene_order is None:
        genes = list(adata.var_names.astype(str))
    else:
        missing = [g for g in gene_order if g not in adata.var_names]
        if missing:
            raise ValueError(f"Genes not found in adata.var_names: {missing}")
        genes = list(gene_order)

    if cluster_order is None:
        clusters = sorted(adata.obs[cluster_col].astype(str).unique().tolist())
    else:
        clusters = [str(c) for c in cluster_order]

    labels = adata.obs[cluster_col].astype(str).values
    cluster_mask = np.isin(labels, clusters)
    if not np.any(cluster_mask):
        raise ValueError("No cells left after applying cluster_order filter")

    adata_use = adata[cluster_mask, genes].copy()
    labels_use = adata_use.obs[cluster_col].astype(str).values

    if clip_range is not None:
        if len(clip_range) != 2:
            raise ValueError("clip_range must be a (min, max) tuple")
        clip_min, clip_max = clip_range
    else:
        clip_min, clip_max = None, None

    # Build cluster label map from table if provided.
    label_map = dict(cluster_label_map) if cluster_label_map is not None else {}
    if cluster_label_df is not None:
        cols = set(cluster_label_df.columns)
        if cluster_col in cols:
            key_col = cluster_col
        elif "cluster" in cols:
            key_col = "cluster"
        else:
            raise ValueError(
                f"cluster_label_df must contain '{cluster_col}' or 'cluster', and '{cluster_label_col}'; "
                f"found {cols}"
            )
        if cluster_label_col not in cols:
            raise ValueError(
                f"cluster_label_df must contain '{cluster_label_col}'; found {cols}"
            )

        tmp = cluster_label_df[[key_col, cluster_label_col]].copy()
        tmp[key_col] = tmp[key_col].astype(str)
        tmp[cluster_label_col] = tmp[cluster_label_col].astype(str)
        label_map.update(dict(zip(tmp[key_col], tmp[cluster_label_col])))

    # Order cells by cluster for the left heatmap.
    cat = pd.Categorical(labels_use, categories=clusters, ordered=True)
    order_idx = np.argsort(cat.codes)

    X = adata_use.X if not hasattr(adata_use.X, "toarray") else adata_use.X.toarray()
    X = np.asarray(X, dtype=float)
    X_cells = X[order_idx, :]
    labels_ord = labels_use[order_idx]

    if clip_min is not None and clip_max is not None:
        X_cells_plot = np.clip(X_cells, clip_min, clip_max)
    else:
        X_cells_plot = X_cells

    # Mean expression per cluster for right heatmap.
    mean_rows = []
    present_clusters = []
    for cl in clusters:
        m = labels_use == cl
        if not np.any(m):
            continue
        present_clusters.append(cl)
        mean_rows.append(X[m].mean(axis=0))

    mean_mat = np.vstack(mean_rows)
    if clip_min is not None and clip_max is not None:
        mean_mat_plot = np.clip(mean_mat, clip_min, clip_max)
    else:
        mean_mat_plot = mean_mat

    mean_df = pd.DataFrame(mean_mat, index=present_clusters, columns=genes)

    display_labels = [f"{cl}: {label_map.get(cl, cl)}" for cl in present_clusters]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [3.2, 1.3]},
    )

    # Left: cell x gene
    im1 = ax1.imshow(X_cells_plot, aspect="auto", cmap=cmap)
    ax1.set_xlabel("Gene")
    ax1.set_ylabel("Cells (ordered by cluster)")
    ax1.set_title("Cell x gene (ordered by cluster)")
    ax1.set_xticks(range(len(genes)))
    ax1.set_xticklabels(genes, rotation=90, ha="right", fontsize=8)

    # Draw cluster boundaries on cell heatmap.
    boundary_positions = []
    prev = None
    for i, cl in enumerate(labels_ord):
        if prev is None:
            prev = cl
            continue
        if cl != prev:
            boundary_positions.append(i - 0.5)
            prev = cl
    for y in boundary_positions:
        ax1.axhline(y, color="white", linewidth=0.7, alpha=0.8)

    # Right: mean per cluster
    im2 = ax2.imshow(mean_mat_plot, aspect="auto", cmap=cmap)
    ax2.set_xlabel("Gene")
    ax2.set_ylabel(cluster_col)
    ax2.set_title("Mean expression per cluster")
    ax2.set_xticks(range(len(genes)))
    ax2.set_xticklabels(genes, rotation=90, ha="right", fontsize=8)
    ax2.set_yticks(range(len(display_labels)))
    ax2.set_yticklabels(display_labels, fontsize=8)

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.03, pad=0.01)
    cbar1.set_label("Expression")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.08, pad=0.03)
    cbar2.set_label("Mean expression")

    if title is None:
        title = f"Cell x gene and cluster means ({cluster_col})"
    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    return fig, mean_df



def make_filtered_views_for_smartseq(adata: ad.AnnData) -> dict[str, ad.AnnData]:
    """
    Return a dict of filtered AnnData views.

    Keys
    ----
    "all"        : neuronal cells, non-neuronal subclasses and deep-layer regions removed
    "inhibitory" : GABAergic subclasses only (Pvalb, Sst, Vip, Lamp5, ...)`
    "excitatory" : Glutamatergic cells only
    """

    # --- filter configuration ---
    _exclude_subclasses = {
        "VLMC", "Peri", "Oligo", "SMC", "No Class",
        "Macrophage", "Low Quality", "Endo", "Doublet", "Astro",
        "High Intronic", "Batch Grouping"
    }
    _exclude_brain_regions = {"L6", "L6b"}
    _inh_subclasses = {"Pvalb", "Sst", "Vip", "Lamp5", "CR", "Serpinf1", "Sncg", "Meis2"}

    base = filter_subclass(adata, _exclude_subclasses, _exclude_brain_regions)
    return {
        "all": base,
        "inhibitory": base[base.obs["subclass"].isin(_inh_subclasses)].copy(),
        "excitatory": base[base.obs["class"] == "Glutamatergic"].copy(),
    }


def cluster_gene_positive_cells(
    adata: ad.AnnData,
    gene: str,
    expression_threshold: float,
    reference_cluster_col: str = "cluster",
    min_cells_per_reference_cluster: int | None = None,
    n_neighbors_range: list[int] = None,
    resolution_range: list[float] = None,
    compute_umap: bool = True,
) -> tuple[ad.AnnData, dict]:
    """
    Filter cells by gene expression threshold and perform Leiden clustering with parameter sweep.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object
    gene : str
        Gene name to filter by
    expression_threshold : float
        Minimum expression value for filtering
    reference_cluster_col : str, optional
        Column name in .obs containing reference clusters for ARI calculation (default: "cluster")
    min_cells_per_reference_cluster : int | None, optional
        If provided, remove cells whose reference cluster has fewer than this many
        cells (computed after gene filtering). Useful to reduce tiny unstable
        clusters before parameter sweep.
    n_neighbors_range : list[int], optional
        Values of n_neighbors to test (default: [5, 10, 15, 20, 30, 40, 50, 60, 80, 100])
    resolution_range : list[float], optional
        Resolution values to test (default: [0.3, 0.5, 0.8, 1.0, 1.5])
    compute_umap : bool, optional
        Whether to compute UMAP embedding (default: True)

    Returns
    -------
    adata_filtered : AnnData
        Filtered AnnData with Leiden clustering results in .obs['leiden']
    results_dict : dict
        Dictionary containing:
        - 'sweep_df': DataFrame with all parameter combinations and ARI scores
        - 'best_params': dict with best_n_neighbors, best_resolution, best_ari
        - 'n_cells': number of cells after filtering
        - 'n_clusters': number of Leiden clusters found with best parameters
    """
    from sklearn.metrics import adjusted_rand_score
    import pandas as pd

    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names")

    # Filter cells by gene expression
    gene_mask = adata[:, gene].X.flatten() > expression_threshold
    adata_filtered = adata[gene_mask].copy()

    print(f"Filtered to {len(adata_filtered):,} cells with {gene} > {expression_threshold}")

    n_cells_after_gene_filter = len(adata_filtered)

    # Optionally remove reference clusters with too few cells
    if min_cells_per_reference_cluster is not None:
        if min_cells_per_reference_cluster < 1:
            raise ValueError("min_cells_per_reference_cluster must be >= 1 when provided")
        if reference_cluster_col not in adata_filtered.obs.columns:
            raise ValueError(
                f"reference_cluster_col '{reference_cluster_col}' not found in adata_filtered.obs"
            )

        cluster_counts = adata_filtered.obs[reference_cluster_col].value_counts()
        keep_clusters = cluster_counts[cluster_counts >= min_cells_per_reference_cluster].index
        adata_filtered = adata_filtered[
            adata_filtered.obs[reference_cluster_col].isin(keep_clusters)
        ].copy()

        print(
            "After min cluster-size filter "
            f"({reference_cluster_col} >= {min_cells_per_reference_cluster}), "
            f"kept {len(adata_filtered):,} cells in {adata_filtered.obs[reference_cluster_col].nunique()} clusters"
        )

        if adata_filtered.n_obs == 0:
            raise ValueError(
                "No cells remain after applying min_cells_per_reference_cluster filter. "
                "Lower threshold or adjust expression_threshold."
            )

    # Set default parameter ranges
    if n_neighbors_range is None:
        n_neighbors_range = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    if resolution_range is None:
        resolution_range = [0.3, 0.5, 0.8, 1.0, 1.5]

    # Parameter sweep
    sweep_results = []
    for n_neighbors in n_neighbors_range:
        for resolution in resolution_range:
            sc.pp.neighbors(adata_filtered, use_rep="X", n_neighbors=n_neighbors)
            sc.tl.leiden(
                adata_filtered,
                resolution=resolution,
                key_added="leiden_test",
                flavor="igraph",
            )

            # Calculate ARI against reference clusters if available
            if reference_cluster_col in adata_filtered.obs.columns:
                ari = adjusted_rand_score(
                    adata_filtered.obs[reference_cluster_col],
                    adata_filtered.obs["leiden_test"],
                )
            else:
                ari = None

            sweep_results.append({
                "n_neighbors": n_neighbors,
                "resolution": resolution,
                "ari": ari
            })

    sweep_df = pd.DataFrame(sweep_results).sort_values("ari", ascending=False)
    best = sweep_df.iloc[0]

    print(f"Best params: n_neighbors={int(best.n_neighbors)}, "
          f"resolution={best.resolution:.1f}, ARI={best.ari:.3f}")

    # Run final clustering with best parameters
    best_n_neighbors = int(best.n_neighbors)
    best_resolution = best.resolution

    sc.pp.neighbors(adata_filtered, use_rep="X", n_neighbors=best_n_neighbors)
    sc.tl.leiden(adata_filtered, resolution=best_resolution, flavor="igraph")

    n_leiden_clusters = adata_filtered.obs["leiden"].nunique()
    print(f"Leiden found {n_leiden_clusters} clusters")

    # Compute UMAP if requested
    if compute_umap:
        sc.tl.umap(adata_filtered)
        print("UMAP computed")

    results_dict = {
        "sweep_df": sweep_df,
        "best_params": {
            "best_n_neighbors": best_n_neighbors,
            "best_resolution": best_resolution,
            "best_ari": best.ari,
        },
        "min_cells_per_reference_cluster": min_cells_per_reference_cluster,
        "n_cells_after_gene_filter": n_cells_after_gene_filter,
        "n_cells": len(adata_filtered),
        "n_clusters": n_leiden_clusters,
    }

    return adata_filtered, results_dict


def plot_clustering_sweep_heatmap(
    sweep_df,
    metric_col: str = "ari",
    title: str = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Plot heatmap of clustering parameter sweep results.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        DataFrame with columns: n_neighbors, resolution, and metric_col
    metric_col : str, optional
        Column name for the metric to plot (default: "ari")
    title : str, optional
        Plot title. If None, uses f"{metric_col.upper()} of Leiden clustering"
    figsize : tuple, optional
        Figure size (default: (8, 5))

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import pandas as pd

    pivot = sweep_df.pivot(index="n_neighbors", columns="resolution", values=metric_col)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)

    if title is None:
        title = f"{metric_col.upper()} of Leiden clustering"
    ax.set_title(title)
    ax.set_xlabel("Leiden resolution")
    ax.set_ylabel("Leiden n_neighbors")

    plt.tight_layout()
    return fig


def summarize_old_label_dispersion(
    adata: ad.AnnData,
    old_label_col: str,
    new_label_col: str,
):
    """
    Quantify how each old label is distributed across new labels.

    Parameters
    ----------
    adata : AnnData
        AnnData with old and new cluster labels in .obs
    old_label_col : str
        Column name for the original (more granular) labels
    new_label_col : str
        Column name for the new labels

    Returns
    -------
    dict with:
      - 'contingency': old x new count table
      - 'fraction_by_old': row-normalized old x new table
      - 'old_label_summary': one row per old label with dispersion metrics
      - 'new_label_composition': counts/fractions of old labels within each new label
    """
    import pandas as pd

    for col in (old_label_col, new_label_col):
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    df = adata.obs[[old_label_col, new_label_col]].copy()
    df = df.dropna(subset=[old_label_col, new_label_col])
    if df.empty:
        raise ValueError("No cells remain after dropping NA labels")

    old = df[old_label_col].astype(str)
    new = df[new_label_col].astype(str)

    contingency = pd.crosstab(old, new, dropna=False)
    fraction_by_old = contingency.div(contingency.sum(axis=1), axis=0)

    per_old_rows = []
    for old_label, row in contingency.iterrows():
        total_cells = int(row.sum())
        positive = row[row > 0]
        n_new_labels = int(positive.shape[0])

        frac_row = fraction_by_old.loc[old_label]
        max_idx = frac_row.idxmax()
        max_frac = float(frac_row.loc[max_idx])

        p = frac_row.values
        p_nonzero = p[p > 0]
        entropy = float(-(p_nonzero * np.log2(p_nonzero)).sum())
        if len(p) > 1:
            entropy_norm = float(entropy / np.log2(len(p)))
        else:
            entropy_norm = 0.0
        purity = float((p ** 2).sum())

        per_old_rows.append(
            {
                "old_label": old_label,
                "n_cells": total_cells,
                "n_new_labels_present": n_new_labels,
                "dominant_new_label": str(max_idx),
                "dominant_fraction": max_frac,
                "is_exclusive_to_one_new_label": n_new_labels == 1,
                "entropy": entropy,
                "normalized_entropy": entropy_norm,
                "purity": purity,
            }
        )

    old_label_summary = pd.DataFrame(per_old_rows)
    old_label_summary = old_label_summary.sort_values(
        ["is_exclusive_to_one_new_label", "dominant_fraction", "n_cells"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    old_label_summary["dispersion_rank"] = (
        old_label_summary["normalized_entropy"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    by_new = pd.crosstab(new, old, dropna=False)
    by_new_frac = by_new.div(by_new.sum(axis=1), axis=0)
    new_label_composition = (
        by_new.stack().rename("count").reset_index().rename(
            columns={new_label_col: "new_label", old_label_col: "old_label"}
        )
    )
    new_label_composition["fraction_within_new_label"] = by_new_frac.stack().values
    new_label_composition = new_label_composition.sort_values(
        ["new_label", "fraction_within_new_label", "count"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return {
        "contingency": contingency,
        "fraction_by_old": fraction_by_old,
        "old_label_summary": old_label_summary,
        "new_label_composition": new_label_composition,
    }


def plot_new_cluster_composition_stacked(
    adata: ad.AnnData,
    old_label_col: str,
    new_label_col: str,
    normalize: bool = True,
    min_fraction: float = 0.0,
    figsize: tuple = (10, 5),
    title: str | None = None,
) -> plt.Figure:
    """
    Plot composition of old labels within each new cluster as a stacked bar plot.

    Parameters
    ----------
    adata : AnnData
        AnnData with old and new cluster labels in .obs
    old_label_col : str
        Column name for old labels
    new_label_col : str
        Column name for new labels
    normalize : bool, optional
        If True, plot fractions (bars sum to 1 per new cluster), else raw counts
    min_fraction : float, optional
        For normalized plots, merge old labels below this fraction into "Other"
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    """
    import pandas as pd

    for col in (old_label_col, new_label_col):
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    df = adata.obs[[old_label_col, new_label_col]].copy()
    df = df.dropna(subset=[old_label_col, new_label_col])
    if df.empty:
        raise ValueError("No cells remain after dropping NA labels")

    old = df[old_label_col].astype(str)
    new = df[new_label_col].astype(str)
    table = pd.crosstab(new, old, dropna=False)

    plot_table = table.copy()
    ylabel = "Cell count"
    if normalize:
        plot_table = table.div(table.sum(axis=1), axis=0)
        ylabel = "Fraction of cells in new cluster"

        if min_fraction > 0:
            rare = plot_table.max(axis=0) < min_fraction
            if rare.any():
                other_vals = plot_table.loc[:, rare].sum(axis=1)
                plot_table = plot_table.loc[:, ~rare]
                plot_table["Other"] = other_vals

    # Plot dominant old labels first for readability
    col_order = plot_table.sum(axis=0).sort_values(ascending=False).index
    plot_table = plot_table[col_order]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(plot_table.index))
    bottom = np.zeros(len(plot_table.index), dtype=float)

    palette = sns.color_palette("tab20", n_colors=plot_table.shape[1])
    for i, old_label in enumerate(plot_table.columns):
        values = plot_table[old_label].values
        ax.bar(x, values, bottom=bottom, label=old_label, color=palette[i])
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(plot_table.index.astype(str), rotation=0)
    ax.set_xlabel(new_label_col)
    ax.set_ylabel(ylabel)

    if title is None:
        mode = "fraction" if normalize else "count"
        title = f"Old-label composition within each new cluster ({mode})"
    ax.set_title(title)

    ax.legend(
        title=old_label_col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
    )

    plt.tight_layout()
    return fig


def plot_old_new_overlap_heatmap_hierarchical(
    adata: ad.AnnData,
    old_label_col: str,
    new_label_col: str,
    normalize: Literal["old", "new", "none"] = "old",
    figsize: tuple = (10, 8),
    cmap: str = "mako",
    title: str | None = None,
) -> plt.Figure:
    """
    Plot old-vs-new label overlap heatmap with hierarchical ordering.

    Parameters
    ----------
    adata : AnnData
        AnnData with labels in .obs
    old_label_col : str
        Original label column (rows)
    new_label_col : str
        New label column (columns)
    normalize : {"old", "new", "none"}, optional
        Normalization axis:
        - "old": each old label row sums to 1
        - "new": each new label column sums to 1
        - "none": raw counts
    figsize : tuple, optional
        Figure size
    cmap : str, optional
        Colormap for heatmap
    title : str, optional
        Custom title
    """
    import pandas as pd
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    for col in (old_label_col, new_label_col):
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    df = adata.obs[[old_label_col, new_label_col]].copy().dropna()
    if df.empty:
        raise ValueError("No cells remain after dropping NA labels")

    table = pd.crosstab(df[old_label_col].astype(str), df[new_label_col].astype(str), dropna=False)

    if normalize == "old":
        plot_data = table.div(table.sum(axis=1), axis=0).fillna(0)
        cbar_label = "Fraction within old label"
    elif normalize == "new":
        plot_data = table.div(table.sum(axis=0), axis=1).fillna(0)
        cbar_label = "Fraction within new label"
    elif normalize == "none":
        plot_data = table.copy()
        cbar_label = "Cell count"
    else:
        raise ValueError("normalize must be one of: 'old', 'new', 'none'")

    # Hierarchical ordering of rows and columns to reveal block structure.
    row_order = plot_data.index
    col_order = plot_data.columns
    if plot_data.shape[0] > 1:
        row_dist = pdist(plot_data.values, metric="euclidean")
        row_link = linkage(row_dist, method="average")
        row_order = plot_data.index[leaves_list(row_link)]
    if plot_data.shape[1] > 1:
        col_dist = pdist(plot_data.values.T, metric="euclidean")
        col_link = linkage(col_dist, method="average")
        col_order = plot_data.columns[leaves_list(col_link)]

    ordered = plot_data.loc[row_order, col_order]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(ordered, cmap=cmap, ax=ax, cbar_kws={"label": cbar_label})
    ax.set_xlabel(new_label_col)
    ax.set_ylabel(old_label_col)
    if title is None:
        title = f"Old vs new overlap (hierarchical ordering, normalize={normalize})"
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_old_new_log2_enrichment_heatmap(
    adata: ad.AnnData,
    old_label_col: str,
    new_label_col: str,
    eps: float = 1e-9,
    vmax_abs: float | None = 3.0,
    figsize: tuple = (10, 8),
    title: str | None = None,
) -> plt.Figure:
    """
    Plot log2 enrichment heatmap for old-vs-new labels.

    Each cell shows log2(observed / expected), where expected is based on
    independent marginals. Positive values indicate enrichment, negative values
    indicate depletion.
    """
    import pandas as pd
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    for col in (old_label_col, new_label_col):
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    df = adata.obs[[old_label_col, new_label_col]].copy().dropna()
    if df.empty:
        raise ValueError("No cells remain after dropping NA labels")

    obs = pd.crosstab(df[old_label_col].astype(str), df[new_label_col].astype(str), dropna=False)
    total = obs.values.sum()

    row_marg = obs.sum(axis=1).values[:, None]
    col_marg = obs.sum(axis=0).values[None, :]
    expected = (row_marg * col_marg) / total

    enrich = np.log2((obs.values + eps) / (expected + eps))
    enrich_df = pd.DataFrame(enrich, index=obs.index, columns=obs.columns)

    row_order = enrich_df.index
    col_order = enrich_df.columns
    if enrich_df.shape[0] > 1:
        row_dist = pdist(enrich_df.values, metric="euclidean")
        row_link = linkage(row_dist, method="average")
        row_order = enrich_df.index[leaves_list(row_link)]
    if enrich_df.shape[1] > 1:
        col_dist = pdist(enrich_df.values.T, metric="euclidean")
        col_link = linkage(col_dist, method="average")
        col_order = enrich_df.columns[leaves_list(col_link)]

    ordered = enrich_df.loc[row_order, col_order]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        ordered,
        cmap="RdBu_r",
        center=0,
        vmin=(-vmax_abs if vmax_abs is not None else None),
        vmax=(vmax_abs if vmax_abs is not None else None),
        ax=ax,
        cbar_kws={"label": "log2(observed / expected)"},
    )
    ax.set_xlabel(new_label_col)
    ax.set_ylabel(old_label_col)
    if title is None:
        title = "Old vs new label enrichment (log2 observed/expected)"
    ax.set_title(title)
    plt.tight_layout()
    return fig


def _leading_token(label: str) -> str:
    """Return the first whitespace-delimited token from a label."""
    parts = str(label).strip().split()
    return parts[0] if parts else ""


def map_colors_to_labels(
    labels,
    token_color_map: dict[str, str],
    default_color: str = "#9e9e9e",
    match_mode: Literal["contains", "leading_token"] = "contains",
    case_sensitive: bool = False,
) -> list[str]:
    """
    Map colors to labels using either substring matching or leading-token matching.

    Parameters
    ----------
    labels : iterable
        Sequence of labels to color.
    token_color_map : dict[str, str]
        Mapping from token/key to color.
    default_color : str, optional
        Fallback color when no key matches.
    match_mode : {"contains", "leading_token"}, optional
        Matching strategy:
        - "contains": key appears anywhere in the label
        - "leading_token": first token of label equals key
    case_sensitive : bool, optional
        Whether matching should preserve case.

    Returns
    -------
    list[str]
        Color for each input label.
    """
    if match_mode not in {"contains", "leading_token"}:
        raise ValueError("match_mode must be 'contains' or 'leading_token'")

    if case_sensitive:
        rules = [(str(k), v) for k, v in token_color_map.items()]
    else:
        rules = [(str(k).lower(), v) for k, v in token_color_map.items()]

    colors = []
    for label in labels:
        label_str = str(label)
        probe = label_str if case_sensitive else label_str.lower()
        lead_probe = _leading_token(label_str)
        if not case_sensitive:
            lead_probe = lead_probe.lower()

        color = default_color
        for key, mapped_color in rules:
            if match_mode == "contains" and key in probe:
                color = mapped_color
                break
            if match_mode == "leading_token" and key == lead_probe:
                color = mapped_color
                break
        colors.append(color)

    return colors


def color_axis_ticklabels(
    ax,
    axis: Literal["x", "y"] = "x",
    token_color_map: dict[str, str] | None = None,
    default_color: str = "#9e9e9e",
    match_mode: Literal["contains", "leading_token"] = "contains",
    case_sensitive: bool = False,
) -> list[str]:
    """
    Color tick label text for an axis based on label text matching rules.

    Works with any matplotlib plot after ticks/ticklabels are set.
    Returns the applied color list.
    """
    if token_color_map is None:
        token_color_map = {}

    ticklabels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    label_text = [t.get_text() for t in ticklabels]

    colors = map_colors_to_labels(
        label_text,
        token_color_map=token_color_map,
        default_color=default_color,
        match_mode=match_mode,
        case_sensitive=case_sensitive,
    )

    for tick, color in zip(ticklabels, colors):
        tick.set_color(color)

    return colors


def subclass_palette_from_labels(
    labels,
    subclass_color_map: dict[str, str] | None = None,
    default_color: str = "#9e9e9e",
) -> list[str]:
    """
    Build a color list from label strings using subclass keyword matching.

    A label is colored when it contains one of the subclass keys (case-insensitive).
    Example defaults: Sst=red, Pvalb=blue, Vip=purple, Lamp5=green.
    """
    if subclass_color_map is None:
        subclass_color_map = {
            "sst": "red",
            "pvalb": "blue",
            "vip": "purple",
            "lamp5": "green",
        }

    return map_colors_to_labels(
        labels,
        token_color_map=subclass_color_map,
        default_color=default_color,
        match_mode="contains",
        case_sensitive=False,
    )


def plot_gene_expression_subplots(adata, genes, bins=50, ncols=4, figsize=None, title_prefix=""):
    """
    Plot expression distributions for multiple genes as subplots.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data in .X
    genes : list of str
        List of gene names to plot
    bins : int, optional
        Number of bins for histograms (default: 50)
    ncols : int, optional
        Number of columns in subplot grid (default: 3)
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated
    title_prefix : str, optional
        Prefix for subplot titles (default: "Distribution of")
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    import math
    
    # Filter to genes that exist in the data
    available_genes = [g for g in genes if g in adata.var_names]
    missing_genes = [g for g in genes if g not in adata.var_names]
    
    if missing_genes:
        print(f"Warning: {len(missing_genes)} gene(s) not found: {missing_genes}")
    
    if not available_genes:
        print("Error: No valid genes found in dataset")
        return None
    
    n_genes = len(available_genes)
    nrows = math.ceil(n_genes / ncols)
    
    if figsize is None:
        figsize = (ncols * 4, nrows * 3)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()  # handle single subplot case
    
    for idx, gene in enumerate(available_genes):
        ax = axes[idx]
        gene_idx = adata.var_names.get_loc(gene)
        gene_expr = adata.X[:, gene_idx]
        
        sns.histplot(gene_expr, bins=bins, ax=ax, kde=False)
        ax.set_xlabel(f"Expression (log2 CPM + 1)")
        ax.set_title(f"{title_prefix} {gene}")
        ax.grid(alpha=0.3)

        # set ylims to 0-1000
        axes[idx].set_ylim(0, 1000)
    
    # Hide empty subplots
    for idx in range(n_genes, len(axes)):
        axes[idx].set_visible(False)

        
    
    plt.tight_layout()


def compute_auc_per_old_label(
    adata: ad.AnnData,
    old_label_col: str,
    new_label_col: str,
) -> dict[str, float]:
    """
    Compute AUC (one-vs-rest) for each old label using new labels as predictor.
    
    For each old label, treat it as a binary classification problem (this label vs. all others),
    and predict using the fraction of that old label within each new cluster.
    
    Parameters
    ----------
    adata : AnnData object with cluster labels in .obs
    old_label_col : str, column in adata.obs with fine-grained original labels
    new_label_col : str, column in adata.obs with coarse new labels
    
    Returns
    -------
    dict[str, float] : {old_label: auc_score}
    """
    from sklearn.metrics import roc_auc_score
    import pandas as pd
    
    # Compute contingency table
    contingency = pd.crosstab(adata.obs[old_label_col], adata.obs[new_label_col])
    
    # Normalize by new label (column) to get P(old_label | new_label)
    contingency_norm = contingency.div(contingency.sum(axis=0), axis=1)
    
    auc_dict = {}
    
    for old_label in adata.obs[old_label_col].unique():
        # Binary outcome: 1 if this old label, 0 otherwise
        y_true = (adata.obs[old_label_col] == old_label).astype(int).values
        
        # Prediction: for each cell, use the fraction of its new cluster that is this old label
        new_labels_in_new_label_col = adata.obs[new_label_col].values
        y_pred = np.array([
            contingency_norm.loc[old_label, nl] if old_label in contingency_norm.index else 0.0
            for nl in new_labels_in_new_label_col
        ])
        
        # Compute AUC
        if len(np.unique(y_true)) < 2:
            # Skip labels with no positive or negative examples
            auc = np.nan
        else:
            auc = roc_auc_score(y_true, y_pred)
        
        auc_dict[old_label] = auc
    
    return auc_dict


def compute_purity_by_new_label(
    adata: ad.AnnData,
    old_label_col: str,
    new_label_col: str,
) -> dict:
    """
    Compute purity for each new label (fraction of most common old label).
    
    Purity = (count of dominant old label) / (total count in this new label)
    
    Parameters
    ----------
    adata : AnnData object
    old_label_col : str, column in adata.obs with fine-grained original labels
    new_label_col : str, column in adata.obs with coarse new labels
    
    Returns
    -------
    dict with keys:
      - 'purity_by_new_label': {new_label: purity_value}
      - 'dominant_old_label_per_new': {new_label: dominant_old_label}
      - 'n_cells_per_new_label': {new_label: n_cells}
    """
    import pandas as pd
    
    contingency = pd.crosstab(adata.obs[old_label_col], adata.obs[new_label_col])
    
    purity_dict = {}
    dominant_dict = {}
    n_cells_dict = {}
    
    for new_label in adata.obs[new_label_col].unique():
        if new_label not in contingency.columns:
            purity_dict[new_label] = 0.0
            dominant_dict[new_label] = None
            n_cells_dict[new_label] = 0
        else:
            counts = contingency[new_label]
            total = counts.sum()
            max_count = counts.max()
            purity = max_count / total if total > 0 else 0.0
            dominant = counts.idxmax()
            
            purity_dict[new_label] = purity
            dominant_dict[new_label] = dominant
            n_cells_dict[new_label] = int(total)
    
    return {
        'purity_by_new_label': purity_dict,
        'dominant_old_label_per_new': dominant_dict,
        'n_cells_per_new_label': n_cells_dict,
    }


def plot_accuracy_metrics_subplots(
    adata: ad.AnnData,
    old_label_col: str,
    new_label_col: str,
    figsize: tuple = (14, 5),
    title_prefix: str = "Cluster Accuracy Metrics",
) -> tuple[plt.Figure, dict]:
    """
    Create side-by-side subplots showing:
    1. AUC per old label (left) — how well the small gene panel distinguishes each granular cluster
    2. Purity per new label (right) — internal homogeneity of new clusters w.r.t. old labels
    
    Parameters
    ----------
    adata : AnnData object
    old_label_col : str, column with granular original labels
    new_label_col : str, column with coarse new labels
    figsize : tuple, figure size
    title_prefix : str, title prefix for the figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    results_dict : dict with 'auc_per_old_label' and 'purity_metrics'
    """
    import pandas as pd
    
    # Compute metrics
    auc_per_old = compute_auc_per_old_label(adata, old_label_col, new_label_col)
    purity_metrics = compute_purity_by_new_label(adata, old_label_col, new_label_col)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # --- Subplot 1: AUC per old label ---
    ax1 = axes[0]
    auc_df = pd.Series(auc_per_old).sort_values(ascending=False)
    valid_auc = auc_df[~auc_df.isna()]
    
    sns.barplot(x=valid_auc.values, y=valid_auc.index, ax=ax1, palette="viridis")
    ax1.set_xlabel("AUC (one-vs-rest)")
    ax1.set_ylabel(f"{old_label_col}")
    ax1.set_title(f"Discriminability of {old_label_col}\n(higher = better distinction by new panel)")
    ax1.set_xlim([0, 1])
    ax1.axvline(0.8, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="AUC=0.8")
    ax1.axvline(0.7, color="orange", linestyle="--", linewidth=1.5, alpha=0.5, label="AUC=0.7")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(axis="x", alpha=0.3)
    
    # --- Subplot 2: Purity per new label ---
    ax2 = axes[1]
    purity_by_new = purity_metrics['purity_by_new_label']
    dominant_by_new = purity_metrics['dominant_old_label_per_new']
    n_cells_by_new = purity_metrics['n_cells_per_new_label']
    
    purity_series = pd.Series(purity_by_new).sort_values(ascending=False)
    
    # Create color map based on dominant old label
    subclass_color_map = {
        "sst": "#E41A1C",
        "pvalb": "#377EB8",
        "vip": "#984EA3",
        "lamp5": "#4DAF4A",
    }
    
    colors = []
    for new_label in purity_series.index:
        dominant = dominant_by_new.get(new_label)
        if dominant:
            dominant_lower = str(dominant).lower()
            # Check if any key matches the leading token
            color = None
            for token, col in subclass_color_map.items():
                if dominant_lower.startswith(token):
                    color = col
                    break
            if color is None:
                color = "#CCCCCC"
        else:
            color = "#CCCCCC"
        colors.append(color)
    
    bars = ax2.bar(range(len(purity_series)), purity_series.values, color=colors)
    ax2.set_xticks(range(len(purity_series)))
    ax2.set_xticklabels(purity_series.index, rotation=45, ha="right")
    ax2.set_ylabel("Purity")
    ax2.set_xlabel(f"{new_label_col}")
    ax2.set_title(f"Homogeneity of {new_label_col}\n(higher = cells from single old cluster)")
    ax2.set_ylim([0, 1.05])
    ax2.axhline(0.8, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="Purity=0.8")
    ax2.axhline(0.7, color="orange", linestyle="--", linewidth=1.5, alpha=0.5, label="Purity=0.7")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    
    # Add cell count labels on bars (if space allows)
    for idx, (new_label, purity) in enumerate(purity_series.items()):
        n_cells = n_cells_by_new.get(new_label, 0)
        if n_cells > 0:
            ax2.text(idx, purity + 0.02, f"n={n_cells}", ha="center", va="bottom", fontsize=8)
    
    fig.suptitle(title_prefix, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    return fig, {
        'auc_per_old_label': auc_per_old,
        'purity_metrics': purity_metrics,
    }


def compute_knn_soft_votes(
    adata: ad.AnnData,
    label_col: str,
    n_neighbors: int = 15,
) -> "pd.DataFrame":
    """
    Compute per-cell soft-vote probabilities using k-nearest neighbors in gene expression space.

    For each cell, the k nearest neighbors (by Euclidean distance in adata.X) cast votes
    for their label. The result is a probability vector over all labels.

    This is label-source-agnostic: pass old Tasic cluster labels to measure how well the
    small gene panel recovers the original fine-grained taxonomy, or pass Leiden/k-means
    labels to measure new-cluster confidence.

    Parameters
    ----------
    adata : AnnData object (cells × genes, log-normalised)
    label_col : str, column in adata.obs with cluster labels
    n_neighbors : int, number of nearest neighbours to use (default 15)

    Returns
    -------
    pd.DataFrame, shape (n_cells, n_labels)
        Soft-vote probability matrix. Index matches adata.obs_names.
        Columns are unique label values.
        Additional series accessible as:
          - df.max(axis=1)          → per-cell confidence (max posterior)
          - df.max(axis=1) - df.apply(lambda r: r.nlargest(2).iloc[-1], axis=1)
                                    → per-cell margin (P1 − P2)
          - (-df * np.log2(df + 1e-9)).sum(axis=1) → per-cell entropy
    """
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors

    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    labels = adata.obs[label_col].values
    unique_labels = sorted(adata.obs[label_col].unique())

    # Fit k-NN (k+1 to exclude the cell itself)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    # Drop the first column (self)
    indices = indices[:, 1:]

    # For each cell, count neighbor votes per label
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    n_cells = X.shape[0]
    n_labels = len(unique_labels)
    vote_matrix = np.zeros((n_cells, n_labels), dtype=np.float32)

    for cell_i, neighbor_idxs in enumerate(indices):
        for nb_idx in neighbor_idxs:
            lbl = labels[nb_idx]
            vote_matrix[cell_i, label_to_idx[lbl]] += 1.0

    # Normalise to probabilities
    vote_matrix /= n_neighbors

    return pd.DataFrame(vote_matrix, index=adata.obs_names, columns=unique_labels)


def plot_knn_confidence_subplots(
    adata: ad.AnnData,
    label_col: str,
    n_neighbors: int = 15,
    figsize: tuple = (14, 5),
    title: str = "k-NN soft-vote confidence",
    sort_by: str = "mean_confidence",
) -> tuple["plt.Figure", "pd.DataFrame"]:
    """
    Visualise per-cluster k-NN soft-vote confidence using two subplots:

    Left  — Mean confidence (max posterior) per cluster, sorted descending.
            Whiskers show ±1 SD across cells in that cluster.
    Right — Mean margin (P_top1 − P_top2) per cluster, sorted the same way.
            Margin distinguishes genuinely confident clusters from ambiguous ones.

    The label_col can be the **old** Tasic cluster labels (ground-truth recovery) or
    any new clustering output (Leiden, k-means, hierarchical).

    Parameters
    ----------
    adata : AnnData object
    label_col : str, column in adata.obs with cluster labels
    n_neighbors : int, k for k-NN voting (default 15)
    figsize : tuple
    title : str, figure suptitle
    sort_by : 'mean_confidence' or 'mean_margin' — which metric to sort bars by

    Returns
    -------
    fig : matplotlib.figure.Figure
    soft_probs : pd.DataFrame, soft-vote probability matrix (cells × labels)
    """
    import pandas as pd

    soft_probs = compute_knn_soft_votes(adata, label_col=label_col, n_neighbors=n_neighbors)

    # Per-cell derived metrics
    confidence = soft_probs.max(axis=1)          # max posterior per cell
    # Margin: P_top1 − P_top2
    sorted_probs = np.sort(soft_probs.values, axis=1)
    margin = pd.Series(sorted_probs[:, -1] - sorted_probs[:, -2], index=adata.obs_names)
    entropy = -(soft_probs * np.log2(soft_probs + 1e-9)).sum(axis=1)

    labels = adata.obs[label_col].values
    unique_labels = soft_probs.columns.tolist()

    # Aggregate per cluster
    stats = []
    for lbl in unique_labels:
        mask = labels == lbl
        stats.append({
            "label": lbl,
            "n_cells": int(mask.sum()),
            "mean_confidence": float(confidence[mask].mean()),
            "std_confidence": float(confidence[mask].std()),
            "mean_margin": float(margin[mask].mean()),
            "std_margin": float(margin[mask].std()),
            "mean_entropy": float(entropy[mask].mean()),
        })

    stats_df = pd.DataFrame(stats)

    # Sort
    sort_col = sort_by if sort_by in stats_df.columns else "mean_confidence"
    stats_df = stats_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # Subclass color map for bar colors
    _subclass_colors = {"sst": "#E41A1C", "pvalb": "#377EB8", "vip": "#984EA3", "lamp5": "#4DAF4A"}

    def _label_color(lbl):
        lbl_lower = str(lbl).lower()
        for token, col in _subclass_colors.items():
            if lbl_lower.startswith(token):
                return col
        return "#AAAAAA"

    colors = [_label_color(lbl) for lbl in stats_df["label"]]
    x = np.arange(len(stats_df))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Left: mean confidence ± SD ---
    ax1 = axes[0]
    ax1.bar(x, stats_df["mean_confidence"], color=colors, yerr=stats_df["std_confidence"],
            capsize=3, error_kw={"elinewidth": 1, "alpha": 0.6})
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df["label"], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Mean confidence (max posterior)")
    ax1.set_xlabel(label_col)
    ax1.set_ylim([0, 1.05])
    ax1.axhline(0.8, color="red", linestyle="--", linewidth=1.2, alpha=0.5, label="0.8")
    ax1.axhline(0.6, color="orange", linestyle="--", linewidth=1.2, alpha=0.5, label="0.6")
    ax1.legend(title="Threshold", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("Confidence\n(how strongly do neighbors agree?)")
    color_axis_ticklabels(ax1, axis="x", token_color_map=_subclass_colors,
                          default_color="#4d4d4d", match_mode="leading_token")

    # --- Right: mean margin ± SD ---
    ax2 = axes[1]
    ax2.bar(x, stats_df["mean_margin"], color=colors, yerr=stats_df["std_margin"],
            capsize=3, error_kw={"elinewidth": 1, "alpha": 0.6})
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats_df["label"], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Mean margin (P₁ − P₂)")
    ax2.set_xlabel(label_col)
    ax2.set_ylim([0, 1.05])
    ax2.axhline(0.4, color="red", linestyle="--", linewidth=1.2, alpha=0.5, label="0.4")
    ax2.axhline(0.2, color="orange", linestyle="--", linewidth=1.2, alpha=0.5, label="0.2")
    ax2.legend(title="Threshold", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("Margin\n(how much better is top-1 vs top-2?)")
    color_axis_ticklabels(ax2, axis="x", token_color_map=_subclass_colors,
                          default_color="#4d4d4d", match_mode="leading_token")

    fig.suptitle(
        f"{title}\n(n_neighbors={n_neighbors}, n={len(adata)} cells, label='{label_col}')",
        fontsize=13, fontweight="bold", y=1.03,
    )
    plt.tight_layout()

    # Attach per-cell series to soft_probs for downstream use
    soft_probs["_confidence"] = confidence.values
    soft_probs["_margin"] = margin.values
    soft_probs["_entropy"] = entropy.values

    return fig, soft_probs


def top_discriminable_genes_per_cluster(
    adata: ad.AnnData,
    cluster_col: str = "leiden",
    top_n: int = 3,
    exclude_genes: tuple[str, ...] | list[str] = ("Gad2", "Sst"),
    use_abs_effect_size: bool = False,
    include_depleted: bool = False,
    bootstrap_iterations: int = 0,
    random_state: int = 0,
) -> "pd.DataFrame":
    """
    Rank most discriminable genes per cluster using one-vs-rest separation.

    For each cluster and each gene:
      score = (mean_in_cluster - mean_out_cluster) / std_all_cells

    Higher positive score means the gene is enriched in that cluster and better
    separates it from the rest. By default, canonical exclusion genes (Gad2, Sst)
    are removed before ranking.

    Parameters
    ----------
    adata : AnnData
        Cells x genes matrix.
    cluster_col : str
        Column in adata.obs with cluster labels (default: "leiden").
    top_n : int
        Number of top genes to report per cluster (default: 3).
    exclude_genes : tuple[str, ...] | list[str]
        Genes to exclude from ranking (case-insensitive exact match).
    use_abs_effect_size : bool
        If True, rank by absolute effect size; if False (default), rank by
        positive enrichment effect size.
    include_depleted : bool
        If True, also return most depleted genes per cluster (most negative
        effect sizes) with direction='depleted'.
    bootstrap_iterations : int
        Number of bootstrap resamples (cells sampled with replacement) used to
        estimate marker stability. Set 0 to disable (default).
    random_state : int
        Random seed for bootstrap reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-form table with columns:
                    cluster, rank, gene, effect_size, mean_in, mean_out, std_all,
                    direction, stability_pct

    Notes
    -----
    - This is descriptive separability, not a p-value test.
    - Works best on normalized/log-transformed expression (e.g., log CPM).
    """
    import pandas as pd

    if cluster_col not in adata.obs.columns:
        raise ValueError(f"cluster_col '{cluster_col}' not found in adata.obs")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be >= 0")

    # Build dense matrix safely if needed
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X = np.asarray(X, dtype=np.float64)

    gene_names = np.asarray(adata.var_names.astype(str))
    cluster_labels = adata.obs[cluster_col].astype(str).values
    unique_clusters = pd.Index(cluster_labels).unique().tolist()

    # Case-insensitive exclusion mask
    exclude_set = {g.lower() for g in exclude_genes}
    keep_gene_mask = np.array([g.lower() not in exclude_set for g in gene_names], dtype=bool)
    if not np.any(keep_gene_mask):
        raise ValueError("All genes were excluded; adjust exclude_genes")

    X_keep = X[:, keep_gene_mask]
    genes_keep = gene_names[keep_gene_mask]

    std_all = X_keep.std(axis=0)
    std_all_safe = np.where(std_all <= 1e-12, 1e-12, std_all)

    def _compute_top_rows(
        X_local: np.ndarray,
        labels_local: np.ndarray,
    ) -> list[dict]:
        """Compute top enriched/depleted rows for a given matrix/labels."""
        local_rows = []
        local_clusters = pd.Index(labels_local).unique().tolist()

        for cl in local_clusters:
            in_mask = labels_local == cl
            out_mask = ~in_mask

            if in_mask.sum() == 0 or out_mask.sum() == 0:
                continue

            mean_in = X_local[in_mask].mean(axis=0)
            mean_out = X_local[out_mask].mean(axis=0)
            effect = (mean_in - mean_out) / std_all_safe

            ranking_score = np.abs(effect) if use_abs_effect_size else effect
            top_idx = np.argsort(ranking_score)[::-1][:top_n]

            for r, gi in enumerate(top_idx, start=1):
                local_rows.append(
                    {
                        "cluster": cl,
                        "rank": r,
                        "gene": genes_keep[gi],
                        "effect_size": float(effect[gi]),
                        "mean_in": float(mean_in[gi]),
                        "mean_out": float(mean_out[gi]),
                        "std_all": float(std_all[gi]),
                        "n_in_cluster": int(in_mask.sum()),
                        "n_out_cluster": int(out_mask.sum()),
                        "direction": "enriched",
                    }
                )

            if include_depleted:
                bottom_idx = np.argsort(effect)[:top_n]
                for r, gi in enumerate(bottom_idx, start=1):
                    local_rows.append(
                        {
                            "cluster": cl,
                            "rank": r,
                            "gene": genes_keep[gi],
                            "effect_size": float(effect[gi]),
                            "mean_in": float(mean_in[gi]),
                            "mean_out": float(mean_out[gi]),
                            "std_all": float(std_all[gi]),
                            "n_in_cluster": int(in_mask.sum()),
                            "n_out_cluster": int(out_mask.sum()),
                            "direction": "depleted",
                        }
                    )

        return local_rows

    rows = _compute_top_rows(X_keep, cluster_labels)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["stability_pct"] = np.nan

    # Bootstrap stability: frequency that each selected marker appears in top_n
    if bootstrap_iterations > 0:
        rng = np.random.default_rng(random_state)
        n_cells = X_keep.shape[0]

        # Count appearances by (cluster, direction, gene)
        stability_counter: dict[tuple[str, str, str], int] = {}

        for _ in range(bootstrap_iterations):
            boot_idx = rng.choice(n_cells, size=n_cells, replace=True)
            X_boot = X_keep[boot_idx]
            labels_boot = cluster_labels[boot_idx]

            boot_rows = _compute_top_rows(X_boot, labels_boot)
            if not boot_rows:
                continue

            # Count each selected gene once per cluster/direction per iteration
            seen = set()
            for row in boot_rows:
                key = (str(row["cluster"]), str(row["direction"]), str(row["gene"]))
                if key not in seen:
                    stability_counter[key] = stability_counter.get(key, 0) + 1
                    seen.add(key)

        def _stability_lookup(row: "pd.Series") -> float:
            key = (str(row["cluster"]), str(row["direction"]), str(row["gene"]))
            return 100.0 * stability_counter.get(key, 0) / float(bootstrap_iterations)

        result["stability_pct"] = result.apply(_stability_lookup, axis=1)

    return result.sort_values(["cluster", "direction", "rank"]).reset_index(drop=True)


def top_discriminable_gene_pairs_per_cluster(
    adata: ad.AnnData,
    cluster_col: str = "leiden",
    top_n: int = 3,
    exclude_genes: tuple[str, ...] | list[str] = ("Gad2", "Sst"),
    min_nonzero_fraction: float = 0.05,
) -> "pd.DataFrame":
    """
    Find the gene pairs that best discriminate each cluster (one-vs-rest),
    allowing each gene to contribute positively (+) or negatively (-).

    For every ordered pair (gene_A, gene_B) and each sign combination
    (s_A, s_B) ∈ {+1, -1}², the combined signal per cell is:

        signal = s_A * expr_A + s_B * expr_B

    The one-vs-rest effect size is then:

        effect = (mean_in - mean_out) / std_all

    where std_all is the std of the combined signal across all cells.
    Canonical form: we always use sign_A = +1 and allow sign_B ∈ {+1, -1},
    which avoids double-counting (flipping both signs inverts the effect size
    symmetrically and the best positive pair is the mirror of the best
    negative pair).

    This answers questions like:
    "Which clusters are uniquely identified by Calb2-high AND Crh-low cells?"

    Parameters
    ----------
    adata : AnnData
    cluster_col : str
        Column in adata.obs with cluster labels (default: "leiden").
    top_n : int
        Number of top gene pairs per cluster to return (default: 3).
    exclude_genes : tuple | list
        Genes to exclude from ranking (case-insensitive exact match).
    min_nonzero_fraction : float
        Minimum fraction of cells that must have nonzero expression for a gene
        to be eligible for pairing (default: 0.05).  Genes that are near-zero
        in almost all cells can produce spuriously high effect sizes (especially
        as the "−" component of a pair) and are suppressed by this filter.

    Returns
    -------
    pd.DataFrame
        Long-form table with columns:
          cluster, rank, gene_A, sign_A, gene_B, sign_B,
          pair_label, effect_size, mean_in, mean_out, std_all,
          n_in_cluster, n_out_cluster

        pair_label is a human-readable string, e.g. "Calb2+ / Crh-".
    """
    import pandas as pd
    from itertools import combinations

    if cluster_col not in adata.obs.columns:
        raise ValueError(f"cluster_col '{cluster_col}' not found in adata.obs")

    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X = np.asarray(X, dtype=np.float64)

    gene_names = np.asarray(adata.var_names.astype(str))
    cluster_labels = adata.obs[cluster_col].astype(str).values
    unique_clusters = pd.Index(cluster_labels).unique().tolist()

    exclude_set = {g.lower() for g in exclude_genes}
    keep_mask = np.array([g.lower() not in exclude_set for g in gene_names], dtype=bool)
    if not np.any(keep_mask):
        raise ValueError("All genes were excluded; adjust exclude_genes")

    X_keep = X[:, keep_mask]
    genes_keep = gene_names[keep_mask]

    # Drop genes that are nonzero in fewer than min_nonzero_fraction of all cells
    if min_nonzero_fraction > 0:
        nonzero_frac = (X_keep > 0).mean(axis=0)
        expr_mask = nonzero_frac >= min_nonzero_fraction
        X_keep = X_keep[:, expr_mask]
        genes_keep = genes_keep[expr_mask]

    n_genes = X_keep.shape[1]
    gene_indices = list(range(n_genes))

    rows = []
    for cl in unique_clusters:
        in_mask = cluster_labels == cl
        out_mask = ~in_mask
        if in_mask.sum() == 0 or out_mask.sum() == 0:
            continue

        best_pairs: list[tuple[float, dict]] = []

        for i, j in combinations(gene_indices, 2):
            # sign_A is always +1; sign_B ∈ {+1, -1} to avoid double-counting
            for sign_B in (1, -1):
                signal = X_keep[:, i] + sign_B * X_keep[:, j]
                std_all = signal.std()
                if std_all <= 1e-12:
                    continue
                mean_in = signal[in_mask].mean()
                mean_out = signal[out_mask].mean()
                effect = (mean_in - mean_out) / std_all

                sign_A_str = "+"
                sign_B_str = "+" if sign_B == 1 else "-"
                pair_label = f"{genes_keep[i]}{sign_A_str} / {genes_keep[j]}{sign_B_str}"

                best_pairs.append((effect, {
                    "cluster": cl,
                    "gene_A": genes_keep[i],
                    "sign_A": "+",
                    "gene_B": genes_keep[j],
                    "sign_B": sign_B_str,
                    "pair_label": pair_label,
                    "effect_size": float(effect),
                    "mean_in": float(mean_in),
                    "mean_out": float(mean_out),
                    "std_all": float(std_all),
                    "n_in_cluster": int(in_mask.sum()),
                    "n_out_cluster": int(out_mask.sum()),
                }))

        # Sort by effect size descending, take top_n
        best_pairs.sort(key=lambda x: x[0], reverse=True)
        for rank, (_, row_dict) in enumerate(best_pairs[:top_n], start=1):
            row_dict["rank"] = rank
            rows.append(row_dict)

    if not rows:
        return pd.DataFrame()

    col_order = [
        "cluster", "rank", "pair_label",
        "gene_A", "sign_A", "gene_B", "sign_B",
        "effect_size", "mean_in", "mean_out", "std_all",
        "n_in_cluster", "n_out_cluster",
    ]
    return (
        pd.DataFrame(rows)[col_order]
        .sort_values(["cluster", "rank"])
        .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Confidence comparison: All cells vs Co-registered cells
# ---------------------------------------------------------------------------

def load_coreg_cell_ids(
    mouse_id: "str | int",
    data_dir: "Path | str",
    catalog_path: "Path | str | None" = None,
) -> set:
    """
    Load co-registered cell IDs for a mouse as a set of full cell IDs
    formatted as ``"{mouse_id}_{hcr_id}"``, matching the ``cell_id`` column
    produced by the HCR-Tasic matching pipeline.

    Parameters
    ----------
    mouse_id : str or int
    data_dir : path to the capsule data directory
    catalog_path : path to the mouse JSON catalog file.
        Defaults to ``/src/ophys-mfish-dataset-catalog/mice/{mouse_id}.json``.

    Returns
    -------
    set[str]
    """
    import json
    import pandas as pd

    mouse_id = str(mouse_id)
    data_dir = Path(data_dir)

    if catalog_path is None:
        catalog_path = Path(
            f"/src/ophys-mfish-dataset-catalog/mice/{mouse_id}.json"
        )

    with open(catalog_path) as fh:
        catalog = json.load(fh)

    coreg_asset = catalog.get("derived_assets", {}).get("czstack_hcr_coreg")
    if coreg_asset is None:
        raise ValueError(f"No czstack_hcr_coreg asset found for mouse {mouse_id}")

    coreg_csv = data_dir / coreg_asset / f"{mouse_id}_coreg_table.csv"
    if not coreg_csv.exists():
        raise FileNotFoundError(f"Coreg table not found: {coreg_csv}")

    coreg_df = pd.read_csv(coreg_csv)
    return {f"{mouse_id}_{hid}" for hid in coreg_df["hcr_id"].astype(str)}


def plot_confidence_comparison(
    assignments: "pd.DataFrame",
    mouse_ids: "list[str]",
    coreg_ids_by_mouse: dict,
    branches: "list[str] | None" = None,
    branch_colors: "dict[str, str] | None" = None,
    metric: str = "confidence",
    metric_label: str = "Matching confidence (Pearson r)",
    figsize: "tuple[float, float] | None" = None,
    ylim: "tuple[float, float] | None" = (0.0, 1.0),
) -> "plt.Figure":
    """
    Grouped box plots comparing matching confidence (Pearson r to best
    centroid) for *all* assigned cells versus *co-registered* cells only,
    split by subclass, for one or more mice.

    Each mouse gets its own panel.  Within each panel the x-axis shows the
    subclass (branch) in the requested order, and the two hue groups are
    ``"All cells"`` and ``"Co-reg cells"``.

    Parameters
    ----------
    assignments : pd.DataFrame
        Assignment table with columns ``cell_id``, ``branch``, ``confidence``,
        ``mouse_id`` (as produced by Stage 5 of the HCR-Tasic pipeline).
    mouse_ids : list[str]
        One or more mouse IDs to plot (each gets its own panel).
    coreg_ids_by_mouse : dict[str, set[str]]
        Mapping from mouse_id (str) -> set of full cell IDs (e.g. "788406_1234")
        that are co-registered.  Build with load_coreg_cell_ids().
    branches : list[str], optional
        Ordered subclass names for the x-axis.
        Defaults to ["Vip", "Lamp5", "Sst", "Pvalb"].
    branch_colors : dict[str, str], optional
        Colour for each branch x-tick label.
    metric : str
        Column in assignments to plot on the y-axis.  Default "confidence".
    metric_label : str
        Y-axis label.
    figsize : tuple[float, float], optional
        Overall figure size.  Defaults to (5.5 * n_mice, 5).
    ylim : tuple[float, float], optional
        Y-axis limits.  Pass None to auto-scale.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import pandas as pd

    if branches is None:
        branches = ["Vip", "Lamp5", "Sst", "Pvalb"]

    mouse_ids = [str(m) for m in mouse_ids]
    n_mice = len(mouse_ids)

    if figsize is None:
        figsize = (5.5 * n_mice, 5)

    palette = {"All cells": "#b0b0b0", "Co-reg cells": "#2196F3"}

    fig, axes = plt.subplots(1, n_mice, figsize=figsize, sharey=True)
    if n_mice == 1:
        axes = [axes]

    for ax, mouse_id in zip(axes, mouse_ids):
        sub = assignments[assignments["mouse_id"].astype(str) == mouse_id].copy()
        if sub.empty:
            ax.set_title(f"Mouse {mouse_id}\n(no data)")
            continue

        coreg_ids = {str(c) for c in coreg_ids_by_mouse.get(mouse_id, [])}

        # Build two slices: all cells and co-reg subset
        all_df = sub[["branch", metric]].rename(columns={metric: "value"}).copy()
        all_df["group"] = "All cells"

        coreg_mask = sub["cell_id"].astype(str).isin(coreg_ids)
        coreg_df = (
            sub.loc[coreg_mask, ["branch", metric]]
            .rename(columns={metric: "value"})
            .copy()
        )
        coreg_df["group"] = "Co-reg cells"

        plot_df = pd.concat([all_df, coreg_df], ignore_index=True)
        plot_df["branch"] = pd.Categorical(
            plot_df["branch"], categories=branches, ordered=True
        )
        plot_df = plot_df.dropna(subset=["branch"])

        sns.boxplot(
            data=plot_df,
            x="branch",
            y="value",
            hue="group",
            order=branches,
            hue_order=["All cells", "Co-reg cells"],
            palette=palette,
            ax=ax,
            linewidth=0.8,
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
            width=0.6,
        )

        n_all = len(sub)
        n_coreg = int(coreg_mask.sum())
        ax.set_title(f"Mouse {mouse_id}\n({n_coreg:,} co-reg / {n_all:,} total)")
        ax.set_xlabel("Subclass")
        ax.set_ylabel(metric_label if ax is axes[0] else "")

        if ylim is not None:
            ax.set_ylim(ylim)

        # Colour x-tick labels by branch
        if branch_colors:
            for tick in ax.get_xticklabels():
                tick.set_color(branch_colors.get(tick.get_text(), "black"))

        ax.legend(title="Cell set", fontsize=9, title_fontsize=9)

    fig.suptitle(
        "Matching confidence: All cells vs Co-registered cells", fontsize=12
    )
    plt.tight_layout()
    return fig
