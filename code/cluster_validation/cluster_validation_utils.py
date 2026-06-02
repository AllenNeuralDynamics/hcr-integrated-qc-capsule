
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
            sc.tl.leiden(adata_filtered, resolution=resolution, key_added="leiden_test")

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
    sc.tl.leiden(adata_filtered, resolution=best_resolution)

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