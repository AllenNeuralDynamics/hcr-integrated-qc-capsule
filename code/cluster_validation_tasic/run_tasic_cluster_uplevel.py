"""
run_hcr_tasic_matching.py

HCR-Tasic matching protocol pipeline:
  Stage 1: Load & normalize both platforms into gene-wise z-scored space.
  Stage 2: Diagnose and correct cross-mouse batch effects within HCR data.
  Stage 3: Approach A — collapse Tasic taxonomy to panel resolution.
  Stage 4: Approach C — supervised hierarchical clustering (soft subclass gating
           + within-branch Leiden on Tasic reference).
  Stage 5: Matching — centroid correlation label transfer (both A and C),
           per-cell confidence, marker-score cross-check.

Outputs saved to: /root/capsule/results/hcr_tasic_matching/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score

from aind_hcr_data_loader import get_hcr_dataset_pairwise
import aind_hcr_qc.viz as viz
import cluster_validation_utils


# =============================================================================
# Constants
# =============================================================================

DATA_DIR = Path("/root/capsule/data")
SS_PATH = Path("/root/capsule/scratch/mouse_VISp_gene_expression_matrices_2018-06-14")
OUT_ROOT = Path("/root/capsule/results/hcr_tasic_matching")
TENX_HMB_DIR = Path("/root/capsule/scratch/reference_atlas_cellxgene/10x-hmb")

# Mice with confirmed pairwise-unmixed inhibitory cell data
MOUSE_IDS = []#["790322", "788406"] # "782149"

# Genes to exclude from the shared panel (non-biological / control)
EXCLUDE_GENES = {"GFP"}

# Canonical subclass markers — excluded from batch centering to avoid
# distortion when mice have different subclass compositions.
SUBCLASS_MARKERS = {"Pvalb", "Sst", "Vip", "Lamp5"}

# Minor subclasses to optionally drop from Tasic reference
# (too few cells to reliably match; can confuse gating/matching)
MINOR_SUBCLASSES = {"Serpinf1", "CR", "Meis2"}


# =============================================================================
# Stage 1 — Data Loading & Normalization
# =============================================================================


def load_hcr_multi_mouse(
    mouse_ids: list[str],
    data_dir: Path = DATA_DIR,
) -> ad.AnnData:
    """
    Load HCR inhibitory cell-by-gene data for multiple mice and concatenate.

    Each cell is tagged with a `mouse_id` column in .obs.
    Returns raw spot counts (not normalized).
    """
    adatas = []
    for mouse_id in mouse_ids:
        print(f"  Loading HCR mouse {mouse_id}...")
        _, pw_ds, _ = get_hcr_dataset_pairwise(
            mouse_id=mouse_id,
            data_dir=data_dir,
            load_spots=False,
            return_removed=False,
            coreg_cells_only=False,
        )
        adata = pw_ds.load_inhibitory_cells(unmixed=True, all_spots=False, as_anndata=True)
        adata.obs["mouse_id"] = mouse_id
        # Ensure unique cell IDs across mice
        adata.obs_names = [f"{mouse_id}_{cid}" for cid in adata.obs_names]
        adatas.append(adata)
        print(f"    → {adata.n_obs} cells, {adata.n_vars} genes")

    combined = ad.concat(adatas, join="inner")
    print(f"  Combined HCR: {combined.n_obs} cells, {combined.n_vars} genes")
    return combined


def load_tasic_inhibitory(
    ss_path: Path = SS_PATH,
    genes: list[str] | None = None,
    layer: str = "exon",
) -> ad.AnnData:
    """
    Load Tasic 2018 Smart-seq VISp data, filtered to inhibitory neurons.

    Returns raw counts (not normalized).
    """
    print(f"  Loading Smart-seq reference (layer={layer})...")
    smartseq = cluster_validation_utils.load_visp_expression(ss_path, genes=genes, layer=layer)
    filtered = cluster_validation_utils.make_filtered_views_for_smartseq(smartseq)
    adata_inh = filtered["inhibitory"]
    print(f"  Tasic inhibitory: {adata_inh.n_obs} cells, {adata_inh.n_vars} genes")
    return adata_inh


def normalize_tasic(adata: ad.AnnData) -> ad.AnnData:
    """
    Normalize Smart-seq counts: CPM → log1p (i.e. log_cp10k with target=1e4).

    Protocol decision: normalize_total to 10k, then log1p.
    Stores raw counts in .layers["raw"].
    """
    adata = adata.copy()
    adata.layers["raw"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def normalize_hcr(adata: ad.AnnData) -> ad.AnnData:
    """
    Normalize HCR spot counts: log1p only (no library-size normalization).

    Protocol decision: log1p on per-cell spot counts directly.
    Stores raw counts in .layers["raw"].
    """
    adata = adata.copy()
    adata.layers["raw"] = adata.X.copy()
    sc.pp.log1p(adata)
    return adata


def load_10x_hmb_query(
    tenx_dir: Path = TENX_HMB_DIR,
    label_column: str = "supertype",
) -> ad.AnnData:
    """
    Load 10x-HMB cell-by-gene matrix and labels from pre-collected CSVs.

    Expected files in tenx_dir:
      - cell_x_gene.csv
      - labels_supertype.csv
    """
    matrix_path = tenx_dir / "cell_x_gene.csv"
    labels_path = tenx_dir / "labels_supertype.csv"

    if not matrix_path.exists():
        raise FileNotFoundError(f"10x matrix file not found: {matrix_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"10x labels file not found: {labels_path}")

    cxg = pd.read_csv(matrix_path)
    if "cell_label" in cxg.columns:
        cxg = cxg.set_index("cell_label")
    else:
        cxg = cxg.set_index(cxg.columns[0])
    cxg.index = cxg.index.astype(str)

    labels_df = pd.read_csv(labels_path)
    if "cell_label" in labels_df.columns:
        labels_df = labels_df.set_index("cell_label")
    else:
        labels_df = labels_df.set_index(labels_df.columns[0])
    labels_df.index = labels_df.index.astype(str)

    if label_column not in labels_df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in {labels_path}. "
            f"Available columns: {labels_df.columns.tolist()}"
        )

    # Align labels to matrix rows and keep matrix row order.
    labels = labels_df[label_column].reindex(cxg.index)
    missing = labels.isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} cells missing '{label_column}' labels; filling with 'unlabeled'.")
        labels = labels.fillna("unlabeled")

    cxg = cxg.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    adata = ad.AnnData(
        X=cxg.values.astype(np.float32),
        obs=pd.DataFrame(index=cxg.index),
        var=pd.DataFrame(index=cxg.columns.astype(str)),
    )
    adata.obs[label_column] = labels.astype(str).values
    adata.obs["mouse_id"] = "10x-hmb"
    adata.obs["dataset"] = "10x-hmb"
    print(f"  10x-HMB loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata


def normalize_10x_hmb(
    adata: ad.AnnData,
    expression_scale: str = "log2",
) -> ad.AnnData:
    """
    Normalize 10x-HMB expression matrix before per-gene z-scoring.

    expression_scale:
      - "log2": matrix is already log-scaled; use as-is.
      - "raw": apply log1p.
    """
    adata = adata.copy()
    adata.layers["raw"] = adata.X.copy()
    if expression_scale == "raw":
        sc.pp.log1p(adata)
    elif expression_scale == "log2":
        pass
    else:
        raise ValueError("expression_scale must be one of: 'log2', 'raw'")
    return adata


def zscore_genes(adata: ad.AnnData) -> ad.AnnData:
    """
    Gene-wise z-score: for each gene, subtract mean and divide by std across cells.

    This is the cross-platform comparison currency.
    Stores the log-normalized values in .layers["log_norm"] before z-scoring.
    """
    adata = adata.copy()
    adata.layers["log_norm"] = adata.X.copy()

    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X = np.asarray(X, dtype=np.float64)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds < 1e-12] = 1.0  # avoid division by zero for constant genes
    X_z = (X - means) / stds
    adata.X = X_z.astype(np.float32)
    return adata


def intersect_genes(
    tasic: ad.AnnData,
    hcr: ad.AnnData,
    exclude: set[str] | None = None,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Subset both datasets to shared panel genes, excluding controls."""
    if exclude is None:
        exclude = set()
    shared = sorted(
        set(tasic.var_names) & set(hcr.var_names) - exclude
    )
    print(f"  Shared panel genes ({len(shared)}): {shared}")
    return tasic[:, shared].copy(), hcr[:, shared].copy()


def filter_tasic_reference(
    adata: ad.AnnData,
    drop_minor_subclasses: bool = False,
    min_cells_per_cluster: int = 0,
) -> ad.AnnData:
    """
    Filter Tasic reference data:
    - Optionally drop minor subclasses (Serpinf1, CR, Meis2).
    - Optionally drop clusters with fewer than min_cells_per_cluster cells.
    """
    n_before = adata.n_obs

    if drop_minor_subclasses:
        keep_mask = ~adata.obs["subclass"].isin(MINOR_SUBCLASSES)
        n_dropped = (~keep_mask).sum()
        dropped_subs = sorted(adata.obs.loc[~keep_mask, "subclass"].unique())
        adata = adata[keep_mask].copy()
        print(f"  Dropped minor subclasses {dropped_subs}: {n_dropped} cells removed")

    if min_cells_per_cluster > 0:
        cluster_counts = adata.obs["cluster"].value_counts()
        keep_clusters = cluster_counts[cluster_counts >= min_cells_per_cluster].index
        n_clusters_before = adata.obs["cluster"].nunique()
        adata = adata[adata.obs["cluster"].isin(keep_clusters)].copy()
        n_clusters_after = adata.obs["cluster"].nunique()
        n_dropped = n_clusters_before - n_clusters_after
        print(f"  Dropped {n_dropped} clusters with < {min_cells_per_cluster} cells "
              f"({n_clusters_before} → {n_clusters_after} clusters, "
              f"{n_before - adata.n_obs} cells removed)")

    print(f"  Tasic after filtering: {adata.n_obs} cells, "
          f"{adata.obs['cluster'].nunique()} clusters")
    return adata


def run_stage1(
    mouse_ids: list[str] = MOUSE_IDS,
    tasic_layer: str = "exon",
    drop_minor_subclasses: bool = False,
    min_cells_per_cluster: int = 0,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData, ad.AnnData]:
    """
    Execute full Stage 1 pipeline.

    Parameters
    ----------
    drop_minor_subclasses : bool
        If True, remove Serpinf1/CR/Meis2 cells from Tasic reference.
    min_cells_per_cluster : int
        Drop Tasic clusters with fewer than this many cells (0 = keep all).

    Returns
    -------
    tasic_z : z-scored Tasic (shared genes)
    hcr_z : z-scored HCR (shared genes)
    tasic_log : log-normalized Tasic (shared genes, before z-score)
    hcr_log : log-normalized HCR (shared genes, before z-score)
    """
    print("\n" + "=" * 60)
    print("STAGE 1: Data Loading & Normalization")
    print("=" * 60)

    # 1.2 Load HCR multi-mouse
    print("\n[1.2] Loading HCR data for multiple mice...")
    hcr_raw = load_hcr_multi_mouse(mouse_ids)

    # 1.1 Load Tasic — use HCR gene names to subset
    hcr_genes = [g for g in hcr_raw.var_names if g not in EXCLUDE_GENES]
    print(f"\n[1.1] Loading Tasic reference (genes from HCR panel)...")
    tasic_raw = load_tasic_inhibitory(genes=hcr_genes, layer=tasic_layer)

    # 1.1b Filter Tasic reference
    if drop_minor_subclasses or min_cells_per_cluster > 0:
        print(f"\n[1.1b] Filtering Tasic reference...")
        tasic_raw = filter_tasic_reference(
            tasic_raw,
            drop_minor_subclasses=drop_minor_subclasses,
            min_cells_per_cluster=min_cells_per_cluster,
        )

    # 1.6 Intersect to shared genes
    print("\n[1.6] Intersecting to shared panel genes...")
    tasic_raw, hcr_raw = intersect_genes(tasic_raw, hcr_raw, exclude=EXCLUDE_GENES)

    # 1.3 Normalize Tasic
    print("\n[1.3] Normalizing Tasic (log_cp10k)...")
    tasic_log = normalize_tasic(tasic_raw)

    # 1.4 Normalize HCR
    print("\n[1.4] Normalizing HCR (log1p)...")
    hcr_log = normalize_hcr(hcr_raw)

    # 1.5 Gene-wise z-score
    print("\n[1.5] Gene-wise z-scoring (per platform)...")
    tasic_z = zscore_genes(tasic_log)
    hcr_z = zscore_genes(hcr_log)

    print(f"\n  Final Tasic z-scored: {tasic_z.shape}")
    print(f"  Final HCR z-scored:   {hcr_z.shape}")

    return tasic_z, hcr_z, tasic_log, hcr_log


# =============================================================================
# Stage 2 — Cross-Mouse Batch Correction (within HCR)
# =============================================================================


def diagnose_batch(hcr_z: ad.AnnData, out_dir: Path) -> None:
    """
    Embed HCR cells and color by mouse to diagnose batch effects.
    Saves UMAP plot before correction.
    """
    print("\n[2.1] Diagnosing batch effects (pre-correction UMAP)...")
    adata = hcr_z.copy()
    sc.pp.pca(adata, n_comps=min(15, adata.n_vars - 1))
    sc.pp.neighbors(adata, n_neighbors=30)
    sc.tl.umap(adata)

    # Plot colored by mouse
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sc.pl.umap(adata, color="mouse_id", ax=axes[0], show=False, title="Pre-correction: colored by mouse")

    # Also show marker-based subclass signal
    # Compute dominant subclass marker for coloring
    marker_genes = [g for g in ["Pvalb", "Sst", "Vip", "Lamp5"] if g in adata.var_names]
    if marker_genes:
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        marker_idx = [list(adata.var_names).index(g) for g in marker_genes]
        marker_vals = X[:, marker_idx]
        dominant_marker = np.array(marker_genes)[marker_vals.argmax(axis=1)]
        adata.obs["dominant_marker"] = pd.Categorical(dominant_marker, categories=marker_genes)
        sc.pl.umap(adata, color="dominant_marker", ax=axes[1], show=False,
                   title="Pre-correction: dominant subclass marker")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "01_pre_correction_umap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / '01_pre_correction_umap.png'}")


def correct_batch_centering(
    hcr_z: ad.AnnData,
    mode: str = "all",
) -> ad.AnnData:
    """
    Per-gene per-mouse centering: subtract each mouse's gene-wise mean.

    Parameters
    ----------
    hcr_z : AnnData
        Z-scored HCR data with 'mouse_id' in .obs.
    mode : str
        Batch correction mode:
        - "all" (default): center ALL genes. Simple and appropriate when
          compositional differences across mice reflect real biology (e.g.
          tissue from different cortical layers).
        - "exclude_markers": exclude canonical subclass markers (Pvalb, Sst,
          Vip, Lamp5) from centering. Use when you suspect compositional
          differences are artifactual and centering would distort marker signal.
        - "per_mouse": Z-score each mouse independently (no pooled z-score).
          Each mouse is its own self-contained experiment — sidesteps the
          multi-mouse alignment problem entirely. Correlation-based matching
          against Tasic centroids works identically since each mouse's relative
          gene structure is preserved.
        - "none": skip correction entirely (pass-through).
    """
    print(f"\n[2.2] Batch correction mode: '{mode}'")

    if mode == "none":
        print("       Skipping batch correction.")
        return hcr_z.copy()

    if mode == "per_mouse":
        print("       Per-mouse independent z-scoring (no cross-mouse alignment).")
        adata = hcr_z.copy()
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        X = np.asarray(X, dtype=np.float64)

        # Re-do z-scoring within each mouse independently
        mouse_ids = adata.obs["mouse_id"].values
        for mouse in np.unique(mouse_ids):
            mask = mouse_ids == mouse
            X_mouse = X[mask]
            means = X_mouse.mean(axis=0)
            stds = X_mouse.std(axis=0)
            stds[stds < 1e-12] = 1.0
            X[mask] = (X_mouse - means) / stds
            print(f"    Mouse {mouse}: z-scored {mask.sum()} cells independently")

        adata.X = X.astype(np.float32)
        return adata

    adata = hcr_z.copy()
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    X = np.asarray(X, dtype=np.float64)
    gene_names = list(adata.var_names)

    if mode == "exclude_markers":
        exclude = SUBCLASS_MARKERS
        center_mask = np.array([g not in exclude for g in gene_names])
        print(f"       Excluding from centering: {sorted(exclude)}")
    elif mode == "all":
        center_mask = np.ones(len(gene_names), dtype=bool)
    else:
        raise ValueError(f"Unknown batch correction mode: {mode!r}. "
                         f"Use 'all', 'exclude_markers', 'per_mouse', or 'none'.")

    n_centered = int(center_mask.sum())
    print(f"       Centering {n_centered}/{len(gene_names)} genes per mouse")

    mouse_ids = adata.obs["mouse_id"].values
    for mouse in np.unique(mouse_ids):
        mask = mouse_ids == mouse
        mouse_mean = X[mask][:, center_mask].mean(axis=0)
        X[np.ix_(mask, center_mask)] -= mouse_mean
        print(f"    Mouse {mouse}: centered {mask.sum()} cells "
              f"(max offset={np.abs(mouse_mean).max():.3f}, "
              f"mean offset={np.abs(mouse_mean).mean():.3f})")

    adata.X = X.astype(np.float32)
    return adata


def post_correction_qc(hcr_corrected: ad.AnnData, out_dir: Path) -> None:
    """
    Post-correction QC: UMAP showing mouse mixing + subclass separation.
    """
    print("\n[2.3] Post-correction QC...")
    adata = hcr_corrected.copy()
    sc.pp.pca(adata, n_comps=min(15, adata.n_vars - 1))
    sc.pp.neighbors(adata, n_neighbors=30)
    sc.tl.umap(adata)

    marker_genes = [g for g in ["Pvalb", "Sst", "Vip", "Lamp5"] if g in adata.var_names]

    # Build figure: mouse_id, dominant_marker, + individual markers
    n_panels = 2 + len(marker_genes)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))

    sc.pl.umap(adata, color="mouse_id", ax=axes[0], show=False,
               title="Post-correction: by mouse")

    if marker_genes:
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        marker_idx = [list(adata.var_names).index(g) for g in marker_genes]
        marker_vals = X[:, marker_idx]
        dominant_marker = np.array(marker_genes)[marker_vals.argmax(axis=1)]
        adata.obs["dominant_marker"] = pd.Categorical(dominant_marker, categories=marker_genes)
        sc.pl.umap(adata, color="dominant_marker", ax=axes[1], show=False,
                   title="Post-correction: dominant marker")

        for i, gene in enumerate(marker_genes):
            sc.pl.umap(adata, color=gene, ax=axes[2 + i], show=False,
                       title=f"Post-correction: {gene}", color_map="magma")

    plt.tight_layout()
    fig.savefig(out_dir / "02_post_correction_umap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / '02_post_correction_umap.png'}")


def plot_gene_distributions_by_mouse(
    hcr_log: ad.AnnData,
    hcr_corrected: ad.AnnData,
    out_dir: Path,
) -> None:
    """
    Plot per-gene violin distributions by mouse (grid layout, vertical violins)
    plus a summary correlation heatmap showing per-gene cross-mouse agreement.
    """
    print("\n  Plotting per-gene distributions by mouse...")
    genes = list(hcr_log.var_names)
    n_genes = len(genes)
    ncols = int(np.ceil(np.sqrt(n_genes)))
    nrows = int(np.ceil(n_genes / ncols))

    X_log = hcr_log.X if not hasattr(hcr_log.X, "toarray") else hcr_log.X.toarray()
    X_corr = hcr_corrected.X if not hasattr(hcr_corrected.X, "toarray") else hcr_corrected.X.toarray()
    mice = np.unique(hcr_log.obs["mouse_id"].values)
    mouse_colors = {"790322": "#1b9e77", "782149": "#d95f02", "788406": "#7570b3"}
    mouse_order = [f"M{m}" for m in sorted(mice)]
    mouse_palette = {f"M{k}": v for k, v in mouse_colors.items()}

    # --- Pre-correction grid ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes_flat = np.array(axes).flatten()
    for i, gene in enumerate(genes):
        ax = axes_flat[i]
        gene_idx = list(hcr_log.var_names).index(gene)
        df = pd.DataFrame({
            "expression": X_log[:, gene_idx],
            "mouse_id": hcr_log.obs["mouse_id"].values,
        })
        df["mouse_label"] = "M" + df["mouse_id"].astype(str)
        sns.violinplot(data=df, x="mouse_label", y="expression", hue="mouse_label", ax=ax,
                       inner="quartile", density_norm="width", cut=0,
                       palette=mouse_palette, order=mouse_order, legend=False)
        ax.set_title(gene, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("Per-gene distributions by mouse (log-normalized, pre-correction)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "03_gene_distributions_pre.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Post-correction grid ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes_flat = np.array(axes).flatten()
    for i, gene in enumerate(genes):
        ax = axes_flat[i]
        gene_idx = list(hcr_corrected.var_names).index(gene)
        df = pd.DataFrame({
            "expression": X_corr[:, gene_idx],
            "mouse_id": hcr_corrected.obs["mouse_id"].values,
        })
        df["mouse_label"] = "M" + df["mouse_id"].astype(str)
        sns.violinplot(data=df, x="mouse_label", y="expression", hue="mouse_label", ax=ax,
                       inner="quartile", density_norm="width", cut=0,
                       palette=mouse_palette, order=mouse_order, legend=False)
        ax.set_title(gene, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("Per-gene distributions by mouse (z-scored, post-correction)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "04_gene_distributions_post.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Summary: per-gene cross-mouse Pearson correlation (pre & post) ---
    # For each pair of mice, compute per-gene mean correlation of cell
    # expression vectors — summarises batch alignment in one figure.
    from itertools import combinations
    mouse_pairs = list(combinations(sorted(mice), 2))

    def _per_gene_mouse_means(X, mouse_ids, gene_names):
        """Return DataFrame of per-mouse mean expression per gene."""
        df = pd.DataFrame(X, columns=gene_names)
        df["mouse_id"] = mouse_ids
        return df.groupby("mouse_id")[gene_names].mean()

    # Pre-correction means
    means_pre = _per_gene_mouse_means(X_log, hcr_log.obs["mouse_id"].values, genes)
    # Post-correction means
    means_post = _per_gene_mouse_means(X_corr, hcr_corrected.obs["mouse_id"].values, genes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pre-correction: pairwise scatter of gene means
    ax = axes[0]
    for m1, m2 in mouse_pairs:
        x = means_pre.loc[m1].values
        y = means_pre.loc[m2].values
        r = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=30, alpha=0.7, label=f"{m1}↔{m2} (r={r:.3f})")
        for k, g in enumerate(genes):
            ax.annotate(g, (x[k], y[k]), fontsize=6, alpha=0.6)
    ax.set_xlabel("Mouse A gene mean")
    ax.set_ylabel("Mouse B gene mean")
    ax.set_title("Pre-correction: per-gene mean agreement")
    ax.legend(fontsize=8)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_aspect("equal")

    # Post-correction: pairwise scatter of gene means
    ax = axes[1]
    for m1, m2 in mouse_pairs:
        x = means_post.loc[m1].values
        y = means_post.loc[m2].values
        r = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=30, alpha=0.7, label=f"{m1}↔{m2} (r={r:.3f})")
        for k, g in enumerate(genes):
            ax.annotate(g, (x[k], y[k]), fontsize=6, alpha=0.6)
    ax.set_xlabel("Mouse A gene mean")
    ax.set_ylabel("Mouse B gene mean")
    ax.set_title("Post-correction: per-gene mean agreement")
    ax.legend(fontsize=8)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_aspect("equal")

    plt.suptitle("Batch correction summary: cross-mouse gene-mean correlation",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "05_batch_correlation_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 03/04 (gene distributions) + 05 (correlation summary)")


def plot_platform_comparison(
    tasic_z: ad.AnnData,
    hcr_corrected: ad.AnnData,
    out_dir: Path,
) -> None:
    """
    Compare z-scored gene profiles between platforms at the subclass level.
    Shows that relative marker structure is preserved across platforms.
    """
    print("\n  Plotting cross-platform comparison (subclass centroids)...")
    genes = list(tasic_z.var_names)

    # Tasic centroids by subclass
    X_tasic = tasic_z.X if not hasattr(tasic_z.X, "toarray") else tasic_z.X.toarray()
    tasic_df = pd.DataFrame(X_tasic, columns=genes, index=tasic_z.obs_names)
    tasic_df["subclass"] = tasic_z.obs["subclass"].values
    tasic_centroids = tasic_df.groupby("subclass")[genes].mean()

    # HCR centroids by dominant marker (proxy for subclass)
    X_hcr = hcr_corrected.X if not hasattr(hcr_corrected.X, "toarray") else hcr_corrected.X.toarray()
    hcr_df = pd.DataFrame(X_hcr, columns=genes, index=hcr_corrected.obs_names)
    marker_genes = [g for g in ["Pvalb", "Sst", "Vip", "Lamp5"] if g in genes]
    if marker_genes:
        marker_vals = hcr_df[marker_genes].values
        hcr_df["subclass_proxy"] = np.array(marker_genes)[marker_vals.argmax(axis=1)]
        hcr_centroids = hcr_df.groupby("subclass_proxy")[genes].mean()
    else:
        print("  WARNING: No canonical markers found, skipping cross-platform plot.")
        return

    # Keep only subclasses present in both
    shared_subclasses = sorted(set(tasic_centroids.index) & set(hcr_centroids.index))
    if not shared_subclasses:
        print("  WARNING: No shared subclasses, skipping cross-platform plot.")
        return

    fig, axes = plt.subplots(1, len(shared_subclasses), figsize=(5 * len(shared_subclasses), 4.5))
    if len(shared_subclasses) == 1:
        axes = [axes]

    for i, sub in enumerate(shared_subclasses):
        ax = axes[i]
        t_vals = tasic_centroids.loc[sub, genes].values
        h_vals = hcr_centroids.loc[sub, genes].values
        ax.scatter(t_vals, h_vals, s=40, alpha=0.8)
        for j, g in enumerate(genes):
            ax.annotate(g, (t_vals[j], h_vals[j]), fontsize=7, alpha=0.7)

        # Correlation
        r = np.corrcoef(t_vals, h_vals)[0, 1]
        ax.set_title(f"{sub} (r={r:.3f})")
        ax.set_xlabel("Tasic z-scored centroid")
        ax.set_ylabel("HCR z-scored centroid")
        lims = [min(t_vals.min(), h_vals.min()) - 0.3, max(t_vals.max(), h_vals.max()) + 0.3]
        ax.plot(lims, lims, "k--", alpha=0.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")

    plt.suptitle("Cross-platform z-scored centroid comparison (Tasic vs HCR)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "01_cross_platform_centroids.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / '01_cross_platform_centroids.png'}")


def plot_normalization_summary(
    tasic_z: ad.AnnData,
    hcr_z: ad.AnnData,
    out_dir: Path,
) -> None:
    """
    Heatmap of mean z-scored expression by subclass (Tasic) and dominant marker (HCR).
    Side-by-side comparison showing the data is in comparable form.
    """
    print("\n  Plotting normalization summary heatmaps...")
    genes = list(tasic_z.var_names)

    # Tasic
    X_t = tasic_z.X if not hasattr(tasic_z.X, "toarray") else tasic_z.X.toarray()
    df_t = pd.DataFrame(X_t, columns=genes)
    df_t["subclass"] = tasic_z.obs["subclass"].values
    centroids_t = df_t.groupby("subclass")[genes].mean()
    # Keep main subclasses
    keep_sub = [s for s in ["Pvalb", "Sst", "Vip", "Lamp5", "Sncg", "Serpinf1", "CR"] if s in centroids_t.index]
    centroids_t = centroids_t.loc[keep_sub]

    # HCR
    X_h = hcr_z.X if not hasattr(hcr_z.X, "toarray") else hcr_z.X.toarray()
    df_h = pd.DataFrame(X_h, columns=genes)
    marker_genes = [g for g in ["Pvalb", "Sst", "Vip", "Lamp5"] if g in genes]
    marker_vals = df_h[marker_genes].values
    df_h["subclass_proxy"] = np.array(marker_genes)[marker_vals.argmax(axis=1)]
    centroids_h = df_h.groupby("subclass_proxy")[genes].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(keep_sub) * 0.5)))

    sns.heatmap(centroids_t, ax=axes[0], cmap="RdBu_r", center=0, square=True,
                cbar_kws={"label": "z-score", "shrink": 0.8})
    axes[0].set_title("Tasic subclass centroids (z-scored)")
    axes[0].tick_params(axis="x", rotation=90)

    sns.heatmap(centroids_h, ax=axes[1], cmap="RdBu_r", center=0, square=True,
                cbar_kws={"label": "z-score", "shrink": 0.8})
    axes[1].set_title("HCR dominant-marker centroids (z-scored)")
    axes[1].tick_params(axis="x", rotation=90)

    plt.suptitle("Data normalization: z-scored subclass centroids (both platforms)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "02_normalization_summary_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / '02_normalization_summary_heatmaps.png'}")


def plot_composition_by_mouse(hcr_z: ad.AnnData, out_dir: Path) -> None:
    """
    Plot subclass composition per mouse — key diagnostic for compositional bias.
    If one mouse has drastically different composition, naive centering will distort.
    """
    print("\n  Plotting subclass composition by mouse...")
    genes = list(hcr_z.var_names)
    marker_genes = [g for g in ["Pvalb", "Sst", "Vip", "Lamp5"] if g in genes]
    if not marker_genes:
        return

    X = hcr_z.X if not hasattr(hcr_z.X, "toarray") else hcr_z.X.toarray()
    marker_idx = [genes.index(g) for g in marker_genes]
    marker_vals = X[:, marker_idx]
    dominant = np.array(marker_genes)[marker_vals.argmax(axis=1)]

    df = pd.DataFrame({"mouse_id": hcr_z.obs["mouse_id"].values, "subclass": dominant})
    ct = pd.crosstab(df["mouse_id"], df["subclass"])
    ct_frac = ct.div(ct.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Absolute counts
    ct.plot.bar(ax=axes[0], stacked=True,
                color={"Pvalb": "#377EB8", "Sst": "#E41A1C", "Vip": "#984EA3", "Lamp5": "#4DAF4A"})
    axes[0].set_title("Subclass cell counts by mouse")
    axes[0].set_ylabel("Cell count")
    axes[0].legend(title="Subclass")
    axes[0].tick_params(axis="x", rotation=0)

    # Fractions
    ct_frac.plot.bar(ax=axes[1], stacked=True,
                     color={"Pvalb": "#377EB8", "Sst": "#E41A1C", "Vip": "#984EA3", "Lamp5": "#4DAF4A"})
    axes[1].set_title("Subclass fractions by mouse")
    axes[1].set_ylabel("Fraction")
    axes[1].set_ylim(0, 1)
    axes[1].legend(title="Subclass")
    axes[1].tick_params(axis="x", rotation=0)

    plt.suptitle("Compositional overlap across mice (batch correction diagnostic)", fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "00_composition_by_mouse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / '00_composition_by_mouse.png'}")

    # Log the composition table
    print("\n  Subclass composition by mouse (counts):")
    print(ct.to_string())
    print("\n  Subclass composition by mouse (fractions):")
    print(ct_frac.round(3).to_string())


def run_stage2(
    hcr_z: ad.AnnData,
    hcr_log: ad.AnnData,
    tasic_z: ad.AnnData,
    out_dir: Path,
    batch_mode: str = "all",
) -> ad.AnnData:
    """
    Execute full Stage 2 pipeline: diagnose → correct → QC.

    Parameters
    ----------
    batch_mode : str
        Passed to correct_batch_centering. Options:
        - "all" (default): center all genes per mouse.
        - "exclude_markers": skip Pvalb/Sst/Vip/Lamp5 during centering.
        - "none": no batch correction.

    Returns
    -------
    hcr_corrected : batch-corrected, z-scored HCR AnnData
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Cross-Mouse Batch Correction")
    print("=" * 60)

    # 2.0 Composition diagnostic (protocol §Cross-sample, point 2)
    plot_composition_by_mouse(hcr_z, out_dir)

    # 2.1 Diagnose
    diagnose_batch(hcr_z, out_dir)

    # 2.2 Correct
    hcr_corrected = correct_batch_centering(hcr_z, mode=batch_mode)

    # 2.3 Post-correction QC
    post_correction_qc(hcr_corrected, out_dir)

    # Additional diagnostic plots
    plot_gene_distributions_by_mouse(hcr_log, hcr_corrected, out_dir)
    plot_platform_comparison(tasic_z, hcr_corrected, out_dir)

    return hcr_corrected


# =============================================================================
# Stage 3 — Approach A: Collapse Tasic Labels to Panel Resolution
# =============================================================================


def compute_cluster_centroids(
    tasic_z: ad.AnnData,
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """
    Compute mean z-scored expression per cluster.

    Returns DataFrame: clusters × genes.
    """
    X = tasic_z.X if not hasattr(tasic_z.X, "toarray") else tasic_z.X.toarray()
    df = pd.DataFrame(X, columns=tasic_z.var_names, index=tasic_z.obs_names)
    df["_cluster"] = tasic_z.obs[cluster_col].values
    centroids = df.groupby("_cluster").mean()
    return centroids


def compute_pairwise_separability(
    tasic_z: ad.AnnData,
    cluster_col: str = "cluster",
    effect_size_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    For every pair of clusters within the same subclass, compute per-gene
    separability (effect size = |mean_A - mean_B| / pooled_std).

    A pair is 'separable' if at least one panel gene exceeds the threshold.

    Returns
    -------
    DataFrame with columns: subclass, cluster_a, cluster_b, best_gene,
                            best_effect_size, separable, n_genes_above_threshold
    """
    from itertools import combinations

    X = tasic_z.X if not hasattr(tasic_z.X, "toarray") else tasic_z.X.toarray()
    X = np.asarray(X, dtype=np.float64)
    genes = np.array(tasic_z.var_names)
    clusters = tasic_z.obs[cluster_col].values
    subclasses = tasic_z.obs["subclass"].values

    # Build per-cluster stats
    unique_clusters = np.unique(clusters)
    cluster_to_subclass = {}
    cluster_means = {}
    cluster_stds = {}
    cluster_counts = {}
    for cl in unique_clusters:
        mask = clusters == cl
        cluster_to_subclass[cl] = subclasses[mask][0]
        cluster_means[cl] = X[mask].mean(axis=0)
        cluster_stds[cl] = X[mask].std(axis=0)
        cluster_counts[cl] = int(mask.sum())

    # Group clusters by subclass
    from collections import defaultdict
    subclass_clusters = defaultdict(list)
    for cl, sub in cluster_to_subclass.items():
        subclass_clusters[sub].append(cl)

    rows = []
    for sub, sub_clusters in subclass_clusters.items():
        if len(sub_clusters) < 2:
            continue
        for cl_a, cl_b in combinations(sorted(sub_clusters), 2):
            mean_a = cluster_means[cl_a]
            mean_b = cluster_means[cl_b]
            # Pooled std
            n_a = cluster_counts[cl_a]
            n_b = cluster_counts[cl_b]
            std_a = cluster_stds[cl_a]
            std_b = cluster_stds[cl_b]
            pooled_std = np.sqrt(
                ((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2)
            )
            pooled_std[pooled_std < 1e-12] = 1e-12

            effect_sizes = np.abs(mean_a - mean_b) / pooled_std
            best_idx = np.argmax(effect_sizes)
            best_effect = float(effect_sizes[best_idx])
            best_gene = genes[best_idx]
            n_above = int((effect_sizes >= effect_size_threshold).sum())

            rows.append({
                "subclass": sub,
                "cluster_a": cl_a,
                "cluster_b": cl_b,
                "best_gene": best_gene,
                "best_effect_size": best_effect,
                "n_genes_above_threshold": n_above,
                "separable": best_effect >= effect_size_threshold,
                "n_cells_a": n_a,
                "n_cells_b": n_b,
            })

    return pd.DataFrame(rows).sort_values(
        ["subclass", "separable", "best_effect_size"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def collapse_inseparable_clusters(
    separability_df: pd.DataFrame,
    tasic_z: ad.AnnData,
    cluster_col: str = "cluster",
) -> tuple[dict[str, str], pd.DataFrame]:
    """
    Merge clusters that cannot be separated on the panel into collapsed groups.

    Uses a union-find approach: if A and B are inseparable, they merge.
    Transitive: if A-B inseparable and B-C inseparable, all three merge.

    Returns
    -------
    mapping : dict
        Original cluster → collapsed label
    summary : DataFrame
        One row per collapsed group with member clusters and group name
    """
    # Union-find
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # All clusters start as their own parent
    all_clusters = sorted(tasic_z.obs[cluster_col].unique())
    for cl in all_clusters:
        parent[cl] = cl

    # Merge inseparable pairs
    inseparable = separability_df[~separability_df["separable"]]
    for _, row in inseparable.iterrows():
        union(row["cluster_a"], row["cluster_b"])

    # Build groups
    from collections import defaultdict
    groups = defaultdict(list)
    for cl in all_clusters:
        groups[find(cl)].append(cl)

    # Create mapping and summary
    cluster_to_subclass = dict(
        zip(tasic_z.obs[cluster_col].values, tasic_z.obs["subclass"].values)
    )
    mapping = {}
    summary_rows = []

    for root, members in sorted(groups.items(), key=lambda kv: kv[0]):
        members = sorted(members)
        subclass = cluster_to_subclass.get(members[0], "Unknown")

        if len(members) == 1:
            # Singleton — keep original name
            label = members[0]
        else:
            # Merged group — name by subclass + member count
            label = f"{subclass} ({len(members)} merged)"

        for cl in members:
            mapping[cl] = label

        summary_rows.append({
            "collapsed_label": label,
            "subclass": subclass,
            "n_members": len(members),
            "member_clusters": " | ".join(members),
            "total_cells": sum(
                int((tasic_z.obs[cluster_col] == cl).sum()) for cl in members
            ),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["subclass", "n_members"], ascending=[True, False]
    ).reset_index(drop=True)

    return mapping, summary


def plot_separability_summary(
    separability_df: pd.DataFrame,
    collapse_summary: pd.DataFrame,
    centroids: pd.DataFrame,
    mapping: dict[str, str],
    out_dir: Path,
) -> None:
    """Save diagnostic plots for the collapse step."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Effect size distribution — separable vs inseparable pairs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for sub in sorted(separability_df["subclass"].unique()):
        sub_df = separability_df[separability_df["subclass"] == sub]
        ax.scatter(
            sub_df.index, sub_df["best_effect_size"],
            label=sub, alpha=0.7, s=30,
        )
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.6, label="threshold=1.0")
    ax.set_xlabel("Pair index")
    ax.set_ylabel("Best single-gene effect size")
    ax.set_title("Pairwise separability (within subclass)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # Plot 2: Collapsed taxonomy overview
    ax = axes[1]
    sub_counts = collapse_summary.groupby("subclass").agg(
        n_groups=("collapsed_label", "count"),
        n_original=("n_members", "sum"),
    ).reset_index()
    x = np.arange(len(sub_counts))
    width = 0.35
    ax.bar(x - width/2, sub_counts["n_original"], width, label="Original clusters", alpha=0.8)
    ax.bar(x + width/2, sub_counts["n_groups"], width, label="After collapse", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sub_counts["subclass"], rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Clusters before/after collapse")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "stage3_01_separability_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: Collapsed centroid heatmap
    collapsed_labels = [mapping.get(cl, cl) for cl in centroids.index]
    centroids_collapsed = centroids.copy()
    centroids_collapsed["_label"] = collapsed_labels
    mean_collapsed = centroids_collapsed.groupby("_label").mean()

    # Sort by subclass
    label_subclass = {}
    for _, row in collapse_summary.iterrows():
        label_subclass[row["collapsed_label"]] = row["subclass"]
    sort_key = [label_subclass.get(l, "ZZZ") + l for l in mean_collapsed.index]
    mean_collapsed = mean_collapsed.iloc[np.argsort(sort_key)]

    fig, ax = plt.subplots(figsize=(14, max(6, len(mean_collapsed) * 0.35)))
    sns.heatmap(
        mean_collapsed, cmap="RdBu_r", center=0, ax=ax,
        cbar_kws={"label": "z-score", "shrink": 0.8},
    )
    ax.set_title("Panel-resolution reference centroids (collapsed taxonomy)")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "stage3_02_collapsed_centroids.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: stage3_01_separability_overview.png, stage3_02_collapsed_centroids.png")


def run_stage3(
    tasic_z: ad.AnnData,
    out_dir: Path,
    effect_size_threshold: float = 1.0,
) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
    """
    Execute Stage 3: Collapse Tasic taxonomy to panel resolution.

    Parameters
    ----------
    tasic_z : AnnData
        Z-scored Tasic inhibitory cells.
    out_dir : Path
        Output directory.
    effect_size_threshold : float
        Minimum effect size for a pair to be considered separable.
        Default 1.0 (one pooled-SD difference on at least one gene).

    Returns
    -------
    mapping : dict
        Original cluster → collapsed label.
    centroids_collapsed : DataFrame
        Mean z-scored expression per collapsed group.
    separability_df : DataFrame
        Full pairwise separability table.
    """
    print("\n" + "=" * 60)
    print("STAGE 3: Collapse Tasic Labels to Panel Resolution (Approach A)")
    print("=" * 60)

    stage3_dir = out_dir / "stage3"
    stage3_dir.mkdir(parents=True, exist_ok=True)

    # 3.1 Compute per-type centroids
    print("\n[3.1] Computing per-cluster centroids...")
    centroids = compute_cluster_centroids(tasic_z)
    print(f"  {len(centroids)} clusters, {centroids.shape[1]} genes")

    # 3.2 Test pairwise separability
    print(f"\n[3.2] Testing pairwise separability (threshold={effect_size_threshold})...")
    separability_df = compute_pairwise_separability(
        tasic_z, effect_size_threshold=effect_size_threshold
    )
    n_pairs = len(separability_df)
    n_separable = int(separability_df["separable"].sum())
    n_inseparable = n_pairs - n_separable
    print(f"  {n_pairs} within-subclass pairs tested")
    print(f"  {n_separable} separable, {n_inseparable} inseparable")
    print(f"\n  Inseparable pairs (will be merged):")
    insep = separability_df[~separability_df["separable"]]
    for _, row in insep.iterrows():
        print(f"    {row['cluster_a']}  ↔  {row['cluster_b']}  "
              f"(best: {row['best_gene']}={row['best_effect_size']:.2f})")

    # 3.3 Merge non-separable types
    print(f"\n[3.3] Collapsing inseparable clusters...")
    mapping, collapse_summary = collapse_inseparable_clusters(separability_df, tasic_z)
    n_original = len(set(mapping.keys()))
    n_collapsed = len(set(mapping.values()))
    print(f"  {n_original} original clusters → {n_collapsed} panel-resolution groups")

    # 3.4 Save mapping table + plots
    print(f"\n[3.4] Saving outputs...")
    mapping_df = pd.DataFrame([
        {"original_cluster": k, "collapsed_label": v}
        for k, v in sorted(mapping.items())
    ])
    mapping_df.to_csv(stage3_dir / "cluster_collapse_mapping.csv", index=False)
    collapse_summary.to_csv(stage3_dir / "collapse_summary.csv", index=False)
    separability_df.to_csv(stage3_dir / "pairwise_separability.csv", index=False)

    # Compute collapsed centroids for downstream matching
    centroids_with_label = centroids.copy()
    centroids_with_label["_collapsed"] = [mapping[cl] for cl in centroids.index]
    centroids_collapsed = centroids_with_label.groupby("_collapsed").mean()

    centroids_collapsed.to_csv(stage3_dir / "collapsed_centroids.csv")
    print(f"  Panel-resolution reference: {len(centroids_collapsed)} groups")

    # Diagnostic plots
    plot_separability_summary(separability_df, collapse_summary, centroids, mapping, stage3_dir)

    # Print summary by subclass
    print("\n  Collapse summary by subclass:")
    for sub in sorted(collapse_summary["subclass"].unique()):
        sub_rows = collapse_summary[collapse_summary["subclass"] == sub]
        n_groups = len(sub_rows)
        n_orig = sub_rows["n_members"].sum()
        merged_groups = sub_rows[sub_rows["n_members"] > 1]
        print(f"    {sub}: {n_orig} → {n_groups} groups"
              f" ({len(merged_groups)} merged groups)")

    return mapping, centroids_collapsed, separability_df


# =============================================================================
# Stage 4 — Approach C: Supervised Hierarchical Clustering
# =============================================================================

# Canonical branches for gating (the four major inhibitory subclasses in cortex)
CANONICAL_BRANCHES = ["Pvalb", "Sst", "Vip", "Lamp5"]


def soft_subclass_gating(
    adata: ad.AnnData,
    tasic_z: ad.AnnData,
    n_neighbors: int = 15,
    confidence_threshold: float = 0.5,
    margin_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Soft (probabilistic) subclass gating via k-NN classifier trained on
    Tasic subclass labels, applied to query cells.

    Protocol C.1: assign by confidence margin, not just the max. Route
    problem cases to explicit bins (Inh-unassigned, Inh-ambiguous).

    Parameters
    ----------
    adata : AnnData
        Query cells (HCR or Tasic) — z-scored, shared panel genes.
    tasic_z : AnnData
        Tasic reference cells — z-scored, same genes. Must have 'subclass' in .obs.
    n_neighbors : int
        Number of neighbors for k-NN voting.
    confidence_threshold : float
        Minimum max probability to assign (default 0.5).
    margin_threshold : float
        Minimum gap between top and second probability (default 0.2).

    Returns
    -------
    DataFrame with columns: cell_id, assigned_branch, confidence, margin, top_prob,
                            second_prob, top_label, second_label
    """
    from sklearn.neighbors import NearestNeighbors

    # Use only canonical branches for gating targets
    # Smaller subclasses (Sncg, Serpinf1, CR, Meis2) get folded into the
    # closest canonical branch or routed to unassigned
    target_labels = tasic_z.obs["subclass"].values
    unique_labels = sorted(set(target_labels))

    X_ref = tasic_z.X if not hasattr(tasic_z.X, "toarray") else tasic_z.X.toarray()
    X_query = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()

    # Fit k-NN on reference
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(X_ref)
    _, indices = nn.kneighbors(X_query)

    # Vote for each query cell
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    n_cells = X_query.shape[0]
    n_labels = len(unique_labels)
    vote_matrix = np.zeros((n_cells, n_labels), dtype=np.float64)

    for cell_i, neighbor_idxs in enumerate(indices):
        for nb_idx in neighbor_idxs:
            vote_matrix[cell_i, label_to_idx[target_labels[nb_idx]]] += 1.0
    vote_matrix /= n_neighbors

    # Decision rule: confidence + margin
    sorted_probs = np.sort(vote_matrix, axis=1)
    top_prob = sorted_probs[:, -1]
    second_prob = sorted_probs[:, -2]
    margin = top_prob - second_prob
    top_idx = np.argmax(vote_matrix, axis=1)
    top_label = np.array(unique_labels)[top_idx]

    # Second-best label
    vote_copy = vote_matrix.copy()
    vote_copy[np.arange(n_cells), top_idx] = -1
    second_idx = np.argmax(vote_copy, axis=1)
    second_label = np.array(unique_labels)[second_idx]

    # Assign: confident = top_prob > threshold AND margin > margin_threshold
    assigned = []
    for i in range(n_cells):
        if top_prob[i] >= confidence_threshold and margin[i] >= margin_threshold:
            # Canonical branch assignment
            if top_label[i] in CANONICAL_BRANCHES:
                assigned.append(top_label[i])
            else:
                # Minor subclasses — keep as-is (Sncg, Serpinf1, etc.)
                assigned.append(top_label[i])
        elif top_prob[i] < 0.3:
            # Very low confidence → marker-negative / unassigned
            assigned.append("Inh-unassigned")
        else:
            # Ambiguous (moderate probability but low margin)
            assigned.append("Inh-ambiguous")

    result = pd.DataFrame({
        "cell_id": adata.obs_names,
        "assigned_branch": assigned,
        "confidence": top_prob,
        "margin": margin,
        "top_prob": top_prob,
        "second_prob": second_prob,
        "top_label": top_label,
        "second_label": second_label,
    })
    return result


def within_branch_leiden_clustering(
    tasic_log: ad.AnnData,
    tasic_z: ad.AnnData,
    branch: str,
    branch_assignments: pd.DataFrame,
    n_neighbors_range: list[int] | None = None,
    resolution_range: list[float] | None = None,
    selection_bootstraps: int = 10,
    selection_subsample_frac: float = 0.8,
    min_cells_per_branch_cluster: int = 0,
) -> tuple[ad.AnnData, dict]:
    """
    Run Leiden parameter sweep within a single subclass branch on Tasic data.

    Uses ARI against original Tasic cluster labels to find best params.

    Parameters
    ----------
    tasic_log : AnnData
        Full Tasic log-normalized data (used for clustering).
    tasic_z : AnnData
        Full Tasic z-scored data (used for downstream matching outputs).
    branch : str
        Branch name (e.g. "Pvalb").
    branch_assignments : DataFrame
        Output from soft_subclass_gating applied to Tasic cells.
    n_neighbors_range, resolution_range : optional parameter ranges for Leiden.
    min_cells_per_branch_cluster : int
        After final clustering, drop Leiden clusters with fewer than this many
        cells (0 = keep all). Dropped cells are removed from adata_branch so
        they do not bias downstream centroid computation or matching.

    Returns
    -------
    adata_branch : AnnData
        Z-scored branch subset with clustering labels in .obs.
    results_dict : dict
        Sweep results, best params, ARI.
    """
    from sklearn.metrics import adjusted_rand_score

    # Select cells assigned to this branch
    branch_cells = branch_assignments[
        branch_assignments["assigned_branch"] == branch
    ]["cell_id"].values
    branch_obs_names = tasic_log.obs_names[tasic_log.obs_names.isin(branch_cells)]
    adata_branch_cluster = tasic_log[branch_obs_names].copy()
    adata_branch_match = tasic_z[branch_obs_names].copy()

    if adata_branch_cluster.n_obs < 10:
        print(f"    WARNING: Branch {branch} has only {adata_branch_cluster.n_obs} cells, skipping.")
        return adata_branch_match, {"skip": True, "n_cells": adata_branch_cluster.n_obs}

        print(f"    Branch {branch}: {adata_branch_cluster.n_obs} cells, "
            f"{adata_branch_cluster.obs['cluster'].nunique()} original Tasic clusters")

    if n_neighbors_range is None:
        n_neighbors_range = [5, 10, 15, 20, 30, 40, 50]
    if resolution_range is None:
        resolution_range = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    # Parameter sweep: evaluate each parameter pair with ARI, silhouette,
    # and a quick bootstrap-stability estimate.
    sweep_rows = []
    rng_sel = np.random.default_rng(0)
    for n_neighbors in n_neighbors_range:
        if n_neighbors >= adata_branch_cluster.n_obs:
            continue
        for resolution in resolution_range:
            sc.pp.neighbors(adata_branch_cluster, use_rep="X", n_neighbors=n_neighbors)
            sc.tl.leiden(
                adata_branch_cluster, resolution=resolution,
                key_added="leiden_test", flavor="igraph",
            )
            ari = adjusted_rand_score(
                adata_branch_cluster.obs["cluster"].astype(str),
                adata_branch_cluster.obs["leiden_test"].astype(str),
            )
            
            # Compute silhouette score (cluster tightness in gene expression space)
            n_clusters_test = adata_branch_cluster.obs["leiden_test"].nunique()
            if n_clusters_test > 1 and n_clusters_test < adata_branch_cluster.n_obs:
                silhouette = silhouette_score(
                    adata_branch_cluster.X,
                    adata_branch_cluster.obs["leiden_test"].astype(str).values
                )
            else:
                silhouette = np.nan

            # Quick stability estimate: re-cluster subsamples and compare
            # against full-data labels for the same parameter setting.
            full_labels_test = adata_branch_cluster.obs["leiden_test"].astype(str).values
            n_cells = adata_branch_cluster.n_obs
            subsample_size = max(10, int(n_cells * selection_subsample_frac))
            stab_scores = []
            for _ in range(selection_bootstraps):
                idx = rng_sel.choice(n_cells, size=subsample_size, replace=False)
                adata_sub = adata_branch_cluster[idx].copy()
                n_nb_sub = min(n_neighbors, adata_sub.n_obs - 1)
                if n_nb_sub < 2:
                    continue
                sc.pp.neighbors(adata_sub, use_rep="X", n_neighbors=n_nb_sub)
                sc.tl.leiden(adata_sub, resolution=resolution, flavor="igraph")
                stab_scores.append(
                    adjusted_rand_score(full_labels_test[idx], adata_sub.obs["leiden"].astype(str).values)
                )

            stability_mean = float(np.mean(stab_scores)) if stab_scores else np.nan
            stability_std = float(np.std(stab_scores)) if stab_scores else np.nan
            
            sweep_rows.append({
                "n_neighbors": n_neighbors,
                "resolution": resolution,
                "ari": ari,
                "silhouette": silhouette,
                "stability_mean": stability_mean,
                "stability_std": stability_std,
                "n_clusters": n_clusters_test,
            })

    sweep_df = pd.DataFrame(sweep_rows).reset_index(drop=True)

    # ── Stability-first parameter selection ──────────────────────────────────
    # Strategy:
    #   1. Pick the setting with the highest bootstrap stability_mean.
    #   2. Break ties by silhouette, then ARI.
    #   3. If stability is unavailable, fall back to silhouette then ARI.
    valid_stab = sweep_df[sweep_df["stability_mean"].notna()].copy()
    if len(valid_stab) > 0:
        best = valid_stab.sort_values(
            ["stability_mean", "silhouette", "ari"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        valid_sil = sweep_df[sweep_df["silhouette"].notna()]
        if len(valid_sil) == 0:
            print("      WARNING: stability/silhouette unavailable, falling back to ARI")
            best = sweep_df.sort_values("ari", ascending=False).iloc[0]
        else:
            print("      WARNING: stability unavailable, selecting by silhouette")
            best = valid_sil.sort_values(["silhouette", "ari"], ascending=[False, False]).iloc[0]

    best_n = int(best["n_neighbors"])
    best_r = float(best["resolution"])

    # Final clustering with best params
    sc.pp.neighbors(adata_branch_cluster, use_rep="X", n_neighbors=best_n)
    sc.tl.leiden(adata_branch_cluster, resolution=best_r, flavor="igraph")
    sc.tl.umap(adata_branch_cluster)

    # Label Leiden clusters with branch prefix
    adata_branch_cluster.obs["branch_cluster"] = [
        f"{branch}-{lid}" for lid in adata_branch_cluster.obs["leiden"].values
    ]

    # Transfer clustering labels from log-space clustering object to z-space
    # matching object so downstream centroid correlation remains in z-space.
    leiden_map = adata_branch_cluster.obs["leiden"].astype(str)
    branch_cluster_map = adata_branch_cluster.obs["branch_cluster"].astype(str)
    adata_branch_match.obs["leiden"] = leiden_map.reindex(adata_branch_match.obs_names).values
    adata_branch_match.obs["branch_cluster"] = (
        branch_cluster_map.reindex(adata_branch_match.obs_names).values
    )
    adata_branch_match.obsm["X_umap"] = adata_branch_cluster.obsm["X_umap"]
    adata_branch_match.obsm["X_cluster"] = adata_branch_cluster.X

    # Drop underpopulated branch clusters if threshold is set
    if min_cells_per_branch_cluster > 0:
        cluster_counts = adata_branch_cluster.obs["branch_cluster"].value_counts()
        keep_clusters = cluster_counts[cluster_counts >= min_cells_per_branch_cluster].index
        n_before = adata_branch_cluster.n_obs
        n_clusters_before = cluster_counts.shape[0]
        adata_branch_cluster = adata_branch_cluster[
            adata_branch_cluster.obs["branch_cluster"].isin(keep_clusters)
        ].copy()
        adata_branch_match = adata_branch_match[
            adata_branch_match.obs_names.isin(adata_branch_cluster.obs_names)
        ].copy()
        adata_branch_match.obsm["X_umap"] = adata_branch_cluster.obsm["X_umap"]
        adata_branch_match.obsm["X_cluster"] = adata_branch_cluster.X
        n_dropped_clusters = n_clusters_before - len(keep_clusters)
        n_dropped_cells = n_before - adata_branch_cluster.n_obs
        if n_dropped_clusters > 0:
            print(f"      Dropped {n_dropped_clusters} branch cluster(s) with "
                  f"< {min_cells_per_branch_cluster} cells "
                  f"({n_dropped_cells} cells removed)")

    n_clusters = adata_branch_cluster.obs["leiden"].nunique()
    best_sil = best["silhouette"] if pd.notna(best["silhouette"]) else float("nan")
    best_stab = best["stability_mean"] if pd.notna(best["stability_mean"]) else float("nan")
    print(f"      Best params: n_neighbors={best_n}, resolution={best_r:.1f}, "
          f"stability={best_stab:.3f}, silhouette={best_sil:.3f}, "
          f"ARI={best['ari']:.3f}, n_clusters={n_clusters}")

    results_dict = {
        "sweep_df": sweep_df,
        "best_params": {
            "best_n_neighbors": best_n,
            "best_resolution": best_r,
            "best_stability": float(best_stab),
            "best_ari": float(best["ari"]),
            "best_silhouette": float(best_sil),
        },
        "n_cells": adata_branch_cluster.n_obs,
        "n_clusters": n_clusters,
    }
    return adata_branch_match, results_dict


def bootstrap_branch_stability(
    adata_branch: ad.AnnData,
    n_bootstraps: int = 50,
    subsample_frac: float = 0.8,
    n_neighbors: int | None = None,
    resolution: float | None = None,
) -> pd.DataFrame:
    """
    Bootstrap stability test for within-branch Leiden clusters.

    Subsamples cells, re-clusters, measures ARI against full-data labels.

    Returns DataFrame with per-bootstrap ARI and cluster count.
    """
    from sklearn.metrics import adjusted_rand_score

    if n_neighbors is None:
        n_neighbors = 15
    if resolution is None:
        resolution = 1.0

    full_labels = adata_branch.obs["leiden"].values
    n_cells = adata_branch.n_obs
    subsample_size = max(10, int(n_cells * subsample_frac))

    results = []
    rng = np.random.default_rng(42)

    for i in range(n_bootstraps):
        idx = rng.choice(n_cells, size=subsample_size, replace=False)
        adata_sub = adata_branch[idx].copy()

        n_nb = min(n_neighbors, adata_sub.n_obs - 1)
        if n_nb < 2:
            continue

        sc.pp.neighbors(adata_sub, use_rep="X", n_neighbors=n_nb)
        sc.tl.leiden(adata_sub, resolution=resolution, flavor="igraph")

        ari = adjusted_rand_score(
            full_labels[idx],
            adata_sub.obs["leiden"].values,
        )
        results.append({
            "bootstrap": i,
            "ari": ari,
            "n_clusters": adata_sub.obs["leiden"].nunique(),
            "n_cells": adata_sub.n_obs,
        })

    return pd.DataFrame(results)


def build_contingency_table(
    adata_branch: ad.AnnData,
    branch_cluster_col: str = "branch_cluster",
    original_col: str = "cluster",
) -> pd.DataFrame:
    """
    Contingency table: branch Leiden clusters × original Tasic clusters.
    Rows = branch_cluster labels, columns = original Tasic clusters.
    """
    return pd.crosstab(
        adata_branch.obs[branch_cluster_col],
        adata_branch.obs[original_col],
    )


def plot_stage4_diagnostics(
    gating_df: pd.DataFrame,
    branch_results: dict,
    stability_results: dict,
    out_dir: Path,
) -> None:
    """Save Stage 4 diagnostic plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Gating summary — pie chart of assignments + confidence distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Assignment counts
    ax = axes[0]
    counts = gating_df["assigned_branch"].value_counts()
    colors = {
        "Pvalb": "#377EB8", "Sst": "#E41A1C", "Vip": "#984EA3",
        "Lamp5": "#4DAF4A", "Sncg": "#FF7F00", "Serpinf1": "#A65628",
        "Inh-unassigned": "#999999", "Inh-ambiguous": "#666666",
    }
    bar_colors = [colors.get(c, "#BBBBBB") for c in counts.index]
    ax.barh(range(len(counts)), counts.values, color=bar_colors)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=9)
    ax.set_xlabel("Cell count")
    ax.set_title("Subclass gating assignments")

    # Confidence distribution
    ax = axes[1]
    for branch in CANONICAL_BRANCHES:
        branch_data = gating_df[gating_df["assigned_branch"] == branch]
        if len(branch_data) > 0:
            ax.hist(branch_data["confidence"], bins=30, alpha=0.5,
                    label=branch, color=colors.get(branch))
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.6, label="threshold")
    ax.set_xlabel("Top probability")
    ax.set_ylabel("Count")
    ax.set_title("Gating confidence (assigned cells)")
    ax.legend(fontsize=8)

    # Margin distribution
    ax = axes[2]
    ax.hist(gating_df["margin"], bins=50, alpha=0.7, color="#555555")
    ax.axvline(0.2, color="red", linestyle="--", alpha=0.6, label="margin threshold")
    ax.set_xlabel("Margin (P_top - P_second)")
    ax.set_ylabel("Count")
    ax.set_title("Gating margin distribution (all cells)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "stage4_01_gating_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Per-branch Leiden sweep heatmaps + UMAP
    n_branches = len(branch_results)
    if n_branches > 0:
        fig, axes = plt.subplots(2, n_branches, figsize=(5 * n_branches, 9))
        if n_branches == 1:
            axes = axes.reshape(2, 1)

        for i, (branch, (adata_br, res_dict)) in enumerate(branch_results.items()):
            if res_dict.get("skip"):
                continue

            # Top: UMAP colored by Leiden
            ax = axes[0, i]
            sc.pl.umap(adata_br, color="branch_cluster", ax=ax, show=False,
                       title=f"{branch}: Leiden clusters", legend_loc="on data",
                       legend_fontsize=7)

            # Bottom: UMAP colored by original Tasic cluster
            ax = axes[1, i]
            sc.pl.umap(adata_br, color="cluster", ax=ax, show=False,
                       title=f"{branch}: original Tasic labels", legend_loc="none")

        plt.tight_layout()
        fig.savefig(out_dir / "stage4_02_branch_umaps.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Plot 3: Bootstrap stability
    if stability_results:
        fig, axes = plt.subplots(1, len(stability_results),
                                 figsize=(4 * len(stability_results), 4))
        if len(stability_results) == 1:
            axes = [axes]
        for i, (branch, stab_df) in enumerate(stability_results.items()):
            ax = axes[i]
            ax.boxplot(stab_df["ari"].values, vert=True)
            ax.set_title(f"{branch} stability\n(mean ARI={stab_df['ari'].mean():.3f})")
            ax.set_ylabel("Bootstrap ARI")
            ax.set_ylim(0, 1)
            ax.axhline(0.8, color="green", linestyle="--", alpha=0.5, label="good")
            ax.legend(fontsize=8)

        plt.tight_layout()
        fig.savefig(out_dir / "stage4_03_stability.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: stage4_01-03 diagnostic plots")


def plot_sweep_diagnostics(
    sweep_df: pd.DataFrame,
    branch: str,
    out_dir: Path,
    best_n_neighbors: int | None = None,
    best_resolution: float | None = None,
) -> None:
    """
    Three-panel sweep diagnostic:
      Left  — Silhouette heatmap (resolution × n_neighbors), selection marked
      Centre — ARI heatmap (reference only)
      Right  — Elbow plot: mean silhouette vs. n_clusters (aggregated over all
                n_neighbors at each resolution), with within-cluster SS on a
                secondary axis as the classic elbow curve.
    """
    has_sil = "silhouette" in sweep_df.columns and sweep_df["silhouette"].notna().any()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: Silhouette heatmap ──────────────────────────────────────────
    if has_sil:
        sil_pivot = sweep_df.pivot_table(
            index="resolution", columns="n_neighbors", values="silhouette"
        )
        sns.heatmap(sil_pivot, annot=True, fmt=".3f", cmap="coolwarm",
                    ax=axes[0], cbar_kws={"label": "Silhouette"}, center=0)
        # Mark selected params
        if best_resolution is not None and best_n_neighbors is not None:
            rows = list(sil_pivot.index)
            cols = list(sil_pivot.columns)
            if best_resolution in rows and best_n_neighbors in cols:
                r_idx = rows.index(best_resolution)
                c_idx = cols.index(best_n_neighbors)
                axes[0].add_patch(
                    plt.Rectangle((c_idx, r_idx), 1, 1,
                                   fill=False, edgecolor="gold", lw=3)
                )
        axes[0].set_title(f"{branch}: Silhouette (★ = selected)")
    else:
        axes[0].text(0.5, 0.5, "Silhouette unavailable",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title(f"{branch}: Silhouette")
    axes[0].set_xlabel("n_neighbors")
    axes[0].set_ylabel("resolution")

    # ── Panel 2: ARI heatmap (reference) ────────────────────────────────────
    ari_pivot = sweep_df.pivot_table(
        index="resolution", columns="n_neighbors", values="ari"
    )
    sns.heatmap(ari_pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                ax=axes[1], cbar_kws={"label": "ARI"}, vmin=0, vmax=1)
    axes[1].set_title(f"{branch}: ARI vs. Tasic labels (reference only)")
    axes[1].set_xlabel("n_neighbors")
    axes[1].set_ylabel("resolution")

    # ── Panel 3: Elbow plot ──────────────────────────────────────────────────
    # Aggregate by n_clusters: mean silhouette and mean within-SS as elbow signal
    ax3 = axes[2]
    elbow_df = (
        sweep_df
        .groupby("n_clusters", as_index=False)
        .agg(mean_silhouette=("silhouette", "mean"), mean_ari=("ari", "mean"))
        .sort_values("n_clusters")
    )

    color_sil = "#1f77b4"
    color_ari = "#d62728"

    ax3.plot(elbow_df["n_clusters"], elbow_df["mean_silhouette"],
             "o-", color=color_sil, linewidth=2, markersize=5, label="Mean silhouette")
    ax3.set_xlabel("Number of clusters")
    ax3.set_ylabel("Mean silhouette score", color=color_sil)
    ax3.tick_params(axis="y", labelcolor=color_sil)
    ax3.axhline(0, color=color_sil, linestyle=":", alpha=0.4)

    ax3b = ax3.twinx()
    ax3b.plot(elbow_df["n_clusters"], elbow_df["mean_ari"],
              "s--", color=color_ari, linewidth=1.5, markersize=4,
              alpha=0.7, label="Mean ARI")
    ax3b.set_ylabel("Mean ARI", color=color_ari)
    ax3b.tick_params(axis="y", labelcolor=color_ari)

    # Mark selected n_clusters if known
    if best_resolution is not None and best_n_neighbors is not None:
        sel = sweep_df[
            (sweep_df["resolution"] == best_resolution) &
            (sweep_df["n_neighbors"] == best_n_neighbors)
        ]
        if len(sel) > 0:
            sel_k = int(sel.iloc[0]["n_clusters"])
            ax3.axvline(sel_k, color="gold", linestyle="--", linewidth=2,
                        label=f"Selected k={sel_k}")

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax3.set_title(f"{branch}: Elbow plot (silhouette vs. n_clusters)")
    ax3.set_xticks(sorted(elbow_df["n_clusters"].unique()))
    ax3.tick_params(axis="x", rotation=45)

    plt.suptitle(
        f"{branch}: Parameter sweep diagnostics (selection = max mean silhouette per resolution)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    out_path = out_dir / f"sweep_diagnostics_{branch.lower()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def extract_top_markers_per_cluster(
    adata: ad.AnnData,
    cluster_col: str = "leiden",
    n_top: int = 5,
) -> pd.DataFrame:
    """
    Extract top discriminable genes per cluster (one-vs-rest effect size).
    
    Returns DataFrame with cluster, gene, effect_size (mean fold-change in z-scores).
    """
    clusters = adata.obs[cluster_col].unique()
    X_z = adata.X  # Already z-scored
    
    markers = []
    for cluster in sorted(clusters):
        in_cluster = adata.obs[cluster_col] == cluster
        out_cluster = ~in_cluster
        
        for gene_idx, gene_name in enumerate(adata.var_names):
            in_expr = X_z[in_cluster, gene_idx]
            out_expr = X_z[out_cluster, gene_idx]
            
            fold_change = in_expr.mean() - out_expr.mean()
            markers.append({
                "cluster": cluster,
                "gene": gene_name,
                "effect_size": fold_change,
            })
    
    markers_df = pd.DataFrame(markers)
    
    # Keep top n per cluster
    top_markers = (
        markers_df
        .sort_values("effect_size", ascending=False, key=abs)
        .groupby("cluster", group_keys=False)
        .head(n_top)
        .sort_values(["cluster", "effect_size"], ascending=[True, False])
    )
    return top_markers


def print_sweep_summary(
    sweep_df: pd.DataFrame,
    branch: str,
    top_n: int = 5,
) -> None:
    """
    Print a summary of sweep results ranked by different metrics.
    """
    print(f"\n    === Sweep Summary for {branch} ===")
    print(f"    Tested {len(sweep_df)} parameter combinations")
    
    # Top by ARI
    top_ari = sweep_df.nlargest(top_n, "ari")[
        ["n_neighbors", "resolution", "ari", "n_clusters", "silhouette"]
    ]
    print(f"\n    Top {top_n} by ARI (taxonomy agreement):")
    for i, row in top_ari.iterrows():
        sil_str = f"silhouette={row['silhouette']:.3f}" if pd.notna(row['silhouette']) else "silhouette=N/A"
        print(f"      n_neighbors={int(row['n_neighbors']):2d}, resolution={row['resolution']:.1f} → "
              f"ARI={row['ari']:.3f}, n_clusters={int(row['n_clusters']):2d}, {sil_str}")
    
    # Top by silhouette (if available) + mean silhouette per resolution (selection basis)
    if sweep_df["silhouette"].notna().any():
        top_sil = sweep_df.nlargest(top_n, "silhouette")[
            ["n_neighbors", "resolution", "ari", "n_clusters", "silhouette"]
        ]
        print(f"\n    Top {top_n} by Silhouette (cluster tightness) — used for selection:")
        for i, row in top_sil.iterrows():
            print(f"      n_neighbors={int(row['n_neighbors']):2d}, resolution={row['resolution']:.1f} → "
                  f"silhouette={row['silhouette']:.3f}, ARI={row['ari']:.3f}, n_clusters={int(row['n_clusters']):2d}")

        mean_sil = sweep_df.groupby("resolution")["silhouette"].mean().sort_values(ascending=False)
        print(f"\n    Mean silhouette per resolution (selection rank):")
        for res, sil in mean_sil.items():
            marker = " ← selected" if res == mean_sil.index[0] else ""
            print(f"      resolution={res:.1f}  mean_silhouette={sil:.3f}{marker}")


def run_stage4(
    tasic_log: ad.AnnData,
    tasic_z: ad.AnnData,
    hcr_corrected: ad.AnnData | None,
    out_dir: Path,
    n_neighbors_gating: int = 15,
    confidence_threshold: float = 0.5,
    margin_threshold: float = 0.2,
    n_bootstraps: int = 50,
    min_cells_per_branch_cluster: int = 0,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Execute Stage 4: Approach C supervised hierarchical clustering.

    Steps:
    1. Soft subclass gating on Tasic (self-validation) and HCR
    2. Within-branch Leiden clustering on Tasic reference
    3. Bootstrap stability test per branch
    4. Contingency table mapping branch clusters → original Tasic labels

    Parameters
    ----------
    tasic_log : AnnData
        Log-normalized Tasic inhibitory cells (used for within-branch clustering).
    tasic_z : AnnData
        Z-scored Tasic inhibitory cells (used for matching features and gating).
    hcr_corrected : AnnData | None
        Batch-corrected z-scored HCR cells. If None, skip HCR gating.
    out_dir : Path
        Output directory.
    n_neighbors_gating : int
        k for k-NN gating classifier.
    confidence_threshold : float
        Min top probability for assignment.
    margin_threshold : float
        Min gap between top and second probability.
    n_bootstraps : int
        Number of bootstrap iterations for stability.
    min_cells_per_branch_cluster : int
        Drop Leiden clusters within each branch that have fewer than this many
        Tasic cells after final clustering (0 = keep all).

    Returns
    -------
    branch_results : dict
        {branch: (adata_branch, results_dict)} for each canonical branch.
    tasic_gating : DataFrame
        Gating assignments for Tasic cells (self-validation).
    hcr_gating : DataFrame
        Gating assignments for HCR cells.
    """
    print("\n" + "=" * 60)
    print("STAGE 4: Approach C — Supervised Hierarchical Clustering")
    print("=" * 60)

    stage4_dir = out_dir / "stage4"
    stage4_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 4.1 Soft subclass gating — first validate on Tasic (known labels)
    # -------------------------------------------------------------------------
    print(f"\n[4.1] Soft subclass gating (k={n_neighbors_gating}, "
          f"conf≥{confidence_threshold}, margin≥{margin_threshold})")

    # Self-validation: gate Tasic cells using leave-one-out style k-NN
    # (since we're using the Tasic data itself, the k-NN will be biased
    # but still useful to validate thresholds and measure mis-gating rate)
    print("\n  Gating Tasic cells (self-validation)...")
    tasic_gating = soft_subclass_gating(
        tasic_z, tasic_z,
        n_neighbors=n_neighbors_gating,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )

    # Compare to known subclass labels
    tasic_gating["true_subclass"] = tasic_z.obs["subclass"].values
    # Map minor subclasses to canonical for accuracy calc
    canonical_map = {b: b for b in CANONICAL_BRANCHES}
    tasic_gating["true_canonical"] = tasic_gating["true_subclass"].map(
        lambda x: x if x in CANONICAL_BRANCHES else "Other"
    )
    # Accuracy: among cells assigned to canonical branches, how many are correct?
    assigned_canonical = tasic_gating[
        tasic_gating["assigned_branch"].isin(CANONICAL_BRANCHES)
    ]
    if len(assigned_canonical) > 0:
        accuracy = (
            assigned_canonical["assigned_branch"] == assigned_canonical["true_canonical"]
        ).mean()
        print(f"  Tasic self-gating accuracy (canonical branches): {accuracy:.3f} "
              f"({len(assigned_canonical)} cells assigned)")

    # Report bin sizes
    print("\n  Tasic gating summary:")
    for branch, count in tasic_gating["assigned_branch"].value_counts().items():
        frac = count / len(tasic_gating)
        print(f"    {branch}: {count} ({frac:.1%})")

    if hcr_corrected is not None:
        # Gate HCR cells
        print("\n  Gating HCR cells...")
        hcr_gating = soft_subclass_gating(
            hcr_corrected, tasic_z,
            n_neighbors=n_neighbors_gating,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
        )
        print("\n  HCR gating summary:")
        for branch, count in hcr_gating["assigned_branch"].value_counts().items():
            frac = count / len(hcr_gating)
            print(f"    {branch}: {count} ({frac:.1%})")
    else:
        hcr_gating = pd.DataFrame(
            columns=[
                "cell_id", "assigned_branch", "confidence", "margin", "top_prob",
                "second_prob", "top_label", "second_label",
            ]
        )
        print("\n  HCR gating skipped (no HCR query provided).")

    # -------------------------------------------------------------------------
    # 4.2 Within-branch Leiden clustering on Tasic reference
    # -------------------------------------------------------------------------
    print(f"\n[4.2] Within-branch Leiden clustering (Tasic reference)...")

    branch_results = {}
    stability_results = {}
    contingency_tables = {}
    branch_marker_names = {}  # {branch: {branch_cluster_id: "Branch-N: gene1 | gene2 | gene3"}}

    for branch in CANONICAL_BRANCHES:
        print(f"\n  --- Branch: {branch} ---")
        adata_branch, res_dict = within_branch_leiden_clustering(
            tasic_log, tasic_z, branch, tasic_gating,
            min_cells_per_branch_cluster=min_cells_per_branch_cluster,
        )

        if res_dict.get("skip"):
            print(f"    Skipped (too few cells)")
            continue

        branch_results[branch] = (adata_branch, res_dict)

        # 4.2b Print sweep summary and plot sweep diagnostics (silhouette elbow + heatmaps)
        sweep_df = res_dict["sweep_df"]
        best_p = res_dict["best_params"]
        print_sweep_summary(sweep_df, branch, top_n=3)
        plot_sweep_diagnostics(
            sweep_df, branch, stage4_dir,
            best_n_neighbors=best_p["best_n_neighbors"],
            best_resolution=best_p["best_resolution"],
        )

        # 4.3 Bootstrap stability
        print(f"    Running bootstrap stability ({n_bootstraps} iterations)...")
        best_params = res_dict["best_params"]
        stab_df = bootstrap_branch_stability(
            adata_branch,
            n_bootstraps=n_bootstraps,
            subsample_frac=0.8,
            n_neighbors=best_params["best_n_neighbors"],
            resolution=best_params["best_resolution"],
        )
        stability_results[branch] = stab_df
        print(f"      Stability: mean ARI={stab_df['ari'].mean():.3f} ± {stab_df['ari'].std():.3f}")

        # 4.4 Contingency table
        ct = build_contingency_table(adata_branch)
        contingency_tables[branch] = ct
        print(f"      Contingency: {ct.shape[0]} Leiden clusters × {ct.shape[1]} Tasic types")

        # 4.5 Discriminable markers per Leiden cluster (enriched + depleted)
        print(f"      Computing discriminable markers...")
        markers_df = cluster_validation_utils.top_discriminable_genes_per_cluster(
            adata_branch,
            cluster_col="branch_cluster",
            top_n=3,
            exclude_genes=("Gad2",),
            use_abs_effect_size=False,
            include_depleted=True,
            bootstrap_iterations=100,
            random_state=0,
        )

        # Build marker-derived names while excluding subclass terms.
        high_conf = markers_df[
            (markers_df["stability_pct"] >= 80.0) & (markers_df["direction"] == "enriched")
        ]
        # Exclude canonical/minor subclass labels from marker-name strings.
        excluded_name_genes = {
            g.lower() for g in (
                list(CANONICAL_BRANCHES)
                + ["Sncg"]
                + list(MINOR_SUBCLASSES)
            )
        }

        def _top_name_genes(df_in: pd.DataFrame, n: int = 3) -> list[str]:
            genes = [
                g for g in df_in["gene"].tolist()
                if isinstance(g, str) and g.lower() not in excluded_name_genes
            ]
            return genes[:n]

        name_map = {}
        for cl in sorted(adata_branch.obs["branch_cluster"].unique()):
            cl_markers = high_conf[high_conf["cluster"] == cl].sort_values("rank")
            marker_str = "-".join(_top_name_genes(cl_markers, n=3))
            if marker_str:
                name_map[cl] = f"{cl}: {marker_str}"
            else:
                # Fall back to top enriched regardless of stability
                cl_enriched = markers_df[
                    (markers_df["cluster"] == cl) & (markers_df["direction"] == "enriched")
                ].sort_values("rank")
                marker_str = "-".join(_top_name_genes(cl_enriched, n=3))
                name_map[cl] = f"{cl}: {marker_str}" if marker_str else cl
        branch_marker_names[branch] = name_map

        print(f"      Marker-derived names:")
        for cl, name in sorted(name_map.items()):
            print(f"        {name}")

        # 4.6 k-NN confidence plots (old Tasic labels + new Leiden labels)
        print(f"      Computing k-NN confidence...")
        fig_old, soft_old = cluster_validation_utils.plot_knn_confidence_subplots(
            adata_branch,
            label_col="cluster",
            n_neighbors=15,
            title=f"{branch}: panel confidence recovering original Tasic labels",
        )
        fig_new, soft_new = cluster_validation_utils.plot_knn_confidence_subplots(
            adata_branch,
            label_col="branch_cluster",
            n_neighbors=15,
            title=f"{branch}: panel confidence for Leiden clusters",
        )

        # 4.7 Overlap & enrichment heatmaps (Leiden vs original Tasic)
        fig_overlap = cluster_validation_utils.plot_old_new_overlap_heatmap_hierarchical(
            adata_branch,
            old_label_col="cluster",
            new_label_col="branch_cluster",
            normalize="old",
            figsize=(8, 6),
            title=f"{branch}: Tasic clusters → Leiden overlap (row-normalized)",
        )
        fig_enrichment = cluster_validation_utils.plot_old_new_log2_enrichment_heatmap(
            adata_branch,
            old_label_col="cluster",
            new_label_col="branch_cluster",
            vmax_abs=3.0,
            figsize=(8, 6),
            title=f"{branch}: Tasic × Leiden enrichment (log2 obs/exp)",
        )

        # Save per-branch diagnostic plots and tables
        branch_dir = stage4_dir / branch.lower()
        branch_dir.mkdir(parents=True, exist_ok=True)

        fig_old.savefig(branch_dir / f"{branch}_knn_confidence_old_labels.png",
                        dpi=200, bbox_inches="tight")
        plt.close(fig_old)
        fig_new.savefig(branch_dir / f"{branch}_knn_confidence_leiden.png",
                        dpi=200, bbox_inches="tight")
        plt.close(fig_new)
        fig_overlap.savefig(branch_dir / f"{branch}_overlap_heatmap.png",
                            dpi=200, bbox_inches="tight")
        plt.close(fig_overlap)
        fig_enrichment.savefig(branch_dir / f"{branch}_enrichment_heatmap.png",
                               dpi=200, bbox_inches="tight")
        plt.close(fig_enrichment)

        # 4.7b Cell×gene labeled plot (Tasic branch, Leiden labels)
        print(f"      Generating cell×gene labeled plot...")
        X_br_raw = adata_branch.X if not hasattr(adata_branch.X, "toarray") else adata_branch.X.toarray()
        cxg_branch = pd.DataFrame(
            X_br_raw, columns=adata_branch.var_names, index=adata_branch.obs_names
        )
        # Use marker-derived names as labels
        branch_labels = pd.Series(
            [name_map.get(cl, cl) for cl in adata_branch.obs["branch_cluster"].values],
            index=adata_branch.obs_names,
        )
        fig_cxg, _, _ = viz.plot_cell_x_gene_labeled(
            cxg_branch,
            labels=branch_labels,
            clip_range=(-2.5, 2.5),
            fig_size=(8, max(6, adata_branch.n_obs * 0.005)),
            label_fontsize=8,
            cbar_label="z-score",
            title=f"{branch} branch: Tasic cells by Leiden cluster",
            cmap="RdBu_r",
        )
        fig_cxg.savefig(branch_dir / f"{branch}_cell_x_gene_labeled.png",
                        dpi=150, bbox_inches="tight")
        plt.close(fig_cxg)

        markers_df.to_csv(branch_dir / f"{branch}_markers_long.csv", index=False)
        high_conf.to_csv(branch_dir / f"{branch}_high_conf_markers.csv", index=False)
        pd.DataFrame([
            {"branch_cluster": k, "marker_name": v} for k, v in name_map.items()
        ]).to_csv(branch_dir / f"{branch}_cluster_names.csv", index=False)

        # k-NN summary stats
        knn_summary = pd.DataFrame({
            "analysis": ["recover_tasic_labels", "leiden_clusters"],
            "mean_confidence": [
                soft_old.max(axis=1).mean(),
                soft_new.max(axis=1).mean(),
            ],
            "mean_margin": [
                (np.sort(soft_old.values, axis=1)[:, -1] - np.sort(soft_old.values, axis=1)[:, -2]).mean(),
                (np.sort(soft_new.values, axis=1)[:, -1] - np.sort(soft_new.values, axis=1)[:, -2]).mean(),
            ],
        })
        knn_summary.to_csv(branch_dir / f"{branch}_knn_summary.csv", index=False)
        print(f"      k-NN confidence (Tasic labels): {soft_old.max(axis=1).mean():.3f}")
        print(f"      k-NN confidence (Leiden):       {soft_new.max(axis=1).mean():.3f}")

    # -------------------------------------------------------------------------
    # 4.8 Save outputs
    # -------------------------------------------------------------------------
    print(f"\n[4.8] Saving Stage 4 outputs...")

    # Gating tables
    tasic_gating.to_csv(stage4_dir / "tasic_gating.csv", index=False)
    hcr_gating.to_csv(stage4_dir / "hcr_gating.csv", index=False)

    # Per-branch results (h5ad, sweep, contingency, stability)
    for branch, (adata_br, res_dict) in branch_results.items():
        branch_dir = stage4_dir / branch.lower()
        branch_dir.mkdir(parents=True, exist_ok=True)
        adata_br.write(branch_dir / f"{branch}_branch_tasic.h5ad")
        res_dict["sweep_df"].to_csv(branch_dir / f"{branch}_leiden_sweep.csv", index=False)
        if branch in contingency_tables:
            contingency_tables[branch].to_csv(branch_dir / f"{branch}_contingency.csv")
        if branch in stability_results:
            stability_results[branch].to_csv(
                branch_dir / f"{branch}_stability.csv", index=False
            )

    # Diagnostic plots (gating summary, branch UMAPs, stability)
    plot_stage4_diagnostics(tasic_gating, branch_results, stability_results, stage4_dir)

    # Summary
    print(f"\n  Stage 4 complete:")
    for branch, (adata_br, res_dict) in branch_results.items():
        stab = stability_results.get(branch)
        stab_str = f", stability={stab['ari'].mean():.3f}" if stab is not None else ""
        print(f"    {branch}: {res_dict['n_cells']} cells → "
              f"{res_dict['n_clusters']} clusters "
              f"(ARI={res_dict['best_params']['best_ari']:.3f}{stab_str})")

    return branch_results, tasic_gating, hcr_gating, branch_marker_names


def load_stage4_cached(out_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    """
    Load Stage 4 cached outputs from disk.

    Required files:
    - stage4/tasic_gating.csv
    - stage4/hcr_gating.csv
    - stage4/{branch}/{Branch}_branch_tasic.h5ad
    - stage4/{branch}/{Branch}_cluster_names.csv
    """
    stage4_dir = out_dir / "stage4"
    tasic_gating_path = stage4_dir / "tasic_gating.csv"
    hcr_gating_path = stage4_dir / "hcr_gating.csv"

    if not tasic_gating_path.exists() or not hcr_gating_path.exists():
        raise FileNotFoundError(
            "Missing stage4 gating tables; expected "
            f"{tasic_gating_path} and {hcr_gating_path}"
        )

    tasic_gating = pd.read_csv(tasic_gating_path)
    hcr_gating = pd.read_csv(hcr_gating_path)

    branch_results = {}
    branch_marker_names = {}

    for branch in CANONICAL_BRANCHES:
        branch_dir = stage4_dir / branch.lower()
        h5ad_path = branch_dir / f"{branch}_branch_tasic.h5ad"
        names_path = branch_dir / f"{branch}_cluster_names.csv"

        if not h5ad_path.exists() or not names_path.exists():
            continue

        adata_br = ad.read_h5ad(h5ad_path)
        if "leiden" in adata_br.obs.columns:
            n_clusters = int(pd.Series(adata_br.obs["leiden"]).nunique())
        elif "branch_cluster" in adata_br.obs.columns:
            n_clusters = int(pd.Series(adata_br.obs["branch_cluster"]).nunique())
        else:
            n_clusters = 0

        res_dict = {
            "skip": False,
            "n_cells": int(adata_br.n_obs),
            "n_clusters": n_clusters,
        }
        branch_results[branch] = (adata_br, res_dict)

        names_df = pd.read_csv(names_path)
        if {"branch_cluster", "marker_name"}.issubset(names_df.columns):
            branch_marker_names[branch] = dict(
                zip(names_df["branch_cluster"].astype(str), names_df["marker_name"].astype(str))
            )
        else:
            branch_marker_names[branch] = {}

    if len(branch_results) == 0:
        raise FileNotFoundError(
            f"No cached branch h5ad files found under {stage4_dir} for canonical branches {CANONICAL_BRANCHES}."
        )

    return branch_results, tasic_gating, hcr_gating, branch_marker_names


# =============================================================================
# Stage 5 — Matching: HCR cells → Panel-Resolution Reference
# =============================================================================


def centroid_correlation_matching(
    query_cells: np.ndarray,
    ref_centroids: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign each query cell to the reference centroid with highest Pearson correlation.

    Parameters
    ----------
    query_cells : ndarray, shape (n_cells, n_genes)
        Z-scored query cell profiles.
    ref_centroids : DataFrame, shape (n_centroids, n_genes)
        Z-scored reference centroid profiles. Index = label names.

    Returns
    -------
    assignments : ndarray of str, shape (n_cells,)
        Assigned label per cell.
    confidences : ndarray of float, shape (n_cells,)
        Max Pearson correlation per cell.
    corr_matrix : ndarray, shape (n_cells, n_centroids)
        Full correlation matrix.
    """
    # Compute Pearson correlation: cells × centroids
    # Pearson r between two z-scored vectors = dot product / n
    # But for robustness, use numpy corrcoef on already z-scored data
    C = ref_centroids.values  # (n_centroids, n_genes)
    Q = query_cells  # (n_cells, n_genes)

    # Standardize each row (cell/centroid) to have mean=0, std=1 for Pearson
    def _row_standardize(X):
        m = X.mean(axis=1, keepdims=True)
        s = X.std(axis=1, keepdims=True)
        s[s < 1e-12] = 1.0
        return (X - m) / s

    C_std = _row_standardize(C.astype(np.float64))
    Q_std = _row_standardize(Q.astype(np.float64))

    # corr_matrix[i, j] = Pearson(cell_i, centroid_j)
    n_genes = C_std.shape[1]
    corr_matrix = (Q_std @ C_std.T) / n_genes  # (n_cells, n_centroids)

    # Assign to max
    best_idx = np.argmax(corr_matrix, axis=1)
    labels = np.array(ref_centroids.index)
    assignments = labels[best_idx]
    confidences = corr_matrix[np.arange(len(best_idx)), best_idx]

    return assignments, confidences, corr_matrix


def score_marker_sets(
    X_z: np.ndarray,
    gene_names: list[str],
    marker_sets: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Compute marker-set scores per cell (M.2 cross-check).

    For each marker set, average the z-scored expression of member genes.
    """
    if marker_sets is None:
        marker_sets = {
            "Pvalb": ["Pvalb"],
            "Sst": ["Sst"],
            "Vip": ["Vip"],
            "Lamp5": ["Lamp5"],
        }

    gene_idx_map = {g: i for i, g in enumerate(gene_names)}
    scores = {}
    for name, genes in marker_sets.items():
        valid_idx = [gene_idx_map[g] for g in genes if g in gene_idx_map]
        if valid_idx:
            scores[name] = X_z[:, valid_idx].mean(axis=1)
        else:
            scores[name] = np.zeros(X_z.shape[0])

    return pd.DataFrame(scores)


def summarize_label_mapping(
    assignments_df: pd.DataFrame,
    label_col: str,
    out_dir: Path,
    prefix: str,
    low_conf_threshold: float = 0.3,
) -> None:
    """
    Summarize how source labels map to assigned Tasic-derived clusters.

    Saves count/fraction tables, per-label quality metrics, and heatmap plots.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    required_cols = {label_col, "assignment", "confidence"}
    missing_cols = required_cols - set(assignments_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns for mapping summary: {sorted(missing_cols)}")

    counts = pd.crosstab(assignments_df[label_col], assignments_df["assignment"])
    counts.to_csv(out_dir / f"{prefix}_counts.csv")

    row_frac = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    col_frac = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    row_frac.to_csv(out_dir / f"{prefix}_row_fraction.csv")
    col_frac.to_csv(out_dir / f"{prefix}_col_fraction.csv")

    # Top mappings per source label.
    top_rows = []
    for src_label in row_frac.index:
        row = row_frac.loc[src_label].sort_values(ascending=False)
        for rank, (dst_label, frac) in enumerate(row.head(3).items(), start=1):
            top_rows.append(
                {
                    label_col: src_label,
                    "rank": rank,
                    "assignment": dst_label,
                    "fraction": float(frac),
                    "count": int(counts.loc[src_label, dst_label]),
                }
            )
    pd.DataFrame(top_rows).to_csv(out_dir / f"{prefix}_top3_mappings.csv", index=False)

    # Per-source label quality metrics.
    metrics_rows = []
    for src_label, df_sub in assignments_df.groupby(label_col, observed=False):
        n = len(df_sub)
        cnt = df_sub["assignment"].value_counts()
        top_assignment = cnt.index[0] if len(cnt) > 0 else "NA"
        top_frac = float(cnt.iloc[0] / n) if n > 0 else 0.0
        probs = (cnt / n).values if n > 0 else np.array([0.0])
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        metrics_rows.append(
            {
                label_col: src_label,
                "n_cells": int(n),
                "mean_confidence": float(df_sub["confidence"].mean()),
                "median_confidence": float(df_sub["confidence"].median()),
                "low_conf_fraction": float((df_sub["confidence"] < low_conf_threshold).mean()),
                "top_assignment": top_assignment,
                "top_assignment_fraction": top_frac,
                "assignment_entropy": entropy,
            }
        )
    metrics_df = pd.DataFrame(metrics_rows).sort_values("n_cells", ascending=False)
    metrics_df.to_csv(out_dir / f"{prefix}_quality_by_{label_col}.csv", index=False)

    def _infer_subclass(label: str) -> str:
        s = str(label).lower()
        for sub in ["Lamp5", "Vip", "Sst", "Pvalb", "Sncg", "Serpinf1", "Meis2"]:
            if sub.lower() in s:
                return sub
        return "Other"

    subclass_order = ["Lamp5", "Vip", "Sst", "Pvalb", "Sncg", "Serpinf1", "Meis2", "Other"]
    subclass_rank = {k: i for i, k in enumerate(subclass_order)}
    subclass_colors = {
        "Lamp5": "#4DAF4A",
        "Vip": "#984EA3",
        "Sst": "#E41A1C",
        "Pvalb": "#377EB8",
        "Sncg": "#FF7F00",
        "Serpinf1": "#A65628",
        "Meis2": "#666666",
        "Other": "#333333",
    }

    def _sort_labels(labels: list[str], score_map: dict[str, float]) -> list[str]:
        return sorted(
            labels,
            key=lambda x: (
                subclass_rank.get(_infer_subclass(x), len(subclass_rank)),
                -float(score_map.get(x, 0.0)),
                str(x),
            ),
        )

    def _group_boundaries(labels: list[str]) -> list[int]:
        if not labels:
            return []
        groups = [_infer_subclass(x) for x in labels]
        bounds = []
        for i in range(1, len(groups)):
            if groups[i] != groups[i - 1]:
                bounds.append(i)
        return bounds

    def _style_axis(ax, row_labels: list[str], col_labels: list[str]) -> None:
        # Color tick labels by inferred subclass for fast visual matching.
        for tick in ax.get_yticklabels():
            sub = _infer_subclass(tick.get_text())
            tick.set_color(subclass_colors.get(sub, "#333333"))
        for tick in ax.get_xticklabels():
            sub = _infer_subclass(tick.get_text())
            tick.set_color(subclass_colors.get(sub, "#333333"))

        # Draw separators where inferred subclass changes.
        for b in _group_boundaries(row_labels):
            ax.axhline(b, color="white", lw=1.2)
        for b in _group_boundaries(col_labels):
            ax.axvline(b, color="white", lw=1.2)

    # Heatmaps for top labels/clusters to keep figure readable.
    top_src = counts.sum(axis=1).sort_values(ascending=False).head(40).index
    top_dst = counts.sum(axis=0).sort_values(ascending=False).head(30).index

    src_scores = counts.sum(axis=1).to_dict()
    dst_scores = counts.sum(axis=0).to_dict()
    row_order = _sort_labels([str(x) for x in top_src.tolist()], src_scores)
    col_order = _sort_labels([str(x) for x in top_dst.tolist()], dst_scores)

    row_view = row_frac.loc[row_order, col_order]
    count_view = counts.loc[row_order, col_order]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.25 * len(top_src))))
    sns.heatmap(
        row_view,
        ax=axes[0],
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Row-normalized fraction"},
    )
    axes[0].set_title(f"{prefix}: {label_col} -> assignment (row fraction, subclass-grouped)")
    axes[0].tick_params(axis="x", rotation=90, labelsize=7)
    axes[0].tick_params(axis="y", rotation=0, labelsize=7)
    _style_axis(axes[0], row_order, col_order)

    sns.heatmap(
        np.log10(count_view + 1),
        ax=axes[1],
        cmap="magma",
        cbar_kws={"label": "log10(count + 1)"},
    )
    axes[1].set_title(f"{prefix}: {label_col} -> assignment (counts, subclass-grouped)")
    axes[1].tick_params(axis="x", rotation=90, labelsize=7)
    axes[1].tick_params(axis="y", rotation=0, labelsize=7)
    _style_axis(axes[1], row_order, col_order)

    # Compact legend for subclass colors.
    legend_order = [s for s in subclass_order if s in {_infer_subclass(x) for x in row_order + col_order}]
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markerfacecolor=subclass_colors[s],
                   markeredgecolor=subclass_colors[s], markersize=7, label=s)
        for s in legend_order
    ]
    if handles:
        axes[1].legend(handles=handles, title="Inferred subclass", loc="upper left",
                       bbox_to_anchor=(1.02, 1.0), fontsize=7, title_fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / f"{prefix}_mapping_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_merge_overall_report(
    assignments_df: pd.DataFrame,
    label_col: str,
    out_dir: Path,
    prefix: str,
) -> None:
    """
    Build destination-centric merge reports.

    These tables answer: for each assigned Tasic cluster, which source labels
    (e.g. 10x supertypes) flow into it, and how concentrated that merge is.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    required_cols = {label_col, "assignment", "confidence"}
    missing_cols = required_cols - set(assignments_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns for merge report: {sorted(missing_cols)}")

    counts = pd.crosstab(assignments_df[label_col], assignments_df["assignment"])
    if counts.empty:
        return

    # Metrics from both perspectives.
    # - within_cluster_frac: source label contribution within a destination cluster.
    # - within_label_frac: fraction of a source label assigned into a destination cluster.
    within_cluster = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    within_label = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    ranked_rows = []
    summary_rows = []

    for assignment in counts.columns:
        col_counts = counts[assignment]
        total = int(col_counts.sum())
        nonzero = col_counts[col_counts > 0].sort_values(ascending=False)

        if total == 0:
            continue

        probs = (nonzero / total).values
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        top3_frac = float((nonzero.head(3).sum()) / total)
        dominant_label = str(nonzero.index[0])
        dominant_frac = float(nonzero.iloc[0] / total)

        assignment_df = assignments_df[assignments_df["assignment"] == assignment]
        mean_conf = float(assignment_df["confidence"].mean()) if len(assignment_df) > 0 else float("nan")

        summary_rows.append(
            {
                "assignment": assignment,
                "n_cells": total,
                "n_source_labels": int(len(nonzero)),
                "dominant_source_label": dominant_label,
                "dominant_source_fraction": dominant_frac,
                "top3_source_fraction": top3_frac,
                "source_entropy": entropy,
                "mean_confidence": mean_conf,
            }
        )

        for rank, (src_label, cnt) in enumerate(nonzero.items(), start=1):
            ranked_rows.append(
                {
                    "assignment": assignment,
                    "rank": rank,
                    label_col: src_label,
                    "count": int(cnt),
                    "within_cluster_frac": float(within_cluster.loc[src_label, assignment]),
                    "within_label_frac": float(within_label.loc[src_label, assignment]),
                }
            )

    ranked_df = pd.DataFrame(ranked_rows).sort_values(["assignment", "rank"])
    summary_df = pd.DataFrame(summary_rows).sort_values("n_cells", ascending=False)

    ranked_df.to_csv(out_dir / f"{prefix}_ranked_sources_per_assignment.csv", index=False)
    summary_df.to_csv(out_dir / f"{prefix}_assignment_merge_summary.csv", index=False)

    # Compact view: top 5 source labels per destination cluster.
    top5 = ranked_df[ranked_df["rank"] <= 5].copy()
    top5.to_csv(out_dir / f"{prefix}_top5_sources_per_assignment.csv", index=False)

    # Plot: dominant source fraction vs size for each destination cluster.
    fig, ax = plt.subplots(figsize=(8, 5))
    x = summary_df["n_cells"].values
    y = summary_df["dominant_source_fraction"].values
    ax.scatter(x, y, s=40, alpha=0.8)
    for _, row in summary_df.iterrows():
        ax.text(row["n_cells"], row["dominant_source_fraction"], str(row["assignment"]), fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Assigned cluster size (log scale)")
    ax.set_ylabel("Dominant source fraction")
    ax.set_title(f"{prefix}: merge concentration by assigned cluster")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"{prefix}_merge_concentration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_stage5_10x_hmb(
    tenx_z: ad.AnnData,
    tasic_z: ad.AnnData,
    centroids_collapsed: pd.DataFrame,
    branch_results: dict,
    tenx_gating: pd.DataFrame,
    branch_marker_names: dict,
    out_dir: Path,
    label_col: str = "supertype",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Match 10x-HMB query cells into Tasic-derived cluster references.

    Outputs:
    - Approach A assignments (collapsed Tasic centroids)
    - Approach C assignments (branch-wise Leiden-named clusters)
    - Supertype-to-assignment mapping summaries and quality metrics
    """
    print("\n" + "=" * 60)
    print("STAGE 5-10X: Matching — 10x-HMB Cells → Tasic Reference")
    print("=" * 60)

    stage5_dir = out_dir / "stage5_10x_hmb"
    stage5_dir.mkdir(parents=True, exist_ok=True)

    genes = list(tasic_z.var_names)
    X_q = tenx_z.X if not hasattr(tenx_z.X, "toarray") else tenx_z.X.toarray()
    X_q = np.asarray(X_q, dtype=np.float64)

    if not set(genes).issubset(set(tenx_z.var_names)):
        raise ValueError("10x query genes are not aligned with Tasic genes.")

    # Reorder query matrix to Tasic gene order.
    tenx_aligned = tenx_z[:, genes].copy()
    X_q = tenx_aligned.X if not hasattr(tenx_aligned.X, "toarray") else tenx_aligned.X.toarray()
    X_q = np.asarray(X_q, dtype=np.float64)

    # Approach A
    print("\n[5x.1] Approach A matching (collapsed centroids)...")
    centroids_a = centroids_collapsed[genes]
    assignments_a, confidences_a, _ = centroid_correlation_matching(X_q, centroids_a)

    tenx_assignments_a = pd.DataFrame(
        {
            "cell_id": tenx_aligned.obs_names,
            "assignment": assignments_a,
            "confidence": confidences_a,
            label_col: tenx_aligned.obs[label_col].values,
            "mouse_id": tenx_aligned.obs["mouse_id"].values,
        }
    )

    print(f"  Mean confidence: {confidences_a.mean():.3f} ± {confidences_a.std():.3f}")
    print(f"  Low-confidence (r<0.3): {(confidences_a < 0.3).sum()} ({(confidences_a < 0.3).mean():.1%})")

    # Approach C
    tenx_assignments_c = None
    if branch_results:
        print("\n[5x.2] Approach C matching (branch-wise Leiden-named)...")
        c_rows = []
        for branch in CANONICAL_BRANCHES:
            if branch not in branch_results:
                continue
            adata_branch, res_dict = branch_results[branch]
            if res_dict.get("skip"):
                continue

            X_br = adata_branch.X if not hasattr(adata_branch.X, "toarray") else adata_branch.X.toarray()
            df_br = pd.DataFrame(X_br, columns=adata_branch.var_names)
            df_br["_cluster"] = adata_branch.obs["branch_cluster"].values
            branch_centroids = df_br.groupby("_cluster", observed=True)[genes].mean()
            name_map = branch_marker_names.get(branch, {})

            q_cell_ids = tenx_gating[tenx_gating["assigned_branch"] == branch]["cell_id"].values
            q_mask = tenx_aligned.obs_names.isin(q_cell_ids)
            X_branch_q = X_q[q_mask]
            if X_branch_q.shape[0] == 0:
                continue

            br_assignments, br_confidences, _ = centroid_correlation_matching(X_branch_q, branch_centroids)
            for j, cell_id in enumerate(tenx_aligned.obs_names[q_mask]):
                raw_label = br_assignments[j]
                named_label = name_map.get(raw_label, raw_label)
                c_rows.append(
                    {
                        "cell_id": cell_id,
                        "branch": branch,
                        "leiden_cluster": raw_label,
                        "assignment": named_label,
                        "confidence": br_confidences[j],
                        label_col: tenx_aligned.obs.loc[cell_id, label_col],
                        "mouse_id": "10x-hmb",
                    }
                )

            print(
                f"    {branch}: {X_branch_q.shape[0]} cells -> "
                f"{len(set(br_assignments))} clusters, mean r={br_confidences.mean():.3f}"
            )

        if c_rows:
            tenx_assignments_c = pd.DataFrame(c_rows)

    # Save assignments
    tenx_assignments_a.to_csv(stage5_dir / "tenx_assignments_approach_a.csv", index=False)
    if tenx_assignments_c is not None:
        tenx_assignments_c.to_csv(stage5_dir / "tenx_assignments_leiden_named.csv", index=False)

    # Mapping summaries
    print("\n[5x.3] Supertype-to-cluster mapping summaries...")
    summarize_label_mapping(
        tenx_assignments_a,
        label_col=label_col,
        out_dir=stage5_dir,
        prefix="approach_a_supertype_to_tasic",
    )
    build_merge_overall_report(
        tenx_assignments_a,
        label_col=label_col,
        out_dir=stage5_dir,
        prefix="approach_a_merge_overall",
    )
    if tenx_assignments_c is not None:
        summarize_label_mapping(
            tenx_assignments_c,
            label_col=label_col,
            out_dir=stage5_dir,
            prefix="approach_c_supertype_to_tasic",
        )
        build_merge_overall_report(
            tenx_assignments_c,
            label_col=label_col,
            out_dir=stage5_dir,
            prefix="approach_c_merge_overall",
        )

    # Gating-level supertype -> branch map
    gating_df = tenx_gating.copy()
    label_lookup = tenx_aligned.obs[[label_col]].copy()
    label_lookup["cell_id"] = label_lookup.index.astype(str)
    label_lookup = label_lookup[["cell_id", label_col]]
    gating_df = gating_df.merge(
        label_lookup,
        on="cell_id",
        how="left",
    )
    gating_df.to_csv(stage5_dir / "tenx_gating_with_labels.csv", index=False)
    branch_ct = pd.crosstab(gating_df[label_col], gating_df["assigned_branch"])
    branch_ct.to_csv(stage5_dir / "supertype_to_branch_counts.csv")
    branch_frac = branch_ct.div(branch_ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    branch_frac.to_csv(stage5_dir / "supertype_to_branch_row_fraction.csv")

    fig, ax = plt.subplots(figsize=(10, max(6, 0.25 * len(branch_frac))))
    sns.heatmap(branch_frac, cmap="Blues", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "Row-normalized fraction"})
    ax.set_title("10x-HMB supertype -> gated branch")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()
    fig.savefig(stage5_dir / "supertype_to_branch_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_10x_cells": int(tenx_aligned.n_obs),
        "n_genes": int(tenx_aligned.n_vars),
        "approach_a_mean_confidence": float(tenx_assignments_a["confidence"].mean()),
        "approach_a_low_conf_fraction": float((tenx_assignments_a["confidence"] < 0.3).mean()),
        "approach_a_n_unique_labels": int(tenx_assignments_a["assignment"].nunique()),
    }
    if tenx_assignments_c is not None:
        summary["approach_c_n_cells"] = int(len(tenx_assignments_c))
        summary["approach_c_mean_confidence"] = float(tenx_assignments_c["confidence"].mean())
        summary["approach_c_n_unique_labels"] = int(tenx_assignments_c["assignment"].nunique())
    pd.Series(summary).to_csv(stage5_dir / "tenx_matching_summary.csv", header=["value"])

    # Matching quality tri-panel diagnostic
    print("\n[5x.4] Generating matching quality diagnostics...")
    _plot_10x_matching_diagnostics(
        tenx_assignments=tenx_assignments_c if tenx_assignments_c is not None else tenx_assignments_a,
        tenx_z=tenx_aligned,
        tasic_z=tasic_z,
        branch_results=branch_results,
        branch_marker_names=branch_marker_names,
        label_col=label_col,
        genes=genes,
        out_dir=stage5_dir,
    )

    print(f"\n  Stage 5-10x complete. Outputs in {stage5_dir}")
    return tenx_assignments_a, tenx_assignments_c


def _plot_10x_matching_diagnostics(
    tenx_assignments: pd.DataFrame,
    tenx_z: ad.AnnData,
    tasic_z: ad.AnnData,
    branch_results: dict,
    branch_marker_names: dict,
    label_col: str,
    genes: list[str],
    out_dir: Path,
    n_worst: int = 20,
) -> None:
    """
    Tri-panel matching quality diagnostic for 10x-HMB:
      Panel A: supertype confidence vs assignment fragmentation scatter
      Panel B: gene residual heatmap for worst-matching supertype×cluster links
      Panel C: per-supertype confidence CDF by subclass
    """
    subclass_colors = {
        "Lamp5": "#4DAF4A", "Vip": "#984EA3", "Sst": "#E41A1C",
        "Pvalb": "#377EB8", "Sncg": "#FF7F00", "Serpinf1": "#A65628",
        "Meis2": "#666666", "Other": "#333333",
    }
    subclass_order = ["Lamp5", "Vip", "Sst", "Pvalb", "Sncg", "Serpinf1", "Meis2", "Other"]

    def _infer_sub(label: str) -> str:
        s = str(label).lower()
        for sub in subclass_order:
            if sub.lower() in s:
                return sub
        return "Other"

    required = {label_col, "assignment", "confidence"}
    if not required.issubset(tenx_assignments.columns):
        print("  Skipping matching diagnostics (missing columns).")
        return

    # ------------------------------------------------------------------ #
    # Pre-compute per-supertype stats
    # ------------------------------------------------------------------ #
    rows = []
    for src, grp in tenx_assignments.groupby(label_col, observed=False):
        n = len(grp)
        cnt = grp["assignment"].value_counts()
        probs = (cnt / n).values
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        rows.append({
            label_col: str(src),
            "n_cells": n,
            "mean_conf": float(grp["confidence"].mean()),
            "entropy": entropy,
            "top_frac": float(probs[0]) if len(probs) else 0.0,
            "subclass": _infer_sub(str(src)),
            "top_assignment": str(cnt.index[0]) if len(cnt) else "NA",
        })
    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return

    # ------------------------------------------------------------------ #
    # Build per-assignment centroids from Tasic branch results
    # ------------------------------------------------------------------ #
    centroid_rows = []
    for branch in CANONICAL_BRANCHES:
        if branch not in branch_results:
            continue
        adata_br, res_dict = branch_results[branch]
        if res_dict.get("skip"):
            continue
        name_map = branch_marker_names.get(branch, {})
        X_br = adata_br.X if not hasattr(adata_br.X, "toarray") else adata_br.X.toarray()
        df_br = pd.DataFrame(X_br, columns=adata_br.var_names, index=adata_br.obs_names)
        df_br["_named"] = [name_map.get(cl, cl) for cl in adata_br.obs["branch_cluster"].values]
        for named_lbl, g in df_br.groupby("_named", observed=True)[genes].mean().iterrows():
            centroid_rows.append({"assignment": named_lbl, **g.to_dict()})
    tasic_centroids = pd.DataFrame(centroid_rows).set_index("assignment") if centroid_rows else pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Gene residuals: per supertype mean(10x z) - matched centroid z
    # ------------------------------------------------------------------ #
    X_q = tenx_z.X if not hasattr(tenx_z.X, "toarray") else tenx_z.X.toarray()
    X_q = np.asarray(X_q, dtype=np.float64)
    cell_df = pd.DataFrame(X_q, columns=tenx_z.var_names, index=tenx_z.obs_names)

    residual_rows = []
    for src, grp in tenx_assignments.groupby(label_col, observed=False):
        cells = grp["cell_id"].values
        cells_present = [c for c in cells if c in cell_df.index]
        if not cells_present:
            continue
        x_mean = cell_df.loc[cells_present, genes].mean()
        top_assign = grp["assignment"].value_counts().index[0]
        mean_conf = float(grp["confidence"].mean())
        if tasic_centroids is not None and top_assign in tasic_centroids.index:
            centroid = tasic_centroids.loc[top_assign, genes]
            residual = (x_mean - centroid).values
        else:
            residual = x_mean.values * 0.0
        residual_rows.append({
            "label": str(src),
            "top_assignment": top_assign,
            "mean_conf": mean_conf,
            "subclass": _infer_sub(str(src)),
            **{f"res_{g}": float(v) for g, v in zip(genes, residual)},
        })
    resid_df = pd.DataFrame(residual_rows)

    if not resid_df.empty:
        resid_df = resid_df.sort_values("mean_conf")
        worst_n = resid_df.head(n_worst)
        gene_cols = [c for c in worst_n.columns if c.startswith("res_")]
        resid_matrix = worst_n[gene_cols].values
        row_labels = [f"{r['label']} → {r['top_assignment']} (r={r['mean_conf']:.2f})"
                      for _, r in worst_n.iterrows()]
        col_labels = [g[4:] for g in gene_cols]  # strip "res_"
    else:
        resid_matrix = np.zeros((1, len(genes)))
        row_labels = ["no data"]
        col_labels = genes
        worst_n = pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Per-supertype ranking metrics: profile fidelity + residual spread
    # ------------------------------------------------------------------ #
    ranking_rows = []
    if tasic_centroids is not None and not tasic_centroids.empty:
        for _, row in stats_df.iterrows():
            src = str(row[label_col])
            top_assign = str(row["top_assignment"])
            cells = tenx_assignments.loc[
                tenx_assignments[label_col] == src, "cell_id"
            ].values
            cells_present = [c for c in cells if c in cell_df.index]
            if not cells_present or top_assign not in tasic_centroids.index:
                continue

            mean_expr = cell_df.loc[cells_present, genes].mean().values.astype(np.float64)
            centroid_vec = tasic_centroids.loc[top_assign, genes].values.astype(np.float64)
            cell_matrix = cell_df.loc[cells_present, genes].values.astype(np.float64)

            profile_r = float(np.corrcoef(mean_expr, centroid_vec)[0, 1])
            if np.isnan(profile_r):
                profile_r = 0.0
            residual_mean = mean_expr - centroid_vec
            residual_cell = cell_matrix - centroid_vec[None, :]
            residual_rms = float(np.sqrt(np.mean(residual_mean ** 2)))
            residual_spread = float(np.mean(np.std(residual_cell, axis=0)))

            n_assignments = tenx_assignments.loc[
                tenx_assignments[label_col] == src, "assignment"
            ].nunique()
            max_entropy = np.log2(max(n_assignments, 2))
            norm_entropy = float(row["entropy"] / max_entropy) if max_entropy > 1e-12 else 0.0

            ranking_rows.append(
                {
                    label_col: src,
                    "subclass": row["subclass"],
                    "top_assignment": top_assign,
                    "n_cells": int(row["n_cells"]),
                    "mean_conf": float(row["mean_conf"]),
                    "top_frac": float(row["top_frac"]),
                    "entropy": float(row["entropy"]),
                    "normalized_entropy": norm_entropy,
                    "profile_r": profile_r,
                    "profile_r2": profile_r ** 2,
                    "residual_rms": residual_rms,
                    "residual_spread": residual_spread,
                    **{f"spread_{g}": float(v) for g, v in zip(genes, np.std(residual_cell, axis=0))},
                }
            )

    ranking_df = pd.DataFrame(ranking_rows)
    if not ranking_df.empty:
        def _zscore_series(series: pd.Series, invert: bool = False) -> pd.Series:
            values = series.astype(float)
            std = values.std(ddof=0)
            if std < 1e-12:
                z = pd.Series(np.zeros(len(values)), index=values.index)
            else:
                z = (values - values.mean()) / std
            return -z if invert else z

        ranking_df["overall_score"] = (
            _zscore_series(ranking_df["mean_conf"])
            + _zscore_series(ranking_df["profile_r"])
            + _zscore_series(ranking_df["top_frac"])
            + _zscore_series(ranking_df["normalized_entropy"], invert=True)
            + _zscore_series(ranking_df["residual_spread"], invert=True)
        )
        ranking_df = ranking_df.sort_values(
            ["overall_score", "profile_r", "mean_conf"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        ranking_df["rank"] = np.arange(1, len(ranking_df) + 1)
        ranking_df.to_csv(out_dir / "tenx_supertype_ranking.csv", index=False)

    # ------------------------------------------------------------------ #
    # Figure
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(22, max(10, 0.35 * max(len(row_labels), 20))))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.6, 1.0], wspace=0.45)

    # --- Panel A: confidence vs entropy scatter ---
    ax_a = fig.add_subplot(gs[0])
    for sub in subclass_order:
        sub_df = stats_df[stats_df["subclass"] == sub]
        if sub_df.empty:
            continue
        ax_a.scatter(
            sub_df["mean_conf"], sub_df["entropy"],
            s=sub_df["n_cells"].clip(upper=1000) * 0.05 + 15,
            color=subclass_colors[sub], label=sub, alpha=0.75, edgecolors="none",
        )
    ax_a.set_xlabel("Mean Pearson r (match confidence)", fontsize=9)
    ax_a.set_ylabel("Assignment entropy (fragmentation)", fontsize=9)
    ax_a.set_title("A: Confidence vs Fragmentation", fontsize=10, fontweight="bold")
    ax_a.legend(title="Subclass", fontsize=7, title_fontsize=8,
                loc="upper left", markerscale=1.2)
    ax_a.axvline(0.3, color="red", linestyle="--", lw=0.8, alpha=0.6)
    ax_a.grid(alpha=0.3)

    # Annotate the worst 5 points
    worst5 = stats_df.nsmallest(5, "mean_conf")
    for _, row in worst5.iterrows():
        ax_a.annotate(
            str(row[label_col]),
            (row["mean_conf"], row["entropy"]),
            fontsize=5, ha="left", va="bottom", alpha=0.8,
        )

    # --- Panel B: gene residual heatmap (worst supertypes) ---
    ax_b = fig.add_subplot(gs[1])
    vmax = np.abs(resid_matrix).max() if resid_matrix.size > 0 else 1.0
    vmax = max(vmax, 0.1)
    im = ax_b.imshow(
        resid_matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    ax_b.set_xticks(range(len(col_labels)))
    ax_b.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax_b.set_yticks(range(len(row_labels)))
    ax_b.set_yticklabels(row_labels, fontsize=6)
    ax_b.set_title(
        f"B: Gene residuals for {n_worst} lowest-confidence supertypes\n"
        "(10x mean z − matched Tasic centroid z)",
        fontsize=9, fontweight="bold",
    )
    plt.colorbar(im, ax=ax_b, label="Δz-score", shrink=0.6)
    # Color y tick labels by subclass
    if not resid_df.empty:
        for tick, (_, row) in zip(ax_b.get_yticklabels(), worst_n.iterrows()):
            tick.set_color(subclass_colors.get(row["subclass"], "#333333"))

    # --- Panel C: confidence CDF per subclass ---
    ax_c = fig.add_subplot(gs[2])
    for sub in subclass_order:
        sub_cells = tenx_assignments[
            tenx_assignments[label_col].apply(_infer_sub) == sub
        ]["confidence"].values
        if len(sub_cells) == 0:
            continue
        sorted_conf = np.sort(sub_cells)
        cdf = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)
        ax_c.plot(sorted_conf, cdf, color=subclass_colors[sub], label=sub, lw=1.2)
    ax_c.axvline(0.3, color="red", linestyle="--", lw=0.8, alpha=0.6, label="r=0.3")
    ax_c.set_xlabel("Pearson r", fontsize=9)
    ax_c.set_ylabel("CDF", fontsize=9)
    ax_c.set_title("C: Confidence CDF by subclass", fontsize=10, fontweight="bold")
    ax_c.legend(fontsize=7, title="Subclass", title_fontsize=8)
    ax_c.grid(alpha=0.3)

    plt.suptitle(
        "10x-HMB → Tasic matching quality diagnostics",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "tenx_matching_quality_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: tenx_matching_quality_diagnostics.png")

    # Best / worst cluster profiles
    _plot_tenx_best_worst_clusters(
        stats_df=stats_df,
        tenx_assignments=tenx_assignments,
        cell_df=cell_df,
        tasic_centroids=tasic_centroids,
        genes=genes,
        label_col=label_col,
        subclass_colors=subclass_colors,
        out_dir=out_dir,
    )

    _plot_tenx_supertype_ranking(
        ranking_df=ranking_df,
        genes=genes,
        subclass_order=subclass_order,
        subclass_colors=subclass_colors,
        label_col=label_col,
        out_dir=out_dir,
    )


def _plot_tenx_best_worst_clusters(
    stats_df: pd.DataFrame,
    tenx_assignments: pd.DataFrame,
    cell_df: pd.DataFrame,
    tasic_centroids: pd.DataFrame,
    genes: list[str],
    label_col: str,
    subclass_colors: dict,
    out_dir: Path,
    n_show: int = 3,
) -> None:
    """
    2×n_show grid comparing the best and worst matched 10x supertypes.

    Best  = highest concentration (top_frac * mean_conf): one dominant Tasic cluster.
    Worst = highest entropy / lowest confidence: spread across many clusters.

    Each column is one supertype.  Two sub-rows per supertype:
      Upper: mean z-score bars (10x) + matched centroid line + 2nd-best dashed.
      Lower: signed residual bars (mean − matched centroid), gene by gene.
    """
    if stats_df.empty or tasic_centroids is None or tasic_centroids.empty:
        print("  Skipping best/worst cluster profiles (no centroid data).")
        return

    df = stats_df.copy()
    df["best_score"] = df["top_frac"] * df["mean_conf"]
    df["worst_score"] = df["entropy"] / (df["mean_conf"] + 0.01)

    # Keep only supertypes whose cells are actually present in cell_df.
    def _has_cells(lbl):
        ids = tenx_assignments.loc[tenx_assignments[label_col] == lbl, "cell_id"].values
        return any(c in cell_df.index for c in ids)

    valid = df[df[label_col].apply(_has_cells)]
    if len(valid) < 2:
        print("  Skipping best/worst cluster profiles (too few valid supertypes).")
        return

    best_rows = valid.nlargest(n_show, "best_score").reset_index(drop=True)
    worst_rows = valid.nlargest(n_show, "worst_score").reset_index(drop=True)

    n_genes = len(genes)
    x = np.arange(n_genes)

    fig, axes = plt.subplots(
        4, n_show,
        figsize=(5.5 * n_show, 13),
        gridspec_kw={"height_ratios": [2.5, 0.9, 2.5, 0.9], "hspace": 0.08, "wspace": 0.35},
    )
    if n_show == 1:
        axes = axes.reshape(4, 1)

    def _row_corr(v1, v2):
        s1, s2 = v1.std(), v2.std()
        v1n = (v1 - v1.mean()) / (s1 if s1 > 1e-9 else 1.0)
        v2n = (v2 - v2.mean()) / (s2 if s2 > 1e-9 else 1.0)
        return float(np.dot(v1n, v2n) / n_genes)

    def _draw_col(ax_expr, ax_resid, row, tier_label, col_idx):
        lbl = row[label_col]
        top_assign = str(row["top_assignment"])
        sub = str(row.get("subclass", "Other"))
        color = subclass_colors.get(sub, "#555555")

        # Mean expression for this supertype
        cell_ids = tenx_assignments.loc[
            tenx_assignments[label_col] == lbl, "cell_id"
        ].values
        present = [c for c in cell_ids if c in cell_df.index]
        mean_expr = cell_df.loc[present, genes].mean().values

        # Primary centroid (top assignment)
        c1 = tasic_centroids.loc[top_assign, genes].values \
            if top_assign in tasic_centroids.index else np.zeros(n_genes)

        # 2nd-best centroid: highest Pearson r with supertype mean, excluding top
        corrs = {
            lbl_c: _row_corr(mean_expr, tasic_centroids.loc[lbl_c, genes].values)
            for lbl_c in tasic_centroids.index
        }
        sorted_centroids = sorted(corrs, key=corrs.get, reverse=True)
        second_assign = next(
            (k for k in sorted_centroids if k != top_assign), top_assign
        )
        c2 = tasic_centroids.loc[second_assign, genes].values \
            if second_assign in tasic_centroids.index else c1

        residual = mean_expr - c1
        r1 = corrs.get(top_assign, float("nan"))
        r2 = corrs.get(second_assign, float("nan"))

        # --- Upper panel: expression + centroids ---
        ax_expr.bar(x, mean_expr, width=0.55, color=color, alpha=0.65,
                    label=f"10x mean (n={len(present)})", zorder=2)
        ax_expr.plot(x, c1, "o-", color="black", lw=1.8, ms=4,
                     label=f"▶ {top_assign[:28]}\n   r={r1:.2f}", zorder=3)
        ax_expr.plot(x, c2, "s--", color="#888888", lw=1.2, ms=3, alpha=0.85,
                     label=f"2nd: {second_assign[:26]}\n    r={r2:.2f}", zorder=3)
        ax_expr.axhline(0, color="grey", lw=0.5)
        ax_expr.set_xticks([])
        ax_expr.set_ylabel("z-score", fontsize=7.5)
        entropy = row.get("entropy", float("nan"))
        mean_conf = row.get("mean_conf", float("nan"))
        ax_expr.set_title(
            f"{tier_label} #{col_idx + 1}  [{sub}]\n"
            f"{lbl}\n"
            f"r={mean_conf:.2f}  H={entropy:.2f}",
            fontsize=7.5, fontweight="bold",
        )
        ax_expr.legend(fontsize=5.5, loc="upper right", framealpha=0.7,
                       handlelength=1.2)
        ax_expr.grid(axis="y", alpha=0.25)

        # --- Lower panel: residuals ---
        resid_colors = ["#d73027" if v > 0 else "#4575b4" for v in residual]
        ax_resid.bar(x, residual, width=0.55, color=resid_colors, alpha=0.85)
        ax_resid.axhline(0, color="black", lw=0.7)
        ax_resid.set_xticks(x)
        ax_resid.set_xticklabels(genes, rotation=90, fontsize=6.5)
        ax_resid.set_ylabel("Δz", fontsize=7)
        ax_resid.tick_params(axis="y", labelsize=6)
        ax_resid.grid(axis="y", alpha=0.2)

    for ci, row in best_rows.iterrows():
        _draw_col(axes[0, ci], axes[1, ci], row, "BEST", ci)

    for ci, row in worst_rows.iterrows():
        _draw_col(axes[2, ci], axes[3, ci], row, "WORST", ci)

    # Horizontal divider between best and worst sections
    fig.add_artist(plt.Line2D(
        [0.02, 0.98], [0.505, 0.505],
        transform=fig.transFigure,
        color="#333333", lw=1.5, linestyle="--",
    ))
    fig.text(0.01, 0.75, "Best matched", va="center", rotation=90,
             fontsize=9, fontweight="bold", color="#1a6b1a")
    fig.text(0.01, 0.25, "Worst matched", va="center", rotation=90,
             fontsize=9, fontweight="bold", color="#b30000")

    plt.suptitle(
        "10x-HMB: Best vs worst matched supertypes\n"
        "(mean supertype z-score  |  matched Tasic centroid  |  residuals)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "tenx_best_worst_cluster_profiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: tenx_best_worst_cluster_profiles.png")


def _plot_tenx_supertype_ranking(
    ranking_df: pd.DataFrame,
    genes: list[str],
    subclass_order: list[str],
    subclass_colors: dict,
    label_col: str,
    out_dir: Path,
) -> None:
    """Save a full best-to-worst ordering of 10x supertypes with residual spread."""
    if ranking_df.empty:
        print("  Skipping full supertype ranking (no ranking data).")
        return

    n_rows = len(ranking_df)
    height = max(9, 0.24 * n_rows)
    fig = plt.figure(figsize=(18, height))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.75], wspace=0.08)

    ax_left = fig.add_subplot(gs[0])
    y = np.arange(n_rows)
    colors = [subclass_colors.get(sub, "#333333") for sub in ranking_df["subclass"]]
    sizes = 18 + 90 * np.sqrt(ranking_df["n_cells"].clip(lower=1) / ranking_df["n_cells"].max())

    ax_left.hlines(y=y, xmin=0, xmax=ranking_df["overall_score"], color=colors, alpha=0.35, linewidth=1.1)
    ax_left.scatter(
        ranking_df["overall_score"],
        y,
        s=sizes,
        c=colors,
        edgecolors="black",
        linewidths=0.35,
        zorder=3,
    )
    ax_left.axvline(0, color="#666666", linestyle="--", lw=0.8)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(ranking_df[label_col], fontsize=6.5)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("Overall match score", fontsize=9)
    ax_left.set_title(
        "A: Full ordering of supertypes\n"
        "score = fit + specificity - entropy - residual spread",
        fontsize=10,
        fontweight="bold",
    )
    ax_left.grid(axis="x", alpha=0.25)

    ax_right = fig.add_subplot(gs[1])
    spread_cols = [f"spread_{g}" for g in genes if f"spread_{g}" in ranking_df.columns]
    spread_matrix = ranking_df[spread_cols].values if spread_cols else np.zeros((n_rows, len(genes)))
    vmax = max(float(np.nanmax(spread_matrix)) if spread_matrix.size else 0.0, 0.1)
    im = ax_right.imshow(spread_matrix, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    ax_right.set_yticks(y)
    ax_right.set_yticklabels([])
    ax_right.set_xticks(range(len(genes)))
    ax_right.set_xticklabels(genes, rotation=90, fontsize=7)
    ax_right.set_title(
        "B: Residual spread by gene\n"
        "mean per-gene SD of (cell z - assigned centroid z)",
        fontsize=10,
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax_right, label="Residual spread", shrink=0.6)

    for tick, (_, row) in zip(ax_left.get_yticklabels(), ranking_df.iterrows()):
        tick.set_color(subclass_colors.get(row["subclass"], "#333333"))

    for ax in [ax_left, ax_right]:
        boundaries = []
        current = 0
        for sub in subclass_order:
            sub_n = int((ranking_df["subclass"] == sub).sum())
            if sub_n > 0:
                current += sub_n
                if current < n_rows:
                    boundaries.append(current - 0.5)
        for boundary in boundaries:
            ax.axhline(boundary, color="white", lw=1.0, alpha=0.9)

    top5 = ranking_df.head(5)
    bottom5 = ranking_df.tail(5)
    left_notes = []
    for _, row in top5.iterrows():
        left_notes.append(
            f"{int(row['rank'])}. {row[label_col]} | r={row['profile_r']:.2f} | spread={row['residual_spread']:.2f}"
        )
    right_notes = []
    for _, row in bottom5.iterrows():
        right_notes.append(
            f"{int(row['rank'])}. {row[label_col]} | r={row['profile_r']:.2f} | spread={row['residual_spread']:.2f}"
        )

    fig.text(0.11, 0.012, "Best 5: " + "   ".join(left_notes), fontsize=7)
    fig.text(0.11, 0.001, "Worst 5: " + "   ".join(right_notes), fontsize=7)
    plt.suptitle(
        "10x-HMB supertype ranking: best to worst matched",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.985])
    fig.savefig(out_dir / "tenx_supertype_ranking.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: tenx_supertype_ranking.png")


def plot_stage5_diagnostics(
    hcr_assignments_a: pd.DataFrame,
    hcr_assignments_c: pd.DataFrame | None,
    marker_scores: pd.DataFrame,
    corr_matrix_a: np.ndarray,
    ref_labels_a: list[str],
    out_dir: Path,
) -> None:
    """Save Stage 5 diagnostic plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Confidence distribution (Approach A)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.hist(hcr_assignments_a["confidence"], bins=50, alpha=0.7, color="#377EB8")
    ax.axvline(0.3, color="red", linestyle="--", alpha=0.6, label="low-conf cutoff")
    ax.set_xlabel("Max Pearson correlation")
    ax.set_ylabel("Cell count")
    ax.set_title("Approach A: matching confidence")
    ax.legend()

    # Plot 2: Assignment counts (Approach A) — top 20
    ax = axes[1]
    counts = hcr_assignments_a["assignment"].value_counts().head(20)
    ax.barh(range(len(counts)), counts.values, color="#377EB8", alpha=0.8)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=7)
    ax.set_xlabel("Cell count")
    ax.set_title("Approach A: top 20 assignments")
    ax.invert_yaxis()

    # Plot 3: Marker score agreement
    ax = axes[2]
    # For each cell, check if assigned subclass matches highest marker score
    assigned_subclass = hcr_assignments_a["assignment"].apply(
        lambda x: x.split()[0] if " " in x else x.split("-")[0] if "-" in x else x
    )
    # Only check canonical branches
    canonical_mask = assigned_subclass.isin(CANONICAL_BRANCHES)
    if canonical_mask.any():
        marker_max = marker_scores.loc[canonical_mask.values].idxmax(axis=1)
        agreement = (assigned_subclass[canonical_mask].values == marker_max.values).mean()
        ax.bar(["Agreement", "Disagreement"], [agreement, 1 - agreement],
               color=["#4DAF4A", "#E41A1C"], alpha=0.8)
        ax.set_ylabel("Fraction")
        ax.set_title(f"Marker-score cross-check\n(agreement={agreement:.1%})")
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "No canonical assignments", ha="center", va="center")
        ax.set_title("Marker-score cross-check")

    plt.tight_layout()
    fig.savefig(out_dir / "stage5_01_matching_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 4: Correspondence heatmap (correlation matrix collapsed to subclass level)
    fig, ax = plt.subplots(figsize=(14, 8))
    # Show mean correlation per assigned label × reference centroid
    corr_df = pd.DataFrame(corr_matrix_a, columns=ref_labels_a)
    corr_df["assignment"] = hcr_assignments_a["assignment"].values
    mean_corr = corr_df.groupby("assignment").mean()
    # Keep top 20 assignments by count
    top_labels = hcr_assignments_a["assignment"].value_counts().head(20).index
    mean_corr_top = mean_corr.loc[mean_corr.index.isin(top_labels)]
    if len(mean_corr_top) > 0:
        sns.heatmap(mean_corr_top, cmap="RdBu_r", center=0, ax=ax,
                    cbar_kws={"label": "Mean Pearson r"})
        ax.set_title("Correspondence: HCR assignments × reference centroids")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "stage5_02_correspondence_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot 5: Approach C branch-by-branch confidence (if available)
    if hcr_assignments_c is not None and len(hcr_assignments_c) > 0:
        branches_present = [b for b in CANONICAL_BRANCHES if b in hcr_assignments_c["branch"].values]
        if branches_present:
            fig, axes = plt.subplots(1, len(branches_present),
                                     figsize=(4.5 * len(branches_present), 4))
            if len(branches_present) == 1:
                axes = [axes]
            for i, branch in enumerate(branches_present):
                ax = axes[i]
                branch_data = hcr_assignments_c[hcr_assignments_c["branch"] == branch]
                ax.hist(branch_data["confidence"], bins=30, alpha=0.7,
                        color={"Pvalb": "#377EB8", "Sst": "#E41A1C",
                               "Vip": "#984EA3", "Lamp5": "#4DAF4A"}.get(branch, "#555"))
                ax.set_xlabel("Pearson r")
                ax.set_title(f"{branch} ({len(branch_data)} cells)")
                ax.set_ylabel("Count")
            plt.suptitle("Approach C: branch-by-branch confidence", fontsize=11)
            plt.tight_layout()
            fig.savefig(out_dir / "stage5_03_branchwise_confidence.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

    print(f"  Saved: stage5_01-03 diagnostic plots")


def run_stage5(
    hcr_corrected: ad.AnnData,
    tasic_z: ad.AnnData,
    centroids_collapsed: pd.DataFrame,
    branch_results: dict,
    hcr_gating: pd.DataFrame,
    branch_marker_names: dict,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Execute Stage 5: Match HCR cells to panel-resolution reference.

    Three parallel outputs:
    - Approach A: match all HCR cells to collapsed Tasic centroids (from Stage 3).
    - Approach C (Tasic labels): match branch-by-branch to within-branch centroids.
    - Approach C (Leiden-named, preferred): same as C but using marker-derived names.

    Parameters
    ----------
    hcr_corrected : AnnData
        Batch-corrected z-scored HCR cells.
    tasic_z : AnnData
        Z-scored Tasic inhibitory cells.
    centroids_collapsed : DataFrame
        Collapsed centroids from Stage 3 (Approach A reference).
    branch_results : dict
        {branch: (adata_branch, res_dict)} from Stage 4.
    hcr_gating : DataFrame
        HCR subclass gating from Stage 4.
    branch_marker_names : dict
        {branch: {branch_cluster_id: "Branch-N: gene1 | gene2"}} from Stage 4.
    out_dir : Path
        Output directory.

    Returns
    -------
    hcr_assignments_a : DataFrame
        Per-cell assignments from Approach A.
    hcr_assignments_c : DataFrame or None
        Per-cell assignments from Approach C (Leiden-named, preferred).
    """
    print("\n" + "=" * 60)
    print("STAGE 5: Matching — HCR Cells → Panel-Resolution Reference")
    print("=" * 60)

    stage5_dir = out_dir / "stage5"
    stage5_dir.mkdir(parents=True, exist_ok=True)

    genes = list(hcr_corrected.var_names)
    X_hcr = hcr_corrected.X if not hasattr(hcr_corrected.X, "toarray") else hcr_corrected.X.toarray()
    X_hcr = np.asarray(X_hcr, dtype=np.float64)

    # -------------------------------------------------------------------------
    # 5.1 Approach A — match all HCR cells to collapsed centroids
    # -------------------------------------------------------------------------
    print(f"\n[5.1] Approach A: centroid correlation matching...")
    print(f"  Reference: {len(centroids_collapsed)} collapsed centroids")
    print(f"  Query: {X_hcr.shape[0]} HCR cells, {X_hcr.shape[1]} genes")

    # Ensure gene order matches
    centroids_a = centroids_collapsed[genes]

    assignments_a, confidences_a, corr_matrix_a = centroid_correlation_matching(
        X_hcr, centroids_a
    )

    hcr_assignments_a = pd.DataFrame({
        "cell_id": hcr_corrected.obs_names,
        "assignment": assignments_a,
        "confidence": confidences_a,
        "mouse_id": hcr_corrected.obs["mouse_id"].values,
    })

    # Report
    print(f"\n  Approach A results:")
    print(f"    Mean confidence: {confidences_a.mean():.3f} ± {confidences_a.std():.3f}")
    print(f"    Low-confidence (r<0.3): {(confidences_a < 0.3).sum()} "
          f"({(confidences_a < 0.3).mean():.1%})")
    print(f"    Unique assignments: {len(set(assignments_a))}")
    print(f"\n    Top 10 assignments:")
    for label, count in hcr_assignments_a["assignment"].value_counts().head(10).items():
        frac = count / len(hcr_assignments_a)
        print(f"      {label}: {count} ({frac:.1%})")

    # -------------------------------------------------------------------------
    # 5.2 Approach C — branch-by-branch matching (Leiden-named, preferred)
    # -------------------------------------------------------------------------
    hcr_assignments_c = None
    if branch_results:
        print(f"\n[5.2] Approach C: branch-by-branch matching (Leiden-named)...")

        c_rows = []
        for branch in CANONICAL_BRANCHES:
            if branch not in branch_results:
                continue

            adata_branch, res_dict = branch_results[branch]
            if res_dict.get("skip"):
                continue

            # Compute reference centroids for this branch's Leiden clusters
            X_br = adata_branch.X if not hasattr(adata_branch.X, "toarray") else adata_branch.X.toarray()
            df_br = pd.DataFrame(X_br, columns=adata_branch.var_names)
            df_br["_cluster"] = adata_branch.obs["branch_cluster"].values
            branch_centroids = df_br.groupby("_cluster", observed=True)[genes].mean()

            # Get marker-derived name mapping for this branch
            name_map = branch_marker_names.get(branch, {})

            # Select HCR cells gated to this branch
            branch_cell_ids = hcr_gating[
                hcr_gating["assigned_branch"] == branch
            ]["cell_id"].values
            branch_mask = hcr_corrected.obs_names.isin(branch_cell_ids)
            X_branch_hcr = X_hcr[branch_mask]

            if X_branch_hcr.shape[0] == 0:
                continue

            # Match within branch
            br_assignments, br_confidences, _ = centroid_correlation_matching(
                X_branch_hcr, branch_centroids
            )

            for j, cell_id in enumerate(hcr_corrected.obs_names[branch_mask]):
                raw_label = br_assignments[j]
                named_label = name_map.get(raw_label, raw_label)
                c_rows.append({
                    "cell_id": cell_id,
                    "branch": branch,
                    "leiden_cluster": raw_label,
                    "assignment": named_label,
                    "confidence": br_confidences[j],
                    "mouse_id": hcr_corrected.obs.loc[cell_id, "mouse_id"],
                })

            print(f"    {branch}: {X_branch_hcr.shape[0]} cells → "
                  f"{len(set(br_assignments))} unique assignments, "
                  f"mean r={br_confidences.mean():.3f}")

        if c_rows:
            hcr_assignments_c = pd.DataFrame(c_rows)

    # -------------------------------------------------------------------------
    # 5.3 Marker-score cross-check (M.2)
    # -------------------------------------------------------------------------
    print(f"\n[5.3] Marker-score cross-check...")
    marker_scores = score_marker_sets(X_hcr, genes)
    marker_scores.index = hcr_corrected.obs_names

    # For each HCR cell, what's the dominant marker?
    dominant_marker = marker_scores.idxmax(axis=1)

    # Compare to Approach A assigned subclass
    assigned_subclass_a = hcr_assignments_a["assignment"].apply(
        lambda x: x.split()[0] if " " in x else x.split("-")[0] if "-" in x else x
    )
    canonical_mask = assigned_subclass_a.isin(CANONICAL_BRANCHES)
    if canonical_mask.any():
        agreement = (
            assigned_subclass_a[canonical_mask].values ==
            dominant_marker.loc[hcr_assignments_a.loc[canonical_mask, "cell_id"]].values
        ).mean()
        print(f"  Approach A vs marker-score agreement: {agreement:.1%}")
    else:
        print("  WARNING: No canonical subclass assignments — cannot compute marker agreement.")

    # -------------------------------------------------------------------------
    # 5.4 Convergence check: Approach A vs C
    # -------------------------------------------------------------------------
    if hcr_assignments_c is not None:
        print(f"\n[5.4] Convergence check (A vs C)...")
        # Compare subclass-level agreement between A and C
        # For cells assigned in C, extract the subclass from the branch_cluster name
        c_subclass = hcr_assignments_c.set_index("cell_id")["branch"]
        # For the same cells, get Approach A's subclass
        shared_cells = set(c_subclass.index) & set(hcr_assignments_a["cell_id"])
        if shared_cells:
            a_sub = assigned_subclass_a.loc[
                hcr_assignments_a["cell_id"].isin(shared_cells)
            ]
            a_sub.index = hcr_assignments_a.loc[
                hcr_assignments_a["cell_id"].isin(shared_cells), "cell_id"
            ]
            c_sub = c_subclass.loc[c_subclass.index.isin(shared_cells)]
            # Align
            common = sorted(set(a_sub.index) & set(c_sub.index))
            if common:
                agreement_ac = (a_sub.loc[common].values == c_sub.loc[common].values).mean()
                print(f"  Subclass-level agreement (A vs C): {agreement_ac:.1%} "
                      f"({len(common)} cells)")

    # -------------------------------------------------------------------------
    # 5.5 Save outputs
    # -------------------------------------------------------------------------
    print(f"\n[5.5] Saving Stage 5 outputs...")

    hcr_assignments_a.to_csv(stage5_dir / "hcr_assignments_approach_a.csv", index=False)
    if hcr_assignments_c is not None:
        # Leiden-named is the preferred output
        hcr_assignments_c.to_csv(
            stage5_dir / "hcr_assignments_leiden_named.csv", index=False
        )
        print(f"  ★ Preferred output: hcr_assignments_leiden_named.csv "
              f"({len(hcr_assignments_c)} cells)")
    marker_scores.to_csv(stage5_dir / "hcr_marker_scores.csv")

    # Diagnostic plots
    plot_stage5_diagnostics(
        hcr_assignments_a, hcr_assignments_c, marker_scores,
        corr_matrix_a, list(centroids_a.index), stage5_dir,
    )

    # Summary statistics
    summary = {
        "approach_a_n_cells": len(hcr_assignments_a),
        "approach_a_mean_confidence": float(confidences_a.mean()),
        "approach_a_low_conf_fraction": float((confidences_a < 0.3).mean()),
        "approach_a_n_unique_labels": len(set(assignments_a)),
    }
    if hcr_assignments_c is not None:
        summary["approach_c_n_cells"] = len(hcr_assignments_c)
        summary["approach_c_mean_confidence"] = float(hcr_assignments_c["confidence"].mean())

    pd.Series(summary).to_csv(stage5_dir / "matching_summary.csv", header=["value"])

    # -------------------------------------------------------------------------
    # 5.6 Per-mouse cell×gene plots + summary mean expression figure
    # -------------------------------------------------------------------------
    if hcr_assignments_c is not None:
        print(f"\n[5.6] Generating per-mouse cell×gene and mean-expression plots...")
        _plot_hcr_cellxgene_per_mouse(hcr_corrected, hcr_assignments_c, stage5_dir)
        _plot_mean_expression_summary(
            hcr_corrected, hcr_assignments_c, tasic_z, branch_results,
            branch_marker_names, stage5_dir,
        )

    # -------------------------------------------------------------------------
    # 5.7 Residual analysis
    # -------------------------------------------------------------------------
    if hcr_assignments_c is not None:
        print(f"\n[5.7] Computing residual analysis (cell vs assigned centroid)...")
        _compute_residual_analysis(hcr_corrected, hcr_assignments_c, stage5_dir)

    # -------------------------------------------------------------------------
    # 5.8 Clustering quality: R² and silhouette
    # -------------------------------------------------------------------------
    if hcr_assignments_c is not None:
        print(f"\n[5.8] Computing clustering quality (R² and silhouette scores)...")
        _compute_clustering_quality(hcr_corrected, hcr_assignments_c, stage5_dir)

    print(f"\n  Stage 5 complete. Outputs in {stage5_dir}")
    return hcr_assignments_a, hcr_assignments_c


def _compute_clustering_quality(
    hcr_corrected: ad.AnnData,
    hcr_assignments_c: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Assess how well the Tasic-derived cluster labels explain variance in HCR data.

    Three complementary metrics, computed per mouse and globally:

    1. R² (variance explained)
       R² = SS_between / SS_total = 1 - SS_within / SS_total
       Global single number + per-gene breakdown showing which genes are
       well-structured by the labels vs. which are noisy.

    2. Per-gene R² ranked barplot
       One-way ANOVA decomposition per gene: genes ranked most → least
       explained by the cluster assignment.

    3. Silhouette scores
       Per-cell: how similar is each cell to its own cluster vs. the nearest
       alternative cluster? Score in [-1, 1].
       Summarised as violin per cluster to show which clusters are
       well-separated vs. ambiguous.

    Outputs per mouse in mouse_{id}/:
        clustering_quality_summary.csv
        clustering_quality_overview.png   (3-panel figure)
    """
    genes = list(hcr_corrected.var_names)
    n_genes = len(genes)
    X = hcr_corrected.X if not hasattr(hcr_corrected.X, "toarray") else hcr_corrected.X.toarray()
    X = np.asarray(X, dtype=np.float64)
    cell_df = pd.DataFrame(X, columns=genes, index=hcr_corrected.obs_names)

    assignment_map = hcr_assignments_c.set_index("cell_id")["assignment"]
    mice = sorted(hcr_corrected.obs["mouse_id"].unique())

    # Global summary rows accumulated across mice
    global_rows = []

    for mouse in mice:
        mouse_mask = hcr_corrected.obs["mouse_id"] == mouse
        mouse_cell_ids = hcr_corrected.obs_names[mouse_mask]
        assigned_ids = mouse_cell_ids[mouse_cell_ids.isin(assignment_map.index)]
        if len(assigned_ids) < 4:
            continue

        mouse_dir = out_dir / f"mouse_{mouse}"
        mouse_dir.mkdir(parents=True, exist_ok=True)

        X_m = cell_df.loc[assigned_ids].values          # (n_cells, n_genes)
        labels_m = assignment_map.loc[assigned_ids].values
        unique_labels = sorted(set(labels_m))
        n_cells = X_m.shape[0]
        n_clusters = len(unique_labels)

        # ---------------------------------------------------------------
        # 1 & 2. R²: global + per gene
        # ---------------------------------------------------------------
        grand_mean = X_m.mean(axis=0)                   # (n_genes,)
        ss_total = ((X_m - grand_mean) ** 2).sum(axis=0)  # per gene

        # Build label index array once
        label_index = np.array([unique_labels.index(l) for l in labels_m])

        # Centroid for each cluster per gene
        centroids = np.zeros((n_clusters, n_genes))
        for ci, label in enumerate(unique_labels):
            mask_ci = label_index == ci
            centroids[ci] = X_m[mask_ci].mean(axis=0)

        ss_within = ((X_m - centroids[label_index]) ** 2).sum(axis=0)  # per gene
        ss_between = ss_total - ss_within

        # Guard against zero-variance genes
        r2_per_gene = np.where(ss_total > 1e-12, ss_between / ss_total, 0.0)
        global_r2 = float(ss_between.sum() / ss_total.sum()) if ss_total.sum() > 0 else 0.0

        gene_r2_df = pd.DataFrame({
            "gene": genes,
            "r2": r2_per_gene,
        }).sort_values("r2", ascending=False).reset_index(drop=True)

        # ---------------------------------------------------------------
        # 3. Silhouette scores
        # ---------------------------------------------------------------
        # Skip if only 1 cluster or too few cells per cluster
        can_silhouette = n_clusters >= 2 and n_cells >= 2 * n_clusters
        if can_silhouette:
            sil_scores = silhouette_samples(X_m, labels_m)
            sil_df = pd.DataFrame({
                "cell_id": assigned_ids,
                "assignment": labels_m,
                "silhouette": sil_scores,
            })
            mean_sil = float(sil_scores.mean())
        else:
            sil_df = pd.DataFrame(columns=["cell_id", "assignment", "silhouette"])
            mean_sil = float("nan")
            print(f"    Mouse {mouse}: silhouette skipped "
                  f"(n_clusters={n_clusters}, n_cells={n_cells})")

        global_rows.append({
            "mouse_id": mouse,
            "n_cells": n_cells,
            "n_clusters": n_clusters,
            "global_r2": round(global_r2, 4),
            "mean_silhouette": round(mean_sil, 4) if not np.isnan(mean_sil) else "NA",
            "best_explained_gene": gene_r2_df.iloc[0]["gene"],
            "worst_explained_gene": gene_r2_df.iloc[-1]["gene"],
        })

        # ---------------------------------------------------------------
        # Figure: 3-panel overview
        # ---------------------------------------------------------------
        has_sil = can_silhouette and len(sil_df) > 0
        n_panels = 3 if has_sil else 2
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(5 * n_panels, max(4, n_genes * 0.35)),
            gridspec_kw={"width_ratios": [1.4, 1, 1][:n_panels]},
        )

        # Panel 1: Per-gene R² barplot (horizontal, ranked)
        ax = axes[0]
        r2_sorted = gene_r2_df["r2"].values
        gene_sorted = gene_r2_df["gene"].values
        bar_colors = plt.cm.RdYlGn(r2_sorted)  # green=high R², red=low
        ax.barh(range(n_genes), r2_sorted, color=bar_colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(n_genes))
        ax.set_yticklabels(gene_sorted, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("R² (variance explained by cluster)", fontsize=9)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(
            f"Per-gene R²\nGlobal R² = {global_r2:.3f}",
            fontsize=10, fontweight="bold",
        )
        for i, (gene, r2) in enumerate(zip(gene_sorted, r2_sorted)):
            ax.text(r2 + 0.02, i, f"{r2:.2f}", va="center", fontsize=7)
        ax.grid(axis="x", alpha=0.3)

        # Panel 2: Per-cluster mean R² barplot
        ax = axes[1]
        cluster_r2 = {}
        for ci, label in enumerate(unique_labels):
            mask_ci = label_index == ci
            ss_t_cl = ((X_m[mask_ci] - grand_mean) ** 2).sum()
            ss_w_cl = ((X_m[mask_ci] - centroids[ci]) ** 2).sum()
            cluster_r2[label] = float((ss_t_cl - ss_w_cl) / ss_t_cl) if ss_t_cl > 1e-12 else 0.0
        sorted_clusters = sorted(cluster_r2, key=cluster_r2.get, reverse=True)
        cl_r2_vals = [cluster_r2[cl] for cl in sorted_clusters]
        cl_colors = plt.cm.RdYlGn(np.array(cl_r2_vals))
        ax.barh(range(len(sorted_clusters)), cl_r2_vals, color=cl_colors,
                edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(len(sorted_clusters)))
        ax.set_yticklabels(sorted_clusters, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("R² (cluster vs grand mean)", fontsize=9)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title("Per-cluster R²", fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Panel 3: Silhouette violin per cluster
        if has_sil:
            ax = axes[2]
            sil_by_cluster = [
                sil_df.loc[sil_df["assignment"] == lbl, "silhouette"].values
                for lbl in sorted_clusters
                if (sil_df["assignment"] == lbl).any()
            ]
            present_labels = [
                lbl for lbl in sorted_clusters
                if (sil_df["assignment"] == lbl).any()
            ]
            parts = ax.violinplot(
                sil_by_cluster,
                vert=False,
                showmedians=True,
                showextrema=True,
            )
            for pc, color in zip(parts["bodies"], cl_colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            ax.set_yticks(range(1, len(present_labels) + 1))
            ax.set_yticklabels(present_labels, fontsize=7)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.axvline(0.5, color="green", linewidth=0.8, linestyle=":", alpha=0.6)
            ax.set_xlabel("Silhouette score", fontsize=9)
            ax.set_xlim(-1, 1)
            ax.set_title(
                f"Silhouette per cluster\nMean = {mean_sil:.3f}",
                fontsize=10, fontweight="bold",
            )
            ax.grid(axis="x", alpha=0.3)

        plt.suptitle(
            f"Mouse {mouse}: Clustering quality  "
            f"(n={n_cells} cells, {n_clusters} clusters)",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(mouse_dir / "clustering_quality_overview.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save CSVs
        gene_r2_df.to_csv(mouse_dir / "clustering_r2_per_gene.csv", index=False)
        pd.DataFrame({
            "cluster": list(cluster_r2.keys()),
            "r2": list(cluster_r2.values()),
        }).sort_values("r2", ascending=False).to_csv(
            mouse_dir / "clustering_r2_per_cluster.csv", index=False
        )
        if has_sil:
            sil_df.to_csv(mouse_dir / "clustering_silhouette.csv", index=False)

        print(f"    Mouse {mouse}: global R²={global_r2:.3f}, "
              f"mean silhouette={mean_sil:.3f} "
              f"({n_cells} cells, {n_clusters} clusters)")

    # Save cross-mouse summary
    if global_rows:
        pd.DataFrame(global_rows).to_csv(
            out_dir / "clustering_quality_summary.csv", index=False
        )
        print(f"  Cross-mouse quality summary saved to {out_dir / 'clustering_quality_summary.csv'}")


def _compute_residual_analysis(
    hcr_corrected: ad.AnnData,
    hcr_assignments_c: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Compute per-cell residuals (cell z-score - assigned centroid z-score) and
    summarize in three diagnostics saved per mouse:

    1. Residual std heatmap (gene × cluster): which (gene, cluster) combinations
       show the highest spread across assigned cells.
    2. Ranked gene instability barplot: genes ranked by their mean residual std
       pooled across all clusters.
    3. Per-cell residual norm plot: ||cell - centroid||_2 sorted descending,
       revealing outlier / potentially misclassified cells.

    Uses z-scored HCR data (hcr_corrected.X) and assignment-derived centroids.
    """
    genes = list(hcr_corrected.var_names)
    n_genes = len(genes)
    X = hcr_corrected.X if not hasattr(hcr_corrected.X, "toarray") else hcr_corrected.X.toarray()
    X = np.asarray(X, dtype=np.float64)
    cell_df = pd.DataFrame(X, columns=genes, index=hcr_corrected.obs_names)

    assignment_map = hcr_assignments_c.set_index("cell_id")["assignment"]
    mice = sorted(hcr_corrected.obs["mouse_id"].unique())

    # Compute per-assignment centroids from hcr_corrected (z-scored)
    # (same space as individual cells, so residuals are directly interpretable)
    assigned_cells_all = hcr_corrected.obs_names[hcr_corrected.obs_names.isin(assignment_map.index)]
    cell_df_assigned = cell_df.loc[assigned_cells_all].copy()
    cell_df_assigned["_assignment"] = assignment_map.loc[assigned_cells_all].values
    centroids = cell_df_assigned.groupby("_assignment")[genes].mean()

    # Compute residual for every assigned cell: cell - its centroid
    residuals = cell_df_assigned[genes].values - centroids.loc[
        cell_df_assigned["_assignment"].values
    ].values  # shape: (n_cells, n_genes)
    residual_df = pd.DataFrame(
        residuals, columns=genes, index=assigned_cells_all
    )
    residual_df["_assignment"] = cell_df_assigned["_assignment"].values
    residual_df["_mouse"] = hcr_corrected.obs.loc[assigned_cells_all, "mouse_id"].values

    # Global gene instability (std of residuals per gene, pooled across all cells)
    gene_std_global = residual_df[genes].std(axis=0).sort_values(ascending=False)

    for mouse in mice:
        mouse_mask = residual_df["_mouse"] == mouse
        if mouse_mask.sum() == 0:
            continue

        mouse_dir = out_dir / f"mouse_{mouse}"
        mouse_dir.mkdir(parents=True, exist_ok=True)

        res_mouse = residual_df[mouse_mask]
        assignments_mouse = res_mouse["_assignment"]
        labels_present = sorted(assignments_mouse.unique())

        # ------------------------------------------------------------------
        # 1. Residual std heatmap: gene × cluster
        # ------------------------------------------------------------------
        # For each (gene, cluster): std of residuals across all cells in that cluster
        std_rows = {}
        for label in labels_present:
            mask_cl = assignments_mouse == label
            if mask_cl.sum() < 2:
                std_rows[label] = np.zeros(n_genes)
            else:
                std_rows[label] = res_mouse.loc[mask_cl, genes].std(axis=0).values
        std_matrix = pd.DataFrame(std_rows, index=genes).T  # clusters × genes
        std_matrix = std_matrix.reindex(labels_present)

        # Gene instability for this mouse (mean std across clusters)
        gene_std_mouse = std_matrix.mean(axis=0).sort_values(ascending=False)

        # Reorder columns by global instability rank for consistent gene ordering
        gene_order = gene_std_global.index.tolist()
        std_matrix_ordered = std_matrix[gene_order]

        fig_h, ax_h = plt.subplots(
            figsize=(max(6, n_genes * 0.5), max(4, len(labels_present) * 0.35))
        )
        im = ax_h.imshow(
            std_matrix_ordered.values,
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
        )
        ax_h.set_xticks(range(n_genes))
        ax_h.set_xticklabels(gene_order, rotation=90, fontsize=8)
        ax_h.set_yticks(range(len(labels_present)))
        ax_h.set_yticklabels(labels_present, fontsize=7)
        ax_h.set_xlabel("Gene (ranked by instability ▶)", fontsize=9)
        ax_h.set_title(
            f"Mouse {mouse}: Residual std per (gene, cluster)\n"
            "Genes ranked left→right by pooled instability",
            fontsize=10,
        )
        plt.colorbar(im, ax=ax_h, label="Residual std (z-score units)")
        plt.tight_layout()
        fig_h.savefig(mouse_dir / "residual_std_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig_h)

        # ------------------------------------------------------------------
        # 2. Ranked gene instability barplot
        # ------------------------------------------------------------------
        fig_b, ax_b = plt.subplots(figsize=(max(6, n_genes * 0.5), 4))
        bar_colors = plt.cm.YlOrRd(
            (gene_std_mouse.loc[gene_order] - gene_std_mouse.min()) /
            (gene_std_mouse.max() - gene_std_mouse.min() + 1e-12)
        )
        ax_b.bar(range(n_genes), gene_std_mouse.loc[gene_order].values, color=bar_colors,
                 edgecolor="black", linewidth=0.4)
        ax_b.set_xticks(range(n_genes))
        ax_b.set_xticklabels(gene_order, rotation=90, fontsize=8)
        ax_b.set_ylabel("Mean residual std across clusters", fontsize=9)
        ax_b.set_title(
            f"Mouse {mouse}: Gene instability (ranked most → least deviant)",
            fontsize=10,
        )
        ax_b.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig_b.savefig(mouse_dir / "residual_gene_instability.png", dpi=150, bbox_inches="tight")
        plt.close(fig_b)

        # ------------------------------------------------------------------
        # 3. Per-cell residual norm plot (outlier detection)
        # ------------------------------------------------------------------
        cell_norms = np.linalg.norm(res_mouse[genes].values, axis=1)
        sort_idx = np.argsort(cell_norms)[::-1]
        sorted_norms = cell_norms[sort_idx]
        sorted_assignments = assignments_mouse.iloc[sort_idx].values
        sorted_cell_ids = res_mouse.index[sort_idx]
        n_cells = len(sorted_norms)

        # Color bars by cluster assignment
        unique_labels = sorted(set(sorted_assignments))
        cmap_disc = plt.cm.get_cmap("tab20", len(unique_labels))
        label_to_color = {lbl: cmap_disc(i) for i, lbl in enumerate(unique_labels)}
        bar_cell_colors = [label_to_color[a] for a in sorted_assignments]

        fig_n, ax_n = plt.subplots(figsize=(max(8, n_cells * 0.04), 4))
        ax_n.bar(range(n_cells), sorted_norms, color=bar_cell_colors, linewidth=0)
        # Mark the 90th-percentile threshold
        p90 = np.percentile(sorted_norms, 90)
        ax_n.axhline(p90, color="red", linestyle="--", linewidth=1,
                     label=f"90th percentile ({p90:.2f})")
        ax_n.set_xlabel(f"Cells ranked by residual norm (n={n_cells})", fontsize=9)
        ax_n.set_ylabel("||cell − centroid||₂", fontsize=9)
        ax_n.set_title(
            f"Mouse {mouse}: Per-cell residual norms (sorted, colored by assignment)",
            fontsize=10,
        )
        ax_n.legend(fontsize=8)
        # Compact cluster legend
        legend_patches = [
            plt.matplotlib.patches.Patch(color=label_to_color[lbl], label=lbl)
            for lbl in unique_labels
        ]
        ax_n.legend(handles=legend_patches, fontsize=6, ncol=2,
                    loc="upper right", title="Assignment")
        plt.tight_layout()
        fig_n.savefig(mouse_dir / "residual_cell_norms.png", dpi=150, bbox_inches="tight")
        plt.close(fig_n)

        # ------------------------------------------------------------------
        # Save CSVs
        # ------------------------------------------------------------------
        std_matrix.to_csv(mouse_dir / "residual_std_by_gene_cluster.csv")
        pd.DataFrame({
            "gene": gene_order,
            "mean_residual_std": gene_std_mouse.loc[gene_order].values,
        }).to_csv(mouse_dir / "residual_gene_instability.csv", index=False)
        pd.DataFrame({
            "cell_id": sorted_cell_ids,
            "assignment": sorted_assignments,
            "residual_norm": sorted_norms,
        }).to_csv(mouse_dir / "residual_cell_norms.csv", index=False)

        print(f"    Mouse {mouse}: residual analysis saved to {mouse_dir} "
              f"({n_cells} cells, {len(labels_present)} clusters)")


def _plot_hcr_cellxgene_per_mouse(
    hcr_corrected: ad.AnnData,
    hcr_assignments_c: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    For each mouse, plot a cell×gene heatmap ordered by Leiden-named assignments.
    Uses raw counts (clipped 0-200), not z-scored values.
    Saves results into per-mouse subfolders.
    """
    # Load raw counts from hcr_log (saved during Stage 1 normalization)
    hcr_log_path = out_dir.parent / "hcr_log.h5ad"
    print(f"    Loading raw counts from {hcr_log_path}...")
    hcr_log = ad.read_h5ad(hcr_log_path)
    
    # Get raw counts for the same cells in hcr_corrected
    # hcr_log.layers["raw"] contains the raw counts before log1p
    X_raw = hcr_log.layers["raw"]
    if hasattr(X_raw, "toarray"):
        X_raw = X_raw.toarray()
    X_raw = np.asarray(X_raw, dtype=np.float64)
    
    # hcr_log and hcr_corrected should have the same cells in the same order
    # But we'll verify by matching on cell names to be safe
    assert len(hcr_log.obs_names) == len(hcr_corrected.obs_names), \
        f"Cell count mismatch: hcr_log={len(hcr_log.obs_names)}, hcr_corrected={len(hcr_corrected.obs_names)}"
    
    # Check if cells are in same order
    if all(hcr_log.obs_names == hcr_corrected.obs_names):
        # Cells already aligned, just need to reorder genes
        gene_indices = [np.where(hcr_log.var_names == g)[0][0] for g in hcr_corrected.var_names]
        X_raw = X_raw[:, gene_indices]
    else:
        # Need to reorder both genes and cells
        cell_indices = [np.where(hcr_log.obs_names == cid)[0][0] for cid in hcr_corrected.obs_names]
        gene_indices = [np.where(hcr_log.var_names == g)[0][0] for g in hcr_corrected.var_names]
        X_raw = X_raw[np.ix_(cell_indices, gene_indices)]
    
    cxg_all = pd.DataFrame(X_raw, columns=hcr_corrected.var_names, index=hcr_corrected.obs_names)
    
    assignment_map = hcr_assignments_c.set_index("cell_id")["assignment"]
    mice = sorted(hcr_corrected.obs["mouse_id"].unique())

    for mouse in mice:
        mouse_mask = hcr_corrected.obs["mouse_id"] == mouse
        mouse_cells = hcr_corrected.obs_names[mouse_mask]
        # Only cells that have Approach C assignments
        assigned_cells = mouse_cells[mouse_cells.isin(assignment_map.index)]
        if len(assigned_cells) == 0:
            continue

        # Create mouse-specific subdirectory
        mouse_dir = out_dir / f"mouse_{mouse}"
        mouse_dir.mkdir(parents=True, exist_ok=True)

        cxg_mouse = cxg_all.loc[assigned_cells]
        labels_mouse = assignment_map.loc[assigned_cells]

        fig, _, _ = viz.plot_cell_x_gene_labeled(
            cxg_mouse,
            labels=labels_mouse,
            clip_range=(0, 200),
            fig_size=(8, max(6, len(assigned_cells) * 0.003)),
            label_fontsize=7,
            cbar_label="raw count",
            title=f"Mouse {mouse}: HCR cells by Leiden-named assignment ({len(assigned_cells)} cells)",
        )
        fig.savefig(mouse_dir / "cellxgene_labeled.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
        
        # Also save the raw data as CSV for downstream analysis
        cxg_mouse.to_csv(mouse_dir / "cellxgene_raw_counts.csv")
        
        print(f"    Mouse {mouse}: results saved to {mouse_dir} ({len(assigned_cells)} cells)")


def _plot_mean_expression_summary(
    hcr_corrected: ad.AnnData,
    hcr_assignments_c: pd.DataFrame,
    tasic_z: ad.AnnData,
    branch_results: dict,
    branch_marker_names: dict,
    out_dir: Path,
) -> None:
    """
    Summary figure: mean gene expression per Leiden-named label.
    Subplots: one per mouse + one for Tasic reference.
    Also creates per-mouse summary heatmaps and saves centroids to CSV.
    """
    genes = list(hcr_corrected.var_names)
    mice = sorted(hcr_corrected.obs["mouse_id"].unique())

    X_hcr = hcr_corrected.X if not hasattr(hcr_corrected.X, "toarray") else hcr_corrected.X.toarray()
    hcr_df = pd.DataFrame(X_hcr, columns=genes, index=hcr_corrected.obs_names)

    # Build assignment series
    assignment_map = hcr_assignments_c.set_index("cell_id")["assignment"]

    # Build Tasic reference centroids with same Leiden names
    tasic_named_rows = []
    for branch, (adata_br, _) in branch_results.items():
        name_map = branch_marker_names.get(branch, {})
        X_br = adata_br.X if not hasattr(adata_br.X, "toarray") else adata_br.X.toarray()
        df_br = pd.DataFrame(X_br, columns=adata_br.var_names, index=adata_br.obs_names)
        df_br["_label"] = [
            name_map.get(cl, cl) for cl in adata_br.obs["branch_cluster"].values
        ]
        means = df_br.groupby("_label")[genes].mean()
        tasic_named_rows.append(means)
    if tasic_named_rows:
        tasic_centroids = pd.concat(tasic_named_rows)
    else:
        tasic_centroids = pd.DataFrame(columns=genes)

    # Compute per-mouse centroids
    mouse_centroids = {}
    for mouse in mice:
        mouse_cells = hcr_corrected.obs_names[hcr_corrected.obs["mouse_id"] == mouse]
        assigned = mouse_cells[mouse_cells.isin(assignment_map.index)]
        if len(assigned) == 0:
            continue
        df_m = hcr_df.loc[assigned].copy()
        df_m["_label"] = assignment_map.loc[assigned].values
        mouse_centroids[mouse] = df_m.groupby("_label")[genes].mean()

    # Use sorted label order from Tasic for consistent ordering
    all_labels = sorted(tasic_centroids.index.tolist())

    # Create combined summary panel
    n_panels = len(mouse_centroids) + 1  # mice + Tasic
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, max(5, len(all_labels) * 0.3)))
    if n_panels == 1:
        axes = [axes]

    # Tasic reference
    ax = axes[0]
    plot_data = tasic_centroids.reindex(all_labels).fillna(0)
    sns.heatmap(plot_data, cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "z-score", "shrink": 0.7})
    ax.set_title("Tasic reference", fontsize=10)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    ax.tick_params(axis="x", labelsize=8, rotation=90)

    # Per-mouse
    for i, mouse in enumerate(mice):
        if mouse not in mouse_centroids:
            continue
        ax = axes[i + 1]
        plot_data = mouse_centroids[mouse].reindex(all_labels).fillna(0)
        sns.heatmap(plot_data, cmap="RdBu_r", center=0, ax=ax,
                    cbar_kws={"label": "z-score", "shrink": 0.7})
        n_cells = (hcr_assignments_c["mouse_id"] == mouse).sum()
        ax.set_title(f"Mouse {mouse} (n={n_cells})", fontsize=10)
        ax.tick_params(axis="y", labelsize=7, rotation=0)
        ax.tick_params(axis="x", labelsize=8, rotation=90)
        ax.set_ylabel("")

    plt.suptitle("Mean expression per Leiden-named cluster (Tasic + per-mouse HCR)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "stage5_04_mean_expression_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Mean expression summary saved ({n_panels} panels)")

    # Create per-mouse individual summary plots and save centroids
    for mouse in mice:
        if mouse not in mouse_centroids:
            continue
        
        # Create mouse-specific subdirectory
        mouse_dir = out_dir / f"mouse_{mouse}"
        mouse_dir.mkdir(parents=True, exist_ok=True)
        
        # Get data for this mouse aligned with all_labels
        tasic_data = tasic_centroids.reindex(all_labels).fillna(0)
        mouse_data = mouse_centroids[mouse].reindex(all_labels).fillna(0)
        diff_data = mouse_data - tasic_data
        
        # Compute per-cluster Pearson correlations
        correlations = []
        for label in all_labels:
            if label in tasic_centroids.index and label in mouse_centroids[mouse].index:
                tasic_row = tasic_centroids.loc[label].values
                mouse_row = mouse_centroids[mouse].loc[label].values
                # Compute Pearson correlation
                corr = np.corrcoef(tasic_row, mouse_row)[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0.0)
            else:
                correlations.append(0.0)
        
        # Create 4-panel figure: Tasic | Mouse | Difference | Correlations
        fig = plt.figure(figsize=(16, max(5, len(all_labels) * 0.3)))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.6], wspace=0.3)
        
        # Panel 1: Tasic reference
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(tasic_data, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        ax1.set_yticks(range(len(all_labels)))
        ax1.set_yticklabels(all_labels, fontsize=7)
        ax1.set_xticks(range(tasic_data.shape[1]))
        ax1.set_xticklabels([f"G{i+1}" for i in range(tasic_data.shape[1])], fontsize=7, rotation=90)
        ax1.set_title("Tasic reference", fontsize=11, fontweight="bold")
        plt.colorbar(im1, ax=ax1, label="z-score")
        
        # Panel 2: Mouse data
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(mouse_data, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        ax2.set_yticks(range(len(all_labels)))
        ax2.set_yticklabels([], fontsize=7)
        ax2.set_xticks(range(mouse_data.shape[1]))
        ax2.set_xticklabels([f"G{i+1}" for i in range(mouse_data.shape[1])], fontsize=7, rotation=90)
        n_cells = (hcr_assignments_c["mouse_id"] == mouse).sum()
        ax2.set_title(f"Mouse {mouse} (n={n_cells})", fontsize=11, fontweight="bold")
        plt.colorbar(im2, ax=ax2, label="z-score")
        
        # Panel 3: Difference (Mouse - Reference)
        ax3 = fig.add_subplot(gs[0, 2])
        # Use diverging colormap centered at 0 for differences
        diff_max = np.abs(diff_data.values).max()
        im3 = ax3.imshow(diff_data, aspect="auto", cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
        ax3.set_yticks(range(len(all_labels)))
        ax3.set_yticklabels([], fontsize=7)
        ax3.set_xticks(range(diff_data.shape[1]))
        ax3.set_xticklabels([f"G{i+1}" for i in range(diff_data.shape[1])], fontsize=7, rotation=90)
        ax3.set_title("Difference\n(Mouse - Reference)", fontsize=11, fontweight="bold")
        plt.colorbar(im3, ax=ax3, label="ΔZ-score")
        
        # Panel 4: Per-cluster correlations
        ax4 = fig.add_subplot(gs[0, 3])
        colors = ["green" if corr > 0.8 else "orange" if corr > 0.6 else "red" for corr in correlations]
        ax4.barh(range(len(all_labels)), correlations, color=colors, edgecolor="black", linewidth=0.5)
        ax4.set_yticks(range(len(all_labels)))
        ax4.set_yticklabels([], fontsize=7)
        ax4.set_xlabel("Pearson r", fontsize=9)
        ax4.set_title("Cluster-wise\ncorrelation", fontsize=11, fontweight="bold")
        ax4.set_xlim([-1, 1])
        ax4.axvline(0.8, color="green", linestyle="--", alpha=0.5, linewidth=1)
        ax4.axvline(0.6, color="orange", linestyle="--", alpha=0.5, linewidth=1)
        ax4.tick_params(axis="x", labelsize=8)
        ax4.grid(axis="x", alpha=0.3)
        
        # Add text annotations for correlation values
        for i, (label, corr) in enumerate(zip(all_labels, correlations)):
            ax4.text(corr + 0.05, i, f"{corr:.2f}", va="center", fontsize=6)
        
        plt.suptitle(f"Mouse {mouse}: Comprehensive mean expression analysis", fontsize=12, fontweight="bold", y=0.98)
        plt.savefig(mouse_dir / "mean_expression_4panel_comparison.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        
        # Save per-mouse centroids and differences as CSV
        mouse_centroids[mouse].to_csv(mouse_dir / "mean_expression_centroids.csv")
        diff_data.to_csv(mouse_dir / "mean_expression_diff.csv")
        pd.DataFrame({
            "cluster": all_labels,
            "pearson_r": correlations,
        }).to_csv(mouse_dir / "cluster_correlations.csv", index=False)
        
        print(f"    Mouse {mouse}: per-mouse summary saved to {mouse_dir}")


# =============================================================================
# Main
# =============================================================================


def setup_logging(out_dir: Path) -> None:
    """Configure logging to both console and file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(fh)

    # Also tee stdout/stderr to the log file
    class TeeWriter:
        def __init__(self, original, log_file):
            self.original = original
            self.log_file = log_file

        def write(self, msg):
            self.original.write(msg)
            self.log_file.write(msg)

        def flush(self):
            self.original.flush()
            self.log_file.flush()

    log_file = open(log_path, "a")
    sys.stdout = TeeWriter(sys.__stdout__, log_file)
    sys.stderr = TeeWriter(sys.__stderr__, log_file)
    print(f"Logging to: {log_path}")


def main(
    batch_mode: str = "all",
    effect_threshold: float = 1.0,
    drop_minor_subclasses: bool = False,
    min_cells_per_cluster: int = 0,
    min_cells_per_branch_cluster: int = 0,
    run_10x_hmb: bool = False,
    tenx_dir: Path = TENX_HMB_DIR,
    tenx_expression_scale: str = "log2",
    tenx_label_column: str = "supertype",
    recompute_stage4: bool = False,
) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT_ROOT)

    print(f"  Batch correction mode: {batch_mode}")
    print(f"  Effect size threshold: {effect_threshold}")
    print(f"  Drop minor subclasses: {drop_minor_subclasses}")
    print(f"  Min cells per cluster (Stage 1): {min_cells_per_cluster}")
    print(f"  Min cells per branch cluster (Stage 4): {min_cells_per_branch_cluster}")
    print(f"  Run 10x-HMB matching: {run_10x_hmb}")
    if run_10x_hmb:
        print(f"  10x-HMB dir: {tenx_dir}")
        print(f"  10x-HMB scale: {tenx_expression_scale}")
        print(f"  10x-HMB label column: {tenx_label_column}")
    print(f"  Recompute Stage 4: {recompute_stage4}")

    # Create stage-specific subdirectories
    normalization_dir = OUT_ROOT / "normalization"
    batch_correction_dir = OUT_ROOT / "batch-correction"
    normalization_dir.mkdir(parents=True, exist_ok=True)
    batch_correction_dir.mkdir(parents=True, exist_ok=True)

    has_hcr_query = len(MOUSE_IDS) > 0
    if (not has_hcr_query) and (not run_10x_hmb):
        raise ValueError(
            "MOUSE_IDS is empty and --run-10x-hmb is False. "
            "Enable --run-10x-hmb to run 10x-only mode, or provide at least one mouse id."
        )

    tenx_raw_cached = None
    tenx_z_cached = None

    if has_hcr_query:
        # Stage 1
        tasic_z, hcr_z, tasic_log, hcr_log = run_stage1(
            drop_minor_subclasses=drop_minor_subclasses,
            min_cells_per_cluster=min_cells_per_cluster,
        )

        # Stage 1 summary plots
        plot_normalization_summary(tasic_z, hcr_z, normalization_dir)

        # Stage 2
        hcr_corrected = run_stage2(hcr_z, hcr_log, tasic_z, batch_correction_dir, batch_mode=batch_mode)

        # Save processed data for downstream stages
        print("\n  Saving processed AnnData objects...")
        tasic_z.write(OUT_ROOT / "tasic_z.h5ad")
        hcr_corrected.write(OUT_ROOT / "hcr_corrected.h5ad")
        hcr_log.write(OUT_ROOT / "hcr_log.h5ad")
        tasic_log.write(OUT_ROOT / "tasic_log.h5ad")

        # Save a summary table
        summary = pd.DataFrame({
            "item": [
                "n_mice", "n_hcr_cells", "n_tasic_cells",
                "n_shared_genes", "shared_genes", "batch_mode",
            ],
            "value": [
                str(len(MOUSE_IDS)),
                str(hcr_corrected.n_obs),
                str(tasic_z.n_obs),
                str(tasic_z.n_vars),
                ", ".join(tasic_z.var_names.tolist()),
                batch_mode,
            ],
        })
        summary.to_csv(OUT_ROOT / "stage1_2_summary.csv", index=False)

        print("\n" + "=" * 60)
        print("STAGES 1-2 COMPLETE")
        print("=" * 60)
        print(f"  Outputs: {OUT_ROOT}")
        print(f"  Mice: {MOUSE_IDS}")
        print(f"  HCR cells: {hcr_corrected.n_obs}")
        print(f"  Tasic cells: {tasic_z.n_obs}")
        print(f"  Panel genes: {tasic_z.n_vars} ({', '.join(tasic_z.var_names.tolist())})")
        print(f"  Batch mode: {batch_mode}")
    else:
        print("\n" + "=" * 60)
        print("10X-ONLY MODE: Building Tasic reference without HCR")
        print("=" * 60)

        # Load 10x first so we can use its genes to subset Tasic.
        tenx_raw_cached = load_10x_hmb_query(tenx_dir=tenx_dir, label_column=tenx_label_column)
        tenx_genes = [g for g in tenx_raw_cached.var_names if g not in EXCLUDE_GENES]

        print("\n[1x.1] Loading Tasic reference (genes from 10x panel)...")
        tasic_raw = load_tasic_inhibitory(genes=tenx_genes, layer="exon")

        if drop_minor_subclasses or min_cells_per_cluster > 0:
            print("\n[1x.1b] Filtering Tasic reference...")
            tasic_raw = filter_tasic_reference(
                tasic_raw,
                drop_minor_subclasses=drop_minor_subclasses,
                min_cells_per_cluster=min_cells_per_cluster,
            )

        print("\n[1x.2] Intersecting Tasic with 10x genes...")
        tasic_raw, tenx_raw_cached = intersect_genes(tasic_raw, tenx_raw_cached, exclude=EXCLUDE_GENES)

        print("\n[1x.3] Normalizing and z-scoring Tasic + 10x...")
        tasic_log = normalize_tasic(tasic_raw)
        tasic_z = zscore_genes(tasic_log)
        tenx_log = normalize_10x_hmb(tenx_raw_cached, expression_scale=tenx_expression_scale)
        tenx_z_cached = zscore_genes(tenx_log)

        tasic_z.write(OUT_ROOT / "tasic_z.h5ad")
        tasic_log.write(OUT_ROOT / "tasic_log.h5ad")
        tenx_z_cached.write(OUT_ROOT / "10x_hmb_z.h5ad")

        print(f"  Tasic cells: {tasic_z.n_obs}")
        print(f"  10x cells:   {tenx_z_cached.n_obs}")
        print(f"  Shared genes: {tasic_z.n_vars}")

    # Stage 3
    mapping, centroids_collapsed, separability_df = run_stage3(
        tasic_z, OUT_ROOT, effect_size_threshold=effect_threshold
    )

    # Stage 4 (reuse cache if available unless recompute is requested)
    if not recompute_stage4:
        try:
            print("\n[4.cache] Attempting to load existing Stage 4 outputs...")
            branch_results, tasic_gating, hcr_gating, branch_marker_names = load_stage4_cached(OUT_ROOT)
            print(f"  Loaded cached Stage 4 from {OUT_ROOT / 'stage4'}")
        except Exception as e:
            print(f"  Stage 4 cache not usable ({e}); recomputing Stage 4...")
            branch_results, tasic_gating, hcr_gating, branch_marker_names = run_stage4(
                tasic_log, tasic_z, hcr_corrected if has_hcr_query else None, OUT_ROOT,
                min_cells_per_branch_cluster=min_cells_per_branch_cluster,
            )
    else:
        branch_results, tasic_gating, hcr_gating, branch_marker_names = run_stage4(
            tasic_log, tasic_z, hcr_corrected if has_hcr_query else None, OUT_ROOT,
            min_cells_per_branch_cluster=min_cells_per_branch_cluster,
        )

    # Stage 5 (HCR-only)
    hcr_assignments_a = None
    hcr_assignments_c = None
    if has_hcr_query:
        hcr_assignments_a, hcr_assignments_c = run_stage5(
            hcr_corrected, tasic_z, centroids_collapsed,
            branch_results, hcr_gating, branch_marker_names, OUT_ROOT
        )

    # Optional: 10x-HMB -> Tasic matching
    if run_10x_hmb:
        print("\n" + "=" * 60)
        print("EXTRA: 10x-HMB QUERY MATCHING")
        print("=" * 60)

        if tenx_z_cached is None:
            print("\n[10x.1] Loading 10x-HMB query matrix...")
            tenx_raw = load_10x_hmb_query(tenx_dir=tenx_dir, label_column=tenx_label_column)

            print("\n[10x.2] Intersecting genes with Tasic panel...")
            # Reuse intersect utility by putting Tasic first to preserve tasic-var conventions.
            tasic_tmp, tenx_raw = intersect_genes(tasic_z, tenx_raw, exclude=EXCLUDE_GENES)
            del tasic_tmp

            print("\n[10x.3] Normalizing + z-scoring 10x-HMB...")
            tenx_log = normalize_10x_hmb(tenx_raw, expression_scale=tenx_expression_scale)
            tenx_z = zscore_genes(tenx_log)
            tenx_z.write(OUT_ROOT / "10x_hmb_z.h5ad")
        else:
            tenx_z = tenx_z_cached
            print("\n[10x.1-3] Reusing precomputed 10x z-scored query from 10x-only bootstrap path.")

        print("\n[10x.4] Soft subclass gating for 10x-HMB...")
        tenx_gating = soft_subclass_gating(
            tenx_z,
            tasic_z,
            n_neighbors=15,
            confidence_threshold=0.5,
            margin_threshold=0.2,
        )

        print("\n[10x.5] Matching 10x-HMB -> Tasic-derived clusters...")
        run_stage5_10x_hmb(
            tenx_z=tenx_z,
            tasic_z=tasic_z,
            centroids_collapsed=centroids_collapsed,
            branch_results=branch_results,
            tenx_gating=tenx_gating,
            branch_marker_names=branch_marker_names,
            out_dir=OUT_ROOT,
            label_col=tenx_label_column,
        )

    print("\n" + "=" * 60)
    print("ALL STAGES COMPLETE (1-5)")
    print("=" * 60)
    if hcr_assignments_a is not None:
        print(f"  HCR Approach A assignments: {len(hcr_assignments_a)}")
    if hcr_assignments_c is not None:
        print(f"  HCR Approach C assignments: {len(hcr_assignments_c)}")
    if run_10x_hmb:
        print("  10x-HMB matching completed")
    print(f"  Outputs: {OUT_ROOT}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HCR-Tasic matching pipeline")
    parser.add_argument(
        "--batch-mode", type=str, default="all",
        choices=["all", "exclude_markers", "per_mouse", "none"],
        help="Batch correction mode: 'all' (default, center all genes), "
             "'exclude_markers' (skip Pvalb/Sst/Vip/Lamp5), "
             "'per_mouse' (z-score each mouse independently), or 'none'.",
    )
    parser.add_argument(
        "--effect-threshold", type=float, default=1.0,
        help="Effect size threshold for Stage 3 separability (default: 1.0).",
    )
    parser.add_argument(
        "--drop-minor-subclasses", action="store_true", default=False,
        help="Drop Serpinf1, CR, and Meis2 subclass cells from Tasic reference.",
    )
    parser.add_argument(
        "--min-cells-per-cluster", type=int, default=10,
        help="(Stage 1) Drop Tasic clusters with fewer than N cells before any "
             "analysis (default: 10, 0 = keep all).",
    )
    parser.add_argument(
        "--min-cells-per-branch-cluster", type=int, default=10,
        help="(Stage 4) After within-branch Leiden clustering, drop branch "
             "clusters with fewer than N Tasic cells. Prevents rare/noisy "
             "clusters from entering the matching reference (default: 10).",
    )
    parser.add_argument(
        "--run-10x-hmb", action="store_true", default=False,
        help="Also run 10x-HMB query matching into Tasic-derived clusters.",
    )
    parser.add_argument(
        "--tenx-dir", type=Path, default=TENX_HMB_DIR,
        help="Path to 10x-HMB folder containing cell_x_gene.csv and labels_supertype.csv.",
    )
    parser.add_argument(
        "--tenx-expression-scale", type=str, default="log2", choices=["log2", "raw"],
        help="Expression scale of input 10x-HMB matrix.",
    )
    parser.add_argument(
        "--tenx-label-column", type=str, default="supertype",
        help="Label column to use from labels_supertype.csv for mapping summaries.",
    )
    parser.add_argument(
        "--recompute-stage4", action="store_true", default=False,
        help="Force recomputation of Stage 4 even if cached outputs exist.",
    )
    args = parser.parse_args()
    main(
        batch_mode=args.batch_mode,
        effect_threshold=args.effect_threshold,
        drop_minor_subclasses=args.drop_minor_subclasses,
        min_cells_per_cluster=args.min_cells_per_cluster,
        min_cells_per_branch_cluster=args.min_cells_per_branch_cluster,
        run_10x_hmb=args.run_10x_hmb,
        tenx_dir=args.tenx_dir,
        tenx_expression_scale=args.tenx_expression_scale,
        tenx_label_column=args.tenx_label_column,
        recompute_stage4=args.recompute_stage4,
    )
