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

from aind_hcr_data_loader import get_hcr_dataset_pairwise
import aind_hcr_qc.viz as viz
import cluster_validation_utils


# =============================================================================
# Constants
# =============================================================================

DATA_DIR = Path("/root/capsule/data")
SS_PATH = Path("/root/capsule/scratch/mouse_VISp_gene_expression_matrices_2018-06-14")
OUT_ROOT = Path("/root/capsule/results/hcr_tasic_matching")

# Mice with confirmed pairwise-unmixed inhibitory cell data
MOUSE_IDS = ["790322", "782149", "788406"]

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
    fig.savefig(out_dir / "stage2_01_pre_correction_umap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'stage2_01_pre_correction_umap.png'}")


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
    fig.savefig(out_dir / "stage2_02_post_correction_umap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'stage2_02_post_correction_umap.png'}")


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
        sns.violinplot(data=df, x="mouse_id", y="expression", ax=ax,
                       inner="quartile", density_norm="width", cut=0,
                       palette=mouse_colors, order=sorted(mice))
        ax.set_title(gene, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("Per-gene distributions by mouse (log-normalized, pre-correction)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "stage2_03_gene_distributions_pre.png", dpi=150, bbox_inches="tight")
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
        sns.violinplot(data=df, x="mouse_id", y="expression", ax=ax,
                       inner="quartile", density_norm="width", cut=0,
                       palette=mouse_colors, order=sorted(mice))
        ax.set_title(gene, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("Per-gene distributions by mouse (z-scored, post-correction)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "stage2_04_gene_distributions_post.png", dpi=150, bbox_inches="tight")
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
    fig.savefig(out_dir / "stage2_05_batch_correlation_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: stage2_03/04 (grid violins) + stage2_05 (correlation summary)")


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
    fig.savefig(out_dir / "stage1_01_cross_platform_centroids.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'stage1_01_cross_platform_centroids.png'}")


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

    plt.suptitle("Stage 1 output: z-scored subclass centroids (both platforms)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "stage1_02_normalization_summary_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'stage1_02_normalization_summary_heatmaps.png'}")


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
    fig.savefig(out_dir / "stage2_00_composition_by_mouse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'stage2_00_composition_by_mouse.png'}")

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
    tasic_z: ad.AnnData,
    branch: str,
    branch_assignments: pd.DataFrame,
    n_neighbors_range: list[int] | None = None,
    resolution_range: list[float] | None = None,
) -> tuple[ad.AnnData, dict]:
    """
    Run Leiden parameter sweep within a single subclass branch on Tasic data.

    Uses ARI against original Tasic cluster labels to find best params.

    Parameters
    ----------
    tasic_z : AnnData
        Full Tasic z-scored data.
    branch : str
        Branch name (e.g. "Pvalb").
    branch_assignments : DataFrame
        Output from soft_subclass_gating applied to Tasic cells.
    n_neighbors_range, resolution_range : optional parameter ranges for Leiden.

    Returns
    -------
    adata_branch : AnnData
        Branch subset with 'leiden' column in .obs.
    results_dict : dict
        Sweep results, best params, ARI.
    """
    from sklearn.metrics import adjusted_rand_score

    # Select cells assigned to this branch
    branch_cells = branch_assignments[
        branch_assignments["assigned_branch"] == branch
    ]["cell_id"].values
    mask = tasic_z.obs_names.isin(branch_cells)
    adata_branch = tasic_z[mask].copy()

    if adata_branch.n_obs < 10:
        print(f"    WARNING: Branch {branch} has only {adata_branch.n_obs} cells, skipping.")
        return adata_branch, {"skip": True, "n_cells": adata_branch.n_obs}

    print(f"    Branch {branch}: {adata_branch.n_obs} cells, "
          f"{adata_branch.obs['cluster'].nunique()} original Tasic clusters")

    if n_neighbors_range is None:
        n_neighbors_range = [5, 10, 15, 20, 30, 40, 50]
    if resolution_range is None:
        resolution_range = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    # Parameter sweep
    sweep_rows = []
    for n_neighbors in n_neighbors_range:
        if n_neighbors >= adata_branch.n_obs:
            continue
        for resolution in resolution_range:
            sc.pp.neighbors(adata_branch, use_rep="X", n_neighbors=n_neighbors)
            sc.tl.leiden(
                adata_branch, resolution=resolution,
                key_added="leiden_test", flavor="igraph",
            )
            ari = adjusted_rand_score(
                adata_branch.obs["cluster"].astype(str),
                adata_branch.obs["leiden_test"].astype(str),
            )
            sweep_rows.append({
                "n_neighbors": n_neighbors,
                "resolution": resolution,
                "ari": ari,
                "n_clusters": adata_branch.obs["leiden_test"].nunique(),
            })

    sweep_df = pd.DataFrame(sweep_rows).sort_values("ari", ascending=False).reset_index(drop=True)
    best = sweep_df.iloc[0]
    best_n = int(best["n_neighbors"])
    best_r = float(best["resolution"])

    # Final clustering with best params
    sc.pp.neighbors(adata_branch, use_rep="X", n_neighbors=best_n)
    sc.tl.leiden(adata_branch, resolution=best_r, flavor="igraph")
    sc.tl.umap(adata_branch)

    # Label Leiden clusters with branch prefix
    adata_branch.obs["branch_cluster"] = [
        f"{branch}-{lid}" for lid in adata_branch.obs["leiden"].values
    ]

    n_clusters = adata_branch.obs["leiden"].nunique()
    print(f"      Best params: n_neighbors={best_n}, resolution={best_r:.1f}, "
          f"ARI={best['ari']:.3f}, n_clusters={n_clusters}")

    results_dict = {
        "sweep_df": sweep_df,
        "best_params": {
            "best_n_neighbors": best_n,
            "best_resolution": best_r,
            "best_ari": float(best["ari"]),
        },
        "n_cells": adata_branch.n_obs,
        "n_clusters": n_clusters,
    }
    return adata_branch, results_dict


def bootstrap_branch_stability(
    adata_branch: ad.AnnData,
    n_bootstraps: int = 20,
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


def run_stage4(
    tasic_z: ad.AnnData,
    hcr_corrected: ad.AnnData,
    out_dir: Path,
    n_neighbors_gating: int = 15,
    confidence_threshold: float = 0.5,
    margin_threshold: float = 0.2,
    n_bootstraps: int = 20,
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
    tasic_z : AnnData
        Z-scored Tasic inhibitory cells.
    hcr_corrected : AnnData
        Batch-corrected z-scored HCR cells.
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
            tasic_z, branch, tasic_gating,
        )

        if res_dict.get("skip"):
            print(f"    Skipped (too few cells)")
            continue

        branch_results[branch] = (adata_branch, res_dict)

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

        # Build marker-derived names: "Pvalb-0: Calb1 | Reln | Tac1"
        high_conf = markers_df[
            (markers_df["stability_pct"] >= 80.0) & (markers_df["direction"] == "enriched")
        ]
        name_map = {}
        for cl in sorted(adata_branch.obs["branch_cluster"].unique()):
            cl_markers = high_conf[high_conf["cluster"] == cl].sort_values("rank")
            marker_str = "-".join(cl_markers["gene"].head(3).tolist())
            if marker_str:
                name_map[cl] = f"{cl}: {marker_str}"
            else:
                # Fall back to top enriched regardless of stability
                cl_enriched = markers_df[
                    (markers_df["cluster"] == cl) & (markers_df["direction"] == "enriched")
                ].sort_values("rank")
                marker_str = "-".join(cl_enriched["gene"].head(3).tolist())
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

    print(f"\n  Stage 5 complete. Outputs in {stage5_dir}")
    return hcr_assignments_a, hcr_assignments_c


def _plot_hcr_cellxgene_per_mouse(
    hcr_corrected: ad.AnnData,
    hcr_assignments_c: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    For each mouse, plot a cell×gene heatmap ordered by Leiden-named assignments.
    Uses viz.plot_cell_x_gene_labeled.
    """
    assignment_map = hcr_assignments_c.set_index("cell_id")["assignment"]
    mice = sorted(hcr_corrected.obs["mouse_id"].unique())

    X = hcr_corrected.X if not hasattr(hcr_corrected.X, "toarray") else hcr_corrected.X.toarray()
    cxg_all = pd.DataFrame(X, columns=hcr_corrected.var_names, index=hcr_corrected.obs_names)

    for mouse in mice:
        mouse_mask = hcr_corrected.obs["mouse_id"] == mouse
        mouse_cells = hcr_corrected.obs_names[mouse_mask]
        # Only cells that have Approach C assignments
        assigned_cells = mouse_cells[mouse_cells.isin(assignment_map.index)]
        if len(assigned_cells) == 0:
            continue

        cxg_mouse = cxg_all.loc[assigned_cells]
        labels_mouse = assignment_map.loc[assigned_cells]

        fig, _, _ = viz.plot_cell_x_gene_labeled(
            cxg_mouse,
            labels=labels_mouse,
            clip_range=(-2.5, 2.5),
            fig_size=(8, max(6, len(assigned_cells) * 0.003)),
            label_fontsize=7,
            cbar_label="z-score",
            title=f"Mouse {mouse}: HCR cells by Leiden-named assignment ({len(assigned_cells)} cells)",
        )
        fig.savefig(out_dir / f"stage5_cellxgene_mouse_{mouse}.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"    Mouse {mouse}: cell×gene plot saved ({len(assigned_cells)} cells)")


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
) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT_ROOT)

    print(f"  Batch correction mode: {batch_mode}")
    print(f"  Effect size threshold: {effect_threshold}")
    print(f"  Drop minor subclasses: {drop_minor_subclasses}")
    print(f"  Min cells per cluster: {min_cells_per_cluster}")

    # Stage 1
    tasic_z, hcr_z, tasic_log, hcr_log = run_stage1(
        drop_minor_subclasses=drop_minor_subclasses,
        min_cells_per_cluster=min_cells_per_cluster,
    )

    # Stage 1 summary plots
    plot_normalization_summary(tasic_z, hcr_z, OUT_ROOT)

    # Stage 2
    hcr_corrected = run_stage2(hcr_z, hcr_log, tasic_z, OUT_ROOT, batch_mode=batch_mode)

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

    # Stage 3
    mapping, centroids_collapsed, separability_df = run_stage3(
        tasic_z, OUT_ROOT, effect_size_threshold=effect_threshold
    )

    # Stage 4
    branch_results, tasic_gating, hcr_gating, branch_marker_names = run_stage4(
        tasic_z, hcr_corrected, OUT_ROOT
    )

    # Stage 5
    hcr_assignments_a, hcr_assignments_c = run_stage5(
        hcr_corrected, tasic_z, centroids_collapsed,
        branch_results, hcr_gating, branch_marker_names, OUT_ROOT
    )

    print("\n" + "=" * 60)
    print("ALL STAGES COMPLETE (1-5)")
    print("=" * 60)
    print(f"  Approach A assignments: {len(hcr_assignments_a)}")
    if hcr_assignments_c is not None:
        print(f"  Approach C assignments: {len(hcr_assignments_c)}")
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
        "--min-cells-per-cluster", type=int, default=0,
        help="Drop Tasic clusters with fewer than N cells (default: 0 = keep all).",
    )
    args = parser.parse_args()
    main(
        batch_mode=args.batch_mode,
        effect_threshold=args.effect_threshold,
        drop_minor_subclasses=args.drop_minor_subclasses,
        min_cells_per_cluster=args.min_cells_per_cluster,
    )
