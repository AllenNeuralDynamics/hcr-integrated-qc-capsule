"""Collect VISp inhibitory reference atlas cell×gene matrices and clustered PNGs.

This script builds reference cell×gene matrices for:
- MERFISH (ABC Atlas, VISp inhibitory classes)
- TASIC Smart-seq (VISp inhibitory subclasses)

and saves:
- matrix CSVs (cells x genes)
- label/metadata CSVs
- clustered cell×gene PNGs using simple k-means clustering from
  ``aind_hcr_qc.viz.plot_cell_x_gene_clustered``.

10x-HMB is loaded from ABC WMB-10X with configurable region and
expression scale (raw or log2).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import aind_hcr_qc.viz as viz
from aind_hcr_data_loader import get_hcr_dataset_pairwise

import atlas_compare
import cluster_validation_utils


# -----------------------------------------------------------------------------
# Paths and defaults
# -----------------------------------------------------------------------------

ABC_ATLAS_DIR = Path("/root/capsule/data/abc_atlas")
V1_CELLS_CSV = Path("/root/capsule/code/v1_merfish_cells.csv")
SS_PATH = Path("/root/capsule/scratch/mouse_VISp_gene_expression_matrices_2018-06-14")
DATA_DIR = Path("/root/capsule/data")
DEFAULT_OUT_DIR = Path("/root/capsule/scratch/reference_atlas_cellxgene")
REFERENCE_MATRIX_CACHE_DIR = DEFAULT_OUT_DIR / "_cached_reference_matrices"

DROP_LAYERS = ["VISp6a", "VISp6b"]
INHIBITORY_REF_CLASSES = ["07 CTX-MGE GABA", "06 CTX-CGE GABA"]
TENX_INHIBITORY_CLASSES = ["06 CTX-CGE GABA", "07 CTX-MGE GABA"]
SUBCLASS_ORDER = ["Pvalb", "Sst", "Vip", "Lamp5"]

# HCR panel genes used to subset reference atlas matrices (both MERFISH + TASIC).
# Keep this list in the desired output column order.
HCR_PANEL_GENES: list[str] = [
    "Calb1",
    "Calb2",
    "Cck",
    "Chat",
    "Crh",
    "Gad2",
    "Hpse",
    "Lamp5",
    "Mme",
    "Ndnf",
    "Npy",
    "Pdyn",
    "Penk",
    "Pthlh",
    "Pvalb",
    "Reln",
    "Slc17a7",
    "Sst",
    "Tac1",
    "Tac2",
    "Vip",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _save_fig(fig, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _fig_size_for_matrix(n_cells: int, n_genes: int) -> tuple[float, float]:
    # Keep readable for medium matrices while preventing absurdly large figures.
    width = float(np.clip(4.0 + 0.06 * n_genes, 8.0, 42.0))
    height = float(np.clip(4.0 + 0.015 * n_cells, 8.0, 48.0))
    return width, height


def _sort_columns_natural(df: pd.DataFrame) -> pd.DataFrame:
    import re

    def _key(s: str):
        return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(s))]

    return df.reindex(columns=sorted(df.columns, key=_key))


def _load_v1_merfish_cells() -> pd.DataFrame:
    if not V1_CELLS_CSV.exists():
        raise FileNotFoundError(f"V1 MERFISH cells CSV not found: {V1_CELLS_CSV}")

    v1_cells = pd.read_csv(V1_CELLS_CSV, index_col=0)
    v1_cells = v1_cells[~v1_cells["parcellation_substructure"].isin(DROP_LAYERS)].copy()
    return v1_cells


def _load_all_merfish_gene_symbols(abc_cache_dir: Path) -> list[str]:
    # Load the full MERFISH gene panel from the atlas cache h5ad var metadata.
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
    import anndata as ad

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        abc_cache = AbcProjectCache.from_cache_dir(abc_cache_dir)

    h5ad_path = abc_cache.get_file_path(
        directory="MERFISH-C57BL6J-638850",
        file_name="C57BL6J-638850/log2",
    )
    adata = ad.read_h5ad(h5ad_path, backed="r")
    gene_symbols = adata.var["gene_symbol"].astype(str).tolist()
    adata.file.close()
    del adata

    # Keep first occurrence if duplicates exist.
    gene_symbols = list(dict.fromkeys(gene_symbols))
    return gene_symbols


def _plot_clustered_cellxgene(
    cxg: pd.DataFrame,
    out_png: Path,
    title: str,
    k: int,
    dpi: int,
    clip_max: float,
    max_plot_cells: int,
    max_plot_genes: int,
    random_seed: int,
) -> None:
    cxg_plot = cxg.copy()

    if max_plot_cells > 0 and len(cxg_plot) > max_plot_cells:
        cxg_plot = cxg_plot.sample(n=max_plot_cells, random_state=random_seed)

    if max_plot_genes > 0 and cxg_plot.shape[1] > max_plot_genes:
        variances = cxg_plot.var(axis=0).sort_values(ascending=False)
        keep_cols = variances.index[:max_plot_genes]
        cxg_plot = cxg_plot.loc[:, keep_cols]

    cxg_plot = _sort_columns_natural(cxg_plot)
    fig_size = _fig_size_for_matrix(cxg_plot.shape[0], cxg_plot.shape[1])

    n_clusters = max(1, min(k, cxg_plot.shape[0]))
    km = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=20)
    cluster_ids = km.fit_predict(cxg_plot.values)

    centers = pd.DataFrame(km.cluster_centers_, columns=cxg_plot.columns)
    marker_genes = [g for g in SUBCLASS_ORDER if g in centers.columns]

    cluster_rows = []
    for cid in range(n_clusters):
        if marker_genes:
            vals = centers.loc[cid, marker_genes].astype(float)
            top_marker = str(vals.idxmax())
            top_score = float(vals.max())
            marker_rank = marker_genes.index(top_marker)
        else:
            top_marker = "NA"
            top_score = 0.0
            marker_rank = len(SUBCLASS_ORDER)

        cluster_rows.append(
            {
                "cluster_id": cid,
                "marker_rank": marker_rank,
                "top_marker": top_marker,
                "top_score": top_score,
            }
        )

    cluster_order = (
        pd.DataFrame(cluster_rows)
        .sort_values(["marker_rank", "top_score", "cluster_id"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    cluster_to_prefixed_label = {}
    for rank, row in cluster_order.iterrows():
        cid = int(row["cluster_id"])
        marker = str(row["top_marker"])
        cluster_to_prefixed_label[cid] = f"{rank:02d}__{marker}__K{cid:02d}"

    labels = pd.Series(cluster_ids, index=cxg_plot.index).map(cluster_to_prefixed_label)

    fig, _, _ = viz.plot_cell_x_gene_labeled(
        cxg_plot,
        labels=labels,
        clip_range=(0, clip_max),
        fig_size=fig_size,
        add_cluster_labels=True,
        label_fontsize=7,
        title=f"{title} (ordered by {SUBCLASS_ORDER})",
    )

    # Remove rank/marker prefixes from the displayed y-axis labels while
    # preserving the subclass-prioritized ordering.
    ax = fig.axes[0]
    for txt in ax.texts:
        s = txt.get_text()
        if "__" in s:
            parts = s.split("__", 2)
            if len(parts) == 3:
                txt.set_text(parts[2])

    _save_fig(fig, out_png, dpi=dpi)


def _write_run_metadata(out_dir: Path, metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))


def _cache_dir_for(dataset_name: str, cache_payload: dict) -> Path:
    key_json = json.dumps(cache_payload, sort_keys=True)
    key_hash = hashlib.md5(key_json.encode("utf-8")).hexdigest()[:12]
    return REFERENCE_MATRIX_CACHE_DIR / dataset_name / key_hash


def _try_read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _subset_to_hcr_panel(cxg: pd.DataFrame, panel_genes: list[str]) -> pd.DataFrame:
    if len(panel_genes) == 0:
        raise ValueError("HCR_PANEL_GENES is empty. Define panel genes at the top of this script.")

    present = [g for g in panel_genes if g in cxg.columns]
    missing = [g for g in panel_genes if g not in cxg.columns]
    if len(present) == 0:
        raise ValueError(
            "None of HCR_PANEL_GENES were found in the matrix columns. "
            f"Panel size={len(panel_genes)}"
        )

    if missing:
        print(f"  Note: {len(missing)} panel genes not present and were skipped: {missing}")

    return cxg.loc[:, present].copy()


# -----------------------------------------------------------------------------
# MERFISH
# -----------------------------------------------------------------------------


def run_merfish(
    out_root: Path,
    label_level: str,
    min_label_cells: int,
    merfish_expression_scale: str,
    k: int,
    dpi: int,
    clip_max: float,
    max_plot_cells: int,
    max_plot_genes: int,
    random_seed: int,
) -> None:
    print("\n" + "=" * 72)
    print("MERFISH: collecting VISp inhibitory reference cell×gene")
    print("=" * 72)

    out_dir = out_root / "merfish"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_payload = {
        "dataset": "merfish",
        "label_level": label_level,
        "min_label_cells": int(min_label_cells),
        "expression_scale": merfish_expression_scale,
        "drop_layers": DROP_LAYERS,
        "ref_classes": INHIBITORY_REF_CLASSES,
        "panel_genes": HCR_PANEL_GENES,
        "v1_cells_csv": str(V1_CELLS_CSV),
    }
    cache_dir = _cache_dir_for("merfish", cache_payload)
    cache_cxg_path = cache_dir / "cell_x_gene.parquet"
    cache_labels_path = cache_dir / f"labels_{label_level}.parquet"
    used_cache = False

    cached_cxg = _try_read_parquet(cache_cxg_path)
    cached_labels = _try_read_parquet(cache_labels_path)
    if cached_cxg is not None and cached_labels is not None:
        used_cache = True
        ref_counts = cached_cxg.copy()
        if label_level in cached_labels.columns:
            ref_labels = cached_labels[label_level].copy()
        else:
            ref_labels = cached_labels.iloc[:, 0].rename(label_level)
        print(f"[MERFISH cache] Loaded cached matrices from {cache_dir}")
    else:
        print("[MERFISH 1/4] Loading V1 VISp cell index...")
        v1_cells = _load_v1_merfish_cells()
        print(f"  V1 cells retained: {len(v1_cells):,}")

        print("[MERFISH 2/4] Using HCR_PANEL_GENES for atlas subsetting...")
        print(f"  HCR panel genes requested: {len(HCR_PANEL_GENES):,}")
        print(f"  Expression scale: {merfish_expression_scale}")

        print("[MERFISH 3/4] Loading inhibitory reference counts from ABC Atlas...")
        ref_counts, ref_labels = atlas_compare.load_abc_merfish_reference(
            abc_cache_dir=ABC_ATLAS_DIR,
            genes=HCR_PANEL_GENES,
            cell_index=v1_cells.index,
            ref_classes=INHIBITORY_REF_CLASSES,
            label_level=label_level,
            min_label_cells=min_label_cells,
            expression_scale=merfish_expression_scale,
            save_dir=None,
        )

        ref_counts = ref_counts.loc[:, ~ref_counts.columns.duplicated()].copy()
        ref_counts = ref_counts.fillna(0)
        ref_counts = _subset_to_hcr_panel(ref_counts, HCR_PANEL_GENES)
        ref_labels = ref_labels.reindex(ref_counts.index)

        _write_parquet(ref_counts, cache_cxg_path)
        _write_parquet(ref_labels.to_frame(name=label_level), cache_labels_path)
        print(f"[MERFISH cache] Saved cached matrices -> {cache_dir}")

    print(
        f"  Reference matrix: {ref_counts.shape[0]:,} cells x "
        f"{ref_counts.shape[1]:,} genes"
    )

    print("[MERFISH 4/4] Writing outputs and clustered PNG...")
    ref_counts.to_csv(out_dir / "cell_x_gene.csv")
    ref_labels.to_frame(name=label_level).to_csv(out_dir / f"labels_{label_level}.csv")

    _plot_clustered_cellxgene(
        cxg=ref_counts,
        out_png=out_dir / "cell_x_gene_clustered_kmeans.png",
        title="MERFISH VISp inhibitory reference (k-means clustered)",
        k=k,
        dpi=dpi,
        clip_max=clip_max,
        max_plot_cells=max_plot_cells,
        max_plot_genes=max_plot_genes,
        random_seed=random_seed,
    )

    _write_run_metadata(
        out_dir,
        {
            "dataset": "MERFISH",
            "cell_filter": {
                "v1_cells_csv": str(V1_CELLS_CSV),
                "drop_layers": DROP_LAYERS,
                "ref_classes": INHIBITORY_REF_CLASSES,
            },
            "label_level": label_level,
            "min_label_cells": min_label_cells,
            "expression_scale": merfish_expression_scale,
            "cache": {
                "cache_dir": str(cache_dir),
                "used_cache": used_cache,
            },
            "hcr_panel_genes": HCR_PANEL_GENES,
            "matrix_shape": [int(ref_counts.shape[0]), int(ref_counts.shape[1])],
            "plot": {
                "cluster_method": "kmeans",
                "k": k,
                "clip_range": [0, clip_max],
                "max_plot_cells": max_plot_cells,
                "max_plot_genes": max_plot_genes,
                "random_seed": random_seed,
                "dpi": dpi,
            },
        },
    )

    gc.collect()
    print(f"  Saved MERFISH outputs -> {out_dir}")


# -----------------------------------------------------------------------------
# TASIC Smart-seq
# -----------------------------------------------------------------------------


def run_tasic(
    out_root: Path,
    tasic_layer: str,
    k: int,
    dpi: int,
    clip_max: float,
    max_plot_cells: int,
    max_plot_genes: int,
    random_seed: int,
) -> None:
    print("\n" + "=" * 72)
    print("TASIC: collecting VISp inhibitory reference cell×gene")
    print("=" * 72)

    out_dir = out_root / "tasic"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_payload = {
        "dataset": "tasic",
        "source_path": str(SS_PATH),
        "tasic_layer": tasic_layer,
        "panel_genes": HCR_PANEL_GENES,
        "filter": "inhibitory",
    }
    cache_dir = _cache_dir_for("tasic", cache_payload)
    cache_cxg_path = cache_dir / "cell_x_gene.parquet"
    cache_meta_path = cache_dir / "cell_metadata.parquet"
    used_cache = False

    cached_cxg = _try_read_parquet(cache_cxg_path)
    cached_meta = _try_read_parquet(cache_meta_path)
    if cached_cxg is not None:
        used_cache = True
        cxg = cached_cxg.copy()
        meta_df = cached_meta.copy() if cached_meta is not None else pd.DataFrame(index=cxg.index)
        print(f"[TASIC cache] Loaded cached matrices from {cache_dir}")
    else:
        print("[TASIC 1/3] Loading VISp Smart-seq expression...")
        adata_all = cluster_validation_utils.load_visp_expression(
            SS_PATH,
            genes=HCR_PANEL_GENES,
            layer=tasic_layer,
        )

        print("[TASIC 2/3] Filtering to inhibitory cells...")
        views = cluster_validation_utils.make_filtered_views_for_smartseq(adata_all)
        adata_inh = views["inhibitory"]
        print(f"  Inhibitory matrix: {adata_inh.n_obs:,} cells x {adata_inh.n_vars:,} genes")

        X = adata_inh.X if not hasattr(adata_inh.X, "toarray") else adata_inh.X.toarray()
        cxg = pd.DataFrame(
            np.asarray(X),
            index=adata_inh.obs_names.astype(str),
            columns=adata_inh.var_names.astype(str),
        )
        cxg = cxg.fillna(0)
        cxg = _subset_to_hcr_panel(cxg, HCR_PANEL_GENES)

        obs_cols = [
            c for c in ["class", "subclass", "cluster", "brain_region", "brain_subregion"]
            if c in adata_inh.obs.columns
        ]
        meta_df = adata_inh.obs.loc[:, obs_cols].copy() if obs_cols else pd.DataFrame(index=cxg.index)
        meta_df = meta_df.reindex(cxg.index)

        _write_parquet(cxg, cache_cxg_path)
        _write_parquet(meta_df, cache_meta_path)
        print(f"[TASIC cache] Saved cached matrices -> {cache_dir}")

        del adata_all, adata_inh, views, X
        gc.collect()

    print("[TASIC 3/3] Writing outputs and clustered PNG...")
    cxg.to_csv(out_dir / "cell_x_gene.csv")
    if not meta_df.empty:
        meta_df.to_csv(out_dir / "cell_metadata.csv")

    _plot_clustered_cellxgene(
        cxg=cxg,
        out_png=out_dir / "cell_x_gene_clustered_kmeans.png",
        title="TASIC VISp inhibitory reference (k-means clustered)",
        k=k,
        dpi=dpi,
        clip_max=clip_max,
        max_plot_cells=max_plot_cells,
        max_plot_genes=max_plot_genes,
        random_seed=random_seed,
    )

    _write_run_metadata(
        out_dir,
        {
            "dataset": "TASIC",
            "source_path": str(SS_PATH),
            "tasic_layer": tasic_layer,
            "filter": "make_filtered_views_for_smartseq()['inhibitory']",
            "cache": {
                "cache_dir": str(cache_dir),
                "used_cache": used_cache,
            },
            "hcr_panel_genes": HCR_PANEL_GENES,
            "matrix_shape": [int(cxg.shape[0]), int(cxg.shape[1])],
            "plot": {
                "cluster_method": "kmeans",
                "k": k,
                "clip_range": [0, clip_max],
                "max_plot_cells": max_plot_cells,
                "max_plot_genes": max_plot_genes,
                "random_seed": random_seed,
                "dpi": dpi,
            },
        },
    )

    # Explicitly release large objects.
    del cxg, meta_df
    gc.collect()
    print(f"  Saved TASIC outputs -> {out_dir}")


# -----------------------------------------------------------------------------
# 10x-HMB (ABC WMB-10X)
# -----------------------------------------------------------------------------


def run_10x_hmb(
    out_root: Path,
    label_level: str,
    min_label_cells: int,
    tenx_region: str,
    tenx_expression_scale: str,
    tenx_ref_classes: list[str] | None,
    tenx_exclude_supertype_substrings: list[str] | None,
    tenx_min_supertype_cells: int | None,
    k: int,
    dpi: int,
    clip_max: float,
    max_plot_cells: int,
    max_plot_genes: int,
    random_seed: int,
) -> None:
    print("\n" + "=" * 72)
    print("10x-HMB: collecting WMB-10X reference cell×gene")
    print("=" * 72)

    out_dir = out_root / "10x-hmb"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_payload = {
        "dataset": "10x-hmb",
        "region": tenx_region,
        "expression_scale": tenx_expression_scale,
        "ref_classes": tenx_ref_classes,
        "exclude_supertype_substrings": tenx_exclude_supertype_substrings,
        "min_supertype_cells": tenx_min_supertype_cells,
        "label_level": label_level,
        "min_label_cells": int(min_label_cells),
        "panel_genes": HCR_PANEL_GENES,
    }
    cache_dir = _cache_dir_for("10x-hmb", cache_payload)
    cache_cxg_path = cache_dir / "cell_x_gene.parquet"
    cache_labels_path = cache_dir / f"labels_{label_level}.parquet"
    used_cache = False

    cached_cxg = _try_read_parquet(cache_cxg_path)
    cached_labels = _try_read_parquet(cache_labels_path)
    if cached_cxg is not None and cached_labels is not None:
        used_cache = True
        ref_counts = cached_cxg.copy()
        if label_level in cached_labels.columns:
            ref_labels = cached_labels[label_level].copy()
        else:
            ref_labels = cached_labels.iloc[:, 0].rename(label_level)
        print(f"[10x-HMB cache] Loaded cached matrices from {cache_dir}")
    else:
        print("[10x-HMB 1/3] Loading WMB-10X reference counts from ABC Atlas...")
        print(f"  Region: {tenx_region}")
        print(f"  Expression scale: {tenx_expression_scale}")
        if tenx_ref_classes:
            print(f"  Class filter: {tenx_ref_classes}")
        if tenx_exclude_supertype_substrings:
            print(f"  Exclude supertype substrings: {tenx_exclude_supertype_substrings}")
        if tenx_min_supertype_cells is not None:
            print(f"  Min supertype cells: {tenx_min_supertype_cells}")

        ref_counts, ref_labels = atlas_compare.load_abc_wmb_10x_reference(
            abc_cache_dir=ABC_ATLAS_DIR,
            genes=HCR_PANEL_GENES,
            region_of_interest=tenx_region,
            ref_classes=tenx_ref_classes,
            exclude_supertype_substrings=tenx_exclude_supertype_substrings,
            min_supertype_cells=tenx_min_supertype_cells,
            label_level=label_level,
            min_label_cells=min_label_cells,
            expression_scale=tenx_expression_scale,
            save_dir=None,
        )

        ref_counts = ref_counts.loc[:, ~ref_counts.columns.duplicated()].copy()
        ref_counts = ref_counts.fillna(0)
        ref_counts = _subset_to_hcr_panel(ref_counts, HCR_PANEL_GENES)
        ref_labels = ref_labels.reindex(ref_counts.index)

        _write_parquet(ref_counts, cache_cxg_path)
        _write_parquet(ref_labels.to_frame(name=label_level), cache_labels_path)
        print(f"[10x-HMB cache] Saved cached matrices -> {cache_dir}")

    print(
        f"  Reference matrix: {ref_counts.shape[0]:,} cells x "
        f"{ref_counts.shape[1]:,} genes"
    )

    print("[10x-HMB 2/3] Writing outputs...")
    ref_counts.to_csv(out_dir / "cell_x_gene.csv")
    ref_labels.to_frame(name=label_level).to_csv(out_dir / f"labels_{label_level}.csv")

    print("[10x-HMB 3/3] Writing clustered PNG...")
    _plot_clustered_cellxgene(
        cxg=ref_counts,
        out_png=out_dir / "cell_x_gene_clustered_kmeans.png",
        title=f"10x-HMB {tenx_region} reference (k-means clustered)",
        k=k,
        dpi=dpi,
        clip_max=clip_max,
        max_plot_cells=max_plot_cells,
        max_plot_genes=max_plot_genes,
        random_seed=random_seed,
    )

    _write_run_metadata(
        out_dir,
        {
            "dataset": "10x-HMB",
            "source": "ABC Atlas WMB-10X",
            "region_of_interest": tenx_region,
            "expression_scale": tenx_expression_scale,
            "ref_classes": tenx_ref_classes,
            "exclude_supertype_substrings": tenx_exclude_supertype_substrings,
            "min_supertype_cells": tenx_min_supertype_cells,
            "cache": {
                "cache_dir": str(cache_dir),
                "used_cache": used_cache,
            },
            "label_level": label_level,
            "min_label_cells": min_label_cells,
            "hcr_panel_genes": HCR_PANEL_GENES,
            "matrix_shape": [int(ref_counts.shape[0]), int(ref_counts.shape[1])],
            "plot": {
                "cluster_method": "kmeans",
                "k": k,
                "clip_range": [0, clip_max],
                "max_plot_cells": max_plot_cells,
                "max_plot_genes": max_plot_genes,
                "random_seed": random_seed,
                "dpi": dpi,
            },
        },
    )

    gc.collect()
    print(f"  Saved 10x-HMB outputs -> {out_dir}")


# -----------------------------------------------------------------------------
# HCR
# -----------------------------------------------------------------------------


def run_hcr(
    out_root: Path,
    mouse_id: str,
    k: int,
    dpi: int,
    clip_max: float,
    max_plot_cells: int,
    max_plot_genes: int,
    random_seed: int,
) -> None:
    print("\n" + "=" * 72)
    print(f"HCR: collecting inhibitory cell×gene for mouse {mouse_id}")
    print("=" * 72)

    out_dir = out_root / "hcr" / str(mouse_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_payload = {
        "dataset": "hcr",
        "mouse_id": str(mouse_id),
        "data_dir": str(DATA_DIR),
        "panel_genes": HCR_PANEL_GENES,
        "source": "load_inhibitory_cells(unmixed=True, all_spots=False)",
    }
    cache_dir = _cache_dir_for("hcr", cache_payload)
    cache_cxg_path = cache_dir / "cell_x_gene.parquet"
    cache_meta_path = cache_dir / "cell_metadata.parquet"
    used_cache = False

    cached_cxg = _try_read_parquet(cache_cxg_path)
    cached_meta = _try_read_parquet(cache_meta_path)
    if cached_cxg is not None:
        used_cache = True
        cxg = cached_cxg.copy()
        meta_df = cached_meta.copy() if cached_meta is not None else pd.DataFrame(index=cxg.index)
        print(f"[HCR cache] Loaded cached matrices from {cache_dir}")
    else:
        print("[HCR 1/3] Loading pairwise HCR dataset...")
        _, pw_ds, _ = get_hcr_dataset_pairwise(
            mouse_id=mouse_id,
            data_dir=DATA_DIR,
            load_spots=False,
            return_removed=False,
            coreg_cells_only=False,
        )

        print("[HCR 2/3] Loading inhibitory cell×gene matrix...")
        adata = pw_ds.load_inhibitory_cells(unmixed=True, all_spots=False, as_anndata=True)
        print(f"  Raw inhibitory matrix: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        cxg = pd.DataFrame(
            np.asarray(X),
            index=adata.obs_names.astype(str),
            columns=adata.var_names.astype(str),
        ).fillna(0)

        cxg = _subset_to_hcr_panel(cxg, HCR_PANEL_GENES)
        print(f"  Panel-subset matrix: {cxg.shape[0]:,} cells x {cxg.shape[1]:,} genes")

        obs_cols = [c for c in ["subclass", "cluster", "section", "x", "y"] if c in adata.obs.columns]
        meta_df = adata.obs.loc[:, obs_cols].copy() if obs_cols else pd.DataFrame(index=cxg.index)
        meta_df = meta_df.reindex(cxg.index)

        _write_parquet(cxg, cache_cxg_path)
        _write_parquet(meta_df, cache_meta_path)
        print(f"[HCR cache] Saved cached matrices -> {cache_dir}")

        del adata, pw_ds, X
        gc.collect()

    print("[HCR 3/3] Writing outputs and clustered PNG...")
    cxg.to_csv(out_dir / "cell_x_gene.csv")
    if not meta_df.empty:
        meta_df.to_csv(out_dir / "cell_metadata.csv")

    _plot_clustered_cellxgene(
        cxg=cxg,
        out_png=out_dir / "cell_x_gene_clustered_kmeans.png",
        title=f"HCR mouse {mouse_id} inhibitory (k-means clustered)",
        k=k,
        dpi=dpi,
        clip_max=clip_max,
        max_plot_cells=max_plot_cells,
        max_plot_genes=max_plot_genes,
        random_seed=random_seed,
    )

    _write_run_metadata(
        out_dir,
        {
            "dataset": "HCR",
            "mouse_id": str(mouse_id),
            "data_dir": str(DATA_DIR),
            "source": "get_hcr_dataset_pairwise + load_inhibitory_cells(unmixed=True, all_spots=False)",
            "cache": {
                "cache_dir": str(cache_dir),
                "used_cache": used_cache,
            },
            "hcr_panel_genes": HCR_PANEL_GENES,
            "matrix_shape": [int(cxg.shape[0]), int(cxg.shape[1])],
            "plot": {
                "cluster_method": "kmeans",
                "k": k,
                "clip_range": [0, clip_max],
                "subclass_order": SUBCLASS_ORDER,
                "max_plot_cells": max_plot_cells,
                "max_plot_genes": max_plot_genes,
                "random_seed": random_seed,
                "dpi": dpi,
            },
        },
    )

    del cxg, meta_df
    gc.collect()
    print(f"  Saved HCR outputs -> {out_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect VISp inhibitory reference atlas cell×gene matrices and clustered PNGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["10x-hmb", "merfish", "tasic", "hcr"],
        choices=["merfish", "tasic", "hcr", "10x-hmb"],
        help="Reference datasets to run.",
    )
    parser.add_argument(
        "--hcr-mouse-id",
        default="790322",
        help="Mouse ID for HCR dataset collection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output root directory.",
    )
    parser.add_argument(
        "--label-level",
        default="supertype",
        choices=["class", "subclass", "supertype", "cluster"],
        help="MERFISH reference label granularity.",
    )
    parser.add_argument(
        "--min-label-cells",
        type=int,
        default=10,
        help="MERFISH minimum cells per kept label.",
    )
    parser.add_argument(
        "--merfish-expression-scale",
        default="log2",
        choices=["log2", "raw"],
        help="MERFISH expression matrix to load from ABC Atlas.",
    )
    parser.add_argument(
        "--tenx-region",
        default="VIS",
        help="Region acronym for WMB-10X (e.g., VIS).",
    )
    parser.add_argument(
        "--tenx-expression-scale",
        default="log2",
        choices=["log2", "raw"],
        help="WMB-10X expression matrix to load from ABC Atlas.",
    )
    parser.add_argument(
        "--tenx-ref-classes",
        nargs="+",
        default=TENX_INHIBITORY_CLASSES,
        help="Class labels to keep for WMB-10X filtering.",
    )
    parser.add_argument(
        "--tenx-exclude-supertype-substrings",
        nargs="+",
        default=["L6"],
        help="Drop WMB-10X cells whose supertype contains any of these substrings.",
    )
    parser.add_argument(
        "--tenx-min-supertype-cells",
        type=int,
        default=10,
        help="Keep only WMB-10X supertypes with at least this many cells.",
    )
    parser.add_argument(
        "--tasic-layer",
        default="exon",
        choices=["sum", "exon", "intron"],
        help="TASIC expression layer.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="k for k-means cell clustering in cell×gene plot.",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=200.0,
        help="Upper bound of display clip range (lower bound is always 0).",
    )
    parser.add_argument(
        "--max-plot-cells",
        type=int,
        default=15000,
        help="If > 0, subsample cells to this count before plotting.",
    )
    parser.add_argument(
        "--max-plot-genes",
        type=int,
        default=256,
        help="If > 0, keep top-variance genes up to this count before plotting.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for plotting subsampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 72)
    print("Reference atlas cell×gene collector")
    print(f"Datasets: {args.datasets}")
    print(f"Output root: {out_root}")
    print("#" * 72)

    if "10x-hmb" in args.datasets:
        run_10x_hmb(
            out_root=out_root,
            label_level=args.label_level,
            min_label_cells=args.min_label_cells,
            tenx_region=args.tenx_region,
            tenx_expression_scale=args.tenx_expression_scale,
            tenx_ref_classes=args.tenx_ref_classes,
            tenx_exclude_supertype_substrings=args.tenx_exclude_supertype_substrings,
            tenx_min_supertype_cells=args.tenx_min_supertype_cells,
            k=args.k,
            dpi=args.dpi,
            clip_max=args.clip_max,
            max_plot_cells=args.max_plot_cells,
            max_plot_genes=args.max_plot_genes,
            random_seed=args.random_seed,
        )

    if "merfish" in args.datasets:
        run_merfish(
            out_root=out_root,
            label_level=args.label_level,
            min_label_cells=args.min_label_cells,
            merfish_expression_scale=args.merfish_expression_scale,
            k=args.k,
            dpi=args.dpi,
            clip_max=args.clip_max,
            max_plot_cells=args.max_plot_cells,
            max_plot_genes=args.max_plot_genes,
            random_seed=args.random_seed,
        )

    if "tasic" in args.datasets:
        run_tasic(
            out_root=out_root,
            tasic_layer=args.tasic_layer,
            k=args.k,
            dpi=args.dpi,
            clip_max=args.clip_max,
            max_plot_cells=args.max_plot_cells,
            max_plot_genes=args.max_plot_genes,
            random_seed=args.random_seed,
        )

    if "hcr" in args.datasets:
        run_hcr(
            out_root=out_root,
            mouse_id=args.hcr_mouse_id,
            k=args.k,
            dpi=args.dpi,
            clip_max=args.clip_max,
            max_plot_cells=args.max_plot_cells,
            max_plot_genes=args.max_plot_genes,
            random_seed=args.random_seed,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
