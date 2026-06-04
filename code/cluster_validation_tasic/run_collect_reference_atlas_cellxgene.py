"""Collect VISp inhibitory reference atlas cell×gene matrices and clustered PNGs.

This script builds reference cell×gene matrices for:
- MERFISH (ABC Atlas, VISp inhibitory classes)
- TASIC Smart-seq (VISp inhibitory subclasses)

and saves:
- matrix CSVs (cells x genes)
- label/metadata CSVs
- clustered cell×gene PNGs using simple k-means clustering from
  ``aind_hcr_qc.viz.plot_cell_x_gene_clustered``.

10x-HMB is intentionally left as a stub for later implementation.
"""

from __future__ import annotations

import argparse
import gc
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

import atlas_compare
import cluster_validation_utils


# -----------------------------------------------------------------------------
# Paths and defaults
# -----------------------------------------------------------------------------

ABC_ATLAS_DIR = Path("/root/capsule/data/abc_atlas")
V1_CELLS_CSV = Path("/root/capsule/code/v1_merfish_cells.csv")
SS_PATH = Path("/root/capsule/scratch/mouse_VISp_gene_expression_matrices_2018-06-14")
DEFAULT_OUT_DIR = Path("/root/capsule/scratch/reference_atlas_cellxgene")

DROP_LAYERS = ["VISp6a", "VISp6b"]
INHIBITORY_REF_CLASSES = ["07 CTX-MGE GABA", "06 CTX-CGE GABA"]
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
        labels=labels.values,
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

    print("[MERFISH 1/4] Loading V1 VISp cell index...")
    v1_cells = _load_v1_merfish_cells()
    print(f"  V1 cells retained: {len(v1_cells):,}")

    print("[MERFISH 2/4] Using HCR_PANEL_GENES for atlas subsetting...")
    print(f"  HCR panel genes requested: {len(HCR_PANEL_GENES):,}")

    print("[MERFISH 3/4] Loading inhibitory reference counts from ABC Atlas...")
    ref_counts, ref_labels = atlas_compare.load_abc_merfish_reference(
        abc_cache_dir=ABC_ATLAS_DIR,
        genes=HCR_PANEL_GENES,
        cell_index=v1_cells.index,
        ref_classes=INHIBITORY_REF_CLASSES,
        label_level=label_level,
        min_label_cells=min_label_cells,
        save_dir=None,
    )

    ref_counts = ref_counts.loc[:, ~ref_counts.columns.duplicated()].copy()
    ref_counts = ref_counts.fillna(0)
    ref_counts = _subset_to_hcr_panel(ref_counts, HCR_PANEL_GENES)
    ref_labels = ref_labels.reindex(ref_counts.index)

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

    print("[TASIC 3/3] Writing outputs and clustered PNG...")
    X = adata_inh.X if not hasattr(adata_inh.X, "toarray") else adata_inh.X.toarray()
    cxg = pd.DataFrame(
        np.asarray(X),
        index=adata_inh.obs_names.astype(str),
        columns=adata_inh.var_names.astype(str),
    )
    cxg = cxg.fillna(0)
    cxg = _subset_to_hcr_panel(cxg, HCR_PANEL_GENES)

    cxg.to_csv(out_dir / "cell_x_gene.csv")

    obs_cols = [c for c in ["class", "subclass", "cluster", "brain_region", "brain_subregion"] if c in adata_inh.obs.columns]
    adata_inh.obs.loc[:, obs_cols].to_csv(out_dir / "cell_metadata.csv")

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
    del adata_all, adata_inh, views, cxg, X
    gc.collect()
    print(f"  Saved TASIC outputs -> {out_dir}")


# -----------------------------------------------------------------------------
# 10x-HMB stub
# -----------------------------------------------------------------------------


def run_10x_hmb_stub(out_root: Path) -> None:
    out_dir = out_root / "10x-hmb"
    out_dir.mkdir(parents=True, exist_ok=True)
    msg = (
        "10x-HMB implementation is pending.\n"
        "Planned behavior: load VISp inhibitory cells, export cell_x_gene.csv, "
        "and save k-means clustered cell_x_gene_clustered_kmeans.png.\n"
    )
    (out_dir / "README_TODO.txt").write_text(msg)
    print(f"10x-HMB placeholder written -> {out_dir / 'README_TODO.txt'}")


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
        default=["merfish", "tasic"],
        choices=["merfish", "tasic", "10x-hmb"],
        help="Reference datasets to run.",
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
        default=50.0,
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

    if "merfish" in args.datasets:
        run_merfish(
            out_root=out_root,
            label_level=args.label_level,
            min_label_cells=args.min_label_cells,
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

    if "10x-hmb" in args.datasets:
        run_10x_hmb_stub(out_root=out_root)

    print("\nDone.")


if __name__ == "__main__":
    main()
