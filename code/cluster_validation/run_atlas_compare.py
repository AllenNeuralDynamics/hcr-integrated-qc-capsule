"""Run the atlas-comparison pipeline for a single mouse and save all figures.

Replicates the full workflow from 20260429-match-clusters-to-ref.ipynb
as a standalone command-line script.

Example usage
-------------
    python run_atlas_compare.py --mouse-id 782149
    python run_atlas_compare.py --mouse-id 782149 --dpi 200 --label-level supertype
    python run_atlas_compare.py --mouse-id 782149 --output-dir /tmp/atlas_out
"""

import argparse
import gc
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aind_hcr_data_loader.codeocean_utils import (
    MouseRecord,
    attach_mouse_record_to_workstation,
    print_attach_results,
)
from aind_hcr_data_loader.hcr_dataset import create_hcr_dataset_from_schema
from aind_hcr_data_loader.pairwise_dataset import create_pairwise_unmixing_dataset

from aind_hcr_qc.constants import Z1_CHANNEL_CMAP_VIBRANT

import atlas_compare

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------

CATALOG_BASE    = Path("/src/ophys-mfish-dataset-catalog/mice")
DATA_DIR        = Path("/root/capsule/data")
ABC_ATLAS_DIR   = Path("/root/capsule/data/abc_atlas")
V1_CELLS_CSV    = Path("/root/capsule/code/v1_merfish_cells.csv")
SCRATCH_BASE    = Path("/root/capsule/scratch/ref_atlas_validation")
REF_CACHE_DIR   = SCRATCH_BASE / "_ref_cache"  # shared across all mice

# ---------------------------------------------------------------------------
# Reference configuration  (mirrors notebook defaults)
# ---------------------------------------------------------------------------

DROP_LAYERS      = ["VISp6a", "VISp6b"]
REF_CLASSES      = ["07 CTX-MGE GABA", "06 CTX-CGE GABA"]
MIN_LABEL_CELLS  = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data(mouse_id: str, legacy: bool = False):
    """Attach assets and build the dataset for *mouse_id*.

    Parameters
    ----------
    legacy:
        When True, return the base HCRDataset instead of wrapping it in a
        PairwiseUnmixingDataset.  Use this to run the pipeline on the original
        (pre-pairwise) unmixed data.
    """
    catalog_path = CATALOG_BASE / f"{mouse_id}.json"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    record  = MouseRecord.from_json_file(catalog_path)
    results = attach_mouse_record_to_workstation(record)
    print_attach_results(results)

    dataset = create_hcr_dataset_from_schema(catalog_path, DATA_DIR)
    dataset.summary()

    if legacy:
        print("[legacy mode] Using base HCRDataset (skipping pairwise unmixing asset).")
        return dataset

    pairwise_asset_name = record.derived_assets.get("pairwise_unmixing")
    if pairwise_asset_name is None:
        raise ValueError(
            f"Mouse {mouse_id} has no 'pairwise_unmixing' asset in the catalog — "
            "this pipeline requires pairwise-unmixed data."
        )

    pairwise_asset_path = DATA_DIR / pairwise_asset_name
    # Some pipeline outputs nest data under a "pairwise_unmixing" subfolder.
    if (pairwise_asset_path / "pairwise_unmixing").exists():
        pairwise_asset_path = pairwise_asset_path / "pairwise_unmixing"

    pw_ds = create_pairwise_unmixing_dataset(
        mouse_id=mouse_id,
        pairwise_asset_path=pairwise_asset_path,
        source_dataset=dataset,
        min_dist=1,
    )
    pw_ds.summary()

    return pw_ds


def _load_spots(pw_ds):
    """Load the mixed and unmixed spot tables and apply cell/validity filters.

    Returns
    -------
    mixed_spots : DataFrame
        Mixed spots with cell_id > 0.
    unmixed_spots_all : DataFrame
        All unmixed spots with cell_id > 0 (no valid_spot filter).
    unmixed_spots_valid : DataFrame
        Unmixed spots filtered to valid_spot == True and cell_id > 0.
    """
    mixed_spots   = pw_ds.load_all_rounds_spots_mp(table_type="mixed_spots",   remove_fg_bg_cols=True)
    unmixed_spots = pw_ds.load_all_rounds_spots_mp(table_type="unmixed_spots",  remove_fg_bg_cols=False)

    # Drop unassigned spots (cell_id == 0); copy() materialises filtered results
    # so we can immediately free the full raw tables.
    mixed_spots_filt    = mixed_spots[mixed_spots["cell_id"] > 0].copy()
    unmixed_spots_all   = unmixed_spots[unmixed_spots["cell_id"] > 0].copy()
    unmixed_spots_valid = unmixed_spots[(unmixed_spots["cell_id"] > 0) & unmixed_spots["valid_spot"]].copy()

    # Free the full raw tables — they are no longer needed.
    del mixed_spots, unmixed_spots
    gc.collect()

    # Drop valid_spot column from both unmixed tables — it has been used for
    # splitting and is not needed downstream.
    for _df in (unmixed_spots_all, unmixed_spots_valid):
        if "valid_spot" in _df.columns:
            _df.drop(columns="valid_spot", inplace=True)

    print(
        f"Spots loaded — mixed: {len(mixed_spots_filt):,}  "
        f"unmixed (all): {len(unmixed_spots_all):,}  "
        f"unmixed (valid only): {len(unmixed_spots_valid):,}"
    )

    return mixed_spots_filt, unmixed_spots_all, unmixed_spots_valid


def _load_cluster_meta(pw_ds) -> pd.DataFrame:
    """Build a cluster metadata DataFrame indexed by cell_id."""
    cluster_labels = pw_ds.load_cluster_labels().squeeze()
    cluster_cids   = pw_ds.load_sorted_cell_ids().squeeze()
    cluster_meta   = pd.DataFrame({"cluster_label": cluster_labels, "cell_id": cluster_cids})
    return cluster_meta.set_index("cell_id")


def _load_cluster_meta_legacy(mouse_id: str) -> pd.DataFrame:
    """Load cluster metadata for a legacy (non-pairwise) analysis run.

    Two strategies are available:

    **Strategy A — Borrow labels from the pairwise dataset (current implementation)**
        The pairwise unmixing dataset for this mouse is instantiated solely to
        read its pre-computed cluster labels and matching sorted cell IDs.  No
        spot data is loaded from the pairwise asset.  Because ``_load_data``
        already called ``attach_mouse_record_to_workstation``, the pairwise
        asset path is already mounted and no second attachment step is needed.
        The returned cell IDs serve as an implicit filter: only cells present in
        the pairwise clustering will appear in ``cluster_meta``, and downstream
        spot tables are filtered to those cell IDs naturally via ``cell_id > 0``.

    **Strategy B — Re-cluster from legacy spots (NOT YET IMPLEMENTED)**
        Cluster labels could be derived de-novo from the legacy unmixed spot
        count matrix using k-means (or a graph-based method such as Leiden).
        This would make the analysis fully independent of the pairwise pipeline
        and is the only option for mice that have no ``pairwise_unmixing``
        derived asset.

        TODO: implement Strategy B.  Key decisions needed:
          - choice of *k* (or resolution for graph methods)
          - spot-count normalisation strategy (CPM, scran, etc.)
          - reproducible random seed handling
          - whether to re-use the ABC-Atlas label vocabulary or assign numeric
            cluster IDs

    Returns
    -------
    pd.DataFrame
        Indexed by ``cell_id`` with a ``cluster_label`` column.
    """
    # ── Strategy A: instantiate pairwise dataset just for its cluster labels ──
    catalog_path = CATALOG_BASE / f"{mouse_id}.json"
    record = MouseRecord.from_json_file(catalog_path)

    pairwise_asset_name = record.derived_assets.get("pairwise_unmixing")
    if pairwise_asset_name is None:
        raise ValueError(
            f"Mouse {mouse_id} has no 'pairwise_unmixing' derived asset — "
            "cannot borrow cluster labels for legacy analysis.  "
            "Implement Strategy B (k-means re-clustering) to support this mouse."
        )

    pairwise_asset_path = DATA_DIR / pairwise_asset_name
    if (pairwise_asset_path / "pairwise_unmixing").exists():
        pairwise_asset_path = pairwise_asset_path / "pairwise_unmixing"

    base_ds = create_hcr_dataset_from_schema(catalog_path, DATA_DIR)
    pw_ds   = create_pairwise_unmixing_dataset(
        mouse_id            = mouse_id,
        pairwise_asset_path = pairwise_asset_path,
        source_dataset      = base_ds,
        min_dist            = 1,
    )
    return _load_cluster_meta(pw_ds)

    # ── Strategy B (stub) ────────────────────────────────────────────────────
    # raise NotImplementedError("Strategy B k-means re-clustering not yet implemented.")


def _build_gene_labels(mixed_spots: pd.DataFrame) -> dict:
    """Build Round-Chan-Gene display labels from the spot table."""
    return (
        mixed_spots[["mixed_gene", "round", "chan"]]
        .drop_duplicates("mixed_gene")
        .set_index("mixed_gene")
        .apply(lambda r: f"{r['round']}-{r['chan']}-{r.name}", axis=1)
        .to_dict()
    )


def _load_v1_merfish_cells() -> pd.DataFrame:
    """Load the V1 MERFISH cell index, dropping irrelevant deep layers."""
    if not V1_CELLS_CSV.exists():
        raise FileNotFoundError(f"V1 MERFISH cells CSV not found: {V1_CELLS_CSV}")
    v1_cells = pd.read_csv(V1_CELLS_CSV, index_col=0)
    v1_cells = v1_cells[~v1_cells["parcellation_substructure"].isin(DROP_LAYERS)]
    print(f"V1 cells after dropping {DROP_LAYERS}: {len(v1_cells):,}")
    print(v1_cells["parcellation_substructure"].value_counts().to_string())
    return v1_cells


def _ref_cache_subdir(label_level: str, hcr_genes) -> Path:
    """Return the cache subdirectory for a given set of reference parameters.

    The subdirectory name encodes label_level, ref_classes, min_label_cells,
    and the sorted HCR gene list so that different gene sets never collide.
    """
    import hashlib, json
    key = json.dumps(
        {"label_level": label_level,
         "ref_classes": sorted(REF_CLASSES),
         "min_label_cells": MIN_LABEL_CELLS,
         "genes": sorted(str(g) for g in hcr_genes)},
        sort_keys=True,
    )
    short_hash = hashlib.md5(key.encode()).hexdigest()[:8]
    return REF_CACHE_DIR / f"{label_level}_{short_hash}"


def _load_abc_reference(hcr_genes, v1_merfish_cells, label_level: str):
    """Load the MERFISH reference, using a per-gene-set on-disk cache.

    Cache is keyed by label_level + REF_CLASSES + MIN_LABEL_CELLS + sorted
    gene list, so every unique mouse gene set gets its own cache dir.
    """
    import json

    cache_dir   = _ref_cache_subdir(label_level, hcr_genes)
    counts_path = cache_dir / f"ref_counts_{label_level}.csv"
    labels_path = cache_dir / f"ref_labels_{label_level}.csv"
    meta_path   = cache_dir / "params.json"

    if counts_path.exists() and labels_path.exists():
        print(f"[ref cache] Loading cached reference from {cache_dir}")
        ref_counts_filt = pd.read_csv(counts_path, index_col=0)
        ref_labels_filt = pd.read_csv(labels_path, index_col=0).squeeze()
        return ref_counts_filt, ref_labels_filt

    print(f"[ref cache] Cache not found — loading from ABC Atlas …")
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        abc_cache = AbcProjectCache.from_cache_dir(ABC_ATLAS_DIR)

    ref_cell_meta = abc_cache.get_metadata_dataframe(
        directory="MERFISH-C57BL6J-638850",
        file_name="cell_metadata_with_cluster_annotation",
        dtype={"cell_label": str, "neurotransmitter": str},
    )
    ref_cell_meta.set_index("cell_label", inplace=True)
    print(f"ref_cell_meta: {ref_cell_meta.shape}")

    ref_counts_filt, ref_labels_filt = atlas_compare.load_abc_merfish_reference(
        abc_cache_dir   = ABC_ATLAS_DIR,
        genes           = hcr_genes,
        cell_index      = v1_merfish_cells.index,
        ref_classes     = REF_CLASSES,
        label_level     = label_level,
        min_label_cells = MIN_LABEL_CELLS,
        save_dir        = cache_dir,   # save_dir writes the CSVs for us
        abc_cache       = abc_cache,
        ref_cell_meta   = ref_cell_meta,
    )

    # Write a sidecar so the cache is self-documenting.
    meta_path.write_text(json.dumps(
        {"label_level": label_level,
         "ref_classes": REF_CLASSES,
         "min_label_cells": MIN_LABEL_CELLS,
         "genes": sorted(str(g) for g in hcr_genes)},
        indent=2,
    ))
    print(f"[ref cache] Saved to {cache_dir}")
    return ref_counts_filt, ref_labels_filt


# ---------------------------------------------------------------------------
# Spot pre-aggregation helper
# ---------------------------------------------------------------------------

def _preaggregate_spots(
    spots: pd.DataFrame,
    gene_col: str,
    cell_ids,
    genes: list,
) -> pd.DataFrame:
    """Aggregate a spot table to a cell × gene count matrix.

    Uses ``groupby`` rather than ``pd.crosstab`` so that Categorical gene
    columns are never expanded to Python object strings.  Spots are
    pre-filtered to *cell_ids* and *genes* before aggregation, further
    reducing working memory.

    Returns
    -------
    pd.DataFrame
        Shape ``(len(cell_ids), len(genes))``, integer counts, zero-filled.
    """
    cell_id_set = set(cell_ids)
    gene_set    = set(genes)
    mask        = spots["cell_id"].isin(cell_id_set) & spots[gene_col].isin(gene_set)
    s           = spots.loc[mask, ["cell_id", gene_col]]
    counts      = s.groupby(["cell_id", gene_col]).size().unstack(fill_value=0)
    return counts.reindex(index=cell_ids, columns=genes, fill_value=0)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _run_one(
    label: str,
    spot_filter: str,
    mouse_id: str,
    raw_counts: pd.DataFrame,
    unmixed_spots: pd.DataFrame,
    unmixed_counts: pd.DataFrame,
    cluster_meta: pd.DataFrame,
    gene_labels: dict,
    ref_counts_filt: pd.DataFrame,
    ref_labels_filt: pd.Series,
    output_dir: Path,
    dpi: int,
    unmixed_spots_valid: pd.DataFrame = None,
) -> None:
    """Run the comparison + save figures and CSVs for one set of unmixed spots.

    Parameters
    ----------
    spot_filter:
        Short label written into the saved CSVs, e.g. ``"all"`` or ``"valid"``.
        Used as a grouping key when CSVs from multiple mice are concatenated.
    mouse_id:
        Written into saved CSVs so rows are identifiable after concatenation.
    raw_counts:
        Pre-aggregated raw (mixed) cell × gene count matrix.
    unmixed_counts:
        Pre-aggregated unmixed cell × gene count matrix for this run.
    unmixed_spots_valid:
        Optional valid-only unmixed spots DataFrame.  Still required for the
        cell × gene heatmap (``12_cxg_clusters_valid.png``).
    """
    print(f"\n{'─'*60}")
    print(f"  Run: {label}")
    print(f"  Unmixed spots: {len(unmixed_spots):,}")
    print(f"  Output: {output_dir}")
    print(f"{'─'*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    comparison, cluster_matches = atlas_compare.raw_unmixed_reference_comparison(
        raw_spots                  = None,
        unmixed_spots              = None,
        cell_meta                  = cluster_meta,
        group_col                  = "cluster_label",
        ref_counts                 = ref_counts_filt,
        ref_labels                 = ref_labels_filt,
        raw_chan_col               = "mixed_gene",
        unmixed_chan_col           = "unmixed_gene",
        matching_source            = "unmixed",
        precomputed_raw_counts     = raw_counts,
        precomputed_unmixed_counts = unmixed_counts,
    )

    print("Cluster → reference matches:")
    print(cluster_matches.to_string())

    # ── Save CSVs with provenance columns ─────────────────────────────────────
    comp_out = comparison.copy()
    comp_out.insert(0, "mouse_id",    mouse_id)
    comp_out.insert(1, "spot_filter", spot_filter)
    comp_out.to_csv(output_dir / "comparison.csv", index=False)

    cm_out = cluster_matches.reset_index().copy()
    cm_out.insert(0, "mouse_id",    mouse_id)
    cm_out.insert(1, "spot_filter", spot_filter)
    cm_out.to_csv(output_dir / "cluster_matches.csv", index=False)

    print(f"CSVs written to: {output_dir}")

    # ── Save figures ──────────────────────────────────────────────────────────
    atlas_compare.save_all_figures(
        comparison      = comparison,
        cluster_matches = cluster_matches,
        gene_labels     = gene_labels,
        output_dir      = output_dir,
        dpi             = dpi,
    )
    print(f"Figures written to: {output_dir}")

    # ── Cell × gene heatmap (clusters pre-computed, no re-clustering) ─────────
    atlas_compare.plot_cell_x_gene_with_clusters(
        spots        = unmixed_spots,
        cluster_meta = cluster_meta,
        gene_labels  = gene_labels,
        chan_col     = "unmixed_gene",
        title        = f"{mouse_id} cell × gene ({spot_filter} unmixed spots)",
        save         = True,
        output_dir   = output_dir,
        filename     = "11_cxg_clusters",
        dpi          = dpi,
        show         = False,
        formats      = ("png",),
    )
    print(f"Cell × gene figure written to: {output_dir}")

    if unmixed_spots_valid is not None:
        atlas_compare.plot_cell_x_gene_with_clusters(
            spots        = unmixed_spots_valid,
            cluster_meta = cluster_meta,
            gene_labels  = gene_labels,
            chan_col     = "unmixed_gene",
            title        = f"{mouse_id} cell × gene (valid unmixed spots)",
            save         = True,
            output_dir   = output_dir,
            filename     = "12_cxg_clusters_valid",
            dpi          = dpi,
            show         = False,
            formats      = ("png",),
        )
        print(f"Cell × gene (valid) figure written to: {output_dir}")


def run(mouse_id: str, output_dir: Path, label_level: str, dpi: int, legacy: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # When legacy mode is active, tag CSV rows with "<mouse_id>-legacy" so that
    # outputs from pairwise and legacy runs can be concatenated without ambiguity.
    csv_mouse_id = f"{mouse_id}-legacy" if legacy else mouse_id

    print(f"\n{'='*60}")
    print(f"Mouse: {mouse_id}{' (legacy mode)' if legacy else ''}")
    print(f"CSV mouse_id tag: {csv_mouse_id}")
    print(f"Output root: {output_dir}")
    print(f"Reference label level: {label_level}")
    print(f"{'='*60}\n")

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print("[1/5] Loading dataset …")
    pw_ds = _load_data(mouse_id, legacy=legacy)

    # ── 2. Load and annotate spots ────────────────────────────────────────────
    print("[2/5] Loading spots …")
    mixed_spots, unmixed_spots_all, unmixed_spots_valid = _load_spots(pw_ds)

    # ── 3. Cluster metadata + gene labels ─────────────────────────────────────
    print("[3/5] Loading cluster labels …")
    if legacy:
        cluster_meta = _load_cluster_meta_legacy(mouse_id)
    else:
        cluster_meta = _load_cluster_meta(pw_ds)
    gene_labels  = _build_gene_labels(mixed_spots)

    # Slim spot tables to only the columns needed downstream:
    #   mixed_spots    → cell_id + mixed_gene  (raw_chan_col)
    #   unmixed tables → cell_id + unmixed_gene (unmixed_chan_col)
    # This must happen AFTER gene_labels is built (needs round/chan columns).
    mixed_spots         = mixed_spots[["cell_id", "mixed_gene"]].copy()
    unmixed_spots_all   = unmixed_spots_all[["cell_id", "unmixed_gene"]].copy()
    unmixed_spots_valid = unmixed_spots_valid[["cell_id", "unmixed_gene"]].copy()

    # Cast to memory-efficient dtypes.  Categorical stores one int8 code per
    # row instead of a Python string object (~68 bytes/row), reducing each
    # gene column from several GB to tens of MB.
    for _df in (mixed_spots, unmixed_spots_all, unmixed_spots_valid):
        _df["cell_id"] = _df["cell_id"].astype("int32")
        _gene_col = "mixed_gene" if "mixed_gene" in _df.columns else "unmixed_gene"
        _df[_gene_col] = _df[_gene_col].astype("category")
    gc.collect()

    # ── 4. ABC Atlas reference (loaded once, shared by both runs) ─────────────
    print("[4/5] Loading V1 MERFISH reference …")
    v1_merfish_cells = _load_v1_merfish_cells()
    hcr_genes = unmixed_spots_all["unmixed_gene"].unique()

    # Fix known typo in gene name across all data structures
    _GENE_RENAME = {"Slac17a7": "Slc17a7"}
    hcr_genes = [_GENE_RENAME.get(g, g) for g in hcr_genes]
    gene_labels = {_GENE_RENAME.get(k, k): v.replace(k, _GENE_RENAME[k]) if k in _GENE_RENAME else v
                   for k, v in gene_labels.items()}
    for spot_df in (mixed_spots, unmixed_spots_all, unmixed_spots_valid):
        for col in ("mixed_gene", "unmixed_gene"):
            if col in spot_df.columns:
                spot_df[col] = spot_df[col].replace(_GENE_RENAME)
    print(f"HCR genes: {hcr_genes}")

    ref_counts_filt, ref_labels_filt = _load_abc_reference(
        hcr_genes        = hcr_genes,
        v1_merfish_cells = v1_merfish_cells,
        label_level      = label_level,
    )

    # ── 5. Pre-aggregate spot tables → cell × gene count matrices ─────────────
    # _preaggregate_spots uses groupby (not pd.crosstab / add_gene_column) so
    # Categorical columns are never expanded to Python object strings, avoiding
    # the ~70 byte/row spike that caused OOM in the previous approach.
    print("[5/6] Pre-aggregating spot tables …")
    _shared_genes        = sorted(set(hcr_genes) & set(ref_counts_filt.columns))
    raw_counts           = _preaggregate_spots(mixed_spots,        "mixed_gene",   cluster_meta.index, _shared_genes)
    unmixed_counts_all   = _preaggregate_spots(unmixed_spots_all,  "unmixed_gene", cluster_meta.index, _shared_genes)
    unmixed_counts_valid = _preaggregate_spots(unmixed_spots_valid, "unmixed_gene", cluster_meta.index, _shared_genes)
    print(f"  Count matrix shape: {raw_counts.shape}  (cells × shared genes)")
    del mixed_spots          # replaced by raw_counts; spot table no longer needed
    gc.collect()

    # ── 6. Run comparison twice ──────────────────────────────────────────────
    print("[6/6] Running comparisons …")

    atlas_compare_dir = output_dir / "atlas_compare"

    _shared = dict(
        mouse_id        = csv_mouse_id,
        raw_counts      = raw_counts,
        cluster_meta    = cluster_meta,
        gene_labels     = gene_labels,
        ref_counts_filt = ref_counts_filt,
        ref_labels_filt = ref_labels_filt,
        dpi             = dpi,
    )

    _run_one(
        label                = "all unmixed spots",
        spot_filter          = "all",
        unmixed_spots        = unmixed_spots_all,
        unmixed_counts       = unmixed_counts_all,
        unmixed_spots_valid  = unmixed_spots_valid,
        output_dir           = atlas_compare_dir / "all_unmixed",
        **_shared,
    )

    # Free all-unmixed structures before the second run — not used again.
    del unmixed_spots_all, unmixed_counts_all
    gc.collect()

    _run_one(
        label          = "valid unmixed spots only",
        spot_filter    = "valid",
        unmixed_spots  = unmixed_spots_valid,
        unmixed_counts = unmixed_counts_valid,
        output_dir     = atlas_compare_dir / "valid_unmixed",
        **_shared,
    )

    print(f"\n{'='*60}")
    print(f"All done for mouse {mouse_id}{' (legacy)' if legacy else ''}.")
    print(f"  all_unmixed/   → {atlas_compare_dir / 'all_unmixed'}")
    print(f"  valid_unmixed/ → {atlas_compare_dir / 'valid_unmixed'}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run atlas comparison pipeline for a single mouse and save all figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mouse-id", required=True,
        help="Mouse ID, e.g. 782149.  Must have a matching catalog JSON and "
             "a pairwise_unmixing derived asset.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write outputs to.  Defaults to "
             "scratch/ref_atlas_validation/<mouse-id>.",
    )
    parser.add_argument(
        "--label-level", default="supertype",
        choices=["class", "subclass", "supertype", "cluster"],
        help="Reference label granularity used for cluster matching.",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Figure resolution in dots per inch.",
    )
    parser.add_argument(
        "--legacy", action="store_true", default=False,
        help="Use the original HCRDataset instead of the pairwise unmixing dataset. "
             "Mouse IDs in output CSVs will be suffixed with '-legacy'. "
             "Output defaults to scratch/ref_atlas_validation/<mouse-id>-legacy.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    default_subdir = f"{args.mouse_id}-legacy" if args.legacy else args.mouse_id
    out = (
        Path(args.output_dir)
        if args.output_dir is not None
        else SCRATCH_BASE / default_subdir
    )

    run(
        mouse_id    = args.mouse_id,
        output_dir  = out,
        label_level = args.label_level,
        dpi         = args.dpi,
        legacy      = args.legacy,
    )
