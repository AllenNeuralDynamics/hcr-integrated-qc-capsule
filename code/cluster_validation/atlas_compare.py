"""
spatial_transcriptomics_validation.py

Utilities for validating a cell-by-gene table produced from spatial transcriptomics
spot tables before and after unmixing.

Expected spot table columns
---------------------------
raw_spots:
    <chan_col>, cell_id, x, y, z

unmixed_spots:
    <chan_col>, cell_id, x, y, z

The channel column can differ between raw and unmixed tables (e.g.
``mixed_gene`` vs ``unmixed_gene``).  Compound functions accept separate
``raw_chan_col`` / ``unmixed_chan_col`` parameters for this purpose.

Each channel column can either contain the gene name directly or a channel
identifier that can be mapped to a gene with ``chan_to_gene``.

Typical workflow
----------------
1. Convert raw and unmixed spot tables into cell x gene count matrices.
2. Measure raw-to-unmixed gene retention by cluster/cell type.
3. Compare raw and unmixed cluster expression to a reference atlas.
4. Plot whether unmixing moved expression closer to reference.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from aind_hcr_qc.utils.utils import saveable_plot as _saveable_plot
except ImportError:  # pragma: no cover
    # Fallback no-op decorator if aind_hcr_qc is not installed
    def _saveable_plot(defaults=None):  # type: ignore
        def _decorator(fn):
            return fn
        return _decorator

# ABC Atlas loading is optional — only imported when load_abc_merfish_reference is called.
try:
    import anndata as _anndata
except ImportError:  # pragma: no cover
    _anndata = None  # type: ignore

try:
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache as _AbcProjectCache
except ImportError:  # pragma: no cover
    _AbcProjectCache = None  # type: ignore


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def prepare_cell_meta(
    cell_meta: pd.DataFrame,
    cell_id_col: str = "cell_id",
) -> pd.DataFrame:
    """
    Ensure cell_meta is indexed by cell_id.

    Parameters
    ----------
    cell_meta:
        DataFrame containing metadata per cell. Either already indexed by cell_id
        or containing a column called `cell_id_col`.
    cell_id_col:
        Name of the column containing cell ids, if not already in the index.

    Returns
    -------
    DataFrame indexed by cell_id.
    """
    cell_meta = cell_meta.copy()

    if cell_meta.index.name == cell_id_col:
        return cell_meta

    if cell_id_col in cell_meta.columns:
        return cell_meta.set_index(cell_id_col, drop=False)

    # If no explicit column exists, assume index already contains cell ids.
    return cell_meta


def add_gene_column(
    spots: pd.DataFrame,
    chan_to_gene: Optional[Dict] = None,
    chan_col: str = "chan",
) -> pd.DataFrame:
    """
    Add a `gene` column to a spot table.

    Parameters
    ----------
    spots:
        Spot table with columns including `chan` and `cell_id`.
    chan_to_gene:
        Optional mapping from channel identifier to gene name. If None,
        `chan` is assumed to already contain gene names.
    chan_col:
        Name of the channel column.

    Returns
    -------
    Copy of spot table with an added `gene` column.
    """
    required = {chan_col, "cell_id"}
    missing = required - set(spots.columns)
    if missing:
        raise ValueError(f"spots is missing required columns: {sorted(missing)}")

    spots = spots.copy()

    if chan_to_gene is None:
        spots["gene"] = spots[chan_col].astype(str)
    else:
        spots["gene"] = spots[chan_col].map(chan_to_gene)

    spots = spots.dropna(subset=["gene", "cell_id"])
    return spots


def spots_to_cell_gene_counts(
    spots: pd.DataFrame,
    cell_ids: Optional[Sequence] = None,
    genes: Optional[Sequence[str]] = None,
    chan_to_gene: Optional[Dict] = None,
    chan_col: str = "chan",
) -> pd.DataFrame:
    """
    Convert a spot table to a cells x genes count matrix.

    Parameters
    ----------
    spots:
        DataFrame with columns `chan`, `cell_id`, `x`, `y`, `z`.
    cell_ids:
        Optional ordered list/index of cells to include.
    genes:
        Optional ordered list of genes to include.
    chan_to_gene:
        Optional mapping from channel to gene.
    chan_col:
        Name of the channel column.

    Returns
    -------
    DataFrame: cells x genes counts.
    """
    spots = add_gene_column(spots, chan_to_gene=chan_to_gene, chan_col=chan_col)
    counts = pd.crosstab(spots["cell_id"], spots["gene"])

    if cell_ids is not None:
        counts = counts.reindex(cell_ids, fill_value=0)

    if genes is not None:
        counts = counts.reindex(columns=list(genes), fill_value=0)

    return counts


def normalize_cp10k(counts: pd.DataFrame, scale: float = 10_000) -> pd.DataFrame:
    """
    Library-size normalize a cells x genes count matrix to counts per 10k.

    Parameters
    ----------
    counts:
        cells x genes count matrix.
    scale:
        Scaling factor, default 10,000.

    Returns
    -------
    DataFrame of normalized expression.
    """
    lib = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(lib, axis=0).mul(scale).fillna(0)


def group_mean_expression(
    counts: pd.DataFrame,
    cell_meta: pd.DataFrame,
    group_col: str,
    scale: float = 10_000,
) -> pd.DataFrame:
    """
    Compute mean normalized expression per group.

    Parameters
    ----------
    counts:
        cells x genes count matrix.
    cell_meta:
        DataFrame indexed by cell_id, containing `group_col`.
    group_col:
        Column in cell_meta defining groups, e.g. cluster or cell_type.
    scale:
        Normalization scale.

    Returns
    -------
    groups x genes mean normalized expression.
    """
    cell_meta = prepare_cell_meta(cell_meta)
    missing_cells = counts.index.difference(cell_meta.index)
    if len(missing_cells) > 0:
        raise ValueError(
            f"{len(missing_cells)} cells in counts are missing from cell_meta."
        )

    if group_col not in cell_meta.columns:
        raise ValueError(f"`{group_col}` is not a column in cell_meta.")

    meta = cell_meta.loc[counts.index]
    norm = normalize_cp10k(counts, scale=scale)
    return norm.groupby(meta[group_col]).mean()


def group_sum_counts(
    counts: pd.DataFrame,
    cell_meta: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    Compute summed raw counts per group.
    """
    cell_meta = prepare_cell_meta(cell_meta)
    meta = cell_meta.loc[counts.index]

    if group_col not in meta.columns:
        raise ValueError(f"`{group_col}` is not a column in cell_meta.")

    return counts.groupby(meta[group_col]).sum()


def cosine_scores(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between vector `a` and each row of matrix `B`.
    """
    a = np.asarray(a, dtype=float)
    B = np.asarray(B, dtype=float)

    a_norm = np.linalg.norm(a)
    B_norm = np.linalg.norm(B, axis=1)
    denom = np.maximum(a_norm * B_norm, 1e-12)

    return (B @ a) / denom


def softmax(x: np.ndarray, temperature: float = 0.15) -> np.ndarray:
    """
    Convert scores to weights.

    Lower temperature gives sharper weights. Higher temperature gives more diffuse
    weighting across reference subtypes.
    """
    x = np.asarray(x, dtype=float) / temperature
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


# -----------------------------------------------------------------------------
# Raw-to-unmixed gene retention
# -----------------------------------------------------------------------------


def gene_retention_by_group(
    raw_spots: pd.DataFrame,
    unmixed_spots: pd.DataFrame,
    cell_meta: pd.DataFrame,
    group_col: str,
    genes: Optional[Sequence[str]] = None,
    chan_to_gene: Optional[Dict] = None,
    raw_chan_col: str = "chan",
    unmixed_chan_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute aggregate raw-to-unmixed retention per group x gene.

    Retention is:

        unmixed spots assigned to cells in group / raw spots assigned to cells in group

    This is aggregate retention, not single-spot tracking. To track individual
    spots moving between cells, your tables would need a stable spot_id column.

    Parameters
    ----------
    raw_spots, unmixed_spots:
        Spot tables with columns `cell_id`, `x`, `y`, `z` and the
        respective channel column.
    cell_meta:
        Cell metadata indexed by cell_id or containing a `cell_id` column.
    group_col:
        Metadata column defining groups, e.g. "cluster".
    genes:
        Optional list of genes to include.
    chan_to_gene:
        Optional channel-to-gene mapping.
    raw_chan_col:
        Channel / gene column in raw_spots.
    unmixed_chan_col:
        Channel / gene column in unmixed_spots. Defaults to raw_chan_col.

    Returns
    -------
    Long DataFrame with group, gene, raw_spots, unmixed_spots, retention.
    """
    if unmixed_chan_col is None:
        unmixed_chan_col = raw_chan_col

    cell_meta = prepare_cell_meta(cell_meta)
    cell_ids = cell_meta.index

    raw_counts = spots_to_cell_gene_counts(
        raw_spots,
        cell_ids=cell_ids,
        genes=genes,
        chan_to_gene=chan_to_gene,
        chan_col=raw_chan_col,
    )

    unmixed_counts = spots_to_cell_gene_counts(
        unmixed_spots,
        cell_ids=cell_ids,
        genes=raw_counts.columns,
        chan_to_gene=chan_to_gene,
        chan_col=unmixed_chan_col,
    )

    raw_group_counts = group_sum_counts(raw_counts, cell_meta, group_col)
    unmixed_group_counts = group_sum_counts(unmixed_counts, cell_meta, group_col)

    records = []
    for group in raw_group_counts.index:
        for gene in raw_group_counts.columns:
            raw_n = float(raw_group_counts.loc[group, gene])
            unmixed_n = float(unmixed_group_counts.loc[group, gene])
            retention = unmixed_n / raw_n if raw_n > 0 else np.nan

            records.append(
                {
                    "group": group,
                    "gene": gene,
                    "raw_spots": raw_n,
                    "unmixed_spots": unmixed_n,
                    "retention": retention,
                }
            )

    return pd.DataFrame(records)


def gene_retention_for_candidate_cells(
    raw_spots: pd.DataFrame,
    unmixed_spots: pd.DataFrame,
    cell_meta: pd.DataFrame,
    gene: str,
    candidate_col: str,
    chan_to_gene: Optional[Dict] = None,
    raw_chan_col: str = "chan",
    unmixed_chan_col: Optional[str] = None,
) -> dict:
    """
    Compute retention for one gene in candidate cells.

    Example use:
        candidate_col="is_vip_like"

    This asks:
        Of raw Cck spots initially assigned to VIP-like cells, how many Cck
        spots remain assigned to VIP-like cells after unmixing?

    Parameters
    ----------
    raw_spots, unmixed_spots:
        Spot tables with columns `cell_id`, `x`, `y`, `z` and the
        respective channel column.
    cell_meta:
        Cell metadata indexed by cell_id or containing a `cell_id` column.
    gene:
        Gene to evaluate, e.g. "Cck".
    candidate_col:
        Boolean metadata column marking candidate cells.
    chan_to_gene:
        Optional channel-to-gene mapping.
    raw_chan_col:
        Channel / gene column in raw_spots.
    unmixed_chan_col:
        Channel / gene column in unmixed_spots. Defaults to raw_chan_col.

    Returns
    -------
    Dictionary with raw spots, unmixed spots, and retention.
    """
    if unmixed_chan_col is None:
        unmixed_chan_col = raw_chan_col

    cell_meta = prepare_cell_meta(cell_meta)

    if candidate_col not in cell_meta.columns:
        raise ValueError(f"`{candidate_col}` is not a column in cell_meta.")

    candidate_cells = set(cell_meta.index[cell_meta[candidate_col].astype(bool)])

    raw = add_gene_column(raw_spots, chan_to_gene=chan_to_gene, chan_col=raw_chan_col)
    unmixed = add_gene_column(unmixed_spots, chan_to_gene=chan_to_gene, chan_col=unmixed_chan_col)

    raw_gene_to_candidates = raw[
        (raw["gene"] == gene) & (raw["cell_id"].isin(candidate_cells))
    ]

    unmixed_gene_to_candidates = unmixed[
        (unmixed["gene"] == gene) & (unmixed["cell_id"].isin(candidate_cells))
    ]

    n_raw = len(raw_gene_to_candidates)
    n_unmixed = len(unmixed_gene_to_candidates)

    return {
        "gene": gene,
        "candidate_group": candidate_col,
        "raw_spots_in_candidate_cells": n_raw,
        "unmixed_spots_in_candidate_cells": n_unmixed,
        "retention": n_unmixed / n_raw if n_raw > 0 else np.nan,
    }


# -----------------------------------------------------------------------------
# Reference matching and raw/unmixed-to-reference comparison
# -----------------------------------------------------------------------------


def match_groups_to_reference(
    observed_group_expr: pd.DataFrame,
    reference_group_expr: pd.DataFrame,
    genes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Match each observed group to its closest reference subtype using cosine similarity.

    Matching is performed on log1p-normalized group-level expression.

    Parameters
    ----------
    observed_group_expr:
        observed groups x genes expression, e.g. cluster means.
    reference_group_expr:
        reference subtypes x genes expression.
    genes:
        Optional gene subset for matching.

    Returns
    -------
    DataFrame indexed by observed group, with reference_label and match_score.
    """
    if genes is None:
        genes = sorted(set(observed_group_expr.columns) & set(reference_group_expr.columns))
    else:
        genes = [g for g in genes if g in observed_group_expr.columns and g in reference_group_expr.columns]

    if len(genes) == 0:
        raise ValueError("No shared genes available for reference matching.")

    obs = np.log1p(observed_group_expr[genes])
    ref = np.log1p(reference_group_expr[genes])

    ref_labels = list(ref.index)
    records = []

    for group in obs.index:
        sims = cosine_scores(obs.loc[group].to_numpy(), ref.to_numpy())
        best_idx = int(np.argmax(sims))

        records.append(
            {
                "group": group,
                "reference_label": ref_labels[best_idx],
                "match_score": float(sims[best_idx]),
            }
        )

    return pd.DataFrame(records).set_index("group")


def reference_weighted_expected_expression(
    observed_group_expr: pd.DataFrame,
    reference_group_expr: pd.DataFrame,
    test_gene: str,
    temperature: float = 0.15,
    matching_genes: Optional[Sequence[str]] = None,
    exclude_test_gene: bool = True,
) -> pd.DataFrame:
    """
    Compute reference-weighted expected expression for one gene.

    This is useful when not every cluster within a broad class, such as VIP,
    should express the same marker.

    For each observed group:
        1. Compare group to all reference subtypes using matching genes.
        2. Convert similarities to weights.
        3. Expected expression of `test_gene` is the weighted mean across
           reference subtype expression.

    Parameters
    ----------
    observed_group_expr:
        observed groups x genes expression.
    reference_group_expr:
        reference subtypes x genes expression.
    test_gene:
        Gene whose expected expression is computed.
    temperature:
        Softmax temperature for subtype weights.
    matching_genes:
        Optional genes to use for matching.
    exclude_test_gene:
        If True, do not use `test_gene` for reference weighting.

    Returns
    -------
    DataFrame indexed by observed group.
    """
    if test_gene not in reference_group_expr.columns:
        raise ValueError(f"{test_gene!r} not present in reference_group_expr.")

    if matching_genes is None:
        matching_genes = sorted(set(observed_group_expr.columns) & set(reference_group_expr.columns))
    else:
        matching_genes = [
            g for g in matching_genes
            if g in observed_group_expr.columns and g in reference_group_expr.columns
        ]

    if exclude_test_gene:
        matching_genes = [g for g in matching_genes if g != test_gene]

    if len(matching_genes) == 0:
        raise ValueError("No matching genes available after exclusions.")

    obs = np.log1p(observed_group_expr[matching_genes])
    ref = np.log1p(reference_group_expr[matching_genes])
    ref_labels = list(ref.index)

    records = []

    for group in obs.index:
        sims = cosine_scores(obs.loc[group].to_numpy(), ref.to_numpy())
        weights = softmax(sims, temperature=temperature)
        expected = float(np.sum(weights * reference_group_expr[test_gene].to_numpy()))
        top_idx = int(np.argmax(weights))

        records.append(
            {
                "group": group,
                "gene": test_gene,
                "expected_cp10k": expected,
                "top_reference_subtype": ref_labels[top_idx],
                "top_reference_weight": float(weights[top_idx]),
            }
        )

    return pd.DataFrame(records).set_index("group")


def raw_unmixed_reference_comparison(
    raw_spots: pd.DataFrame,
    unmixed_spots: pd.DataFrame,
    cell_meta: pd.DataFrame,
    group_col: str,
    ref_counts: pd.DataFrame,
    ref_labels: pd.Series,
    genes: Optional[Sequence[str]] = None,
    chan_to_gene: Optional[Dict] = None,
    raw_chan_col: str = "chan",
    unmixed_chan_col: Optional[str] = None,
    group_to_ref_label: Optional[Dict] = None,
    matching_source: str = "unmixed",
    pseudocount: float = 0.1,
    scale: float = 10_000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare raw and unmixed group expression to a matched reference subtype.

    The output table includes:
        raw_cp10k
        unmixed_cp10k
        reference_cp10k
        raw_log2_oe
        unmixed_log2_oe
        raw_abs_error
        unmixed_abs_error
        improvement

    Positive improvement means unmixing moved that group/gene closer to reference:
        improvement = abs(raw_log2_oe) - abs(unmixed_log2_oe)

    Parameters
    ----------
    raw_spots, unmixed_spots:
        Spot tables with columns `cell_id`, `x`, `y`, `z` and the
        respective channel column.
    cell_meta:
        Cell metadata indexed by cell_id or containing a `cell_id` column.
    group_col:
        Metadata column defining groups, e.g. "cluster".
    ref_counts:
        Reference cells x genes count matrix.
    ref_labels:
        Series indexed by reference cell id, giving reference subtype labels.
    genes:
        Optional gene list.
    chan_to_gene:
        Optional channel-to-gene mapping.
    raw_chan_col:
        Channel / gene column in raw_spots.
    unmixed_chan_col:
        Channel / gene column in unmixed_spots. Defaults to raw_chan_col.
    group_to_ref_label:
        Optional manual mapping from observed group to reference label.
    matching_source:
        "unmixed" or "raw"; which observed table to use when auto-matching
        observed groups to reference.
    pseudocount:
        Small value added before log2 O/E.
    scale:
        CP10K scale factor.

    Returns
    -------
    comparison:
        Long DataFrame, group x gene.
    match_df:
        Group-to-reference mapping and match scores.
    """
    if unmixed_chan_col is None:
        unmixed_chan_col = raw_chan_col

    cell_meta = prepare_cell_meta(cell_meta)
    cell_ids = cell_meta.index

    if genes is None:
        if chan_to_gene is None:
            raw_genes = set(raw_spots[raw_chan_col].astype(str))
            unmixed_genes = set(unmixed_spots[unmixed_chan_col].astype(str))
        else:
            raw_genes = set(raw_spots[raw_chan_col].map(chan_to_gene).dropna())
            unmixed_genes = set(unmixed_spots[unmixed_chan_col].map(chan_to_gene).dropna())

        genes = sorted(raw_genes & unmixed_genes & set(ref_counts.columns))
    else:
        genes = [g for g in genes if g in ref_counts.columns]

    if len(genes) == 0:
        raise ValueError("No shared genes between observed spots and reference.")

    raw_counts = spots_to_cell_gene_counts(
        raw_spots,
        cell_ids=cell_ids,
        genes=genes,
        chan_to_gene=chan_to_gene,
        chan_col=raw_chan_col,
    )

    unmixed_counts = spots_to_cell_gene_counts(
        unmixed_spots,
        cell_ids=cell_ids,
        genes=genes,
        chan_to_gene=chan_to_gene,
        chan_col=unmixed_chan_col,
    )

    raw_group_expr = group_mean_expression(raw_counts, cell_meta, group_col, scale=scale)
    unmixed_group_expr = group_mean_expression(unmixed_counts, cell_meta, group_col, scale=scale)

    ref_labels = ref_labels.loc[ref_counts.index]
    ref_norm = normalize_cp10k(ref_counts[genes], scale=scale)
    ref_group_expr = ref_norm.groupby(ref_labels).mean()

    if group_to_ref_label is None:
        if matching_source not in {"raw", "unmixed"}:
            raise ValueError("matching_source must be 'raw' or 'unmixed'.")

        source_expr = unmixed_group_expr if matching_source == "unmixed" else raw_group_expr
        match_df = match_groups_to_reference(source_expr, ref_group_expr, genes=genes)
        group_to_ref_label = match_df["reference_label"].to_dict()
    else:
        match_df = pd.DataFrame.from_dict(
            {
                group: {"reference_label": ref_label, "match_score": np.nan}
                for group, ref_label in group_to_ref_label.items()
            },
            orient="index",
        )
        match_df.index.name = "group"

    records = []

    for group in raw_group_expr.index:
        if group not in group_to_ref_label:
            continue

        ref_label = group_to_ref_label[group]
        if ref_label not in ref_group_expr.index:
            raise ValueError(f"Reference label {ref_label!r} not found in reference.")

        for gene in genes:
            raw_val = float(raw_group_expr.loc[group, gene])
            unmixed_val = float(unmixed_group_expr.loc[group, gene])
            ref_val = float(ref_group_expr.loc[ref_label, gene])

            raw_log2_oe = np.log2((raw_val + pseudocount) / (ref_val + pseudocount))
            unmixed_log2_oe = np.log2((unmixed_val + pseudocount) / (ref_val + pseudocount))

            records.append(
                {
                    "group": group,
                    "gene": gene,
                    "reference_label": ref_label,
                    "raw_cp10k": raw_val,
                    "unmixed_cp10k": unmixed_val,
                    "reference_cp10k": ref_val,
                    "raw_log2_oe": raw_log2_oe,
                    "unmixed_log2_oe": unmixed_log2_oe,
                    "raw_abs_error": abs(raw_log2_oe),
                    "unmixed_abs_error": abs(unmixed_log2_oe),
                    "improvement": abs(raw_log2_oe) - abs(unmixed_log2_oe),
                }
            )

    return pd.DataFrame(records), match_df


def reference_weighted_oe_score(
    cell_counts: pd.DataFrame,
    cell_clusters: pd.Series,
    ref_counts: pd.DataFrame,
    ref_labels: pd.Series,
    genes: Optional[Sequence[str]] = None,
    scale: float = 10_000,
    pseudocount: float = 0.1,
    temperature: float = 0.15,
    min_expected: float = 0.25,
) -> pd.DataFrame:
    """
    Cluster x gene observed/expected score using soft reference subtype weighting.

    This version works from an existing cell x gene table.

    For each cluster and gene:
        1. Exclude the gene being tested.
        2. Compare the cluster to all reference subtype centroids.
        3. Convert similarities to weights.
        4. Compute expected expression as weighted reference expression.
        5. Compute log2 observed/expected.

    Useful when some VIP clusters should have Cck and some should not.
    """
    cell_clusters = cell_clusters.loc[cell_counts.index]
    ref_labels = ref_labels.loc[ref_counts.index]

    shared_genes = sorted(set(cell_counts.columns) & set(ref_counts.columns))
    if genes is None:
        genes = shared_genes
    else:
        genes = [g for g in genes if g in shared_genes]

    if len(genes) < 3:
        raise ValueError("Need at least 3 shared genes for leave-one-gene-out scoring.")

    X = normalize_cp10k(cell_counts[genes].astype(float), scale=scale)
    R = normalize_cp10k(ref_counts[genes].astype(float), scale=scale)

    cluster_mean = X.groupby(cell_clusters).mean()
    ref_mean = R.groupby(ref_labels).mean()

    log_cluster = np.log1p(cluster_mean)
    log_ref = np.log1p(ref_mean)

    ref_subtypes = list(ref_mean.index)
    records = []

    for cluster in cluster_mean.index:
        for gene in genes:
            other_genes = [g for g in genes if g != gene]
            cluster_vec = log_cluster.loc[cluster, other_genes].to_numpy()
            ref_mat = log_ref.loc[:, other_genes].to_numpy()

            sims = cosine_scores(cluster_vec, ref_mat)
            weights = softmax(sims, temperature=temperature)

            observed = float(cluster_mean.loc[cluster, gene])
            expected = float(np.sum(weights * ref_mean.loc[:, gene].to_numpy()))
            log2_oe = np.log2((observed + pseudocount) / (expected + pseudocount))

            top_idx = int(np.argmax(weights))

            records.append(
                {
                    "cluster": cluster,
                    "gene": gene,
                    "top_reference_subtype": ref_subtypes[top_idx],
                    "top_reference_weight": float(weights[top_idx]),
                    "observed_cp10k": observed,
                    "expected_cp10k": expected,
                    "log2_observed_expected": log2_oe,
                    "expected_gene_is_meaningful": expected >= min_expected,
                }
            )

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


@_saveable_plot()
def plot_reference_scatter(
    comparison: pd.DataFrame,
    min_reference: float = 0.1,
    title_prefix: str = "",
    annotate_genes: Optional[Iterable[str]] = None,
    use_unmixed: bool = True,
) -> None:
    """
    Plot raw-vs-reference or unmixed-vs-reference expression scatter.

    When *use_unmixed* is ``True`` (default) plots unmixed vs reference.
    Set to ``False`` for raw vs reference.  Call twice to get both panels.

    Parameters
    ----------
    comparison:
        Output from raw_unmixed_reference_comparison.
    min_reference:
        Minimum reference CP10K to include.
    title_prefix:
        Optional prefix for the plot title.
    annotate_genes:
        Optional iterable of gene names to label on the plot.
    use_unmixed:
        If ``True`` (default) plot unmixed vs reference; else raw vs reference.
    """
    required = {"reference_cp10k", "raw_cp10k", "unmixed_cp10k", "gene"}
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(f"comparison missing columns: {sorted(missing)}")

    df = comparison[comparison["reference_cp10k"] >= min_reference].copy()
    if df.empty:
        raise ValueError("No rows remain after applying min_reference filter.")

    all_vals = pd.concat([
        df["reference_cp10k"], df["raw_cp10k"], df["unmixed_cp10k"]
    ])
    _log_margin = 0.15
    lim_min = 10 ** (np.log10(all_vals[all_vals > 0].min()) - _log_margin)
    lim_max = 10 ** (np.log10(all_vals.max()) + _log_margin)

    annotate_genes = set(annotate_genes or [])

    if use_unmixed:
        y_col, ylabel, panel = "unmixed_cp10k", "Unmixed expression, CP10K", "Unmixed vs reference"
    else:
        y_col, ylabel, panel = "raw_cp10k",     "Raw expression, CP10K",     "Raw vs reference"

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(df["reference_cp10k"], df[y_col], alpha=0.5)
    plt.plot([lim_min, lim_max], [lim_min, lim_max])
    if annotate_genes:
        sub = df[df["gene"].isin(annotate_genes)]
        for _, row in sub.iterrows():
            plt.text(row["reference_cp10k"], row[y_col], str(row["gene"]), fontsize=8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.xlabel("Reference expression, CP10K")
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix}{panel}".strip())
    plt.tight_layout()
    return fig


@_saveable_plot()
def plot_error_improvement(
    comparison: pd.DataFrame,
    title: str = "Reference error before vs after unmixing",
) -> None:
    """
    Plot absolute log2 O/E error before and after unmixing.

    Points below the diagonal improved after unmixing.
    """
    required = {"raw_abs_error", "unmixed_abs_error"}
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(f"comparison missing columns: {sorted(missing)}")

    df = comparison.copy()
    lim_max = max(df["raw_abs_error"].max(), df["unmixed_abs_error"].max())

    plt.figure(figsize=(5, 5))
    plt.scatter(df["raw_abs_error"], df["unmixed_abs_error"], alpha=0.5)
    plt.plot([0, lim_max], [0, lim_max])
    plt.xlabel("Raw absolute log2 O/E error")
    plt.ylabel("Unmixed absolute log2 O/E error")
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def plot_gene_improvement(
    comparison: pd.DataFrame,
    gene: str,
    title: Optional[str] = None,
) -> None:
    """
    Plot raw vs unmixed reference error for one gene.
    """
    df = comparison[comparison["gene"] == gene].copy()
    if df.empty:
        raise ValueError(f"No rows found for gene {gene!r}.")

    plot_error_improvement(
        df,
        title=title or f"{gene}: reference error before vs after unmixing",
    )


def plot_retention_by_group(
    retention: pd.DataFrame,
    gene: str,
    sort: bool = True,
    title: Optional[str] = None,
) -> None:
    """
    Bar plot of retention values for one gene across groups.
    """
    required = {"group", "gene", "retention"}
    missing = required - set(retention.columns)
    if missing:
        raise ValueError(f"retention missing columns: {sorted(missing)}")

    df = retention[retention["gene"] == gene].copy()
    if df.empty:
        raise ValueError(f"No rows found for gene {gene!r}.")

    if sort:
        df = df.sort_values("retention")

    plt.figure(figsize=(max(6, 0.35 * len(df)), 4))
    plt.bar(df["group"].astype(str), df["retention"])
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.ylabel("Unmixed / raw spot count")
    plt.xlabel("Group")
    plt.title(title or f"{gene} raw-to-unmixed retention")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


@_saveable_plot()
def plot_retention_heatmap(
    retention: pd.DataFrame,
    min_raw_spots: int = 5,
    vmin: float = 0.0,
    vmax: float = 1.0,
    col_sort: Optional[str] = "retention",
    row_sort: Optional[str] = "natural",
    gene_labels: Optional[Dict[str, str]] = None,
    title: str = "Raw-to-unmixed retention (clusters × genes)",
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Heatmap of retention values across all clusters and genes.

    Values < 1 mean spots were lost after unmixing; 1 means fully retained.
    Cells where raw_spots < min_raw_spots are shown as grey (unreliable).

    Parameters
    ----------
    retention:
        Output of gene_retention_by_group.
    min_raw_spots:
        Cluster x gene entries with fewer raw spots are masked as unreliable.
    vmin, vmax:
        Color scale bounds (default 0–1).
    col_sort:
        Column (gene) sort order.
        ``"retention"``  — ascending mean retention across clusters.
        ``"label"``      — alphabetical sort on display label (groups by
                           round then channel when gene_labels is provided).
        ``None``         — preserve original gene order.
    row_sort:
        Row (cluster) sort order.
        ``"natural"``    — natural / alphanumeric sort on cluster name
                           (e.g. 1, 2, 10 not 1, 10, 2).
        ``"retention"``  — ascending mean retention across genes.
        ``None``         — preserve original cluster order.
    gene_labels:
        Optional dict mapping gene name -> display label, e.g.
        {"Vip": "R1-488-Vip"}.  Applied to x-axis tick labels and used
        for ``col_sort="label"``.
    title:
        Plot title.
    figsize:
        Override automatic figure size.
    """
    import re

    required = {"group", "gene", "retention", "raw_spots"}
    missing = required - set(retention.columns)
    if missing:
        raise ValueError(f"retention missing columns: {sorted(missing)}")

    valid_col_sort = {"retention", "label", None}
    valid_row_sort = {"retention", "natural", None}
    if col_sort not in valid_col_sort:
        raise ValueError(f"col_sort must be one of {valid_col_sort}, got {col_sort!r}")
    if row_sort not in valid_row_sort:
        raise ValueError(f"row_sort must be one of {valid_row_sort}, got {row_sort!r}")

    mat = retention.pivot(index="group", columns="gene", values="retention")
    raw = retention.pivot(index="group", columns="gene", values="raw_spots")

    mask = raw < min_raw_spots

    # ── column ordering ───────────────────────────────────────────────────────
    if col_sort == "retention":
        col_order = mat.mean(axis=0).sort_values().index
        mat = mat[col_order]
        mask = mask[col_order]
    elif col_sort == "label":
        label_map = gene_labels or {}
        col_order = sorted(mat.columns, key=lambda g: label_map.get(g, g))
        mat = mat[col_order]
        mask = mask[col_order]

    # ── row ordering ──────────────────────────────────────────────────────────
    def _natural_key(s: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]

    if row_sort == "natural":
        row_order = sorted(mat.index, key=_natural_key)
        mat = mat.loc[row_order]
        mask = mask.loc[row_order]
    elif row_sort == "retention":
        row_order = mat.mean(axis=1).sort_values().index
        mat = mat.loc[row_order]
        mask = mask.loc[row_order]

    n_rows, n_cols = mat.shape
    if figsize is None:
        figsize = (max(6, 0.55 * n_cols), max(4, 0.35 * n_rows))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        mat.to_numpy(),
        aspect="auto",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # grey out unreliable cells
    grey = np.zeros((*mat.shape, 4))
    grey[..., :3] = 0.75
    grey[..., 3] = mask.to_numpy().astype(float) * 0.85
    ax.imshow(grey, aspect="auto", interpolation="nearest")

    xticklabels = [
        gene_labels.get(g, g) if gene_labels else g for g in mat.columns
    ]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(xticklabels, rotation=90, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Unmixed cluster label")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("% spots retained (Unmixed / raw spots)")

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Comparison summary plots
# -----------------------------------------------------------------------------


@_saveable_plot()
def plot_supertype_match_summary(
    cluster_matches: pd.DataFrame,
    title: str = "Cluster → reference supertype matches",
    figsize: Optional[Tuple] = None,
) -> None:
    """
    Dot/lollipop chart showing which reference supertype each HCR cluster
    matched to.  The x-axis is zoomed to the actual score range so small
    differences are visible even when all scores are near 1.  The matched
    supertype is embedded in the y-axis label so nothing overflows the axes.

    Parameters
    ----------
    cluster_matches:
        Second return value of ``raw_unmixed_reference_comparison``.
        Must have index = group label, columns include ``reference_label``
        and ``match_score``.
    title:
        Plot title.
    figsize:
        Override default figure size.
    """
    required = {"reference_label", "match_score"}
    missing = required - set(cluster_matches.columns)
    if missing:
        raise ValueError(f"cluster_matches missing columns: {sorted(missing)}")

    import re as _re
    import matplotlib.patches as mpatches
    import seaborn as _sns

    def _natural_key(s):
        return [int(t) if t.isdigit() else t for t in _re.split(r"(\d+)", str(s))]

    df = cluster_matches.copy().reset_index()
    df.columns = [c if c != "index" else "group" for c in df.columns]
    if "group" not in df.columns:
        df = df.rename(columns={df.columns[0]: "group"})

    df = df.sort_values("group", key=lambda s: s.map(_natural_key))

    # Colour by matched reference label
    # Colour by subclass marker found in the supertype label string.
    # Known markers listed in priority order; "other" catches anything unrecognised.
    SUBCLASS_COLORS = {
        "Vip":   "#2196F3",   # blue
        "Pvalb": "#F44336",   # red
        "Sst":   "#4CAF50",   # green
        "Lamp5": "#FF9800",   # orange
        "Ndnf":  "#9C27B0",   # purple
    }
    OTHER_COLOR = "#9E9E9E"   # grey for unlabelled

    def _subclass_color(label: str) -> str:
        for marker, color in SUBCLASS_COLORS.items():
            if marker.lower() in str(label).lower():
                return color
        return OTHER_COLOR

    # Legend patches
    present_markers = []
    for marker in list(SUBCLASS_COLORS) + ["other"]:
        col = SUBCLASS_COLORS.get(marker, OTHER_COLOR)
        if marker == "other":
            if any(_subclass_color(lab) == OTHER_COLOR for lab in df["reference_label"]):
                present_markers.append(mpatches.Patch(color=col, label="other"))
        else:
            if any(marker.lower() in str(lab).lower() for lab in df["reference_label"]):
                present_markers.append(mpatches.Patch(color=col, label=marker))

    # Y-axis label: "cluster  |  supertype"
    y_labels = [f"{row['group']}  |  {row['reference_label']}" for _, row in df.iterrows()]
    colors   = [_subclass_color(lab) for lab in df["reference_label"]]
    scores   = df["match_score"].values
    y_pos    = range(len(df))

    n = len(df)
    fig_h = max(4, 0.38 * n)

    with _sns.plotting_context("notebook"):
        fig, ax = plt.subplots(figsize=figsize or (7, fig_h))

        # Lollipop: thin stem + coloured dot
        x_ref = 0.8  # pin stems to the left axis edge
        for i, (score, color) in enumerate(zip(scores, colors)):
            ax.plot([x_ref, score], [i, i], color=color, linewidth=1.2, alpha=0.6)
            ax.scatter(score, i, color=color, s=60, zorder=3)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(y_labels)

        ax.set_xlim(0.8, 1.0)
        ax.invert_yaxis()

        if present_markers:
            ax.legend(
                handles=present_markers, title="Subclass",
                loc="upper left", bbox_to_anchor=(1.01, 1),
                borderaxespad=0, frameon=True, framealpha=0.9,
            )

        ax.axvline(scores.mean(), color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Cosine similarity (match score)")
        ax.set_ylabel("Cluster  |  matched supertype")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        return fig


@_saveable_plot()
def plot_improvement_heatmap(
    comparison: pd.DataFrame,
    cluster_matches: Optional[pd.DataFrame] = None,
    row_sort: str = "natural",
    col_sort: str = "mean_improvement",
    vmin: float = -2.0,
    vmax: float = 2.0,
    title: str = "Unmixing improvement (Δ|log₂ O/E|, raw − unmixed)",
    figsize: Optional[Tuple] = None,
) -> None:
    """
    Heatmap of unmixing improvement per cluster × gene.

    Each cell shows ``improvement = |raw_log2_oe| − |unmixed_log2_oe|``.
    Positive (warm) = gene moved closer to reference after unmixing.
    Negative (cool) = gene moved further away.

    Parameters
    ----------
    comparison:
        Output of ``raw_unmixed_reference_comparison``.
    cluster_matches:
        Optional; if provided, each row label is annotated with its matched
        reference supertype.
    row_sort:
        ``"natural"`` – alphanumeric cluster order (default).
        ``"mean_improvement"`` – clusters with most improvement first.
        ``None`` – preserve DataFrame order.
    col_sort:
        ``"mean_improvement"`` – genes most improved on average first (default).
        ``"name"`` – alphabetical gene order.
        ``None`` – preserve DataFrame order.
    vmin, vmax:
        Colour scale limits for the diverging RdBu_r palette.
    title:
        Plot title.
    figsize:
        Override default figure size.
    """
    import re as _re

    required = {"group", "gene", "improvement"}
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(f"comparison missing columns: {sorted(missing)}")

    mat = comparison.pivot_table(index="group", columns="gene", values="improvement")

    def _natural_key(s):
        return [int(t) if t.isdigit() else t for t in _re.split(r"(\d+)", str(s))]

    if row_sort == "natural":
        mat = mat.loc[sorted(mat.index, key=_natural_key)]
    elif row_sort == "mean_improvement":
        mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]

    if col_sort == "mean_improvement":
        mat = mat[mat.mean(axis=0).sort_values(ascending=False).index]
    elif col_sort == "name":
        mat = mat[sorted(mat.columns)]

    # Row labels: "cluster  →  supertype" when cluster_matches provided
    if cluster_matches is not None and "reference_label" in cluster_matches.columns:
        row_labels = [
            f"{g}  →  {cluster_matches.loc[g, 'reference_label']}"
            if g in cluster_matches.index
            else str(g)
            for g in mat.index
        ]
    else:
        row_labels = [str(g) for g in mat.index]

    n_rows, n_cols = mat.shape
    fig_w = max(8, 0.55 * n_cols)
    fig_h = max(4, 0.45 * n_rows)
    fig, ax = plt.subplots(figsize=figsize or (fig_w, fig_h))

    import matplotlib.colors as _mcolors
    norm = _mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Cluster  →  matched supertype")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Improvement (|raw| − |unmixed| log₂ O/E)")

    plt.tight_layout()
    return fig


@_saveable_plot()
def plot_cluster_reference_expression(
    comparison: pd.DataFrame,
    cluster_matches: pd.DataFrame,
    gene_labels: Optional[Dict] = None,
    use_unmixed: bool = True,
    log_transform: bool = False,
    normalize: Optional[str] = "row",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_diff: str = "RdBu_r",
    diff_vmax: Optional[float] = None,
    figsize: Optional[Tuple] = None,
    title_left: Optional[str] = None,
    title_right: Optional[str] = None,
    title_diff: Optional[str] = None,
) -> None:
    """
    Three-panel heatmap comparing HCR cluster expression to matched reference
    supertypes, with a difference panel.

    Left   — observed CP10K (raw or unmixed) per cluster × gene.
    Middle — reference CP10K for each cluster's matched supertype.
    Right  — difference (observed − reference) using a diverging colormap.

    Rows are aligned: cluster N on the left faces its matched supertype in the
    middle.  Genes are ordered by *gene_labels* (round-channel order) when
    provided, falling back to alphabetical.

    Parameters
    ----------
    comparison:
        Output of ``raw_unmixed_reference_comparison``.  Must contain columns
        ``group``, ``gene``, ``raw_cp10k`` (or ``unmixed_cp10k``),
        ``reference_cp10k``, and ``reference_label``.
    cluster_matches:
        Second return value of ``raw_unmixed_reference_comparison``.  Used
        only for its ``reference_label`` column to build right-panel row labels.
    gene_labels:
        Optional dict mapping gene name → display label (e.g.
        ``{"Vip": "R1-488-Vip", ...}``).  When supplied the columns are sorted
        by this string so related round/channel groups sit together.
    use_unmixed:
        If ``True`` (default) use ``unmixed_cp10k`` for the left panel,
        otherwise use ``raw_cp10k``.
    log_transform:
        If ``True``, apply ``log2(CP10K + 1)`` before any normalisation.
        Compresses dynamic range so dim genes are more visible.  Default
        ``False`` (linear CP10K).
    normalize:
        How to normalise each matrix before display:

        * ``"row"`` *(default)* — divide each row by its sum so values are a
          fraction of total per-cluster expression.
        * ``"col"`` — divide each column by its sum so values are a fraction of
          total per-gene expression.
        * ``None`` — no normalisation; show raw CP10K (or log₂).
    cmap:
        Colormap for the HCR and reference panels.  Default ``"viridis"``.
    vmin, vmax:
        Explicit colour-scale limits for the HCR and reference panels.
        Defaults: ``vmin=0``, ``vmax`` auto-computed from the data maximum.
    cmap_diff:
        Colormap for the difference panel.  Should be diverging.
        Default ``"RdBu_r"`` (red = HCR > reference, blue = HCR < reference).
    diff_vmax:
        Symmetric colour-scale limit (±) for the difference panel.
        Default: auto-computed as ``max(|diff|)``.
    figsize:
        Override automatic figure size.
    title_left, title_right, title_diff:
        Override individual panel titles.
    """
    import re as _re
    import seaborn as _sns

    if normalize not in ("row", "col", None):
        raise ValueError(f"normalize must be 'row', 'col', or None, got {normalize!r}")

    obs_col = "unmixed_cp10k" if use_unmixed else "raw_cp10k"
    required = {"group", "gene", obs_col, "reference_cp10k", "reference_label"}
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(f"comparison missing columns: {sorted(missing)}")

    def _natural_key(s):
        return [int(t) if t.isdigit() else t for t in _re.split(r"(\d+)", str(s))]

    # ── Build observed / reference matrices (cluster × gene) ─────────────────
    obs_mat = comparison.pivot_table(index="group", columns="gene", values=obs_col)
    ref_mat = comparison.pivot_table(index="group", columns="gene", values="reference_cp10k")

    # ── Gene column order ─────────────────────────────────────────────────────
    genes_present = obs_mat.columns.tolist()
    if gene_labels:
        def _gene_sort_key(g):
            lbl = gene_labels.get(g, g)
            return _natural_key(lbl)
        genes_present = sorted(genes_present, key=_gene_sort_key)
    else:
        genes_present = sorted(genes_present, key=_natural_key)

    obs_mat = obs_mat[genes_present]
    ref_mat = ref_mat[genes_present]

    # ── Row order: natural sort of cluster labels ─────────────────────────────
    row_order = sorted(obs_mat.index, key=_natural_key)
    obs_mat = obs_mat.loc[row_order]
    ref_mat = ref_mat.loc[row_order]

    # ── Optional log2 transform ───────────────────────────────────────────────
    if log_transform:
        obs_mat = np.log2(obs_mat + 1)
        ref_mat = np.log2(ref_mat + 1)

    # ── Normalisation ─────────────────────────────────────────────────────────
    def _norm(df):
        if normalize == "row":
            totals = df.sum(axis=1).replace(0, np.nan)
            return df.div(totals, axis=0).fillna(0)
        elif normalize == "col":
            totals = df.sum(axis=0).replace(0, np.nan)
            return df.div(totals, axis=1).fillna(0)
        else:
            return df

    obs_norm = _norm(obs_mat)
    ref_norm = _norm(ref_mat)
    diff_mat = obs_norm - ref_norm          # positive = HCR > ref

    # ── Colour-scale limits ───────────────────────────────────────────────────
    _vmin = vmin if vmin is not None else 0.0
    _vmax = vmax if vmax is not None else max(obs_norm.values.max(), ref_norm.values.max())
    _diff_vmax = diff_vmax if diff_vmax is not None else float(np.abs(diff_mat.values).max())

    # ── Display labels ────────────────────────────────────────────────────────
    x_labels = [gene_labels.get(g, g) if gene_labels else g for g in genes_present]
    y_left   = [str(r) for r in row_order]
    def _fmt_supertype(label: str) -> str:
        return label.replace("chandelier", "chan.").replace("Chandelier", "Chan.")

    y_mid    = [
        _fmt_supertype(cluster_matches.loc[r, "reference_label"])
        if r in cluster_matches.index else str(r)
        for r in row_order
    ]

    # ── Auto label strings ────────────────────────────────────────────────────
    _base        = "log₂(CP10K+1)" if log_transform else "CP10K"
    _norm_suffix = {"row": " (row-norm)", "col": " (col-norm)", None: ""}[normalize]
    _cbar_lbl    = _base + _norm_suffix
    _obs_label   = "unmixed" if use_unmixed else "raw"
    _norm_label  = {"row": "row-normalised", "col": "col-normalised", None: "no normalisation"}[normalize]

    # ── Layout ────────────────────────────────────────────────────────────────
    n_rows, n_cols = obs_norm.shape
    fig_w = max(18, 0.55 * n_cols * 3)
    fig_h = max(5,  0.40 * n_rows)

    with _sns.plotting_context("notebook"):
        fig, (ax_l, ax_m, ax_d) = plt.subplots(
            1, 3,
            figsize=figsize or (fig_w, fig_h),
            gridspec_kw={"wspace": 0.55},
        )

        # ── Left and middle panels (shared colormap + scale) ─────────────────
        for ax, mat, title in [
            (ax_l, obs_norm, title_left  or f"HCR clusters ({_obs_label})"),
            (ax_m, ref_norm, title_right or "MERFISH supertypes"),
        ]:
            im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                           vmin=_vmin, vmax=_vmax)
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(x_labels, rotation=90, ha="center")
            ax.set_yticks(range(n_rows))
            ax.set_xlabel("")
            ax.set_title(title)

        # Left panel: cluster labels
        ax_l.set_yticklabels(y_left)

        # Middle panel: cluster → matched supertype labels
        ax_m.set_yticklabels(y_mid)

        cbar_main = fig.colorbar(im, ax=[ax_l, ax_m], fraction=0.015, pad=0.02)
        cbar_main.set_label(_cbar_lbl)

        # ── Difference panel ─────────────────────────────────────────────────
        im_d = ax_d.imshow(
            diff_mat.values, aspect="auto", cmap=cmap_diff,
            vmin=-_diff_vmax, vmax=_diff_vmax,
        )
        ax_d.set_xticks(range(n_cols))
        ax_d.set_xticklabels(x_labels, rotation=90, ha="center")
        ax_d.set_yticks(range(n_rows))
        ax_d.set_yticklabels(y_left)
        ax_d.set_xlabel("")
        ax_d.set_title(title_diff or "Difference: HCR − MERFISH")

        cbar_diff = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
        cbar_diff.set_label(f"Δ {_cbar_lbl}  (red = HCR higher)")

        plt.suptitle(
            f"Cluster expression vs. MERFISH supertypes  ({_norm_label})", y=1.01
        )
        plt.tight_layout()
        return fig


@_saveable_plot()
def plot_gene_expression_bias(
    comparison: pd.DataFrame,
    gene_labels: Optional[Dict] = None,
    use_unmixed: bool = True,
    normalize: Optional[str] = None,
    log_transform: bool = False,
    figsize: Optional[Tuple] = None,
) -> None:
    """
    Aggregate point plot showing per-gene expression bias relative to the
    MERFISH reference across all matched clusters.

    For each gene the values from every cluster are shown as a strip of dots;
    the mean is overlaid as a larger marker with a stem connecting to zero.
    Genes above zero are systematically over-expressed in HCR; below = under.

    Two metrics are available, controlled by *normalize*:

    * ``normalize=None`` *(default)* — uses ``log₂(observed / expected)``,
      already in the ``comparison`` table.  Scale is log-fold-change.
    * ``normalize="row"`` or ``"col"`` — recomputes the normalised difference
      matrix (HCR − reference) in the same way as
      ``plot_cluster_reference_expression``, then aggregates by gene.  Values
      are fraction-of-total differences and tell a slightly different story:
      which genes "take more share" in HCR vs the reference profile.

    Parameters
    ----------
    comparison:
        Output of ``raw_unmixed_reference_comparison``.
    gene_labels:
        Optional dict mapping gene → display label for ordering/x-ticks.
    use_unmixed:
        If ``True`` (default) use unmixed expression; otherwise raw.
    normalize:
        ``None`` — use log₂ O/E (default).
        ``"row"`` or ``"col"`` — use row- or col-normalised CP10K difference.
    log_transform:
        Only relevant when *normalize* is not ``None``.  If ``True``, apply
        ``log2(CP10K + 1)`` before normalising (mirrors the heatmap option).
    figsize:
        Override automatic figure size.
    """
    import re as _re
    import seaborn as _sns

    if normalize not in (None, "row", "col"):
        raise ValueError(f"normalize must be None, 'row', or 'col', got {normalize!r}")

    obs_col = "unmixed_cp10k" if use_unmixed else "raw_cp10k"

    def _natural_key(s):
        return [int(t) if t.isdigit() else t for t in _re.split(r"(\d+)", str(s))]

    # ── Order genes ───────────────────────────────────────────────────────────
    genes = comparison["gene"].dropna().unique().tolist()
    if gene_labels:
        genes = sorted(genes, key=lambda g: _natural_key(gene_labels.get(g, g)))
    else:
        genes = sorted(genes, key=_natural_key)
    x_labels = [gene_labels.get(g, g) if gene_labels else g for g in genes]

    # ── Build per-cluster values ──────────────────────────────────────────────
    if normalize is None:
        # Use pre-computed log2 O/E
        oe_col = "unmixed_log2_oe" if use_unmixed else "raw_log2_oe"
        if oe_col not in comparison.columns:
            raise ValueError(f"comparison missing column '{oe_col}'")
        sub = comparison[comparison["gene"].isin(genes)][["gene", "group", oe_col]].copy()
        sub = sub.rename(columns={oe_col: "_val"})
        ylabel = f"log₂(HCR / MERFISH ref)  ({'unmixed' if use_unmixed else 'raw'})"
        title_metric = "log₂ O/E"
    else:
        # Recompute normalised difference: same logic as the heatmap
        required = {"group", "gene", obs_col, "reference_cp10k"}
        missing = required - set(comparison.columns)
        if missing:
            raise ValueError(f"comparison missing columns: {sorted(missing)}")

        obs_mat = comparison.pivot_table(index="group", columns="gene", values=obs_col)
        ref_mat = comparison.pivot_table(index="group", columns="gene", values="reference_cp10k")
        obs_mat = obs_mat[[g for g in genes if g in obs_mat.columns]]
        ref_mat = ref_mat[[g for g in genes if g in ref_mat.columns]]

        if log_transform:
            obs_mat = np.log2(obs_mat + 1)
            ref_mat = np.log2(ref_mat + 1)

        def _norm(df):
            if normalize == "row":
                totals = df.sum(axis=1).replace(0, np.nan)
                return df.div(totals, axis=0).fillna(0)
            else:  # col
                totals = df.sum(axis=0).replace(0, np.nan)
                return df.div(totals, axis=1).fillna(0)

        diff_mat = _norm(obs_mat) - _norm(ref_mat)

        # Melt to long form
        diff_long = diff_mat.reset_index().melt(id_vars="group", var_name="gene", value_name="_val")
        sub = diff_long[diff_long["gene"].isin(genes)].copy()

        _base      = "log₂(CP10K+1)" if log_transform else "CP10K"
        _norm_lbl  = "row-norm" if normalize == "row" else "col-norm"
        ylabel     = f"Δ {_base} ({_norm_lbl})  HCR − MERFISH ref  ({'unmixed' if use_unmixed else 'raw'})"
        title_metric = f"{_norm_lbl} difference"

    # ── Aggregate ─────────────────────────────────────────────────────────────
    gene_stats = (
        sub.groupby("gene")["_val"]
        .agg(
            mean="mean",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reindex(genes)
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_genes = len(genes)
    with _sns.plotting_context("notebook"):
        fig, ax = plt.subplots(figsize=figsize or (max(10, 0.55 * n_genes), 5))

        # Strip
        for i, g in enumerate(genes):
            vals = sub.loc[sub["gene"] == g, "_val"].dropna().values
            jitter = np.random.default_rng(42).uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter, vals,
                s=18, alpha=0.45, color="steelblue", linewidths=0, zorder=2,
            )

        # Mean stem + dot
        for i, g in enumerate(genes):
            m = gene_stats.loc[g, "mean"]
            color = "#c0392b" if m > 0 else "#2980b9"
            ax.plot([i, i], [0, m], color=color, lw=1.5, zorder=3)
            ax.scatter([i], [m], s=80, color=color, zorder=4,
                       edgecolors="white", linewidths=0.8)

        # IQR bar
        for i, g in enumerate(genes):
            ax.plot(
                [i, i],
                [gene_stats.loc[g, "q25"], gene_stats.loc[g, "q75"]],
                color="gray", lw=3, alpha=0.35, zorder=1, solid_capstyle="round",
            )

        ax.axhline(0, color="black", lw=1, zorder=0)
        ax.set_xticks(np.arange(n_genes))
        ax.set_xticklabels(x_labels, rotation=90, ha="center")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, n_genes - 0.4)
        ax.set_title(
            f"Gene expression bias vs. MERFISH reference  ({title_metric})\n"
            "(red = HCR over-expressed, blue = under-expressed; dot = mean, bar = IQR)"
        )

        plt.tight_layout()
        return fig


@_saveable_plot()
def plot_cell_x_gene_with_clusters(
    spots: pd.DataFrame,
    cluster_meta: pd.DataFrame,
    gene_labels: Optional[Dict[str, str]] = None,
    chan_col: str = "unmixed_gene",
    clip_range: Tuple[int, int] = (0, 200),
    figsize: Tuple[float, float] = (8, 10),
    title: str = "Cell × gene",
) -> "plt.Figure":
    """
    Plot a cell × gene heatmap with cells grouped by pre-computed cluster labels.

    Columns are named and sorted using round-chan-gene display labels from
    *gene_labels*.  No re-clustering is performed — the cluster assignments
    already present in *cluster_meta* are used directly.

    Parameters
    ----------
    spots:
        Spot table with columns ``cell_id`` and *chan_col*.
    cluster_meta:
        DataFrame indexed by ``cell_id`` with a ``cluster_label`` column.
    gene_labels:
        Dict mapping plain gene name → round-chan-gene display label
        (e.g. ``{"Gad2": "R1-488-Gad2"}``).  When ``None``, plain gene
        names are used as column headers.
    chan_col:
        Column in *spots* containing the gene/channel name.
    clip_range:
        ``(min, max)`` count clip applied before plotting.
    figsize:
        ``(width, height)`` in inches.
    title:
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import re as _re
    import seaborn as _sns
    from aind_hcr_qc.viz.cell_x_gene import plot_cell_x_gene_clustered as _plot_cxg

    _sns.set_context("notebook")

    def _natural_key(s: str):
        return [int(t) if t.isdigit() else t.lower() for t in _re.split(r"(\d+)", str(s))]

    # 1. Build cell × gene count matrix restricted to cells in cluster_meta
    counts = spots_to_cell_gene_counts(
        spots,
        cell_ids=cluster_meta.index,
        chan_col=chan_col,
    )

    # 2. Rename columns to round-chan-gene display labels
    if gene_labels is not None:
        counts = counts.rename(columns=gene_labels)

    # Drop any column whose name is the string "nan" (from NaN gene values)
    counts = counts[[c for c in counts.columns if str(c).lower() != "nan"]]

    # 3. Natural-sort columns so rounds and channels are in order
    col_order = sorted(counts.columns, key=_natural_key)
    counts = counts[col_order]

    # 4. Sort rows by cluster_label (natural order), preserving cell_id order within each cluster
    meta_aligned = cluster_meta.loc[cluster_meta.index.intersection(counts.index)]
    cluster_order = sorted(meta_aligned["cluster_label"].unique(), key=_natural_key)
    sorted_ids = pd.Index(
        [cid for cl in cluster_order for cid in meta_aligned[meta_aligned["cluster_label"] == cl].index]
    )
    counts_sorted = counts.reindex(sorted_ids)

    # Drop genes (columns) with all-NaN values, then fill remaining NaN with 0
    counts_sorted = counts_sorted.dropna(axis=1, how="all").fillna(0).astype(int)
    cluster_labels_arr = meta_aligned.loc[counts_sorted.index, "cluster_label"].to_numpy()

    # Pre-clip so the cluster_result passed below already has the clipped values
    # (plot_cell_x_gene_clustered re-assigns cxg from cluster_result, which would
    #  bypass its own clip step if we don't do it here).
    counts_clipped = counts_sorted.clip(lower=clip_range[0], upper=clip_range[1])

    # 5. Call the library function with pre-computed clustering — no re-clustering
    fig, _, _ = _plot_cxg(
        counts_clipped,
        clip_range=clip_range,
        fig_size=figsize,
        cluster_result=(counts_clipped, cluster_labels_arr, counts_clipped.index),
        gene_sort=None,
        add_cluster_labels=True,
        title=title,
    )

    # Pin the colormap to exactly clip_range so all plots use a consistent scale
    for ax in fig.axes:
        for im in ax.get_images():
            im.set_clim(clip_range[0], clip_range[1])

    return fig


# -----------------------------------------------------------------------------
# Multi-mouse combined-results plots
# -----------------------------------------------------------------------------


@_saveable_plot()
def plot_multi_mouse_error_improvement(
    comparison: pd.DataFrame,
    spot_filter: Optional[str] = None,
    figsize: Optional[Tuple] = None,
    title: str = "Reference error before vs after unmixing (all mice)",
) -> "plt.Figure":
    """
    Scatter of raw vs unmixed absolute log2 O/E error for all mice.

    One subplot per ``spot_filter`` value present in *comparison*.  Points are
    coloured by ``mouse_id``.

    Parameters
    ----------
    comparison:
        Combined comparison CSV loaded as a DataFrame.  Must have columns
        ``mouse_id``, ``spot_filter``, ``raw_abs_error``, ``unmixed_abs_error``.
    spot_filter:
        If given, restrict to this filter value (e.g. ``"valid"``).
    figsize:
        Override figure size.
    title:
        Figure suptitle.
    """
    import re as _re

    df = comparison.copy()
    if spot_filter is not None:
        df = df[df["spot_filter"] == spot_filter]

    filters = sorted(df["spot_filter"].unique())
    mice    = sorted(df["mouse_id"].astype(str).unique(), key=lambda s: [int(t) if t.isdigit() else t for t in _re.split(r"(\d+)", s)])
    cmap    = plt.cm.get_cmap("tab10", len(mice))
    colors  = {m: cmap(i) for i, m in enumerate(mice)}

    n_panels = len(filters)
    fw, fh = figsize or (5.5 * n_panels, 5.0)
    fig, axes = plt.subplots(1, n_panels, figsize=(fw, fh), squeeze=False)

    lim = max(df["raw_abs_error"].max(), df["unmixed_abs_error"].max()) * 1.05

    for ax, filt in zip(axes[0], filters):
        sub = df[df["spot_filter"] == filt]
        for mouse in mice:
            msub = sub[sub["mouse_id"].astype(str) == mouse]
            ax.scatter(
                msub["raw_abs_error"], msub["unmixed_abs_error"],
                color=colors[mouse], alpha=0.45, s=18, label=str(mouse),
            )
        ax.plot([0, lim], [0, lim], color="steelblue", linewidth=1.2)
        ax.set_xlim(-0.1, lim)
        ax.set_ylim(-0.1, lim)
        ax.set_xlabel("Raw absolute log2 O/E error")
        ax.set_ylabel("Unmixed absolute log2 O/E error")
        ax.set_title(f"spot_filter = {filt}")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[m], markersize=7, label=m) for m in mice]
    fig.legend(handles=handles, title="mouse_id", bbox_to_anchor=(1.01, 0.9), loc="upper left", fontsize=9)
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


@_saveable_plot()
def plot_multi_mouse_improvement_by_gene(
    comparison: pd.DataFrame,
    spot_filter: Optional[str] = None,
    figsize: Optional[Tuple] = None,
    title: str = "Mean unmixing improvement by gene (all mice)",
) -> "plt.Figure":
    """
    Horizontal bar chart of mean improvement per gene across all mice.

    One subplot per ``spot_filter``.  Each bar shows the cross-mouse mean;
    individual mouse values are overlaid as jittered dots.

    Parameters
    ----------
    comparison:
        Combined comparison CSV.  Must have ``mouse_id``, ``spot_filter``,
        ``gene``, ``improvement``.
    spot_filter:
        If given, restrict to this filter value.
    figsize:
        Override figure size.
    title:
        Figure suptitle.
    """
    import re as _re

    df = comparison.copy()
    if spot_filter is not None:
        df = df[df["spot_filter"] == spot_filter]

    filters = sorted(df["spot_filter"].unique())
    mice    = sorted(df["mouse_id"].astype(str).unique(), key=lambda s: [int(t) if t.isdigit() else t for t in _re.split(r"(\d+)", s)])
    cmap    = plt.cm.get_cmap("tab10", len(mice))
    colors  = {m: cmap(i) for i, m in enumerate(mice)}

    # Gene list: derive from the FULL comparison so genes measured by only some
    # mice or only one spot_filter still appear in every panel.
    gene_mean = comparison.groupby("gene")["improvement"].mean().sort_values(ascending=True)
    genes = gene_mean.index.tolist()

    n_panels = len(filters)
    fw, fh = figsize or (7.0, max(4, 0.4 * len(genes)))
    fig, axes = plt.subplots(1, n_panels, figsize=(fw * n_panels, fh), squeeze=False)

    rng = __import__("numpy").random.default_rng(0)

    for ax, filt in zip(axes[0], filters):
        sub = df[df["spot_filter"] == filt]
        means = sub.groupby("gene")["improvement"].mean().reindex(genes)
        colors_bar = ["#d62728" if v < 0 else "#2ca02c" for v in means]
        ax.barh(genes, means, color=colors_bar, alpha=0.65, zorder=2)

        for mouse in mice:
            msub = sub[sub["mouse_id"].astype(str) == mouse]
            gene_vals = msub.groupby("gene")["improvement"].mean().reindex(genes)
            y_jitter  = rng.uniform(-0.25, 0.25, size=len(genes))
            ax.scatter(
                gene_vals, __import__("numpy").arange(len(genes)) + y_jitter,
                color=colors[mouse], s=18, zorder=3, alpha=0.85, label=str(mouse),
            )

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes)
        ax.set_xlabel("Mean improvement (Δ|log₂ O/E|)")
        ax.set_title(f"spot_filter = {filt}")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[m], markersize=7, label=m) for m in mice]
    fig.legend(handles=handles, title="mouse_id", bbox_to_anchor=(1.01, 0.9), loc="upper left", fontsize=9)
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


@_saveable_plot()
def plot_multi_mouse_match_scores(
    cluster_matches: pd.DataFrame,
    spot_filter: Optional[str] = None,
    figsize: Optional[Tuple] = None,
    title: str = "Cluster → reference match scores (all mice)",
) -> "plt.Figure":
    """
    Strip + box plot of cluster-to-reference cosine match scores per mouse.

    One subplot per ``spot_filter``.  Each mouse is one group on the x-axis;
    each dot is one cluster.

    Parameters
    ----------
    cluster_matches:
        Combined cluster_matches CSV.  Must have ``mouse_id``, ``spot_filter``,
        ``match_score``.
    spot_filter:
        If given, restrict to this filter value.
    figsize:
        Override figure size.
    title:
        Figure suptitle.
    """
    import re as _re
    import numpy as _np

    df = cluster_matches.copy()
    if spot_filter is not None:
        df = df[df["spot_filter"] == spot_filter]

    filters = sorted(df["spot_filter"].unique())
    mice    = sorted(df["mouse_id"].astype(str).unique(), key=lambda s: [int(t) if t.isdigit() else t for t in _re.split(r"(\d+)", s)])
    cmap    = plt.cm.get_cmap("tab10", len(mice))
    colors  = {m: cmap(i) for i, m in enumerate(mice)}

    n_panels = len(filters)
    fw, fh = figsize or (max(5, 1.2 * len(mice)) * n_panels, 4.5)
    fig, axes = plt.subplots(1, n_panels, figsize=(fw, fh), squeeze=False)

    rng = _np.random.default_rng(0)

    for ax, filt in zip(axes[0], filters):
        sub = df[df["spot_filter"] == filt]
        for xi, mouse in enumerate(mice):
            vals = sub[sub["mouse_id"].astype(str) == mouse]["match_score"].dropna().values
            bp = ax.boxplot(vals, positions=[xi], widths=0.4, patch_artist=True,
                            boxprops=dict(facecolor=colors[mouse], alpha=0.35),
                            medianprops=dict(color="black", linewidth=1.5),
                            whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
                            flierprops=dict(marker=""), showfliers=False)
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(xi + jitter, vals, color=colors[mouse], s=22, alpha=0.75, zorder=3)

        ax.set_xticks(range(len(mice)))
        ax.set_xticklabels(mice, rotation=30, ha="right")
        ax.set_ylabel("Cosine match score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"spot_filter = {filt}")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


@_saveable_plot()
def plot_multi_mouse_error_ecdf(
    comparison: pd.DataFrame,
    spot_filter: Optional[str] = None,
    figsize: Optional[Tuple] = None,
    title: str = "ECDF of absolute log2 O/E error: raw vs unmixed (pooled mice)",
) -> "plt.Figure":
    """
    ECDF of raw vs unmixed absolute log2 O/E error, pooled across all mice.

    One panel per ``spot_filter``.  Two bold curves show the pooled raw
    (dashed, red) and unmixed (solid, green) distributions.  Thin per-mouse
    lines are drawn at low opacity to convey animal-to-animal variability.
    A leftward shift in the unmixed curve indicates error reduction.

    Parameters
    ----------
    comparison:
        Combined comparison CSV.  Must have ``mouse_id``, ``spot_filter``,
        ``raw_abs_error``, ``unmixed_abs_error``.
    spot_filter:
        If given, restrict to this filter value.
    figsize:
        Override figure size.
    title:
        Figure suptitle.
    """
    import re as _re
    import numpy as _np

    RAW_COLOR    = "#d62728"
    UNMIX_COLOR  = "#2ca02c"

    df = comparison.copy()
    if spot_filter is not None:
        df = df[df["spot_filter"] == spot_filter]

    filters = sorted(df["spot_filter"].unique())
    mice    = sorted(df["mouse_id"].astype(str).unique(), key=lambda s: [int(t) if t.isdigit() else t for t in _re.split(r"(\d+)", s)])

    n_panels = len(filters)
    fw, fh = figsize or (5.5 * n_panels, 4.5)
    fig, axes = plt.subplots(1, n_panels, figsize=(fw, fh), squeeze=False)

    for ax, filt in zip(axes[0], filters):
        sub = df[df["spot_filter"] == filt]

        # Thin per-mouse lines for variability context
        for mouse in mice:
            msub = sub[sub["mouse_id"].astype(str) == mouse]
            for col, color in [("raw_abs_error", RAW_COLOR), ("unmixed_abs_error", UNMIX_COLOR)]:
                vals = _np.sort(msub[col].dropna().values)
                if len(vals) == 0:
                    continue
                ecdf_y = _np.arange(1, len(vals) + 1) / len(vals)
                ax.plot(vals, ecdf_y, color=color, linewidth=0.7, alpha=0.25)

        # Bold pooled curves
        for col, color, ls in [
            ("raw_abs_error",    RAW_COLOR,   "--"),
            ("unmixed_abs_error", UNMIX_COLOR, "-"),
        ]:
            vals = _np.sort(sub[col].dropna().values)
            if len(vals) == 0:
                continue
            ecdf_y = _np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, ecdf_y, color=color, linestyle=ls, linewidth=2.2)

        ax.set_xlabel("Absolute log2 O/E error")
        ax.set_ylabel("Cumulative proportion")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"spot_filter = {filt}")
        ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":")

    handles = [
        plt.Line2D([0], [0], color=RAW_COLOR,   linestyle="--", linewidth=2, label="raw"),
        plt.Line2D([0], [0], color=UNMIX_COLOR,  linestyle="-",  linewidth=2, label="unmixed"),
    ]
    fig.legend(handles=handles, title="condition", bbox_to_anchor=(1.01, 0.95), loc="upper left", fontsize=9)
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


@_saveable_plot()
def plot_multi_mouse_filter_ecdf(
    comparison: pd.DataFrame,
    figsize: Optional[Tuple] = None,
    title: str = "ECDF by spot_filter: raw vs unmixed (pooled mice)",
) -> "plt.Figure":
    """
    Single-panel ECDF comparing raw and unmixed error across all ``spot_filter``
    values, pooled across mice.

    Each ``spot_filter`` gets a distinct colour.  Within each filter, the
    dashed line is raw and the solid line is unmixed.  This lets you see both
    how filters differ in baseline error *and* how much unmixing helps within
    each filter.

    Parameters
    ----------
    comparison:
        Combined comparison CSV.  Must have ``mouse_id``, ``spot_filter``,
        ``raw_abs_error``, ``unmixed_abs_error``.
    figsize:
        Override figure size.
    title:
        Figure suptitle.
    """
    import numpy as _np

    df = comparison.copy()

    filters = sorted(df["spot_filter"].unique())
    cmap    = plt.cm.get_cmap("Set1", len(filters))
    colors  = {f: cmap(i) for i, f in enumerate(filters)}

    fw, fh = figsize or (6.0, 4.5)
    fig, ax = plt.subplots(figsize=(fw, fh))

    for filt in filters:
        sub = df[df["spot_filter"] == filt]
        for col, ls in [("raw_abs_error", "--"), ("unmixed_abs_error", "-")]:
            vals = _np.sort(sub[col].dropna().values)
            if len(vals) == 0:
                continue
            ecdf_y = _np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, ecdf_y, color=colors[filt], linestyle=ls, linewidth=1.9, alpha=0.9)

    ax.set_xlabel("Absolute log2 O/E error")
    ax.set_ylabel("Cumulative proportion")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":")

    filter_handles = [
        plt.Line2D([0], [0], color=colors[f], linewidth=2, label=f) for f in filters
    ]
    style_handles = [
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5, label="raw"),
        plt.Line2D([0], [0], color="gray", linestyle="-",  linewidth=1.5, label="unmixed"),
    ]
    ax.legend(handles=filter_handles + style_handles, title="filter / condition", fontsize=9)
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


def save_all_figures(
    comparison: pd.DataFrame,
    cluster_matches: pd.DataFrame,
    gene_labels: Optional[Dict] = None,
    output_dir: "str | Path" = "figures",
    dpi: int = 150,
    formats: Tuple = ("png",),
    expression_vmax: Optional[float] = 0.3,
) -> None:
    """
    Save all standard atlas_compare figures to *output_dir*.

    Produces ten numbered PNG files (configurable via *formats*):

    01  supertype_match_summary
    02  improvement_heatmap
    03  expression_heatmap_row_norm
    04  expression_heatmap_col_norm
    05  expression_heatmap_no_norm_log2
    06  gene_bias_log2_oe
    07  gene_bias_row_norm_diff
    08  reference_scatter_raw
    09  reference_scatter_unmixed
    10  error_improvement

    Parameters
    ----------
    comparison:
        Output of ``raw_unmixed_reference_comparison``.
    cluster_matches:
        Second return value of ``raw_unmixed_reference_comparison``.
    gene_labels:
        Optional dict mapping gene name → display label.
    output_dir:
        Directory to write figures into (created if absent).
    dpi:
        Resolution for saved figures.
    formats:
        One or more file extensions, e.g. ``("png",)`` or ``("png", "pdf")``.
    expression_vmax:
        ``vmax`` used for the row-norm expression heatmap.  Set ``None`` for
        auto-scaling.
    """
    from pathlib import Path as _Path

    out = _Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _kw = dict(save=True, output_dir=out, formats=formats, dpi=dpi, show=False)

    plot_supertype_match_summary(
        cluster_matches,
        filename="01_supertype_match_summary", **_kw,
    )
    plot_improvement_heatmap(
        comparison,
        cluster_matches=cluster_matches,
        row_sort="natural",
        col_sort="mean_improvement",
        filename="02_improvement_heatmap", **_kw,
    )
    plot_cluster_reference_expression(
        comparison=comparison,
        cluster_matches=cluster_matches,
        gene_labels=gene_labels,
        use_unmixed=True,
        normalize="row",
        vmax=expression_vmax,
        filename="03_expression_heatmap_row_norm", **_kw,
    )
    plot_cluster_reference_expression(
        comparison=comparison,
        cluster_matches=cluster_matches,
        gene_labels=gene_labels,
        use_unmixed=True,
        normalize="col",
        filename="04_expression_heatmap_col_norm", **_kw,
    )
    plot_cluster_reference_expression(
        comparison=comparison,
        cluster_matches=cluster_matches,
        gene_labels=gene_labels,
        use_unmixed=True,
        log_transform=True,
        normalize=None,
        cmap_diff="RdBu_r",
        filename="05_expression_heatmap_no_norm_log2", **_kw,
    )
    plot_gene_expression_bias(
        comparison=comparison,
        gene_labels=gene_labels,
        use_unmixed=True,
        normalize=None,
        filename="06_gene_bias_log2_oe", **_kw,
    )
    plot_gene_expression_bias(
        comparison=comparison,
        gene_labels=gene_labels,
        use_unmixed=True,
        normalize="row",
        filename="07_gene_bias_row_norm_diff", **_kw,
    )
    plot_reference_scatter(
        comparison,
        annotate_genes=[],
        use_unmixed=False,
        filename="08_reference_scatter_raw", **_kw,
    )
    plot_reference_scatter(
        comparison,
        annotate_genes=[],
        use_unmixed=True,
        filename="09_reference_scatter_unmixed", **_kw,
    )
    plot_error_improvement(
        comparison,
        filename="10_error_improvement", **_kw,
    )

    import os as _os
    saved = sorted(_os.listdir(out))
    print(f"Saved {len(saved)} figures to: {out}")
    for f in saved:
        print(f"  {f}")


def spot_feature_improvement_analysis(
    comparison: pd.DataFrame,
    raw_spots: pd.DataFrame,
    cell_meta: pd.DataFrame,
    group_col: str,
    raw_chan_col: str = "mixed_gene",
    round_col: str = "round",
    chan_col: str = "chan",
    feature_cols: Optional[Sequence[str]] = None,
    top_n_scatter: int = 5,
    figsize: Optional[Tuple] = None,
) -> pd.DataFrame:
    """
    Discover which spot-table or expression-context features correlate with
    unmixing improvement (or lack thereof).

    Workflow
    --------
    1. Aggregate numeric spot-table columns to per-(cluster, gene) means.
    2. Build same-round neighbour expression features from the comparison
       table: for each (cluster, gene), compute the mean raw CP10K of other
       genes measured in the same imaging round.  Tests the hypothesis that
       dominant co-round neighbours drive poor unmixing.
    3. Include raw_cp10k, reference_cp10k, and raw_log2_oe from comparison
       as additional context features.
    4. Compute Pearson r between every feature and ``improvement``.
    5. Plot a ranked bar chart of correlations (red = hurt, blue = help).
    6. Scatter panels for the top-|r| features, with the worst gene-cluster
       points annotated.

    Parameters
    ----------
    comparison:
        Output of ``raw_unmixed_reference_comparison``.
    raw_spots:
        Raw spot table with cell_id, gene/channel, round, chan, and any
        numeric quality/intensity columns.
    cell_meta:
        Cell metadata indexed by cell_id, with *group_col*.
    group_col:
        Column in cell_meta defining groups (e.g. "cluster_label").
    raw_chan_col:
        Column in raw_spots giving the gene name (e.g. "mixed_gene").
    round_col, chan_col:
        Round and channel columns in raw_spots; used for neighbour construction.
    feature_cols:
        Numeric spot columns to aggregate. ``None`` → auto-detect all numeric
        columns except coordinates, identifiers, and gene/round/chan columns.
    top_n_scatter:
        Number of scatter panels to draw for the highest-|r| features.
    figsize:
        Override default figure sizes.

    Returns
    -------
    corr_df : pd.DataFrame
        Correlation table with columns: feature, pearson_r, p_value, n.
        Sorted by pearson_r ascending (most negative first).
    """
    try:
        from scipy import stats as _stats
    except ImportError:
        raise ImportError("scipy is required.  pip install scipy")

    cell_meta = prepare_cell_meta(cell_meta)

    # ── 1. Aggregate spot-table numeric features per (group, gene) ───────────
    spots = raw_spots.copy()
    spots = spots[spots["cell_id"] > 0]
    group_map = cell_meta[group_col]
    spots["_group"] = spots["cell_id"].map(group_map)
    spots = spots.dropna(subset=["_group"])

    _exclude = {
        "cell_id", "x", "y", "z", "_group",
        raw_chan_col, round_col, chan_col,
        "valid_spot", "mixed_gene", "unmixed_gene", "round", "chan",
    }
    if feature_cols is None:
        feature_cols = [
            c for c in spots.select_dtypes(include="number").columns
            if c not in _exclude
        ]

    if feature_cols:
        spot_agg = (
            spots.groupby(["_group", raw_chan_col])[list(feature_cols)]
            .mean()
            .reset_index()
            .rename(columns={"_group": "group", raw_chan_col: "gene"})
        )
    else:
        spot_agg = pd.DataFrame(columns=["group", "gene"])

    # ── 2. Same-round neighbour expression features ──────────────────────────
    # Core hypothesis: if co-round neighbours are highly expressed, the
    # unmixer has a harder job → lower improvement.
    gene_round_chan = (
        raw_spots[[raw_chan_col, round_col, chan_col]]
        .drop_duplicates(subset=[raw_chan_col])
        .set_index(raw_chan_col)
    )
    comp_pivot = comparison.pivot_table(
        index="group", columns="gene", values="raw_cp10k", aggfunc="mean"
    )

    neighbor_records = []
    for gene in comp_pivot.columns:
        if gene not in gene_round_chan.index:
            continue
        g_round = gene_round_chan.loc[gene, round_col]
        same_round_genes = [
            g for g in comp_pivot.columns
            if g != gene
            and g in gene_round_chan.index
            and gene_round_chan.loc[g, round_col] == g_round
        ]
        if not same_round_genes:
            continue
        for group in comp_pivot.index:
            neighbor_mean = float(comp_pivot.loc[group, same_round_genes].mean())
            self_expr = float(comp_pivot.loc[group, gene])
            neighbor_records.append({
                "group": group,
                "gene": gene,
                "neighbor_mean_cp10k": neighbor_mean,
                # ratio < 1 → gene is weaker than its co-round neighbours
                "self_to_neighbor_ratio": (self_expr + 1.0) / (neighbor_mean + 1.0),
            })

    neighbor_df = (
        pd.DataFrame(neighbor_records)
        if neighbor_records
        else pd.DataFrame(columns=["group", "gene",
                                   "neighbor_mean_cp10k",
                                   "self_to_neighbor_ratio"])
    )

    # ── 3. Merge everything with improvement ─────────────────────────────────
    base_cols = ["group", "gene", "raw_cp10k", "reference_cp10k",
                 "raw_log2_oe", "unmixed_log2_oe", "improvement"]
    merged = comparison[base_cols].copy()
    if not spot_agg.empty:
        merged = merged.merge(spot_agg, on=["group", "gene"], how="left")
    if not neighbor_df.empty:
        merged = merged.merge(neighbor_df, on=["group", "gene"], how="left")

    # ── 4. Pearson r for every feature vs improvement ────────────────────────
    exclude_corr = {"group", "gene", "improvement"}
    feat_names = [c for c in merged.columns if c not in exclude_corr]

    rows = []
    for feat in feat_names:
        sub = merged[["improvement", feat]].dropna()
        if len(sub) < 5 or sub[feat].std() == 0:
            continue
        r, p = _stats.pearsonr(sub["improvement"], sub[feat])
        rows.append({"feature": feat, "pearson_r": r, "p_value": p, "n": len(sub)})

    corr_df = pd.DataFrame(rows).sort_values("pearson_r").reset_index(drop=True)

    if corr_df.empty:
        print("[spot_feature_improvement_analysis] No features with sufficient data.")
        return corr_df

    # ── 5. Bar chart of all correlations ─────────────────────────────────────
    n_feats = len(corr_df)
    fig_w = max(5, 0.6 * n_feats)
    fig, ax = plt.subplots(figsize=figsize or (fig_w, 4))
    bar_colors = ["#F44336" if r < 0 else "#2196F3" for r in corr_df["pearson_r"]]
    ax.bar(corr_df["feature"], corr_df["pearson_r"], color=bar_colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticklabels(corr_df["feature"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r with improvement")
    ax.set_title(
        "Feature correlation with unmixing improvement\n"
        "(red = associated with lower improvement, blue = higher)"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()

    # ── 6. Scatter panels for top-|r| features ───────────────────────────────
    top = corr_df.loc[
        corr_df["pearson_r"].abs().sort_values(ascending=False).index
    ].head(min(top_n_scatter, n_feats))

    ncols = len(top)
    if ncols > 0:
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
        axes = axes[0]
        for ax, (_, row) in zip(axes, top.iterrows()):
            feat = row["feature"]
            sub = merged[["improvement", feat, "gene", "group"]].dropna()
            ax.scatter(sub[feat], sub["improvement"], alpha=0.35, s=18)
            # annotate the 5 worst-improving points
            for _, pt in sub.nsmallest(5, "improvement").iterrows():
                ax.annotate(
                    f"{pt['gene']}\n{pt['group']}",
                    (pt[feat], pt["improvement"]),
                    fontsize=5, alpha=0.8,
                    xytext=(4, 0), textcoords="offset points",
                )
            ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
            ax.set_xlabel(feat, fontsize=8)
            if ax is axes[0]:
                ax.set_ylabel("Improvement")
            ax.set_title(f"r = {row['pearson_r']:.2f}", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        plt.suptitle("Top correlates with unmixing improvement", fontsize=10)
        plt.tight_layout()
        plt.show()

    print("\nCorrelation summary:")
    print(corr_df.to_string(index=False))
    return corr_df


# -----------------------------------------------------------------------------
# Convenience wrapper
# -----------------------------------------------------------------------------


def run_basic_validation(
    raw_spots: pd.DataFrame,
    unmixed_spots: pd.DataFrame,
    cell_meta: pd.DataFrame,
    group_col: str,
    ref_counts: pd.DataFrame,
    ref_labels: pd.Series,
    genes: Optional[Sequence[str]] = None,
    chan_to_gene: Optional[Dict] = None,
    raw_chan_col: str = "chan",
    unmixed_chan_col: Optional[str] = None,
    matching_source: str = "unmixed",
    pseudocount: float = 0.1,
    scale: float = 10_000,
) -> dict:
    """
    Run the core validation outputs in one call.

    Returns
    -------
    Dictionary containing:
        retention
        comparison
        cluster_matches
    """
    retention = gene_retention_by_group(
        raw_spots=raw_spots,
        unmixed_spots=unmixed_spots,
        cell_meta=cell_meta,
        group_col=group_col,
        genes=genes,
        chan_to_gene=chan_to_gene,
        raw_chan_col=raw_chan_col,
        unmixed_chan_col=unmixed_chan_col,
    )

    comparison, cluster_matches = raw_unmixed_reference_comparison(
        raw_spots=raw_spots,
        unmixed_spots=unmixed_spots,
        cell_meta=cell_meta,
        group_col=group_col,
        ref_counts=ref_counts,
        ref_labels=ref_labels,
        genes=genes,
        chan_to_gene=chan_to_gene,
        raw_chan_col=raw_chan_col,
        unmixed_chan_col=unmixed_chan_col,
        matching_source=matching_source,
        pseudocount=pseudocount,
        scale=scale,
    )

    return {
        "retention": retention,
        "comparison": comparison,
        "cluster_matches": cluster_matches,
    }


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
#
# raw_spots = pd.read_csv("raw_spots.csv")
# unmixed_spots = pd.read_csv("unmixed_spots.csv")
# cell_meta = pd.read_csv("cell_meta.csv")
# ref_counts = pd.read_csv("reference_counts.csv", index_col=0)
# ref_labels = pd.read_csv("reference_labels.csv", index_col=0)["subtype"]
#
# # If chan already stores gene names:
# chan_to_gene = None
#
# # If chan is a numeric/channel identifier:
# # chan_to_gene = {
# #     0: "Vip",
# #     1: "Cck",
# #     2: "Gad2",
# #     ...
# # }
#
# results = run_basic_validation(
#     raw_spots=raw_spots,
#     unmixed_spots=unmixed_spots,
#     cell_meta=cell_meta,
#     group_col="cluster",
#     ref_counts=ref_counts,
#     ref_labels=ref_labels,
#     chan_to_gene=chan_to_gene,
# )
#
# retention = results["retention"]
# comparison = results["comparison"]
# cluster_matches = results["cluster_matches"]
#
# # Look for Cck over-removal:
# print(retention.query("gene == 'Cck'").sort_values("retention"))
# print(comparison.query("gene == 'Cck'").sort_values("unmixed_log2_oe"))
#
# # Plot:
# plot_retention_by_group(retention, gene="Cck")
# plot_reference_scatter(comparison, annotate_genes=["Cck", "Vip", "Gad2", "Slc17a7"])
# plot_gene_improvement(comparison, gene="Cck")


# -----------------------------------------------------------------------------
# ABC Atlas reference loading
# -----------------------------------------------------------------------------


def load_abc_merfish_reference(
    abc_cache_dir,
    genes: Sequence[str],
    cell_index=None,
    ref_classes: Optional[Sequence[str]] = None,
    label_level: str = "subclass",
    min_label_cells: int = 5,
    save_dir=None,
    abc_cache=None,
    ref_cell_meta: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and filter a reference expression matrix from the ABC Atlas MERFISH dataset.

    This function wraps the multi-step loading workflow into a single call:

    1. Initialise ``AbcProjectCache`` from *abc_cache_dir*.
    2. Load MERFISH cell metadata (cluster / subclass / class / supertype columns).
    3. Open the log2 expression h5ad in backed mode and identify genes whose
       ``gene_symbol`` matches any entry in *genes*.
    4. Optionally restrict to a subset of cells given by *cell_index* (a
       pandas Index or any container understood by ``pd.Index.isin``).  Pass
       None to keep all cells.
    5. Extract the expression sub-matrix for the selected cells × panel genes.
    6. Optionally filter cells to those whose ``"class"`` column is in
       *ref_classes*.
    7. Build ``ref_labels`` at *label_level* (``"class"``, ``"subclass"``,
       ``"supertype"``, or ``"cluster"``).
    8. Drop labels whose cell count is below *min_label_cells*.
    9. Optionally save ``ref_counts`` and ``ref_labels`` as CSVs to *save_dir*.

    Parameters
    ----------
    abc_cache_dir:
        Path (str or Path-like) to the ABC Atlas local cache root (passed to
        ``AbcProjectCache.from_cache_dir``).
    genes:
        Iterable of gene symbol strings to include (e.g. your HCR panel).
        Genes not present in the MERFISH panel are silently ignored.
    cell_index:
        Optional index / iterable of cell labels to restrict to (e.g. from a
        region-of-interest CSV).  ``None`` → use all cells.
    ref_classes:
        Optional list of ``"class"`` values to keep (e.g.
        ``["07 CTX-MGE GABA", "06 CTX-CGE GABA"]``).  ``None`` → keep all.
    label_level:
        Column in the cell-metadata table to use as reference labels for
        centroid construction.  One of ``"class"``, ``"subclass"``,
        ``"supertype"``, ``"cluster"``.  Default ``"subclass"``.
    min_label_cells:
        Drop any label whose cell count (after all other filters) is below this
        threshold.  Prevents noisy centroids from sparse groups.  Default 5.
    save_dir:
        Optional path (str or Path-like) of a directory.  If provided the
        function writes two files there:
        ``ref_counts_<label_level>.csv`` and ``ref_labels_<label_level>.csv``.

    Returns
    -------
    ref_counts : pd.DataFrame
        Cells × genes expression matrix (log2, raw values from h5ad), filtered
        and matched to *ref_labels*.
    ref_labels : pd.Series
        Cell labels → reference group name, indexed identically to *ref_counts*.
    """
    if _AbcProjectCache is None:
        raise ImportError(
            "abc_atlas_access is not installed. "
            "Install with:  pip install git+https://github.com/AllenInstitute/abc_atlas_access.git"
        )
    if _anndata is None:
        raise ImportError("anndata is not installed. Install with:  pip install anndata")

    import warnings
    from pathlib import Path as _Path

    abc_cache_dir = _Path(abc_cache_dir)

    # ── 1. Cache ──────────────────────────────────────────────────────────────
    if abc_cache is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            abc_cache = _AbcProjectCache.from_cache_dir(abc_cache_dir)

    # ── 2. Cell metadata ──────────────────────────────────────────────────────
    if ref_cell_meta is None:
        _meta = abc_cache.get_metadata_dataframe(
            directory="MERFISH-C57BL6J-638850",
            file_name="cell_metadata_with_cluster_annotation",
            dtype={"cell_label": str, "neurotransmitter": str},
        )
        _meta.set_index("cell_label", inplace=True)
        ref_cell_meta = _meta

    # ── 3. Expression matrix (backed) ────────────────────────────────────────
    h5ad_path = abc_cache.get_file_path(
        directory="MERFISH-C57BL6J-638850",
        file_name="C57BL6J-638850/log2",
    )
    adata = _anndata.read_h5ad(h5ad_path, backed="r")
    ref_gene_meta = adata.var  # Ensembl index, gene_symbol column

    genes = [g for g in genes if isinstance(g, str)]
    gene_mask = ref_gene_meta["gene_symbol"].isin(genes)
    panel_gene_meta = ref_gene_meta[gene_mask]
    missing = sorted(set(genes) - set(panel_gene_meta["gene_symbol"]))
    if missing:
        print(f"[load_abc_merfish_reference] {len(missing)} genes not in MERFISH panel (ignored): {missing}")

    # ── 4. Cell subset ────────────────────────────────────────────────────────
    if cell_index is not None:
        cell_mask = adata.obs_names.isin(cell_index)
    else:
        cell_mask = np.ones(adata.n_obs, dtype=bool)
    print(f"[load_abc_merfish_reference] cells selected: {cell_mask.sum():,} / {adata.n_obs:,}")

    # ── 5. Extract expression ─────────────────────────────────────────────────
    ref_counts = adata[cell_mask, panel_gene_meta.index].to_df()
    ref_counts.columns = panel_gene_meta["gene_symbol"].values
    adata.file.close()
    del adata

    # ── 6. Class filter ───────────────────────────────────────────────────────
    if ref_classes is not None:
        class_col = ref_cell_meta.loc[ref_counts.index, "class"]
        class_mask = class_col.isin(ref_classes)
        ref_counts = ref_counts.loc[class_mask]
        print(
            f"[load_abc_merfish_reference] class filter → {class_mask.sum():,} cells "
            f"({class_col[class_mask].value_counts().to_dict()})"
        )

    # ── 7. Build ref_labels ───────────────────────────────────────────────────
    if label_level not in ref_cell_meta.columns:
        available = [c for c in ref_cell_meta.columns if c in ("class", "subclass", "supertype", "cluster")]
        raise ValueError(
            f"label_level={label_level!r} not found in cell metadata. "
            f"Available: {available}"
        )
    ref_labels = ref_cell_meta.loc[ref_counts.index, label_level]

    # ── 8. Min-cell filter ────────────────────────────────────────────────────
    label_counts = ref_labels.value_counts()
    keep = label_counts[label_counts >= min_label_cells].index
    dropped = (label_counts < min_label_cells).sum()
    keep_mask = ref_labels.isin(keep)
    ref_counts = ref_counts.loc[keep_mask]
    ref_labels = ref_labels.loc[keep_mask]
    print(
        f"[load_abc_merfish_reference] label_level={label_level!r}  "
        f"kept={keep.size} labels  dropped={dropped} (< {min_label_cells} cells)"
    )
    print(ref_labels.value_counts().to_string())

    # ── 9. Optional save ──────────────────────────────────────────────────────
    if save_dir is not None:
        save_dir = _Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        counts_path = save_dir / f"ref_counts_{label_level}.csv"
        labels_path = save_dir / f"ref_labels_{label_level}.csv"
        ref_counts.to_csv(counts_path)
        ref_labels.to_csv(labels_path, header=True)
        print(f"[load_abc_merfish_reference] saved → {counts_path}")
        print(f"[load_abc_merfish_reference] saved → {labels_path}")

    return ref_counts, ref_labels
