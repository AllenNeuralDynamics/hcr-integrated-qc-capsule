"""Mouse-level QC metrics: collection and IO.

Per-mouse JSON files are stored on S3 under::

    s3://aind-scratch-data/ctl/hcr/qc/_metrics/{mouse_id}_metrics.json

The ``_metrics`` folder is prefixed with ``_`` so the viewer catalog scanner
(which skips folders starting with ``_``) ignores it.

Structure::

    {
      "mouse_id": "755252",
      "computed_at": "2026-04-03T12:00:00+00:00",
      "scalar": {
        "n_spots_total": 987654
      },
      "per_gene": {
        "Slc17a7": {
          "mean_intensity":   1234.5,
          "median_intensity": 1100.2,
          "std_intensity":     312.7,
          "n_spots":           5678
        },
        ...
      }
    }

Adding new metrics
------------------
* **Scalar metric**: compute it and do ``metrics["scalar"]["my_key"] = value``, then
  add ``"my_key"`` to the relevant ``_*_SCALAR_KEYS`` set so the cache-check knows
  to look for it.
* **New spots group key**: add it to ``_SPOTS_PER_GENE_KEYS`` and compute it in
  ``collect_spots_metrics``.  The missing key will trigger a reload on the next run.
"""

import json
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError
import pandas as pd

METRICS_S3_BUCKET: str = "aind-scratch-data"
METRICS_S3_PREFIX: str = "ctl/hcr/qc/_metrics"


def _s3_key(mouse_id: str) -> str:
    return f"{METRICS_S3_PREFIX}/{mouse_id}_metrics.json"

# ---------------------------------------------------------------------------
# Completeness sentinels — extend these when adding new metrics
# ---------------------------------------------------------------------------

# Scalar keys that are produced by the spots-table loading step.
_SPOTS_SCALAR_KEYS: frozenset[str] = frozenset({"n_spots_total"})

# Per-gene keys produced by the spots-table loading step.
_SPOTS_PER_GENE_KEYS: frozenset[str] = frozenset(
    {"mean_intensity", "median_intensity", "std_intensity", "n_spots"}
)

# Per-gene keys produced by the cell-by-gene (CXG) loading step.
_CXG_PER_GENE_KEYS: frozenset[str] = frozenset(
    {"mean_normalized_counts", "median_normalized_counts"}
)

# Scaling factor for CPM-style normalization (spots per cell × SCALE)
_CPM_SCALE: float = 1_000.0


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_mouse_metrics(mouse_id: str) -> dict:
    """Load existing per-mouse metrics JSON from S3, or return an empty skeleton."""
    s3 = boto3.client("s3")
    try:
        resp = s3.get_object(Bucket=METRICS_S3_BUCKET, Key=_s3_key(mouse_id))
        return json.loads(resp["Body"].read())
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return {"mouse_id": mouse_id, "scalar": {}, "per_gene": {}}
        raise


def save_mouse_metrics(mouse_id: str, metrics: dict) -> None:
    """Stamp the timestamp and upload the JSON to S3."""
    metrics["computed_at"] = datetime.now(timezone.utc).isoformat()
    key = _s3_key(mouse_id)
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=METRICS_S3_BUCKET,
        Key=key,
        Body=json.dumps(metrics, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"  [metrics] Saved -> s3://{METRICS_S3_BUCKET}/{key}")


# ---------------------------------------------------------------------------
# Completeness checks (cache guards)
# ---------------------------------------------------------------------------

def _spots_metrics_complete(metrics: dict) -> bool:
    """Return True if all spots metric keys are already present in *metrics*."""
    scalar = metrics.get("scalar", {})
    if not _SPOTS_SCALAR_KEYS.issubset(scalar):
        return False
    per_gene = metrics.get("per_gene", {})
    if not per_gene:
        return False
    return all(
        _SPOTS_PER_GENE_KEYS.issubset(gene_vals)
        for gene_vals in per_gene.values()
    )


def _cxg_metrics_complete(metrics: dict) -> bool:
    """Return True if all CXG metric keys are already present in *metrics*."""
    per_gene = metrics.get("per_gene", {})
    if not per_gene:
        return False
    return all(
        _CXG_PER_GENE_KEYS.issubset(gene_vals)
        for gene_vals in per_gene.values()
    )


# ---------------------------------------------------------------------------
# Metric collectors
# ---------------------------------------------------------------------------

def collect_spots_metrics(
    pw_dataset,
    metrics: dict,
    overwrite: bool = False,
) -> bool:
    """Compute per-gene spot intensity metrics and merge into *metrics*.

    Metrics computed
    ----------------
    Scalar:
      * ``n_spots_total`` — total spot count across all rounds and genes

    Per gene (grouped by ``unmixed_gene``):
      * ``mean_intensity``   — mean of each spot's own-channel intensity
      * ``median_intensity`` — median of each spot's own-channel intensity
      * ``std_intensity``    — standard deviation of each spot's own-channel intensity
      * ``n_spots``          — number of spots assigned to that gene

    Intensity is derived from each spot's unmixed channel (``_intensity``
    column added by ``add_unmixed_channel_intensity``).  No intensity
    threshold is applied so that ``n_spots`` reflects all assigned spots.

    Parameters
    ----------
    pw_dataset:
        A ``PairwiseUnmixingDataset`` instance.
    metrics:
        The dict returned by ``load_mouse_metrics`` — updated in-place.
    overwrite:
        If False and all spots metric keys are already present, skip.

    Returns
    -------
    bool
        True if metrics were (re)computed, False if skipped.
    """
    from aind_hcr_qc.viz.intergrated_datasets import add_unmixed_channel_intensity

    if not overwrite and _spots_metrics_complete(metrics):
        print("  [metrics] Spots metrics already complete — skipping.")
        return False

    print("  [metrics] Loading spots table...")
    spots_df = pw_dataset.load_all_rounds_spots_mp(
        table_type="unmixed_spots", remove_fg_bg_cols=False
    )
    # Add _intensity = each spot's own-channel value; no threshold so counts are unfiltered.
    spots_df = add_unmixed_channel_intensity(
        spots_df, chan_col="unmixed_chan", intensity_threshold=None
    )

    grouped = spots_df.groupby("unmixed_gene")["_intensity"]
    per_gene: dict[str, dict] = {}
    for gene, grp in grouped:
        per_gene[str(gene)] = {
            "mean_intensity":   round(float(grp.mean()), 4),
            "median_intensity": round(float(grp.median()), 4),
            "std_intensity":    round(float(grp.std()), 4),
            "n_spots":          int(len(grp)),
        }

    metrics.setdefault("scalar", {})["n_spots_total"] = int(len(spots_df))
    # Merge so any future per_gene keys added by other collectors are preserved.
    existing_per_gene = metrics.get("per_gene", {})
    for gene, vals in per_gene.items():
        existing_per_gene.setdefault(gene, {}).update(vals)
    metrics["per_gene"] = existing_per_gene

    del spots_df
    print(f"  [metrics] Spots metrics computed for {len(per_gene)} gene(s).")
    return True


def collect_cxg_metrics(
    pw_dataset,
    metrics: dict,
    overwrite: bool = False,
) -> bool:
    """Compute per-gene normalized-count metrics from the cell-by-gene table.

    Normalization
    -------------
    For each cell divide its spot counts by the cell's total spot count across
    all genes, then multiply by ``_CPM_SCALE`` (1 000).  This is analogous to
    CPM library-size normalization and removes the confound of overall
    detection efficiency varying between cells.

    Metrics computed
    ----------------
    Per gene:
      * ``mean_normalized_counts``   — mean of normalized value across all cells
      * ``median_normalized_counts`` — median of normalized value across all cells

    Scalar:
      * ``n_cells`` — number of cells in the CXG table

    Parameters
    ----------
    pw_dataset:
        A ``PairwiseUnmixingDataset`` instance.
    metrics:
        The dict returned by ``load_mouse_metrics`` — updated in-place.
    overwrite:
        If False and all CXG metric keys are already present, skip.

    Returns
    -------
    bool
        True if metrics were (re)computed, False if skipped.
    """
    import numpy as np

    if not overwrite and _cxg_metrics_complete(metrics):
        print("  [metrics] CXG metrics already complete — skipping.")
        return False

    print("  [metrics] Loading cell-by-gene table...")
    cxg = pw_dataset.load_aggregated_cxg(unmixed=True)
    # Columns are in ``{round}-{chan}-{gene}`` form, e.g. "R3-561-Crh".
    # Strip the round/channel prefix and group (sum) across rounds so each
    # gene is represented by a single count per cell.
    all_cols = [c for c in cxg.columns if c != "cell_id"]

    def _gene_from_col(col: str) -> str:
        parts = col.split("-", 2)
        return parts[2] if len(parts) == 3 else col

    # Build gene-level counts by summing all round-chan columns for the same gene
    gene_map: dict[str, list] = {}
    for col in all_cols:
        gene_map.setdefault(_gene_from_col(col), []).append(col)

    counts_by_gene = pd.DataFrame(
        {gene: cxg[cols].sum(axis=1) for gene, cols in gene_map.items()}
    ).fillna(0.0)

    # CPM-style: normalize each row by its total spot count, scale by _CPM_SCALE
    row_totals = counts_by_gene.sum(axis=1).replace(0, np.nan)
    normalized = counts_by_gene.div(row_totals, axis=0) * _CPM_SCALE

    per_gene: dict[str, dict] = {}
    for gene in counts_by_gene.columns:
        col = normalized[gene].dropna()
        per_gene[str(gene)] = {
            "mean_normalized_counts":   round(float(col.mean()), 4),
            "median_normalized_counts": round(float(col.median()), 4),
        }

    metrics.setdefault("scalar", {})["n_cells"] = int(len(cxg))
    existing_per_gene = metrics.get("per_gene", {})
    for gene, vals in per_gene.items():
        existing_per_gene.setdefault(gene, {}).update(vals)
    metrics["per_gene"] = existing_per_gene

    del cxg, counts_by_gene, normalized
    print(f"  [metrics] CXG metrics computed for {len(per_gene)} gene(s).")
    return True


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def _is_round_chan_gene(key: str) -> bool:
    """Return True if *key* looks like a round-chan-gene label, e.g. 'R3-561-Crh'."""
    parts = key.split("-", 2)
    return (
        len(parts) == 3
        and parts[0].startswith("R")
        and parts[0][1:].isdigit()
    )


def clean_stale_cxg_keys(
    mouse_id: str,
    dry_run: bool = False,
) -> int:
    """Remove round-chan-gene artefact keys from a mouse's per_gene metrics.

    When ``collect_cxg_metrics`` was first written it stored keys in the
    ``{round}-{chan}-{gene}`` form (e.g. ``"R3-561-Crh"``) instead of the
    plain gene name.  This function loads the metrics JSON from S3, drops any
    ``per_gene`` key that matches that pattern, and re-uploads the cleaned
    JSON.

    Parameters
    ----------
    mouse_id:
        Mouse to clean.
    dry_run:
        If True, print what would be removed but do not write to S3.

    Returns
    -------
    int
        Number of stale keys removed (0 if already clean).
    """
    metrics = load_mouse_metrics(mouse_id)
    per_gene = metrics.get("per_gene", {})
    stale = [k for k in per_gene if _is_round_chan_gene(k)]
    if not stale:
        print(f"  [clean] {mouse_id}: no stale keys found.")
        return 0
    print(f"  [clean] {mouse_id}: removing {len(stale)} stale key(s): {stale[:5]}{'...' if len(stale) > 5 else ''}")
    if not dry_run:
        for k in stale:
            del per_gene[k]
        metrics["per_gene"] = per_gene
        save_mouse_metrics(mouse_id, metrics)
    return len(stale)


# ---------------------------------------------------------------------------
# CLI — run to clean stale round-chan-gene keys from all mice on S3
#
#   python metrics.py --clean [--dry-run]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean stale round-chan-gene keys from metrics JSONs on S3."
    )
    parser.add_argument(
        "--clean", action="store_true", required=True,
        help="Remove round-chan-gene artefact keys from all mouse JSONs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print what would be removed without writing to S3.",
    )
    args = parser.parse_args()

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    mouse_ids = []
    for page in paginator.paginate(Bucket=METRICS_S3_BUCKET, Prefix=METRICS_S3_PREFIX + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("_metrics.json"):
                stem = key.split("/")[-1].replace("_metrics.json", "")
                mouse_ids.append(stem)

    print(f"Found {len(mouse_ids)} mouse metric file(s) on S3.")
    total = 0
    for mid in sorted(mouse_ids):
        total += clean_stale_cxg_keys(mid, dry_run=args.dry_run)

    action = "Would remove" if args.dry_run else "Removed"
    print(f"\n{action} {total} stale key(s) across {len(mouse_ids)} mouse file(s).")
