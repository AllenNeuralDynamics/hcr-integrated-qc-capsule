"""Mouse-level QC metrics: collection and IO.

Per-mouse JSON files are written to METRICS_DIR:

    scratch/metrics/{mouse_id}_metrics.json

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
from pathlib import Path

import pandas as pd

METRICS_DIR = Path("/root/capsule/scratch/metrics")

# ---------------------------------------------------------------------------
# Completeness sentinels — extend these when adding new metrics
# ---------------------------------------------------------------------------

# Scalar keys that are produced by the spots-table loading step.
_SPOTS_SCALAR_KEYS: frozenset[str] = frozenset({"n_spots_total"})

# Per-gene keys produced by the spots-table loading step.
_SPOTS_PER_GENE_KEYS: frozenset[str] = frozenset(
    {"mean_intensity", "median_intensity", "std_intensity", "n_spots"}
)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_mouse_metrics(mouse_id: str) -> dict:
    """Load existing per-mouse metrics JSON, or return an empty skeleton."""
    path = METRICS_DIR / f"{mouse_id}_metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"mouse_id": mouse_id, "scalar": {}, "per_gene": {}}


def save_mouse_metrics(mouse_id: str, metrics: dict) -> None:
    """Stamp the timestamp, write the JSON, and print the output path."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics["computed_at"] = datetime.now(timezone.utc).isoformat()
    path = METRICS_DIR / f"{mouse_id}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    print(f"  [metrics] Saved -> {path}")


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
