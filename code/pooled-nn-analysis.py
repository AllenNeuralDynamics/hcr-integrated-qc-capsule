import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from pathlib import Path
import numpy as np
import pandas as pd

from aind_hcr_data_loader.codeocean_utils import (
    MouseRecord,
    attach_mouse_record_to_workstation,
    print_attach_results,
)
from aind_hcr_data_loader.hcr_dataset import create_hcr_dataset_from_schema
from aind_hcr_data_loader.pairwise_dataset import create_pairwise_unmixing_dataset

import aind_hcr_qc.viz as viz
from aind_hcr_qc.viz.intergrated_datasets import plot_intensity_violins
from aind_hcr_qc.viz.single_cell_unmixing import (
    plot_spot_projection,
    plot_spot_measure_distributions,
    plot_cell_qc,
    plot_spot_nn_distances,
    plot_adjacent_channel_scatter,
    collect_nn_dists_all_cells,
    plot_nn_euclidean_dists,
    plot_nn_grid_pooled,
    plot_nn_euclidean_dists_pooled,
)

import aind_hcr_qc.viz.single_cell_unmixing as scu

from aind_hcr_qc.viz.spot_detection import (
    annotate_spots_df,
    plot_removal_metric_distributions,
)
from aind_hcr_qc.constants import Z1_CHANNEL_CMAP_VIBRANT



DATA_DIR = Path('/root/capsule/data')
BUCKET_NAME = "aind-open-data"

CHAN_ORDER  = ["488", "514", "561", "594", "638"]
CHAN_COLORS = {k: v for k, v in Z1_CHANNEL_CMAP_VIBRANT.items() if k in CHAN_ORDER}
VOXEL_SIZE  = {"x": 0.24, "y": 0.24, "z": 1.0}  # µm/px  (z, y, x = 1, 0.24, 0.24)

# Crosstalk pairs to investigate: mouse_id -> round_key -> focal_channel -> [partner_channels]
CROSSTALK_TARGETS = {
    "767022": {
        "R2": {
            "514": ["488"],
            "561": ["594"],
            "594": ["561"],
            "638": ["561"],
        },
    },
}


def get_data(
    mouse_id: str,
    data_dir: Path = DATA_DIR,
    bucket_name: str = BUCKET_NAME,
) -> tuple:
    catalog_path = Path(f"/src/ophys-mfish-dataset-catalog/mice/{mouse_id}.json")

    # ── attach & load ────────────────────────────────────────────────────────────
    record  = MouseRecord.from_json_file(catalog_path)
    results = attach_mouse_record_to_workstation(record)
    print_attach_results(results)

    dataset = create_hcr_dataset_from_schema(catalog_path, data_dir)
    dataset.summary()

    # ── pairwise unmixing (optional) ─────────────────────────────────────────────
    # The pairwise asset name lives in derived_assets["pairwise_unmixing"] when present.
    pairwise_asset_name = record.derived_assets.get("pairwise_unmixing")

    if pairwise_asset_name is not None:
        pairwise_asset_path = data_dir / pairwise_asset_name
        # Some pipeline outputs nest data under a "pairwise_unmixing" subfolder
        if (pairwise_asset_path / "pairwise_unmixing").exists():
            pairwise_asset_path = pairwise_asset_path / "pairwise_unmixing"
        pw_ds = create_pairwise_unmixing_dataset(
            mouse_id=mouse_id,
            pairwise_asset_path=pairwise_asset_path,
            source_dataset=dataset,
        )
        pw_ds.summary()
    else:
        print("No pairwise_unmixing asset found in catalog record — skipping.")
        pw_ds = None

    return dataset, pw_ds


def get_spots(pw_ds, round_key,mixed=True,unmixed=False):
    """option for mixed, unmixed one of the other"""
    if mixed and unmixed:
        raise ValueError("Cannot specify both mixed and unmixed as True.")
    elif not mixed and not unmixed:
        raise ValueError("Must specify either mixed or unmixed as True.")
    
    if mixed:
        spots_df = pw_ds.get_spots_df(round_key=round_key, mixed=True)
    else:
        spots_df = pw_ds.get_spots_df(round_key=round_key, mixed=False)
    
    return annotate_spots_df(spots_df)


def get_adjacent_channel_pairs(chan_order: list[str]) -> list[tuple[str, str]]:
    """Return (source, neighbor) pairs for spectrally adjacent channels."""
    return [(chan_order[i], chan_order[i + 1]) for i in range(len(chan_order) - 1)]


def get_nonadjacent_channel_pairs(chan_order: list[str]) -> list[tuple[str, str]]:
    """Return (source, neighbor) pairs skipping ≥ 2 channels (gap ≥ 3 steps).
    e.g. 488→594, 488→638, 514→638
    """
    return [
        (chan_order[i], chan_order[j])
        for i in range(len(chan_order))
        for j in range(i + 3, len(chan_order))
    ]


def get_high_signal_cells(
    pw_ds,
    round_key: str,
    focal_chan: str,
    percentile: float = 75.0,
    min_spots: int = 1,
    unmixed: bool = True,
) -> pd.Index:
    """Return cell IDs whose spot count in (round_key, focal_chan) exceeds
    the given percentile threshold (and a hard minimum).

    Uses the aggregated cell-x-gene table from the pairwise asset.
    Columns in that table are named '{round_key}-{chan}-{gene}', so we
    sum all columns matching the requested round and channel.
    """
    cxg = pw_ds.load_aggregated_cxg(unmixed=unmixed)
    prefix = f"{round_key}-{focal_chan}-"
    focal_cols = [c for c in cxg.columns if c.startswith(prefix)]
    if not focal_cols:
        raise ValueError(
            f"No columns found for round={round_key!r}, channel={focal_chan!r}. "
            f"Available prefixes: {sorted({c.rsplit('-', 1)[0] for c in cxg.columns})}"
        )
    spot_counts = cxg[focal_cols].sum(axis=1)
    threshold = max(np.percentile(spot_counts, percentile), min_spots)
    return spot_counts[spot_counts >= threshold].index


def select_cells_for_analysis(
    pw_ds,
    spots_df: pd.DataFrame,
    round_key: str,
    focal_chans: list[str],
    percentile: float = 75.0,
    max_cells: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[pd.Index, pd.DataFrame]:
    """Union high-signal cells across focal channels, sample down to max_cells,
    and return the filtered spots DataFrame.
    """
    if rng is None:
        rng = np.random.default_rng()
    cell_ids: set = set()
    for ch in focal_chans:
        try:
            ids = get_high_signal_cells(pw_ds, round_key, ch, percentile=percentile)
            cell_ids.update(ids.tolist())
        except ValueError:
            pass  # channel not present in this round
    cell_ids = pd.Index(sorted(cell_ids))
    if len(cell_ids) > max_cells:
        cell_ids = pd.Index(rng.choice(cell_ids, size=max_cells, replace=False))
    filtered = spots_df[spots_df["cell_id"].isin(cell_ids)]
    print(f"  Selected {len(cell_ids)} high-signal cells, {len(filtered):,} spots")
    return cell_ids, filtered


def shuffle_spots_within_cells(
    spots_df: pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Randomly permute x/y/z positions within each cell.

    Preserves per-cell spot counts and marginal coordinate distributions
    but destroys inter-channel spatial co-localization — null control for
    co-proximity driven by real biology vs. chance.
    """
    if rng is None:
        rng = np.random.default_rng()
    shuffled = spots_df.copy()
    for _, idx in spots_df.groupby("cell_id").groups.items():
        for col in ("x", "y", "z"):
            vals = shuffled.loc[idx, col].values.copy()
            rng.shuffle(vals)
            shuffled.loc[idx, col] = vals
    return shuffled


def plot_nn_dist_comparison(
    dists_adj: dict,
    dists_nonadj: dict,
    dists_shuf: dict,
    output_path: Path,
    title: str = "",
    xlim: float = 15.0,
) -> None:
    """2-row figure comparing NN distance distributions.

    Row 1 — ECDF: all three conditions overlaid on one axes for direct comparison.
    Row 2 — Count histograms: one panel per condition, shared x from -1 to xlim.
    """
    def _pool_d3d(dists: dict) -> np.ndarray:
        arrays = [
            v["d3d"] for v in dists.values()
            if v is not None and len(v.get("d3d", [])) > 0
        ]
        return np.concatenate(arrays) if arrays else np.array([])

    conditions = [
        ("Adjacent",       dists_adj,    "steelblue"),
        ("Non-adjacent",   dists_nonadj, "darkorange"),
        ("Shuffled (adj)", dists_shuf,   "gray"),
    ]

    fig = plt.figure(figsize=(17, 8))
    gs  = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.38)

    # ── Row 1: ECDF overlay (cols 0-2) + adjacent zoom-in (col 3) ────────────
    ax_ecdf = fig.add_subplot(gs[0, :3])
    for label, dists, color in conditions:
        d3d = _pool_d3d(dists)
        if len(d3d) == 0:
            continue
        x = np.sort(d3d)
        x = x[x <= xlim]
        y = np.arange(1, len(x) + 1) / len(d3d)  # fraction of *all* spots
        ax_ecdf.plot(x, y, color=color, linewidth=1.5,
                     label=f"{label}  (n={len(d3d):,}, med={np.median(d3d):.2f} µm)")
    ax_ecdf.set_xlabel("NN distance (µm)")
    ax_ecdf.set_ylabel("Cumulative fraction")
    ax_ecdf.set_xlim(-1, xlim)
    ax_ecdf.set_title("ECDF — all conditions")
    ax_ecdf.legend(fontsize=8)
    ax_ecdf.grid(True, linewidth=0.4, alpha=0.5)

    # ── Adjacent zoom-in: 0–2.5 µm histogram ─────────────────────────────────
    ax_zoom = fig.add_subplot(gs[0, 3])
    d3d_adj = _pool_d3d(dists_adj)
    zoom_xlim = 2.5
    if len(d3d_adj) > 0:
        med_adj = np.median(d3d_adj)
        n_adj   = len(d3d_adj)
        ax_zoom.hist(d3d_adj, bins=60, density=False, range=(0, zoom_xlim),
                     color="steelblue", alpha=0.75, edgecolor="none")
        ax_zoom.axvline(med_adj, color="k", linestyle="--", linewidth=1.2, alpha=0.5)
        ax_zoom.set_title(f"Adjacent — zoom 0–2.5 µm\nn={n_adj:,}  med={med_adj:.2f} µm",
                          color="k", fontsize=9)
    else:
        ax_zoom.set_title("Adjacent — zoom 0–2.5 µm\n(no data)", color="0.55", fontsize=9)
    ax_zoom.set_xlim(-1, zoom_xlim)
    ax_zoom.set_xlabel("NN distance (µm)")
    ax_zoom.set_ylabel("Count")

    # ── Row 2: count histograms, shared x ────────────────────────────────────
    hist_axes = [fig.add_subplot(gs[1, c]) for c in range(3)]
    for ax, (label, dists, color) in zip(hist_axes, conditions):
        d3d = _pool_d3d(dists)
        if len(d3d) == 0:
            ax.set_title(f"{label}\n(no data)", color="0.55")
            continue
        n   = len(d3d)
        med = np.median(d3d)
        ax.hist(d3d, bins=120, density=False, range=(0, xlim),
                color=color, alpha=0.75, edgecolor="none")
        ax.axvline(med, color="k", linestyle="--", linewidth=1.2)
        ax.set_xlim(-1, xlim)
        ax.set_title(f"{label}\nn={n:,}  med={med:.2f} µm", color="0.55", fontsize=9)
        ax.set_xlabel("NN distance (µm)")

    # only leftmost histogram gets a y-axis label; hide tick labels on others
    hist_axes[0].set_ylabel("Count")
    for ax in hist_axes[1:]:
        ax.tick_params(labelleft=False)

    # share x and y across the three histogram panels
    for ax in hist_axes[1:]:
        ax.sharex(hist_axes[0])
        ax.sharey(hist_axes[0])

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")



def build_dist_dataframe(
    dists: dict,
    condition: str,
    mouse_id: str,
    round_key: str,
) -> pd.DataFrame:
    """Convert a dists dict to a long-form DataFrame (one row per spot distance).

    TODO: persist to parquet for downstream analysis, e.g.:
        df.to_parquet(out / f"{mouse_id}_{round_key}_{condition}_dists.parquet")

    Columns: mouse_id, round_key, condition, ch_a, ch_b, direction, d3d
    """
    rows = []
    for (ch_a, ch_b, direction), vals in dists.items():
        if vals is None or len(vals.get("d3d", [])) == 0:
            continue
        for d in vals["d3d"]:
            rows.append({
                "mouse_id": mouse_id, "round_key": round_key,
                "condition": condition,
                "ch_a": ch_a, "ch_b": ch_b, "direction": direction,
                "d3d": d,
            })
    return pd.DataFrame(rows)


def build_summary_rows(
    dists: dict,
    condition: str,
    mouse_id: str,
    round_key: str,
) -> list[dict]:
    """Compute per-pair summary stats from a dists dict."""
    rows = []
    for (ch_a, ch_b, direction), vals in dists.items():
        if vals is None or len(vals.get("d3d", [])) == 0:
            continue
        d3d = vals["d3d"]
        rows.append({
            "mouse_id":       mouse_id,
            "round_key":      round_key,
            "condition":      condition,
            "ch_a":           ch_a,
            "ch_b":           ch_b,
            "direction":      direction,
            "n":              len(d3d),
            "median_d3d":     float(np.median(d3d)),
            "frac_below_0.5um": float(np.mean(d3d < 0.5)),
            "frac_below_1um": float(np.mean(d3d < 1.0)),
            "frac_below_2um": float(np.mean(d3d < 2.0)),
        })
    return rows


def _merge_dists(target: dict, source: dict) -> None:
    """Append source distance arrays into target accumulator."""
    for key, vals in source.items():
        if vals is None or len(vals.get("d3d", [])) == 0:
            continue
        target.setdefault(key, {"d3d": []})["d3d"].append(vals["d3d"])


def _concat_dists(d: dict) -> dict:
    """Concatenate accumulated d3d lists into a single array per key."""
    return {
        k: {"d3d": np.concatenate([a for a in v["d3d"] if a is not None and len(a)])}
        for k, v in d.items()
    }


def plot_combined_mice_figure(
    dists_adj: dict,
    dists_nonadj: dict,
    dists_shuf: dict,
    summary_df: pd.DataFrame,
    output_path: Path,
    title: str = "",
    xlim: float = 15.0,
) -> None:
    """All-mice combined: standard NN dist panels + per-mouse median strip.

    Layout (2 rows × 4 cols):
      Row 0: ECDF (cols 0-2) | adjacent zoom hist (col 3)
      Row 1: adj hist | nonadj hist | shuf hist | per-mouse median strip (col 3)
    """
    def _pool_d3d(dists: dict) -> np.ndarray:
        arrays = [
            v["d3d"] for v in dists.values()
            if v is not None and len(v.get("d3d", [])) > 0
        ]
        return np.concatenate(arrays) if arrays else np.array([])

    conditions = [
        ("adjacent",     dists_adj,    "steelblue"),
        ("nonadjacent",  dists_nonadj, "darkorange"),
        ("shuffled",     dists_shuf,   "gray"),
    ]
    cond_labels = {"adjacent": "Adjacent", "nonadjacent": "Non-adjacent", "shuffled": "Shuffled (adj)"}

    fig = plt.figure(figsize=(17, 8))
    gs  = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.38)

    # ── Row 0: ECDF (cols 0-2) ──────────────────────────────────────────────
    ax_ecdf = fig.add_subplot(gs[0, :3])
    for cond, dists, color in conditions:
        d3d = _pool_d3d(dists)
        if len(d3d) == 0:
            continue
        x = np.sort(d3d)
        x = x[x <= xlim]
        y = np.arange(1, len(x) + 1) / len(d3d)
        ax_ecdf.plot(x, y, color=color, linewidth=1.5,
                     label=f"{cond_labels[cond]}  (n={len(d3d):,}, med={np.median(d3d):.2f} µm)")
    ax_ecdf.set_xlabel("NN distance (µm)")
    ax_ecdf.set_ylabel("Cumulative fraction")
    ax_ecdf.set_xlim(-1, xlim)
    ax_ecdf.set_title("ECDF — all conditions")
    ax_ecdf.legend(fontsize=8)
    ax_ecdf.grid(True, linewidth=0.4, alpha=0.5)

    # ── Row 0, col 3: adjacent zoom histogram (0–2.5 µm) ────────────────────
    ax_zoom = fig.add_subplot(gs[0, 3])
    d3d_adj = _pool_d3d(dists_adj)
    zoom_xlim = 2.5
    if len(d3d_adj) > 0:
        med_adj = np.median(d3d_adj)
        ax_zoom.hist(d3d_adj, bins=60, density=False, range=(0, zoom_xlim),
                     color="steelblue", alpha=0.75, edgecolor="none")
        ax_zoom.axvline(med_adj, color="k", linestyle="--", linewidth=1.2)
        ax_zoom.set_title(f"Adjacent — zoom 0–2.5 µm\nn={len(d3d_adj):,}  med={med_adj:.2f} µm",
                          color="0.55", fontsize=9)
    ax_zoom.set_xlim(0, zoom_xlim)
    ax_zoom.set_xlabel("NN distance (µm)")
    ax_zoom.set_ylabel("Count")

    # ── Row 1, cols 0-2: count histograms ──────────────────────────────────
    hist_axes = [fig.add_subplot(gs[1, c]) for c in range(3)]
    for ax, (cond, dists, color) in zip(hist_axes, conditions):
        d3d = _pool_d3d(dists)
        if len(d3d) == 0:
            ax.set_title(f"{cond_labels[cond]}\n(no data)", color="0.55")
            continue
        med = np.median(d3d)
        ax.hist(d3d, bins=120, density=False, range=(0, xlim),
                color=color, alpha=0.75, edgecolor="none")
        ax.axvline(med, color="k", linestyle="--", linewidth=1.2)
        ax.set_xlim(-1, xlim)
        ax.set_title(f"{cond_labels[cond]}\nn={len(d3d):,}  med={med:.2f} µm",
                     color="0.55", fontsize=9)
        ax.set_xlabel("NN distance (µm)")
    hist_axes[0].set_ylabel("Count")
    for ax in hist_axes[1:]:
        ax.sharex(hist_axes[0])
        ax.sharey(hist_axes[0])
        ax.tick_params(labelleft=False)

    # ── Row 1, col 3: per-mouse median strip ──────────────────────────────
    ax_strip = fig.add_subplot(gs[1, 3])
    cond_colors = {"adjacent": "steelblue", "shuffled": "gray"}
    strip_conds = ["adjacent", "shuffled"]
    if not summary_df.empty:
        per_mouse = (
            summary_df[summary_df["condition"].isin(strip_conds)]
            .groupby(["mouse_id", "condition"])["median_d3d"]
            .median()
            .reset_index()
        )
        for _, grp in per_mouse.groupby("mouse_id"):
            adj_med  = grp.loc[grp["condition"] == "adjacent",  "median_d3d"].values
            shuf_med = grp.loc[grp["condition"] == "shuffled", "median_d3d"].values
            if len(adj_med) and len(shuf_med):
                ax_strip.plot([0, 1], [adj_med[0], shuf_med[0]],
                              "o-", color="0.7", linewidth=0.8, markersize=4, zorder=2)
        for i, cond in enumerate(strip_conds):
            vals = per_mouse.loc[per_mouse["condition"] == cond, "median_d3d"]
            ax_strip.scatter([i] * len(vals), vals,
                             color=cond_colors[cond], zorder=3, s=40)
    ax_strip.set_xticks([0, 1])
    ax_strip.set_xticklabels(["Adjacent", "Shuffled"], fontsize=8)
    ax_strip.set_ylabel("Median d3d (µm)")
    ax_strip.set_title("Per-mouse medians", color="0.55", fontsize=9)
    ax_strip.set_xlim(-0.5, 1.5)

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def get_clusters(pw_ds, unmixed=False):
    cluster_labels = pw_ds.load_cluster_labels(unmixed=unmixed)
    cluster_cids = pw_ds.load_sorted_cell_ids(unmixed=unmixed)
    return cluster_labels, cluster_cids


def print_cluster_counts(cluster_labels: pd.DataFrame) -> None:
    counts = cluster_labels.iloc[:, 0].value_counts().sort_index()
    print(f"\nCells per cluster ({len(cluster_labels)} total):")
    for cluster, count in counts.items():
        print(f"  Cluster {cluster}: {count} cells")


def main():
    parser = argparse.ArgumentParser(description="Pooled NN analysis")
    parser.add_argument("--mouse-ids", nargs="+", required=True,
                        help="One or more mouse IDs (e.g. 767022 747667)")
    args = parser.parse_args()

    pooled_dir = Path("/root/capsule/scratch/single_cell_unmixing/pooled")
    pooled_dir.mkdir(parents=True, exist_ok=True)

    adj_pairs    = get_adjacent_channel_pairs(CHAN_ORDER)
    nonadj_pairs = get_nonadjacent_channel_pairs(CHAN_ORDER)
    print(f"Adjacent pairs:     {adj_pairs}")
    print(f"Non-adjacent pairs: {nonadj_pairs}")

    global_dists_adj    = {}
    global_dists_nonadj = {}
    global_dists_shuf   = {}
    all_summary_rows: list[dict] = []

    for mouse_id in args.mouse_ids:
        print(f"\n{'='*60}\nMouse {mouse_id}\n{'='*60}")
        output_dir = Path(f"/root/capsule/scratch/single_cell_unmixing/{mouse_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset, pw_ds = get_data(mouse_id)
        if pw_ds is None:
            print(f"  No pairwise dataset for {mouse_id} — skipping.")
            continue

        cluster_labels, cluster_cids = get_clusters(pw_ds, unmixed=False)
        print_cluster_counts(cluster_labels)

        mouse_dists_adj    = {}
        mouse_dists_nonadj = {}
        mouse_dists_shuf   = {}

        for round_key in pw_ds.rounds:
            print(f"\n  Analysing {round_key}...")
            spots_df = pw_ds.rounds[round_key].load_spots(table_type="mixed")
            print(f"    {len(spots_df):,} spots, {spots_df['cell_id'].nunique():,} cells")

            _, spots_filt = select_cells_for_analysis(
                pw_ds, spots_df, round_key, CHAN_ORDER, max_cells=500
            )

            dists_adj    = collect_nn_dists_all_cells(spots_filt, adj_pairs,    VOXEL_SIZE)
            dists_nonadj = collect_nn_dists_all_cells(spots_filt, nonadj_pairs, VOXEL_SIZE)
            spots_shuf   = shuffle_spots_within_cells(spots_filt)
            dists_shuf   = collect_nn_dists_all_cells(spots_shuf, adj_pairs, VOXEL_SIZE)

            # per-round figure
            plot_nn_dist_comparison(
                dists_adj, dists_nonadj, dists_shuf,
                output_path=output_dir / f"{mouse_id}_{round_key}_nn_dist_comparison.png",
                title=f"Mouse {mouse_id} | {round_key} — NN distance distributions",
            )

            # summary stats
            all_summary_rows.extend(build_summary_rows(dists_adj,    "adjacent",    mouse_id, round_key))
            all_summary_rows.extend(build_summary_rows(dists_nonadj, "nonadjacent", mouse_id, round_key))
            all_summary_rows.extend(build_summary_rows(dists_shuf,   "shuffled",    mouse_id, round_key))

            _merge_dists(mouse_dists_adj,    dists_adj)
            _merge_dists(mouse_dists_nonadj, dists_nonadj)
            _merge_dists(mouse_dists_shuf,   dists_shuf)
            _merge_dists(global_dists_adj,    dists_adj)
            _merge_dists(global_dists_nonadj, dists_nonadj)
            _merge_dists(global_dists_shuf,   dists_shuf)

        # per-mouse all-rounds combined figure
        plot_nn_dist_comparison(
            _concat_dists(mouse_dists_adj),
            _concat_dists(mouse_dists_nonadj),
            _concat_dists(mouse_dists_shuf),
            output_path=output_dir / f"{mouse_id}_all_rounds_nn_dist_comparison.png",
            title=f"Mouse {mouse_id} | All rounds combined — NN distance distributions",
        )

    # ── save summary CSV ───────────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_summary_rows)
    summary_path = pooled_dir / "nn_dist_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

    # ── combined all-mice figure ───────────────────────────────────────────────
    plot_combined_mice_figure(
        _concat_dists(global_dists_adj),
        _concat_dists(global_dists_nonadj),
        _concat_dists(global_dists_shuf),
        summary_df=summary_df,
        output_path=pooled_dir / "all_mice_nn_dist_comparison.png",
        title=f"All mice ({', '.join(args.mouse_ids)}) — NN distance distributions",
    )


if __name__ == "__main__":
    main()


