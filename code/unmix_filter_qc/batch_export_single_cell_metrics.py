#!/usr/bin/env python
"""
Batch export single-cell unmixing metric figures for all coregistered cells of a given mouse.

Organizes output by mouse ID, with one PNG per metric per cell.

Usage:
    python batch_export_single_cell_metrics.py --mouse-id 782149 --round-key R5
    python batch_export_single_cell_metrics.py --mouse-id 782149 --output-base /mnt/results
    python batch_export_single_cell_metrics.py --mouse-id 782149 --metrics r d_assign_neighbor_ratio_1
    python batch_export_single_cell_metrics.py --mouse-id 782149 --sample-cells 50
"""

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from aind_hcr_data_loader import get_hcr_dataset_pairwise
from aind_hcr_qc.constants import Z1_CHANNEL_CMAP_VIBRANT

# Import local utilities
sys.path.insert(0, str(Path(__file__).parent))
import unmix_qc_utils


# Default metrics to export per cell
DEFAULT_METRICS = [
    "r",
    "dist",
    "d_assign_neighbor_ratio_1",
    "d_assign_neighbor_ratio_2",
    "dye_line_dist_ratio",
    "intensity_assigned_chan_norm",
]

# Default channel order
DEFAULT_CHANNEL_ORDER = ["488", "514", "561", "594", "638"]


_WORKER_STATE = {}


def load_data(
    mouse_id: str,
    data_dir: Path = Path("/root/capsule/data"),
    coreg_cells_only: bool = True,
) -> tuple:
    """
    Load HCR dataset and spots for a mouse.

    Args:
        mouse_id: Mouse ID (e.g., "782149")
        data_dir: Base data directory
        coreg_cells_only: Load only coregistered cells

    Returns:
        Tuple of (dataset, pw_ds, spots, coreg_df)
    """
    print(f"Loading dataset for mouse {mouse_id}...")
    dataset, pw_ds, spots = get_hcr_dataset_pairwise(
        mouse_id=mouse_id,
        data_dir=data_dir,
        load_spots=True,
        return_removed=True,
        coreg_cells_only=coreg_cells_only,
    )

    coreg_df = dataset.load_coreg_table()
    print(f"  Loaded {len(spots)} spots across {len(spots['cell_id'].unique())} cells")
    print(f"  Available rounds: {sorted(spots['round'].unique())}")

    return dataset, pw_ds, spots, coreg_df


def add_intensity_column(spots: pd.DataFrame) -> pd.DataFrame:
    """Add normalized intensity for assigned channel to spots dataframe."""
    if "intensity_assigned_chan_norm" not in spots.columns:
        print("  Computing intensity_assigned_chan_norm...")
        spots = unmix_qc_utils.add_intensity_assigned_chan(spots)
    return spots


def export_cell_metrics(
    cell_id: int,
    round_key: str,
    spots: pd.DataFrame,
    dataset,
    chan_order: list,
    chan_colors: dict,
    metric_cols: list,
    output_dir: Path,
    dpi: int = 150,
    spot_size: float = 30,
    fast_plot: bool = False,
    verbose: bool = True,
) -> int:
    """
    Export all metric figures for a single cell.

    Args:
        cell_id: Cell ID to export
        round_key: Round key (e.g., "R5")
        spots: Spots dataframe
        dataset: HCRDataset object
        chan_order: Channel order list
        chan_colors: Channel color mapping
        metric_cols: List of metric column names
        output_dir: Output directory for PNG files
        dpi: DPI for figure export
        spot_size: Spot size for plotting
        fast_plot: Use fast plot mode
        verbose: Print progress

    Returns:
        Number of figures successfully exported
    """
    # Filter spots for this cell and round
    m_x = spots[(spots["round"] == round_key) & (spots["cell_id"] == cell_id)]
    u_x = spots[
        (spots["round"] == round_key)
        & (spots["cell_id"] == cell_id)
        & (spots["removed"] == False)
    ]

    if len(m_x) == 0:
        if verbose:
            print(f"    Warning: No spots found for cell {cell_id} in {round_key}")
        return 0

    if verbose:
        print(f"    Exporting {len(metric_cols)} metrics for cell {cell_id} ({len(m_x)} total spots)...")

    try:
        batch_paths = unmix_qc_utils.batch_save_single_cell_unmixing_mg2(
            m_x,
            u_x,
            cell_id=cell_id,
            round_key=round_key,
            dataset=dataset,
            chan_order=chan_order,
            chan_colors=chan_colors,
            metric_cols=metric_cols,
            output_dir=output_dir,
            spot_size=spot_size,
            fast_plot=fast_plot,
            dpi=dpi,
            file_ext="png",
            verbose=False,  # Suppress per-figure output
        )
        return len(batch_paths)
    except Exception as e:
        print(f"    Error exporting cell {cell_id}: {e}")
        return 0


def _process_cell_all_rounds(
    cell_id: int,
    rounds_to_export: list,
    spots: pd.DataFrame,
    dataset,
    chan_colors: dict,
    metric_cols: list,
    output_dir: Path,
    dpi: int,
    spot_size: float,
    fast_plot: bool,
) -> dict:
    """Process all rounds for a single cell and return per-round timing/results."""
    cell_start = time.perf_counter()
    round_results = []
    cell_output_dir = Path(output_dir) / f"cell_{int(cell_id)}"
    cell_output_dir.mkdir(parents=True, exist_ok=True)

    for rk in rounds_to_export:
        round_start = time.perf_counter()
        n_spots = int(((spots["round"] == rk) & (spots["cell_id"] == cell_id)).sum())

        if n_spots == 0:
            round_results.append(
                {
                    "round": rk,
                    "cell_id": int(cell_id),
                    "spots": 0,
                    "figures_exported": 0,
                    "elapsed_sec": round(time.perf_counter() - round_start, 4),
                    "status": "no_spots",
                }
            )
            continue

        n_figs = export_cell_metrics(
            cell_id=cell_id,
            round_key=rk,
            spots=spots,
            dataset=dataset,
            chan_order=DEFAULT_CHANNEL_ORDER,
            chan_colors=chan_colors,
            metric_cols=metric_cols,
            output_dir=cell_output_dir,
            dpi=dpi,
            spot_size=spot_size,
            fast_plot=fast_plot,
            verbose=False,
        )
        elapsed = time.perf_counter() - round_start

        status = "ok" if n_figs > 0 else "error"
        round_results.append(
            {
                "round": rk,
                "cell_id": int(cell_id),
                "spots": n_spots,
                "figures_exported": int(n_figs),
                "elapsed_sec": round(elapsed, 4),
                "status": status,
            }
        )

    total_figs = int(sum(r["figures_exported"] for r in round_results))
    total_errors = int(sum(1 for r in round_results if r["status"] == "error"))
    return {
        "cell_id": int(cell_id),
        "round_results": round_results,
        "figures_exported": total_figs,
        "errors": total_errors,
        "duration_sec": round(time.perf_counter() - cell_start, 4),
    }


def _init_cell_worker(
    mouse_id: str,
    data_dir: str,
    metric_cols: list,
    output_dir: str,
    dpi: int,
    spot_size: float,
    fast_plot: bool,
    rounds_to_export: list,
):
    """Initialize one worker process with dataset, spots, and static parameters."""
    dataset, pw_ds, spots, coreg_df = load_data(mouse_id=mouse_id, data_dir=Path(data_dir))
    spots = add_intensity_column(spots)
    chan_colors = {
        k: v for k, v in Z1_CHANNEL_CMAP_VIBRANT.items() if k in DEFAULT_CHANNEL_ORDER
    }

    _WORKER_STATE["dataset"] = dataset
    _WORKER_STATE["spots"] = spots
    _WORKER_STATE["chan_colors"] = chan_colors
    _WORKER_STATE["metric_cols"] = metric_cols
    _WORKER_STATE["output_dir"] = Path(output_dir)
    _WORKER_STATE["dpi"] = dpi
    _WORKER_STATE["spot_size"] = spot_size
    _WORKER_STATE["fast_plot"] = fast_plot
    _WORKER_STATE["rounds_to_export"] = rounds_to_export


def _process_cell_worker(cell_id: int) -> dict:
    """Worker entrypoint: process all selected rounds for one cell."""
    return _process_cell_all_rounds(
        cell_id=cell_id,
        rounds_to_export=_WORKER_STATE["rounds_to_export"],
        spots=_WORKER_STATE["spots"],
        dataset=_WORKER_STATE["dataset"],
        chan_colors=_WORKER_STATE["chan_colors"],
        metric_cols=_WORKER_STATE["metric_cols"],
        output_dir=_WORKER_STATE["output_dir"],
        dpi=_WORKER_STATE["dpi"],
        spot_size=_WORKER_STATE["spot_size"],
        fast_plot=_WORKER_STATE["fast_plot"],
    )


def batch_export_mouse(
    mouse_id: str,
    round_key: Optional[str] = None,
    metric_cols: Optional[list] = None,
    output_base: Path = Path("/root/capsule/results"),
    data_dir: Path = Path("/root/capsule/data"),
    dpi: int = 150,
    spot_size: float = 30,
    fast_plot: bool = False,
    sample_cells: Optional[int] = 50,
    sample_seed: int = 42,
    num_workers: int = 1,
) -> dict:
    """
    Export metric figures for all coregistered cells of a mouse.

    Args:
        mouse_id: Mouse ID
        round_key: Specific round to export (if None, all rounds)
        metric_cols: Metrics to export (if None, use defaults)
        output_base: Base output directory (mouse-specific subdir created here)
        data_dir: Data directory path
        dpi: Figure DPI
        spot_size: Spot size for plotting
        fast_plot: Use fast plot mode
        sample_cells: Number of coreg cells to sample overall (None or <=0 for all)
        sample_seed: Random seed for reproducible sampling
        num_workers: Number of worker processes (1 = sequential)

    Returns:
        Dictionary with export statistics
    """
    if metric_cols is None:
        metric_cols = DEFAULT_METRICS

    # Load data in parent for sampling/stats/spots export
    dataset, pw_ds, spots, coreg_df = load_data(
        mouse_id=mouse_id,
        data_dir=data_dir,
    )

    # Add intensity column if needed
    spots = add_intensity_column(spots)

    # Set up channel colors
    chan_colors = {
        k: v for k, v in Z1_CHANNEL_CMAP_VIBRANT.items() if k in DEFAULT_CHANNEL_ORDER
    }

    # Determine rounds to export
    available_rounds = sorted(spots["round"].unique())
    rounds_to_export = [round_key] if round_key else available_rounds

    print(f"\nExporting metrics for mouse {mouse_id}:")
    print(f"  Rounds: {rounds_to_export}")
    print(f"  Metrics: {metric_cols}")

    # Create output structure
    output_base = Path(output_base)
    mouse_output_dir = output_base / f"mouse_{mouse_id}"
    mouse_output_dir.mkdir(parents=True, exist_ok=True)

    # Save full spots dataframe for all coreg cells
    spots_export_path = mouse_output_dir / "coreg_spots_all_cells.csv"
    spots.to_csv(spots_export_path, index=False)
    print(f"  Saved coreg spots dataframe: {spots_export_path} ({len(spots)} rows)")

    # Select cells once, then process all selected rounds for each cell.
    cells_all = sorted(
        spots.loc[spots["round"].isin(rounds_to_export), "cell_id"].unique()
    )
    if sample_cells is not None and sample_cells > 0 and len(cells_all) > sample_cells:
        rng = np.random.default_rng(sample_seed)
        sampled_cells = rng.choice(cells_all, size=sample_cells, replace=False)
        selected_cells = sorted(sampled_cells.tolist())
        print(f"  Cells: sampling {len(selected_cells)} of {len(cells_all)} across selected rounds")
    else:
        selected_cells = cells_all
        print(f"  Cells: exporting all {len(selected_cells)} across selected rounds")

    round_cell_totals = {
        rk: int(spots.loc[spots["round"] == rk, "cell_id"].nunique()) for rk in rounds_to_export
    }

    # Track statistics
    stats = {
        "mouse_id": mouse_id,
        "rounds_exported": {},
        "selected_cells": len(selected_cells),
        "total_cells": 0,
        "total_figures": 0,
        "total_errors": 0,
        "sample_cells": sample_cells,
        "sample_seed": sample_seed,
        "num_workers": num_workers,
        "spots_export_path": str(spots_export_path),
        "spots_rows": int(len(spots)),
    }
    all_cell_timings = []
    export_start = time.perf_counter()

    for rk in rounds_to_export:
        stats["rounds_exported"][rk] = {
            "total_coreg_cells": round_cell_totals[rk],
            "cells_selected": len(selected_cells),
            "cells_with_spots": 0,
            "figures_exported": 0,
            "errors": 0,
            "duration_sec": 0.0,
            "avg_cell_duration_sec": 0.0,
        }

    cell_results = []
    if num_workers > 1:
        print(f"  Processing cells in parallel with {num_workers} workers...")
        with cf.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_cell_worker,
            initargs=(
                mouse_id,
                str(data_dir),
                metric_cols,
                str(mouse_output_dir),
                dpi,
                spot_size,
                fast_plot,
                rounds_to_export,
            ),
        ) as executor:
            futures = {executor.submit(_process_cell_worker, int(cid)): int(cid) for cid in selected_cells}
            for i, fut in enumerate(cf.as_completed(futures), 1):
                cid = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"    Cell {cid} failed: {e}")
                    result = {
                        "cell_id": int(cid),
                        "round_results": [
                            {
                                "round": rk,
                                "cell_id": int(cid),
                                "spots": 0,
                                "figures_exported": 0,
                                "elapsed_sec": 0.0,
                                "status": "error",
                            }
                            for rk in rounds_to_export
                        ],
                        "figures_exported": 0,
                        "errors": len(rounds_to_export),
                        "duration_sec": 0.0,
                    }
                cell_results.append(result)
                if i % 10 == 0 or i == len(selected_cells):
                    print(f"    Completed {i}/{len(selected_cells)} cells")
    else:
        print("  Processing cells sequentially (num_workers=1)...")
        for i, cid in enumerate(selected_cells, 1):
            result = _process_cell_all_rounds(
                cell_id=int(cid),
                rounds_to_export=rounds_to_export,
                spots=spots,
                dataset=dataset,
                chan_colors=chan_colors,
                metric_cols=metric_cols,
                output_dir=mouse_output_dir,
                dpi=dpi,
                spot_size=spot_size,
                fast_plot=fast_plot,
            )
            cell_results.append(result)
            print(
                f"    Cell {int(cid)} took {result['duration_sec']:.2f}s "
                f"-> {result['figures_exported']} figures ({result['errors']} errors)"
            )

    for cell_result in cell_results:
        for rr in cell_result["round_results"]:
            rk = rr["round"]
            rstats = stats["rounds_exported"][rk]
            all_cell_timings.append(rr)

            if rr["spots"] > 0:
                rstats["cells_with_spots"] += 1
                stats["total_cells"] += 1
            rstats["figures_exported"] += rr["figures_exported"]
            rstats["duration_sec"] += rr["elapsed_sec"]
            if rr["status"] == "error":
                rstats["errors"] += 1

            stats["total_figures"] += rr["figures_exported"]
            if rr["status"] == "error":
                stats["total_errors"] += 1

    for rk in rounds_to_export:
        rstats = stats["rounds_exported"][rk]
        if rstats["cells_with_spots"] > 0:
            rstats["avg_cell_duration_sec"] = round(
                rstats["duration_sec"] / rstats["cells_with_spots"], 4
            )
        rstats["duration_sec"] = round(rstats["duration_sec"], 4)
        print(
            f"  Round {rk}: {rstats['figures_exported']} figures, {rstats['errors']} errors, "
            f"{rstats['cells_with_spots']}/{rstats['cells_selected']} cells with spots, "
            f"{rstats['duration_sec']:.2f}s total"
        )

    total_elapsed = time.perf_counter() - export_start
    stats["duration_sec"] = round(total_elapsed, 4)
    if stats["total_cells"] > 0:
        stats["avg_cell_duration_sec"] = round(total_elapsed / stats["total_cells"], 4)
    else:
        stats["avg_cell_duration_sec"] = 0.0

    # Write per-cell timing log
    timing_log_path = mouse_output_dir / "cell_timing_log.csv"
    pd.DataFrame(all_cell_timings).to_csv(timing_log_path, index=False)
    stats["timing_log_path"] = str(timing_log_path)

    # Write manifest
    manifest_path = mouse_output_dir / "export_manifest.json"
    stats["output_dir"] = str(mouse_output_dir)
    with open(manifest_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Export complete!")
    print(f"  Output: {mouse_output_dir}")
    print(f"  Total figures: {stats['total_figures']}")
    print(f"  Total duration: {total_elapsed:.2f}s ({stats['avg_cell_duration_sec']:.2f}s/cell)")
    print(f"  Timing log: {timing_log_path}")
    print(f"  Manifest: {manifest_path}")

    return stats


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch export single-cell metrics for all coregistered cells of a mouse.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_export_single_cell_metrics.py --mouse-id 782149
  python batch_export_single_cell_metrics.py --mouse-id 782149 --round-key R5
  python batch_export_single_cell_metrics.py --mouse-id 782149 --output-base /mnt/results
  python batch_export_single_cell_metrics.py --mouse-id 782149 --metrics r d_assign_neighbor_ratio_1
    python batch_export_single_cell_metrics.py --mouse-id 782149 --sample-cells 50 --sample-seed 42
        """,
    )

    parser.add_argument(
        "--mouse-id",
        required=True,
        help="Mouse ID (e.g., 782149)",
    )
    parser.add_argument(
        "--round-key",
        default=None,
        help="Specific round to export (e.g., R5). If None, exports all rounds.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=f"Metrics to export (default: {DEFAULT_METRICS})",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("/root/capsule/results"),
        help="Base output directory. Outputs are written under mouse_<id>/ without round subfolders. (default: /root/capsule/results)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/root/capsule/data"),
        help="Data directory path (default: /root/capsule/data)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI (default: 200)",
    )
    parser.add_argument(
        "--spot-size",
        type=float,
        default=30,
        help="Spot size for plotting (default: 30)",
    )
    parser.add_argument(
        "--fast-plot",
        action="store_true",
        help="Use fast plot mode (skip some overlays)",
    )
    parser.add_argument(
        "--sample-cells",
        type=int,
        default=50,
        help="Number of coreg cells to sample per round (default: 50). Use 0 or negative to export all.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for reproducible cell sampling (default: 42).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker processes across cells (default: 1).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Use defaults if metrics not specified
    metric_cols = args.metrics if args.metrics else DEFAULT_METRICS

    try:
        stats = batch_export_mouse(
            mouse_id=args.mouse_id,
            round_key=args.round_key,
            metric_cols=metric_cols,
            output_base=args.output_base,
            data_dir=args.data_dir,
            dpi=args.dpi,
            spot_size=args.spot_size,
            fast_plot=args.fast_plot,
            sample_cells=args.sample_cells,
            sample_seed=args.sample_seed,
            num_workers=args.num_workers,
        )

        print("\n" + "=" * 60)
        print("EXPORT SUMMARY")
        print("=" * 60)
        print(json.dumps(stats, indent=2))

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
