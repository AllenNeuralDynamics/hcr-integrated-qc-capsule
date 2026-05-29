#!/usr/bin/env python
"""
Batch export single-cell unmixing metric figures for all coregistered cells of a given mouse.

Organizes output by mouse ID, with one PNG per metric per cell.

Usage:
    python batch_export_single_cell_metrics.py --mouse-id 782149 --round-key R5
    python batch_export_single_cell_metrics.py --mouse-id 782149 --output-base /mnt/results
    python batch_export_single_cell_metrics.py --mouse-id 782149 --metrics r d_assign_neighbor_ratio_1
"""

import argparse
import json
import sys
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


def batch_export_mouse(
    mouse_id: str,
    round_key: Optional[str] = None,
    metric_cols: Optional[list] = None,
    output_base: Path = Path("/root/capsule/results"),
    data_dir: Path = Path("/root/capsule/data"),
    dpi: int = 150,
    spot_size: float = 30,
    fast_plot: bool = False,
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

    Returns:
        Dictionary with export statistics
    """
    if metric_cols is None:
        metric_cols = DEFAULT_METRICS

    # Load data
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

    # Track statistics
    stats = {
        "mouse_id": mouse_id,
        "rounds_exported": {},
        "total_cells": 0,
        "total_figures": 0,
        "total_errors": 0,
    }

    # Export per round
    for rk in rounds_to_export:
        # Get unique cells for this round
        cells_in_round = sorted(
            spots.loc[spots["round"] == rk, "cell_id"].unique()
        )
        print(f"\n  Round {rk}: {len(cells_in_round)} cells")

        round_stats = {
            "cells": len(cells_in_round),
            "figures_exported": 0,
            "errors": 0,
        }

        # Export each cell
        for i, cell_id in enumerate(cells_in_round, 1):
            n_figs = export_cell_metrics(
                cell_id=cell_id,
                round_key=rk,
                spots=spots,
                dataset=dataset,
                chan_order=DEFAULT_CHANNEL_ORDER,
                chan_colors=chan_colors,
                metric_cols=metric_cols,
                output_dir=mouse_output_dir,
                dpi=dpi,
                spot_size=spot_size,
                fast_plot=fast_plot,
                verbose=(i % 10 == 0),  # Print every 10th cell
            )
            if n_figs > 0:
                round_stats["figures_exported"] += n_figs
            else:
                round_stats["errors"] += 1

        stats["rounds_exported"][rk] = round_stats
        stats["total_cells"] += len(cells_in_round)
        stats["total_figures"] += round_stats["figures_exported"]
        stats["total_errors"] += round_stats["errors"]

        print(f"    ✓ Exported {round_stats['figures_exported']} figures ({round_stats['errors']} errors)")

    # Write manifest
    manifest_path = mouse_output_dir / "export_manifest.json"
    stats["output_dir"] = str(mouse_output_dir)
    with open(manifest_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Export complete!")
    print(f"  Output: {mouse_output_dir}")
    print(f"  Total figures: {stats['total_figures']}")
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
        default=150,
        help="Figure DPI (default: 150)",
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
