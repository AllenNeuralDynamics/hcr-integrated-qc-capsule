"""top level run script

Example usage:
    python run_capsule.py --mouse-id 755252
    python run_capsule.py --mouse-id 755252 --overwrite
    python run_capsule.py --mouse-id 755252 --bucket my-bucket --overwrite
"""

import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from aind_hcr_data_loader.codeocean_utils import (
    MouseRecord,
    attach_mouse_record_to_workstation,
    print_attach_results,
)
from aind_hcr_data_loader.hcr_dataset import create_hcr_dataset_from_schema
from aind_hcr_data_loader.pairwise_dataset import create_pairwise_unmixing_dataset
from aind_hcr_qc.viz.intergrated_datasets import plot_intensity_violins
from aind_hcr_qc.viz.spectral_unmixing import plot_channel_intensity_histograms_by_round
import aind_hcr_qc.viz.cell_x_gene
from aind_hcr_qc.utils.s3_qc import QC_S3_BUCKET, check_plot_exists, upload_plot

from plot_configs import SPOTS_PLOTS, TAXONOMY_PLOTS

CATALOG_BASE = Path("/src/ophys-mfish-dataset-catalog/mice")
DATA_DIR = Path("/root/capsule/data")
OUTPUT_DIR = Path("/root/capsule/results")


def load_data(mouse_id):
    catalog_path = CATALOG_BASE / f"{mouse_id}.json"

    record = MouseRecord.from_json_file(catalog_path)
    results = attach_mouse_record_to_workstation(record)
    print_attach_results(results)

    dataset = create_hcr_dataset_from_schema(catalog_path, DATA_DIR)

    pairwise_asset_name = record.derived_assets.get("pairwise_unmixing")
    if pairwise_asset_name is None:
        print("No pairwise_unmixing asset found in catalog record — skipping.")
        return dataset, None, None

    pw_dataset = create_pairwise_unmixing_dataset(
        mouse_id=mouse_id,
        pairwise_asset_path=DATA_DIR / pairwise_asset_name,
        source_dataset=dataset,
    )

    # Collect all assets that feed into this dataset for provenance tracking.
    source_assets = {
        "rounds": dict(record.rounds),
        **{k: v for k, v in record.derived_assets.items() if v is not None},
    }
    return dataset, pw_dataset, source_assets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_plots_by_s3(plot_specs, bucket, mouse_id, overwrite):
    """Return the subset of *plot_specs* not yet on S3 (or all if *overwrite*)."""
    to_run = []
    for spec in plot_specs:
        plot_type = spec["plot_type"]
        if not overwrite and check_plot_exists(bucket, mouse_id, plot_type):
            print(
                f"QC plot '{plot_type}' already exists on S3 for mouse {mouse_id}."
                " Pass --overwrite to regenerate."
            )
        else:
            to_run.append(spec)
    return to_run


def _upload_and_close(fig, bucket, mouse_id, plot_type, plot_kwargs, source_assets):
    """Upload a figure to S3 with a JSON sidecar and close it."""
    upload_plot(
        fig=fig,
        bucket=bucket,
        mouse_id=mouse_id,
        plot_type=plot_type,
        metadata={
            "plot_kwargs": plot_kwargs,
            "source_assets": source_assets,
        },
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Category runners
# ---------------------------------------------------------------------------

def run_spots_plots(mouse_id, pw_dataset, source_assets, bucket, overwrite, spot_specs=None):
    """Spots intensity violin plots."""
    plots_to_run = _filter_plots_by_s3(spot_specs or SPOTS_PLOTS, bucket, mouse_id, overwrite)
    if not plots_to_run:
        return

    spots_df = pw_dataset.load_all_rounds_spots_mp(
        table_type="unmixed_spots", remove_fg_bg_cols=False
    )
    if "rd_ch_unmixed_gene" not in spots_df.columns:
        spots_df["rd_ch_unmixed_gene"] = (
            spots_df["round"] + "-" + spots_df["unmixed_chan"] + "-" + spots_df["unmixed_gene"]
        )

    for spec in plots_to_run:
        if spec["plot_type"] == "spots_intensity_hist_log":
            fig = plot_channel_intensity_histograms_by_round(
                spots_df,
                **spec["plot_kwargs"],
                title_prefix=f"{mouse_id} - ",
            )
        else:
            plot_intensity_violins(
                spots_df,
                **spec["plot_kwargs"],
                save=False,
                show=True,
                close=False,
            )
            fig = plt.gcf()
        _upload_and_close(
            fig, bucket, mouse_id,
            spec["plot_type"], spec["plot_kwargs"], source_assets,
        )

    del spots_df


def run_taxonomy_plots(mouse_id, pw_dataset, source_assets, bucket, overwrite, taxonomy_specs=None):
    """Taxonomy centroid scatter plots."""
    plots_to_run = _filter_plots_by_s3(taxonomy_specs or TAXONOMY_PLOTS, bucket, mouse_id, overwrite)
    if not plots_to_run:
        return

    cell_info = pw_dataset.get_cell_info()
    cell_type_df = pw_dataset.load_taxonomy_cell_types()
    cells = cell_info.merge(
        cell_type_df.reset_index(),
        on="cell_id",
        how="left",
        validate="1:1",
    )

    for spec in plots_to_run:
        aind_hcr_qc.viz.cell_x_gene.plot_all_subclass_centroids(
            cells,
            **spec["plot_kwargs"],
            save=False,
            show=True,
            close=False,
        )
        _upload_and_close(
            plt.gcf(), bucket, mouse_id,
            spec["plot_type"], spec["plot_kwargs"], source_assets,
        )

    del cells


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_plots(
    mouse_id: str,
    pw_dataset,
    source_assets: dict | None = None,
    bucket: str = QC_S3_BUCKET,
    overwrite: bool = False,
    plot_types: list[str] | None = None,
) -> None:
    """Generate all QC plots and upload each to S3 with a JSON sidecar."""
    if pw_dataset is None:
        print("No pairwise dataset — skipping plots.")
        return

    if plot_types:
        allowed = set(plot_types)
        spot_specs = [s for s in SPOTS_PLOTS if s["plot_type"] in allowed]
        taxonomy_specs = [s for s in TAXONOMY_PLOTS if s["plot_type"] in allowed]
    else:
        spot_specs = SPOTS_PLOTS
        taxonomy_specs = TAXONOMY_PLOTS

    run_spots_plots(mouse_id, pw_dataset, source_assets, bucket, overwrite, spot_specs)
    run_taxonomy_plots(mouse_id, pw_dataset, source_assets, bucket, overwrite, taxonomy_specs)


def run(
    mouse_id: str,
    bucket: str = QC_S3_BUCKET,
    overwrite: bool = False,
    plot_types: list[str] | None = None,
) -> None:
    dataset, pw_dataset, source_assets = load_data(mouse_id)
    run_plots(
        mouse_id, pw_dataset,
        source_assets=source_assets,
        bucket=bucket,
        overwrite=overwrite,
        plot_types=plot_types,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse-id", required=True)
    parser.add_argument(
        "--bucket",
        default=QC_S3_BUCKET,
        help="S3 bucket for QC plot storage (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-generate and re-upload even if the plot already exists on S3",
    )
    parser.add_argument(
        "--plot-type",
        nargs="+",
        metavar="PLOT_TYPE",
        default=None,
        help="One or more plot type names to run (default: all). "
             "Example: --plot-type spots_intensity_hist_log spots_intensity_violins_alpha_order",
    )
    args = parser.parse_args()
    run(
        args.mouse_id,
        bucket=args.bucket,
        overwrite=args.overwrite,
        plot_types=args.plot_type,
    )
