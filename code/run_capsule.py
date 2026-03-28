"""top level run script"""

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
from aind_hcr_qc.utils.s3_qc import QC_S3_BUCKET, check_plot_exists, upload_plot

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
# Plot constants
# ---------------------------------------------------------------------------

_INTENSITY_VIOLINS_PLOT_TYPE = "spots_intensity_violins_round_chan"
_INTENSITY_VIOLINS_KWARGS = {
    "intensity_threshold": 25.0,
    "order": "round_chan",
    "n_sample": 25_000,
}


def run_plots(
    mouse_id: str,
    pw_dataset,
    source_assets: dict | None = None,
    bucket: str = QC_S3_BUCKET,
    overwrite: bool = False,
) -> None:
    """Generate QC plots and upload each to S3 with a JSON sidecar.

    The S3 existence check happens *before* loading the heavy spots table so
    that the run can be skipped entirely without touching any large files.

    Parameters
    ----------
    mouse_id:
        Subject identifier.
    pw_dataset:
        Pairwise unmixing dataset.  If ``None`` all plots are skipped.
    source_assets:
        Dict of all upstream assets that contributed to the plot, e.g.
        ``{"rounds": {"R1": "...", ...}, "pairwise_unmixing": "..."}``.  Goes
        into the sidecar JSON for provenance.
    bucket:
        Target S3 bucket.
    overwrite:
        When ``True``, regenerate and re-upload even if the plot already
        exists on S3.
    """
    if pw_dataset is None:
        print("No pairwise dataset — skipping plots.")
        return

    # ------------------------------------------------------------------
    # Check S3 BEFORE loading the heavy spots table
    # ------------------------------------------------------------------
    plot_type = _INTENSITY_VIOLINS_PLOT_TYPE
    if not overwrite and check_plot_exists(bucket, mouse_id, plot_type):
        print(
            f"QC plot '{plot_type}' already exists on S3 for mouse {mouse_id}."
            " Pass --overwrite to regenerate."
        )
        return

    # ------------------------------------------------------------------
    # Load data (only reached when the plot is missing or overwrite=True)
    # ------------------------------------------------------------------

    spots_df = pw_dataset.load_all_rounds_spots_mp(
        table_type="unmixed_spots", remove_fg_bg_cols=False
    )
    if 'rd_ch_unmixed_gene' not in spots_df.columns:
        spots_df["rd_ch_unmixed_gene"] = (
            spots_df["round"] + "-" + spots_df["unmixed_chan"] + "-" + spots_df["unmixed_gene"]
        )

    plot_intensity_violins(
        spots_df,
        **_INTENSITY_VIOLINS_KWARGS,
        save=False,
        show=True,   # keeps figure open so we can capture it below
        close=False,
    )
    del spots_df

    # ------------------------------------------------------------------
    # Upload PNG + JSON sidecar to S3
    # ------------------------------------------------------------------
    fig = plt.gcf()
    upload_plot(
        fig=fig,
        bucket=bucket,
        mouse_id=mouse_id,
        plot_type=plot_type,
        metadata={
            "plot_kwargs": _INTENSITY_VIOLINS_KWARGS,
            "source_assets": source_assets,
        },
    )
    plt.close(fig)


def run(mouse_id: str, bucket: str = QC_S3_BUCKET, overwrite: bool = False) -> None:
    dataset, pw_dataset, source_assets = load_data(mouse_id)
    run_plots(mouse_id, pw_dataset, source_assets=source_assets, bucket=bucket, overwrite=overwrite)


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
    args = parser.parse_args()
    run(args.mouse_id, bucket=args.bucket, overwrite=args.overwrite)
