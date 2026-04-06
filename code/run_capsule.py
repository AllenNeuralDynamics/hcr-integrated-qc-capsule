"""top level run script

Example usage:
    python run_capsule.py --mouse-id 755252
    python run_capsule.py --mouse-id 755252 --overwrite
    python run_capsule.py --mouse-id 755252 --bucket my-bucket --overwrite
"""

import argparse
import boto3
import matplotlib.pyplot as plt
from pathlib import Path
import json
import re

from aind_hcr_data_loader.codeocean_utils import (
    MouseRecord,
    attach_mouse_record_to_workstation,
    print_attach_results,
)
from aind_hcr_data_loader.hcr_dataset import create_hcr_dataset_from_schema
from aind_hcr_data_loader.pairwise_dataset import create_pairwise_unmixing_dataset
from aind_hcr_qc.viz.intergrated_datasets import plot_intensity_violins, plot_gene_spot_count_pairplot
from aind_hcr_qc.viz.spectral_unmixing import plot_channel_intensity_histograms_by_round
import aind_hcr_qc.viz.cell_x_gene
from aind_hcr_qc.utils.s3_qc import (
    QC_S3_BUCKET,
    QC_S3_PREFIX,
    check_plot_exists,
    upload_plot,
    check_round_plot_exists,
    upload_round_plot,
)
from aind_hcr_qc.viz.sample_overview import plot_tile_overview, raw_asset_name_from_processed

from plot_configs import SPOTS_PLOTS, TAXONOMY_PLOTS, CXG_PLOTS
from metrics import collect_spots_metrics, collect_cxg_metrics, load_mouse_metrics, save_mouse_metrics

CATALOG_BASE = Path("/src/ophys-mfish-dataset-catalog/mice")
DATA_DIR = Path("/root/capsule/data")
OUTPUT_DIR = Path("/root/capsule/results")

NG_LINK_FILENAME = "ng_links.json"


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
    return dataset, pw_dataset, source_assets, record


_SUBSET_PREFIX = {
    "all": "taxonomy_all_map",
    "inhibitory": "taxonomy_inh_map",
}


# ---------------------------------------------------------------------------
# Neuroglancer link collection
# ---------------------------------------------------------------------------

def _collect_ng_links_for_round(asset_dir: Path) -> list[dict]:
    """Return a list of {name, url} dicts for every NG JSON in *asset_dir*.

    Scans all *.json files at the top level of the folder.  A file is treated
    as a neuroglancer link file if it contains a top-level ``ng_link`` key.
    """
    links = []
    if not asset_dir.is_dir():
        return links
    for json_path in sorted(asset_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        url = data.get("ng_link")
        if isinstance(url, str) and url.startswith("http"):
            links.append({"name": json_path.stem, "url": url})
    return links


def collect_and_upload_ng_links(
    mouse_id: str,
    record: MouseRecord,
    data_dir: Path = DATA_DIR,
    bucket: str = QC_S3_BUCKET,
    overwrite: bool = False,
) -> None:
    """Scan each round's dataset folder for neuroglancer JSON files and upload
    a consolidated ``ng_links.json`` to S3.

    S3 key::

        ctl/hcr/qc/{mouse_id}/ng_links.json

    Structure of the uploaded JSON::

        {
          "mouse_id": "755252",
          "rounds": {
            "R1": [{"name": "fused_ng", "url": "https://..."}],
            "R2": [...]
          }
        }

    All ng files are included (fused, camera_aligned, radially_corrected,
    multichannel_spot_annotation, cc_ng_*, etc.).
    """
    s3_key = f"{QC_S3_PREFIX}/{mouse_id}/{NG_LINK_FILENAME}"
    s3 = boto3.client("s3")

    if not overwrite:
        from botocore.exceptions import ClientError as _ClientError
        try:
            s3.head_object(Bucket=bucket, Key=s3_key)
            print(
                f"  [skip] ng_links.json already exists on S3 for mouse {mouse_id}."
                " Pass --overwrite to replace."
            )
            return
        except _ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                raise

    rounds_links: dict[str, list] = {}
    for round_label, asset_name in record.rounds.items():
        asset_dir = data_dir / asset_name
        links = _collect_ng_links_for_round(asset_dir)
        rounds_links[round_label] = links
        print(f"  [ng_links] {round_label} ({asset_name}): {len(links)} link(s) found")

    payload = {
        "mouse_id": mouse_id,
        "rounds": rounds_links,
    }

    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(payload, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"  [upload] ng_links.json -> s3://{bucket}/{s3_key}")


# ---------------------------------------------------------------------------
# Cell-typing plot copy
# ---------------------------------------------------------------------------

def copy_cell_typing_plots(
    mouse_id: str,
    record: MouseRecord,
    data_dir: Path = DATA_DIR,
    bucket: str = QC_S3_BUCKET,
    overwrite: bool = False,
) -> None:
    """Upload pre-generated cell-typing plots from the data asset to S3.

    Source files are uploaded under the canonical QC prefix with a
    ``taxonomy_all_map_`` / ``taxonomy_inh_map_`` prefix so the data viewer
    can distinguish subset and plot type.

    S3 key pattern::

        ctl/hcr/qc/{mouse_id}/{subset_prefix}_{original_stem}.png
    """
    cell_typing_asset = record.derived_assets.get("cell_typing")
    if cell_typing_asset is None:
        print(f"Mouse {mouse_id}: no cell_typing asset in catalog — skipping upload.")
        return

    cell_typing_dir = data_dir / cell_typing_asset
    s3 = boto3.client("s3")

    for subset, prefix in _SUBSET_PREFIX.items():
        plots_dirs = sorted((cell_typing_dir / f"{subset}_cells").glob("*/plots"))
        if not plots_dirs:
            print(f"Mouse {mouse_id}: no plots directory found under {subset}_cells — skipping.")
            continue

        plots_dir = plots_dirs[0]
        for src in sorted(plots_dir.glob("*.png")):
            plot_type = f"{prefix}_{src.stem}"
            if not overwrite and check_plot_exists(bucket, mouse_id, plot_type):
                print(f"  [skip] {plot_type} already exists on S3. Pass --overwrite to replace.")
                continue
            s3.upload_file(
                Filename=str(src),
                Bucket=bucket,
                Key=f"{QC_S3_PREFIX}/{mouse_id}/{plot_type}.png",
                ExtraArgs={"ContentType": "image/png"},
            )
            print(f"  [upload] {src.relative_to(data_dir)} -> s3://{bucket}/{QC_S3_PREFIX}/{mouse_id}/{plot_type}.png")


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


def run_cxg_plots(mouse_id, pw_dataset, source_assets, bucket, overwrite, cxg_specs=None):
    """Pairwise gene spot-count plots from the aggregated cell-by-gene table."""
    plots_to_run = _filter_plots_by_s3(cxg_specs or CXG_PLOTS, bucket, mouse_id, overwrite)
    if not plots_to_run:
        return

    cxg_wide = pw_dataset.load_aggregated_cxg(unmixed=True)

    for spec in plots_to_run:
        fig = plot_gene_spot_count_pairplot(
            cxg_wide,
            **spec["plot_kwargs"],
            title_prefix=f"{mouse_id} - ",
        )
        _upload_and_close(
            fig, bucket, mouse_id,
            spec["plot_type"], spec["plot_kwargs"], source_assets,
        )

    del cxg_wide


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
        cxg_specs = [s for s in CXG_PLOTS if s["plot_type"] in allowed]
    else:
        spot_specs = SPOTS_PLOTS
        taxonomy_specs = TAXONOMY_PLOTS
        cxg_specs = CXG_PLOTS

    run_spots_plots(mouse_id, pw_dataset, source_assets, bucket, overwrite, spot_specs)
    run_taxonomy_plots(mouse_id, pw_dataset, source_assets, bucket, overwrite, taxonomy_specs)
    run_cxg_plots(mouse_id, pw_dataset, source_assets, bucket, overwrite, cxg_specs)


# ---------------------------------------------------------------------------
# Round-specific QC
# ---------------------------------------------------------------------------

def run_round_qc(
    mouse_id: str,
    dataset,
    record: "MouseRecord",
    bucket: str = QC_S3_BUCKET,
    overwrite: bool = False,
    channel: str = "405",
    pyramid_level: int = 3,
) -> None:
    """Generate per-round tile overview plots and upload to S3.

    Each round's plot is stored under its own asset sub-folder so that
    processed and raw assets occupy distinct S3 paths::

        ctl/hcr/qc/{mouse_id}/{processed_asset_name}/tile_overview_ch{channel}.png

    The raw asset name (stripped of the ``_processed_*`` suffix) is recorded
    in the JSON sidecar for provenance but is not used as a data source
    (fused zarrs live in the processed asset).
    """
    plot_type = f"tile_overview_ch{channel}"

    for round_key, processed_name in record.rounds.items():
        raw_name = raw_asset_name_from_processed(processed_name)

        if not overwrite and check_round_plot_exists(
            bucket, mouse_id, processed_name, plot_type
        ):
            print(
                f"  [skip] {round_key} {plot_type} already on S3 for mouse {mouse_id}."
                " Pass --overwrite to replace."
            )
            continue

        hcr_round = dataset.rounds.get(round_key)
        if hcr_round is None:
            print(f"  [skip] {round_key} not found in dataset — skipping.")
            continue

        print(f"  [round qc] {round_key}: generating {plot_type} ...")
        fig = plot_tile_overview(hcr_round, channel=channel, pyramid_level=pyramid_level)
        gene_dict = hcr_round.processing_manifest.get("gene_dict", {})
        upload_round_plot(
            fig=fig,
            bucket=bucket,
            mouse_id=mouse_id,
            asset_name=processed_name,
            plot_type=plot_type,
            metadata={
                "round_label": round_key,
                "source_assets": {
                    "processed": processed_name,
                    "raw": raw_name,
                },
                "plot_kwargs": {
                    "channel": channel,
                    "pyramid_level": pyramid_level,
                },
                "gene_dict": gene_dict,
            },
        )
        plt.close(fig)


def run_metrics(
    mouse_id: str,
    pw_dataset,
    overwrite: bool = False,
) -> None:
    """Collect all QC metrics for *mouse_id* and write to scratch/metrics/."""
    if pw_dataset is None:
        print("No pairwise dataset — skipping metrics.")
        return

    metrics = load_mouse_metrics(mouse_id)
    changed = collect_spots_metrics(pw_dataset, metrics, overwrite=overwrite)
    changed |= collect_cxg_metrics(pw_dataset, metrics, overwrite=overwrite)
    if changed:
        save_mouse_metrics(mouse_id, metrics)


def run(
    mouse_id: str,
    bucket: str = QC_S3_BUCKET,
    overwrite: bool = False,
    plot_types: list[str] | None = None,
) -> None:
    dataset, pw_dataset, source_assets, record = load_data(mouse_id)
    run_metrics(mouse_id, pw_dataset, overwrite=overwrite)
    run_round_qc(mouse_id, dataset, record, bucket=bucket, overwrite=overwrite)
    run_plots(
        mouse_id, pw_dataset,
        source_assets=source_assets,
        bucket=bucket,
        overwrite=overwrite,
        plot_types=plot_types,
    )
    copy_cell_typing_plots(mouse_id, record, bucket=bucket, overwrite=overwrite)
    collect_and_upload_ng_links(mouse_id, record, bucket=bucket, overwrite=overwrite)


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
