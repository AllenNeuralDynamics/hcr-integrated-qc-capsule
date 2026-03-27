"""top level run script"""

import argparse
from pathlib import Path

from aind_hcr_data_loader.codeocean_utils import (
    MouseRecord,
    attach_mouse_record_to_workstation,
    print_attach_results,
)
from aind_hcr_data_loader.hcr_dataset import create_hcr_dataset_from_schema
from aind_hcr_data_loader.pairwise_dataset import create_pairwise_unmixing_dataset
from aind_hcr_qc.viz.intergrated_datasets import plot_intensity_violins

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
        return dataset, None

    pw_dataset = create_pairwise_unmixing_dataset(
        mouse_id=mouse_id,
        pairwise_asset_path=DATA_DIR / pairwise_asset_name,
        source_dataset=dataset,
    )
    return dataset, pw_dataset


def run_plots(mouse_id, pw_dataset):
    if pw_dataset is None:
        print("No pairwise dataset — skipping plots.")
        return

    spots_df = pw_dataset.load_all_rounds_spots_mp(
        table_type="unmixed_spots", remove_fg_bg_cols=False
    )
    if 'rd_ch_unmixed_gene' not in spots_df.columns:
        spots_df["rd_ch_unmixed_gene"] = (
            spots_df["round"] + "-" + spots_df["unmixed_chan"] + "-" + spots_df["unmixed_gene"]
        )

    save_kwargs = {"save": True,
                   "filename": f"{mouse_id}_intensity_violins_round_chan",
                   "output_dir":OUTPUT_DIR,}
    plot_intensity_violins(
        spots_df,
        order="round_chan",
        intensity_threshold=25.0,
        **save_kwargs
    )
    del spots_df


def run(mouse_id):
    dataset, pw_dataset = load_data(mouse_id)
    run_plots(mouse_id, pw_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse-id", required=True)
    args = parser.parse_args()
    run(args.mouse_id)
