"""Collect and concatenate per-mouse atlas-comparison CSVs.

After running run_atlas_compare.py for multiple mice, use this script to
gather all per-run CSVs into two combined files:

    _combined/comparison_all.csv        -- all cluster×gene rows
    _combined/cluster_matches_all.csv   -- all cluster→supertype matches

The combined CSVs have two extra columns that identify each row:
    mouse_id     -- e.g. "782149"
    spot_filter  -- "all" or "valid"

Example usage
-------------
    python collect_results.py
    python collect_results.py --scratch-dir /some/other/path
    python collect_results.py --overwrite
"""

import argparse
from pathlib import Path

import pandas as pd

SCRATCH_BASE = Path("/root/capsule/scratch/ref_atlas_validation")
COMBINED_DIR_NAME = "_combined"


def collect(scratch_dir: Path, overwrite: bool = False) -> None:
    combined_dir = scratch_dir / COMBINED_DIR_NAME
    combined_dir.mkdir(parents=True, exist_ok=True)

    out_comparison = combined_dir / "comparison_all.csv"
    out_matches    = combined_dir / "cluster_matches_all.csv"

    if not overwrite and out_comparison.exists() and out_matches.exists():
        print(
            f"Combined CSVs already exist in {combined_dir}.\n"
            "Pass --overwrite to regenerate."
        )
        return

    comparison_dfs    = []
    cluster_match_dfs = []

    # Walk: scratch_dir/<mouse_id>/atlas_compare/<all_unmixed|valid_unmixed>/
    for mouse_dir in sorted(scratch_dir.iterdir()):
        if not mouse_dir.is_dir() or mouse_dir.name.startswith("_"):
            continue
        mouse_id = mouse_dir.name
        atlas_dir = mouse_dir / "atlas_compare"
        if not atlas_dir.is_dir():
            continue

        for run_dir in sorted(atlas_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            comp_path = run_dir / "comparison.csv"
            cm_path   = run_dir / "cluster_matches.csv"

            if comp_path.exists():
                comparison_dfs.append(pd.read_csv(comp_path))
                print(f"  [comparison]      {comp_path.relative_to(scratch_dir)}")
            else:
                print(f"  [missing]         {comp_path.relative_to(scratch_dir)}")

            if cm_path.exists():
                cluster_match_dfs.append(pd.read_csv(cm_path))
                print(f"  [cluster_matches] {cm_path.relative_to(scratch_dir)}")
            else:
                print(f"  [missing]         {cm_path.relative_to(scratch_dir)}")

    if not comparison_dfs:
        print("No comparison CSVs found. Run run_atlas_compare.py for at least one mouse first.")
        return

    combined_comparison = pd.concat(comparison_dfs, ignore_index=True)
    combined_matches    = pd.concat(cluster_match_dfs, ignore_index=True) if cluster_match_dfs else pd.DataFrame()

    combined_comparison.to_csv(out_comparison, index=False)
    combined_matches.to_csv(out_matches, index=False)

    n_mice   = combined_comparison["mouse_id"].nunique()
    n_filter = combined_comparison["spot_filter"].nunique()
    print(
        f"\nCombined {n_mice} mouse/mice × {n_filter} filter(s) "
        f"→ {len(combined_comparison):,} rows"
    )
    print(f"  {out_comparison}")
    print(f"  {out_matches}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Collect per-mouse atlas-comparison CSVs into combined files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scratch-dir", default=str(SCRATCH_BASE),
        help="Root directory containing per-mouse result folders.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing combined CSVs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    collect(scratch_dir=Path(args.scratch_dir), overwrite=args.overwrite)
