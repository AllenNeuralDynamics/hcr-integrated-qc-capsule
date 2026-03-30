"""Safely delete a QC plot (PNG + JSON sidecar) from S3.

Runs in dry-run mode by default — use --confirm to actually delete.

Examples
--------
# Preview what would be deleted for one mouse
python delete_plot.py --mouse-id 755252 --plot-type spots_intensity_violins_round_chan

# Preview across all mice in a list
python delete_plot.py --mouse-id 755252 767022 782149 --plot-type spots_intensity_violins_round_chan

# Actually delete (will prompt for confirmation)
python delete_plot.py --mouse-id 755252 --plot-type spots_intensity_violins_round_chan --confirm
"""

import argparse

from aind_hcr_qc.utils.s3_qc import QC_S3_BUCKET, delete_plot


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mouse-id", nargs="+", required=True, metavar="MOUSE_ID",
                        help="One or more mouse IDs to target.")
    parser.add_argument("--plot-type", required=True,
                        help="Plot type name to delete (e.g. spots_intensity_violins_round_chan).")
    parser.add_argument("--bucket", default=QC_S3_BUCKET,
                        help="S3 bucket (default: %(default)s).")
    parser.add_argument("--confirm", action="store_true", default=False,
                        help="Actually delete. Without this flag only a dry-run preview is shown.")
    args = parser.parse_args()

    dry_run = not args.confirm

    if dry_run:
        print("DRY RUN — pass --confirm to actually delete.\n")
    else:
        print(f"About to delete plot_type='{args.plot_type}' for mice: {args.mouse_id}")
        answer = input("Type 'yes' to proceed: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            return
        print()

    for mouse_id in args.mouse_id:
        print(f"--- mouse {mouse_id} ---")
        delete_plot(
            bucket=args.bucket,
            mouse_id=mouse_id,
            plot_type=args.plot_type,
            dry_run=dry_run,
        )


if __name__ == "__main__":
    main()
