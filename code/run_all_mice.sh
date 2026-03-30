#!/usr/bin/env bash
set -e

# Run QC plots for a list of mice.
# Usage:
#   bash run_all_mice.sh                    # run all mice below
#   bash run_all_mice.sh --overwrite        # re-generate existing plots

MICE=(
    #747667
    #749315
    #754803
    755252
    
    767022
    #767108
    782149
    #783551-v1
    #783552-01
    #783552-02
    #783884-01
    #783884-02
    #785054-v1
    788406
    #788639-25
    790322
    #795476-01
    #797321-01
    #800993-01
    767018
)

EXTRA_ARGS=("$@")

for mouse_id in "${MICE[@]}"; do
    echo "===== Running mouse ${mouse_id} ====="
    python -u run_capsule.py --mouse-id "$mouse_id" "${EXTRA_ARGS[@]}" || {
        echo "FAILED: mouse ${mouse_id}"
        continue
    }
done

echo "===== Done ====="
