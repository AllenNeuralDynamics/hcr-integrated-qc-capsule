#!/usr/bin/env bash
# Run the atlas-comparison pipeline for every mouse that has a pairwise_unmixing asset.
#
# Usage:
#   bash run_all_mice.sh
#   bash run_all_mice.sh --dpi 200 --label-level subclass
#
# Any extra flags are forwarded to run_atlas_compare.py unchanged.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MICE=(
    788406
    790322
)

# 755252
# 767022
# 782149

echo "========================================"
echo "Mice to process (${#MICE[@]}): ${MICE[*]}"
echo "Extra args: ${*:-none}"
echo "========================================"

FAILED=()

for MOUSE_ID in "${MICE[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Starting mouse: $MOUSE_ID"
    echo "----------------------------------------"

    if python "$SCRIPT_DIR/run_atlas_compare.py" --mouse-id "$MOUSE_ID" "$@"; then
        echo "✓ $MOUSE_ID done"
    else
        echo "✗ $MOUSE_ID FAILED (exit $?)"
        FAILED+=("$MOUSE_ID")
    fi
done

echo ""
echo "========================================"
echo "All mice processed."
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All succeeded."
else
    echo "FAILED mice: ${FAILED[*]}"
    exit 1
fi
echo "========================================"
