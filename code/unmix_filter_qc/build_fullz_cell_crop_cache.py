"""Build and manage full-Z per-cell crop caches for unmixing/QC workflows.

This script creates reusable per-cell cache artifacts that avoid repeated reads from
large zarr volumes during downstream plotting and analysis.

What is cached per cell
-----------------------
- Full-Z cropped image stack for selected channels: ``images`` with shape ``(C, Z, Y, X)``
- Segmentation labels: ``segmentation`` with shape ``(Z, Y, X)``
- Binary segmentation outlines (all labels): ``mask_outline`` with shape ``(Z, Y, X)``
- Binary outlines for target cell only: ``cell_outline`` with shape ``(Z, Y, X)``
- Spatial metadata: ``origin_zyx`` for mapping crop coordinates back to global volume

Each cell is written as a compressed ``.npz`` file so it can be reused by any plotting
or ML workflow without re-reading full-resolution source zarr arrays.

Example
-------
python build_fullz_cell_crop_cache.py \
  --mouse-id 782149 \
  --round-key R5 \
  --cell-ids 35357 35358 35359 \
  --output-dir /root/capsule/scratch/fullz_cache_r5 \
  --plot-buffer 50
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from aind_hcr_data_loader import get_hcr_dataset_pairwise
from aind_hcr_qc.io.zarr_data import get_mask_outlines


DEFAULT_CHUNK_SHAPE = (200, 200, 200)
DEFAULT_PLOT_BUFFER_ZYX = (5, 50, 50)


@dataclass
class CellCacheRecord:
    """Summary metadata for one cached cell."""

    cell_id: int
    round_key: str
    channels: list[str]
    image_shape_czyx: tuple[int, int, int, int]
    segmentation_shape_zyx: tuple[int, int, int]
    origin_zyx: tuple[int, int, int]
    uncompressed_bytes: int
    compressed_file_bytes: int
    cache_file: str


def _to_int_cell_ids(cell_ids: Iterable[int | str]) -> list[int]:
    """Normalize an iterable of cell ids into sorted unique integers."""
    unique = {int(c) for c in cell_ids}
    return sorted(unique)


def _select_channels(dataset, round_key: str, channels: Sequence[str] | None) -> list[str]:
    """Return channel list to cache in numeric sort order."""
    if channels is None:
        channels = dataset.get_channels(round_key)
    return sorted((str(c) for c in channels), key=lambda x: int(x))


def _get_cell_centroid_array(dataset, round_key: str) -> np.ndarray:
    """Return centroid array with columns [z_centroid, y_centroid, x_centroid, cell_id]."""
    cell_info_df = dataset.rounds[round_key].get_cell_info(source="mixed_cxg")
    return cell_info_df[["z_centroid", "y_centroid", "x_centroid", "cell_id"]].to_numpy()


def _compute_bbox_full_z(
    segmentation_zarr,
    cell_centroids: np.ndarray,
    cell_id: int,
    *,
    plot_buffer: int | tuple[int, int, int],
    chunk_shape: tuple[int, int, int] = DEFAULT_CHUNK_SHAPE,
) -> tuple[int, int, int, int, int, int]:
    """Compute global buffered bbox [zmin, ymin, xmin, zmax, ymax, xmax] for one cell.

    The method mirrors the chunk-first approach used in plotting helpers:
    1. Use centroid to read a bounded segmentation chunk.
    2. Find the cell's exact local bbox inside that chunk.
    3. Expand with ``plot_buffer`` and clip to global volume bounds.
    """
    idx = np.where(cell_centroids[:, -1] == cell_id)[0]
    if len(idx) == 0:
        raise ValueError(f"Cell id {cell_id} not found in centroid table")

    centroid = cell_centroids[idx[0], :-1].astype(int)
    sz, sy, sx = (centroid - np.array(chunk_shape) / 2).astype(int)

    seg_chunk = segmentation_zarr[
        0,
        0,
        sz : sz + chunk_shape[0],
        sy : sy + chunk_shape[1],
        sx : sx + chunk_shape[2],
    ]

    zz, yy, xx = np.where(seg_chunk == cell_id)
    if len(zz) == 0:
        raise ValueError(
            f"Cell id {cell_id} not found inside segmentation chunk; "
            "increase chunk_shape or verify centroid table."
        )

    bbox_global = (
        sz + int(zz.min()),
        sy + int(yy.min()),
        sx + int(xx.min()),
        sz + int(zz.max()),
        sy + int(yy.max()),
        sx + int(xx.max()),
    )

    gshape = np.array(segmentation_zarr.shape[2:])
    if isinstance(plot_buffer, int):
        buffer_zyx = np.array([plot_buffer, plot_buffer, plot_buffer], dtype=int)
    else:
        if len(plot_buffer) != 3:
            raise ValueError(
                f"plot_buffer must be int or 3-tuple (z,y,x), got: {plot_buffer}"
            )
        buffer_zyx = np.array(plot_buffer, dtype=int)

    zmin, ymin, xmin = np.maximum(np.array(bbox_global[:3]) - buffer_zyx, 0)
    zmax, ymax, xmax = np.minimum(np.array(bbox_global[3:]) + buffer_zyx, gshape)

    return int(zmin), int(ymin), int(xmin), int(zmax), int(ymax), int(xmax)


def _estimate_cell_uncompressed_bytes(
    image_shape_czyx: tuple[int, int, int, int],
    segmentation_shape_zyx: tuple[int, int, int],
    image_dtype: np.dtype,
    segmentation_dtype: np.dtype,
    mask_dtype: np.dtype,
) -> int:
    """Estimate total bytes for arrays stored in one cache file (before compression)."""
    c, z, y, x = image_shape_czyx
    seg_voxels = int(np.prod(segmentation_shape_zyx))
    img_voxels = int(c * z * y * x)

    img_bytes = img_voxels * np.dtype(image_dtype).itemsize
    seg_bytes = seg_voxels * np.dtype(segmentation_dtype).itemsize
    # Two mask volumes are stored: full-mask outlines + target-cell outlines.
    mask_bytes = 2 * seg_voxels * np.dtype(mask_dtype).itemsize
    # Small metadata arrays are ignored as negligible.
    return int(img_bytes + seg_bytes + mask_bytes)


def estimate_cache_bytes_for_cells(
    *,
    num_cells: int,
    avg_shape_zyx: tuple[int, int, int],
    num_channels: int,
    image_dtype: np.dtype = np.uint16,
    segmentation_dtype: np.dtype = np.uint32,
    mask_dtype: np.dtype = np.uint8,
) -> int:
    """Return uncompressed byte estimate for a batch of full-Z caches.

    This is useful for planning storage before writing the cache.
    """
    z, y, x = avg_shape_zyx
    per_cell = _estimate_cell_uncompressed_bytes(
        image_shape_czyx=(num_channels, z, y, x),
        segmentation_shape_zyx=(z, y, x),
        image_dtype=image_dtype,
        segmentation_dtype=segmentation_dtype,
        mask_dtype=mask_dtype,
    )
    return int(num_cells * per_cell)


def build_full_z_cell_cache(
    dataset,
    *,
    round_key: str,
    cell_ids: Iterable[int | str],
    output_dir: str | Path,
    channels: Sequence[str] | None = None,
    pyramid_level: str = "0",
    plot_buffer: int | tuple[int, int, int] = DEFAULT_PLOT_BUFFER_ZYX,
    chunk_shape: tuple[int, int, int] = DEFAULT_CHUNK_SHAPE,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict:
    """Build compressed full-Z crop caches for selected cells.

    Parameters
    ----------
    dataset
        Loaded HCR dataset object.
    round_key
        Round identifier (for example: ``"R5"``).
    cell_ids
        Iterable of cell ids to cache.
    output_dir
        Destination directory for ``.npz`` cache files and manifest.
    channels
        Optional subset of channels to cache. If None, all available channels are used.
    pyramid_level
        Pyramid level passed through to zarr loading.
    plot_buffer
        Voxels added around the segmented-cell bbox. Accepts either:
        - single int (same padding for z/y/x)
        - tuple ``(z_pad, y_pad, x_pad)``
        Default is ``(5, 50, 50)``.
    chunk_shape
        Local chunk size used during bbox detection around centroid.
    overwrite
        If False, existing cache files are skipped.
    verbose
        If True, print progress lines.

    Returns
    -------
    dict
        Run summary with file locations, byte totals, and per-cell records.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_ids_int = _to_int_cell_ids(cell_ids)
    channels_sorted = _select_channels(dataset, round_key, channels)

    if verbose:
        print(f"Round: {round_key}")
        print(f"Cells requested: {len(cell_ids_int)}")
        print(f"Channels cached: {channels_sorted}")
        print(f"Output dir: {output_dir}")

    segmentation_zarr = dataset.load_segmentation_mask(round_key, pyramid_level)
    cell_centroids = _get_cell_centroid_array(dataset, round_key)

    # Open channel zarr handles once and reuse for all cells.
    channel_zarr = {
        ch: dataset.load_zarr_channel(round_key, ch, data_type="fused", pyramid_level=pyramid_level)
        for ch in channels_sorted
    }

    records: list[CellCacheRecord] = []

    for i, cell_id in enumerate(cell_ids_int, start=1):
        cache_name = f"cell_{cell_id}_round_{round_key}_fullz.npz"
        cache_path = output_dir / cache_name

        if cache_path.exists() and not overwrite:
            if verbose:
                print(f"[{i}/{len(cell_ids_int)}] skip existing: {cache_path.name}")
            continue

        zmin, ymin, xmin, zmax, ymax, xmax = _compute_bbox_full_z(
            segmentation_zarr,
            cell_centroids,
            cell_id,
            plot_buffer=plot_buffer,
            chunk_shape=chunk_shape,
        )

        seg_crop = np.asarray(segmentation_zarr[0, 0, zmin:zmax, ymin:ymax, xmin:xmax])

        # Reuse existing outline utility, then convert to compact binary masks.
        masks_only, cell_mask_only = get_mask_outlines(seg_crop, cell_id)
        mask_outline = np.isfinite(masks_only).astype(np.uint8)
        cell_outline = np.isfinite(cell_mask_only).astype(np.uint8)

        image_crops = []
        for ch in channels_sorted:
            ch_crop = np.asarray(channel_zarr[ch][0, 0, zmin:zmax, ymin:ymax, xmin:xmax])
            image_crops.append(ch_crop)
        images = np.stack(image_crops, axis=0)

        np.savez_compressed(
            cache_path,
            images=images,
            channels=np.asarray(channels_sorted, dtype="U16"),
            segmentation=seg_crop,
            mask_outline=mask_outline,
            cell_outline=cell_outline,
            origin_zyx=np.asarray((zmin, ymin, xmin), dtype=np.int32),
            cell_id=np.asarray(cell_id, dtype=np.int64),
            round_key=np.asarray(round_key),
            pyramid_level=np.asarray(str(pyramid_level)),
        )

        image_shape = tuple(int(v) for v in images.shape)
        seg_shape = tuple(int(v) for v in seg_crop.shape)
        uncompressed = _estimate_cell_uncompressed_bytes(
            image_shape_czyx=image_shape,
            segmentation_shape_zyx=seg_shape,
            image_dtype=images.dtype,
            segmentation_dtype=seg_crop.dtype,
            mask_dtype=mask_outline.dtype,
        )

        record = CellCacheRecord(
            cell_id=int(cell_id),
            round_key=str(round_key),
            channels=list(channels_sorted),
            image_shape_czyx=image_shape,
            segmentation_shape_zyx=seg_shape,
            origin_zyx=(int(zmin), int(ymin), int(xmin)),
            uncompressed_bytes=int(uncompressed),
            compressed_file_bytes=int(cache_path.stat().st_size),
            cache_file=cache_path.name,
        )
        records.append(record)

        if verbose:
            print(
                f"[{i}/{len(cell_ids_int)}] cached cell {cell_id}: "
                f"shape={image_shape} file={cache_path.name} "
                f"size={record.compressed_file_bytes / (1024**2):.2f} MB"
            )

    summary = {
        "round_key": round_key,
        "num_requested_cells": len(cell_ids_int),
        "num_cached_cells": len(records),
        "channels": channels_sorted,
        "output_dir": str(output_dir),
        "total_uncompressed_bytes": int(sum(r.uncompressed_bytes for r in records)),
        "total_compressed_bytes": int(sum(r.compressed_file_bytes for r in records)),
        "records": [asdict(r) for r in records],
    }

    manifest_path = output_dir / f"manifest_round_{round_key}_fullz.json"
    manifest_path.write_text(json.dumps(summary, indent=2))

    if verbose:
        total_mb = summary["total_compressed_bytes"] / (1024**2)
        print(f"Wrote manifest: {manifest_path}")
        print(f"Compressed total: {total_mb:.2f} MB")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build full-Z per-cell crop cache files.")

    parser.add_argument("--mouse-id", required=True, help="Mouse id passed to get_hcr_dataset_pairwise")
    parser.add_argument("--round-key", required=True, help="Round key to cache (example: R5)")
    parser.add_argument(
        "--cell-ids",
        required=True,
        nargs="+",
        type=int,
        help="List of cell ids to cache",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for cache .npz files")
    parser.add_argument(
        "--data-dir",
        default="/root/capsule/data",
        help="Base data directory used by get_hcr_dataset_pairwise",
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="Optional channel subset (example: 488 514 561 594 638)",
    )
    parser.add_argument("--pyramid-level", default="0", help="Image pyramid level")
    parser.add_argument(
        "--plot-buffer",
        nargs="*",
        type=int,
        default=list(DEFAULT_PLOT_BUFFER_ZYX),
        help=(
            "Buffer around cell bbox. Provide either one int for isotropic padding "
            "or three ints for z y x padding. Default: 5 50 50"
        ),
    )
    parser.add_argument(
        "--chunk-shape",
        nargs=3,
        type=int,
        default=list(DEFAULT_CHUNK_SHAPE),
        metavar=("Z", "Y", "X"),
        help="Chunk shape used during bbox detection",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print rough uncompressed estimate and exit without writing files",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    dataset, _, _ = get_hcr_dataset_pairwise(
        mouse_id=str(args.mouse_id),
        data_dir=Path(args.data_dir),
        load_spots=False,
        return_removed=False,
        coreg_cells_only=True,
    )

    channels_sorted = _select_channels(dataset, args.round_key, args.channels)
    num_cells = len(_to_int_cell_ids(args.cell_ids))

    # Conservative default estimate when no sample geometry is available.
    # This can differ from final cache sizes because each cell has unique Z/Y/X span.
    est_bytes = estimate_cache_bytes_for_cells(
        num_cells=num_cells,
        avg_shape_zyx=(80, 101, 101),
        num_channels=len(channels_sorted),
        image_dtype=np.uint16,
        segmentation_dtype=np.uint32,
        mask_dtype=np.uint8,
    )
    print(
        "Rough uncompressed estimate "
        f"(assuming avg shape 80x101x101): {est_bytes / (1024**3):.2f} GB"
    )

    if args.estimate_only:
        return

    if len(args.plot_buffer) == 1:
        plot_buffer: int | tuple[int, int, int] = int(args.plot_buffer[0])
    elif len(args.plot_buffer) == 3:
        plot_buffer = (int(args.plot_buffer[0]), int(args.plot_buffer[1]), int(args.plot_buffer[2]))
    else:
        raise ValueError("--plot-buffer expects either one int or three ints (z y x)")

    summary = build_full_z_cell_cache(
        dataset,
        round_key=args.round_key,
        cell_ids=args.cell_ids,
        output_dir=args.output_dir,
        channels=args.channels,
        pyramid_level=args.pyramid_level,
        plot_buffer=plot_buffer,
        chunk_shape=tuple(args.chunk_shape),
        overwrite=args.overwrite,
        verbose=True,
    )

    total_gb = summary["total_compressed_bytes"] / (1024**3)
    print(f"Done. Compressed cache size: {total_gb:.2f} GB")


if __name__ == "__main__":
    main()