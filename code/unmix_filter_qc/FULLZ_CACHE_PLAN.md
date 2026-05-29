# Full-Z Cell Crop Cache Build Plan

## Overview
Build a cache of full-Z (complete axial depth) per-cell crops to avoid repeated expensive zarr/segmentation mask loads during batch metric figure generation and analysis.

**Status:** Script complete, ready for deployment

---

## Phase 1: Script Validation ✅ DONE
- [x] `build_fullz_cell_crop_cache.py` implemented with:
  - Full-Z per-cell spatial cropping
  - Memoized zarr/mask loads for batch efficiency
  - Blosc compression (.npz format)
  - Storage estimation utilities
  - CLI with configurable padding and chunk-shape

---

## Phase 2: Build Cache for Target Cells

### Step 1: Estimate Storage Requirements
```bash
cd /root/capsule/code/unmix_filter_qc
python build_fullz_cell_crop_cache.py \
  --mouse-id 782149 \
  --round-key R5 \
  --cell-ids <CELL_LIST> \
  --estimate-only
```
- Output: Storage needed for all cells
- Decision: Accept or reduce cell count / padding

### Step 2: Run Cache Builder
```bash
python build_fullz_cell_crop_cache.py \
  --mouse-id 782149 \
  --round-key R5 \
  --cell-ids <CELL_LIST> \
  --output-dir /root/capsule/scratch/fullz_cache_r5 \
  --chunk-shape 300 300 300 \
  --plot-buffer 150 \
  --pyramid-level 0
```

**Key Parameters:**
- `--chunk-shape`: Fixed search window for centroid-based cropping. If cells fail to be found, increase this value.
- `--plot-buffer`: Padding applied after bbox detection (xy and z separately)
- `--pyramid-level`: 0 = full resolution; higher = downsampled


### Step 3: Handle Centroid Errors (if any)
**Known Issue:** If a cell's true centroid is >100 voxels away from spot-based guess, the fixed chunk search may miss it.

**Solution:**
- Increase `--chunk-shape` (e.g., from 200 to 300 or 400)
- Or exclude problematic cells and investigate centroid accuracy separately

**Example recovery:**
```bash
python build_fullz_cell_crop_cache.py \
  --mouse-id 782149 \
  --round-key R5 \
  --cell-ids 35358 35359 \
  --output-dir /root/capsule/scratch/fullz_cache_r5 \
  --chunk-shape 400 400 400 \
  --plot-buffer 150
```

---

## Phase 3: Validate Cache

### Run Interactive Viewer
Open and execute cells in `test_crop_cache.ipynb`:
1. Cache auto-discovery via dropdown selector
2. Z-slider for manual inspection
3. Autoscale or fixed intensity bounds
4. Cell outline overlay toggle

**Expected Output:**
- Confirms image data is present for all channels
- Visual check for proper cell cropping (not truncated)
- Spot/background visibility for quality assessment

---

## Phase 4: Use Cache in Batch Analysis

### Option A: Direct Cache Usage (Not Yet Implemented)
Modify plotting functions to load from `.npz` instead of live zarr (faster):
```python
# Pseudo-code (future work)
cached_data = load_fullz_cache(cell_id, round_key, cache_dir)
fig = plot_from_cache(cached_data, metric_col)
```

### Option B: Current Approach (Already Implemented)
Batch export figures with memoized *live* zarr loads:
```python
batch_paths = unmix_qc_utils.batch_save_single_cell_unmixing_mg2(
    m_cell, u_cell,
    cell_id=35357,
    round_key='R5',
    dataset=dataset,
    chan_order=CHAN_ORDER,
    chan_colors=CHAN_COLORS,
    metric_cols=['r', 'd_assign', 'intensity_norm'],
    output_dir=OUTPUT_DIR / 'single_cell_metric_batch'
)
```
- Avoids repeated zarr/mask opens *within* batch loop
- Cache file serves as fallback reference or alternative storage

---

## Storage Expectations

**Per-Cell Footprint:**
- Full-Z source (5 channels, ~80 Z planes, uint16): ~0.78 MB uncompressed
- With segmentation + outlines: ~1.5 MB uncompressed
- After Blosc compression: **~12 MB** typical (.npz file)

**For 100 cells:** ~1.2 GB disk space  
**For 1000 cells:** ~12 GB disk space

**Optimization Options (if needed later):**
1. Drop full segmentation; keep only cell outline
2. Cache fewer channels (e.g., only target channel)
3. Tighten xy padding (if 50 voxels is excessive)
4. Switch to Zarr format for lazy/partial reads

---

## Next Steps After Cache Build

1. **Validate Integrity:** Run test_crop_cache.ipynb on representative cells
2. **Archive:** Move cache to persistent storage or S3 if needed
3. **Document Cell IDs:** Keep manifest of which cells are cached (already in JSON)
4. **Batch Export:** Run batch_save_single_cell_unmixing_mg2() on all cells with metrics of interest

---

## File Locations

| File | Purpose |
|------|---------|
| `build_fullz_cell_crop_cache.py` | Cache builder CLI script |
| `test_crop_cache.ipynb` | Interactive validator & explorer |
| `unmix_qc_utils.py` | Contains batch export helper |
| `/root/capsule/scratch/fullz_cache_r5/` | Cache output directory |

---

## Timeline Estimate

- **Build cache (100 cells):** ~10–30 min (depends on zarr I/O and chunk errors)
- **Validate (interactive):** ~5 min
- **Batch export (5 metrics × 100 cells):** ~5–15 min
- **Total:** ~20–50 min start-to-finish

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Cell id X not found` | Centroid outside search window | Increase `--chunk-shape` to 300–400 |
| `FileNotFoundError: zarr` | Dataset path wrong | Verify `--mouse-id` and `--round-key` |
| `MemoryError` | Too many cells or high res | Reduce `--cell-ids` count or use `--pyramid-level 1` |
| Cache file huge (>50 MB) | Excessive padding or channels | Reduce `--plot-buffer` or cache fewer channels (future) |
