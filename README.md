# HCR QC Capsule

Generates and uploads QC plots for HCR mouse datasets to S3. Each plot is stored with a JSON sidecar recording the `plot_kwargs` and source asset provenance.

---

## Repository layout

```
code/
    run_capsule.py      # entry point — load data, run plots, upload to S3
    run_all_mice.sh     # batch runner across a list of mouse IDs
    plot_configs.py     # plot registry: all plot types and their parameters
    delete_plot.py      # safely delete a plot type from S3 (dry-run by default)
data/                   # mounted dataset assets (pairwise unmixing, etc.)
results/                # local output directory (unused by default; plots go to S3)
```

---

## Running plots for a single mouse

```bash
cd /root/capsule/code

# Run all plots, skipping any already on S3
python run_capsule.py --mouse-id 755252

# Re-generate and re-upload all plots even if they already exist on S3
python run_capsule.py --mouse-id 755252 --overwrite

# Run only specific plot type(s) — omitting --plot-type considers all plots
python run_capsule.py --mouse-id 755252 --plot-type spots_intensity_hist_log
python run_capsule.py --mouse-id 755252 --plot-type spots_intensity_hist_log spots_intensity_violins_alpha_order

# Force-regenerate specific plot type(s)
python run_capsule.py --mouse-id 755252 --plot-type spots_intensity_hist_log --overwrite

# Target a non-default S3 bucket
python run_capsule.py --mouse-id 755252 --bucket my-other-bucket
```

### When to use `--overwrite`

| Situation | Use `--overwrite-all`? |
|---|---|
| First time running a mouse | No — default behaviour uploads everything |
| Plot already exists and you haven't changed anything | No — skip saves time and cost |
| You changed `plot_kwargs` for an existing plot type | No — use `--plot-type <name> --overwrite` to target just that plot |
| You fixed a bug in the underlying plot function | **Yes** (or `--plot-type <name> --overwrite` if only one function changed) |
| You added a brand-new `plot_type` | No — it doesn't exist on S3 yet so it runs automatically |

---

## Running all mice (batch)

`run_all_mice.sh` iterates over a curated list of mouse IDs and calls `run_capsule.py` for each one. A failure on one mouse is logged and the script continues to the next.

```bash
cd /root/capsule/code

# Run all active mice (skips plots that already exist on S3)
bash run_all_mice.sh

# Re-generate all plots for all mice
bash run_all_mice.sh --overwrite
```

Any extra arguments after the script name are forwarded directly to `run_capsule.py`, so `--bucket` and `--plot-type` also work:

```bash
bash run_all_mice.sh --bucket my-other-bucket
bash run_all_mice.sh --plot-type spots_intensity_hist_log --overwrite
```

### Enabling / disabling mice in the batch

Open `run_all_mice.sh` and comment or uncomment mouse IDs in the `MICE=(...)` array:

```bash
MICE=(
    755252        # active — will run
    #749315       # commented out — will be skipped
)
```

---

## Adding a new plot

All plot configuration lives in `plot_configs.py`. `run_capsule.py` never needs to change for a new plot within an existing data category.

### 1. Choose the right category

| List | Data loaded | Example plot types |
|---|---|---|
| `SPOTS_PLOTS` | `pw_dataset.load_all_rounds_spots_mp` | intensity violins, intensity histograms |
| `TAXONOMY_PLOTS` | `pw_dataset.get_cell_info` + `load_taxonomy_cell_types` | centroid scatters |

### 2. Add an entry to the list

```python
# plot_configs.py

SPOTS_PLOTS = [
    ...
    {
        "plot_type": "my_new_plot_name",   # becomes the S3 key / filename stem
        "plot_kwargs": {
            "some_param": 42,
        },
    },
]
```

- `plot_type` must be **unique** across all plots — it is used as the S3 key and to check whether the plot already exists.
- `plot_kwargs` are passed as keyword arguments to the relevant plot function in `run_capsule.py`.

### 3. Wire up the function (if it's a new function)

If the new plot calls a function that isn't already dispatched in `run_spots_plots` or `run_taxonomy_plots`, add an `elif` branch for the new `plot_type` in the appropriate runner inside `run_capsule.py`.

> **Note:** A future refactor ([`plot_configs_refactor.md`](plot_configs_refactor.md)) would store the function reference directly in the config, eliminating this step entirely.

### 4. Test on one mouse first

```bash
python run_capsule.py --mouse-id 755252
```

Only the new plot will run (all others are already on S3). Use `--overwrite` only if you also want to regenerate existing plots.

---

## Deleting a plot from S3

`delete_plot.py` removes a plot's PNG and JSON sidecar from S3. It is **dry-run by default** — it will print what would be deleted without touching anything until you pass `--confirm`.

```bash
cd /root/capsule/code

# Preview what would be deleted for one mouse
python delete_plot.py --mouse-id 755252 --plot-type spots_intensity_violins_round_chan

# Preview across multiple mice at once
python delete_plot.py --mouse-id 755252 767022 782149 --plot-type spots_intensity_violins_round_chan

# Actually delete (prompts "Type 'yes' to proceed")
python delete_plot.py --mouse-id 755252 767022 782149 --plot-type spots_intensity_violins_round_chan --confirm

# Target a non-default bucket
python delete_plot.py --mouse-id 755252 --plot-type spots_intensity_violins_round_chan --bucket my-other-bucket --confirm
```

After deleting, the plot will no longer show as existing on S3, so the next `run_capsule.py` run will regenerate and re-upload it without needing `--overwrite`.
