# Plot Config Refactor: Function-in-Config Pattern

## Motivation

`run_capsule.py` currently dispatches to different plot functions via an `if plot_type == ...` chain. As new plot types are added, this chain grows and changes must be made in two places (`plot_configs.py` + `run_capsule.py`). Since the config already imports non-serializable objects (`constants.Z1_CHANNEL_CMAP_SOFT`), the "pure data" principle is already broken — moving function references into the config costs nothing and buys a generic runner.

## Proposed Change

### `plot_configs.py`

Add a `plot_fn` key to each spec pointing directly to the callable. For plot functions that need runtime values (e.g. `title_prefix` derived from `mouse_id`), declare them in an optional `runtime_kwargs` list.

```python
import aind_hcr_qc.constants as constants
from aind_hcr_qc.viz.intergrated_datasets import plot_intensity_violins
from aind_hcr_qc.viz.spectral_unmixing import plot_channel_intensity_histograms_by_round

SPOTS_PLOTS = [
    {
        "plot_type": "spots_intensity_violins_round_chan",
        "plot_fn": plot_intensity_violins,
        "plot_kwargs": {
            "intensity_threshold": 25.0,
            "order": "round_chan",
            "n_sample": 25_000,
            "save": False, "show": True, "close": False,
        },
    },
    {
        "plot_type": "spots_intensity_violins_alpha_order",
        "plot_fn": plot_intensity_violins,
        "plot_kwargs": {
            "intensity_threshold": 25.0,
            "order": "alpha",
            "n_sample": 25_000,
            "save": False, "show": True, "close": False,
        },
    },
    {
        "plot_type": "spots_intensity_hist_log",
        "plot_fn": plot_channel_intensity_histograms_by_round,
        "plot_kwargs": {
            "channel_col": "chan",
            "round_col": "round_key",
            "cmap": constants.Z1_CHANNEL_CMAP_SOFT,
            "bins": 200,
            "log_scale": True,
            "xlim": (1, 4.5),
        },
        "runtime_kwargs": ["title_prefix"],  # injected by runner at call time
    },
]
```

### `run_capsule.py` — spots loop

The loop becomes fully generic. `plot_fn` is called with `spots_df`, static `plot_kwargs`, and any resolved `runtime_kwargs`. Because `plot_intensity_violins` returns `None` (uses `plt.gcf()` internally), a fallback is needed.

```python
_RUNTIME_RESOLVERS = {
    "title_prefix": lambda mouse_id: f"{mouse_id} - ",
}

for spec in plots_to_run:
    runtime = {
        k: _RUNTIME_RESOLVERS[k](mouse_id)
        for k in spec.get("runtime_kwargs", [])
    }
    fig = spec["plot_fn"](spots_df, **spec["plot_kwargs"], **runtime)
    if fig is None:  # some functions use plt.gcf() internally
        fig = plt.gcf()
    _upload_and_close(
        fig, bucket, mouse_id,
        spec["plot_type"], spec["plot_kwargs"], source_assets,
    )
```

The import of `plot_channel_intensity_histograms_by_round` and the `if plot_type == ...` block in `run_capsule.py` are removed entirely.

## Adding a New Plot (after refactor)

Changes required: **`plot_configs.py` only**.

1. Import the new function at the top of `plot_configs.py`.
2. Append a new dict to the relevant `*_PLOTS` list with `plot_fn`, `plot_kwargs`, and optionally `runtime_kwargs`.
3. If a new runtime value is needed that isn't already in `_RUNTIME_RESOLVERS`, add it there.

## Notes

- `runtime_kwargs` is optional — omitting it means the function is called with only `plot_kwargs`.
- `_RUNTIME_RESOLVERS` is a small lookup dict; it grows only when a genuinely new kind of runtime value is needed (rare).
- The same pattern applies to `TAXONOMY_PLOTS` and any future categories.
