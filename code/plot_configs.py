"""Plot configuration for QC pipeline.

Each category groups plots that share the same data-loading step.
Every entry has:
    plot_type : str   – S3 key / filename stem
    plot_kwargs : dict – keyword arguments forwarded to the plotting function
"""

import aind_hcr_qc.constants as constants

# ---------------------------------------------------------------------------
# Spots plots  (data: pw_dataset.load_all_rounds_spots_mp)
# ---------------------------------------------------------------------------

SPOTS_PLOTS = [
    {
        "plot_type": "spots_intensity_violins_round_chan",
        "plot_kwargs": {
            "intensity_threshold": 25.0,
            "order": "round_chan",
            "n_sample": 25_000,
        },
    },
    {
        "plot_type": "spots_intensity_violins_alpha_order",
        "plot_kwargs": {
            "intensity_threshold": 25.0,
            "order": "alpha",
            "n_sample": 25_000,
        },
    },
    {
        "plot_type": "spots_intensity_hist_log",
        "plot_kwargs": {
            "channel_col": "unmixed_chan",
            "round_col": "round",
            "cmap": constants.Z1_CHANNEL_CMAP_SOFT,
            "bins": 200,
            "log_scale": True,
            "xlim": (1, 4.5),
        },
    },
]

# ---------------------------------------------------------------------------
# Taxonomy plots  (data: dataset.get_cell_info + load_taxonomy_cell_types)
# ---------------------------------------------------------------------------

TAXONOMY_PLOTS = [
    {
        "plot_type": f"taxonomy_all_{cluster_key.split('_')[0]}_centroids_{orientation}",
        "plot_kwargs": {
            "cluster_key": cluster_key,
            "orientation": orientation,
            "n_cols": 6,
            "panel_size": 3,
            "invert_z": True,
        },
    }
    for orientation in ["ZX", "XY"]
    for cluster_key in ["class_name", "subclass_name", "supertype_name"]
]

# ---------------------------------------------------------------------------
# CXG plots  (data: pw_dataset.load_aggregated_cxg)
# ---------------------------------------------------------------------------

CXG_PLOTS = [
    {
        "plot_type": "spots_count_pairplot_inhibitory",
        "plot_kwargs": {
            "genes_to_plot": ["Gad2", "Sst", "Pvalb", "Vip", "Npy"],
            "n_sample": 40_000,
        },
    },
    {
        "plot_type": "spots_count_pairplot_all",
        "plot_kwargs": {
            "n_sample": 40_000,
        },
    },
]
