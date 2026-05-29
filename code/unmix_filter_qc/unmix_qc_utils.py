"""Utility helpers for the unmixing QC notebook."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Iterable, Sequence
import re

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from aind_hcr_qc.viz.single_cell_unmixing import fig_single_cell_unmixing_mg2


DEFAULT_CMAPS_MPL: Sequence[str] = (
	"viridis",
	"plasma",
	"inferno",
	"magma",
	"cividis",
	"turbo",
	"coolwarm",
)

DEFAULT_CMAPS_PLOTLY: Sequence[str] = (
	"Viridis",
	"Plasma",
	"Inferno",
	"Magma",
	"Cividis",
	"Turbo",
	"RdBu",
)


def _sanitize_for_filename(value: object) -> str:
	"""Return a filesystem-friendly string for output filenames."""
	text = str(value).strip()
	text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
	text = re.sub(r"_+", "_", text)
	return text.strip("_.") or "value"


@contextmanager
def _temporarily_cache_dataset_loads(dataset):
	"""Temporarily memoize heavy dataset image/mask loaders for one batch run.

	This caches calls to ``dataset.load_zarr_channel`` and
	``dataset.load_segmentation_mask`` on the dataset instance passed in, then
	restores the original methods on exit.
	"""
	if dataset is None:
		yield
		return

	orig_load_zarr_channel = dataset.load_zarr_channel
	orig_load_segmentation_mask = dataset.load_segmentation_mask
	zarr_cache = {}
	seg_cache = {}

	def _freeze_kwargs(kwargs):
		return tuple(sorted(kwargs.items()))

	def cached_load_zarr_channel(*args, **kwargs):
		key = (args, _freeze_kwargs(kwargs))
		if key not in zarr_cache:
			zarr_cache[key] = orig_load_zarr_channel(*args, **kwargs)
		return zarr_cache[key]

	def cached_load_segmentation_mask(*args, **kwargs):
		key = (args, _freeze_kwargs(kwargs))
		if key not in seg_cache:
			seg_cache[key] = orig_load_segmentation_mask(*args, **kwargs)
		return seg_cache[key]

	dataset.load_zarr_channel = cached_load_zarr_channel
	dataset.load_segmentation_mask = cached_load_segmentation_mask
	try:
		yield
	finally:
		dataset.load_zarr_channel = orig_load_zarr_channel
		dataset.load_segmentation_mask = orig_load_segmentation_mask


@contextmanager
def _temporarily_set_logger_level(logger_name: str, level: int):
	"""Temporarily override a logger level and restore it on exit."""
	logger = logging.getLogger(logger_name)
	old_level = logger.level
	logger.setLevel(level)
	try:
		yield
	finally:
		logger.setLevel(old_level)


def batch_save_single_cell_unmixing_mg2(
	m_cell,
	u_cell,
	cell_id,
	round_key,
	dataset,
	chan_order,
	chan_colors,
	metric_cols: Iterable[str],
	*,
	output_dir: str | Path,
	filename_prefix: str | None = None,
	file_ext: str = "png",
	dpi: int = 150,
	bbox_inches: str = "tight",
	facecolor: str = "white",
	transparent: bool = False,
	pyramid_level: str = "0",
	img_fixed_vmin: float = 90,
	img_fixed_vmax: float = 1200,
	spot_size: float = 30,
	fast_plot: bool = False,
	top_row_mask_outlines: bool = True,
	cmap: str = "viridis",
	vmin: float | None = None,
	vmax: float | None = None,
	close_figures: bool = True,
	verbose: bool = True,
):
	"""Save one ``fig_single_cell_unmixing_mg2`` image per metric.

	The figure construction remains identical to the interactive single-metric
	path, but expensive dataset loaders are memoized for the duration of the
	batch so repeated zarr and segmentation opens are avoided.

	Parameters
	----------
	m_cell, u_cell
		Spot tables for a single cell and round.
	cell_id, round_key, dataset, chan_order, chan_colors
		Forwarded directly to ``fig_single_cell_unmixing_mg2``.
	metric_cols
		Iterable of metric column names to render.
	output_dir
		Directory where images are written.
	filename_prefix
		Optional prefix for saved files. Defaults to ``cell_<id>_<round>``.
	file_ext
		Image extension such as ``png`` or ``pdf``.
	close_figures
		Close figures after save to avoid memory growth during large batches.

	Returns
	-------
	list[pathlib.Path]
		Saved file paths, in metric order.
	"""
	metrics = list(metric_cols)
	if not metrics:
		raise ValueError("metric_cols must contain at least one metric name")

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	if filename_prefix is None:
		filename_prefix = f"cell_{cell_id}_{round_key}"
	filename_prefix = _sanitize_for_filename(filename_prefix)
	file_ext = file_ext.lstrip(".")

	saved_paths: list[Path] = []
	with _temporarily_cache_dataset_loads(dataset):
		# Suppress noisy Matplotlib limits/aspect warning during batch export.
		with _temporarily_set_logger_level("matplotlib.axes._base", logging.ERROR):
			for metric_col in metrics:
				metric_slug = _sanitize_for_filename(metric_col)
				out_path = output_path / f"{filename_prefix}_metric_{metric_slug}.{file_ext}"

				fig = fig_single_cell_unmixing_mg2(
					m_cell,
					u_cell,
					cell_id=cell_id,
					round_key=round_key,
					chan_order=chan_order,
					chan_colors=chan_colors,
					dataset=dataset,
					pyramid_level=pyramid_level,
					img_fixed_vmin=img_fixed_vmin,
					img_fixed_vmax=img_fixed_vmax,
					spot_size=spot_size,
					fast_plot=fast_plot,
					top_row_mask_outlines=top_row_mask_outlines,
					metric_col=metric_col,
					cmap=cmap,
					vmin=vmin,
					vmax=vmax,
				)
				fig.savefig(
					out_path,
					dpi=dpi,
					bbox_inches=bbox_inches,
					facecolor=facecolor,
					transparent=transparent,
				)
				saved_paths.append(out_path)
				if close_figures:
					plt.close(fig)
				if verbose:
					print(f"Saved {out_path}")

	return saved_paths


def interactive_single_cell_unmixing_mg2(
	m_cell,
	u_cell,
	cell_id,
	round_key,
	dataset,
	chan_order,
	chan_colors,
	*,
	fast_plot: bool = False,
	spot_size: float = 30,
	pyramid_level: str = "0",
	img_fixed_vmin: float = 90,
	img_fixed_vmax: float = 1200,
	metric_options: Iterable[str] = ("r", "dist"),
	cmap_options: Sequence[str] = DEFAULT_CMAPS_MPL,
):
	"""Interactive matplotlib wrapper around ``fig_single_cell_unmixing_mg2``."""

	metric_dropdown = widgets.Dropdown(
		options=list(metric_options),
		value="r" if "r" in metric_options else next(iter(metric_options)),
		description="Metric",
		layout=widgets.Layout(width="220px"),
	)
	cmap_dropdown = widgets.Dropdown(
		options=list(cmap_options),
		value=cmap_options[0],
		description="Cmap",
		layout=widgets.Layout(width="220px"),
	)
	out = widgets.Output()

	def _render(*_):
		with out:
			out.clear_output(wait=True)
			fig = fig_single_cell_unmixing_mg2(
				m_cell,
				u_cell,
				cell_id=cell_id,
				round_key=round_key,
				chan_order=chan_order,
				chan_colors=chan_colors,
				dataset=dataset,
				pyramid_level=pyramid_level,
				img_fixed_vmin=img_fixed_vmin,
				img_fixed_vmax=img_fixed_vmax,
				spot_size=spot_size,
				fast_plot=fast_plot,
				metric_col=metric_dropdown.value,
				cmap=cmap_dropdown.value,
			)
			display(fig)

	metric_dropdown.observe(_render, names="value")
	cmap_dropdown.observe(_render, names="value")

	controls = widgets.HBox([metric_dropdown, cmap_dropdown])
	widget = widgets.VBox([controls, out])

	_render()
	return widget


def _build_single_cell_unmixing_plotly(
	m_cell,
	u_cell,
	cell_id,
	round_key,
	chan_order,
	metric_col: str,
	cmap: str,
	spot_size: float,
	axis_limits: str,
):
	"""Build a Plotly small-multiples figure for single-cell unmixing QC."""
	in_unmixed = set(u_cell["spot_uid"]) if "spot_uid" in u_cell.columns else set()
	removed_by_unmixing = (
		m_cell[~m_cell["spot_uid"].isin(in_unmixed)]
		if "spot_uid" in m_cell.columns
		else m_cell.iloc[0:0]
	)

	if "valid_spot" in u_cell.columns:
		kept_removed_qc = u_cell[u_cell["valid_spot"] == False]
		kept_passed_qc = u_cell[u_cell["valid_spot"] == True]
	else:
		kept_removed_qc = u_cell.iloc[0:0]
		kept_passed_qc = u_cell

	row_specs = [
		("All mixed", m_cell, "chan"),
		("Removed by unmixing", removed_by_unmixing, "chan"),
		("Kept by unmixing", u_cell, "unmixed_chan"),
		("Kept but removed by QC", kept_removed_qc, "unmixed_chan"),
		("Kept and passed QC", kept_passed_qc, "unmixed_chan"),
	]

	if metric_col in u_cell.columns and len(u_cell) > 0:
		vals = u_cell[metric_col].dropna().to_numpy()
		if len(vals):
			cmin = float(np.percentile(vals, 2))
			cmax = float(np.percentile(vals, 98))
		else:
			cmin, cmax = 0.0, 1.0
	else:
		cmin, cmax = 0.0, 1.0

	# Shared spatial limits so every panel uses the same frame.
	x_chunks = []
	y_chunks = []
	for df in (m_cell, u_cell):
		if {"x", "y"}.issubset(df.columns) and len(df) > 0:
			x_chunks.append(df["x"].to_numpy(dtype=float))
			y_chunks.append(df["y"].to_numpy(dtype=float))

	if x_chunks and y_chunks:
		x_all = np.concatenate(x_chunks)
		y_all = np.concatenate(y_chunks)
		x_pad = 10.0
		y_pad = 10.0
		x_range = [float(x_all.min() - x_pad), float(x_all.max() + x_pad)]
		# Reversed order keeps image-style y direction (top-left origin).
		y_range = [float(y_all.max() + y_pad), float(y_all.min() - y_pad)]
	else:
		x_range = [0.0, 1.0]
		y_range = [1.0, 0.0]

	n_rows = len(row_specs)
	n_cols = len(chan_order)
	subplot_titles = []
	for row_title, _, _ in row_specs:
		for ch in chan_order:
			subplot_titles.append(f"{row_title}<br>Ch {ch}")

	fig = make_subplots(
		rows=n_rows,
		cols=n_cols,
		subplot_titles=subplot_titles,
		shared_xaxes="all",
		shared_yaxes="all",
		horizontal_spacing=0.02,
		vertical_spacing=0.06,
	)

	for r_idx, (_, df, chan_col) in enumerate(row_specs, start=1):
		for c_idx, ch in enumerate(chan_order, start=1):
			sub = df[df[chan_col].astype(str) == str(ch)] if chan_col in df.columns else df.iloc[0:0]

			if len(sub) == 0:
				fig.add_trace(
					go.Scattergl(
						x=[],
						y=[],
						mode="markers",
						showlegend=False,
						hoverinfo="skip",
					),
					row=r_idx,
					col=c_idx,
				)
				continue

			has_metric = metric_col in sub.columns
			marker = {
				"size": spot_size,
				"opacity": 0.7,
			}
			if has_metric:
				marker.update(
					{
						"color": sub[metric_col],
						"colorscale": cmap,
						"cmin": cmin,
						"cmax": cmax,
						"showscale": (r_idx == 1 and c_idx == n_cols),
						"colorbar": {
							"title": metric_col,
							"len": 0.25,
							"x": 1.02,
						},
					}
				)
			else:
				marker.update({"color": "gray"})

			fig.add_trace(
				go.Scattergl(
					x=sub["x"],
					y=sub["y"],
					mode="markers",
					marker=marker,
					showlegend=False,
					hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
				),
				row=r_idx,
				col=c_idx,
			)

	if str(axis_limits).lower() == "global":
		fig.update_xaxes(matches="x")
		fig.update_yaxes(matches="y")
		fig.update_xaxes(range=x_range, autorange=False)
		fig.update_yaxes(range=y_range, autorange=False)
		fig.update_yaxes(scaleanchor="x", scaleratio=1)
	else:
		fig.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
	fig.update_layout(
		title=f"Cell {cell_id} | {round_key} | metric={metric_col}",
		height=280 * n_rows,
		width=340 * n_cols,
		template="plotly_white",
		margin={"l": 40, "r": 40, "t": 70, "b": 30},
	)
	return fig


def interactive_single_cell_unmixing_plotly(
	m_cell,
	u_cell,
	cell_id,
	round_key,
	chan_order,
	*,
	spot_size: float = 8,
	axis_limits: str = "global",
	metric_options: Iterable[str] = ("r", "dist", "d_assign_neighbor_ratio_1", "d_assign_neighbor_ratio_2", "dye_line_dist_ratio", "intensity_assigned_chan", "intensity_assigned_chan_norm"),
	cmap_options: Sequence[str] = DEFAULT_CMAPS_PLOTLY,
):
	"""Interactive Plotly version with dropdowns for metric and colormap.

	Parameters
	----------
	axis_limits : str
		"global" for one shared x/y range across all subplots; "auto" for
		per-subplot autoscaling.
	"""
	metric_options = list(metric_options)
	cmap_options = list(cmap_options)

	metric_dropdown = widgets.Dropdown(
		options=metric_options,
		value="r" if "r" in metric_options else metric_options[0],
		description="Metric",
		layout=widgets.Layout(width="220px"),
	)
	cmap_dropdown = widgets.Dropdown(
		options=cmap_options,
		value=cmap_options[0],
		description="Cmap",
		layout=widgets.Layout(width="220px"),
	)
	out = widgets.Output()

	def _render(*_):
		with out:
			out.clear_output(wait=True)
			fig = _build_single_cell_unmixing_plotly(
				m_cell=m_cell,
				u_cell=u_cell,
				cell_id=cell_id,
				round_key=round_key,
				chan_order=chan_order,
				metric_col=metric_dropdown.value,
				cmap=cmap_dropdown.value,
				spot_size=spot_size,
				axis_limits=axis_limits,
			)
			display(fig)

	metric_dropdown.observe(_render, names="value")
	cmap_dropdown.observe(_render, names="value")

	widget = widgets.VBox([widgets.HBox([metric_dropdown, cmap_dropdown]), out])
	_render()
	return widget


def interactive_single_cell_browser_plotly(
	spots,
	chan_order,
	*,
	start_cell_id=None,
	round_key=None,
	cell_ids: Iterable[int] | None = None,
	round_options: Iterable[str] | None = None,
	spot_size: float = 8,
	axis_limits: str = "global",
	metric_options: Iterable[str] = (
		"r",
		"dist",
		"d_assign_neighbor_ratio_1",
		"d_assign_neighbor_ratio_2",
		"dye_line_dist_ratio",
		"intensity_assigned_chan",
		"intensity_assigned_chan_norm",
	),
	cmap_options: Sequence[str] = DEFAULT_CMAPS_PLOTLY,
):
	"""Browse single-cell Plotly QC with notebook controls.

	Provides previous/next navigation, direct cell-id entry, round selection,
	metric/cmap controls, and a small status summary for the current view.
	"""
	required_cols = {"round", "cell_id", "removed", "x", "y"}
	missing_cols = sorted(required_cols - set(spots.columns))
	if missing_cols:
		raise ValueError(f"spots is missing required columns: {missing_cols}")

	metric_options = list(metric_options)
	cmap_options = list(cmap_options)

	def _sorted_unique(values):
		vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
		return sorted(set(int(v) for v in vals))

	spots_df = spots.copy()
	spots_df["round"] = spots_df["round"].astype(str)

	allowed_rounds = [str(r) for r in round_options] if round_options is not None else sorted(spots_df["round"].unique())
	if not allowed_rounds:
		raise ValueError("No rounds available to browse.")

	if round_key is None:
		round_key = allowed_rounds[0]
	else:
		round_key = str(round_key)
		if round_key not in allowed_rounds:
			raise ValueError(f"round_key {round_key!r} is not in available rounds: {allowed_rounds}")

	allowed_cell_ids = _sorted_unique(cell_ids) if cell_ids is not None else _sorted_unique(spots_df["cell_id"].unique())
	if not allowed_cell_ids:
		raise ValueError("No cell_ids available to browse.")

	def _available_cells_for_round(rk):
		mask = (spots_df["round"] == str(rk)) & (spots_df["cell_id"].isin(allowed_cell_ids))
		return _sorted_unique(spots_df.loc[mask, "cell_id"].unique())

	if not _available_cells_for_round(round_key):
		fallback_round = next((rk for rk in allowed_rounds if _available_cells_for_round(rk)), None)
		if fallback_round is None:
			raise ValueError("No cells available for the requested round/cell filters.")
		round_key = fallback_round

	def _nearest_cell_id(target, available_ids):
		if not available_ids:
			return None
		return min(available_ids, key=lambda cid: abs(cid - target))

	def _cell_index(cid, available_ids):
		try:
			return available_ids.index(int(cid))
		except ValueError:
			return 0

	round_dropdown = widgets.Dropdown(
		options=allowed_rounds,
		value=round_key,
		description="Round",
		layout=widgets.Layout(width="140px"),
	)
	cell_input = widgets.BoundedIntText(
		value=int(start_cell_id) if start_cell_id is not None else int(_available_cells_for_round(round_key)[0]),
		min=min(allowed_cell_ids),
		max=max(allowed_cell_ids),
		step=1,
		description="Cell",
		layout=widgets.Layout(width="170px"),
	)
	prev_button = widgets.Button(description="Prev", layout=widgets.Layout(width="70px"))
	next_button = widgets.Button(description="Next", layout=widgets.Layout(width="70px"))
	metric_dropdown = widgets.Dropdown(
		options=metric_options,
		value="r" if "r" in metric_options else metric_options[0],
		description="Metric",
		layout=widgets.Layout(width="240px"),
	)
	cmap_dropdown = widgets.Dropdown(
		options=cmap_options,
		value=cmap_options[0],
		description="Cmap",
		layout=widgets.Layout(width="220px"),
	)
	axis_dropdown = widgets.Dropdown(
		options=["global", "auto"],
		value=axis_limits if axis_limits in {"global", "auto"} else "global",
		description="Axes",
		layout=widgets.Layout(width="150px"),
	)
	position_html = widgets.HTML()
	status_html = widgets.HTML()
	out = widgets.Output()

	state = {
		"available_cells": [],
		"current_index": 0,
		"syncing": False,
	}

	def _refresh_available_cells(preferred_cell_id=None):
		available = _available_cells_for_round(round_dropdown.value)
		state["available_cells"] = available
		if not available:
			state["current_index"] = 0
			return None, None, False

		if preferred_cell_id is None:
			preferred_cell_id = cell_input.value
		chosen = int(preferred_cell_id)
		was_snapped = chosen not in available
		if chosen not in available:
			chosen = _nearest_cell_id(chosen, available)
		state["current_index"] = _cell_index(chosen, available)
		return chosen, available[state["current_index"]], was_snapped

	def _set_cell_input(value):
		state["syncing"] = True
		cell_input.value = int(value)
		state["syncing"] = False

	def _render(*_):
		requested_cell = cell_input.value
		resolved_request, current_cell_id, was_snapped = _refresh_available_cells(preferred_cell_id=requested_cell)

		with out:
			out.clear_output(wait=True)
			if current_cell_id is None:
				position_html.value = "<b>0 / 0</b>"
				status_html.value = f"<span style='color:#a33'>No cells available for round {round_dropdown.value}.</span>"
				prev_button.disabled = True
				next_button.disabled = True
				return

			if int(cell_input.value) != int(current_cell_id):
				_set_cell_input(current_cell_id)

			available = state["available_cells"]
			prev_button.disabled = state["current_index"] <= 0
			next_button.disabled = state["current_index"] >= len(available) - 1
			position_html.value = f"<b>{state['current_index'] + 1} / {len(available)}</b>"

			mask = (spots_df["round"] == round_dropdown.value) & (spots_df["cell_id"] == int(current_cell_id))
			m_cell = spots_df.loc[mask].copy()
			u_cell = m_cell.loc[m_cell["removed"] == False].copy()

			n_removed = int(len(m_cell) - len(u_cell))
			removed_frac = (n_removed / len(m_cell)) if len(m_cell) else np.nan
			removed_frac_text = f"{removed_frac:.3f}" if len(m_cell) else "nan"
			note = ""
			if was_snapped:
				note = f" Requested {int(requested_cell)}; showing nearest available cell {int(current_cell_id)}."
			status_html.value = (
				f"<b>Cell {int(current_cell_id)}</b> | <b>{round_dropdown.value}</b> | "
				f"mixed={len(m_cell)} | kept={len(u_cell)} | removed={n_removed} | "
				f"removed_frac={removed_frac_text}"
			)
			if note:
				status_html.value += f"<span style='color:#666'>{note}</span>"

			fig = _build_single_cell_unmixing_plotly(
				m_cell=m_cell,
				u_cell=u_cell,
				cell_id=int(current_cell_id),
				round_key=round_dropdown.value,
				chan_order=chan_order,
				metric_col=metric_dropdown.value,
				cmap=cmap_dropdown.value,
				spot_size=spot_size,
				axis_limits=axis_dropdown.value,
			)
			display(fig)

	def _step(delta):
		available = state["available_cells"]
		if not available:
			return
		new_index = min(max(state["current_index"] + delta, 0), len(available) - 1)
		state["current_index"] = new_index
		_set_cell_input(available[new_index])
		_render()

	def _on_prev(_):
		_step(-1)

	def _on_next(_):
		_step(1)

	def _on_round_change(change):
		if change["name"] == "value":
			_render()

	def _on_cell_change(change):
		if change["name"] == "value" and not state["syncing"]:
			_render()

	prev_button.on_click(_on_prev)
	next_button.on_click(_on_next)
	round_dropdown.observe(_on_round_change, names="value")
	cell_input.observe(_on_cell_change, names="value")
	metric_dropdown.observe(_render, names="value")
	cmap_dropdown.observe(_render, names="value")
	axis_dropdown.observe(_render, names="value")

	controls_top = widgets.HBox([
		prev_button,
		next_button,
		cell_input,
		round_dropdown,
		position_html,
	])
	controls_bottom = widgets.HBox([
		metric_dropdown,
		cmap_dropdown,
		axis_dropdown,
	])
	widget = widgets.VBox([
		controls_top,
		controls_bottom,
		status_html,
		out,
	])

	_render()
	return widget


def add_intensity_assigned_chan(
	df,
	assigned_col: str | None = None,
	output_col: str = "intensity_assigned_chan",
	normalized_output_col: str = "intensity_assigned_chan_norm",
	normalize_quantiles: tuple[float, float] = (0.02, 0.98),
):
	"""Add a derived intensity column for each spot's assigned channel.

	For each row, this reads the intensity from the channel-specific column
	``chan_<assigned>_intensity`` where ``<assigned>`` comes from the assigned
	channel column. It also adds a normalized companion metric computed within
	each assigned channel.

	Parameters
	----------
	df : pd.DataFrame
		Spot table containing channel intensity columns such as
		``chan_488_intensity``.
	assigned_col : str or None
		Column to use for assigned channel. If ``None``, uses ``unmixed_chan``
		when present, otherwise ``chan``.
	output_col : str
		Name of the derived output column.
	normalized_output_col : str
		Name of the per-channel normalized output column.
	normalize_quantiles : tuple of float
		Lower and upper quantiles used for robust within-channel scaling after
		``log1p`` transform.

	Returns
	-------
	pd.DataFrame
		A copy of ``df`` with raw and normalized assigned-intensity columns added.
	"""
	import pandas as pd

	if assigned_col is None:
		if "unmixed_chan" in df.columns:
			assigned_col = "unmixed_chan"
		elif "chan" in df.columns:
			assigned_col = "chan"
		else:
			raise ValueError(
				"No assigned channel column found. Expected 'unmixed_chan' or 'chan'."
			)

	if assigned_col not in df.columns:
		raise ValueError(f"Assigned channel column not found: {assigned_col}")

	def _fmt_chan(v):
		if pd.isna(v):
			return None
		s = str(v).strip()
		if s.endswith(".0"):
			s = s[:-2]
		return s

	out = df.copy()
	assigned = out[assigned_col].map(_fmt_chan)
	target_cols = assigned.map(lambda c: f"chan_{c}_intensity" if c is not None else None)
	q_lo, q_hi = normalize_quantiles
	if not (0 <= q_lo < q_hi <= 1):
		raise ValueError("normalize_quantiles must satisfy 0 <= low < high <= 1")

	out[output_col] = np.nan
	out[normalized_output_col] = np.nan
	for col_name, row_idx in target_cols.groupby(target_cols).groups.items():
		if col_name is None:
			continue
		if col_name in out.columns:
			vals = pd.to_numeric(out.loc[row_idx, col_name], errors="coerce")
			out.loc[row_idx, output_col] = vals

			# Normalize within each assigned channel so channels remain comparable
			# despite different absolute intensity scales.
			log_vals = np.log1p(vals.to_numpy(dtype=float))
			valid = np.isfinite(log_vals)
			if not np.any(valid):
				continue

			lo = float(np.nanquantile(log_vals[valid], q_lo))
			hi = float(np.nanquantile(log_vals[valid], q_hi))
			if hi <= lo:
				norm_vals = np.full(log_vals.shape, 0.5)
			else:
				norm_vals = np.clip((log_vals - lo) / (hi - lo), 0.0, 1.0)
			norm_vals[~valid] = np.nan
			out.loc[row_idx, normalized_output_col] = norm_vals

	return out
