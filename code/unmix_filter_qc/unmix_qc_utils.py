"""Utility helpers for the unmixing QC notebook."""

from __future__ import annotations

from typing import Iterable, Sequence

import ipywidgets as widgets
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

	fig.update_yaxes(scaleanchor="x", scaleratio=1)
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
	metric_options: Iterable[str] = ("r", "dist", "d_assign_neighbor_ratio_1", "d_assign_neighbor_ratio_2", "dye_line_dist_ratio", "intensity_assigned_chan"),
	cmap_options: Sequence[str] = DEFAULT_CMAPS_PLOTLY,
):
	"""Interactive Plotly version with dropdowns for metric and colormap."""
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
			)
			display(fig)

	metric_dropdown.observe(_render, names="value")
	cmap_dropdown.observe(_render, names="value")

	widget = widgets.VBox([widgets.HBox([metric_dropdown, cmap_dropdown]), out])
	_render()
	return widget


def add_intensity_assigned_chan(
	df,
	assigned_col: str | None = None,
	output_col: str = "intensity_assigned_chan",
):
	"""Add a derived intensity column for each spot's assigned channel.

	For each row, this reads the intensity from the channel-specific column
	``chan_<assigned>_intensity`` where ``<assigned>`` comes from the assigned
	channel column.

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

	Returns
	-------
	pd.DataFrame
		A copy of ``df`` with ``output_col`` added.
	"""
	import pandas as pd

	# if assigned_col is None:
	# 	if "unmixed_chan" in df.columns:
	# 		assigned_col = "unmixed_chan"
	# 	elif "chan" in df.columns:
	# 		assigned_col = "chan"
	# 	else:
	# 		raise ValueError(
	# 			"No assigned channel column found. Expected 'unmixed_chan' or 'chan'."
	# 		)
	assigned_col = "chan"

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

	out[output_col] = np.nan
	for col_name, row_idx in target_cols.groupby(target_cols).groups.items():
		if col_name is None:
			continue
		if col_name in out.columns:
			out.loc[row_idx, output_col] = pd.to_numeric(
				out.loc[row_idx, col_name],
				errors="coerce",
			)

	return out
