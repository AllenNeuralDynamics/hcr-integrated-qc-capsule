# Batch Correction Notes for HCR-Tasic Matching

This note summarizes the current Stage 2 batch-correction behavior in
`run_tasic_cluster_uplevel.py`, how to interpret the QC plots, and what more
sophisticated alternatives could be tested if the simple approach is not
sufficient.

## What the current batch correction does

The current default is a simple cross-mouse correction within the HCR data.

In plain language:

1. For each mouse, look at each gene separately.
2. Compute that mouse's average value for that gene across its cells.
3. Subtract that average from every cell from that mouse.

Effectively, each mouse is shifted back to a common baseline on a per-gene
basis.

This is not a latent-variable integration method. It is a lightweight,
interpretable correction intended to remove broad mouse-specific offsets while
preserving the relative structure among cells within each mouse.

## Current batch modes

The script exposes four modes.

### `all`

Center all genes per mouse.

Use this when the main concern is broad technical mouse-to-mouse offsets, and
when real compositional differences across mice may still exist.

This is the simplest default and is often a good first pass for this pipeline.

### `exclude_markers`

Center all genes except the canonical subclass markers:

- `Pvalb`
- `Sst`
- `Vip`
- `Lamp5`

Use this when there is concern that centering could weaken the main biological
signal used to separate inhibitory subclasses.

This is often the safest alternative to `all` if the major marker structure is
getting blurred after correction.

### `per_mouse`

Re-z-score each mouse independently instead of using pooled z-scoring followed
by centering.

Use this when each mouse behaves like its own self-contained experiment and
cross-mouse alignment is doing more harm than good.

This keeps within-mouse relative structure but makes cross-mouse magnitudes less
directly interpretable.

### `none`

Skip batch correction entirely.

Use this only if there is little visible mouse effect or if correction clearly
removes real biology.

## Practical recommendation for this pipeline

For the current HCR-to-Tasic matching workflow, the most useful comparison is
usually:

1. `all`
2. `exclude_markers`

The decision should be based on three checks:

1. Do cells mix better by mouse after correction?
2. Do canonical subclass markers still separate cleanly?
3. Do final HCR-to-Tasic assignments become more stable and biologically
   sensible?

If marker separation weakens under `all`, try `exclude_markers` next.

## What the post-correction marker UMAP panels show

The subclass-specific UMAP panels in the post-correction QC figure are plotting
values from the batch-corrected HCR expression matrix.

Those values are:

1. HCR spot counts
2. `log1p` transformed
3. gene-wise z-scored across cells
4. then batch-corrected

So the units are corrected gene z-scores. They are dimensionless.

Interpretation:

- `0` means approximately average expression for that gene in the z-scored HCR
  dataset.
- Positive values mean above-average expression.
- Negative values mean below-average expression.

## Colormap scaling for marker panels

The post-correction marker panels should use a shared color scale if the goal is
to compare marker magnitude across genes.

Why:

1. Per-panel autoscaling can make weak genes look artificially strong.
2. A shared scale makes the intensities directly comparable.
3. Because the values are z-scores, a symmetric scale around `0` is the most
   natural choice.

Recommended plotting behavior:

- use one shared `vmin` / `vmax` across the marker panels
- center the colormap at `0`
- use a diverging colormap rather than a purely sequential one

In the current script, the post-correction marker panels were updated to use a
shared symmetric scale and a diverging colormap centered at zero.

## More sophisticated alternatives

If the current centering-based correction is not sufficient, the next level of
methods falls into a few categories.

### Linear gene-level correction

#### `ComBat`

Removes per-gene batch offsets and scaling effects with an empirical Bayes
model.

Pros:

- produces a corrected gene-level matrix
- more principled than simple centering

Risks:

- may remove true biology if mouse composition differs strongly

#### Regression / residualization

Regress out `mouse_id` for each gene and keep the residuals.

Pros:

- simple and explicit

Risks:

- similar confounding risk as `ComBat`

### Embedding or neighbor-graph integration

#### `Harmony`

Adjusts a low-dimensional embedding so cells mix by biology rather than batch.

Pros:

- very practical for PCA/UMAP/clustering workflows
- often the most useful first upgrade for embeddings

Risks:

- mainly produces a corrected embedding, not necessarily the ideal corrected
  gene matrix for downstream centroid correlation

#### `BBKNN`

Builds a batch-balanced neighbor graph rather than directly correcting the
expression matrix.

Pros:

- useful if the main goal is better UMAP structure

Risks:

- not a corrected gene matrix

#### `MNN` / `fastMNN`

Uses mutual nearest neighbors across batches to estimate correction vectors.

Pros:

- can handle more nonlinear shifts than simple centering

Risks:

- depends on shared cell states across mice
- may be unstable with a small targeted panel

#### `Scanorama`

Finds shared cell states across batches and stitches them together.

Pros:

- useful when overlap is partial

Risks:

- can be fragile for a very small gene panel

### Latent-variable models

#### `scVI` / `scANVI`

Deep generative models that learn a latent representation while modeling batch.

Pros:

- flexible and powerful for large single-cell RNA-seq datasets

Risks:

- more infrastructure and tuning
- more opaque than simple methods
- less obviously appropriate for a small HCR panel

## What is most appropriate here

For this specific workflow, sophistication is not automatically better. A small
targeted HCR panel makes overcorrection easier.

The most realistic next tests are:

1. `exclude_markers`
2. `ComBat`
3. `Harmony` for embedding diagnostics
4. `fastMNN` if a stronger nonlinear alignment is really needed

If the downstream objective is correlation-based matching to Tasic centroids,
one reasonable compromise is:

1. keep a simple corrected gene matrix for matching
2. use a more sophisticated embedding-only method like `Harmony` or `BBKNN`
   just for diagnostic UMAPs

## Recommended interpretation hierarchy

Use the simplest method that satisfies the diagnostics.

1. Start with `all`.
2. Move to `exclude_markers` if marker biology is weakened.
3. Test `ComBat` or `Harmony` only if mouse structure remains a real problem.
4. Reserve heavier methods for clear failure cases rather than as defaults.