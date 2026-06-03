# Integrated handoff: matching HCR spatial cell types to the Tasic 2018 reference, with cross-platform normalization

This document merges two pieces:
1. **The strategy layer** — how to coarsen the reference taxonomy to panel resolution, handle subclass gating, and correct for cross-mouse batch effects.
2. **The normalization/comparison engine** — how to normalize each platform in its natural units and make them comparable for centroid correlation and label transfer.

Where the two source documents disagreed, the resolved decision is stated inline and marked **[RESOLVED]** so no contradiction is handed to the executor.

---

## Problem summary

We have spatial transcriptomic data from an HCR-based, expansion / thick-tissue FISH method (EASI-FISH style; Wang et al. 2021, *Cell*) covering a **20-gene panel** targeting **inhibitory (GABAergic) neurons in mouse primary visual cortex (V1 / VISp)**, across **multiple mice**.

Goal: assign each spatially-profiled cell to a known inhibitory cell type, using the Tasic et al. 2018 (*Nature*) VISp scRNA-seq taxonomy as reference, then map the typed cells back into space.

**Core constraint — panel resolution.** A 20-gene panel cannot resolve the full Tasic inhibitory taxonomy (~50 leaf types); many types differ only in off-panel genes. Mapping directly onto the full taxonomy forces false-confidence assignments. We must **coarsen the reference to the resolution the panel supports**, then match. (This is independently endorsed by the normalization doc §9.3: for very small panels, label transfer onto a fine taxonomy is underdetermined — prefer coarsened labels and marker-score logic.)

**Core principle — relative, not absolute.** We match on **relative on/off marker structure**, not absolute abundance. HCR spots, SMART-seq reads, and 10x UMIs are not the same measurement and must not be compared on raw scales.

---

## Inputs

- HCR data: per-cell **resolved spot counts** (not integrated intensity) for the 20 panel genes, with spatial coordinates and a mouse/sample ID per cell.
- Tasic 2018 VISp scRNA-seq data with original cell-type labels (GEO: GSE115746). Separate **exon** and **intron** count matrices.
- Optional: 10x VISp UMI data (for a UMI-based abundance cross-check; see notes).
- The list of 20 panel genes.

---

## Normalization foundations [RESOLVED]

The two source docs conflicted on normalization. **Resolution: follow the natural-units → log1p → gene-wise z-score approach** (do NOT force HCR and Tasic into one shared normalization, and do NOT use TPM as the common currency).

Rationale: z-scoring each gene *within each platform* puts everything on a common "relative high/low" footing, which is exactly the relative-structure comparison we want, without pretending raw scales match.

### N.1 — Per-platform normalization (cells × genes)

```python
import numpy as np, pandas as pd

def log_cp10k(counts):  # SMART-seq counts, 10x UMIs, MERFISH/HCR molecule counts
    totals = counts.sum(axis=1).replace(0, np.nan)
    return np.log1p(counts.div(totals, axis=0) * 1e4)

def zscore_genes(X):    # the cross-platform comparison currency
    return (X - X.mean(axis=0)) / X.std(axis=0).replace(0, np.nan)

def make_centroids(X, labels):
    return X.groupby(labels).mean()
```

- **Tasic (SMART-seq):** `log_cp10k` on counts. **[RESOLVED] Use the EXON-ONLY matrix as the primary reference** — HCR/MERFISH probes target mature transcript, and exon-only best approximates mature mRNA. Keep exon+intron only as a sensitivity check (and for any 10x comparison, since Cell Ranger v7+ includes introns).
- **HCR:** `log1p` on per-cell spot counts (after the batch correction in Pre-step 5), then `zscore_genes`. Optionally divide by cell area/volume *only if* larger segmented cells genuinely contain more RNA rather than reflecting a segmentation artifact.
- **10x (if used):** `log_cp10k` on UMI counts. Never TPM.

### N.2 — Order of operations (critical)

Cross-*sample* (mouse) correction and cross-*platform* (z-score) normalization are different problems and must run in this order:

```
HCR raw spots
  → per-cell normalize
  → per-mouse batch correction        (Pre-step 5 — within HCR only)
  → log1p + gene-wise z-score          (N.1 — cross-platform currency)
  → centroid correlation / matching    (Matching section)
```

Per-mouse batch correction does NOT make platforms comparable; gene-wise z-scoring does NOT remove a per-mouse offset. Both are needed.

---

## Pre-steps

1. Confirm HCR quantitation is **spot-counted**, not intensity-based. If intensity-based, flag before proceeding (changes how quantitative the data can be treated as).
2. Gate to inhibitory cells: retain `Gad1`/`Gad2`+; exclude glutamatergic (`Slc17a7`) and non-neuronal. Apply to both datasets.
3. Subset Tasic to the 20 panel genes.
4. Normalize per platform per **Normalization foundations** above (natural units → log1p → gene-wise z-score). **Do not** force a single shared normalization; **do not** TPM-normalize anything except optionally exon-only Tasic if a length-normalized view is specifically wanted.
5. **Correct cross-mouse batch effects within the HCR data — BEFORE clustering or matching.** See next section.

---

## Cross-sample batch correction (different mice)

Risk: with only 20 genes, a per-mouse technical offset (staining, imaging gain, expansion factor, segmentation) is a large fraction of total variance, so cells can cluster by animal instead of by type.

1. **Diagnose first.** Embed/cluster pooled HCR cells, color by mouse. If types already mix across mice, minimize correction. Only correct as much as the diagnosis warrants.
2. **Check compositional overlap across mice.** Integration assumes shared cell types. If one mouse lacks a subclass another has (uneven V1-depth sampling), correction can force false alignment.
3. **Per-cell normalize first** (size-factor) to remove detection-efficiency differences before per-mouse offsets.
4. **Integrate with `batch = mouse`.** Prefer lighter-touch methods (Harmony, or per-gene per-mouse centering/scaling) over heavy latent-variable models (scVI/scANVI), which overfit with only 20 features.
5. **Guard against over-correction.** Confirm Pvalb/Sst/Vip/Lamp5 still separate afterward. If correction merges subclasses, it removed biology — dial back.
6. **Keep Tasic out of this step.** Batch-correct *among HCR mice* only; the HCR↔Tasic platform offset is handled at Matching (label transfer / z-score), not by treating Tasic as another "batch."

Note: label transfer (recommended at Matching) is somewhat more robust to per-mouse effects than de-novo clustering of pooled cells, but per-mouse normalization is still required so the classifier sees comparable input across animals.

---

## Building the panel-resolution reference taxonomy

Three routes. They should converge; running ≥2 and confirming agreement is the recommended QC. **A is the conservative anchor; C is the best route for data-driven fine structure; B is a looser cross-check.**

### Approach A — Collapse existing Tasic labels (recommended baseline)

Merge only what the panel cannot separate; inherits Tasic's validated structure.

1. Take original Tasic inhibitory cells with published labels.
2. Subset to the 20 panel genes (Pre-step 3).
3. For each pair/group of types, test panel separability: compute per-type mean/median profiles over the 20 genes; two types are **separable** if ≥1 panel gene shows a clear on/off (or strong quantitative) difference.
4. **Merge** non-separable types into one "panel-collapsed" group.
5. Name each group by its constituents (e.g. "Pvalb (subtypes unresolved)", "Sst (subtypes unresolved)", "Vip", "Lamp5").
6. Record mapping table: original type → collapsed group. **Required deliverable.**
7. Proceed to Matching with collapsed groups as labels.

### Approach B — Flat recluster on the limited panel (cross-check)

Re-derive clusters from scratch in 20-gene space. More failure modes — clusters can form on noise.

1. Tasic inhibitory cells, subset to 20 panel genes.
2. Cluster on the 20 genes (Leiden/Louvain or hierarchical). Record resolution parameter.
3. **Test stability** — bootstrap/subsample, recluster, keep reproducible boundaries; collapse the rest.
4. **Validate each cluster against marker structure** — every retained split needs a clear on/off difference in ≥1 panel gene; discard splits not backed by a marker (resolution-parameter artifacts).
5. **Map clusters back to original Tasic labels** — contingency table (new cluster × original type). Flag any cluster mixing biologically distinct types.
6. Name from the contingency table. Record it. **Required deliverable.**
7. Proceed to Matching.

### Approach C — Supervised hierarchical de novo clustering ("pseudo-hierarchical")

The route already attempted: split by known biology first, then cluster within branches.

```
All cells
  └─ Gad1/Gad2+  (inhibitory gate)
       ├─ Pvalb branch  → cluster within → new Pvalb-* labels
       ├─ Sst branch    → cluster within → new Sst-*   labels
       ├─ Vip branch    → cluster within → new Vip-*   labels
       └─ Lamp5 branch  → cluster within → new Lamp5-* labels
```

Bakes validated subclass structure into the top splits (no nonsensical top-level divisions) and reserves data-driven clustering for within-subclass structure where it's most useful. More stable than flat B because each within-branch problem is smaller/more homogeneous. Apply the SAME hierarchy to panel-subset Tasic and to HCR so labels correspond.

1. Gate to inhibitory (`Gad1`/`Gad2`+) — top of hierarchy (Pre-step 2).
2. Assign each cell to a subclass branch via canonical panel markers (`Pvalb`, `Sst`, `Vip`, `Lamp5`; add `Sncg`/`Serpinf1` if present). Use the gating rule in **C.1**.
3. Within each branch, cluster on remaining informative panel genes → sub-labels (e.g. `Sst-1`, `Sst-2`).
4. Stability-test **within each branch** (bootstrap/subsample; keep reproducible sub-clusters).
5. Validate each within-branch split against a real panel-gene marker difference; discard unsupported splits.
6. Run the IDENTICAL hierarchy on panel-subset Tasic; map branches/sub-clusters back to original Tasic labels (contingency table) for naming and sanity-checking.
7. Proceed to Matching **branch-by-branch** (match HCR Sst sub-clusters to Tasic Sst sub-clusters, etc.) to constrain the problem and reduce cross-subclass misassignment.

Cautions:
- **Gating weights a few genes heavily** — a mis-gated cell is wrong downstream. Use explicit bins for ambiguous cells (C.1).
- **Dropout matters more here** — a true Pvalb cell reading low on `Pvalb` gets mis-branched. Soft/probabilistic gating or an "unassigned" bin beats hard thresholds.
- **Within-branch resolution still panel-capped** — don't over-split.
- **C is a supervised variant of B**, not a replacement for A. Validate, then compare to A as convergence check.

#### C.1 — Subclass-gating decision rule (Approach C, step 2)

Four canonical subclasses are *largely* mutually exclusive in cortex, but real cells produce three problem cases — marker-negative, co-expressing/ambiguous, dropout. Handle all explicitly; the unresolved-bin size is itself a QC signal.

**Recommended: soft (probabilistic) assignment.**

1. **Per-subclass score per cell** (normalized marker expression, or a classifier probability trained on panel-subset Tasic subclass labels) — not a single threshold. Degrades gracefully under dropout.
2. **Assign by confidence margin**, not just the max (e.g. top prob > 0.5 AND top − second > a set gap).
3. **Route problem cases to explicit bins:**
   - **Marker-negative** → `Inh-unassigned`. A large fraction flags panel/sensitivity issues or a missed type.
   - **Co-expressing/ambiguous** → `Inh-ambiguous`. In V1 some `Sncg`/`Vip` and `Lamp5`/`Sncg` overlap is genuine biology — keep separate so it doesn't distort within-branch clustering.
   - **Dropout-suspect** (marginal top marker + low total transcript count) → low-confidence; hold out or flag. Tie to per-cell detection depth.
4. **Report bins** — counts/fractions per subclass and per `Inh-unassigned`/`Inh-ambiguous`. Most cells should land in the four subclasses; a large unresolved fraction means revisit thresholds/normalization/panel.
5. **Do NOT cluster the unassigned/ambiguous bins back into branches.** Inspect them separately (missed type, doublets, segmentation artifact).

**Hard-threshold fallback:** assign by highest-expressing marker only when it exceeds a background cutoff AND beats the next marker by a fixed ratio; everything else → unassigned/ambiguous. Validate the cutoff on panel-subset Tasic (known labels → measure mis-gating rate).

**Cross-mouse note:** apply the gate *after* per-mouse correction (Pre-step 5) so thresholds mean the same thing across animals.

---

## Matching: HCR cells → panel-resolution reference

Once a panel-resolution reference exists (collapsed groups from A, validated flat clusters from B, or hierarchical branch/sub-labels from C):

### M.1 — Method: centroid correlation + nearest-centroid label transfer

For a 20-gene panel, centroid-level comparison is more robust than per-cell (reduces dropout/noise), and the normalization doc endorses nearest-centroid transfer for small panels.

```python
panel_genes = sorted(set(hcr_z.columns) & set(tasic_z.columns))  # shared panel

# Reference centroids at PANEL RESOLUTION (collapsed/branch labels, not full taxonomy)
tasic_ref = zscore_genes(tasic_log[panel_genes])
ref_centroids = make_centroids(tasic_ref, tasic_panel_resolution_labels)

# Query = HCR cells (already per-mouse corrected, then z-scored)
hcr_query = hcr_z[panel_genes]

def centroid_correlation(A, B):   # A: centroids×genes, B: cells×genes → centroids×cells
    return pd.DataFrame(
        {cell: A.apply(lambda r: r.corr(B.loc[cell]), axis=1) for cell in B.index}
    )

corr = centroid_correlation(ref_centroids, hcr_query)
hcr_pred = corr.idxmax(axis=0)        # assigned label per cell
hcr_conf = corr.max(axis=0)            # confidence metric
```

1. **Match at panel resolution only** — feed centroids from the coarsened taxonomy, never the full ~50-type taxonomy.
2. For Approach C, match **branch-by-branch** (Sst query cells against Sst reference sub-centroids).
3. **Report per-cell confidence** (`hcr_conf`); flag low-confidence cells (ambiguous, poorly segmented, off-panel, or absent from reference) rather than force-assigning.
4. Build a **correspondence heatmap / confusion matrix** (HCR clusters × reference labels).
5. **Interpret only at panel-supported resolution** — no leaf-type claims the panel can't deliver.

### M.2 — Marker-score cross-check (small-panel safeguard)

Because the panel is small, also compute marker-set scores as an interpretable parallel to label transfer (normalization doc §8). Agreement between marker scores and centroid-transfer labels is reassuring; disagreement localizes ambiguity.

```python
marker_sets = {  # edit to your exact panel
    "Pvalb": ["Pvalb"], "Sst": ["Sst"], "Vip": ["Vip"], "Lamp5": ["Lamp5"],
}
def score_marker_sets(X, sets):
    return pd.DataFrame({n: X[[g for g in gs if g in X.columns]].mean(axis=1)
                         for n, gs in sets.items()}, index=X.index)
hcr_marker_scores = score_marker_sets(hcr_z, marker_sets)
```

---

## Convergence check (recommended)

Run ≥2 of the three taxonomy routes and compare reference taxonomies and HCR assignments. Recommended pairing: **A as trusted baseline, C as exploratory taxonomy**; reconcile differences. Agreement = robust panel-resolution typing; disagreement localizes where the panel is ambiguous.

---

## Sensitivity & robustness analyses

1. **Exon-only vs exon+intron Tasic reference.** Re-run matching with both; report label agreement. If conclusions change, the comparison is sensitive to feature definition — inspect which genes/types drive it. (Default conclusion stands on exon-only per [RESOLVED] above.)
2. **Per-gene cross-platform correlation.** For each panel gene, correlate its relative pattern across cell types between HCR and Tasic (and 10x if available). Flag genes that disagree — unreliable probes or platform-specific artifacts.
3. **Normalization alternative for HCR** (area/volume-normalized vs not) if cell-size variation is non-trivial.
4. **Panel coverage report** — how many genes overlap each comparison.

---

## Payoff

Once HCR cells are typed at panel resolution, overlay spatial coordinates → the **3D spatial distribution of each inhibitory subclass within V1**, the information dissociated Tasic data cannot provide. Validate spatially: predicted labels should occupy plausible laminar positions; marker-score spatial maps should show expected gradients.

---

## Required deliverables

- **Mapping table:** original Tasic type → panel-resolution label (per approach used).
- **Cluster stability report** (Approaches B and C; per-branch for C).
- **Cross-mouse batch diagnostic:** pre/post-correction embeddings colored by mouse, plus confirmation subclass markers still separate.
- **Subclass-gating report** (Approach C): rule, thresholds/margins, counts + fractions per subclass and per `Inh-unassigned`/`Inh-ambiguous`.
- **Per-cell HCR assignments** with confidence scores.
- **Correspondence heatmap** (HCR × reference) and marker-score cross-check.
- **Sensitivity tables:** exon-only vs exon+intron label agreement; per-gene cross-platform correlation.
- **Spatial outputs:** predicted-label map, marker-score maps.
- **Known limitations:** type distinctions the panel could NOT resolve; low-confidence/unassigned fractions.

---

## Key decisions locked in (so the executor has no ambiguity)

| Decision point | Locked choice | Why |
|---|---|---|
| Common normalization currency | Natural units → log1p → **gene-wise z-score per platform** | Compares relative structure; avoids fake cross-platform scale matching |
| TPM | **Not used** (except optional length-normalized view of exon-only Tasic) | Wrong for UMI/molecule/intensity data |
| Tasic matrix for spatial matching | **Exon-only** (exon+intron = sensitivity check) | HCR/MERFISH probes target mature transcript |
| Reference resolution | **Coarsened** (A/B/C), never full ~50-type taxonomy | 20 genes can't resolve leaf types; avoids false-confidence labels |
| Matching method | **Centroid correlation + nearest-centroid transfer** at panel resolution | Robust for small panels; per-cell transfer underdetermined |
| Mouse batch vs platform offset | **Two separate steps, in order** (per-mouse correct → then z-score) | They fix different problems; neither substitutes for the other |
| Subclass gating | **Soft/probabilistic with explicit unassigned/ambiguous bins** | Robust to dropout and genuine co-expression |
