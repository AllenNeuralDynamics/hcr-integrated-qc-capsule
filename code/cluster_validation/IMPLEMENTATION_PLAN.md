# Implementation Plan: HCR-Tasic Matching Protocol

## Overview

Build `run_hcr_tasic_matching.py` — a single script implementing the full matching protocol from `hcr_tasic_matching_protocol.md`. Assigns each spatially-profiled HCR cell to a known inhibitory cell type using the Tasic 2018 VISp scRNA-seq taxonomy as reference.

---

## Stage 1 — Data Loading & Normalization Pipeline

| Step | What | Notes |
|------|------|-------|
| 1.1 | **Load Smart-seq reference** | Exon-only matrix via `cluster_validation_utils.load_visp_expression()`, gate to inhibitory. Already proven in the batch script. |
| 1.2 | **Load multi-mouse HCR data** | Loop over available mice (`790322`, `782149`, `788406`, etc.), call `get_hcr_dataset_pairwise()` → `pw_ds.load_inhibitory_cells(unmixed=True, as_anndata=True)`, tag each cell with `mouse_id`, concatenate into one AnnData. |
| 1.3 | **Normalize Tasic (SMART-seq)** | `log_cp10k` (normalize_total to 10k, log1p). Exon-only as primary. |
| 1.4 | **Normalize HCR** | `log1p` on raw spot counts (already integer counts from pairwise dataset). |
| 1.5 | **Gene-wise z-score** (per platform) | Z-score each gene independently within Tasic cells and within HCR cells — this is the cross-platform comparison currency. |
| 1.6 | **Intersect to shared panel genes** | Find genes present in both platforms (excluding GFP). ~18-19 shared genes. |

---

## Stage 2 — Cross-Mouse Batch Correction (within HCR)

| Step | What | Notes |
|------|------|-------|
| 2.1 | **Diagnose batch effects** | Embed pooled HCR cells (PCA → UMAP), color by mouse. Check whether cells cluster by mouse or by type. |
| 2.2 | **Apply correction** | Per-gene per-mouse centering (lightest touch) or Harmony on the 20-gene space. Prefer simple centering first; escalate if diagnosis warrants. |
| 2.3 | **Post-correction QC** | Confirm Pvalb/Sst/Vip/Lamp5 still separate after correction. If they merge → over-corrected. |

---

## Stage 3 — Approach A: Collapse Tasic Labels to Panel Resolution

| Step | What | Notes |
|------|------|-------|
| 3.1 | **Compute per-type centroids** on z-scored Tasic (panel genes only) | Group by original Tasic `cluster` label → mean expression vector. |
| 3.2 | **Test pairwise separability** | For each pair of types within a subclass, check if ≥1 panel gene has a clear on/off difference (e.g., effect size > threshold or AUC > 0.8). |
| 3.3 | **Merge non-separable types** | Iteratively merge types that can't be distinguished → produce "panel-collapsed" labels. |
| 3.4 | **Save mapping table** | Original Tasic type → collapsed label. Required deliverable. |

---

## Stage 4 — Approach C: Supervised Hierarchical Clustering

| Step | What | Notes |
|------|------|-------|
| 4.1 | **Soft subclass gating** | Train a simple classifier (k-NN or logistic regression on the 4 canonical markers) on Tasic subclass labels → apply to each cell. Route low-confidence cells to `Inh-unassigned`/`Inh-ambiguous` bins. |
| 4.2 | **Within-branch clustering** | For each subclass branch (Pvalb, Sst, Vip, Lamp5), run Leiden sweep on remaining panel genes (reuse `cluster_gene_positive_cells` logic). |
| 4.3 | **Stability test** | Bootstrap within each branch (reuse `top_discriminable_genes_per_cluster` with `bootstrap_iterations`). Keep only reproducible sub-clusters. |
| 4.4 | **Map back to Tasic labels** | Contingency table of C sub-clusters × original Tasic types for naming. |

---

## Stage 5 — Matching: HCR Cells → Panel-Resolution Reference

| Step | What | Notes |
|------|------|-------|
| 5.1 | **Compute reference centroids** | From Approach A collapsed labels (and separately from Approach C branch labels) on z-scored Tasic panel genes. |
| 5.2 | **Centroid correlation** | For each HCR cell (z-scored), correlate against every reference centroid → assign to highest-corr label. |
| 5.3 | **Per-cell confidence** | Report max correlation as confidence; flag low-confidence cells. |
| 5.4 | **Branch-by-branch matching** (Approach C) | Match Sst HCR cells only against Sst reference sub-centroids, etc. |
| 5.5 | **Marker-score cross-check** | Compute per-cell marker-set scores (mean z-scored expression of canonical markers) → compare with centroid-transfer labels. |

---

## Stage 6 — Convergence Check & Outputs

| Step | What | Notes |
|------|------|-------|
| 6.1 | **Compare A vs C assignments** | Compute agreement rate, confusion matrix between the two approaches. |
| 6.2 | **Sensitivity: exon-only vs exon+intron** | Re-run matching with exon+intron Tasic; report label agreement. |
| 6.3 | **Per-gene cross-platform correlation** | For each panel gene, correlate its type-level pattern between HCR and Tasic. Flag discordant genes. |
| 6.4 | **Save deliverables** | Mapping tables, per-cell assignments + confidence, correspondence heatmaps, spatial outputs. |

---

## File Structure

```
/root/capsule/code/cluster_validation/run_hcr_tasic_matching.py
```

One script with clearly separated functions for each stage, plus a `main()` that runs them in sequence and saves outputs to `/root/capsule/results/hcr_tasic_matching/`.

---

## Dependencies on Existing Code

- `cluster_validation_utils`: `load_visp_expression`, `make_filtered_views_for_smartseq`, `cluster_gene_positive_cells`, `top_discriminable_genes_per_cluster`, `compute_knn_soft_votes`
- `aind_hcr_data_loader`: `get_hcr_dataset_pairwise`
- External: `scanpy`, `numpy`, `pandas`, `sklearn`, optionally `harmonypy` for batch correction

---

## Recommended Build Order

1. **Stages 1–2**: Data loading, normalization, batch correction pipeline — get both platforms into z-scored comparable form.
2. **Stage 3**: Approach A (collapse Tasic labels) — the conservative baseline.
3. **Stages 4–5**: Approach C (hierarchical) + matching engine — the core matching logic.
4. **Stage 6**: Convergence, sensitivity, deliverables.

---

## Required Deliverables (from protocol)

- Mapping table: original Tasic type → panel-resolution label (per approach)
- Cluster stability report (per-branch for Approach C)
- Cross-mouse batch diagnostic: pre/post-correction embeddings colored by mouse
- Subclass-gating report (Approach C): rule, thresholds, counts per bin
- Per-cell HCR assignments with confidence scores
- Correspondence heatmap (HCR × reference) and marker-score cross-check
- Sensitivity tables: exon-only vs exon+intron; per-gene cross-platform correlation
- Spatial outputs: predicted-label map, marker-score maps
- Known limitations: type distinctions the panel could NOT resolve
