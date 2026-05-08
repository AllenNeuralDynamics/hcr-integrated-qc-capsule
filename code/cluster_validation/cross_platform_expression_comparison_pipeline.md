# Cross-platform expression comparison pipeline: SMART-seq, 10x, MERFISH, and HCR

This note describes a practical pipeline for comparing the Allen/Tasic mouse VISp SMART-seq dataset with 10x Genomics single-cell RNA-seq and targeted spatial assays such as MERFISH and HCR RNA-FISH.

The main idea is: **do not force all platforms into one universal normalization such as TPM.** Instead, normalize each assay in the units it naturally measures, then compare shared genes, cell-type centroids, marker/signature scores, and spatial patterns.

---

## 1. Assay summary

| Platform | Typical measurement | Gene coverage | Natural unit | Main bias / caveat | Recommended comparison representation |
|---|---:|---:|---:|---|---|
| SMART-seq / Allen-Tasic | Full-length read counts | Whole transcriptome | Reads per gene | Gene length, amplification, batch, plate effects | `log1p(CP10K)` or `log1p(TPM)` for exon-only; `log1p(CP10K)` for exon+intron |
| 10x Genomics | 3' or 5' UMI counts | Whole transcriptome | UMIs per gene | Capture efficiency, dropout, ambient RNA, version-dependent intron handling | `log1p(CP10K)`, SCTransform, or pseudobulk counts |
| MERFISH | Targeted RNA molecule counts | Probe panel | Molecules per cell per gene | Probe efficiency, panel design, segmentation, optical crowding | total-count/cell-size normalized `log1p`, then gene-wise scaling |
| HCR RNA-FISH | Targeted spots or fluorescence intensity | Small marker panel | Spots or intensity per cell | Background, probe efficiency, imaging settings, segmentation | background-corrected per-cell signal, then gene-wise scaling |

---

## 2. Key normalization logic

### 2.1 Do not use TPM as the common currency

TPM is useful for full-length read-based RNA-seq because longer genes tend to accumulate more reads. But TPM is not appropriate for 10x UMI counts, MERFISH molecule counts, or HCR intensity/spot data.

Use TPM only when you specifically want a length-normalized representation of the Allen/Tasic SMART-seq exon counts.

### 2.2 Better common currency: within-platform normalized, gene-wise scaled expression

For cross-platform comparison, the most stable representation is usually:

1. Normalize each platform in its natural way.
2. Restrict to the relevant shared gene panel.
3. Transform with `log1p` when appropriate.
4. Z-score each gene **within each platform**.
5. Compare cell types, signatures, or spatial patterns.

Gene-wise z-scoring emphasizes whether a gene is relatively high or low across cells or cell types within a platform, rather than trying to compare absolute molecules, UMIs, reads, and fluorescence intensity directly.

---

## 3. Exons, introns, and when to sum them

The Allen/Tasic mouse VISp data provide separate exon and intron count matrices. The readme says the sequencing results were aligned to exons and introns in the GRCm38.p3 reference genome using STAR, and that gene-level counts were calculated separately for exon and intron matrices.

| Comparison target | Allen/Tasic matrix to use | Reason |
|---|---:|---|
| Standard mature mRNA expression | Exon only | Exons better approximate mature transcript abundance |
| MERFISH with standard exonic/mature transcript probes | Exon only | Most targeted RNA-FISH panels are designed against mature transcript sequence unless specified otherwise |
| HCR with standard mature transcript probes | Exon only | Same logic as MERFISH |
| 10x Cell Ranger v7+ whole-transcriptome matrix | Exon + intron | Cell Ranger v7+ includes intronic reads by default for whole-transcriptome gene expression |
| Single-nucleus 10x | Exon + intron | Nuclear data often contain substantial unspliced/pre-mRNA signal |
| Sensitivity analysis | Both exon-only and exon+intron | Useful to quantify platform-definition effects |

Recommended practical setup:

```python
allen_exon = exon_counts
allen_exon_intron = exon_counts + intron_counts
```

For multi-platform comparison, I would usually make the **main spatial comparison with exon-only Allen counts**, and keep **exon+intron** as a sensitivity analysis or for direct comparison to 10x matrices that include intronic UMIs.

---

## 4. Gene lengths for SMART-seq TPM

If you compute TPM for Allen/Tasic exon counts, use exonic gene lengths from a GRCm38.p3-compatible annotation such as GENCODE mouse M4 or M5.

The length should generally be the **union of all annotated exons per gene**, so overlapping exons from different isoforms are counted once.

### Python example: compute exonic gene lengths from a GTF

```python
import gzip
import re
import pandas as pd
from collections import defaultdict

# Example GENCODE mouse GRCm38.p3 annotation:
# gencode.vM4.annotation.gtf.gz or gencode.vM5.annotation.gtf.gz

gtf_path = "gencode.vM4.annotation.gtf.gz"

def parse_attr(attr, key):
    m = re.search(fr'{key} "([^"]+)"', attr)
    return m.group(1) if m else None

exons = defaultdict(list)

with gzip.open(gtf_path, "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        chrom, source, feature, start, end, score, strand, frame, attr = fields
        if feature != "exon":
            continue
        gene_name = parse_attr(attr, "gene_name")
        if gene_name is None:
            continue
        exons[gene_name].append((chrom, int(start), int(end)))

def merged_length(intervals):
    by_chrom = defaultdict(list)
    for chrom, start, end in intervals:
        by_chrom[chrom].append((start, end))

    total = 0
    for chrom, ivals in by_chrom.items():
        ivals = sorted(ivals)
        merged = []
        for start, end in ivals:
            if not merged or start > merged[-1][1] + 1:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        total += sum(end - start + 1 for start, end in merged)
    return total

gene_lengths = pd.DataFrame({
    "gene": list(exons.keys()),
    "length_bp": [merged_length(v) for v in exons.values()]
})
```

### TPM from exon counts

Use this only for the SMART-seq exon matrix, not for 10x/MERFISH/HCR.

```python
import numpy as np
import pandas as pd

# counts: genes x cells
# gene_lengths: columns = ["gene", "length_bp"]

lengths = gene_lengths.set_index("gene")["length_bp"]
common = counts.index.intersection(lengths.index)

counts2 = counts.loc[common]
length_kb = lengths.loc[common] / 1000

rpk = counts2.div(length_kb, axis=0)
tpm = rpk.div(rpk.sum(axis=0), axis=1) * 1_000_000
log_tpm = np.log1p(tpm)
```

---

## 5. Recommended per-platform normalization

### 5.1 Helper functions

Assume matrices are **cells x genes** unless otherwise noted.

```python
import numpy as np
import pandas as pd

def log_cp10k(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Library-size normalize cells x genes counts to counts per 10k, then log1p.
    Works for SMART-seq counts, 10x UMI counts, and MERFISH molecule counts.
    """
    totals = counts.sum(axis=1).replace(0, np.nan)
    cp10k = counts.div(totals, axis=0) * 1e4
    return np.log1p(cp10k)


def zscore_genes(X: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each gene across cells or centroids within a platform.
    """
    mean = X.mean(axis=0)
    sd = X.std(axis=0).replace(0, np.nan)
    return (X - mean) / sd


def make_centroids(X: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """
    Average cells into cell-type, subclass, cluster, layer, or region centroids.
    X should be cells x genes. labels should be indexed like X rows.
    """
    return X.groupby(labels).mean()
```

### 5.2 SMART-seq / Allen-Tasic

If raw matrices are genes x cells, transpose first.

```python
# exon_counts_raw: genes x cells
# intron_counts_raw: genes x cells

allen_exon_counts = exon_counts_raw.T
allen_total_counts = (exon_counts_raw + intron_counts_raw).T

allen_exon_log = log_cp10k(allen_exon_counts)
allen_total_log = log_cp10k(allen_total_counts)
```

For the main multi-platform comparison, start with:

```python
allen_main_log = allen_exon_log
```

Then use `allen_total_log` as a sensitivity analysis.

### 5.3 10x Genomics

```python
# tenx_counts: cells x genes raw UMI counts

tenx_log = log_cp10k(tenx_counts)
```

Do not TPM-normalize 10x UMI counts.

### 5.4 MERFISH

If MERFISH is already molecule counts per segmented cell:

```python
# merfish_counts: cells x genes targeted molecule counts

merfish_log = log_cp10k(merfish_counts)
```

If cell sizes vary strongly, consider normalizing by cell area/volume or using total molecules plus area/volume as covariates. A simple area-normalized version might look like:

```python
# cell_area: Series indexed by MERFISH cell IDs

merfish_area_norm = merfish_counts.div(cell_area, axis=0)
merfish_area_log = np.log1p(merfish_area_norm)
```

Which one is better depends on whether larger segmented cells truly contain more RNA or whether area is mostly a segmentation/imaging artifact.

### 5.5 HCR RNA-FISH

For HCR, the input might be spot counts or intensity per cell.

Recommended minimal processing:

1. Background-correct each channel.
2. Aggregate per cell.
3. Optionally divide by area/volume.
4. `log1p` transform.
5. Gene-wise z-score.

```python
# hcr_signal: cells x genes, after background correction and segmentation

hcr_log = np.log1p(hcr_signal)
hcr_z = zscore_genes(hcr_log)
```

Do not assume HCR intensity is on the same absolute scale as RNA-seq counts.

---

## 6. Gene panel strategy

Do not require all four platforms to share the same gene set unless the intersection is large enough.

Use layered comparisons instead:

| Layer | Comparison | Gene set | Goal |
|---|---|---|---|
| RNA-seq reference layer | SMART-seq ↔ 10x | Broad transcriptome or HVGs | Check global cell-type agreement |
| MERFISH projection layer | RNA-seq reference ↔ MERFISH | MERFISH panel genes | Assign or validate spatial cell types |
| HCR validation layer | RNA-seq/MERFISH expectations ↔ HCR | HCR marker genes | Validate markers and spatial/layer patterns |
| Four-way summary layer | SMART-seq ↔ 10x ↔ MERFISH ↔ HCR | Only high-confidence shared markers | Qualitative agreement, not full integration |

---

## 7. Cell-type centroid comparison

This is the safest first analysis because it reduces single-cell dropout and platform-specific noise.

### 7.1 Build centroids

```python
# Metadata examples:
# allen_meta["cluster"] or allen_meta["subclass"]
# tenx_meta["cell_type"]
# merfish_meta["cell_type"] or merfish_meta["region"]

allen_centroids = make_centroids(allen_main_log, allen_meta["cluster"])
tenx_centroids = make_centroids(tenx_log, tenx_meta["cell_type"])
merfish_centroids = make_centroids(merfish_log, merfish_meta["cell_type"])
```

### 7.2 Restrict to a panel and z-score within platform

```python
merfish_genes = sorted(set(merfish_log.columns))

shared = sorted(
    set(allen_centroids.columns)
    & set(tenx_centroids.columns)
    & set(merfish_genes)
)

allen_cz = zscore_genes(allen_centroids[shared])
tenx_cz = zscore_genes(tenx_centroids[shared])
merfish_cz = zscore_genes(merfish_centroids[shared])
```

### 7.3 Correlate centroids

```python
def centroid_correlation(A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    """
    A and B are centroid x gene matrices over the same genes.
    Returns A-centroids x B-centroids Pearson correlations.
    """
    rows = []
    for a_name, a in A.iterrows():
        row = []
        for b_name, b in B.iterrows():
            row.append(a.corr(b))
        rows.append(row)
    return pd.DataFrame(rows, index=A.index, columns=B.index)

allen_vs_merfish = centroid_correlation(allen_cz, merfish_cz)
tenx_vs_merfish = centroid_correlation(tenx_cz, merfish_cz)
allen_vs_tenx = centroid_correlation(allen_cz, tenx_cz)
```

Interpretation:

- Rows = reference cell types or clusters.
- Columns = query cell types, spatial clusters, or regions.
- High correlation means similar relative marker pattern over the selected gene panel.

---

## 8. Marker/signature score comparison

For MERFISH and HCR, especially if HCR has a small marker panel, marker scores are often more interpretable than high-dimensional integration.

### 8.1 Example marker dictionaries

Edit these to match your exact panel and biology.

```python
marker_sets = {
    "Excitatory": ["Slc17a7", "Camk2a", "Neurod6"],
    "Inhibitory": ["Gad1", "Gad2", "Slc6a1"],
    "Astrocyte": ["Aqp4", "Slc1a3", "Gfap"],
    "Oligodendrocyte": ["Mbp", "Plp1", "Mog"],
    "OPC": ["Pdgfra", "Cspg4"],
    "Microglia": ["C1qa", "Cx3cr1", "P2ry12"],
    "Endothelial": ["Pecam1", "Cldn5"],
}
```

### 8.2 Score cells or centroids

```python
def score_marker_sets(X: pd.DataFrame, marker_sets: dict[str, list[str]]) -> pd.DataFrame:
    """
    X is cells x genes or centroids x genes, preferably log-normalized and gene-wise scaled.
    Returns cells/centroids x marker sets.
    """
    scores = {}
    for name, genes in marker_sets.items():
        present = [g for g in genes if g in X.columns]
        if len(present) == 0:
            scores[name] = np.nan
        else:
            scores[name] = X[present].mean(axis=1)
    return pd.DataFrame(scores, index=X.index)

allen_scores = score_marker_sets(zscore_genes(allen_main_log), marker_sets)
tenx_scores = score_marker_sets(zscore_genes(tenx_log), marker_sets)
merfish_scores = score_marker_sets(zscore_genes(merfish_log), marker_sets)
hcr_scores = score_marker_sets(hcr_z, marker_sets)
```

Use these scores to compare expected cell identity and spatial distribution.

---

## 9. Projection / label transfer approach

A simple version is nearest-centroid classification.

### 9.1 Build a reference from RNA-seq

```python
# Use genes measured in the spatial assay
panel_genes = sorted(set(merfish_log.columns) & set(allen_main_log.columns))

allen_ref = zscore_genes(allen_main_log[panel_genes])
allen_ref_centroids = make_centroids(allen_ref, allen_meta["cluster"])
```

### 9.2 Project MERFISH cells

```python
merfish_query = zscore_genes(merfish_log[panel_genes])

corr = centroid_correlation(allen_ref_centroids, merfish_query)
# rows = Allen clusters, columns = MERFISH cells

merfish_predicted_cluster = corr.idxmax(axis=0)
merfish_prediction_score = corr.max(axis=0)
```

Use `merfish_prediction_score` as a confidence metric. Low-confidence cells may be ambiguous, poorly segmented, out of panel, or absent from the reference.

### 9.3 Project HCR cautiously

For HCR, the gene panel may be very small, so label transfer may be underdetermined. Prefer marker-score logic unless the HCR panel is large and discriminative.

---

## 10. Spatial validation outputs

For MERFISH/HCR, do not stop at expression-space agreement. Also check spatial sanity.

Recommended outputs:

| Output | What it tells you |
|---|---|
| Cell-type centroid correlation heatmap | Whether platforms agree on expression signatures |
| Predicted cell-type spatial map | Whether inferred labels occupy plausible anatomical/layer positions |
| Marker-score spatial maps | Whether marker gradients and cell classes appear where expected |
| Confusion matrix against existing annotations | Whether label transfer matches known cell labels |
| Exon-only vs exon+intron sensitivity table | Whether intron inclusion changes conclusions |
| Per-gene platform correlation table | Which genes are reliable or problematic across platforms |

---

## 11. Exon-only vs exon+intron sensitivity analysis

Run the same comparison twice:

```python
allen_versions = {
    "exon_only": allen_exon_log,
    "exon_plus_intron": allen_total_log,
}

results = {}

for version_name, allen_log in allen_versions.items():
    panel_genes = sorted(set(allen_log.columns) & set(merfish_log.columns))

    allen_ref = zscore_genes(allen_log[panel_genes])
    merfish_query = zscore_genes(merfish_log[panel_genes])

    allen_ref_centroids = make_centroids(allen_ref, allen_meta["cluster"])
    corr = centroid_correlation(allen_ref_centroids, merfish_query)

    results[version_name] = {
        "predicted_cluster": corr.idxmax(axis=0),
        "prediction_score": corr.max(axis=0),
    }
```

Then compare predictions:

```python
pred_exon = results["exon_only"]["predicted_cluster"]
pred_total = results["exon_plus_intron"]["predicted_cluster"]

agreement = (pred_exon == pred_total).mean()
print(f"Exon-only vs exon+intron label agreement: {agreement:.3f}")
```

If conclusions change substantially, report that the comparison is sensitive to feature definition and inspect which genes/cell types drive the difference.

---

## 12. Suggested final analysis structure

### Step A: Build the RNA-seq reference

1. Load Allen/Tasic exon counts, intron counts, gene metadata, and cell metadata.
2. Load 10x raw UMI counts and metadata.
3. Normalize Allen exon-only and Allen exon+intron with `log1p(CP10K)`.
4. Normalize 10x with `log1p(CP10K)` or SCTransform.
5. Compare Allen vs 10x over broad shared genes and known markers.
6. Decide whether exon-only or exon+intron is more relevant for the 10x matrix.

### Step B: Compare MERFISH to the RNA-seq reference

1. Normalize MERFISH molecules per cell.
2. Restrict RNA-seq reference to MERFISH panel genes.
3. Gene-wise z-score within platform.
4. Compare centroid correlations.
5. Transfer labels or score marker signatures.
6. Plot spatial maps of predicted labels and marker scores.

### Step C: Compare HCR as targeted validation

1. Background-correct HCR channels.
2. Aggregate per cell or anatomical region.
3. Normalize by area/volume only if justified.
4. Z-score genes within HCR.
5. Compute marker/signature scores.
6. Compare expected patterns from Allen/10x/MERFISH to observed spatial pattern.

### Step D: Report robustness

Include:

- Exon-only vs exon+intron Allen results.
- Alternative normalization choices for MERFISH/HCR if relevant.
- Panel gene coverage: how many genes overlap each comparison.
- Low-confidence or ambiguous cell types.
- Genes that disagree strongly across platforms.

---

## 13. Minimal end-to-end skeleton

```python
# 1. Normalize RNA-seq
allen_exon_log = log_cp10k(exon_counts_raw.T)
allen_total_log = log_cp10k((exon_counts_raw + intron_counts_raw).T)
tenx_log = log_cp10k(tenx_counts)

# 2. Normalize spatial
merfish_log = log_cp10k(merfish_counts)
hcr_log = np.log1p(hcr_signal)

# 3. Choose main Allen reference
allen_main_log = allen_exon_log

# 4. MERFISH-panel comparison
merfish_panel = sorted(set(merfish_log.columns) & set(allen_main_log.columns))

allen_ref = zscore_genes(allen_main_log[merfish_panel])
merfish_query = zscore_genes(merfish_log[merfish_panel])

allen_ref_centroids = make_centroids(allen_ref, allen_meta["cluster"])
merfish_corr = centroid_correlation(allen_ref_centroids, merfish_query)

merfish_predicted_cluster = merfish_corr.idxmax(axis=0)
merfish_prediction_score = merfish_corr.max(axis=0)

# 5. HCR marker-score comparison
hcr_z = zscore_genes(hcr_log)
hcr_scores = score_marker_sets(hcr_z, marker_sets)

# 6. Optional 10x centroids
tenx_panel = sorted(set(tenx_log.columns) & set(allen_main_log.columns))
allen_tenx_shared_ref = zscore_genes(allen_main_log[tenx_panel])
tenx_shared = zscore_genes(tenx_log[tenx_panel])

allen_centroids = make_centroids(allen_tenx_shared_ref, allen_meta["cluster"])
tenx_centroids = make_centroids(tenx_shared, tenx_meta["cell_type"])
allen_vs_tenx = centroid_correlation(allen_centroids, tenx_centroids)
```

---

## 14. Practical interpretation rules

| Observation | Interpretation | Next check |
|---|---|---|
| Allen and 10x agree, but MERFISH differs | Spatial panel/probe/segmentation or tissue-region difference | Check MERFISH panel gene reliability and spatial anatomy |
| MERFISH agrees with Allen exon-only but not exon+intron | Spatial assay likely reflects mature RNA | Use exon-only for spatial reference |
| 10x agrees better with Allen exon+intron | 10x matrix likely includes intronic/pre-mRNA signal | Confirm Cell Ranger version/settings |
| HCR marker agrees spatially but not quantitatively | Normal for intensity-based targeted assays | Treat HCR as validation rather than transcriptome-scale quantification |
| Cell-type label transfer is low-confidence | Panel may be insufficient or cell type absent | Use broader subclass labels or marker scores |
| Four-way shared gene set is tiny | Full integration is underpowered | Use layered comparisons instead of one global model |

---

## 15. Bottom line

For comparing SMART-seq, 10x, MERFISH, and HCR together:

1. **Do not normalize everything to TPM.**
2. Use **exon-only Allen/Tasic** as the main mature RNA reference for spatial comparison.
3. Use **exon+intron Allen/Tasic** when comparing to 10x matrices that include introns or to single-nucleus data.
4. Normalize each platform in its natural units.
5. Compare using **shared panels, centroids, marker scores, and gene-wise scaling**.
6. Treat MERFISH and HCR as targeted spatial measurements, not full-transcriptome RNA-seq replacements.

---

## References and source notes

- Allen/Tasic mouse VISp readme: exon and intron matrices are separate gene-level count matrices; gene metadata include gene symbol, chromosome, Entrez ID, and gene name.
- 10x Genomics Cell Ranger intron guidance: https://www.10xgenomics.com/support/software/cell-ranger/latest/miscellaneous/cr-intron-mode-rec
- 10x Genomics gene expression algorithm: https://www.10xgenomics.com/support/software/cell-ranger/latest/algorithms-overview/cr-gex-algorithm
- 10x Genomics TPM/RPKM/FPKM FAQ: https://kb.10xgenomics.com/hc/en-us/articles/115003684783-Should-I-calculate-TPM-RPKM-or-FPKM-instead-of-counts-for-10x-Genomics-data
- GENCODE mouse release M4, GRCm38.p3-compatible annotation: https://www.gencodegenes.org/mouse/release_M4.html
- GENCODE mouse release history: https://www.gencodegenes.org/mouse/releases.html
