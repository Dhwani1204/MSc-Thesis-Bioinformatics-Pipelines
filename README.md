# MSc-Thesis-Bioinformatics-Pipelines
 Data analysis, statistical workflows, and visualization scripts developed for my Master's thesis.
# Decoding Galanin Receptor Oncobiology: A Multi-Omics Pan-Cancer Characterisation of Galanin Receptors

This repository hosts the complete bioinformatics framework, data preprocessing pipelines, and statistical analysis workflows developed for my Master's thesis. The project provides an end-to-end, multi-omics pan-cancer characterisation of Galanin Receptors (GALR1, GALR2, and GALR3) across 33 cancer types using high-throughput data from major public repositories.

## 🧬 Project Overview
Galanin receptors play complex, context-dependent roles in oncogenesis. This project integrates transcriptional profiling, DNA methylation architectures, copy number variations (CNV), and immune microenvironment infiltration patterns to decode the functional significance and prognostic values of GALR genes using data from **The Cancer Genome Atlas (TCGA)**, **DepMap (CTRPv2)**, and the **TIMER2.0** database.

## 🛠️ Technical Stack & Implementation
The computational framework seamlessly bridges wet-lab domain context with robust dry-lab engineering:
* **Python Pipelines:** Utilized for survival analysis using `lifelines` (Cox proportional hazards modeling), data manipulations via `Pandas` and `NumPy`, multiple testing corrections with `statsmodels`, and custom plotting with `matplotlib`.
* **OS Environment:** Executed entirely inside a native `Linux / Bash` terminal environment to manage and preprocess heavy genomic data matrices.

## 📊 Detailed Pipeline Framework

### 1. Survival & Prognostic Modeling (`Python / lifelines`)
* Extracted RSEM-normalized TPM matrices and applied a $\log_2(\text{TPM} + 0.001)$ transformation.
* Computed **univariate Cox proportional hazard regressions** mapping overall survival time (`OS.time`) and status (`OS`).
* Performed median-stratification to isolate high vs. low expression cohorts, applying a global **Benjamini-Hochberg FDR** correction across all gene-cancer combinations.

### 2. Immune Infiltration & EMT Analysis (`Python`)
* Procured the `HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION` gene set from MSigDB to measure individual sample **EMT Z-scores**.
* Merged multi-algorithm immune deconvolution metrics (from TIMER, CIBERSORT, EPIC, xCELL, etc.) using TCGA barcodes.
* Employed **partial Spearman correlation** backed by **Ordinary Least Squares (OLS) regression** to completely control and regress out the confounding effects of tumor purity.

### 3. Competitive Endogenous RNA (ceRNA) Network Construction (`Python`)
* **miRNA-Target Prediction:** Integrated downstream miRNA expression data with matching TCGA clinical matrices. Utilized target prediction algorithms (such as *TargetScan / miRDB / miRTarBase*) to map potential upstream miRNAs binding to GALR1, GALR2, and GALR3.
* **lncRNA Sponging Activity:** Identified corresponding Long Non-Coding RNAs (lncRNAs) sharing identical miRNA response elements (MREs) with GALR transcripts to isolate potential competitive sponging mechanisms.
* **Network Crosstalk Modeling:** Calculated negative Spearman correlation values for lncRNA-miRNA pairs and positive correlation values for lncRNA-mRNA pairs to validate true ceRNA behavior (where lncRNA expression rescues GALR transcripts from miRNA-mediated degradation).
* **Native Network Visualisation in Python:** Constructed, modeled, and visualized the multi-node regulatory layouts (lncRNA-miRNA-mRNA crosstalk) directly in Python using `NetworkX` for graph structures and topological metrics, alongside `matplotlib` or `seaborn` for generating clean network plots.
---

## 📂 Repository Layout
* `/Python_Scripts/` — Scripts containing the `coxPHFitter` loop, FDR calculations, and forest plot generators.

*Note: Raw TCGA/DepMap/TIMER data matrices are publicly accessible via their respective portals and are not hosted directly within this repository to preserve data compliance.*
