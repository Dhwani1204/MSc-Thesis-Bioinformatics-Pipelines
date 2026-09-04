"""
Survival analysis — GALR1/2/3 expression vs TCGA patient outcomes.

Pipeline:
  1. Univariate Cox proportional-hazards regression (per cancer type, per
     gene) -> forest plots of all results and of significant (p<0.05) hits.
  2. Kaplan-Meier curves (median-split high/low expression + log-rank test)
     for the univariate-significant cancer/gene pairs.
  3. Multivariate Cox regression, adjusting for age and tumor stage where
     available -> forest plots of significant hits.
  4. Tumor-stage vs expression violin plots (+ Kruskal-Wallis test) for the
     multivariate-significant cancer/gene pairs.
  5. Publication-style forest-plot-with-table figures.

Inputs (see DATA_DIR below):
    galr_expression.tsv                                TCGA TPM matrix, target genes only
    Survival_SupplementalTable_S1_20171025_xena_sp.tsv  UCSC Xena TCGA clinical/survival table
    TCGA_phenotype_denseDataOnlyDownload.tsv.gz         sample -> cancer type (full name)

Outputs (written to OUTPUT_DIR):
    <DATASET>_cox_univariate_all.csv / _significant.csv
    <DATASET>_cox_multivariate_all.csv / _significant.csv
    forest plots, KM curves, stage-violin plots (.png)

Note: the univariate stage uses the full cancer-type name from the TCGA
phenotype file ("breast invasive carcinoma"), while the multivariate/violin
stages use the "cancer type abbreviation" column already present in the
clinical table ("BRCA") — that's carried over as-is from the original
analysis; the two naming schemes aren't merged here, so a cancer type may
render differently in early vs. later plots.
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

# ── Configuration — edit these for your own machine/dataset ───────────
DATA_DIR = "data"
OUTPUT_DIR = "output"
DATASET = "GALR"
TARGET_GENES = ["GALR1", "GALR2", "GALR3"]

EXPRESSION_FILE = os.path.join(DATA_DIR, "galr_expression.tsv")
CLINICAL_FILE = os.path.join(DATA_DIR, "Survival_SupplementalTable_S1_20171025_xena_sp.tsv")
PHENOTYPE_FILE = os.path.join(DATA_DIR, "TCGA_phenotype_denseDataOnlyDownload.tsv.gz")

MIN_SAMPLES = 30           # minimum samples per cancer/gene group to fit a Cox model
MIN_KM_GROUP_SIZE = 10     # minimum samples per high/low group for a KM curve
COX_PVAL_CUTOFF = 0.05
MULTIVARIATE_PENALIZER = 0.1


# ════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════

def load_expression(path: str) -> pd.DataFrame:
    """Load the target-gene TPM matrix, patients (12-char barcode) as rows."""
    expr = pd.read_csv(path, sep="\t").set_index("gene_name").drop(columns=["sample"], errors="ignore").T
    expr.index = expr.index.str[:12]
    expr.index.name = "sample"
    print("Expression matrix:", expr.shape)
    return expr


def load_clinical(path: str) -> pd.DataFrame:
    """Load the Xena clinical/survival table with standardized column names."""
    clinical = pd.read_csv(path, sep="\t")
    clinical = clinical.rename(columns={
        "OS.time": "time",
        "OS": "event",
        "age_at_initial_pathologic_diagnosis": "age",
        "ajcc_pathologic_tumor_stage": "stage_raw",
        "cancer type abbreviation": "cancer_abbrev",
    })
    clinical["sample"] = clinical["sample"].str[:12]
    clinical = clinical.drop_duplicates("sample")
    clinical["age"] = pd.to_numeric(clinical["age"], errors="coerce")
    clinical["stage"] = simplify_stage(clinical["stage_raw"])
    return clinical


def load_phenotype_cancer_types(path: str) -> pd.DataFrame:
    """Load the full cancer-type name per sample, from the TCGA phenotype file."""
    pheno = pd.read_csv(path, sep="\t", low_memory=False)[["sample", "_primary_disease"]]
    pheno.columns = ["sample", "cancer_type"]
    pheno["sample"] = pheno["sample"].str[:12]
    return pheno.drop_duplicates("sample")


def simplify_stage(stage_series: pd.Series) -> pd.Series:
    """Extract Stage I-IV as an integer 1-4 from an AJCC tumor-stage string column."""
    upper = stage_series.astype(str).str.upper()
    roman = upper.str.extract(r"(I{1,3}|IV)", expand=False)
    return roman.map({"I": 1, "II": 2, "III": 3, "IV": 4})


# ════════════════════════════════════════════════════════════════════
# Stage 1 — univariate Cox regression
# ════════════════════════════════════════════════════════════════════

def run_univariate_cox(merged: pd.DataFrame, genes: list, cancer_col: str, min_samples: int) -> pd.DataFrame:
    results = []
    for cancer in merged[cancer_col].unique():
        df_cancer = merged[merged[cancer_col] == cancer]
        for gene in genes:
            df = df_cancer[[gene, "time", "event"]].dropna()
            if len(df) < min_samples:
                continue
            cph = CoxPHFitter()
            try:
                cph.fit(df, duration_col="time", event_col="event")
                s = cph.summary.loc[gene]
                results.append({"Cancer": cancer, "Gene": gene, "HR": s["exp(coef)"],
                                 "CI_low": s["exp(coef) lower 95%"], "CI_high": s["exp(coef) upper 95%"],
                                 "Pvalue": s["p"]})
            except Exception as e:
                print(f"  [{cancer}/{gene}] Cox fit failed: {e}")
    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════════
# Stage 2 — Kaplan-Meier curves for significant univariate hits
# ════════════════════════════════════════════════════════════════════

def plot_km_curves(merged: pd.DataFrame, sig_df: pd.DataFrame, cancer_col: str,
                    min_samples: int, min_group_size: int, out_dir: str):
    kmf_high, kmf_low = KaplanMeierFitter(), KaplanMeierFitter()

    for _, row in sig_df.iterrows():
        cancer, gene = row["Cancer"], row["Gene"]
        df = merged[merged[cancer_col] == cancer][[gene, "time", "event"]].dropna()
        if len(df) < min_samples:
            continue

        median = df[gene].median()
        df = df.copy()
        df["group"] = df[gene] > median
        high, low = df[df["group"]], df[~df["group"]]
        if len(high) < min_group_size or len(low) < min_group_size:
            continue

        res = logrank_test(high["time"], low["time"],
                            event_observed_A=high["event"], event_observed_B=low["event"])

        plt.figure(figsize=(5, 4))
        kmf_high.fit(high["time"], high["event"], label=f"High (n={len(high)})")
        kmf_high.plot(ci_show=False, linewidth=1.5, color="#d62728")
        kmf_low.fit(low["time"], low["event"], label=f"Low (n={len(low)})")
        kmf_low.plot(ci_show=False, linewidth=1.5, color="#1f77b4")

        plt.title(f"{gene} - {cancer}", fontsize=12, weight="bold")
        plt.xlabel("Time")
        plt.ylabel("Survival probability")
        plt.text(0.6, 0.1, f"p = {res.p_value:.3e}", transform=plt.gca().transAxes, fontsize=10)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.legend(frameon=False)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{gene}_{cancer}_KM.png"), dpi=300, bbox_inches="tight")
        plt.show()


# ════════════════════════════════════════════════════════════════════
# Stage 3 — multivariate Cox regression (age + stage adjusted)
# ════════════════════════════════════════════════════════════════════

def run_multivariate_cox(merged: pd.DataFrame, genes: list, cancer_col: str,
                          min_samples: int, penalizer: float) -> pd.DataFrame:
    results = []
    for cancer in merged[cancer_col].unique():
        df_cancer = merged[merged[cancer_col] == cancer]
        for gene in genes:
            df = df_cancer[[gene, "time", "event", "age", "stage"]].dropna()
            print(f"{cancer} - {gene} -> n = {len(df)}")
            if len(df) < min_samples:
                print("  skipped: too small")
                continue

            # drop covariates with no variance in this subgroup (Cox can't use them)
            if df["stage"].nunique() <= 1:
                df = df.drop(columns=["stage"])
            if df["age"].nunique() <= 1:
                df = df.drop(columns=["age"])

            cph = CoxPHFitter(penalizer=penalizer)
            try:
                cph.fit(df, duration_col="time", event_col="event")
                if gene not in cph.summary.index:
                    print("  gene missing from summary")
                    continue
                s = cph.summary.loc[gene]
                results.append({"Cancer": cancer, "Gene": gene, "HR": s["exp(coef)"],
                                 "CI_low": s["exp(coef) lower 95%"], "CI_high": s["exp(coef) upper 95%"],
                                 "Pvalue": s["p"]})
            except Exception as e:
                print("  ERROR:", e)
    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════════
# Shared forest-style plots
# ════════════════════════════════════════════════════════════════════

def plot_hr_errorbar(cox_df: pd.DataFrame, genes: list, title_suffix: str, out_dir: str, tag: str):
    """Simple HR errorbar plot, one figure per gene."""
    for gene in genes:
        df = cox_df[cox_df["Gene"] == gene].sort_values("HR")
        if df.empty:
            print(f"No {tag} results for {gene}")
            continue

        plt.figure(figsize=(6, max(4, len(df) * 0.5)))
        plt.errorbar(df["HR"], range(len(df)),
                     xerr=[df["HR"] - df["CI_low"], df["CI_high"] - df["HR"]], fmt="o")
        plt.yticks(range(len(df)), df["Cancer"])
        plt.axvline(1, linestyle="--")
        plt.title(f"{gene} Cox ({title_suffix})")
        plt.xlabel("Hazard Ratio (95% CI)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{gene}_{tag}_forest.png"), dpi=300, bbox_inches="tight")
        plt.show()


def plot_forest_table(cox_df: pd.DataFrame, gene_name: str, out_dir: str):
    """Forest plot with an inline HR(CI)/p-value table — publication style."""
    sub = cox_df[cox_df["Gene"] == gene_name].copy()
    if sub.empty:
        print(f"No data for {gene_name}")
        return
    sub["HR_CI"] = sub.apply(lambda x: f"{x['HR']:.2f} ({x['CI_low']:.2f}-{x['CI_high']:.2f})", axis=1)
    sub = sub.sort_values("HR")
    y_pos = range(len(sub))

    fig, ax = plt.subplots(figsize=(8, max(3, len(sub) * 0.6)))
    ax.hlines(y=y_pos, xmin=sub["CI_low"], xmax=sub["CI_high"], color="black", linewidth=1)
    colors = ["#d73027" if hr > 1 else "#4575b4" for hr in sub["HR"]]
    ax.scatter(sub["HR"], y_pos, color=colors, zorder=3, s=40)
    ax.axvline(x=1, linestyle="--", color="black")

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["Cancer"], fontsize=10)

    ax.text(0.02, len(sub) + 0.5, "HR (95% CI)", fontsize=11, weight="bold")
    ax.text(max(sub["CI_high"]) + 0.3, len(sub) + 0.5, "p-value", fontsize=11, weight="bold")
    for i in range(len(sub)):
        ax.text(0.02, i, sub["HR_CI"].iloc[i], va="center", fontsize=9)
        ax.text(max(sub["CI_high"]) + 0.3, i, f"{sub['Pvalue'].iloc[i]:.1e}", va="center", fontsize=9)

    ax.set_xlabel("Better survival                                 Poor survival", fontsize=11)
    ax.set_xlim(0, max(sub["CI_high"]) + 1)
    ax.set_title(f"{gene_name} – Multivariate Cox", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"forest_{gene_name}_table.png"), dpi=300, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════════════
# Stage 4 — tumor-stage violin plots for multivariate-significant hits
# ════════════════════════════════════════════════════════════════════

def plot_stage_violins(merged: pd.DataFrame, sig_df: pd.DataFrame, out_dir: str):
    merged = merged.copy()
    merged.columns = merged.columns.str.strip()

    for _, row in sig_df.iterrows():
        cancer = row["Cancer"]
        gene = row["Gene"].strip()
        print(f"\nProcessing: {cancer} - {gene}")

        df = merged[merged["cancer_abbrev"] == cancer].copy()
        if df.empty or gene not in df.columns:
            print(f"  skipped: no data / {gene} not found")
            continue

        df = df[[gene, "stage"]].dropna()
        if df.empty:
            print("  skipped: no data after dropna")
            continue

        stages = sorted(df["stage"].dropna().unique())
        plt.figure(figsize=(6, 5))
        sns.violinplot(x="stage", y=gene, data=df, order=stages, inner="box",
                        linewidth=1.2, color="#2c7fb8")

        groups = [df[df["stage"] == s][gene] for s in stages]
        if len(groups) >= 2:
            stat, p = kruskal(*groups)
            plt.text(0.5, 0.92, f"Kruskal p = {p:.3e}", transform=plt.gca().transAxes,
                      ha="center", fontsize=11)

        plt.xticks(range(len(stages)), [f"Stage {int(s)}" for s in stages])
        plt.title(f"{gene} – {cancer}", fontsize=14, weight="bold")
        plt.xlabel("Tumor stage")
        plt.ylabel("Expression")
        sns.despine()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{gene}_{cancer}_stage_violin.png"), dpi=300, bbox_inches="tight")
        plt.show()


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    expr = load_expression(EXPRESSION_FILE)
    clinical = load_clinical(CLINICAL_FILE)
    pheno = load_phenotype_cancer_types(PHENOTYPE_FILE)

    # ── Stage 1: univariate Cox (uses full cancer-type name from phenotype file) ──
    merged_uni = expr.merge(clinical[["sample", "time", "event"]], left_index=True, right_on="sample")
    merged_uni = merged_uni.merge(pheno, on="sample")
    print("Univariate merge:", merged_uni.shape)

    cox_uni = run_univariate_cox(merged_uni, TARGET_GENES, "cancer_type", MIN_SAMPLES)
    cox_uni.to_csv(os.path.join(OUTPUT_DIR, f"{DATASET}_cox_univariate_all.csv"), index=False)
    plot_hr_errorbar(cox_uni, TARGET_GENES, "Unadjusted", OUTPUT_DIR, tag="univariate_all")

    sig_uni = cox_uni[cox_uni["Pvalue"] < COX_PVAL_CUTOFF]
    sig_uni.to_csv(os.path.join(OUTPUT_DIR, f"{DATASET}_cox_univariate_significant.csv"), index=False)
    plot_hr_errorbar(sig_uni, TARGET_GENES, "Unadjusted, significant", OUTPUT_DIR, tag="univariate_sig")

    # ── Stage 2: KM curves for univariate-significant hits ──
    plot_km_curves(merged_uni, sig_uni, "cancer_type", MIN_SAMPLES, MIN_KM_GROUP_SIZE, OUTPUT_DIR)

    # ── Stage 3: multivariate Cox, age + stage adjusted (uses cancer abbreviation) ──
    merged_multi = expr.merge(clinical, left_index=True, right_on="sample")
    print("Multivariate merge:", merged_multi.shape)

    cox_multi = run_multivariate_cox(merged_multi, TARGET_GENES, "cancer_abbrev", MIN_SAMPLES,
                                      MULTIVARIATE_PENALIZER)
    cox_multi.to_csv(os.path.join(OUTPUT_DIR, f"{DATASET}_cox_multivariate_all.csv"), index=False)

    sig_multi = cox_multi[cox_multi["Pvalue"] < COX_PVAL_CUTOFF]
    sig_multi.to_csv(os.path.join(OUTPUT_DIR, f"{DATASET}_cox_multivariate_significant.csv"), index=False)
    plot_hr_errorbar(sig_multi, TARGET_GENES, "Adjusted", OUTPUT_DIR, tag="multivariate_sig")

    # ── Stage 4: stage violin plots for multivariate-significant hits ──
    plot_stage_violins(merged_multi, sig_multi, OUTPUT_DIR)

    # ── Publication-style forest-plot-with-table, one per gene with results ──
    for gene in cox_multi["Gene"].unique():
        plot_forest_table(sig_multi, gene, OUTPUT_DIR)

    print(f"\nAll done. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
