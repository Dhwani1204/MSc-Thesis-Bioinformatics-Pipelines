"""
ceRNA network analysis — lncRNA-miRNA-mRNA competing endogenous RNA
networks for target genes (GALR1/2/3 by default), pan-cancer (TCGA).

Pipeline:
  1. Load mRNA expression, lncRNA expression (subset via GENCODE lncRNA
     annotation), miRNA expression, and TCGA sample/cancer-type metadata.
  2. Correlate target-gene expression with every miRNA, per cancer type
     (Pearson) -> keep the significant, negatively-correlated pairs
     (candidate miRNA "sponges") -> intersect with TargetScan predicted
     targets for high-confidence gene-miRNA interactions.
  3. Correlate every lncRNA against each high-confidence miRNA within the
     same cancer type. This is the single largest computation in the
     pipeline, so results are written incrementally to disk rather than
     held in memory.
  4. Merge gene-miRNA and lncRNA-miRNA correlations into ceRNA triplets,
     export full + top-10-filtered Cytoscape edge/node tables.
  5. Visualize: hub miRNA/lncRNA degree heatmaps, per-cancer network
     diagrams, and a summary table (Cancer, Triplets, Unique_lncRNA,
     Unique_miRNA, Unique_genes).

Inputs (see DATA_DIR below):
    tcga_RSEM_gene_tpm_annotated.tsv
    pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena.gz
    gencode.v22.long_noncoding_RNAs.gtf
    TCGA_phenotype_denseDataOnlyDownload.tsv.gz
    selected_cancers.txt
    Nonconserved_Site_Context_Scores.txt   (TargetScan)

Outputs (written to OUTPUT_DIR):
    <DATASET>_miRNA_correlation_all.tsv / _significant.tsv / _regulatory.tsv
    High_confidence_miRNA_<DATASET>.tsv
    lncRNA_miRNA_cancer_specific_corr.tsv (+ _named, _significant)
    lncRNA_miRNA_<DATASET>_cerna_network(cancerwise).tsv
    <DATASET>_top10_filtered_network.tsv
    <DATASET>_network_summary.tsv
    cytoscape/  edge + node tables (full network and top-10-filtered network)
    hub_miRNA_heatmap.png, hub_lncRNA_heatmap.png
    networks/  one diagram per cancer type
"""

import os
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, ConstantInputWarning
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=ConstantInputWarning)

# ── Configuration — edit these for your own machine/dataset ───────────
DATA_DIR = "data"
OUTPUT_DIR = "output"
DATASET = "GALR"  # label used in filenames/titles; e.g. "FZD_SMO" for the other pipeline
TARGET_GENES = ["GALR1", "GALR2", "GALR3"]

EXPRESSION_FILE = os.path.join(DATA_DIR, "tcga_RSEM_gene_tpm_annotated.tsv")
MIRNA_FILE = os.path.join(
    DATA_DIR, "pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena.gz"
)
LNCRNA_GTF_FILE = os.path.join(DATA_DIR, "gencode.v22.long_noncoding_RNAs.gtf")
PHENOTYPE_FILE = os.path.join(DATA_DIR, "TCGA_phenotype_denseDataOnlyDownload.tsv.gz")
SELECTED_CANCERS_FILE = os.path.join(DATA_DIR, "selected_cancers.txt")
TARGETSCAN_FILE = os.path.join(DATA_DIR, "Nonconserved_Site_Context_Scores.txt")

# Gene-miRNA correlation thresholds
MIN_SAMPLES_GENE_MIRNA = 20
SIGNIFICANT_PVAL = 0.05
SIGNIFICANT_ABS_CORR = 0.3
REGULATORY_CORR = -0.3   # ceRNA hypothesis: gene up when miRNA down
REGULATORY_PVAL = 0.02

# lncRNA-miRNA correlation thresholds
MIN_PATIENTS_PER_CANCER = 5
MIN_PAIR_N = 3
LNCRNA_SIGNIFICANT_CORR = -0.3
LNCRNA_SIGNIFICANT_FDR = 0.05

# Hub-network filtering ("top 10 miRNA per gene, then top 10 lncRNA per miRNA")
TOP_MIR_PER_CANCER_GENE = 10
TOP_LNC_PER_CANCER_MIR = 10

# Network-diagram appearance
NETWORK_DPI = 300
NETWORK_FORMAT = "png"
NETWORK_SEED = 42

TCGA_ABBREVIATIONS = {
    "breast invasive carcinoma": "BRCA",
    "lung adenocarcinoma": "LUAD",
    "lung squamous cell carcinoma": "LUSC",
    "colon adenocarcinoma": "COAD",
    "rectum adenocarcinoma": "READ",
    "prostate adenocarcinoma": "PRAD",
    "stomach adenocarcinoma": "STAD",
    "liver hepatocellular carcinoma": "LIHC",
    "kidney renal clear cell carcinoma": "KIRC",
    "kidney renal papillary cell carcinoma": "KIRP",
    "kidney chromophobe": "KICH",
    "bladder urothelial carcinoma": "BLCA",
    "uterine corpus endometrial carcinoma": "UCEC",
    "cervical squamous cell carcinoma": "CESC",
    "ovarian serous cystadenocarcinoma": "OV",
    "head and neck squamous cell carcinoma": "HNSC",
    "thyroid carcinoma": "THCA",
    "skin cutaneous melanoma": "SKCM",
    "glioblastoma multiforme": "GBM",
    "brain lower grade glioma": "LGG",
    "pancreatic adenocarcinoma": "PAAD",
    "esophageal carcinoma": "ESCA",
    "sarcoma": "SARC",
    "testicular germ cell tumors": "TGCT",
    "acute myeloid leukemia": "LAML",
    "diffuse large b-cell lymphoma": "DLBC",
    "mesothelioma": "MESO",
    "adrenocortical carcinoma": "ACC",
    "uveal melanoma": "UVM",
    "cholangiocarcinoma": "CHOL",
    "thymoma": "THYM",
    "pheochromocytoma & paraganglioma": "PCPG",
}


# ════════════════════════════════════════════════════════════════════
# Stage 1 — load data
# ════════════════════════════════════════════════════════════════════

def load_gencode_lncrna_ids(gtf_path: str) -> set:
    """Parse a GENCODE GTF and return the set of lncRNA Ensembl gene IDs (version stripped)."""
    lncrna_ids = set()
    with open(gtf_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if fields[2] != "gene":
                continue
            for attr in fields[8].split(";"):
                attr = attr.strip()
                if attr.startswith("gene_id"):
                    gene_id = attr.split('"')[1].split(".")[0]
                    lncrna_ids.add(gene_id)
    print("Total lncRNAs from GENCODE:", len(lncrna_ids))
    return lncrna_ids


def load_expression_matrices(expression_path: str, gtf_path: str):
    """Load the TCGA TPM matrix and split it into (gene-symbol matrix, lncRNA-Ensembl-ID matrix)."""
    expr = pd.read_csv(expression_path, sep="\t", index_col=0)
    expr["gene_clean"] = expr["sample"].str.split(".").str[0]
    print("mRNA matrix:", expr.shape)

    # Gene-symbol-indexed matrix, for target-gene lookups
    expr_values = expr.drop(columns=["hgnc_symbol", "gene_clean"], errors="ignore")
    expr_values.index = expr["hgnc_symbol"]
    expr_values = expr_values[~expr_values.index.duplicated(keep="first")]

    # lncRNA subset, via GENCODE annotation, patients as rows
    lncrna_ids = load_gencode_lncrna_ids(gtf_path)
    lnc = expr[expr["gene_clean"].isin(lncrna_ids)].set_index("gene_clean")
    lnc = lnc.drop(columns=["gene", "hgnc_symbol", "sample"], errors="ignore").T
    lnc.index.name = "Sample"
    lnc["Patient"] = lnc.index.str[:12]
    lnc = lnc.reset_index().drop_duplicates("Patient").set_index("Patient")
    print("lncRNA matrix:", lnc.shape)

    # Ensembl ID -> gene symbol mapping, used later to name lncRNA hits
    id_to_symbol = expr[["gene_clean", "hgnc_symbol"]].drop_duplicates().rename(
        columns={"gene_clean": "Ensembl_id", "hgnc_symbol": "lncRNA_symbol"}
    )

    return expr_values, lnc, id_to_symbol


def load_mirna_matrix(mirna_path: str) -> pd.DataFrame:
    """Load the TCGA miRNA expression matrix, patients as rows."""
    mir = pd.read_csv(mirna_path, sep="\t", compression="gzip").set_index("sample").T
    mir.index.name = "Sample"
    print("miRNA matrix:", mir.shape)
    return mir


def load_metadata(phenotype_path: str, selected_cancers_path: str):
    """Load TCGA sample-level phenotype metadata and the list of cancer types to analyze."""
    meta = pd.read_csv(phenotype_path, sep="\t").rename(
        columns={"_primary_disease": "Cancer", "sample_type": "Type", "sample": "Sample"}
    )
    meta["Patient"] = meta["Sample"].str[:12]

    selected_cancers = pd.read_csv(selected_cancers_path, header=None)[0].tolist()
    meta_selected = meta[meta["Cancer"].isin(selected_cancers)].copy()
    print(f"Selected cancers ({len(selected_cancers)}):", selected_cancers)
    return meta_selected, selected_cancers


def align_gene_matrix(expr_values: pd.DataFrame, target_genes: list) -> pd.DataFrame:
    """Subset to target genes, transpose to patients-as-rows, dedupe patient barcodes."""
    genes_present = expr_values.index.intersection(target_genes)
    gene_matrix = expr_values.loc[genes_present].T
    gene_matrix.index = gene_matrix.index.str.slice(0, 12)
    gene_matrix = gene_matrix[~gene_matrix.index.duplicated()]
    return gene_matrix


# ════════════════════════════════════════════════════════════════════
# Stage 2 — target gene vs miRNA correlation, then TargetScan filtering
# ════════════════════════════════════════════════════════════════════

def correlate_gene_mirna(gene_matrix, mir, meta_selected, target_genes, selected_cancers):
    """Pearson-correlate each target gene against every miRNA, within each cancer type."""
    mir = mir.copy()
    mir.index = mir.index.str.slice(0, 12)
    mir = mir[~mir.index.duplicated()]
    meta_selected = meta_selected.copy()
    meta_selected["Patient"] = meta_selected["Patient"].str.slice(0, 12)
    meta_selected = meta_selected.drop_duplicates("Patient")

    common_patients = list(set(gene_matrix.index) & set(mir.index) & set(meta_selected["Patient"]))
    print("Common patients (gene x miRNA):", len(common_patients))

    expr_common = gene_matrix.loc[common_patients].apply(pd.to_numeric, errors="coerce")
    mir_common = mir.loc[common_patients].apply(pd.to_numeric, errors="coerce")
    meta_common = meta_selected[meta_selected["Patient"].isin(common_patients)]

    results = []
    print("Running gene x miRNA correlation...")
    for cancer in selected_cancers:
        cancer_patients = meta_common.loc[meta_common["Cancer"] == cancer, "Patient"]
        cancer_patients = cancer_patients[cancer_patients.isin(expr_common.index)]
        print(f"  {cancer}: {len(cancer_patients)} patients")

        for gene in target_genes:
            if gene not in expr_common.columns:
                continue
            gene_vals = expr_common.loc[cancer_patients, gene]
            for mirna in mir_common.columns:
                mir_vals = mir_common.loc[cancer_patients, mirna]
                valid = pd.concat([gene_vals, mir_vals], axis=1).dropna()
                if len(valid) > MIN_SAMPLES_GENE_MIRNA:
                    r, p = pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
                    results.append({"Cancer": cancer, "Gene": gene, "miRNA": mirna,
                                     "Correlation": r, "pvalue": p, "N_samples": len(valid)})

    results_df = pd.DataFrame(results)
    print("Total gene-miRNA correlations:", results_df.shape)
    return results_df


def filter_significant_and_regulatory(results_df: pd.DataFrame):
    """Split gene-miRNA correlations into 'significant' and the stricter 'regulatory' subset.

    'Regulatory' requires a negative correlation (gene up when miRNA down, consistent
    with the ceRNA/sponge hypothesis) at a tighter p-value than 'significant'.
    """
    significant = results_df[
        (results_df["pvalue"] < SIGNIFICANT_PVAL)
        & (results_df["Correlation"].abs() > SIGNIFICANT_ABS_CORR)
    ]
    regulatory = results_df[
        (results_df["Correlation"] < REGULATORY_CORR) & (results_df["pvalue"] < REGULATORY_PVAL)
    ]
    return significant, regulatory


def load_targetscan_highconfidence(targetscan_path: str, target_genes: list, regulatory_df: pd.DataFrame):
    """Intersect regulatory gene-miRNA correlations with TargetScan predicted targets."""
    targetscan = pd.read_csv(targetscan_path, sep="\t")
    targetscan_targets = targetscan[targetscan["Gene Symbol"].isin(target_genes)]
    print("TargetScan interactions for target genes:", targetscan_targets.shape)

    high_confidence = regulatory_df.merge(
        targetscan_targets, left_on=["Gene", "miRNA"], right_on=["Gene Symbol", "miRNA"], how="inner"
    )
    print("High-confidence gene-miRNA interactions:", high_confidence.shape)
    return high_confidence


# ════════════════════════════════════════════════════════════════════
# Stage 3 — lncRNA vs miRNA correlation (large computation, streamed to disk)
# ════════════════════════════════════════════════════════════════════

def correlate_lncrna_mirna(lnc, mir, meta_selected, high_confidence_pairs, out_path):
    """For each (cancer, miRNA) pair from Stage 2, correlate every lncRNA against that miRNA.

    This is the largest computation in the pipeline (lncRNAs x pairs), so results
    are appended to `out_path` line-by-line rather than accumulated in memory.
    """
    common_patients = list(set(lnc.index) & set(mir.index) & set(meta_selected["Patient"]))
    print("Common patients (lncRNA x miRNA):", len(common_patients))

    lnc_common = lnc.loc[common_patients].apply(pd.to_numeric, errors="coerce")
    mir_common = mir.loc[common_patients].apply(pd.to_numeric, errors="coerce")
    meta_common = meta_selected[meta_selected["Patient"].isin(common_patients)]

    pairs = high_confidence_pairs[["Cancer", "miRNA"]].drop_duplicates()
    print("Cancer x miRNA pairs to process:", len(pairs))

    with open(out_path, "w") as f:
        f.write("Cancer\tlncRNA\tmiRNA\tCorrelation\tPvalue\n")

    lncrna_ids = lnc_common.columns
    for pair_index, row in enumerate(pairs.itertuples(), 1):
        cancer, mirna = row.Cancer, row.miRNA
        print(f"Processing pair {pair_index}/{len(pairs)}: {cancer} | {mirna}")

        if mirna not in mir_common.columns:
            continue

        cancer_patients = meta_common.loc[meta_common["Cancer"] == cancer, "Patient"]
        cancer_patients = [p for p in cancer_patients if p in lnc_common.index]
        if len(cancer_patients) < MIN_PATIENTS_PER_CANCER:
            continue

        lnc_sub = lnc_common.loc[cancer_patients]
        mir_vector = mir_common.loc[cancer_patients, mirna]

        with open(out_path, "a") as f:
            for i, lnc_id in enumerate(lncrna_ids):
                if i % 1000 == 0:
                    print(f"   lncRNA {i}/{len(lncrna_ids)}")
                aligned = pd.concat([lnc_sub[lnc_id], mir_vector], axis=1).dropna()
                if len(aligned) < MIN_PAIR_N:
                    continue
                corr, p = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                f.write(f"{cancer}\t{lnc_id}\t{mirna}\t{corr}\t{p}\n")

    print("lncRNA-miRNA correlation complete:", out_path)


def name_and_filter_lncrna_mirna(raw_path, id_to_symbol, out_dir):
    """Map lncRNA Ensembl IDs to gene symbols, apply global BH-FDR, and keep significant hits."""
    results_df = pd.read_csv(raw_path, sep="\t")
    results_df = results_df.merge(id_to_symbol, left_on="lncRNA", right_on="Ensembl_id", how="left")
    results_df["lncRNA"] = results_df["lncRNA_symbol"]
    results_df = results_df.drop(columns=["Ensembl_id", "lncRNA_symbol"])
    named_path = os.path.join(out_dir, "lncRNA_miRNA_cancer_specific_corr_named.tsv")
    results_df.to_csv(named_path, sep="\t", index=False)

    results_df = results_df.dropna(subset=["Pvalue"])
    results_df["FDR"] = multipletests(results_df["Pvalue"].values, method="fdr_bh")[1]
    print("Correlation summary:\n", results_df["Correlation"].describe())
    print("FDR summary:\n", results_df["FDR"].describe())

    significant = results_df[
        (results_df["Correlation"] < LNCRNA_SIGNIFICANT_CORR)
        & (results_df["FDR"] < LNCRNA_SIGNIFICANT_FDR)
    ]
    print("Significant lncRNA-miRNA pairs:", significant.shape)
    significant.to_csv(os.path.join(out_dir, "lncRNA_miRNA_significant.tsv"), sep="\t", index=False)
    return significant


# ════════════════════════════════════════════════════════════════════
# Stage 4 — build ceRNA triplets, export Cytoscape tables
# ════════════════════════════════════════════════════════════════════

def build_cerna_network(lnc_mir_significant: pd.DataFrame, gene_mir_regulatory: pd.DataFrame) -> pd.DataFrame:
    """Join significant lncRNA-miRNA pairs with regulatory gene-miRNA pairs -> ceRNA triplets."""
    cerna = lnc_mir_significant.merge(gene_mir_regulatory, on=["Cancer", "miRNA"], how="inner")
    print("ceRNA triplets:", cerna.shape)
    return cerna


def export_cytoscape_tables(cerna: pd.DataFrame, gene_col: str, out_dir: str, subdir: str):
    """Write per-cancer edge tables (lncRNA->miRNA, miRNA->gene) and a combined node table."""
    outdir = os.path.join(out_dir, "cytoscape", subdir)
    os.makedirs(outdir, exist_ok=True)

    lnc_mir_edges = cerna[["Cancer", "lncRNA", "miRNA"]].copy()
    lnc_mir_edges["source"], lnc_mir_edges["target"], lnc_mir_edges["interaction"] = (
        lnc_mir_edges["lncRNA"], lnc_mir_edges["miRNA"], "lncRNA-miRNA",
    )
    mir_gene_edges = cerna[["Cancer", "miRNA", gene_col]].copy()
    mir_gene_edges["source"], mir_gene_edges["target"], mir_gene_edges["interaction"] = (
        mir_gene_edges["miRNA"], mir_gene_edges[gene_col], "miRNA-gene",
    )
    edges = pd.concat([
        lnc_mir_edges[["Cancer", "source", "target", "interaction"]],
        mir_gene_edges[["Cancer", "source", "target", "interaction"]],
    ])

    for cancer, df in edges.groupby("Cancer"):
        name = cancer.replace(" ", "_")
        df[["source", "target", "interaction"]].to_csv(
            os.path.join(outdir, f"{name}_ceRNA_network.tsv"), sep="\t", index=False
        )

    nodes = pd.concat([
        pd.DataFrame({"node": cerna["lncRNA"].unique(), "type": "lncRNA"}),
        pd.DataFrame({"node": cerna["miRNA"].unique(), "type": "miRNA"}),
        pd.DataFrame({"node": cerna[gene_col].unique(), "type": "gene"}),
    ]).drop_duplicates()
    nodes.to_csv(os.path.join(outdir, "cytoscape_nodes_all.tsv"), sep="\t", index=False)
    print(f"Cytoscape tables exported to {outdir}")


def apply_top_n_filter(cerna: pd.DataFrame, gene_col: str, top_mir_n: int, top_lnc_n: int) -> pd.DataFrame:
    """Keep only the top-N miRNA per (cancer, gene), then the top-N lncRNA per (cancer, gene, miRNA).

    Ranking is by connection count (degree), so this keeps each gene's most-connected
    "hub" miRNAs, and each miRNA's most-connected "hub" lncRNAs.
    """
    mir_rank = cerna.groupby(["Cancer", gene_col, "miRNA"]).size().reset_index(name="lnc_count")
    top_mir = (
        mir_rank.sort_values(["Cancer", gene_col, "lnc_count"], ascending=[True, True, False])
        .groupby(["Cancer", gene_col]).head(top_mir_n)
    )
    cerna_f = cerna.merge(top_mir[["Cancer", gene_col, "miRNA"]], on=["Cancer", gene_col, "miRNA"])
    cerna_f = cerna_f.drop_duplicates(subset=["Cancer", gene_col, "miRNA", "lncRNA"])
    print(f"After top-{top_mir_n}-miRNA filter: {cerna_f.shape[0]} triplets")

    lnc_rank = cerna_f.groupby(["Cancer", gene_col, "miRNA", "lncRNA"]).size().reset_index(name="count")
    top_lnc = (
        lnc_rank.sort_values(["Cancer", gene_col, "miRNA", "count"], ascending=[True, True, True, False])
        .groupby(["Cancer", gene_col, "miRNA"]).head(top_lnc_n)
    )
    final_df = cerna_f.merge(top_lnc[["Cancer", gene_col, "miRNA", "lncRNA"]],
                              on=["Cancer", gene_col, "miRNA", "lncRNA"])
    print(f"After top-{top_lnc_n}-lncRNA filter: {final_df.shape[0]} triplets")

    check_mir = top_mir.groupby(["Cancer", gene_col])["miRNA"].nunique()
    check_lnc = top_lnc.groupby(["Cancer", gene_col, "miRNA"])["lncRNA"].nunique()
    print(f"miRNA per cancer-gene — max: {check_mir.max()}, mean: {check_mir.mean():.1f}")
    print(f"lncRNA per cancer-gene-miRNA — max: {check_lnc.max()}, mean: {check_lnc.mean():.1f}")
    return final_df


# ════════════════════════════════════════════════════════════════════
# Stage 5 — visualization: hub heatmaps + per-cancer network diagrams
# ════════════════════════════════════════════════════════════════════

def plot_hub_heatmap(final_df: pd.DataFrame, node_col: str, title: str, out_path: str, top_n: int = 20):
    """Clustermap of the top-N most-connected nodes (by cancer type), log1p-scaled connection counts."""
    degree = final_df.groupby(["Cancer", node_col]).size().reset_index(name="count")
    top_nodes = (
        degree.groupby(node_col)["count"].sum().sort_values(ascending=False).head(top_n).index
    )
    pivot = degree[degree[node_col].isin(top_nodes)].pivot(index=node_col, columns="Cancer", values="count").fillna(0)
    pivot.columns = [TCGA_ABBREVIATIONS.get(c.lower().strip(), c) for c in pivot.columns]

    g = sns.clustermap(
        np.log1p(pivot), cmap="viridis", figsize=(max(8, pivot.shape[1] * 0.6), max(6, top_n * 0.35)),
        dendrogram_ratio=0.12, col_cluster=True, row_cluster=True,
        cbar_kws={"label": "log1p(connection count)"},
    )
    g.ax_heatmap.set_title(title, fontsize=13, pad=60)
    plt.savefig(out_path, dpi=NETWORK_DPI, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def draw_cancer_network(cancer: str, df: pd.DataFrame, gene_col: str, dataset: str, out_path: str):
    """Concentric-ring ceRNA network diagram for one cancer type: genes (center) -> miRNA (middle) -> lncRNA (outer)."""
    G = nx.DiGraph()
    for _, row in df.iterrows():
        lnc, mi, gene = str(row["lncRNA"]), str(row["miRNA"]), str(row[gene_col])
        G.add_node(lnc, node_type="lncRNA")
        G.add_node(mi, node_type="miRNA")
        G.add_node(gene, node_type="gene")
        G.add_edge(lnc, mi, edge_type="lncRNA-miRNA")
        G.add_edge(mi, gene, edge_type="miRNA-gene")

    if G.number_of_nodes() == 0:
        print(f"  [{cancer}] empty, skipped.")
        return

    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
    gene_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "gene"]
    mir_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "miRNA"]
    lnc_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "lncRNA"]

    # Concentric ring layout: genes centered, miRNA in a middle ring, lncRNA
    # in an outer ring grouped near the miRNA they connect to.
    pos = {}
    for i, g in enumerate(gene_nodes):
        angle = 2 * np.pi * i / max(len(gene_nodes), 1)
        pos[g] = (0.15 * np.cos(angle), 0.15 * np.sin(angle))
    for i, m in enumerate(mir_nodes):
        angle = 2 * np.pi * i / max(len(mir_nodes), 1)
        pos[m] = (0.65 * np.cos(angle), 0.65 * np.sin(angle))

    lnc_to_mir = {str(row["lncRNA"]): str(row["miRNA"]) for _, row in df.iterrows()}
    mir_index = {m: i for i, m in enumerate(mir_nodes)}
    lnc_by_mir = {}
    for lnc in lnc_nodes:
        lnc_by_mir.setdefault(lnc_to_mir.get(lnc, mir_nodes[0]), []).append(lnc)
    for mi, lncs in lnc_by_mir.items():
        base_angle = 2 * np.pi * mir_index.get(mi, 0) / max(len(mir_nodes), 1)
        spread = np.pi / max(len(mir_nodes), 1)
        for j, lnc in enumerate(lncs):
            offset = spread * (j - len(lncs) / 2) / max(len(lncs), 1)
            angle = base_angle + offset
            radius = 1.2 + 0.15 * (j % 3)
            pos[lnc] = (radius * np.cos(angle), radius * np.sin(angle))

    fig_size = max(14, min(28, n_nodes * 0.28))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    fig.patch.set_facecolor("white")

    for etype, color, alpha, lw in [("lncRNA-miRNA", "#BBBBBB", 0.5, 0.8), ("miRNA-gene", "#E84C4C", 0.8, 1.5)]:
        elist = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == etype]
        if elist:
            nx.draw_networkx_edges(G, pos, edgelist=elist, ax=ax, edge_color=color, alpha=alpha,
                                    arrows=True, arrowsize=10, arrowstyle="-|>",
                                    connectionstyle="arc3,rad=0.05", width=lw,
                                    min_source_margin=10, min_target_margin=10)

    for ntype, marker, color, size in [("lncRNA", "o", "#FFB6C1", 600), ("miRNA", "D", "#2ECC71", 1200),
                                        ("gene", "s", "#E84C4C", 3000)]:
        nlist = [n for n, d in G.nodes(data=True) if d["node_type"] == ntype]
        if nlist:
            nx.draw_networkx_nodes(G, pos, nodelist=nlist, ax=ax, node_color=color, node_size=size,
                                    node_shape=marker, alpha=0.93, linewidths=0.8, edgecolors="white")

    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: n for n in gene_nodes},
                             font_size=14, font_weight="bold", font_color="white")
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: n for n in mir_nodes},
                             font_size=12, font_weight="bold", font_color="#0D3D1F")
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: n for n in lnc_nodes},
                             font_size=12, font_color="#6D0025")

    legend_elements = [
        mpatches.Patch(facecolor="#FFB6C1", edgecolor="white", label=f"lncRNA  (n={len(lnc_nodes)})"),
        mpatches.Patch(facecolor="#2ECC71", edgecolor="white", label=f"miRNA   (n={len(mir_nodes)})"),
        mpatches.Patch(facecolor="#E84C4C", edgecolor="white", label=f"{dataset} gene (n={len(gene_nodes)})"),
        Line2D([0], [0], color="#BBBBBB", lw=1.5, label="lncRNA → miRNA"),
        Line2D([0], [0], color="#E84C4C", lw=1.5, label="miRNA → gene"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.9,
              edgecolor="#CCCCCC", fancybox=True, title=f"{dataset} ceRNA", title_fontsize=11)
    ax.set_title(f"{dataset} ceRNA Network — {cancer}\n"
                 f"top-{TOP_MIR_PER_CANCER_GENE} miRNA × top-{TOP_LNC_PER_CANCER_MIR} lncRNA"
                 f"   |   {n_nodes} nodes, {n_edges} edges", fontsize=16, fontweight="bold", pad=14)
    ax.axis("off")

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    ax.set_xlim(min(all_x) - 0.3, max(all_x) + 0.3)
    ax.set_ylim(min(all_y) - 0.3, max(all_y) + 0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=NETWORK_DPI, bbox_inches="tight", format=NETWORK_FORMAT)
    plt.close(fig)
    print(f"  [{cancer}] saved -> {os.path.basename(out_path)} ({n_nodes} nodes, {n_edges} edges)")


def draw_all_networks_and_summary(final_df: pd.DataFrame, gene_col: str, dataset: str, out_dir: str):
    """Draw one network diagram per cancer type, and write the Cancer/Triplets/Unique_* summary table."""
    networks_dir = os.path.join(out_dir, "networks")
    os.makedirs(networks_dir, exist_ok=True)

    cancers = sorted(final_df["Cancer"].unique())
    print(f"\nDrawing networks for {len(cancers)} cancers...")
    records = []
    for i, cancer in enumerate(cancers, 1):
        df = final_df[final_df["Cancer"] == cancer]
        if df.empty:
            continue
        print(f"Drawing {i}/{len(cancers)}: {cancer}")
        safe = cancer.replace(" ", "_").replace("/", "-")
        out = os.path.join(networks_dir, f"{dataset}_{safe}_ceRNA.{NETWORK_FORMAT}")
        draw_cancer_network(cancer, df, gene_col, dataset, out)
        records.append({
            "Cancer": cancer, "Triplets": len(df),
            "Unique_lncRNA": df["lncRNA"].nunique(),
            "Unique_miRNA": df["miRNA"].nunique(),
            "Unique_genes": df[gene_col].nunique(),
        })

    summary = pd.DataFrame(records).sort_values("Triplets", ascending=False)
    summary.to_csv(os.path.join(out_dir, f"{dataset}_network_summary.tsv"), sep="\t", index=False)
    print("\nSummary:\n", summary.to_string(index=False))
    return summary


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Stage 1
    expr_values, lnc, id_to_symbol = load_expression_matrices(EXPRESSION_FILE, LNCRNA_GTF_FILE)
    mir = load_mirna_matrix(MIRNA_FILE)
    meta_selected, selected_cancers = load_metadata(PHENOTYPE_FILE, SELECTED_CANCERS_FILE)
    gene_matrix = align_gene_matrix(expr_values, TARGET_GENES)

    # Stage 2
    gene_mir_corr = correlate_gene_mirna(gene_matrix, mir, meta_selected, TARGET_GENES, selected_cancers)
    gene_mir_corr.to_csv(os.path.join(OUTPUT_DIR, f"{DATASET}_miRNA_correlation_all.tsv"), sep="\t", index=False)
    significant, regulatory = filter_significant_and_regulatory(gene_mir_corr)
    significant.to_csv(os.path.join(OUTPUT_DIR, f"{DATASET}_miRNA_correlation_significant.tsv"), sep="\t", index=False)
    regulatory.to_csv(os.path.join(OUTPUT_DIR, f"{DATASET}_miRNA_correlation_regulatory.tsv"), sep="\t", index=False)
    high_confidence = load_targetscan_highconfidence(TARGETSCAN_FILE, TARGET_GENES, regulatory)
    high_confidence.to_csv(os.path.join(OUTPUT_DIR, f"High_confidence_miRNA_{DATASET}.tsv"), sep="\t", index=False)

    # Stage 3
    raw_lnc_mir_path = os.path.join(OUTPUT_DIR, "lncRNA_miRNA_cancer_specific_corr.tsv")
    correlate_lncrna_mirna(lnc, mir, meta_selected, high_confidence, raw_lnc_mir_path)
    lnc_mir_significant = name_and_filter_lncrna_mirna(raw_lnc_mir_path, id_to_symbol, OUTPUT_DIR)

    # Stage 4
    gene_col = "Gene_y" if "Gene_y" in high_confidence.columns else "Gene"
    cerna = build_cerna_network(lnc_mir_significant, high_confidence.rename(columns={gene_col: "Gene"}))
    cerna_path = os.path.join(OUTPUT_DIR, f"lncRNA_miRNA_{DATASET}_cerna_network(cancerwise).tsv")
    cerna.to_csv(cerna_path, sep="\t", index=False)
    export_cytoscape_tables(cerna, "Gene", OUTPUT_DIR, subdir="full_network")

    final_df = apply_top_n_filter(cerna, "Gene", TOP_MIR_PER_CANCER_GENE, TOP_LNC_PER_CANCER_MIR)
    final_df.to_csv(os.path.join(OUTPUT_DIR, f"{DATASET}_top10_filtered_network.tsv"), sep="\t", index=False)
    export_cytoscape_tables(final_df, "Gene", OUTPUT_DIR, subdir="top10_filtered")

    # Stage 5
    plot_hub_heatmap(final_df, "miRNA", f"Hub miRNAs — {DATASET}",
                      os.path.join(OUTPUT_DIR, "hub_miRNA_heatmap.png"))
    plot_hub_heatmap(final_df, "lncRNA", f"Hub lncRNAs — {DATASET}",
                      os.path.join(OUTPUT_DIR, "hub_lncRNA_heatmap.png"))
    draw_all_networks_and_summary(final_df, "Gene", DATASET, OUTPUT_DIR)

    print(f"\nAll done. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
