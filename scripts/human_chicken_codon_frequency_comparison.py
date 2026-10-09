#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Compare codon frequencies between human and chicken for 1:1 orthologous
genes, separately for human autosomal and human X-linked genes.

Pipeline:
  1. Load the human-chicken 1:1 ortholog table
     (general/genes_1to1_ortholog_human_chicken/tables/
     human_chicken_one2one_orthologs.csv), the per-gene codon frequencies of
     both species (tables/human/codon_frequencies_human_liver.csv and
     tables/chicken/codon_frequencies_chicken_liver.csv) and both CDS tables
     (human: the MANE Select CDS table human_co_mega_analysis.py uses;
     chicken: data/chicken_CDS_sequence.csv). Joins are ID-based (human
     `ensembl_gene_id`, chicken `gene_id`); both are audited.
  2. Keep ortholog pairs present in both frequency tables and split them by
     the HUMAN gene's chromosome: autosomal (chr 1-22) vs X-linked. (Y-linked
     human genes essentially have no chicken ortholog, and chicken has no Y.)
  3. For each ortholog pair and codon: log2(human codon freq / chicken codon
     freq); the value plotted for a gene type is the MEAN of that across the
     ortholog pairs in the type. A pair whose frequency is 0 in either species
     has no defined log2 for that codon and is left out of that codon's mean
     (the number of pairs used per codon is saved as `n_pairs_log2_defined`;
     the log2 of the ratio of the mean frequencies is also saved, as
     `log2_ratio_of_mean_freq`, for reference).
  4. Chi-square tests (human vs chicken, one set per group):
       - Omnibus: chi-square test of homogeneity on the 61 x 2 table of codon
         counts (human vs chicken).
       - Per codon: 2 x 2 chi-square test (this codon vs. all other codons,
         human vs chicken), Benjamini-Hochberg FDR across the 61 codons.
     Counts are made directly from the CDS sequences (in-frame 3-mers of the
     61 sense codons, last/stop codon dropped) and summed over genes.
     CAVEAT: pooling codons across genes treats them as independent, which
     they are not, so the chi-square p-values are anti-conservative. Since
     orthologs are PAIRED, each codon also gets a gene-level paired Wilcoxon
     signed-rank test on (human freq - chicken freq) per ortholog pair (each
     pair counts once) as a robustness check, and a Wilcoxon test of the
     per-pair log2 ratios against 0 (`log2_ratio_wilcoxon_p_value`/`_fdr`),
     the test that matches the plotted quantity. The * in the heatmaps is the
     pooled-count chi-square FDR, which can disagree with the plotted mean.
  5. Clustered heatmaps (rows = gene types, columns = codons; both clustered
     hierarchically, average linkage on Euclidean distance) of the mean
     per-pair log2(human / chicken codon freq), red = enriched in human, * =
     per-codon chi-square FDR < 0.05. A color bar above the codons marks
     codons ending in A/T (blue) vs G/C (yellow). One figure per comparison:
       a. Autosome vs X
       b. Autosome vs Xi-expressed vs Xi-silent vs NPX-NPY vs PAR1 vs PAR2
       c. Autosome vs Xi-silent vs Xi-expressed
       (b and c for both the Gylemo and the Neha XCI classification; X genes
        are matched to the classification by gene name, as elsewhere in the
        pipeline)
       d. Autosome vs each X evolutionary stratum (S5, S4, S3, Ambiguous S3,
          Ambiguous S1, S1; "Human Stratum" in
          analysis_riboseq/tables/genes_ancestral_x_by_strata_Winston.csv,
          matched on human Ensembl gene ID); plus versions of d restricted to
          Xi-silent genes only and to Xi-expressed genes only (each for both
          XCI classifications)
     Gene types with no ortholog pairs are left out; each row label shows its n (some are tiny, so read them with care). If no
     pair of a gene type has a defined log2 for a codon, that cell is shown
     as 0 (and not starred) so the clustering can run.

Outputs are saved under tables/human_chicken_comparison/ and
figures/human_chicken_comparison/ (prefix `codon_frequency_`).
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from co_mega import SENSE_CODONS
from co_mega_common import audit_join, check_key_uniqueness

BASE_DIR = "/lab/solexa_page/xueqi/analysis_codon"
ORTHOLOG_FILE = "/lab/solexa_page/xueqi/general/genes_1to1_ortholog_human_chicken/tables/human_chicken_one2one_orthologs.csv"
HUMAN_FREQ_FILE = os.path.join(BASE_DIR, "tables/human/codon_frequencies_human_liver.csv")
CHICKEN_FREQ_FILE = os.path.join(BASE_DIR, "tables/chicken/codon_frequencies_chicken_liver.csv")
HUMAN_CDS_FILE = "/lab/solexa_page/xueqi/general/sequence_CDS/human/tables/human_CDS_sequences_MANE_Select_ensembl.csv"
CHICKEN_CDS_FILE = os.path.join(BASE_DIR, "data/chicken_CDS_sequence.csv")
XCI_FILES = {
    # source label: (classification table, classifications to exclude)
    'Gylemo': ("/lab/solexa_page/xueqi/general/genes_XCI_Gylemo/tables/chrX_genes_classification_comparison_gylemo_liver_thresh0p05.csv", ()),
    'Neha':   ("/lab/solexa_page/xueqi/general/genes_XCI_Neha/tables/chrX_genes_classification_Neha.csv", ('No call',)),
}
STRATA_FILE = "/lab/solexa_page/xueqi/analysis_riboseq/tables/genes_ancestral_x_by_strata_Winston.csv"
STRATA = ['S5', 'S4', 'S3', 'Ambiguous S3', 'Ambiguous S1', 'S1']
TABLE_DIR = os.path.join(BASE_DIR, "tables/human_chicken_comparison")
FIG_DIR = os.path.join(BASE_DIR, "figures/human_chicken_comparison")

AUTOSOMES = [str(i) for i in range(1, 23)]
GROUPS = ["Autosome", "X"]
FDR_ALPHA = 0.05
MIN_PAIRS = 1        # drop gene types with fewer ortholog pairs than this (n is shown on every row label)
AT_COLOR = "#0019FC"  # codon ends in A/T
GC_COLOR = "#FFFF00"  # codon ends in G/C


def tbl(name):
    return os.path.join(TABLE_DIR, f"{name}.csv")


def fig(name):
    return os.path.join(FIG_DIR, f"{name}.png")


def codon_counts(freq_df, cds_df, key, label):
    """
    Per-gene codon counts of the 61 sense codons, counted directly from the
    CDS sequence (in-frame, non-overlapping 3-mers, last codon = stop codon
    dropped). Rows are aligned to `freq_df`.
    """
    df = freq_df[[key]].merge(cds_df[[key, 'CDS_sequence']], on=key, how='left', validate='one_to_one')
    if df['CDS_sequence'].isna().any():
        raise ValueError(f"{label}: {df['CDS_sequence'].isna().sum()} genes in the frequency table have no CDS")
    index = {c: i for i, c in enumerate(SENSE_CODONS)}
    counts = np.zeros((len(df), len(SENSE_CODONS)), dtype=np.int64)
    for r, seq in enumerate(df['CDS_sequence'].str.upper()):
        for k in range(0, len(seq) - 3, 3):  # len - 3: stop codon dropped
            i = index.get(seq[k:k + 3])
            if i is not None:
                counts[r, i] += 1
    counts = pd.DataFrame(counts, columns=SENSE_CODONS)
    # Consistency with the frequency table: freq = count / (# codons after dropping the stop).
    n_codons = (df['CDS_sequence'].str.len() // 3 - 1).to_numpy()
    max_diff = np.abs(counts.to_numpy() / n_codons[:, None] - freq_df[SENSE_CODONS].to_numpy()).max()
    print(f"  {label}: counted {int(counts.to_numpy().sum()):,} sense codons directly from {len(df):,} CDS "
          f"(max |count/n_codons - table frequency| = {max_diff:.2g})")
    return counts


def load_pairs():
    orth = pd.read_csv(ORTHOLOG_FILE, usecols=['Gene stable ID', 'Chicken gene stable ID']).rename(
        columns={'Gene stable ID': 'ensembl_gene_id', 'Chicken gene stable ID': 'chicken_gene_id'})
    orth = orth.dropna()
    print(f"  Loaded {len(orth):,} human-chicken 1:1 orthologs.")

    h_freq = pd.read_csv(HUMAN_FREQ_FILE, low_memory=False)
    c_freq = pd.read_csv(CHICKEN_FREQ_FILE, low_memory=False)
    h_cds = pd.read_csv(HUMAN_CDS_FILE, usecols=['ensembl_gene_id', 'CDS_sequence'])
    c_cds = pd.read_csv(CHICKEN_CDS_FILE, usecols=['gene_id', 'CDS_sequence'])
    check_key_uniqueness(h_freq, 'ensembl_gene_id', 'human codon frequencies')
    check_key_uniqueness(c_freq, 'gene_id', 'chicken codon frequencies')
    check_key_uniqueness(orth, 'ensembl_gene_id', 'ortholog table (human side)')
    check_key_uniqueness(orth, 'chicken_gene_id', 'ortholog table (chicken side)')

    h_counts = codon_counts(h_freq, h_cds, 'ensembl_gene_id', 'human')
    c_counts = codon_counts(c_freq, c_cds, 'gene_id', 'chicken')
    h = pd.concat([h_freq[['ensembl_gene_id', 'gene_name', 'chromosome', 'is_X', 'is_Y']],
                   h_freq[SENSE_CODONS].add_prefix('fh_'), h_counts.add_prefix('nh_')], axis=1)
    c = pd.concat([c_freq[['gene_id']].rename(columns={'gene_id': 'chicken_gene_id'}),
                   c_freq[SENSE_CODONS].add_prefix('fc_'), c_counts.add_prefix('nc_')], axis=1)

    m = audit_join(orth, h, 'ensembl_gene_id', 'ensembl_gene_id',
                   'human-chicken orthologs', 'human codon frequencies',
                   tbl("codon_frequency_join_audit_orthologs_vs_human"))
    m = audit_join(m, c, 'chicken_gene_id', 'chicken_gene_id',
                   'orthologs + human codon frequencies', 'chicken codon frequencies',
                   tbl("codon_frequency_join_audit_orthologs_vs_chicken"))
    m['group'] = np.where(m['is_X'] == 1, 'X', np.where(m['is_Y'] == 1, 'Y', 'Autosome'))
    print(f"  {len(m):,} ortholog pairs with codon frequencies in both species: "
          + ", ".join(f"{g} = {(m['group'] == g).sum():,}" for g in ['Autosome', 'X', 'Y']))
    return m


def compare_group(sub, group):
    """Human vs chicken-ortholog codon frequencies for the ortholog pairs in `sub`."""
    hf = sub[[f"fh_{c}" for c in SENSE_CODONS]].to_numpy()
    cf = sub[[f"fc_{c}" for c in SENSE_CODONS]].to_numpy()
    hn = sub[[f"nh_{c}" for c in SENSE_CODONS]].sum().to_numpy()
    cn = sub[[f"nc_{c}" for c in SENSE_CODONS]].sum().to_numpy()

    table = np.column_stack([hn, cn])
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    omnibus = {'group': group, 'n_pairs': len(sub), 'n_codons_human': int(hn.sum()),
               'n_codons_chicken': int(cn.sum()), 'chi2': chi2, 'dof': dof, 'p_value': p,
               'cramers_v': np.sqrt(chi2 / table.sum())}

    rows = []
    with np.errstate(divide='ignore', invalid='ignore'):
        pair_log2 = np.log2(hf / cf)  # per ortholog pair and codon
        pair_log2[~np.isfinite(pair_log2)] = np.nan  # undefined if either frequency is 0
        for i, codon in enumerate(SENSE_CODONS):
            t = np.array([[hn[i], hn.sum() - hn[i]], [cn[i], cn.sum() - cn[i]]])
            try:
                chi2_c, p_c, _, _ = stats.chi2_contingency(t, correction=False)
            except ValueError:  # codon absent from both species in a tiny group
                chi2_c, p_c = np.nan, np.nan
            diff = hf[:, i] - cf[:, i]
            w_p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            l2_pairs = pair_log2[:, i][~np.isnan(pair_log2[:, i])]
            l2_p = stats.wilcoxon(l2_pairs).pvalue if np.any(l2_pairs != 0) else 1.0
            l2_pooled = np.log2(hf[:, i].mean() / cf[:, i].mean())
            rows.append({'group': group, 'codon': codon,
                         'mean_freq_human': hf[:, i].mean(), 'mean_freq_chicken': cf[:, i].mean(),
                         # PLOTTED: mean over ortholog pairs of log2(human freq / chicken freq),
                         # pairs with a zero frequency in either species excluded (log2 undefined)
                         'log2_human_over_chicken': l2_pairs.mean() if len(l2_pairs) else np.nan,
                         'n_pairs_log2_defined': len(l2_pairs),
                         'log2_ratio_of_mean_freq': l2_pooled if np.isfinite(l2_pooled) else np.nan,
                         'chi2': chi2_c, 'p_value': p_c, 'paired_wilcoxon_p_value': w_p,
                         'log2_ratio_wilcoxon_p_value': l2_p if len(l2_pairs) else np.nan})
    per_codon = pd.DataFrame(rows)
    for col, out in [('p_value', 'fdr'), ('paired_wilcoxon_p_value', 'paired_wilcoxon_fdr'),
                     ('log2_ratio_wilcoxon_p_value', 'log2_ratio_wilcoxon_fdr')]:
        ok = per_codon[col].notna()
        per_codon[out] = np.nan
        per_codon.loc[ok, out] = multipletests(per_codon.loc[ok, col], method='fdr_bh')[1]
    per_codon['n_pairs'] = len(sub)
    return omnibus, per_codon


def run_set(pairs, group_masks, name):
    """Chi-square tests for each gene type in `group_masks` ({label: boolean mask}); saves tables."""
    omnibus_rows, per_codon_all, results = [], [], {}
    for label, mask in group_masks.items():
        sub = pairs[mask]
        if len(sub) < MIN_PAIRS:
            print(f"    skipping '{label}': only {len(sub)} ortholog pair(s) (< {MIN_PAIRS})")
            continue
        omnibus, per_codon = compare_group(sub, label)
        omnibus_rows.append(omnibus)
        per_codon_all.append(per_codon)
        results[label] = per_codon
        print(f"    {label} (n = {len(sub):,} pairs): omnibus chi2 = {omnibus['chi2']:.1f} (dof {omnibus['dof']}), "
              f"p = {omnibus['p_value']:.3g}; {int((per_codon['fdr'] < FDR_ALPHA).sum())}/61 codons chi-square FDR < {FDR_ALPHA}, "
              f"{int((per_codon['paired_wilcoxon_fdr'] < FDR_ALPHA).sum())}/61 paired-Wilcoxon FDR < {FDR_ALPHA}")
    pd.DataFrame(omnibus_rows).to_csv(tbl(f"codon_frequency_chi_square_omnibus_human_vs_chicken_{name}"), index=False)
    pd.concat(per_codon_all, ignore_index=True).to_csv(
        tbl(f"codon_frequency_chi_square_per_codon_human_vs_chicken_{name}"), index=False)
    return results


def plot_clustered_heatmap(results, title, out_file):
    """Horizontal clustered heatmap: rows = gene types, columns = codons."""
    lfc = pd.DataFrame({g: r.set_index('codon')['log2_human_over_chicken'] for g, r in results.items()}).T
    lfc = lfc[SENSE_CODONS]
    sig = pd.DataFrame({g: r.set_index('codon')['fdr'] < FDR_ALPHA for g, r in results.items()}).T[SENSE_CODONS]
    labels = {g: f"{g} (n = {int(r['n_pairs'].iloc[0]):,})" for g, r in results.items()}
    lfc.index = [labels[g] for g in lfc.index]
    sig.index = lfc.index
    plot_vals = lfc.fillna(0.0)  # undefined ratios (zero frequency, tiny groups) -> 0 for clustering/plotting

    vmax = np.nanpercentile(np.abs(lfc.to_numpy()), 98)
    stars = np.where(sig.to_numpy(), "*", "")
    col_colors = pd.Series({c: AT_COLOR if c[-1] in 'AT' else GC_COLOR for c in SENSE_CODONS},
                           name="third base")
    n_rows = len(lfc)
    g = sns.clustermap(
        plot_vals, method='average', metric='euclidean',
        row_cluster=n_rows > 1, col_cluster=True,
        col_colors=col_colors, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
        annot=stars, fmt='', annot_kws={'size': 9, 'color': 'black'},
        linewidths=0.4, linecolor='white',
        xticklabels=True, yticklabels=True,
        figsize=(16, 3.4 + 0.55 * n_rows),
        dendrogram_ratio=(0.12 if n_rows > 1 else 0.001, 0.18),
        colors_ratio=0.05,
        cbar_pos=(0.02, 0.02, 0.012, 0.14),
        cbar_kws={'label': 'mean log2(human / chicken)'},
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8, color='black')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9, color='black')
    g.ax_heatmap.set_xlabel("Codon")
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.yaxis.set_label_position("right")
    g.ax_col_colors.set_yticks([0.5])
    g.ax_col_colors.set_yticklabels(["ends in"], fontsize=8)
    g.ax_col_colors.yaxis.tick_right()
    g.ax_heatmap.legend(handles=[Patch(facecolor=AT_COLOR, edgecolor='gray', label='codon ends in A/T'),
                                 Patch(facecolor=GC_COLOR, edgecolor='gray', label='codon ends in G/C')],
                        loc='upper left', bbox_to_anchor=(1.0, -0.35), fontsize=8, frameon=False)
    g.fig.suptitle(f"Codon frequency comparison between human and chicken 1:1 orthologs: {title}\n"
                   "Data plotted: mean log2(human codon freq / chicken codon freq) across ortholog pairs "
                   "(red = enriched in human)\n"
                   f"* = chi-square FDR < {FDR_ALPHA} (pooled codon counts); rows and codons clustered (average linkage)",
                   fontsize=11, y=1.12)
    g.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(g.fig)
    print(f"  Saved: {out_file}")


def winston_category_by_id():
    """
    Ensembl gene ID -> 'PAR1' / 'PAR2' / 'NPX-NPY' from the Winston table (PAR from "Human Stratum",
    NPX-NPY from "Yes" in "X-Y Pair"; PAR takes precedence). Defined by Ensembl ID because gene symbols
    in the XCI tables / CDS table do not always agree (e.g. FAM122B, TMSL2).
    """
    w = pd.read_csv(STRATA_FILE, skiprows=1, usecols=['Gene Symbol', 'X-Y Pair', 'Human Stratum', 'Human Ensembl ID'])
    w = w.dropna(subset=['Gene Symbol', 'Human Ensembl ID']).drop_duplicates('Human Ensembl ID')
    cat = pd.Series(np.where(w['X-Y Pair'] == 'Yes', 'NPX-NPY', None), index=w['Human Ensembl ID'].to_numpy(), dtype=object)
    cat[(w['Human Stratum'] == 'PAR1').to_numpy()] = 'PAR1'
    cat[(w['Human Stratum'] == 'PAR2').to_numpy()] = 'PAR2'
    return cat.dropna()


def add_xci_group(pairs, xci_file, exclude):
    """
    X-linked ortholog pairs' XCI classification; NaN elsewhere. Xi-silent / Xi-expressed come from
    `xci_file` by gene name; PAR1 / PAR2 / NPX-NPY are taken from the Winston table by Ensembl gene
    ID (so they are identical for every classification source and not lost to symbol mismatches).
    """
    xci = pd.read_csv(xci_file, usecols=['gene_name', 'classification'])
    xci = xci[~xci['classification'].isin(exclude)].drop_duplicates('gene_name')
    cls = pairs['gene_name'].map(xci.set_index('gene_name')['classification'])
    cls = cls.where(~cls.isin(['PAR1', 'PAR2', 'NPX-NPY']))  # category genes come from Winston only
    cat = pairs['ensembl_gene_id'].map(winston_category_by_id())
    cls = cat.where(cat.notna(), cls)
    return cls.where(pairs['group'] == 'X')


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=== Codon frequency: human vs chicken, 1:1 orthologs ===")
    print("\n[1] Loading orthologs, codon frequencies and CDS counts...")
    pairs = load_pairs()
    is_auto = pairs['group'] == 'Autosome'

    print("\n[2] Autosome vs X...")
    results = run_set(pairs, {'Autosome': is_auto, 'X': pairs['group'] == 'X'}, "autosome_vs_X")
    # per-codon table / means under the original (unsuffixed) names
    pd.concat(results.values(), ignore_index=True).to_csv(
        tbl("codon_frequency_chi_square_per_codon_human_vs_chicken"), index=False)
    pd.DataFrame({f"{g}_{s}": r.set_index('codon')[f"mean_freq_{s}"] for g, r in results.items()
                  for s in ('human', 'chicken')}).to_csv(tbl("codon_frequency_mean_human_vs_chicken_orthologs"))
    plot_clustered_heatmap(results, "Autosome vs X",
                           fig("codon_frequency_log2_human_over_chicken_heatmap"))

    for source, (xci_file, exclude) in XCI_FILES.items():
        print(f"\n[3] XCI categories ({source})...")
        cls = add_xci_group(pairs, xci_file, exclude)
        print(f"  {int(cls.notna().sum()):,} of {int((pairs['group'] == 'X').sum()):,} X-linked ortholog pairs have a "
              f"{source} XCI classification: " + ", ".join(f"{k} = {v}" for k, v in cls.value_counts().items()))
        full = ['Xi-expressed', 'Xi-silent', 'NPX-NPY', 'PAR1', 'PAR2']
        masks = {'Autosome': is_auto, **{c: cls == c for c in full}}
        res = run_set(pairs, masks, f"XCI_full_{source}")
        plot_clustered_heatmap(res, f"Autosome vs Xi-expressed vs Xi-silent vs NPX-NPY vs PAR1 vs PAR2 ({source} XCI classification)",
                               fig(f"codon_frequency_log2_human_over_chicken_heatmap_XCI_full_{source}"))
        masks = {'Autosome': is_auto, 'Xi-silent': cls == 'Xi-silent', 'Xi-expressed': cls == 'Xi-expressed'}
        res = run_set(pairs, masks, f"XCI_reduced_{source}")
        plot_clustered_heatmap(res, f"Autosome vs Xi-silent vs Xi-expressed ({source} XCI classification)",
                               fig(f"codon_frequency_log2_human_over_chicken_heatmap_XCI_reduced_{source}"))

    print("\n[4] X evolutionary strata...")
    strata = pd.read_csv(STRATA_FILE, skiprows=1, usecols=['Human Ensembl ID', 'Human Stratum'])
    strata = strata.dropna(subset=['Human Ensembl ID']).drop_duplicates('Human Ensembl ID')
    stratum = pairs['ensembl_gene_id'].map(strata.set_index('Human Ensembl ID')['Human Stratum'])
    stratum = stratum.where(pairs['group'] == 'X')
    print(f"  X-linked ortholog pairs per stratum: " + ", ".join(
        f"{s} = {int((stratum == s).sum())}" for s in STRATA))
    masks = {'Autosome': is_auto, **{s: stratum == s for s in STRATA}}
    res = run_set(pairs, masks, "X_strata")
    plot_clustered_heatmap(res, "Autosome vs X evolutionary strata (S5, S4, S3, Ambiguous S3, Ambiguous S1, S1)",
                           fig("codon_frequency_log2_human_over_chicken_heatmap_X_strata"))

    # Same strata comparison, restricted to Xi-silent genes only / Xi-expressed genes only
    # (X genes' XCI status from each classification source; Autosome row unchanged).
    for source, (xci_file, exclude) in XCI_FILES.items():
        cls = add_xci_group(pairs, xci_file, exclude)
        for status in ('Xi-silent', 'Xi-expressed'):
            print(f"\n[4] X evolutionary strata, {status} genes only ({source})...")
            in_status = cls == status
            print(f"  {int(in_status.sum())} {status} ortholog pairs per stratum: " + ", ".join(
                f"{st} = {int((in_status & (stratum == st)).sum())}" for st in STRATA))
            masks = {'Autosome': is_auto, **{st: in_status & (stratum == st) for st in STRATA}}
            res = run_set(pairs, masks, f"X_strata_{status}_only_{source}")
            plot_clustered_heatmap(
                res, f"Autosome vs X evolutionary strata, {status} genes only ({source} XCI classification)",
                fig(f"codon_frequency_log2_human_over_chicken_heatmap_X_strata_{status}_only_{source}"))

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
