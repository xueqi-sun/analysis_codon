#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Do codon frequencies differ between autosomal, X-linked and Y-linked genes
(human liver CDS set)?

Pipeline:
  1. Load per-gene codon frequencies
     (tables/human/codon_frequencies_human_liver.csv; written by
     human_co_mega_analysis.py) and assign each gene to Autosome / X / Y
     (is_X, is_Y; everything else is autosomal).
  2. Mean frequency of each of the 61 sense codons per group -> 61 x 3 table.
  3. Chi-square tests for Autosome vs X, Autosome vs Y and X vs Y:
       - Omnibus: chi-square test of homogeneity on the 61 x 2 table of
         codon counts (are the two groups' codon distributions the same?).
       - Per codon: 2 x 2 chi-square test (this codon vs. all other codons,
         group A vs. group B), Benjamini-Hochberg FDR across the 61 codons.
     The chi-square tests need counts, not frequencies. Per-gene counts are
     recovered exactly as frequency x (# codons in the gene), where
     # codons = len(CDS)/3 - 1 (the stop codon is dropped, as when the
     frequencies were computed); the CDS lengths come from the same CDS table
     human_co_mega_analysis.py uses. Group counts are the sums over genes.
     CAVEAT: this pools codons across genes as if they were independent
     draws, which they are not (codons within a gene are correlated, and
     long genes dominate), so the chi-square p-values are anti-conservative.
     As a robustness check the per-codon table also carries a gene-level
     Mann-Whitney U test on the per-gene frequencies (each gene counts once).
  4. Heatmap of log2(mean frequency ratio) per codon, X vs Autosome (plus
     Y vs Autosome and Y vs X), codons sorted by the X vs Autosome ratio;
     * marks per-codon chi-square FDR < 0.05.

Outputs (suffix `_human_liver`): tables/human/ and figures/human/.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from co_mega import SENSE_CODONS

BASE_DIR = "/lab/solexa_page/xueqi/analysis_codon"
FREQ_FILE = os.path.join(BASE_DIR, "tables", "human", "codon_frequencies_human_liver.csv")
# Same CDS table human_co_mega_analysis.py builds the frequencies from; used
# only for each gene's codon count.
CDS_FILE = "/lab/solexa_page/xueqi/general/sequence_CDS/human/tables/human_CDS_sequences_MANE_Select_ensembl.csv"
TABLE_DIR = os.path.join(BASE_DIR, "tables", "human")
FIG_DIR = os.path.join(BASE_DIR, "figures", "human")
SUFFIX = "human_liver"

GROUPS = ["Autosome", "X", "Y"]
COMPARISONS = [("Autosome", "X"), ("Autosome", "Y"), ("X", "Y")]
FDR_ALPHA = 0.05


def tbl(name):
    return os.path.join(TABLE_DIR, f"{name}_{SUFFIX}.csv")


def fig(name):
    return os.path.join(FIG_DIR, f"{name}_{SUFFIX}.png")


def load_data():
    freq = pd.read_csv(FREQ_FILE, low_memory=False)
    cds = pd.read_csv(CDS_FILE, usecols=['ensembl_gene_id', 'CDS_sequence'])
    df = freq.merge(cds, on='ensembl_gene_id', how='left', validate='one_to_one')
    if df['CDS_sequence'].isna().any():
        raise ValueError(f"{df['CDS_sequence'].isna().sum()} genes in the frequency table have no CDS")
    df['n_codons'] = df['CDS_sequence'].str.len() // 3 - 1  # stop codon dropped

    counts = df[SENSE_CODONS].mul(df['n_codons'], axis=0)
    if (counts - counts.round()).abs().max().max() > 1e-6:
        raise ValueError("frequency x n_codons is not integer; frequencies/CDS table are out of sync")
    counts = counts.round().astype(np.int64)

    df['group'] = np.where(df['is_Y'] == 1, 'Y', np.where(df['is_X'] == 1, 'X', 'Autosome'))
    print(f"  {len(df):,} genes: " + ", ".join(f"{g} = {(df['group'] == g).sum():,}" for g in GROUPS))
    return df, counts


def compare_groups(df, counts, g1, g2):
    """Omnibus + per-codon chi-square tests of codon counts, group g1 vs g2."""
    c1 = counts[(df['group'] == g1).to_numpy()].sum()
    c2 = counts[(df['group'] == g2).to_numpy()].sum()
    table = np.column_stack([c1.to_numpy(), c2.to_numpy()])
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    n_tot = table.sum()
    cramers_v = np.sqrt(chi2 / (n_tot * (min(table.shape) - 1)))
    omnibus = {'comparison': f"{g1} vs {g2}", 'n_genes_1': int((df['group'] == g1).sum()),
               'n_genes_2': int((df['group'] == g2).sum()), 'n_codons_1': int(c1.sum()),
               'n_codons_2': int(c2.sum()), 'chi2': chi2, 'dof': dof, 'p_value': p,
               'cramers_v': cramers_v}

    rows = []
    mean1 = df.loc[df['group'] == g1, SENSE_CODONS].mean()
    mean2 = df.loc[df['group'] == g2, SENSE_CODONS].mean()
    for codon in SENSE_CODONS:
        a, b = c1[codon], c2[codon]
        t = np.array([[a, c1.sum() - a], [b, c2.sum() - b]])
        chi2_c, p_c, _, _ = stats.chi2_contingency(t, correction=False)
        u_p = stats.mannwhitneyu(df.loc[df['group'] == g1, codon], df.loc[df['group'] == g2, codon],
                                 alternative='two-sided').pvalue
        rows.append({'comparison': f"{g1} vs {g2}", 'codon': codon,
                     f'mean_freq_{g1}': mean1[codon], f'mean_freq_{g2}': mean2[codon],
                     'log2_ratio_2_over_1': np.log2(mean2[codon] / mean1[codon]),
                     'chi2': chi2_c, 'p_value': p_c, 'mannwhitney_p_value': u_p})
    per_codon = pd.DataFrame(rows)
    per_codon['fdr'] = multipletests(per_codon['p_value'], method='fdr_bh')[1]
    per_codon['mannwhitney_fdr'] = multipletests(per_codon['mannwhitney_p_value'], method='fdr_bh')[1]
    return omnibus, per_codon


def plot_heatmap(mean_df, results, out_file):
    """log2(mean freq ratio) heatmap, rows = codons sorted by X vs Autosome."""
    cols = [("Autosome", "X"), ("Autosome", "Y"), ("X", "Y")]
    lfc = pd.DataFrame({f"{b} / {a}": np.log2(mean_df[b] / mean_df[a]) for a, b in cols})
    sig = pd.DataFrame({f"{b} / {a}": results[(a, b)].set_index('codon')['fdr'] < FDR_ALPHA for a, b in cols})
    order = lfc[f"X / Autosome"].sort_values(ascending=False).index
    lfc, sig = lfc.loc[order], sig.loc[order]

    vmax = np.nanpercentile(np.abs(lfc.to_numpy()), 98)
    fig_, ax = plt.subplots(figsize=(5.6, 13))
    im = ax.imshow(lfc.to_numpy(), cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(lfc.shape[1]))
    ax.set_xticklabels(lfc.columns)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(lfc)))
    ax.set_yticklabels(lfc.index, family='monospace', fontsize=8)
    for i in range(lfc.shape[0]):
        for j in range(lfc.shape[1]):
            v = lfc.iat[i, j]
            ax.text(j, i, f"{v:+.2f}" + ("*" if sig.iat[i, j] else ""), ha='center', va='center',
                    fontsize=7, color='white' if abs(v) > 0.6 * vmax else 'black')
    cbar = fig_.colorbar(im, ax=ax, shrink=0.4, pad=0.04)
    cbar.set_label("log2(mean codon frequency ratio)\nred = enriched in numerator group")
    ax.set_title("Codon frequency: X vs Autosome (and Y)\ncodons sorted by X / Autosome;  * = chi-square FDR < 0.05",
                 pad=40, fontsize=10)
    fig_.tight_layout()
    fig_.savefig(out_file, dpi=150)
    plt.close(fig_)
    print(f"  Saved: {out_file}")


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=== Codon frequency: Autosome vs X vs Y (human liver) ===")
    print("\n[1] Loading codon frequencies and per-gene codon counts...")
    df, counts = load_data()

    print("\n[2] Mean codon frequency per group (61 x 3)...")
    mean_df = pd.DataFrame({g: df.loc[df['group'] == g, SENSE_CODONS].mean() for g in GROUPS})
    mean_df.index.name = 'codon'
    out_mean = tbl("codon_mean_frequency_autosome_X_Y")
    mean_df.to_csv(out_mean)
    print(f"  Saved {mean_df.shape[0]} x {mean_df.shape[1]} table to {out_mean}")

    print("\n[3] Chi-square tests...")
    omnibus_rows, per_codon_all, results = [], [], {}
    for g1, g2 in COMPARISONS:
        omnibus, per_codon = compare_groups(df, counts, g1, g2)
        omnibus_rows.append(omnibus)
        per_codon_all.append(per_codon)
        results[(g1, g2)] = per_codon
        n_sig = int((per_codon['fdr'] < FDR_ALPHA).sum())
        n_sig_u = int((per_codon['mannwhitney_fdr'] < FDR_ALPHA).sum())
        print(f"  {g1} vs {g2}: omnibus chi2 = {omnibus['chi2']:.1f} (dof {omnibus['dof']}), "
              f"p = {omnibus['p_value']:.3g}, Cramer's V = {omnibus['cramers_v']:.4f}; "
              f"{n_sig}/61 codons with chi-square FDR < {FDR_ALPHA}, "
              f"{n_sig_u}/61 with gene-level Mann-Whitney FDR < {FDR_ALPHA}")
    pd.DataFrame(omnibus_rows).to_csv(tbl("codon_chi_square_omnibus"), index=False)
    pd.concat(per_codon_all, ignore_index=True).to_csv(tbl("codon_chi_square_per_codon"), index=False)
    print(f"  Saved {tbl('codon_chi_square_omnibus')}\n  Saved {tbl('codon_chi_square_per_codon')}")

    xa = results[("Autosome", "X")].sort_values('log2_ratio_2_over_1', ascending=False)
    print("\n  Top codons enriched in X vs Autosome (log2 ratio, chi-square FDR):")
    for _, r in xa.head(5).iterrows():
        print(f"    {r['codon']}: {r['log2_ratio_2_over_1']:+.3f}  FDR = {r['fdr']:.3g}")
    print("  Top codons depleted in X vs Autosome:")
    for _, r in xa.tail(5).iloc[::-1].iterrows():
        print(f"    {r['codon']}: {r['log2_ratio_2_over_1']:+.3f}  FDR = {r['fdr']:.3g}")

    print("\n[4] Heatmap...")
    plot_heatmap(mean_df, results, fig("codon_frequency_log2_ratio_heatmap_X_Y_vs_autosome"))

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
