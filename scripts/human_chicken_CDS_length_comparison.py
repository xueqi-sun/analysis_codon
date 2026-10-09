#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
CDS length of human and chicken genes, compared between gene groups.

Pipeline:
  1. Load the human CDS table (the MANE Select table human_co_mega_analysis.py
     uses), the chicken CDS table (data/chicken_CDS_sequence.csv) and the
     human-chicken 1:1 ortholog table, as in
     human_chicken_codon_frequency_comparison.py. CDS length = number of
     nucleotides of the CDS sequence (stop codon included); the log2 of the
     length is also computed.
  2. Gene groups (always by the HUMAN gene): autosomal (chr 1-22) vs X-linked;
     X genes by XCI status (Xi-expressed, Xi-silent, NPX-NPY, PAR1, PAR2) for
     both the Gylemo and the Neha classification; X genes by evolutionary
     stratum (S5, S4, S3, Ambiguous S3, Ambiguous S1, S1; Winston table).
  3. Two gene sets and measures, each in raw length (nt) and log2(length):
       - ALL human genes: human CDS length.
       - 1:1 ORTHOLOG pairs with a chicken CDS: chicken-ortholog CDS length,
         and log2(human CDS length / chicken CDS length) per pair (log scale
         only; it is a ratio).
  4. For every comparison: boxplots (with the pairwise Mann-Whitney U
     p-values in the legend) and a statistics table in which each group is
     tested against Autosome with a Mann-Whitney U test and a Welch t-test
     (BH-adjusted across the groups of that comparison), plus a Kruskal-Wallis
     test across all groups. The Mann-Whitney / Kruskal-Wallis results are
     identical for raw and log2 lengths (rank-based, monotone transform);
     the Welch t-test is not.

Outputs: tables/CDS_length/ and figures/CDS_length/.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from co_mega_common import plot_co_mega_by_group
from human_chicken_codon_frequency_comparison import (
    ORTHOLOG_FILE, HUMAN_CDS_FILE, CHICKEN_CDS_FILE, XCI_FILES, STRATA, STRATA_FILE,
    add_xci_group, AUTOSOMES,
)

BASE_DIR = "/lab/solexa_page/xueqi/analysis_codon"
TABLE_DIR = os.path.join(BASE_DIR, "tables/CDS_length")
FIG_DIR = os.path.join(BASE_DIR, "figures/CDS_length")

XCI_FULL = ['Xi-expressed', 'Xi-silent', 'NPX-NPY', 'PAR1', 'PAR2']


def load_data():
    h = pd.read_csv(HUMAN_CDS_FILE, usecols=['ensembl_gene_id', 'gene_name', 'chromosome', 'CDS_sequence'])
    h = h.dropna(subset=['ensembl_gene_id', 'CDS_sequence'])
    h['chromosome'] = h['chromosome'].astype(str)
    h['human_len'] = h['CDS_sequence'].str.len()
    h['group'] = np.where(h['chromosome'] == 'X', 'X', np.where(h['chromosome'].isin(AUTOSOMES), 'Autosome', 'other'))
    n_other = int((h['group'] == 'other').sum())
    h = h[h['group'] != 'other'].drop(columns='CDS_sequence')  # Y-linked / other chromosomes excluded
    print(f"  Human: {len(h):,} genes (autosome {int((h['group'] == 'Autosome').sum()):,}, X {int((h['group'] == 'X').sum()):,}); "
          f"{n_other} Y-linked/other-chromosome genes excluded")
    print(f"  Human CDS length: median {h['human_len'].median():.0f} nt; {(h['human_len'] % 3 != 0).sum()} not a multiple of 3")

    c = pd.read_csv(CHICKEN_CDS_FILE, usecols=['gene_id', 'CDS_sequence'])
    c['chicken_len'] = c['CDS_sequence'].str.len()
    c = c.drop(columns='CDS_sequence').rename(columns={'gene_id': 'chicken_gene_id'})
    print(f"  Chicken: {len(c):,} genes; median CDS length {c['chicken_len'].median():.0f} nt")

    orth = pd.read_csv(ORTHOLOG_FILE, usecols=['Gene stable ID', 'Chicken gene stable ID']).rename(
        columns={'Gene stable ID': 'ensembl_gene_id', 'Chicken gene stable ID': 'chicken_gene_id'}).dropna()
    pairs = orth.merge(h, on='ensembl_gene_id', how='inner').merge(c, on='chicken_gene_id', how='inner')
    pairs['log2_len_ratio_human_over_chicken'] = np.log2(pairs['human_len'] / pairs['chicken_len'])
    print(f"  {len(orth):,} 1:1 orthologs -> {len(pairs):,} pairs with a human CDS and a chicken CDS "
          f"(autosome {int((pairs['group'] == 'Autosome').sum()):,}, X {int((pairs['group'] == 'X').sum()):,})")

    strata = pd.read_csv(STRATA_FILE, skiprows=1, usecols=['Human Ensembl ID', 'Human Stratum'])
    strata = strata.dropna(subset=['Human Ensembl ID']).drop_duplicates('Human Ensembl ID').set_index('Human Ensembl ID')['Human Stratum']
    for df in (h, pairs):
        df['stratum'] = df['ensembl_gene_id'].map(strata).where(df['group'] == 'X')
        for source, (xci_file, exclude) in XCI_FILES.items():
            df[f'xci_{source}'] = add_xci_group(df, xci_file, exclude)
    for df in (h, pairs):
        for col in ['human_len', 'chicken_len']:
            if col in df:
                df[f'log2_{col}'] = np.log2(df[col])
    return h, pairs


def group_comparisons(df):
    """(name, group column, ordered group labels) for each comparison."""
    comps = [('autosome_vs_X', 'group', ['Autosome', 'X'])]
    for source in XCI_FILES:
        comps.append((f'autosome_vs_XCI_status_{source}', f'xci_{source}', ['Autosome'] + XCI_FULL))
    comps.append(('autosome_vs_X_strata', 'stratum', ['Autosome'] + STRATA))
    return comps


def test_vs_autosome(values, labels, order, comp, set_name, measure, scale):
    """Each non-autosome group vs Autosome: Mann-Whitney U, Welch t; Kruskal-Wallis overall."""
    present = [g for g in order if (labels == g).any()]
    auto = values[labels == 'Autosome']
    rows = []
    for g in present:
        if g == 'Autosome':
            continue
        x = values[labels == g]
        mw = stats.mannwhitneyu(auto, x, alternative='two-sided').pvalue
        tt = stats.ttest_ind(auto, x, equal_var=False).pvalue if len(x) > 1 else np.nan
        rows.append({'set': set_name, 'measure': measure, 'scale': scale, 'comparison': comp, 'group': g,
                     'n_autosome': len(auto), 'n_group': len(x),
                     'mean_autosome': auto.mean(), 'mean_group': x.mean(),
                     'median_autosome': auto.median(), 'median_group': x.median(),
                     'mannwhitney_p_vs_autosome': mw, 'welch_t_p_vs_autosome': tt})
    out = pd.DataFrame(rows)
    if len(out):
        out['mannwhitney_fdr'] = multipletests(out['mannwhitney_p_vs_autosome'], method='fdr_bh')[1]
        ok = out['welch_t_p_vs_autosome'].notna()
        out['welch_t_fdr'] = np.nan
        out.loc[ok, 'welch_t_fdr'] = multipletests(out.loc[ok, 'welch_t_p_vs_autosome'], method='fdr_bh')[1]
        samples = [values[labels == g] for g in present]
        out['kruskal_wallis_p_all_groups'] = stats.kruskal(*samples).pvalue if len(samples) > 2 else np.nan
    return out


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=== CDS length: human vs chicken, by gene group ===")
    print("\n[1] Loading CDS tables and orthologs...")
    human, pairs = load_data()
    cols = ['ensembl_gene_id', 'gene_name', 'chromosome', 'group', 'stratum'] + [f'xci_{s}' for s in XCI_FILES]
    human[cols + ['human_len', 'log2_human_len']].to_csv(os.path.join(TABLE_DIR, "CDS_length_human_all_genes.csv"), index=False)
    pairs[cols + ['chicken_gene_id', 'human_len', 'log2_human_len', 'chicken_len', 'log2_chicken_len',
                  'log2_len_ratio_human_over_chicken']].to_csv(
        os.path.join(TABLE_DIR, "CDS_length_human_chicken_ortholog_pairs.csv"), index=False)

    # (set name, data, measure column, display name, scales: [(scale label, column, y-axis label)])
    jobs = [
        ('human_all_genes', human, 'human_len', 'Human CDS length',
         [('raw', 'human_len', 'CDS length (nt)'), ('log2', 'log2_human_len', 'log2(CDS length, nt)')]),
        ('ortholog_pairs', pairs, 'chicken_len', "Chicken ortholog CDS length",
         [('raw', 'chicken_len', 'CDS length (nt)'), ('log2', 'log2_chicken_len', 'log2(CDS length, nt)')]),
        ('ortholog_pairs', pairs, 'len_ratio', 'log2(human CDS length / chicken CDS length)',
         [('log2', 'log2_len_ratio_human_over_chicken', 'log2(human CDS length / chicken CDS length)')]),
    ]
    stats_tables = []
    print("\n[2] Comparisons...")
    for set_name, df, measure, display, scales in jobs:
        for comp, group_col, order in group_comparisons(df):
            for scale, col, ylabel in scales:
                # label: Autosome / X (or the X gene's XCI status / stratum); NaN = not in this comparison
                label = df['group'] if group_col == 'group' else df['group'].where(df['group'] == 'Autosome', df[group_col])
                keep = label.notna() & df[col].notna()
                plot_df = pd.DataFrame({col: df.loc[keep, col], 'label': label[keep]}).reset_index(drop=True)
                res = test_vs_autosome(plot_df[col], plot_df['label'], order, comp, set_name, measure, scale)
                stats_tables.append(res)
                tag = f"{set_name}_{measure}_{scale}_{comp}"
                title = f"{display} ({scale} scale): {comp.replace('_', ' ')} [{set_name.replace('_', ' ')}]"
                plot_co_mega_by_group(plot_df, col, 'label', order, title,
                                      os.path.join(FIG_DIR, f"CDS_length_{tag}.png"), ylabel=ylabel)
                for r in res.itertuples():
                    print(f"    {tag}: {r.group} (n = {r.n_group:,}) vs Autosome (n = {r.n_autosome:,}): "
                          f"median {r.median_group:.4g} vs {r.median_autosome:.4g}, "
                          f"Mann-Whitney FDR = {r.mannwhitney_fdr:.3g}")

    all_stats = pd.concat(stats_tables, ignore_index=True)
    out = os.path.join(TABLE_DIR, "CDS_length_comparison_stats.csv")
    all_stats.to_csv(out, index=False)
    print(f"\n  Saved statistics ({len(all_stats)} rows) to {out}")
    print("\n=== Done ===")


if __name__ == '__main__':
    main()
