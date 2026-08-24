#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Compare human and chicken CO_Mega for 1-to-1 orthologous genes, and check
whether human X-linked genes (and their XCI status) differ from autosomal
genes in: (a) their own (human liver) CO_Mega, (b) their chicken ortholog's
(chicken liver) CO_Mega, and (c) a normalized CO_Mega (human / chicken).

Pipeline:
  1. Load human-chicken 1:1 orthologs from
     general/genes_1to1_ortholog_human_chicken/tables/human_chicken_one2one_orthologs.csv
     ('Gene stable ID' = human gene, 'Chicken gene stable ID' = chicken gene).
  2. Merge with human CO_Mega (analysis_codon/tables/co_mega_all_genes_human_liver.csv,
     on 'gene_id') and chicken CO_Mega (co_mega_all_genes_chicken_liver.csv,
     on 'gene_id'), keeping only genes with both. Compute
     CO_Mega_normalized = CO_Mega_human / CO_Mega_chicken.
  3. For each of the 3 CO_Mega measures (human liver, chicken liver ortholog,
     normalized), grouped by the HUMAN gene's chromosome status:
       - Boxplot: Autosome vs X (2 groups), like co_mega_boxplot_autosome_vs_X_*.png.
       - Boxplots by XCI category (Gylemo and Neha classifications, each
         'full' [Autosome, Xi-silent, Xi-expressed, NPX-NPY, PAR1, PAR2] and
         'reduced' [Autosome, Xi-silent, Xi-expressed]), like
         co_mega_boxplot_XCI_categories_full_Gylemo_human_liver.png.
     All boxplots show n, mean, and pairwise Mann-Whitney U p-values
     (scientific notation via `.3g`) in a legend.

All outputs (restricted to genes with a human-chicken 1:1 ortholog) are
saved under dedicated `human_chicken_comparison` subfolders of `figures/`
and `tables/`.
"""

import os

import pandas as pd
import numpy as np

from co_mega_common import plot_co_mega_boxplot, plot_co_mega_by_group, build_xci_groups

BASE_DIR       = "/lab/solexa_page/xueqi/analysis_codon"
ORTHOLOG_FILE  = "/lab/solexa_page/xueqi/general/genes_1to1_ortholog_human_chicken/tables/human_chicken_one2one_orthologs.csv"
HUMAN_CO_MEGA_FILE   = os.path.join(BASE_DIR, "tables/co_mega_all_genes_human_liver.csv")
CHICKEN_CO_MEGA_FILE = os.path.join(BASE_DIR, "tables/co_mega_all_genes_chicken_liver.csv")
XCI_FILE_GYLEMO = "/lab/solexa_page/xueqi/general/genes_XCI_Gylemo/tables/chrX_genes_classification_comparison_gylemo_liver_thresh0p05.csv"
XCI_FILE_NEHA   = "/lab/solexa_page/xueqi/general/genes_XCI_Neha/tables/chrX_genes_classification_Neha.csv"
TABLE_DIR = os.path.join(BASE_DIR, "tables/human_chicken_comparison")
FIG_DIR   = os.path.join(BASE_DIR, "figures/human_chicken_comparison")

# (value column in the merged ortholog df, filename/description suffix, display label)
VALUE_COLUMNS = [
    ('CO_Mega_human',      'human_liver',   'Human liver CO_Mega'),
    ('CO_Mega_chicken',    'chicken_liver', "Human's chicken-ortholog liver CO_Mega"),
    ('CO_Mega_normalized', 'normalized',    'Normalized CO_Mega (human / chicken)'),
]
XCI_SOURCES = [
    ('Gylemo', XCI_FILE_GYLEMO, ()),
    ('Neha',   XCI_FILE_NEHA,   ('No call',)),
]


def tbl(name):
    return os.path.join(TABLE_DIR, f"{name}.csv")


def fig(name):
    return os.path.join(FIG_DIR, f"{name}.png")


def load_ortholog_co_mega():
    orthologs = pd.read_csv(ORTHOLOG_FILE, usecols=['Gene stable ID', 'Chicken gene stable ID']).rename(
        columns={'Gene stable ID': 'human_gene_id', 'Chicken gene stable ID': 'chicken_gene_id'})
    print(f"  Loaded {len(orthologs):,} human-chicken 1:1 orthologs.")

    human_df = pd.read_csv(HUMAN_CO_MEGA_FILE, usecols=['gene_id', 'gene_name', 'is_X', 'CO_Mega']).rename(
        columns={'gene_id': 'human_gene_id', 'CO_Mega': 'CO_Mega_human'})
    chicken_df = pd.read_csv(CHICKEN_CO_MEGA_FILE, usecols=['gene_id', 'CO_Mega']).rename(
        columns={'gene_id': 'chicken_gene_id', 'CO_Mega': 'CO_Mega_chicken'})

    merged = orthologs.merge(human_df, on='human_gene_id', how='inner') \
                       .merge(chicken_df, on='chicken_gene_id', how='inner')
    print(f"  {len(merged):,} orthologous genes have both a human liver and chicken liver CO_Mega.")

    merged['CO_Mega_normalized'] = merged['CO_Mega_human'] / merged['CO_Mega_chicken']
    n_bad = (~np.isfinite(merged['CO_Mega_normalized'])).sum()
    if n_bad:
        print(f"  Dropping {n_bad:,} genes with a non-finite normalized CO_Mega "
              f"(chicken CO_Mega == 0).")
        merged = merged[np.isfinite(merged['CO_Mega_normalized'])].copy()

    n_auto = (merged['is_X'] == 0).sum()
    n_x = (merged['is_X'] == 1).sum()
    print(f"  {n_auto:,} autosomal, {n_x:,} X-linked orthologous genes.")
    return merged


def plot_xci_boxplots_by_value(df, value_col, value_suffix, value_label, source_label, xci_file, exclude_values):
    groups_df, other_genes = build_xci_groups(df, xci_file, value_col=value_col, exclude_values=exclude_values)
    n_x_total = (df['is_X'] == 1).sum()
    print(f"  [{value_label}, {source_label}] {len(other_genes):,} of {n_x_total:,} X-linked orthologous "
          f"genes have a defined XCI status ({n_x_total - len(other_genes):,} excluded).")

    plot_co_mega_by_group(
        groups_df, value_col, 'classification',
        ['Autosome', 'Xi-silent', 'Xi-expressed', 'NPX-NPY', 'PAR1', 'PAR2'],
        f'{value_label} by XCI category ({source_label}, human-chicken orthologs)',
        fig(f"co_mega_boxplot_XCI_categories_full_{source_label}_{value_suffix}"),
        out_csv=tbl(f"co_mega_XCI_categories_full_stats_{source_label}_{value_suffix}")
    )
    plot_co_mega_by_group(
        groups_df, value_col, 'classification',
        ['Autosome', 'Xi-silent', 'Xi-expressed'],
        f'{value_label}: Autosome vs Xi-silent vs Xi-expressed ({source_label}, human-chicken orthologs)',
        fig(f"co_mega_boxplot_XCI_categories_reduced_{source_label}_{value_suffix}"),
        out_csv=tbl(f"co_mega_XCI_categories_reduced_stats_{source_label}_{value_suffix}")
    )


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=== Human vs. chicken CO_Mega, 1:1 orthologous genes ===")

    print("\n[1] Loading orthologs and merging with human/chicken CO_Mega...")
    merged = load_ortholog_co_mega()
    out_merged = tbl("co_mega_human_chicken_orthologs")
    merged.to_csv(out_merged, index=False)
    print(f"  Saved merged ortholog CO_Mega table to {out_merged}")

    print("\n[2] Boxplots: Autosome vs X (human classification), for each CO_Mega measure...")
    for value_col, value_suffix, value_label in VALUE_COLUMNS:
        plot_co_mega_boxplot(
            merged, value_col, 'is_X', 'X chromosome',
            f'{value_label}: X-linked vs Autosomal genes\n(human-chicken 1:1 orthologs)',
            fig(f"co_mega_boxplot_autosome_vs_X_{value_suffix}")
        )

    print("\n[3] Boxplots by XCI category (Gylemo / Neha), for each CO_Mega measure...")
    for value_col, value_suffix, value_label in VALUE_COLUMNS:
        for source_label, xci_file, exclude_values in XCI_SOURCES:
            plot_xci_boxplots_by_value(merged, value_col, value_suffix, value_label,
                                        source_label, xci_file, exclude_values)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
