#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Compare human and chicken CO_Mega for 1-to-1 orthologous genes, and check
whether human X-linked genes (and their XCI status) differ from autosomal
genes in: (a) their own (human liver) CO_Mega, (b) their chicken ortholog's
(chicken liver) CO_Mega, and (c) a normalized codon score (human / chicken).

CO_Mega (in co_mega_all_genes_{human,chicken}_liver.csv) is each species'
OWN model's predicted log2(TE) (fit independently per species; chicken
genes always use chicken's own coefficients, never human's). To combine the
two species multiplicatively rather than by directly dividing (possibly
negative) log2(TE) predictions, each CO_Mega is first exponentiated back to
a predicted (raw, always-positive) TE-like score: exp_codon_score = 2^CO_Mega.

Pipeline:
  1. Load human-chicken 1:1 orthologs from
     general/genes_1to1_ortholog_human_chicken/tables/human_chicken_one2one_orthologs.csv
     ('Gene stable ID' = human gene, 'Chicken gene stable ID' = chicken gene).
  2. Merge with human CO_Mega (analysis_codon/tables/co_mega_all_genes_human_liver.csv,
     on 'gene_id') and chicken CO_Mega (co_mega_all_genes_chicken_liver.csv,
     on 'gene_id'), keeping only genes with both. Compute:
       exp_codon_score_human   = 2 ** CO_Mega_human
       exp_codon_score_chicken = 2 ** CO_Mega_chicken
       normalized_score        = exp_codon_score_human / exp_codon_score_chicken
       log2_normalized_score   = log2(normalized_score)
     (always defined and finite, since exp_codon_score > 0 for every gene.)
  3. For each of the 3 measures (human liver CO_Mega, chicken liver ortholog
     CO_Mega, log2-normalized score), grouped by the HUMAN gene's chromosome
     status:
       - Boxplot + CDF: Autosome vs X (2 groups), like
         co_mega_boxplot_autosome_vs_X_orthologs_only_*.png /
         co_mega_cdf_autosome_vs_X_orthologs_only_*.png.
       - Boxplots by XCI category (Gylemo and Neha classifications, each
         'full' [Autosome, Xi-silent, Xi-expressed, NPX-NPY, PAR1, PAR2] and
         'reduced' [Autosome, Xi-silent, Xi-expressed]), like
         co_mega_boxplot_XCI_categories_full_Gylemo_human_liver.png.
     All boxplots show n, mean, and pairwise Mann-Whitney U p-values
     (scientific notation via `.3g`) in a legend.
  4. For the log2-normalized-score measure only, the XCI-category
     comparisons (XCI full/reduced x Gylemo/Neha) are additionally plotted
     as empirical CDFs (same p-values, n, and mean, shown in the plot
     legend).
  5. For each of the 3 measures, two more variants of the Autosome-vs-X
     comparison (boxplot + CDF, for both Gylemo and Neha where relevant):
       - Restricted to X genes with a *defined* XCI status (Xi-silent or
         Xi-expressed only; NPX-NPY/PAR1/PAR2/No-call genes excluded).
       - Autosomal genes split into their 22 individual (human) chromosomes
         (one box/curve per chromosome) vs. a single X group; each
         autosome's Mann-Whitney U p-value vs. X is saved to a table and
         (on the boxplot) marked with significance asterisks. For the
         chicken-liver/normalized measures this split is still based on
         the HUMAN ortholog's chromosome.
  6. For comparison, the same combined (Autosome vs X) and
     separated-by-chromosome plots (boxplot + CDF) are also made using
     *all* human genes (not just those with a chicken ortholog) for the
     human-liver CO_Mega measure only (chicken-liver/normalized are
     undefined without a chicken ortholog), suffixed `_all_genes`.

All figures/tables from steps [1]-[5] are restricted to genes with a
human-chicken 1:1 ortholog and are named with an `orthologs_only` suffix;
they, and the `_all_genes` figures from step [6], are all saved under
dedicated `human_chicken_comparison` subfolders of `figures/` and
`tables/`.
"""

import os

import pandas as pd
import numpy as np

from co_mega_common import (
    plot_co_mega_boxplot, plot_co_mega_by_group, build_xci_groups,
    plot_co_mega_cdf, plot_co_mega_cdf_by_group,
    plot_co_mega_boxplot_by_chromosome, plot_co_mega_cdf_by_chromosome,
)

BASE_DIR       = "/lab/solexa_page/xueqi/analysis_codon"
ORTHOLOG_FILE  = "/lab/solexa_page/xueqi/general/genes_1to1_ortholog_human_chicken/tables/human_chicken_one2one_orthologs.csv"
HUMAN_CO_MEGA_FILE   = os.path.join(BASE_DIR, "tables/co_mega_all_genes_human_liver.csv")
CHICKEN_CO_MEGA_FILE = os.path.join(BASE_DIR, "tables/co_mega_all_genes_chicken_liver.csv")
XCI_FILE_GYLEMO = "/lab/solexa_page/xueqi/general/genes_XCI_Gylemo/tables/chrX_genes_classification_comparison_gylemo_liver_thresh0p05.csv"
XCI_FILE_NEHA   = "/lab/solexa_page/xueqi/general/genes_XCI_Neha/tables/chrX_genes_classification_Neha.csv"
TABLE_DIR = os.path.join(BASE_DIR, "tables/human_chicken_comparison")
FIG_DIR   = os.path.join(BASE_DIR, "figures/human_chicken_comparison")

XCI_FULL_ORDER    = ['Autosome', 'Xi-silent', 'Xi-expressed', 'NPX-NPY', 'PAR1', 'PAR2']
XCI_REDUCED_ORDER = ['Autosome', 'Xi-silent', 'Xi-expressed']
AUTOSOME_ORDER    = [str(i) for i in range(1, 23)]

# (value column, filename/description suffix, display label, y/x-axis label, also make CDF plots)
VALUE_COLUMNS = [
    ('CO_Mega_human',        'human_liver',   'Human liver CO_Mega',                                'CO_Mega',                     False),
    ('CO_Mega_chicken',      'chicken_liver', "Human's chicken-ortholog liver CO_Mega",              'CO_Mega',                     False),
    ('log2_normalized_score', 'normalized',   'log2(Normalized exp(codon score)) (human / chicken)', 'log2(Normalized score)',      True),
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

    human_df = pd.read_csv(HUMAN_CO_MEGA_FILE, usecols=['gene_id', 'gene_name', 'chromosome', 'is_X', 'CO_Mega']).rename(
        columns={'gene_id': 'human_gene_id', 'CO_Mega': 'CO_Mega_human'})
    chicken_df = pd.read_csv(CHICKEN_CO_MEGA_FILE, usecols=['gene_id', 'CO_Mega']).rename(
        columns={'gene_id': 'chicken_gene_id', 'CO_Mega': 'CO_Mega_chicken'})

    merged = orthologs.merge(human_df, on='human_gene_id', how='inner') \
                       .merge(chicken_df, on='chicken_gene_id', how='inner')
    print(f"  {len(merged):,} orthologous genes have both a human liver and chicken liver CO_Mega.")

    n_auto = (merged['is_X'] == 0).sum()
    n_x = (merged['is_X'] == 1).sum()
    print(f"  {n_auto:,} autosomal, {n_x:,} X-linked orthologous genes.")

    # CO_Mega is each species' own model's predicted log2(TE); exponentiate
    # back to a predicted (raw, always-positive) TE-like score before taking
    # the human/chicken ratio, so the ratio (and its log2) is always defined.
    merged['exp_codon_score_human'] = 2 ** merged['CO_Mega_human']
    merged['exp_codon_score_chicken'] = 2 ** merged['CO_Mega_chicken']
    merged['normalized_score'] = merged['exp_codon_score_human'] / merged['exp_codon_score_chicken']
    merged['log2_normalized_score'] = np.log2(merged['normalized_score'])
    return merged


def load_all_human_co_mega():
    """
    Load human-liver CO_Mega for ALL human protein-coding genes (autosomes +
    X), not restricted to genes with a chicken ortholog.
    """
    df = pd.read_csv(HUMAN_CO_MEGA_FILE, usecols=['gene_id', 'gene_name', 'chromosome', 'is_X', 'CO_Mega']).rename(
        columns={'CO_Mega': 'CO_Mega_human'})
    n_auto = (df['is_X'] == 0).sum()
    n_x = (df['is_X'] == 1).sum()
    print(f"  Loaded {len(df):,} human genes (all, not restricted to chicken orthologs): "
          f"{n_auto:,} autosomal, {n_x:,} X-linked.")
    return df


def plot_autosome_vs_x_defined_xci(df, value_col, value_suffix, value_label, ylabel,
                                    source_label, xci_file, exclude_values):
    """
    Autosome vs. X boxplot + CDF, restricted to X-linked genes with a
    defined XCI status (Xi-silent or Xi-expressed only, per `source_label`'s
    classification; NPX-NPY/PAR1/PAR2/No-call genes are excluded).
    """
    groups_df, _ = build_xci_groups(df, xci_file, value_col=value_col, exclude_values=exclude_values)
    defined = groups_df[groups_df['classification'].isin(['Autosome', 'Xi-silent', 'Xi-expressed'])].copy()
    defined['is_X_defined'] = (defined['classification'] != 'Autosome').astype(int)

    n_x = (defined['is_X_defined'] == 1).sum()
    print(f"  [{value_label}, {source_label}] {n_x:,} X-linked orthologous genes with defined XCI status "
          f"(Xi-silent + Xi-expressed).")

    title = (f'{value_label}: X (Xi-silent + Xi-expressed, {source_label}) vs Autosomal genes\n'
             f'(human-chicken 1:1 orthologs)')
    plot_co_mega_boxplot(
        defined, value_col, 'is_X_defined', 'X (Xi-silent + Xi-expressed)', title,
        fig(f"co_mega_boxplot_autosome_vs_X_orthologs_only_defined_XCI_{source_label}_{value_suffix}"), ylabel=ylabel
    )
    plot_co_mega_cdf(
        defined, value_col, 'is_X_defined', 'X (Xi-silent + Xi-expressed)', ylabel, title,
        fig(f"co_mega_cdf_autosome_vs_X_orthologs_only_defined_XCI_{source_label}_{value_suffix}")
    )


def plot_autosome_by_chromosome_vs_x(df, value_col, value_suffix, value_label, ylabel,
                                      name_suffix, title_suffix):
    """
    Autosome-vs-X boxplot + CDF, but with the autosomal genes split into
    their 22 individual (human) chromosomes, each shown as its own box/curve,
    against the (single, differently-colored) X group. Reports each
    autosome's p-value vs. X to a table. For the chicken-liver and
    normalized measures, the autosome/X split is still based on the HUMAN
    gene's chromosome (i.e. the ortholog's chromosome location in human).

    `name_suffix` (e.g. 'orthologs_only' / 'all_genes') is inserted into
    output filenames right after 'vs_X'; `title_suffix` (e.g.
    '(human-chicken 1:1 orthologs)' / '(all human genes)') is appended to
    the plot title.
    """
    title = f'{value_label} by chromosome: Autosomes (split by chromosome) vs X\n{title_suffix}'
    plot_co_mega_boxplot_by_chromosome(
        df, value_col, 'chromosome', AUTOSOME_ORDER, title,
        fig(f"co_mega_boxplot_autosome_by_chromosome_vs_X_{name_suffix}_{value_suffix}"),
        tbl(f"co_mega_autosome_by_chromosome_vs_X_{name_suffix}_stats_{value_suffix}"),
        ylabel=ylabel
    )
    plot_co_mega_cdf_by_chromosome(
        df, value_col, 'chromosome', AUTOSOME_ORDER, ylabel, title,
        fig(f"co_mega_cdf_autosome_by_chromosome_vs_X_{name_suffix}_{value_suffix}")
    )


def plot_xci_boxplots_by_value(df, value_col, value_suffix, value_label, ylabel, also_cdf,
                                source_label, xci_file, exclude_values):
    groups_df, other_genes = build_xci_groups(df, xci_file, value_col=value_col, exclude_values=exclude_values)
    n_x_total = (df['is_X'] == 1).sum()
    print(f"  [{value_label}, {source_label}] {len(other_genes):,} of {n_x_total:,} X-linked orthologous "
          f"genes have a defined XCI status ({n_x_total - len(other_genes):,} excluded).")

    plot_co_mega_by_group(
        groups_df, value_col, 'classification', XCI_FULL_ORDER,
        f'{value_label} by XCI category ({source_label}, human-chicken orthologs)',
        fig(f"co_mega_boxplot_XCI_categories_full_{source_label}_{value_suffix}"),
        out_csv=tbl(f"co_mega_XCI_categories_full_stats_{source_label}_{value_suffix}"),
        ylabel=ylabel
    )
    plot_co_mega_by_group(
        groups_df, value_col, 'classification', XCI_REDUCED_ORDER,
        f'{value_label}: Autosome vs Xi-silent vs Xi-expressed ({source_label}, human-chicken orthologs)',
        fig(f"co_mega_boxplot_XCI_categories_reduced_{source_label}_{value_suffix}"),
        out_csv=tbl(f"co_mega_XCI_categories_reduced_stats_{source_label}_{value_suffix}"),
        ylabel=ylabel
    )

    if also_cdf:
        plot_co_mega_cdf_by_group(
            groups_df, value_col, 'classification', XCI_FULL_ORDER, ylabel,
            f'{value_label} by XCI category ({source_label}, human-chicken orthologs)',
            fig(f"co_mega_cdf_XCI_categories_full_{source_label}_{value_suffix}")
        )
        plot_co_mega_cdf_by_group(
            groups_df, value_col, 'classification', XCI_REDUCED_ORDER, ylabel,
            f'{value_label}: Autosome vs Xi-silent vs Xi-expressed ({source_label}, human-chicken orthologs)',
            fig(f"co_mega_cdf_XCI_categories_reduced_{source_label}_{value_suffix}")
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

    print("\n[2] Boxplots + CDFs: Autosome vs X (human classification)...")
    for value_col, value_suffix, value_label, ylabel, also_cdf in VALUE_COLUMNS:
        title = f'{value_label}: X-linked vs Autosomal genes\n(human-chicken 1:1 orthologs)'
        plot_co_mega_boxplot(
            merged, value_col, 'is_X', 'X chromosome', title,
            fig(f"co_mega_boxplot_autosome_vs_X_orthologs_only_{value_suffix}"), ylabel=ylabel
        )
        plot_co_mega_cdf(
            merged, value_col, 'is_X', 'X chromosome', ylabel, title,
            fig(f"co_mega_cdf_autosome_vs_X_orthologs_only_{value_suffix}")
        )

    print("\n[3] Boxplots (+ CDFs for log2-normalized score) by XCI category (Gylemo / Neha)...")
    for value_col, value_suffix, value_label, ylabel, also_cdf in VALUE_COLUMNS:
        for source_label, xci_file, exclude_values in XCI_SOURCES:
            plot_xci_boxplots_by_value(merged, value_col, value_suffix, value_label, ylabel, also_cdf,
                                        source_label, xci_file, exclude_values)

    print("\n[4] Autosome vs X, restricted to X genes with defined XCI status (Xi-silent + Xi-expressed only)...")
    for value_col, value_suffix, value_label, ylabel, also_cdf in VALUE_COLUMNS:
        for source_label, xci_file, exclude_values in XCI_SOURCES:
            plot_autosome_vs_x_defined_xci(merged, value_col, value_suffix, value_label, ylabel,
                                            source_label, xci_file, exclude_values)

    print("\n[5] Autosome (split by chromosome) vs X...")
    for value_col, value_suffix, value_label, ylabel, also_cdf in VALUE_COLUMNS:
        plot_autosome_by_chromosome_vs_x(merged, value_col, value_suffix, value_label, ylabel,
                                          'orthologs_only', '(human-chicken 1:1 orthologs)')

    print("\n[6] For comparison: same Autosome-vs-X plots (combined + by-chromosome) using ALL "
          "human genes (human liver CO_Mega only)...")
    all_human = load_all_human_co_mega()
    value_col, value_suffix, value_label, ylabel, _ = VALUE_COLUMNS[0]  # human-liver CO_Mega
    assert value_suffix == 'human_liver'

    title = f'{value_label}: X-linked vs Autosomal genes\n(all human genes)'
    plot_co_mega_boxplot(
        all_human, value_col, 'is_X', 'X chromosome', title,
        fig(f"co_mega_boxplot_autosome_vs_X_all_genes_{value_suffix}"), ylabel=ylabel
    )
    plot_co_mega_cdf(
        all_human, value_col, 'is_X', 'X chromosome', ylabel, title,
        fig(f"co_mega_cdf_autosome_vs_X_all_genes_{value_suffix}")
    )
    plot_autosome_by_chromosome_vs_x(all_human, value_col, value_suffix, value_label, ylabel,
                                      'all_genes', '(all human genes)')

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
