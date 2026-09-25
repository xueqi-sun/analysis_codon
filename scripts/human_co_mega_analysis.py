#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Build our own human CO_Mega model (codon composition -> TE) and use it to
compare CO_Mega between autosomal, X-linked, and Y-linked genes.

Pipeline:
  1. Load CDS sequences for all human protein-coding genes (autosomes + X +
     Y) from their MANE Select transcripts, extracted by
     general/sequence_CDS/human/scripts/extract_human_cds.py from the NCBI
     RefSeq GRCh38.p14 annotation (general/sequence_CDS/human/tables/
     human_CDS_sequences_MANE_Select.csv).
  2. For each gene, split the CDS into codons, drop the last codon (the stop
     codon), and compute each of the 61 sense codons' frequency as
     (# that codon) / (# total codons remaining).
  3. Load per-gene TE_mean (human liver) from
     analysis_riboseq/tables/human_liver/all_human_genes_merged_data.csv,
     merging on gene_name/symbol (the new CDS table's `gene_id` is actually
     the RefSeq gene symbol, not an Ensembl ID, so symbol is the only
     identifier shared with the Ensembl-ID-keyed TE table).
  4. Fit  TE_mean ~ 61 codon frequencies  (OLS) using only autosomal
     (chr 1-22) genes. Report R^2 and save the intercept + 61 coefficients.
  5. Apply the fitted model to ALL genes (autosomes + X + Y) to get each
     gene's CO_Mega, report its correlation with TE_mean, and compare
     CO_Mega between autosomal, X-linked, and Y-linked genes with boxplots
     + Mann-Whitney U.

All output tables/figures are suffixed with `_human_liver`, and saved under
dedicated `tables/human/` and `figures/human/` subfolders.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from co_mega import SENSE_CODONS, compute_co_mega
from co_mega_common import (
    precise_pvalue_str, compute_codon_frequencies, fit_co_mega_te_model,
    save_coefficient_table, plot_actual_vs_predicted, plot_actual_vs_predicted_spearman,
    plot_co_mega_boxplot, bootstrap_codon_pearson_r, bootstrap_codon_spearman_rho,
    plot_codon_pearson_r_barplot, plot_codon_spearman_rho_barplot,
    plot_co_mega_by_group, build_xci_groups,
    make_autosome_x_y_group, run_regression_diagnostics,
)

BASE_DIR  = "/lab/solexa_page/xueqi/analysis_codon"
CDS_FILE  = "/lab/solexa_page/xueqi/general/sequence_CDS/human/tables/human_CDS_sequences_MANE_Select.csv"
TE_FILE   = "/lab/solexa_page/xueqi/analysis_riboseq/tables/human_liver/all_human_genes_merged_data.csv"
# Y-linked genes are NOT in TE_FILE (0 rows there); computed separately
# (samples 1 & 2 only -- sample 3 is female, 0 reads on Y genes) by
# analysis_riboseq/scripts/human_liver_te_chromosome_Y.py.
TE_FILE_Y = "/lab/solexa_page/xueqi/analysis_riboseq/tables/human_liver/chromosome_te_analysis_Y.csv"
XCI_FILE_GYLEMO = "/lab/solexa_page/xueqi/general/genes_XCI_Gylemo/tables/chrX_genes_classification_comparison_gylemo_liver_thresh0p05.csv"
XCI_FILE_NEHA   = "/lab/solexa_page/xueqi/general/genes_XCI_Neha/tables/chrX_genes_classification_Neha.csv"
TABLE_DIR = os.path.join(BASE_DIR, "tables", "human")
FIG_DIR   = os.path.join(BASE_DIR, "figures", "human")

AUTOSOMES = [str(i) for i in range(1, 23)]

SUFFIX = "human_liver"
TISSUE_LABEL = "Human liver"


def tbl(name):
    return os.path.join(TABLE_DIR, f"{name}_{SUFFIX}.csv")


def fig(name):
    return os.path.join(FIG_DIR, f"{name}_{SUFFIX}.png")


# ── Codon frequencies ───────────────────────────────────────────────────────
def load_codon_frequencies(cds_file):
    df = pd.read_csv(cds_file, usecols=['gene_id', 'gene_name', 'chromosome', 'CDS_sequence'])
    print(f"  Loaded {len(df):,} genes from {cds_file}")

    freq_records = df['CDS_sequence'].apply(compute_codon_frequencies)
    n_unusable = freq_records.isna().sum()
    df = df.loc[freq_records.notna()].copy()
    freq_df = pd.DataFrame(list(freq_records.dropna()), index=df.index)
    print(f"  Dropped {n_unusable:,} genes with missing/unusable CDS sequences "
          f"(not a multiple of 3, or too short); {len(df):,} genes remain.")

    out = pd.concat([df[['gene_id', 'gene_name', 'chromosome']], freq_df], axis=1)
    out['chromosome'] = out['chromosome'].astype(str)
    out['is_X'] = (out['chromosome'] == 'X').astype(int)
    out['is_Y'] = (out['chromosome'] == 'Y').astype(int)
    n_auto = (out['chromosome'].isin(AUTOSOMES)).sum()
    n_x = (out['chromosome'] == 'X').sum()
    n_y = (out['chromosome'] == 'Y').sum()
    print(f"  {n_auto:,} autosomal (chr 1-22) genes, {n_x:,} X-linked genes, {n_y:,} Y-linked genes.")
    return out


# ── Plot: XCI-category boxplots (human-specific) ────────────────────────────
def plot_xci_boxplots(codon_df, xci_file, source_label, exclude_values=()):
    """
    For one XCI classification source (e.g. Gylemo or Neha), build and save:
      - a boxplot + stats table comparing Autosome vs X-linked genes with a
        defined XCI status (i.e. found in `xci_file`, excluding
        `exclude_values`)
      - a boxplot + stats table across the full set of XCI categories
        (Autosome, Xi-silent, Xi-expressed, NPX-NPY, PAR1, PAR2)
      - a boxplot + stats table across the reduced set of categories
        (Autosome, Xi-silent, Xi-expressed)
    Output files are all suffixed with `_{source_label}_human_liver`.
    """
    groups_df, x_genes = build_xci_groups(codon_df, xci_file, exclude_values=exclude_values)
    n_x_total = (codon_df['is_X'] == 1).sum()
    print(f"  {len(x_genes):,} of {n_x_total:,} X-linked genes have a defined XCI status in the "
          f"{source_label} table ({n_x_total - len(x_genes):,} excluded from these boxplots).")

    defined_xci_df = groups_df.copy()
    defined_xci_df['group'] = np.where(defined_xci_df['classification'] == 'Autosome', 'Autosome', 'X (defined XCI)')
    plot_co_mega_by_group(
        defined_xci_df, 'CO_Mega', 'group', ['Autosome', 'X (defined XCI)'],
        f'CO_Mega: Autosome vs X (defined XCI status, {source_label}, {TISSUE_LABEL})',
        fig(f"co_mega_boxplot_autosome_vs_X_with_defined_XCI_{source_label}"),
        out_csv=tbl(f"co_mega_autosome_vs_X_with_defined_XCI_stats_{source_label}")
    )

    plot_co_mega_by_group(
        groups_df, 'CO_Mega', 'classification',
        ['Autosome', 'Xi-silent', 'Xi-expressed', 'NPX-NPY', 'PAR1', 'PAR2'],
        f'CO_Mega by XCI category ({source_label}, {TISSUE_LABEL})',
        fig(f"co_mega_boxplot_XCI_categories_full_{source_label}"),
        out_csv=tbl(f"co_mega_XCI_categories_full_stats_{source_label}")
    )
    plot_co_mega_by_group(
        groups_df, 'CO_Mega', 'classification',
        ['Autosome', 'Xi-silent', 'Xi-expressed'],
        f'CO_Mega: Autosome vs Xi-silent vs Xi-expressed ({source_label}, {TISSUE_LABEL})',
        fig(f"co_mega_boxplot_XCI_categories_reduced_{source_label}"),
        out_csv=tbl(f"co_mega_XCI_categories_reduced_stats_{source_label}")
    )


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=== Human CO_Mega model (human liver) ===")

    print("\n[1] Computing codon frequencies from CDS sequences...")
    codon_df = load_codon_frequencies(CDS_FILE)
    out_codon = tbl("codon_frequencies")
    codon_df.to_csv(out_codon, index=False)
    print(f"  Saved codon frequencies to {out_codon}")

    print("\n[2] Loading human liver TE data...")
    # Merge on gene_name/symbol: the new CDS table's `gene_id` (from the NCBI
    # RefSeq GTF) is actually the gene symbol, not an Ensembl ID, so gene_name
    # is the only identifier shared with this Ensembl-ID-keyed TE table.
    te_df = pd.read_csv(TE_FILE)[['gene_name', 'TE_mean']]
    te_df_y = pd.read_csv(TE_FILE_Y)[['gene_name', 'TE_mean']]
    te_df = pd.concat([te_df, te_df_y], ignore_index=True)
    te_df = te_df.dropna(subset=['TE_mean', 'gene_name'])
    print(f"  {len(te_df):,} genes with TE_mean ({len(te_df_y):,} Y-linked, from {TE_FILE_Y}).")

    merged = codon_df.merge(te_df, on='gene_name', how='inner')
    print(f"  {len(merged):,} genes with both codon frequencies and TE_mean.")

    print("\n[3] Fitting TE_mean ~ codon frequencies on autosomal genes...")
    train = merged[merged['chromosome'].isin(AUTOSOMES)]
    print(f"  Training on {len(train):,} autosomal genes, {len(SENSE_CODONS)} codon predictors.")
    coef, model = fit_co_mega_te_model(train[SENSE_CODONS], train['TE_mean'])
    print(f"  R^2 = {model.rsquared:.4f}   Adjusted R^2 = {model.rsquared_adj:.4f}   n = {int(model.nobs)}")

    out_coef = tbl("co_mega_model_coefficients")
    save_coefficient_table(coef, model, SENSE_CODONS, out_coef)

    r_train, _ = stats.pearsonr(train['TE_mean'], model.fittedvalues)
    rho_train, _ = stats.spearmanr(train['TE_mean'], model.fittedvalues)
    p_train_str = precise_pvalue_str(r_train, len(train))
    p_train_rho_str = precise_pvalue_str(rho_train, len(train))
    plot_actual_vs_predicted(
        train['TE_mean'], model.fittedvalues,
        'Actual TE_mean (autosomal training genes)', 'Fitted TE_mean (OLS model)',
        f'Original regression: TE ~ codon frequencies\n(autosomal training genes, {TISSUE_LABEL})',
        r_train, p_train_str, fig("co_mega_regression_fit_train")
    )
    plot_actual_vs_predicted_spearman(
        train['TE_mean'], model.fittedvalues,
        'Actual TE_mean (autosomal training genes)', 'Fitted TE_mean (OLS model)',
        f'Original regression: TE ~ codon frequencies\n(autosomal training genes, {TISSUE_LABEL})',
        rho_train, p_train_rho_str, fig("co_mega_regression_fit_train_spearman")
    )

    print("\n[3b] Regression diagnostics (linearity, homoscedasticity, normality)...")
    run_regression_diagnostics(
        model, train['TE_mean'], f'autosomal training genes, {TISSUE_LABEL}',
        fig("co_mega_regression_diagnostics_te_histogram"),
        fig("co_mega_regression_diagnostics_residual_histogram"),
        fig("co_mega_regression_diagnostics_residual_qq"),
        fig("co_mega_regression_diagnostics_residual_vs_fitted"),
        tbl("co_mega_regression_diagnostics_stats"),
    )

    print("\n[4] Computing CO_Mega for all genes (autosomes + X + Y) and comparing to TE...")
    codon_df['CO_Mega'] = compute_co_mega(codon_df[SENSE_CODONS], coef)

    with_te = codon_df.merge(te_df, on='gene_name', how='inner')
    r_all, p_all = stats.pearsonr(with_te['CO_Mega'], with_te['TE_mean'])
    rho_all, p_rho_all = stats.spearmanr(with_te['CO_Mega'], with_te['TE_mean'])
    print(f"  All genes with TE (n={len(with_te):,}): "
          f"Pearson r = {r_all:.4f} (p = {p_all:.3g}), Spearman rho = {rho_all:.4f} (p = {p_rho_all:.3g})")

    auto_te = with_te[with_te['chromosome'].isin(AUTOSOMES)]
    r_auto, p_auto = stats.pearsonr(auto_te['CO_Mega'], auto_te['TE_mean'])
    rho_auto, p_rho_auto = stats.spearmanr(auto_te['CO_Mega'], auto_te['TE_mean'])
    print(f"  Autosomal genes only (n={len(auto_te):,}, training set): Pearson r = {r_auto:.4f} (p = {p_auto:.3g}), "
          f"Spearman rho = {rho_auto:.4f} (p = {p_rho_auto:.3g})")

    x_te = with_te[with_te['chromosome'] == 'X']
    if len(x_te) > 1:
        r_x, p_x = stats.pearsonr(x_te['CO_Mega'], x_te['TE_mean'])
        rho_x, p_rho_x = stats.spearmanr(x_te['CO_Mega'], x_te['TE_mean'])
        print(f"  X-linked genes only (n={len(x_te):,}, held out): Pearson r = {r_x:.4f} (p = {p_x:.3g}), "
              f"Spearman rho = {rho_x:.4f} (p = {p_rho_x:.3g})")
    else:
        r_x, p_x, rho_x, p_rho_x = np.nan, np.nan, np.nan, np.nan

    y_te = with_te[with_te['chromosome'] == 'Y']
    if len(y_te) > 1:
        r_y, p_y = stats.pearsonr(y_te['CO_Mega'], y_te['TE_mean'])
        rho_y, p_rho_y = stats.spearmanr(y_te['CO_Mega'], y_te['TE_mean'])
        print(f"  Y-linked genes only (n={len(y_te):,}, held out): Pearson r = {r_y:.4f} (p = {p_y:.3g}), "
              f"Spearman rho = {rho_y:.4f} (p = {p_rho_y:.3g})")
    else:
        r_y, p_y, rho_y, p_rho_y = np.nan, np.nan, np.nan, np.nan

    p_all_str = precise_pvalue_str(r_all, len(with_te))
    p_auto_str = precise_pvalue_str(r_auto, len(auto_te))
    p_x_str = precise_pvalue_str(r_x, len(x_te)) if len(x_te) > 1 else np.nan
    p_y_str = precise_pvalue_str(r_y, len(y_te)) if len(y_te) > 1 else np.nan
    p_rho_all_str = precise_pvalue_str(rho_all, len(with_te))
    p_rho_auto_str = precise_pvalue_str(rho_auto, len(auto_te))
    p_rho_x_str = precise_pvalue_str(rho_x, len(x_te)) if len(x_te) > 1 else np.nan
    p_rho_y_str = precise_pvalue_str(rho_y, len(y_te)) if len(y_te) > 1 else np.nan

    corr_stats = pd.DataFrame([
        {'gene_set': 'All genes', 'n': len(with_te), 'pearson_r': r_all, 'p_value': p_all_str},
        {'gene_set': 'Autosomal genes', 'n': len(auto_te), 'pearson_r': r_auto, 'p_value': p_auto_str},
        {'gene_set': 'X-linked genes', 'n': len(x_te), 'pearson_r': r_x, 'p_value': p_x_str},
        {'gene_set': 'Y-linked genes', 'n': len(y_te), 'pearson_r': r_y, 'p_value': p_y_str},
    ])
    out_corr = tbl("co_mega_te_correlation_stats")
    corr_stats.to_csv(out_corr, index=False)
    print(f"  Saved CO_Mega vs. TE correlation stats (n, Pearson r, p) to {out_corr}")

    corr_stats_spearman = pd.DataFrame([
        {'gene_set': 'All genes', 'n': len(with_te), 'spearman_rho': rho_all, 'p_value': p_rho_all_str},
        {'gene_set': 'Autosomal genes', 'n': len(auto_te), 'spearman_rho': rho_auto, 'p_value': p_rho_auto_str},
        {'gene_set': 'X-linked genes', 'n': len(x_te), 'spearman_rho': rho_x, 'p_value': p_rho_x_str},
        {'gene_set': 'Y-linked genes', 'n': len(y_te), 'spearman_rho': rho_y, 'p_value': p_rho_y_str},
    ])
    out_corr_spearman = tbl("co_mega_te_correlation_stats_spearman")
    corr_stats_spearman.to_csv(out_corr_spearman, index=False)
    print(f"  Saved CO_Mega vs. TE correlation stats (n, Spearman rho, p) to {out_corr_spearman}")

    plot_actual_vs_predicted(with_te['TE_mean'], with_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                              f'Actual TE vs. CO_Mega: All genes ({TISSUE_LABEL})', r_all, p_all_str,
                              fig("co_mega_vs_te_all_genes"))
    plot_actual_vs_predicted_spearman(with_te['TE_mean'], with_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                                       f'Actual TE vs. CO_Mega: All genes ({TISSUE_LABEL})', rho_all, p_rho_all_str,
                                       fig("co_mega_vs_te_all_genes_spearman"))
    plot_actual_vs_predicted(auto_te['TE_mean'], auto_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                              f'Actual TE vs. CO_Mega: Autosomal genes ({TISSUE_LABEL})', r_auto, p_auto_str,
                              fig("co_mega_vs_te_autosomal"))
    plot_actual_vs_predicted_spearman(auto_te['TE_mean'], auto_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                                       f'Actual TE vs. CO_Mega: Autosomal genes ({TISSUE_LABEL})', rho_auto,
                                       p_rho_auto_str, fig("co_mega_vs_te_autosomal_spearman"))
    if len(x_te) > 1:
        plot_actual_vs_predicted(x_te['TE_mean'], x_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                                  f'Actual TE vs. CO_Mega: X-linked genes ({TISSUE_LABEL})', r_x, p_x_str,
                                  fig("co_mega_vs_te_X_linked"))
        plot_actual_vs_predicted_spearman(x_te['TE_mean'], x_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                                           f'Actual TE vs. CO_Mega: X-linked genes ({TISSUE_LABEL})', rho_x,
                                           p_rho_x_str, fig("co_mega_vs_te_X_linked_spearman"))
    if len(y_te) > 1:
        plot_actual_vs_predicted(y_te['TE_mean'], y_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                                  f'Actual TE vs. CO_Mega: Y-linked genes ({TISSUE_LABEL})', r_y, p_y_str,
                                  fig("co_mega_vs_te_Y_linked"))
        plot_actual_vs_predicted_spearman(y_te['TE_mean'], y_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                                           f'Actual TE vs. CO_Mega: Y-linked genes ({TISSUE_LABEL})', rho_y,
                                           p_rho_y_str, fig("co_mega_vs_te_Y_linked_spearman"))

    out_all = tbl("co_mega_all_genes")
    codon_df[['gene_id', 'gene_name', 'chromosome', 'is_X', 'is_Y', 'CO_Mega']].merge(
        te_df, on='gene_name', how='left'
    ).to_csv(out_all, index=False)
    print(f"  Saved CO_Mega for all {len(codon_df):,} genes to {out_all}")

    print("\n[5] Boxplot: CO_Mega, autosomal vs X-linked genes...")
    plot_co_mega_boxplot(codon_df, 'CO_Mega', 'is_X', 'X chromosome',
                          f'CO_Mega: X-linked vs Autosomal genes ({TISSUE_LABEL})',
                          fig("co_mega_boxplot_autosome_vs_X"))

    print("\n[5b] Boxplot: CO_Mega, autosomal vs X-linked vs Y-linked genes...")
    codon_df['group'] = make_autosome_x_y_group(codon_df['chromosome'])
    plot_co_mega_by_group(
        codon_df, 'CO_Mega', 'group', ['Autosome', 'X', 'Y'],
        f'CO_Mega: Autosome vs X vs Y ({TISSUE_LABEL})',
        fig("co_mega_boxplot_autosome_vs_X_vs_Y"),
        out_csv=tbl("co_mega_autosome_vs_X_vs_Y_stats")
    )

    print("\n[6] Per-codon Pearson r with TE (bootstrap SD, autosomal training genes)...")
    r_df = bootstrap_codon_pearson_r(train[SENSE_CODONS], train['TE_mean'], SENSE_CODONS)
    out_codon_r = tbl("co_mega_codon_pearson_r")
    r_df.to_csv(out_codon_r, index=False)
    print(f"  Saved per-codon Pearson r (+ bootstrap SD) to {out_codon_r}")
    plot_codon_pearson_r_barplot(
        r_df, f'Pearson R between codon frequency and TE, per codon\n(autosomal training genes, {TISSUE_LABEL})',
        fig("co_mega_codon_pearson_r_barplot")
    )

    print("\n[6b] Per-codon Spearman rho with TE (bootstrap SD, autosomal training genes)...")
    rho_df = bootstrap_codon_spearman_rho(train[SENSE_CODONS], train['TE_mean'], SENSE_CODONS)
    out_codon_rho = tbl("co_mega_codon_spearman_rho")
    rho_df.to_csv(out_codon_rho, index=False)
    print(f"  Saved per-codon Spearman rho (+ bootstrap SD) to {out_codon_rho}")
    plot_codon_spearman_rho_barplot(
        rho_df, f'Spearman rho between codon frequency and TE, per codon\n(autosomal training genes, {TISSUE_LABEL})',
        fig("co_mega_codon_pearson_r_barplot_spearman")
    )

    print("\n[7] Boxplots: CO_Mega by XCI category (Gylemo classification)...")
    plot_xci_boxplots(codon_df, XCI_FILE_GYLEMO, 'Gylemo')

    print("\n[8] Boxplots: CO_Mega by XCI category (Neha classification)...")
    plot_xci_boxplots(codon_df, XCI_FILE_NEHA, 'Neha', exclude_values=('No call',))

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
