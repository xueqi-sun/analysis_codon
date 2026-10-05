#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Build our own chicken CO_Mega model (codon composition -> TE) and use it to
compare CO_Mega between autosomal and Z-linked genes. Mirrors
human_co_mega_analysis.py.

Pipeline:
  1. Parse chicken CDS sequences (autosomes + Z) from
     analysis_codon/data/chicken_CDS_sequence.txt (FASTA, header
     ">gene_id|gene_id.version|transcript_id|transcript_id.version|gene_name|chromosome")
     and save as analysis_codon/data/chicken_CDS_sequence.csv.
  2. For each gene, split the CDS into codons, drop the last codon (the stop
     codon), and compute each of the 61 sense codons' frequency as
     (# that codon) / (# total codons remaining).
  3. Load per-gene TE_mean (chicken liver) from the CANONICAL chicken TE
     table, analysis_riboseq/tables/chicken_liver/translational_efficiency_data_with_ids.csv
     (produced once by analysis_riboseq/scripts/build_chicken_te_with_ids.py,
     which attaches each gene's EnsemblBioMart ID -- from
     general/genes_1to1_ortholog_human_chicken/tables/chicken_id_crosswalk.csv,
     itself built once by
     general/genes_1to1_ortholog_human_chicken/scripts/build_chicken_id_crosswalk.py
     -- to the raw TE table, which is keyed by the OLD GRCg6a Ensembl gene
     ID). This script does NOT redo that ID conversion: it just joins its
     own (already EnsemblBioMart-keyed) `gene_id` directly against the
     augmented TE table's `ensembl_biomart_gene_id`. Genes without a clean
     1:1 GRCg6a<->EnsemblBioMart ID mapping have no `ensembl_biomart_gene_id`
     in that file and are excluded here -- there is no name-based fallback.
  4. Fit  TE_mean ~ 61 codon frequencies  (OLS) using only autosomal
     (chr 1-39) genes. Report R^2 and save the intercept + 61 coefficients.
  5. Apply the fitted model to ALL genes (autosomes + Z) to get each gene's
     CO_Mega, report its correlation with TE_mean, and compare CO_Mega
     between autosomal and Z-linked genes with a boxplot + Mann-Whitney U.

All output tables/figures are suffixed with `_chicken_liver`.
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
    plot_codon_pearson_r_barplot, plot_codon_spearman_rho_barplot, parse_cds_fasta,
    run_regression_diagnostics, audit_join,
)

BASE_DIR    = "/lab/solexa_page/xueqi/analysis_codon"
CDS_FASTA   = os.path.join(BASE_DIR, "data/chicken_CDS_sequence.txt")
CDS_CSV     = os.path.join(BASE_DIR, "data/chicken_CDS_sequence.csv")
# Canonical chicken TE table (chicken Gene_ID (GRCg6a) already joined to its
# EnsemblBioMart ID once, upstream -- see build_chicken_te_with_ids.py).
TE_FILE     = ("/lab/solexa_page/xueqi/analysis_riboseq/tables/chicken_liver/"
               "translational_efficiency_data_with_ids.csv")
TABLE_DIR   = os.path.join(BASE_DIR, "tables", "chicken")
FIG_DIR     = os.path.join(BASE_DIR, "figures", "chicken")

AUTOSOMES = [str(i) for i in range(1, 40)]  # chicken autosomes: chr 1-39

SUFFIX = "chicken_liver"
TISSUE_LABEL = "Chicken liver"


def tbl(name):
    return os.path.join(TABLE_DIR, f"{name}_{SUFFIX}.csv")


def fig(name):
    return os.path.join(FIG_DIR, f"{name}_{SUFFIX}.png")


# ── Step 2: codon frequencies ───────────────────────────────────────────────
def load_codon_frequencies(cds_csv):
    df = pd.read_csv(cds_csv, usecols=['gene_id', 'gene_name', 'chromosome', 'CDS_sequence'])
    print(f"  Loaded {len(df):,} genes from {cds_csv}")

    freq_records = df['CDS_sequence'].apply(compute_codon_frequencies)
    n_unusable = freq_records.isna().sum()
    df = df.loc[freq_records.notna()].copy()
    freq_df = pd.DataFrame(list(freq_records.dropna()), index=df.index)
    print(f"  Dropped {n_unusable:,} genes with missing/unusable CDS sequences "
          f"(not a multiple of 3, or too short); {len(df):,} genes remain.")

    out = pd.concat([df[['gene_id', 'gene_name', 'chromosome']], freq_df], axis=1)
    out['chromosome'] = out['chromosome'].astype(str)
    out['is_Z'] = (out['chromosome'] == 'Z').astype(int)
    n_auto = (out['chromosome'].isin(AUTOSOMES)).sum()
    n_z = (out['chromosome'] == 'Z').sum()
    print(f"  {n_auto:,} autosomal (chr 1-39) genes, {n_z:,} Z-linked genes.")
    return out


# NOTE: this script used to do its own GRCg6a <-> EnsemblBioMart chicken
# gene-ID conversion here (via the raw GEGA table, with a name-based fallback
# for genes whose ID mapping was missing/ambiguous). That conversion now
# happens exactly ONCE, upstream, in
# general/genes_1to1_ortholog_human_chicken/scripts/build_chicken_id_crosswalk.py
# (ID-only, no name fallback -- ambiguous/unmapped genes are dropped and
# reported in chicken_id_crosswalk_audit.csv) and is propagated into the
# canonical TE table by analysis_riboseq/scripts/build_chicken_te_with_ids.py.
# This script simply reads that already-ID-augmented TE table below and
# joins directly on ensembl_biomart_gene_id (see main(), step [3]).


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=== Chicken CO_Mega model (chicken liver) ===")

    print("\n[1] Parsing chicken CDS FASTA into a CSV table...")
    parse_cds_fasta(CDS_FASTA, CDS_CSV)

    print("\n[2] Computing codon frequencies from CDS sequences...")
    codon_df = load_codon_frequencies(CDS_CSV)
    out_codon = tbl("codon_frequencies")
    codon_df.to_csv(out_codon, index=False)
    print(f"  Saved codon frequencies to {out_codon}")

    print("\n[3] Loading canonical chicken liver TE data (EnsemblBioMart IDs already attached)...")
    te_df = pd.read_csv(TE_FILE)[['ensembl_biomart_gene_id', 'TE_mean']].rename(
        columns={'ensembl_biomart_gene_id': 'te_gene_id'})
    n_te_no_id = int(te_df['te_gene_id'].isna().sum())
    te_df = te_df.dropna(subset=['TE_mean', 'te_gene_id'])
    print(f"  {len(te_df):,} chicken genes with both TE_mean and a mapped EnsemblBioMart ID in {TE_FILE} "
          f"({n_te_no_id:,} rows there had no mapped ID -- no clean 1:1 crosswalk mapping -- and are excluded).")

    # ID-based join only: codon_df['gene_id'] is already an EnsemblBioMart ID
    # (from the CDS FASTA header), matching te_df['te_gene_id'] directly.
    # No per-script ID conversion and no name-based fallback (see note above
    # build_id_conversion's old location).
    codon_df['te_gene_id'] = codon_df['gene_id']
    merged = audit_join(
        codon_df, te_df, 'te_gene_id', 'te_gene_id',
        'chicken codon frequencies (training genes)', 'chicken TE (with EnsemblBioMart IDs)',
        tbl("join_audit_codon_vs_te_train"),
    )
    print(f"  {len(merged):,} genes with both codon frequencies and TE_mean.")

    print("\n[4] Fitting TE_mean ~ codon frequencies on autosomal genes...")
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

    print("\n[4b] Regression diagnostics (linearity, homoscedasticity, normality)...")
    run_regression_diagnostics(
        model, train['TE_mean'], f'autosomal training genes, {TISSUE_LABEL}',
        fig("co_mega_regression_diagnostics_te_histogram"),
        fig("co_mega_regression_diagnostics_residual_histogram"),
        fig("co_mega_regression_diagnostics_residual_qq"),
        fig("co_mega_regression_diagnostics_residual_vs_fitted"),
        tbl("co_mega_regression_diagnostics_stats"),
    )

    print("\n[5] Computing CO_Mega for all genes (autosomes + Z) and comparing to TE...")
    codon_df['CO_Mega'] = compute_co_mega(codon_df[SENSE_CODONS], coef)

    with_te = audit_join(
        codon_df, te_df, 'te_gene_id', 'te_gene_id',
        'chicken codon frequencies (all genes)', 'chicken TE (with EnsemblBioMart IDs)',
        tbl("join_audit_codon_vs_te_all_genes"),
    )
    r_all, p_all = stats.pearsonr(with_te['CO_Mega'], with_te['TE_mean'])
    rho_all, p_rho_all = stats.spearmanr(with_te['CO_Mega'], with_te['TE_mean'])
    print(f"  All genes with TE (n={len(with_te):,}): "
          f"Pearson r = {r_all:.4f} (p = {p_all:.3g}), Spearman rho = {rho_all:.4f} (p = {p_rho_all:.3g})")

    auto_te = with_te[with_te['chromosome'].isin(AUTOSOMES)]
    r_auto, p_auto = stats.pearsonr(auto_te['CO_Mega'], auto_te['TE_mean'])
    rho_auto, p_rho_auto = stats.spearmanr(auto_te['CO_Mega'], auto_te['TE_mean'])
    print(f"  Autosomal genes only (n={len(auto_te):,}, training set): Pearson r = {r_auto:.4f} (p = {p_auto:.3g}), "
          f"Spearman rho = {rho_auto:.4f} (p = {p_rho_auto:.3g})")

    z_te = with_te[with_te['chromosome'] == 'Z']
    if len(z_te) > 1:
        r_z, p_z = stats.pearsonr(z_te['CO_Mega'], z_te['TE_mean'])
        rho_z, p_rho_z = stats.spearmanr(z_te['CO_Mega'], z_te['TE_mean'])
        print(f"  Z-linked genes only (n={len(z_te):,}, held out): Pearson r = {r_z:.4f} (p = {p_z:.3g}), "
              f"Spearman rho = {rho_z:.4f} (p = {p_rho_z:.3g})")
    else:
        r_z, p_z, rho_z, p_rho_z = np.nan, np.nan, np.nan, np.nan

    p_all_str = precise_pvalue_str(r_all, len(with_te))
    p_auto_str = precise_pvalue_str(r_auto, len(auto_te))
    p_z_str = precise_pvalue_str(r_z, len(z_te)) if len(z_te) > 1 else np.nan
    p_rho_all_str = precise_pvalue_str(rho_all, len(with_te))
    p_rho_auto_str = precise_pvalue_str(rho_auto, len(auto_te))
    p_rho_z_str = precise_pvalue_str(rho_z, len(z_te)) if len(z_te) > 1 else np.nan

    corr_stats = pd.DataFrame([
        {'gene_set': 'All genes', 'n': len(with_te), 'pearson_r': r_all, 'p_value': p_all_str},
        {'gene_set': 'Autosomal genes', 'n': len(auto_te), 'pearson_r': r_auto, 'p_value': p_auto_str},
        {'gene_set': 'Z-linked genes', 'n': len(z_te), 'pearson_r': r_z, 'p_value': p_z_str},
    ])
    out_corr = tbl("co_mega_te_correlation_stats")
    corr_stats.to_csv(out_corr, index=False)
    print(f"  Saved CO_Mega vs. TE correlation stats (n, Pearson r, p) to {out_corr}")

    corr_stats_spearman = pd.DataFrame([
        {'gene_set': 'All genes', 'n': len(with_te), 'spearman_rho': rho_all, 'p_value': p_rho_all_str},
        {'gene_set': 'Autosomal genes', 'n': len(auto_te), 'spearman_rho': rho_auto, 'p_value': p_rho_auto_str},
        {'gene_set': 'Z-linked genes', 'n': len(z_te), 'spearman_rho': rho_z, 'p_value': p_rho_z_str},
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
    if len(z_te) > 1:
        plot_actual_vs_predicted(z_te['TE_mean'], z_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                                  f'Actual TE vs. CO_Mega: Z-linked genes ({TISSUE_LABEL})', r_z, p_z_str,
                                  fig("co_mega_vs_te_Z_linked"))
        plot_actual_vs_predicted_spearman(z_te['TE_mean'], z_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                                           f'Actual TE vs. CO_Mega: Z-linked genes ({TISSUE_LABEL})', rho_z,
                                           p_rho_z_str, fig("co_mega_vs_te_Z_linked_spearman"))

    out_all = tbl("co_mega_all_genes")
    all_genes_with_te = audit_join(
        codon_df[['gene_id', 'gene_name', 'chromosome', 'is_Z', 'te_gene_id', 'CO_Mega']], te_df,
        'te_gene_id', 'te_gene_id',
        'chicken codon frequencies (all genes, for output table)', 'chicken TE (with EnsemblBioMart IDs)',
        tbl("join_audit_codon_vs_te_output_table"), how='left',
    )
    all_genes_with_te.to_csv(out_all, index=False)
    print(f"  Saved CO_Mega for all {len(codon_df):,} genes to {out_all}")

    print("\n[6] Boxplot: CO_Mega, autosomal vs Z-linked genes...")
    plot_co_mega_boxplot(codon_df, 'CO_Mega', 'is_Z', 'Z chromosome',
                          f'CO_Mega: Z-linked vs Autosomal genes ({TISSUE_LABEL})',
                          fig("co_mega_boxplot_autosome_vs_Z"))

    print("\n[7] Per-codon Pearson r with TE (bootstrap SD, autosomal training genes)...")
    r_df = bootstrap_codon_pearson_r(train[SENSE_CODONS], train['TE_mean'], SENSE_CODONS)
    out_codon_r = tbl("co_mega_codon_pearson_r")
    r_df.to_csv(out_codon_r, index=False)
    print(f"  Saved per-codon Pearson r (+ bootstrap SD) to {out_codon_r}")
    plot_codon_pearson_r_barplot(
        r_df, f'Pearson R between codon frequency and TE, per codon\n(autosomal training genes, {TISSUE_LABEL})',
        fig("co_mega_codon_pearson_r_barplot")
    )

    print("\n[7b] Per-codon Spearman rho with TE (bootstrap SD, autosomal training genes)...")
    rho_df = bootstrap_codon_spearman_rho(train[SENSE_CODONS], train['TE_mean'], SENSE_CODONS)
    out_codon_rho = tbl("co_mega_codon_spearman_rho")
    rho_df.to_csv(out_codon_rho, index=False)
    print(f"  Saved per-codon Spearman rho (+ bootstrap SD) to {out_codon_rho}")
    plot_codon_spearman_rho_barplot(
        rho_df, f'Spearman rho between codon frequency and TE, per codon\n(autosomal training genes, {TISSUE_LABEL})',
        fig("co_mega_codon_pearson_r_barplot_spearman")
    )

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
