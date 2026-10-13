#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Test: does the 5' UTR codon composition predict TE the way CDS codon
composition does?  Same pipeline as human_co_mega_analysis.py, but the
sequence fed into the codon-frequency step is the 5' UTR (`utr5_sequence`)
instead of the CDS.

Pipeline:
  1. Load analysis_codon/data/human_UTR_sequence.csv and use `utr5_sequence`
     as the "CDS" input. Its `tx_id` is merged with `transcript_id` in
     analysis_codon/data/human_CDS_sequence.csv to bring in gene_id (Ensembl
     gene ID, no version), gene_name and chromosome, so that the downstream
     TE mapping (by Ensembl gene ID) carries over unchanged.
  2. Trim each UTR to a multiple of 3 (drop the last 1 or 2 nt), split it into
     non-overlapping 3-letter codons, and compute each of the 61 sense codons'
     frequency as (# that codon) / (# total codons). Unlike a real CDS there
     is no stop codon to drop, so every codon stays in the denominator
     (in-frame TAA/TAG/TGA included). UTRs shorter than 3 nt are dropped.
  3. Load per-gene TE_mean (human liver) and merge on Ensembl gene ID.
  4. Fit  TE_mean ~ 61 codon frequencies  (OLS) on autosomal (chr 1-22) genes.
  5. Apply the model to all genes, report CO_Mega vs TE correlations (Pearson
     and Spearman) for all / autosomal / X / Y genes, with scatter plots.

Outputs are suffixed `_human_liver_UTR` and saved under tables/human_UTR/ and
figures/human_UTR/ so the CDS-based outputs are not overwritten.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from co_mega import SENSE_CODONS, compute_co_mega
from co_mega_common import (
    precise_pvalue_str, chromosome_groups, fit_co_mega_te_model, save_coefficient_table,
    plot_actual_vs_predicted, plot_actual_vs_predicted_spearman,
    run_regression_diagnostics, audit_join, check_key_uniqueness, plot_co_mega_boxplot,
)

BASE_DIR  = "/lab/solexa_page/xueqi/analysis_codon"
UTR_FILE  = os.path.join(BASE_DIR, "data", "human_UTR_sequence.csv")
CDS_FILE  = os.path.join(BASE_DIR, "data", "human_CDS_sequence.csv")
TE_FILE   = "/lab/solexa_page/xueqi/analysis_riboseq/tables/human_liver/all_human_genes_merged_data.csv"
# Y-linked genes are NOT in TE_FILE; computed separately by
# analysis_riboseq/scripts/human_liver_te_chromosome_Y.py.
TE_FILE_Y = "/lab/solexa_page/xueqi/analysis_riboseq/tables/human_liver/chromosome_te_analysis_Y.csv"
TABLE_DIR = os.path.join(BASE_DIR, "tables", "human_UTR")
FIG_DIR   = os.path.join(BASE_DIR, "figures", "human_UTR")

AUTOSOMES = [str(i) for i in range(1, 23)]

SUFFIX = "human_liver_UTR"
TISSUE_LABEL = "Human liver, 5' UTR"


def tbl(name):
    return os.path.join(TABLE_DIR, f"{name}_{SUFFIX}.csv")


def fig(name):
    return os.path.join(FIG_DIR, f"{name}_{SUFFIX}.png")


# ── Codon frequencies ───────────────────────────────────────────────────────
def compute_utr_codon_frequencies(seq, codons=SENSE_CODONS):
    """
    Trim `seq` to a multiple of 3 (drop the last 1-2 nt), split into
    non-overlapping codons, and return each of `codons`' frequency as
    (# that codon) / (# total codons). Drop the last codon (stop codon).
    Returns None if the sequence is missing or shorter than one codon.
    """
    if not isinstance(seq, str):
        return None
    seq = seq.upper()
    n_codons = len(seq) // 3
    if n_codons < 2:
        return None
    counts = Counter(seq[i:i + 3] for i in range(0, n_codons * 3, 3))
    return {c: counts.get(c, 0) / (n_codons - 1) for c in codons}


def load_codon_frequencies(utr_file, cds_file):
    utr = pd.read_csv(utr_file, usecols=['tx_id', 'utr5_sequence'])
    cds = pd.read_csv(cds_file, usecols=['transcript_id', 'gene_id', 'gene_name', 'chromosome'])
    print(f"  Loaded {len(utr):,} UTR rows from {utr_file}")
    print(f"  Loaded {len(cds):,} transcripts from {cds_file}")

    check_key_uniqueness(utr, 'tx_id', 'human_UTR_sequence.csv')
    check_key_uniqueness(cds, 'transcript_id', 'human_CDS_sequence.csv')
    df = audit_join(
        utr, cds, 'tx_id', 'transcript_id',
        'human 5\' UTR (tx_id)', 'human CDS table (transcript_id)',
        tbl("join_audit_utr_vs_cds_table"),
    )
    # gene_id in this table is the unversioned Ensembl gene ID, which is what
    # the TE table's Gene_ID uses.
    df = df.rename(columns={'gene_id': 'ensembl_gene_id'})
    print(f"  {len(df):,} transcripts with both a UTR and gene/chromosome annotation.")

    n_trim = (df['utr5_sequence'].str.len() % 3 != 0).sum()
    print(f"  {n_trim:,} UTRs are not a multiple of 3 and will be trimmed (last 1-2 nt dropped).")

    freq_records = df['utr5_sequence'].apply(compute_utr_codon_frequencies)
    n_unusable = freq_records.isna().sum()
    df = df.loc[freq_records.notna()].copy()
    freq_df = pd.DataFrame(list(freq_records.dropna()), index=df.index)
    print(f"  Dropped {n_unusable:,} transcripts with missing/too-short (<3 nt) UTRs; {len(df):,} remain.")

    out = pd.concat([df[['tx_id', 'ensembl_gene_id', 'gene_name', 'chromosome']], freq_df], axis=1)
    out['chromosome'] = out['chromosome'].astype(str)
    out['is_X'] = (out['chromosome'] == 'X').astype(int)
    out['is_Y'] = (out['chromosome'] == 'Y').astype(int)
    n_auto = out['chromosome'].isin(AUTOSOMES).sum()
    print(f"  {n_auto:,} autosomal (chr 1-22) genes, {out['is_X'].sum():,} X-linked, {out['is_Y'].sum():,} Y-linked.")
    return out


def correlate(df, label):
    """Pearson + Spearman of CO_Mega vs TE_mean; NaNs if fewer than 2 genes."""
    if len(df) < 2:
        return dict(label=label, n=len(df), r=np.nan, p_r=np.nan, rho=np.nan, p_rho=np.nan,
                    p_r_str=np.nan, p_rho_str=np.nan)
    r, p_r = stats.pearsonr(df['CO_Mega'], df['TE_mean'])
    rho, p_rho = stats.spearmanr(df['CO_Mega'], df['TE_mean'])
    print(f"  {label} (n={len(df):,}): Pearson r = {r:.4f} (p = {p_r:.3g}), "
          f"Spearman rho = {rho:.4f} (p = {p_rho:.3g})")
    return dict(label=label, n=len(df), r=r, p_r=p_r, rho=rho, p_rho=p_rho,
                p_r_str=precise_pvalue_str(r, len(df)), p_rho_str=precise_pvalue_str(rho, len(df)))



# ── Main ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=== Human CO_Mega model, 5' UTR codons (human liver) ===")

    print("\n[1] Computing codon frequencies from 5' UTR sequences...")
    codon_df = load_codon_frequencies(UTR_FILE, CDS_FILE)
    out_codon = tbl("codon_frequencies")
    codon_df.to_csv(out_codon, index=False)
    print(f"  Saved codon frequencies to {out_codon}")

    print("\n[2] Loading human liver TE data...")
    te_raw = pd.read_csv(TE_FILE).rename(columns={'Gene_ID': 'ensembl_gene_id'})
    check_key_uniqueness(te_raw, 'ensembl_gene_id', 'all_human_genes_merged_data.csv (TE_FILE)')
    te_df = te_raw[['ensembl_gene_id', 'TE_mean']]
    te_df_y = pd.read_csv(TE_FILE_Y).rename(columns={'Gene_ID': 'ensembl_gene_id'})[['ensembl_gene_id', 'TE_mean']]
    te_df = pd.concat([te_df, te_df_y], ignore_index=True)
    te_df = te_df.dropna(subset=['TE_mean', 'ensembl_gene_id'])
    print(f"  {len(te_df):,} genes with TE_mean ({len(te_df_y):,} Y-linked, from {TE_FILE_Y}).")

    merged = audit_join(
        codon_df, te_df, 'ensembl_gene_id', 'ensembl_gene_id',
        'human UTR codon frequencies (Ensembl gene ID)', 'human TE (Ensembl gene ID)',
        tbl("join_audit_codon_vs_te_train"),
    )
    print(f"  {len(merged):,} genes with both UTR codon frequencies and TE_mean.")

    print("\n[3] Fitting TE_mean ~ UTR codon frequencies on autosomal genes...")
    train = merged[merged['chromosome'].isin(AUTOSOMES)]
    print(f"  Training on {len(train):,} autosomal genes, {len(SENSE_CODONS)} codon predictors.")
    coef, model = fit_co_mega_te_model(train[SENSE_CODONS], train['TE_mean'])
    print(f"  R^2 = {model.rsquared:.4f}   Adjusted R^2 = {model.rsquared_adj:.4f}   n = {int(model.nobs)}")
    save_coefficient_table(coef, model, SENSE_CODONS, tbl("co_mega_model_coefficients"))

    r_train, _ = stats.pearsonr(train['TE_mean'], model.fittedvalues)
    rho_train, _ = stats.spearmanr(train['TE_mean'], model.fittedvalues)
    fit_title = f'Regression: TE ~ UTR codon frequencies\n(autosomal training genes, {TISSUE_LABEL})'
    plot_actual_vs_predicted(
        train['TE_mean'], model.fittedvalues,
        'Actual TE_mean (autosomal training genes)', 'Fitted TE_mean (OLS model)',
        fit_title, r_train, precise_pvalue_str(r_train, len(train)), fig("co_mega_regression_fit_train"))
    plot_actual_vs_predicted_spearman(
        train['TE_mean'], model.fittedvalues,
        'Actual TE_mean (autosomal training genes)', 'Fitted TE_mean (OLS model)',
        fit_title, rho_train, precise_pvalue_str(rho_train, len(train)),
        fig("co_mega_regression_fit_train_spearman"))

    print("\n[3b] Regression diagnostics...")
    run_regression_diagnostics(
        model, train['TE_mean'], f'autosomal training genes, {TISSUE_LABEL}',
        fig("co_mega_regression_diagnostics_te_histogram"),
        fig("co_mega_regression_diagnostics_residual_histogram"),
        fig("co_mega_regression_diagnostics_residual_qq"),
        fig("co_mega_regression_diagnostics_residual_vs_fitted"),
        tbl("co_mega_regression_diagnostics_stats"),
    )

    print("\n[4] Computing CO_Mega for all genes and comparing to TE...")
    codon_df['CO_Mega'] = compute_co_mega(codon_df[SENSE_CODONS], coef)
    with_te = audit_join(
        codon_df, te_df, 'ensembl_gene_id', 'ensembl_gene_id',
        'human UTR codon frequencies (all genes)', 'human TE (Ensembl gene ID)',
        tbl("join_audit_codon_vs_te_all_genes"),
    )

    subsets = [
        ('All genes', 'all_genes', with_te),
        ('Autosomal genes', 'autosomal', with_te[with_te['chromosome'].isin(AUTOSOMES)]),
        ('X-linked genes', 'X_linked', with_te[with_te['chromosome'] == 'X']),
        ('Y-linked genes', 'Y_linked', with_te[with_te['chromosome'] == 'Y']),
    ]
    rows = []
    for label, key, sub in subsets:
        s = correlate(sub, label)
        rows.append(s)
        if len(sub) < 2:
            continue
        title = f'Actual TE vs. CO_Mega (5\' UTR): {label} ({TISSUE_LABEL})'
        grp = chromosome_groups(sub) if key == 'all_genes' else None  # autosome / X colors on the all-genes scatter
        plot_actual_vs_predicted(sub['TE_mean'], sub['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                                 title, s['r'], s['p_r_str'], fig(f"co_mega_vs_te_{key}"), groups=grp)
        plot_actual_vs_predicted_spearman(sub['TE_mean'], sub['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                                          title, s['rho'], s['p_rho_str'], fig(f"co_mega_vs_te_{key}_spearman"),
                                          groups=grp)

    pd.DataFrame([{'gene_set': s['label'], 'n': s['n'], 'pearson_r': s['r'], 'p_value': s['p_r_str']}
                  for s in rows]).to_csv(tbl("co_mega_te_correlation_stats"), index=False)
    pd.DataFrame([{'gene_set': s['label'], 'n': s['n'], 'spearman_rho': s['rho'], 'p_value': s['p_rho_str']}
                  for s in rows]).to_csv(tbl("co_mega_te_correlation_stats_spearman"), index=False)

    all_genes_with_te = audit_join(
        codon_df[['tx_id', 'ensembl_gene_id', 'gene_name', 'chromosome', 'is_X', 'is_Y', 'CO_Mega']], te_df,
        'ensembl_gene_id', 'ensembl_gene_id',
        'human UTR codon frequencies (all genes, for output table)', 'human TE (Ensembl gene ID)',
        tbl("join_audit_codon_vs_te_output_table"), how='left',
    )
    out_all = tbl("co_mega_all_genes")
    all_genes_with_te.to_csv(out_all, index=False)
    print(f"  Saved CO_Mega for all {len(codon_df):,} genes to {out_all}")

    # Plot boxplot for CO_Mega of X vs. autosomes
    plot_co_mega_boxplot(codon_df, 'CO_Mega', 'is_X', 'X chromosome',
                          f'CO_Mega: X-linked vs Autosomal genes ({TISSUE_LABEL})',
                          fig("co_mega_boxplot_autosome_vs_X"))

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
