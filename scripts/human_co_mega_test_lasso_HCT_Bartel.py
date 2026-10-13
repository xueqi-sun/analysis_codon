#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Bartel-TE version of human_co_mega_test_lasso.py: the same lasso pipeline, but the human TE is the
Bartel HCT TE (log2 of `Sidrauski_TE_unstr`, from
analysis_riboseq/scripts/all_genes_translational_efficiency_human_HCT_Bartel.py) instead of the human
liver ribo-seq/RNA-seq TE. Outputs: tables/human_HCT_Bartel_lasso/ and figures/human_HCT_Bartel_lasso/,
suffix `_human_HCT_Bartel_lasso`. (The OLS counterpart is human_co_mega_analysis_HCT_Bartel.py; the two are
compared by compare_ols_vs_lasso_HCT_Bartel.py.)

Original description (liver lasso):
Lasso variant of human_co_mega_analysis.py: identical pipeline, except the
codon-composition -> TE regression is fit with LASSO (L1-penalised least
squares, penalty chosen by 10-fold cross-validation) instead of OLS. All
outputs go to tables/human_HCT_Bartel_lasso/ and figures/human_HCT_Bartel_lasso/ (suffix
`_human_HCT_Bartel_lasso`).

Original description:
Build our own human CO_Mega model (codon composition -> TE) and use it to
compare CO_Mega between autosomal, X-linked, and Y-linked genes.

Pipeline:
  1. Load CDS sequences for all human protein-coding genes (autosomes + X +
     Y) from their MANE Select transcripts, extracted directly from the MANE
     Ensembl GTF (MANE.GRCh38.v1.5.ensembl_genomic.gtf; CDS + stop_codon
     records of `tag "MANE_Select"` transcripts) by
     general/sequence_CDS/human/scripts/extract_human_cds_mane_ensembl.py
     (general/sequence_CDS/human/tables/
     human_CDS_sequences_MANE_Select_ensembl.csv). Ensembl gene IDs are native
     to that GTF: `gene_id` is the versioned Ensembl gene ID and
     `ensembl_gene_id` the same ID without the version suffix; the latter is
     what every join below uses. No RefSeq -> Ensembl conversion is involved.
  2. For each gene, split the CDS into codons, drop the last codon (the stop
     codon), and compute each of the 61 sense codons' frequency as
     (# that codon) / (# total codons remaining).
  3. Load per-gene TE_mean (human HCT, Bartel TE) from
     analysis_riboseq/tables/human_HCT_Bartel/all_human_genes_merged_data.csv,
     merging on `ensembl_gene_id` <-> `Gene_ID` (gene-ID-based like every other
     join in the pipeline; still audited at every merge via `audit_join` /
     `check_key_uniqueness` below).
  4. Fit  TE_mean ~ 61 codon frequencies  (LASSO) using only autosomal
     (chr 1-22) genes. Report R^2 and save the intercept + 61 coefficients.
  5. Apply the fitted model to ALL genes (autosomes + X + Y) to get each
     gene's CO_Mega, report its correlation with TE_mean, and compare
     CO_Mega between autosomal, X-linked, and Y-linked genes with boxplots
     + Mann-Whitney U.

All output tables/figures are suffixed with `_human_HCT_Bartel_lasso`, and saved under
dedicated `tables/human_lasso/` and `figures/human_lasso/` subfolders.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from co_mega import SENSE_CODONS, compute_co_mega
from co_mega_common import (
    precise_pvalue_str, chromosome_groups, compute_codon_frequencies,
    plot_actual_vs_predicted, plot_actual_vs_predicted_spearman,
    plot_co_mega_boxplot, bootstrap_codon_pearson_r, bootstrap_codon_spearman_rho,
    plot_codon_pearson_r_barplot, plot_codon_spearman_rho_barplot,
    plot_co_mega_by_group, build_xci_groups,
    make_autosome_x_y_group,
    audit_join, check_key_uniqueness,
    _normality_stats, _plot_histogram_with_normality_stats, _plot_residual_qq,
    _plot_residual_vs_fitted,
)

BASE_DIR  = "/lab/solexa_page/xueqi/analysis_codon"
# Canonical human CDS table: MANE Select CDS taken straight from the MANE
# Ensembl GTF (extract_human_cds_mane_ensembl.py); native Ensembl gene IDs.
CDS_FILE  = "/lab/solexa_page/xueqi/general/sequence_CDS/human/tables/human_CDS_sequences_MANE_Select_ensembl.csv"
TE_FILE   = "/lab/solexa_page/xueqi/analysis_riboseq/tables/human_HCT_Bartel/all_human_genes_merged_data.csv"
# Y-linked genes are NOT in TE_FILE (0 rows there); computed separately
# (samples 1 & 2 only -- sample 3 is female, 0 reads on Y genes) by
# analysis_riboseq/scripts/all_genes_translational_efficiency_human_HCT_Bartel.py.
TE_FILE_Y = "/lab/solexa_page/xueqi/analysis_riboseq/tables/human_HCT_Bartel/chromosome_te_analysis_Y.csv"
XCI_FILE_GYLEMO = "/lab/solexa_page/xueqi/general/genes_XCI_Gylemo/tables/chrX_genes_classification_comparison_gylemo_liver_thresh0p05.csv"
XCI_FILE_NEHA   = "/lab/solexa_page/xueqi/general/genes_XCI_Neha/tables/chrX_genes_classification_Neha.csv"
TABLE_DIR = os.path.join(BASE_DIR, "tables", "human_HCT_Bartel_lasso")
FIG_DIR   = os.path.join(BASE_DIR, "figures", "human_HCT_Bartel_lasso")

AUTOSOMES = [str(i) for i in range(1, 23)]

SUFFIX = "human_HCT_Bartel_lasso"
TISSUE_LABEL = "Human HCT (Bartel), lasso"


def tbl(name):
    return os.path.join(TABLE_DIR, f"{name}_{SUFFIX}.csv")


def fig(name):
    return os.path.join(FIG_DIR, f"{name}_{SUFFIX}.png")


# ── Lasso fit ───────────────────────────────────────────────────────────────
class LassoFit:
    """Minimal stand-in for a statsmodels result: fitted values, residuals, R^2."""
    def __init__(self, y, fitted, n_params, alpha):
        self.y = np.asarray(y, dtype=float)
        self.fittedvalues = np.asarray(fitted, dtype=float)
        self.resid = self.y - self.fittedvalues
        self.nobs = len(self.y)
        self.alpha = alpha
        ss_res = float(np.sum(self.resid ** 2))
        ss_tot = float(np.sum((self.y - self.y.mean()) ** 2))
        self.rsquared = 1 - ss_res / ss_tot
        # adjusted R^2 uses the number of NON-ZERO coefficients as df
        self.rsquared_adj = 1 - (1 - self.rsquared) * (self.nobs - 1) / (self.nobs - n_params - 1)


def fit_co_mega_te_model_lasso(codon_freq_df, te, codons=SENSE_CODONS, cv=10, seed=0):
    """
    Fit TE ~ codon frequencies with LASSO. Predictors are standardised, the
    penalty (alpha) is chosen by `cv`-fold cross-validation (LassoCV, minimum
    CV MSE), and coefficients are converted back to the raw codon-frequency
    scale. Returns (coef, model) where `coef` is a Series indexed by
    'intercept' + `codons` (ready for `compute_co_mega`).
    """
    X = codon_freq_df[list(codons)].to_numpy(dtype=float)
    y = np.asarray(te, dtype=float)
    scaler = StandardScaler().fit(X)
    lasso = LassoCV(cv=cv, random_state=seed, n_jobs=1, max_iter=100000).fit(scaler.transform(X), y)
    beta = lasso.coef_ / scaler.scale_
    intercept = lasso.intercept_ - float(np.sum(beta * scaler.mean_))
    coef = pd.Series([intercept] + list(beta), index=['intercept'] + list(codons))
    fitted = intercept + X @ beta
    model = LassoFit(y, fitted, int(np.sum(beta != 0)), lasso.alpha_)
    model.n_nonzero = int(np.sum(beta != 0))
    model.exog = np.column_stack([np.ones(len(X)), X])
    return coef, model


def save_lasso_coefficient_table(coef, model, codons, out_file):
    names = ['intercept'] + list(codons)
    df = pd.DataFrame({'term': names, 'coefficient': [coef[n] for n in names],
                       'selected': [bool(coef[n] != 0) for n in names],
                       'lasso_alpha': model.alpha})
    df.to_csv(out_file, index=False)
    print(f"  Saved intercept + {len(codons)} lasso codon coefficients to {out_file}")
    return df


def run_lasso_diagnostics(model, y, title_suffix, hist_te_file, hist_resid_file, qq_file,
                          resid_fitted_file, stats_csv):
    """
    Same diagnostics/plots as `run_regression_diagnostics`, computed from the
    lasso residuals/fitted values: normality of TE and residuals,
    Breusch-Pagan (squared residuals on the design matrix), and a Ramsey-
    RESET-style linearity F-test of adding fitted^2 and fitted^3 to
    y ~ fitted (the lasso-penalised model itself cannot be refit by RESET).
    """
    y = np.asarray(y, dtype=float)
    resid, fitted = model.resid, model.fittedvalues
    y_stats = _normality_stats(y, 'TE')
    resid_stats = _normality_stats(resid, 'residuals')
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(resid, model.exog)
    base = sm.OLS(y, sm.add_constant(fitted)).fit()
    aug = sm.OLS(y, sm.add_constant(np.column_stack([fitted, fitted ** 2, fitted ** 3]))).fit()
    reset_stat, reset_pvalue, _ = aug.compare_f_test(base)
    all_stats = {**y_stats, **resid_stats,
                 'breusch_pagan_lm_stat': bp_stat, 'breusch_pagan_pvalue': bp_pvalue,
                 'ramsey_reset_fstat': float(reset_stat), 'ramsey_reset_pvalue': float(reset_pvalue)}
    print(f"  --- Lasso regression diagnostics ({title_suffix}) ---")
    print(f"  Normality of residuals: skew = {resid_stats['residuals_skew']:.3f}, "
          f"excess kurtosis = {resid_stats['residuals_excess_kurtosis']:.3f}, "
          f"D'Agostino K^2 p = {resid_stats['residuals_dagostino_k2_pvalue']:.3g}")
    print(f"  Homoscedasticity (Breusch-Pagan): LM stat = {bp_stat:.3f}, p = {bp_pvalue:.3g}")
    print(f"  Linearity (RESET-style on fitted values): F = {reset_stat:.3f}, p = {reset_pvalue:.3g}")
    _plot_histogram_with_normality_stats(
        y, 'Actual TE_mean (log2 ribo/RNA)', f'Distribution of TE_mean used for regression\n{title_suffix}',
        hist_te_file, y_stats, 'TE')
    _plot_histogram_with_normality_stats(
        resid, 'Residual (actual \u2212 fitted TE_mean)', f'Distribution of regression residuals\n{title_suffix}',
        hist_resid_file, resid_stats, 'residuals')
    _plot_residual_qq(resid, f'Q-Q plot of regression residuals\n{title_suffix}', qq_file)
    _plot_residual_vs_fitted(fitted, resid, f'Residuals vs. fitted values\n{title_suffix}',
                             resid_fitted_file, bp_pvalue, float(reset_pvalue))
    pd.DataFrame([all_stats]).to_csv(stats_csv, index=False)
    print(f"  Saved diagnostic statistics to {stats_csv}")
    return all_stats


# ── Codon frequencies ───────────────────────────────────────────────────────
def load_codon_frequencies(cds_file):
    df = pd.read_csv(cds_file, usecols=['gene_id', 'ensembl_gene_id', 'gene_name', 'chromosome', 'CDS_sequence'])
    print(f"  Loaded {len(df):,} genes from {cds_file}")
    n_with_ensembl_id = df['ensembl_gene_id'].notna().sum()
    print(f"  {n_with_ensembl_id:,} of {len(df):,} genes have an Ensembl gene ID (native to the MANE GTF).")

    freq_records = df['CDS_sequence'].apply(compute_codon_frequencies)
    n_unusable = freq_records.isna().sum()
    df = df.loc[freq_records.notna()].copy()
    freq_df = pd.DataFrame(list(freq_records.dropna()), index=df.index)
    print(f"  Dropped {n_unusable:,} genes with missing/unusable CDS sequences "
          f"(not a multiple of 3, or too short); {len(df):,} genes remain.")

    out = pd.concat([df[['gene_id', 'ensembl_gene_id', 'gene_name', 'chromosome']], freq_df], axis=1)
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
    Output files are all suffixed with `_{source_label}_human_HCT_Bartel`.
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

    print("=== Human CO_Mega model (human HCT, Bartel TE) ===")

    print("\n[1] Computing codon frequencies from CDS sequences...")
    codon_df = load_codon_frequencies(CDS_FILE)
    out_codon = tbl("codon_frequencies")
    codon_df.to_csv(out_codon, index=False)
    print(f"  Saved codon frequencies to {out_codon}")

    print("\n[2] Loading human HCT (Bartel) TE data...")
    # ID-based join on `ensembl_gene_id` <-> `Gene_ID` (both real Ensembl gene
    # IDs). This used to be the pipeline's one deliberate name-based-join
    # exception -- see the module docstring step 3 -- but now that the CDS
    # table carries a real Ensembl gene ID (via convert_refseq_to_ensembl.py
    # / mygene.info), there is no exception left. Still audited at every
    # merge (check_key_uniqueness() below and audit_join() at every merge)
    # as a matter of course, same as every other join in this pipeline.
    # (te_df carries only `ensembl_gene_id`/`TE_mean` -- not `gene_name` --
    # since codon_df already has the authoritative gene_name for every merge
    # below; this avoids a redundant/duplicated gene_name_right column.)
    te_raw = pd.read_csv(TE_FILE).rename(columns={'Gene_ID': 'ensembl_gene_id'})
    check_key_uniqueness(te_raw, 'ensembl_gene_id', 'all_human_genes_merged_data.csv (TE_FILE)')
    te_df = te_raw[['ensembl_gene_id', 'TE_mean']]
    te_df_y = pd.read_csv(TE_FILE_Y).rename(columns={'Gene_ID': 'ensembl_gene_id'})[['ensembl_gene_id', 'TE_mean']]
    te_df = pd.concat([te_df, te_df_y], ignore_index=True)
    te_df = te_df.dropna(subset=['TE_mean', 'ensembl_gene_id'])
    print(f"  {len(te_df):,} genes with TE_mean ({len(te_df_y):,} Y-linked, from {TE_FILE_Y}).")

    merged = audit_join(
        codon_df, te_df, 'ensembl_gene_id', 'ensembl_gene_id',
        'human codon frequencies (CDS, Ensembl gene ID)', 'human TE (Ensembl gene ID)',
        tbl("join_audit_codon_vs_te_train"),
    )
    print(f"  {len(merged):,} genes with both codon frequencies and TE_mean.")

    print("\n[3] Fitting TE_mean ~ codon frequencies (LASSO) on autosomal genes...")
    train = merged[merged['chromosome'].isin(AUTOSOMES)]
    print(f"  Training on {len(train):,} autosomal genes, {len(SENSE_CODONS)} codon predictors.")
    coef, model = fit_co_mega_te_model_lasso(train[SENSE_CODONS], train['TE_mean'])
    print(f"  R^2 = {model.rsquared:.4f}   Adjusted R^2 = {model.rsquared_adj:.4f}   n = {int(model.nobs)}")
    print(f"  Lasso alpha (10-fold CV) = {model.alpha:.3g}; {model.n_nonzero} of {len(SENSE_CODONS)} codons have non-zero coefficients.")

    out_coef = tbl("co_mega_model_coefficients")
    save_lasso_coefficient_table(coef, model, SENSE_CODONS, out_coef)

    r_train, _ = stats.pearsonr(train['TE_mean'], model.fittedvalues)
    rho_train, _ = stats.spearmanr(train['TE_mean'], model.fittedvalues)
    p_train_str = precise_pvalue_str(r_train, len(train))
    p_train_rho_str = precise_pvalue_str(rho_train, len(train))
    plot_actual_vs_predicted(
        train['TE_mean'], model.fittedvalues,
        'Actual TE_mean (autosomal training genes)', 'Fitted TE_mean (lasso model)',
        f'Lasso regression: TE ~ codon frequencies\n(autosomal training genes, {TISSUE_LABEL})',
        r_train, p_train_str, fig("co_mega_regression_fit_train")
    )
    plot_actual_vs_predicted_spearman(
        train['TE_mean'], model.fittedvalues,
        'Actual TE_mean (autosomal training genes)', 'Fitted TE_mean (lasso model)',
        f'Lasso regression: TE ~ codon frequencies\n(autosomal training genes, {TISSUE_LABEL})',
        rho_train, p_train_rho_str, fig("co_mega_regression_fit_train_spearman")
    )

    print("\n[3b] Regression diagnostics (linearity, homoscedasticity, normality)...")
    run_lasso_diagnostics(
        model, train['TE_mean'], f'autosomal training genes, {TISSUE_LABEL}',
        fig("co_mega_regression_diagnostics_te_histogram"),
        fig("co_mega_regression_diagnostics_residual_histogram"),
        fig("co_mega_regression_diagnostics_residual_qq"),
        fig("co_mega_regression_diagnostics_residual_vs_fitted"),
        tbl("co_mega_regression_diagnostics_stats"),
    )

    print("\n[4] Computing CO_Mega for all genes (autosomes + X + Y) and comparing to TE...")
    codon_df['CO_Mega'] = compute_co_mega(codon_df[SENSE_CODONS], coef)

    # ID-based join again (see [2] above), all genes.
    with_te = audit_join(
        codon_df, te_df, 'ensembl_gene_id', 'ensembl_gene_id',
        'human codon frequencies (all genes, Ensembl gene ID)', 'human TE (Ensembl gene ID)',
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
                              fig("co_mega_vs_te_all_genes"), groups=chromosome_groups(with_te))
    plot_actual_vs_predicted_spearman(with_te['TE_mean'], with_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                                       f'Actual TE vs. CO_Mega: All genes ({TISSUE_LABEL})', rho_all, p_rho_all_str,
                                       fig("co_mega_vs_te_all_genes_spearman"), groups=chromosome_groups(with_te))
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
    # NOTE: this output table's `gene_id` column is still the RefSeq gene
    # SYMBOL (kept as informational metadata only -- see the docstring step 1
    # note) -- any script that reads co_mega_all_genes_human_HCT_Bartel.csv and
    # needs to join it to another Ensembl-ID-keyed table (e.g.
    # human_chicken_comparison.py) should join on `ensembl_gene_id`, not
    # `gene_id` or `gene_name`.
    all_genes_with_te = audit_join(
        codon_df[['gene_id', 'ensembl_gene_id', 'gene_name', 'chromosome', 'is_X', 'is_Y', 'CO_Mega']], te_df,
        'ensembl_gene_id', 'ensembl_gene_id',
        'human codon frequencies (all genes, for output table)', 'human TE (Ensembl gene ID)',
        tbl("join_audit_codon_vs_te_output_table"), how='left',
    )
    all_genes_with_te.to_csv(out_all, index=False)
    print(f"  Saved CO_Mega for all {len(codon_df):,} genes to {out_all}")
    check_key_uniqueness(
        all_genes_with_te, 'ensembl_gene_id',
        'co_mega_all_genes_human_HCT_Bartel.csv (output; downstream scripts should join this on ensembl_gene_id)'
    )

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
