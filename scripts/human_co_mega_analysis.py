#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Build our own human CO_Mega model (codon composition -> TE) and use it to
compare CO_Mega between autosomal and X-linked genes.

Pipeline:
  1. Load CDS sequences for all human protein-coding genes (autosomes + X)
     from analysis_codon/data/human_CDS_sequence.csv.
  2. For each gene, split the CDS into codons, drop the last codon (the stop
     codon), and compute each of the 61 sense codons' frequency as
     (# that codon) / (# total codons remaining).
  3. Load per-gene TE_mean (human liver) from
     analysis_riboseq/tables/human_liver/all_human_genes_merged_data.csv.
  4. Fit  TE_mean ~ 61 codon frequencies  (OLS) using only autosomal
     (chr 1-22) genes. Report R^2 and save the intercept + 61 coefficients.
  5. Apply the fitted model to ALL genes (autosomes + X) to get each gene's
     CO_Mega, report its correlation with TE_mean, and compare CO_Mega
     between autosomal and X-linked genes with a boxplot + Mann-Whitney U.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from co_mega import SENSE_CODONS, compute_co_mega

BASE_DIR  = "/lab/solexa_page/xueqi/analysis_codon"
CDS_FILE  = os.path.join(BASE_DIR, "data/human_CDS_sequence.csv")
TE_FILE   = "/lab/solexa_page/xueqi/analysis_riboseq/tables/human_liver/all_human_genes_merged_data.csv"
TABLE_DIR = os.path.join(BASE_DIR, "tables")
FIG_DIR   = os.path.join(BASE_DIR, "figures")

AUTOSOMES = [str(i) for i in range(1, 23)]


# ── Codon frequencies ───────────────────────────────────────────────────────
def compute_codon_frequencies(cds_seq, codons=SENSE_CODONS):
    """
    Split a CDS sequence into non-overlapping 3-base codons, drop the last
    codon (the stop codon), and return each of `codons`' frequency as
    (# that codon) / (# codons remaining after dropping the stop codon).
    Returns None if the sequence isn't usable (missing, not a multiple of 3,
    or too short to have any non-stop codons).
    """
    if not isinstance(cds_seq, str) or len(cds_seq) % 3 != 0:
        return None
    n_codons_total = len(cds_seq) // 3
    if n_codons_total < 2:
        return None
    all_codons = [cds_seq[i:i + 3] for i in range(0, len(cds_seq), 3)]
    coding_codons = all_codons[:-1]  # drop the last (stop) codon
    total = len(coding_codons)
    counts = Counter(coding_codons)
    return {c: counts.get(c, 0) / total for c in codons}


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
    n_auto = (out['chromosome'].isin(AUTOSOMES)).sum()
    n_x = (out['chromosome'] == 'X').sum()
    print(f"  {n_auto:,} autosomal (chr 1-22) genes, {n_x:,} X-linked genes.")
    return out


# ── Regression ────────────────────────────────────────────────────────────
def fit_co_mega_te_model(codon_freq_df, te, codons=SENSE_CODONS):
    """
    Fit TE ~ codon frequencies (OLS) and return (coef, model), where `coef`
    is a Series indexed by 'intercept' + `codons`, ready for
    `co_mega.compute_co_mega`.
    """
    X = sm.add_constant(codon_freq_df[list(codons)])
    model = sm.OLS(np.asarray(te, dtype=float), X).fit()
    coef = model.params.rename({'const': 'intercept'})
    return coef, model


def save_coefficient_table(coef, model, codons, out_file):
    names = ['intercept'] + list(codons)
    rows = []
    for name in names:
        rows.append({
            'term': name,
            'coefficient': coef[name],
            'std_err': model.bse[name if name != 'intercept' else 'const'],
            't_value': model.tvalues[name if name != 'intercept' else 'const'],
            'p_value': model.pvalues[name if name != 'intercept' else 'const'],
        })
    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(out_file, index=False)
    print(f"  Saved intercept + {len(codons)} codon coefficients to {out_file}")
    return coef_df


# ── Plot ─────────────────────────────────────────────────────────────────
def plot_co_mega_boxplot(df, value_col, output_file):
    autosome = df.loc[df['is_X'] == 0, value_col].dropna()
    x_chrom = df.loc[df['is_X'] == 1, value_col].dropna()
    pval = stats.mannwhitneyu(autosome, x_chrom, alternative='two-sided').pvalue

    fig, ax = plt.subplots(figsize=(5, 6))
    bp = ax.boxplot(
        [autosome, x_chrom],
        tick_labels=[f'Autosome\n(n={len(autosome):,})', f'X chromosome\n(n={len(x_chrom):,})'],
        patch_artist=True, widths=0.5, showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.3, markeredgecolor='none')
    )
    for patch, color in zip(bp['boxes'], ['#377EB8', '#E41A1C']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('CO_Mega')
    ax.set_title('CO_Mega: X-linked vs Autosomal genes', fontweight='bold')
    ax.annotate(f"Mann-Whitney U\np = {pval:.3g}",
                xy=(0.5, 0.98), xycoords='axes fraction', va='top', ha='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}  (Mann-Whitney U p = {pval:.3g}, "
          f"median autosome = {autosome.median():.3f}, median X = {x_chrom.median():.3f})")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=== Human CO_Mega model ===")

    print("\n[1] Computing codon frequencies from CDS sequences...")
    codon_df = load_codon_frequencies(CDS_FILE)
    out_codon = os.path.join(TABLE_DIR, "human_codon_frequencies.csv")
    codon_df.to_csv(out_codon, index=False)
    print(f"  Saved codon frequencies to {out_codon}")

    print("\n[2] Loading human liver TE data...")
    te_df = pd.read_csv(TE_FILE)[['Gene_ID', 'TE_mean']].rename(columns={'Gene_ID': 'gene_id'})
    te_df = te_df.dropna(subset=['TE_mean'])
    print(f"  {len(te_df):,} genes with TE_mean.")

    merged = codon_df.merge(te_df, on='gene_id', how='inner')
    print(f"  {len(merged):,} genes with both codon frequencies and TE_mean.")

    print("\n[3] Fitting TE_mean ~ codon frequencies on autosomal genes...")
    train = merged[merged['chromosome'].isin(AUTOSOMES)]
    print(f"  Training on {len(train):,} autosomal genes, {len(SENSE_CODONS)} codon predictors.")
    coef, model = fit_co_mega_te_model(train[SENSE_CODONS], train['TE_mean'])
    print(f"  R^2 = {model.rsquared:.4f}   Adjusted R^2 = {model.rsquared_adj:.4f}   n = {int(model.nobs)}")

    out_coef = os.path.join(TABLE_DIR, "co_mega_model_coefficients.csv")
    save_coefficient_table(coef, model, SENSE_CODONS, out_coef)

    print("\n[4] Computing CO_Mega for all genes (autosomes + X) and comparing to TE...")
    codon_df['CO_Mega'] = compute_co_mega(codon_df[SENSE_CODONS], coef)

    with_te = codon_df.merge(te_df, on='gene_id', how='inner')
    r_all, p_all = stats.pearsonr(with_te['CO_Mega'], with_te['TE_mean'])
    rho_all, p_rho_all = stats.spearmanr(with_te['CO_Mega'], with_te['TE_mean'])
    print(f"  All genes with TE (n={len(with_te):,}): "
          f"Pearson r = {r_all:.4f} (p = {p_all:.3g}), Spearman rho = {rho_all:.4f} (p = {p_rho_all:.3g})")

    auto_te = with_te[with_te['chromosome'].isin(AUTOSOMES)]
    r_auto, p_auto = stats.pearsonr(auto_te['CO_Mega'], auto_te['TE_mean'])
    print(f"  Autosomal genes only (n={len(auto_te):,}, training set): Pearson r = {r_auto:.4f} (p = {p_auto:.3g})")

    x_te = with_te[with_te['chromosome'] == 'X']
    if len(x_te) > 1:
        r_x, p_x = stats.pearsonr(x_te['CO_Mega'], x_te['TE_mean'])
        print(f"  X-linked genes only (n={len(x_te):,}, held out): Pearson r = {r_x:.4f} (p = {p_x:.3g})")

    out_all = os.path.join(TABLE_DIR, "co_mega_all_genes.csv")
    codon_df[['gene_id', 'gene_name', 'chromosome', 'is_X', 'CO_Mega']].merge(
        te_df, on='gene_id', how='left'
    ).to_csv(out_all, index=False)
    print(f"  Saved CO_Mega for all {len(codon_df):,} genes to {out_all}")

    print("\n[5] Boxplot: CO_Mega, autosomal vs X-linked genes...")
    plot_co_mega_boxplot(codon_df, 'CO_Mega', os.path.join(FIG_DIR, "co_mega_boxplot_autosome_vs_X.png"))

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
