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
from matplotlib.lines import Line2D
from scipy import stats
import mpmath

from co_mega import SENSE_CODONS, compute_co_mega

mpmath.mp.dps = 60

BASE_DIR  = "/lab/solexa_page/xueqi/analysis_codon"
CDS_FILE  = os.path.join(BASE_DIR, "data/human_CDS_sequence.csv")
TE_FILE   = "/lab/solexa_page/xueqi/analysis_riboseq/tables/human_liver/all_human_genes_merged_data.csv"
XCI_FILE_GYLEMO = "/lab/solexa_page/xueqi/general/genes_XCI_Gylemo/tables/chrX_genes_classification_comparison_gylemo_liver_thresh0p05.csv"
XCI_FILE_NEHA   = "/lab/solexa_page/xueqi/general/genes_XCI_Neha/tables/chrX_genes_classification_Neha.csv"
TABLE_DIR = os.path.join(BASE_DIR, "tables")
FIG_DIR   = os.path.join(BASE_DIR, "figures")

AUTOSOMES = [str(i) for i in range(1, 23)]


def precise_pvalue_str(r, n, sig_figs=4):
    """
    Two-sided p-value for a Pearson correlation r (n observations), computed
    with arbitrary-precision arithmetic (mpmath) so it doesn't underflow to
    0.0 in float64 like scipy.stats.pearsonr's p-value does for very
    significant correlations. Returns a string in scientific notation, e.g.
    '2.123e-69'.
    """
    df = n - 2
    t_stat = mpmath.mpf(r) * mpmath.sqrt(mpmath.mpf(df) / (1 - mpmath.mpf(r) ** 2))
    x = mpmath.mpf(df) / (mpmath.mpf(df) + t_stat ** 2)
    sf = mpmath.mpf('0.5') * mpmath.betainc(mpmath.mpf(df) / 2, mpmath.mpf('0.5'), 0, x, regularized=True)
    p = 2 * sf
    return mpmath.nstr(p, sig_figs, min_fixed=0, max_fixed=0)


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
def plot_actual_vs_predicted(actual, predicted, xlabel, ylabel, title, r, p_str, output_file):
    """
    Scatter of some 'actual' quantity vs. a model-derived 'predicted' one
    (e.g. actual TE_mean vs. fitted TE_mean, or actual TE_mean vs. CO_Mega),
    with a y = x reference line, equal axis scales, and R / R^2 / p / n
    annotated.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    r_squared = r ** 2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, s=8, alpha=0.3, color='#377EB8', edgecolor='none')
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(lims, lims, color='gray', linestyle='--', linewidth=1, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.annotate(f"R = {r:.4f}\nR\u00b2 = {r_squared:.4f}\np = {p_str}\nn = {len(actual):,}",
                xy=(0.05, 0.95), xycoords='axes fraction', va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}  (R = {r:.4f}, R^2 = {r_squared:.4f}, p = {p_str}, n = {len(actual):,})")


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


GROUP_COLORS = ['#377EB8', '#E41A1C', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33']


def plot_co_mega_by_group(df, value_col, group_col, group_order, title, output_file, out_csv=None):
    """
    Boxplot of `value_col` across the groups in `group_order` that are
    actually present in `df[group_col]` (missing ones are skipped). Prints
    (and shows, in a plot legend) the Mann-Whitney U p-value for every
    pairwise comparison of groups. If `out_csv` is given, saves a table with
    n / mean per group and the p-value for every pairwise comparison.
    """
    groups = [g for g in group_order if g in set(df[group_col])]
    data = [df.loc[df[group_col] == g, value_col].dropna() for g in groups]
    labels = [f'{g}\n(n={len(d):,})\n(mean={d.mean():.3f})' for g, d in zip(groups, data)]

    pairwise = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            pval = stats.mannwhitneyu(data[i], data[j], alternative='two-sided').pvalue
            pairwise.append((groups[i], groups[j], pval))

    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 3.5, 6))
    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, widths=0.5, showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.3, markeredgecolor='none')
    )
    for patch, color in zip(bp['boxes'], GROUP_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('CO_Mega')
    ax.set_title(title, fontweight='bold')

    legend_text = "Mann-Whitney U p-values:\n" + "\n".join(
        f"{g1} vs {g2}: p = {pval:.3g}" for g1, g2, pval in pairwise
    )
    legend_handle = Line2D([], [], color='none', label=legend_text)
    ax.legend(handles=[legend_handle], loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=7, handlelength=0, handletextpad=0, frameon=True, borderaxespad=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")

    print(f"  Pairwise Mann-Whitney U p-values ({title}):")
    stats_rows = []
    for g1, g2, pval in pairwise:
        i, j = groups.index(g1), groups.index(g2)
        n1, n2 = len(data[i]), len(data[j])
        print(f"    {g1} (n={n1:,}) vs {g2} (n={n2:,}): p = {pval:.3g}")
        stats_rows.append({
            'group1': g1, 'n1': n1, 'mean1': data[i].mean(),
            'group2': g2, 'n2': n2, 'mean2': data[j].mean(),
            'p_value': pval,
        })

    if out_csv is not None:
        pd.DataFrame(stats_rows).to_csv(out_csv, index=False)
        print(f"  Saved pairwise stats to {out_csv}")


def build_xci_groups(codon_df, xci_file, exclude_values=()):
    """
    Merge X-linked genes in `codon_df` with a gene_name -> classification
    table (`xci_file`, columns 'gene_name' and 'classification'), dropping
    any classification in `exclude_values` (e.g. 'No call'). X-linked genes
    not found in the table are excluded. Autosomal genes are labeled
    'Autosome'. Returns (groups_df, x_genes), where `groups_df` has columns
    ['CO_Mega', 'classification'] ready for `plot_co_mega_by_group`, and
    `x_genes` is the merged (post-exclusion) X-linked-only frame.
    """
    xci_df = pd.read_csv(xci_file, usecols=['gene_name', 'classification'])
    x_genes = codon_df.loc[codon_df['is_X'] == 1, ['gene_name', 'CO_Mega']].merge(
        xci_df, on='gene_name', how='inner'
    )
    if exclude_values:
        x_genes = x_genes[~x_genes['classification'].isin(exclude_values)]

    auto_genes = codon_df.loc[codon_df['is_X'] == 0, ['CO_Mega']].copy()
    auto_genes['classification'] = 'Autosome'
    groups_df = pd.concat(
        [auto_genes[['CO_Mega', 'classification']], x_genes[['CO_Mega', 'classification']]],
        ignore_index=True
    )
    return groups_df, x_genes


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
    Output files are all suffixed with `_{source_label}`.
    """
    groups_df, x_genes = build_xci_groups(codon_df, xci_file, exclude_values=exclude_values)
    n_x_total = (codon_df['is_X'] == 1).sum()
    print(f"  {len(x_genes):,} of {n_x_total:,} X-linked genes have a defined XCI status in the "
          f"{source_label} table ({n_x_total - len(x_genes):,} excluded from these boxplots).")

    defined_xci_df = groups_df.copy()
    defined_xci_df['group'] = np.where(defined_xci_df['classification'] == 'Autosome', 'Autosome', 'X (defined XCI)')
    plot_co_mega_by_group(
        defined_xci_df, 'CO_Mega', 'group', ['Autosome', 'X (defined XCI)'],
        f'CO_Mega: Autosome vs X (defined XCI status, {source_label})',
        os.path.join(FIG_DIR, f"co_mega_boxplot_autosome_vs_X_with_defined_XCI_{source_label}.png"),
        out_csv=os.path.join(TABLE_DIR, f"co_mega_autosome_vs_X_with_defined_XCI_stats_{source_label}.csv")
    )

    plot_co_mega_by_group(
        groups_df, 'CO_Mega', 'classification',
        ['Autosome', 'Xi-silent', 'Xi-expressed', 'NPX-NPY', 'PAR1', 'PAR2'],
        f'CO_Mega by XCI category ({source_label})',
        os.path.join(FIG_DIR, f"co_mega_boxplot_XCI_categories_full_{source_label}.png"),
        out_csv=os.path.join(TABLE_DIR, f"co_mega_XCI_categories_full_stats_{source_label}.csv")
    )
    plot_co_mega_by_group(
        groups_df, 'CO_Mega', 'classification',
        ['Autosome', 'Xi-silent', 'Xi-expressed'],
        f'CO_Mega: Autosome vs Xi-silent vs Xi-expressed ({source_label})',
        os.path.join(FIG_DIR, f"co_mega_boxplot_XCI_categories_reduced_{source_label}.png"),
        out_csv=os.path.join(TABLE_DIR, f"co_mega_XCI_categories_reduced_stats_{source_label}.csv")
    )


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

    r_train, _ = stats.pearsonr(train['TE_mean'], model.fittedvalues)
    p_train_str = precise_pvalue_str(r_train, len(train))
    plot_actual_vs_predicted(
        train['TE_mean'], model.fittedvalues,
        'Actual TE_mean (autosomal training genes)', 'Fitted TE_mean (OLS model)',
        'Original regression: TE ~ codon frequencies\n(autosomal training genes)',
        r_train, p_train_str, os.path.join(FIG_DIR, "co_mega_regression_fit_train.png")
    )

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
    r_x, p_x = stats.pearsonr(x_te['CO_Mega'], x_te['TE_mean']) if len(x_te) > 1 else (np.nan, np.nan)
    if len(x_te) > 1:
        print(f"  X-linked genes only (n={len(x_te):,}, held out): Pearson r = {r_x:.4f} (p = {p_x:.3g})")

    p_all_str = precise_pvalue_str(r_all, len(with_te))
    p_auto_str = precise_pvalue_str(r_auto, len(auto_te))
    p_x_str = precise_pvalue_str(r_x, len(x_te)) if len(x_te) > 1 else np.nan

    corr_stats = pd.DataFrame([
        {'gene_set': 'All genes', 'n': len(with_te), 'pearson_r': r_all, 'p_value': p_all_str},
        {'gene_set': 'Autosomal genes', 'n': len(auto_te), 'pearson_r': r_auto, 'p_value': p_auto_str},
        {'gene_set': 'X-linked genes', 'n': len(x_te), 'pearson_r': r_x, 'p_value': p_x_str},
    ])
    out_corr = os.path.join(TABLE_DIR, "co_mega_te_correlation_stats.csv")
    corr_stats.to_csv(out_corr, index=False)
    print(f"  Saved CO_Mega vs. TE correlation stats (n, Pearson r, p) to {out_corr}")

    plot_actual_vs_predicted(with_te['TE_mean'], with_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                              'Actual TE vs. CO_Mega: All genes', r_all, p_all_str,
                              os.path.join(FIG_DIR, "co_mega_vs_te_all_genes.png"))
    plot_actual_vs_predicted(auto_te['TE_mean'], auto_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                              'Actual TE vs. CO_Mega: Autosomal genes', r_auto, p_auto_str,
                              os.path.join(FIG_DIR, "co_mega_vs_te_autosomal.png"))
    if len(x_te) > 1:
        plot_actual_vs_predicted(x_te['TE_mean'], x_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                                  'Actual TE vs. CO_Mega: X-linked genes', r_x, p_x_str,
                                  os.path.join(FIG_DIR, "co_mega_vs_te_X_linked.png"))

    out_all = os.path.join(TABLE_DIR, "co_mega_all_genes.csv")
    codon_df[['gene_id', 'gene_name', 'chromosome', 'is_X', 'CO_Mega']].merge(
        te_df, on='gene_id', how='left'
    ).to_csv(out_all, index=False)
    print(f"  Saved CO_Mega for all {len(codon_df):,} genes to {out_all}")

    print("\n[5] Boxplot: CO_Mega, autosomal vs X-linked genes...")
    plot_co_mega_boxplot(codon_df, 'CO_Mega', os.path.join(FIG_DIR, "co_mega_boxplot_autosome_vs_X.png"))

    print("\n[6] Boxplots: CO_Mega by XCI category (Gylemo classification)...")
    plot_xci_boxplots(codon_df, XCI_FILE_GYLEMO, 'Gylemo')

    print("\n[7] Boxplots: CO_Mega by XCI category (Neha classification)...")
    plot_xci_boxplots(codon_df, XCI_FILE_NEHA, 'Neha', exclude_values=('No call',))

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
