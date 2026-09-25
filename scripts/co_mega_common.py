#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Shared helpers for the human/chicken CO_Mega analyses (human_co_mega_analysis.py,
chicken_co_mega_analysis.py): codon-frequency computation, OLS model
fitting/reporting, precise (non-underflowing) p-values, and the plotting
functions common to both species' pipelines.
"""

import textwrap
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import jarque_bera
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import mpmath

from co_mega import SENSE_CODONS

mpmath.mp.dps = 60


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


# ── FASTA parsing ────────────────────────────────────────────────────────────
def parse_cds_fasta(fasta_file, out_csv):
    """
    Parse a FASTA file whose headers look like:
      >gene_id|gene_id.version|transcript_id|transcript_id.version|gene_name|chromosome
    into a DataFrame with columns matching human_CDS_sequence.csv
    (gene_id, gene_id_version, transcript_id, transcript_id_version,
    chromosome, gene_name, CDS_sequence), and save it as `out_csv`.
    """
    records = []
    header_fields = None
    seq_chunks = []

    def flush():
        if header_fields is not None:
            gene_id, gene_id_version, transcript_id, transcript_id_version, gene_name, chromosome = header_fields
            records.append({
                'gene_id': gene_id, 'gene_id_version': gene_id_version,
                'transcript_id': transcript_id, 'transcript_id_version': transcript_id_version,
                'chromosome': chromosome, 'gene_name': gene_name,
                'CDS_sequence': ''.join(seq_chunks),
            })

    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                flush()
                header_fields = line[1:].split('|')
                seq_chunks = []
            else:
                seq_chunks.append(line)
    flush()

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"  Parsed {len(df):,} CDS records from {fasta_file}")
    print(f"  Saved to {out_csv}")
    return df


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


# ── Regression diagnostics (TE ~ codon frequencies OLS model) ──────────────
def _normality_stats(x, label):
    """
    Normality diagnostics for 1-D array `x` (NaNs dropped): skewness, excess
    (Fisher) kurtosis (0 for a Normal distribution), the D'Agostino-Pearson
    K^2 test (scipy `normaltest`; valid/robust for both small and large n,
    unlike Shapiro-Wilk which becomes numerically unreliable for n > ~5000),
    and the Jarque-Bera test. Returns a dict of stats prefixed by `label`.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    dagostino_stat, dagostino_p = stats.normaltest(x)
    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(x)
    return {
        f'{label}_n': len(x),
        f'{label}_mean': x.mean(),
        f'{label}_sd': x.std(ddof=1),
        f'{label}_skew': stats.skew(x),
        f'{label}_excess_kurtosis': stats.kurtosis(x),
        f'{label}_dagostino_k2_stat': dagostino_stat,
        f'{label}_dagostino_k2_pvalue': dagostino_p,
        f'{label}_jarque_bera_stat': jb_stat,
        f'{label}_jarque_bera_pvalue': jb_p,
    }


def _plot_histogram_with_normality_stats(x, xlabel, title, output_file, stats_dict, prefix):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    mu, sigma = x.mean(), x.std(ddof=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(x, bins=60, color='#377EB8', alpha=0.8, edgecolor='none', density=True)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color='red', linewidth=1.5,
            label=f'Normal(mean={mu:.3f}, sd={sigma:.3f})')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(_wrap_title(title), fontweight='bold')
    annotation = (
        f"n = {stats_dict[f'{prefix}_n']:,}\n"
        f"skew = {stats_dict[f'{prefix}_skew']:.3f}\n"
        f"excess kurtosis = {stats_dict[f'{prefix}_excess_kurtosis']:.3f}\n"
        f"D'Agostino K\u00b2 p = {stats_dict[f'{prefix}_dagostino_k2_pvalue']:.3g}\n"
        f"Jarque-Bera p = {stats_dict[f'{prefix}_jarque_bera_pvalue']:.3g}"
    )
    ax.annotate(annotation, xy=(0.98, 0.97), xycoords='axes fraction', va='top', ha='right', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")


def _plot_residual_qq(resid, title, output_file):
    resid = np.asarray(resid, dtype=float)
    (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist='norm', fit=True)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(osm, osr, s=10, alpha=0.4, color='#377EB8', edgecolor='none')
    xs = np.array([osm.min(), osm.max()])
    ax.plot(xs, intercept + slope * xs, color='red', linestyle='--', linewidth=1.2,
            label=f'Normal fit (R\u00b2 = {r ** 2:.4f})')
    ax.set_xlabel('Theoretical quantiles (Normal)')
    ax.set_ylabel('Sample quantiles (residuals)')
    ax.set_title(_wrap_title(title), fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")


def _plot_residual_vs_fitted(fitted, resid, title, output_file, bp_pvalue, reset_pvalue):
    fitted = np.asarray(fitted, dtype=float)
    resid = np.asarray(resid, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(fitted, resid, s=8, alpha=0.3, color='#377EB8', edgecolor='none')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.2)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(resid, fitted, frac=0.3)
        ax.plot(smooth[:, 0], smooth[:, 1], color='black', linewidth=1.5, label='LOWESS trend')
        ax.legend(loc='lower right', fontsize=8)
    except Exception:
        pass

    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals (actual \u2212 fitted)')
    ax.set_title(_wrap_title(title), fontweight='bold')
    annotation = (
        f"Breusch-Pagan (homoscedasticity)\np = {bp_pvalue:.3g}\n"
        f"Ramsey RESET (linearity)\np = {reset_pvalue:.3g}"
    )
    ax.annotate(annotation, xy=(0.02, 0.98), xycoords='axes fraction', va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")


def run_regression_diagnostics(model, y, title_suffix, hist_te_file, hist_resid_file, qq_file,
                                resid_fitted_file, stats_csv):
    """
    Diagnostics for a `fit_co_mega_te_model` OLS fit (predicting log2(TE)
    from codon frequencies):
      - histogram of the actual response `y` (log2 TE, the data used to fit
        the model), with normality stats.
      - histogram of the model's residuals, with normality stats.
      - Q-Q plot of residuals vs. the Normal distribution.
      - residuals-vs-fitted-values scatter (for homoscedasticity/linearity),
        annotated with the Breusch-Pagan and Ramsey RESET test p-values.

    Statistical tests used (each also printed, and saved as one row to
    `stats_csv`):
      - Normality of `y` and of residuals: skewness, excess kurtosis,
        D'Agostino-Pearson K^2 test, Jarque-Bera test. (Null: the data are
        Normally distributed; p < 0.05 => reject Normality.)
      - Homoscedasticity: Breusch-Pagan test, regressing squared residuals
        on the model's own design matrix (codon frequencies). (Null:
        residual variance is constant; p < 0.05 => reject
        homoscedasticity, i.e. evidence of heteroscedasticity.)
      - Linearity: Ramsey RESET test (F-test of whether adding fitted^2 and
        fitted^3 as extra regressors significantly improves the fit).
        (Null: the linear specification is adequate; p < 0.05 => reject
        linearity, i.e. evidence of a missing nonlinear/misspecified term.)

    Returns a dict of all the above statistics.
    """
    y = np.asarray(y, dtype=float)
    resid = np.asarray(model.resid, dtype=float)
    fitted = np.asarray(model.fittedvalues, dtype=float)

    y_stats = _normality_stats(y, 'TE')
    resid_stats = _normality_stats(resid, 'residuals')

    bp_stat, bp_pvalue, bp_fstat, bp_fpvalue = het_breuschpagan(resid, model.model.exog)
    reset_result = linear_reset(model, power=3, test_type='fitted', use_f=True)
    reset_stat, reset_pvalue = float(reset_result.statistic), float(reset_result.pvalue)

    all_stats = {
        **y_stats, **resid_stats,
        'breusch_pagan_lm_stat': bp_stat, 'breusch_pagan_pvalue': bp_pvalue,
        'ramsey_reset_fstat': reset_stat, 'ramsey_reset_pvalue': reset_pvalue,
    }

    print(f"  --- Regression diagnostics ({title_suffix}) ---")
    print(f"  Normality of TE (actual, log2 scale, n={y_stats['TE_n']:,}): "
          f"skew = {y_stats['TE_skew']:.3f}, excess kurtosis = {y_stats['TE_excess_kurtosis']:.3f}, "
          f"D'Agostino K^2 p = {y_stats['TE_dagostino_k2_pvalue']:.3g}, "
          f"Jarque-Bera p = {y_stats['TE_jarque_bera_pvalue']:.3g}")
    print(f"  Normality of residuals: "
          f"skew = {resid_stats['residuals_skew']:.3f}, "
          f"excess kurtosis = {resid_stats['residuals_excess_kurtosis']:.3f}, "
          f"D'Agostino K^2 p = {resid_stats['residuals_dagostino_k2_pvalue']:.3g}, "
          f"Jarque-Bera p = {resid_stats['residuals_jarque_bera_pvalue']:.3g}")
    print(f"  Homoscedasticity (Breusch-Pagan): LM stat = {bp_stat:.3f}, p = {bp_pvalue:.3g}")
    print(f"  Linearity (Ramsey RESET, power=3 on fitted values): F = {reset_stat:.3f}, p = {reset_pvalue:.3g}")
    print("  (in each test, p < 0.05 => reject the null of normality / homoscedasticity / linearity)")

    _plot_histogram_with_normality_stats(
        y, 'Actual TE_mean (log2 ribo/RNA)', f'Distribution of TE_mean used for regression\n{title_suffix}',
        hist_te_file, y_stats, 'TE'
    )
    _plot_histogram_with_normality_stats(
        resid, 'Residual (actual \u2212 fitted TE_mean)', f'Distribution of regression residuals\n{title_suffix}',
        hist_resid_file, resid_stats, 'residuals'
    )
    _plot_residual_qq(resid, f'Q-Q plot of regression residuals\n{title_suffix}', qq_file)
    _plot_residual_vs_fitted(
        fitted, resid, f'Residuals vs. fitted values\n{title_suffix}', resid_fitted_file, bp_pvalue, reset_pvalue
    )

    pd.DataFrame([all_stats]).to_csv(stats_csv, index=False)
    print(f"  Saved diagnostic statistics to {stats_csv}")
    return all_stats


# ── Plot: predicted vs. actual / CO_Mega scatter ────────────────────────────
def _plot_actual_vs_predicted_impl(actual, predicted, actual_label, predicted_label, title, stat_value, p_str,
                                    output_file, stat_symbol, stat_name):
    """
    Shared implementation for `plot_actual_vs_predicted` (Pearson R) and
    `plot_actual_vs_predicted_spearman` (Spearman rho) -- see those for
    details. `stat_symbol` (e.g. 'R' or '\u03c1') is used in the plot
    annotation; `stat_name` (e.g. 'Pearson R' or 'Spearman rho') is used in
    the printed summary line.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    stat_squared = stat_value ** 2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(predicted, actual, s=8, alpha=0.3, color='#377EB8', edgecolor='none')
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(lims, lims, color='gray', linestyle='--', linewidth=1, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(predicted_label)
    ax.set_ylabel(actual_label)
    ax.set_title(title, fontweight='bold')
    ax.annotate(f"{stat_symbol} = {stat_value:.4f}\n{stat_symbol}\u00b2 = {stat_squared:.4f}\nn = {len(actual):,}",
                xy=(0.05, 0.95), xycoords='axes fraction', va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}  ({stat_name} = {stat_value:.4f}, {stat_name}\u00b2 = {stat_squared:.4f}, "
          f"p = {p_str}, n = {len(actual):,})")


def plot_actual_vs_predicted(actual, predicted, actual_label, predicted_label, title, r, p_str, output_file):
    """
    Scatter of a model-derived 'predicted' quantity (x-axis) vs. the
    corresponding 'actual' quantity (y-axis) (e.g. predicted/fitted TE_mean
    or CO_Mega on x, actual TE_mean on y), with a y = x reference line,
    equal axis scales, and R / R^2 / p (scientific notation) / n annotated.
    R and R^2 are unaffected by which quantity is plotted on which axis
    (Pearson correlation is symmetric).
    """
    _plot_actual_vs_predicted_impl(actual, predicted, actual_label, predicted_label, title, r, p_str,
                                    output_file, stat_symbol='R', stat_name='Pearson R')


def plot_actual_vs_predicted_spearman(actual, predicted, actual_label, predicted_label, title, rho, p_str,
                                       output_file):
    """
    Same as `plot_actual_vs_predicted`, but annotated with Spearman's rho
    (rank correlation) instead of Pearson's R. Intended as a parallel
    "Spearman copy" of a `plot_actual_vs_predicted` figure, saved to a
    different `output_file` (e.g. with a `_spearman` suffix).
    """
    _plot_actual_vs_predicted_impl(actual, predicted, actual_label, predicted_label, title, rho, p_str,
                                    output_file, stat_symbol='\u03c1', stat_name='Spearman rho')


def _wrap_title(title, width=52):
    return "\n".join(textwrap.fill(line, width=width) for line in title.split("\n"))


def _nonempty_groups(df, value_col, group_col, group_order):
    """
    For each group in `group_order` present in `df[group_col]`, return its
    non-NaN `value_col` values, dropping any group left with 0 observations
    (e.g. after an upstream transform like log() made every value in that
    group undefined).
    """
    groups, data = [], []
    for g in group_order:
        if g not in set(df[group_col]):
            continue
        d = df.loc[df[group_col] == g, value_col].dropna()
        if len(d) > 0:
            groups.append(g)
            data.append(d)
    return groups, data


# ── Plot: two-group (autosome vs. sex chromosome) boxplot ──────────────────
def plot_co_mega_boxplot(df, value_col, group_col, group1_label, title, output_file, ylabel='CO_Mega'):
    """
    Boxplot comparing `value_col` between genes with `group_col` == 0
    ('Autosome') and `group_col` == 1 (labeled `group1_label`, e.g. 'X
    chromosome' or 'Z chromosome'), with a Mann-Whitney U p-value annotated.
    Each group's tick label shows n and mean.
    """
    autosome = df.loc[df[group_col] == 0, value_col].dropna()
    other = df.loc[df[group_col] == 1, value_col].dropna()
    pval = stats.mannwhitneyu(autosome, other, alternative='two-sided').pvalue

    fig, ax = plt.subplots(figsize=(6.5, 6))
    bp = ax.boxplot(
        [autosome, other],
        tick_labels=[
            f'Autosome\n(n={len(autosome):,})\n(mean={autosome.mean():.3f})',
            f'{group1_label}\n(n={len(other):,})\n(mean={other.mean():.3f})',
        ],
        patch_artist=True, widths=0.5, showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.3, markeredgecolor='none')
    )
    for patch, color in zip(bp['boxes'], ['#377EB8', '#E41A1C']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel(ylabel)
    ax.set_title(_wrap_title(title), fontweight='bold', fontsize=12)
    ax.annotate(f"Mann-Whitney U\np = {pval:.3g}",
                xy=(0.5, 0.98), xycoords='axes fraction', va='top', ha='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}  (Mann-Whitney U p = {pval:.3g}, "
          f"mean autosome = {autosome.mean():.3f}, mean {group1_label} = {other.mean():.3f})")


# ── Plot: multi-group (e.g. XCI-category) boxplot ───────────────────────────
GROUP_COLORS = ['#377EB8', '#E41A1C', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33']

# Fixed colors for specific, recurring group names (independent of their
# position in `group_order`), so e.g. 'Xi-expressed' is always #00e0e8 and
# 'Xi-silent' is always #ff7b7f wherever they appear.
_NAMED_GROUP_COLORS = {
    'Autosome': '#377EB8',
    'Xi-silent': '#ff7b7f',
    'Xi-expressed': '#00e0e8',
    'X (defined XCI)': '#E41A1C',
    'NPX-NPY': '#984EA3',
    'PAR1': '#FF7F00',
    'PAR2': '#FFFF33',
    'X': '#E41A1C',
    'Y': '#4DAF4A',
}
_FALLBACK_GROUP_COLORS = GROUP_COLORS


def _group_colors(groups):
    """Color for each group in `groups`: a fixed color for known names
    (see `_NAMED_GROUP_COLORS`), else the next unused color from the
    default `GROUP_COLORS` cycle."""
    fallback = (c for c in _FALLBACK_GROUP_COLORS if c not in _NAMED_GROUP_COLORS.values())
    return [_NAMED_GROUP_COLORS.get(g) or next(fallback, '#999999') for g in groups]


def plot_co_mega_by_group(df, value_col, group_col, group_order, title, output_file, out_csv=None, ylabel=None):
    """
    Boxplot of `value_col` across the groups in `group_order` that are
    actually present in `df[group_col]` (missing ones are skipped). Prints
    (and shows, in a plot legend) the Mann-Whitney U p-value for every
    pairwise comparison of groups. If `out_csv` is given, saves a table with
    n / mean per group and the p-value for every pairwise comparison.
    """
    groups, data = _nonempty_groups(df, value_col, group_col, group_order)
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
    for patch, color in zip(bp['boxes'], _group_colors(groups)):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel(ylabel if ylabel is not None else value_col)
    ax.set_title(_wrap_title(title), fontweight='bold')

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


# ── Plot: two-group / multi-group CDF ───────────────────────────────────────
def _ecdf(values):
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_co_mega_cdf(df, value_col, group_col, group1_label, xlabel, title, output_file):
    """
    Empirical-CDF plot comparing `value_col` between genes with `group_col`
    == 0 ('Autosome') and `group_col` == 1 (labeled `group1_label`), with a
    Mann-Whitney U p-value annotated (scientific notation). Each group's
    legend entry shows n and mean.
    """
    autosome = df.loc[df[group_col] == 0, value_col].dropna()
    other = df.loc[df[group_col] == 1, value_col].dropna()
    pval = stats.mannwhitneyu(autosome, other, alternative='two-sided').pvalue

    fig, ax = plt.subplots(figsize=(8, 6))
    for data, label_base, color in [(autosome, 'Autosome', '#377EB8'), (other, group1_label, '#E41A1C')]:
        x, y = _ecdf(data)
        ax.step(x, y, where='post', color=color, linewidth=1.8,
                label=f'{label_base} (n={len(data):,}, mean={data.mean():.3f})')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative probability')
    ax.set_title(_wrap_title(title), fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.annotate(f"Mann-Whitney U\np = {pval:.3g}",
                xy=(0.02, 0.98), xycoords='axes fraction', va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}  (Mann-Whitney U p = {pval:.3g}, "
          f"mean autosome = {autosome.mean():.3f}, mean {group1_label} = {other.mean():.3f})")


def plot_co_mega_cdf_by_group(df, value_col, group_col, group_order, xlabel, title, output_file):
    """
    Empirical-CDF plot of `value_col` across the groups in `group_order`
    that are actually present in `df[group_col]` (missing ones are
    skipped), with every pairwise Mann-Whitney U p-value (scientific
    notation) listed in a legend. Each group's line legend entry shows n
    and mean.
    """
    groups, data = _nonempty_groups(df, value_col, group_col, group_order)

    pairwise = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            pval = stats.mannwhitneyu(data[i], data[j], alternative='two-sided').pvalue
            pairwise.append((groups[i], groups[j], pval))

    fig, ax = plt.subplots(figsize=(1.4 * len(groups) + 6, 6))
    for g, d, color in zip(groups, data, _group_colors(groups)):
        x, y = _ecdf(d)
        ax.step(x, y, where='post', color=color, linewidth=1.8,
                label=f'{g} (n={len(d):,}, mean={d.mean():.3f})')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative probability')
    ax.set_title(_wrap_title(title), fontweight='bold')

    line_legend = ax.legend(loc='lower right', fontsize=8)
    legend_text = "Mann-Whitney U p-values:\n" + "\n".join(
        f"{g1} vs {g2}: p = {pval:.3g}" for g1, g2, pval in pairwise
    )
    legend_handle = Line2D([], [], color='none', label=legend_text)
    ax.legend(handles=[legend_handle], loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=7, handlelength=0, handletextpad=0, frameon=True, borderaxespad=0)
    ax.add_artist(line_legend)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")

    print(f"  Pairwise Mann-Whitney U p-values ({title}):")
    for g1, g2, pval in pairwise:
        i, j = groups.index(g1), groups.index(g2)
        print(f"    {g1} (n={len(data[i]):,}) vs {g2} (n={len(data[j]):,}): p = {pval:.3g}")


# ── Plot: autosomes-split-by-chromosome vs. X ──────────────────────────────
def _significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


def plot_co_mega_boxplot_by_chromosome(df, value_col, chrom_col, autosome_order, title, output_file, out_csv,
                                        ylabel=None, x_value='X', x_label='X chromosome',
                                        autosome_color='#377EB8', x_color='#E41A1C'):
    """
    Boxplot of `value_col`, with one box per autosome in `autosome_order`
    (matched against `df[chrom_col]`, all colored `autosome_color`) plus
    one (colored `x_color`) box for the X-chromosome group
    (`df[chrom_col] == x_value`). Each autosome's Mann-Whitney U p-value vs. the X group is computed and
    annotated on the plot as significance asterisks
    (* p<0.05, ** p<0.01, *** p<0.001), and saved (with n, mean) to
    `out_csv`.
    """
    groups, data = _nonempty_groups(df, value_col, chrom_col, list(autosome_order) + [x_value])
    x_idx = groups.index(x_value)
    x_data = data[x_idx]

    pvals = []
    for g, d in zip(groups, data):
        if g == x_value:
            pvals.append(np.nan)
        else:
            pvals.append(stats.mannwhitneyu(d, x_data, alternative='two-sided').pvalue)

    labels = [x_label if g == x_value else g for g in groups]
    colors = [x_color if g == x_value else autosome_color for g in groups]

    fig, ax = plt.subplots(figsize=(0.55 * len(groups) + 3, 6))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6, showfliers=True,
                     flierprops=dict(marker='o', markersize=2, alpha=0.25, markeredgecolor='none'))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (g, d, p) in enumerate(zip(groups, data, pvals)):
        stars = '' if np.isnan(p) else _significance_stars(p)
        if stars:
            ax.text(i + 1, d.max(), stars, ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Chromosome')
    ax.set_ylabel(ylabel if ylabel is not None else value_col)
    ax.set_title(_wrap_title(title), fontweight='bold')
    ax.annotate("* p<0.05  ** p<0.01  *** p<0.001\n(each autosome vs. X, Mann-Whitney U)",
                xy=(0.99, 0.98), xycoords='axes fraction', va='top', ha='right', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")

    rows = []
    for g, d, p in zip(groups, data, pvals):
        rows.append({
            'chromosome': x_label if g == x_value else g,
            'n': len(d), 'mean': d.mean(), 'p_value_vs_X': p,
            'significance': '' if np.isnan(p) else _significance_stars(p),
        })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out_csv, index=False)
    print(f"  Saved chromosome-vs-X stats to {out_csv}")
    return stats_df


def plot_co_mega_cdf_by_chromosome(df, value_col, chrom_col, autosome_order, xlabel, title, output_file,
                                    x_value='X', x_label='X chromosome',
                                    autosome_color='#377EB8', x_color='#E41A1C'):
    """
    Empirical-CDF plot of `value_col`, with one curve (colored
    `autosome_color`) per autosome in `autosome_order` plus one curve
    (colored `x_color`) for the X-chromosome group. The legend shows a
    single proxy entry for the autosome curves plus the X group's n/mean,
    rather than one legend line per autosome.
    """
    groups, data = _nonempty_groups(df, value_col, chrom_col, list(autosome_order) + [x_value])

    fig, ax = plt.subplots(figsize=(8, 6))
    n_auto_curves, n_auto_genes, x_entry = 0, 0, None
    for g, d in zip(groups, data):
        if g == x_value:
            ax.step(*_ecdf(d), where='post', color=x_color, linewidth=2.0, zorder=3)
            x_entry = f'{x_label} (n={len(d):,}, mean={d.mean():.3f})'
        else:
            ax.step(*_ecdf(d), where='post', color=autosome_color, linewidth=1.0, alpha=0.7, zorder=2)
            n_auto_curves += 1
            n_auto_genes += len(d)

    handles = [
        Line2D([0], [0], color=autosome_color, linewidth=1.5,
               label=f'Autosomes, per chromosome ({n_auto_curves} curves, {n_auto_genes:,} genes)'),
        Line2D([0], [0], color=x_color, linewidth=2.0, label=x_entry),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative probability')
    ax.set_title(_wrap_title(title), fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")


def make_autosome_x_y_group(chromosome_series):
    """
    Map a chromosome column (values '1'..'22', 'X', 'Y') to a 3-level
    'Autosome' / 'X' / 'Y' group array, ready for `plot_co_mega_by_group` /
    `plot_co_mega_cdf_by_group`.
    """
    return np.where(chromosome_series == 'X', 'X', np.where(chromosome_series == 'Y', 'Y', 'Autosome'))


def build_xci_groups(df, xci_file, value_col='CO_Mega', group_col='is_X', exclude_values=()):
    """
    Merge genes with `group_col` == 1 in `df` with a gene_name ->
    classification table (`xci_file`, columns 'gene_name' and
    'classification'), dropping any classification in `exclude_values` (e.g.
    'No call'). Genes with `group_col` == 1 not found in the table are
    excluded. Genes with `group_col` == 0 are labeled 'Autosome'. Returns
    (groups_df, other_genes), where `groups_df` has columns
    [value_col, 'classification'] ready for `plot_co_mega_by_group`, and
    `other_genes` is the merged (post-exclusion) `group_col` == 1-only frame.
    """
    xci_df = pd.read_csv(xci_file, usecols=['gene_name', 'classification'])
    other_genes = df.loc[df[group_col] == 1, ['gene_name', value_col]].merge(
        xci_df, on='gene_name', how='inner'
    )
    if exclude_values:
        other_genes = other_genes[~other_genes['classification'].isin(exclude_values)]

    auto_genes = df.loc[df[group_col] == 0, [value_col]].copy()
    auto_genes['classification'] = 'Autosome'
    groups_df = pd.concat(
        [auto_genes[[value_col, 'classification']], other_genes[[value_col, 'classification']]],
        ignore_index=True
    )
    return groups_df, other_genes


# ── Per-codon Pearson r / Spearman rho vs. TE, with bootstrap SD ───────────
def _pearson_r_columns(X, y):
    """Pearson r of each column of X (n x k) against y (n,), vectorized."""
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    num = Xc.T @ yc
    denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    return num / denom


def _spearman_rho_columns(X, y):
    """
    Spearman rho of each column of X (n x k) against y (n,), vectorized.
    Computed as the Pearson r of the rank-transformed values, which is
    mathematically identical to Spearman's rho.
    """
    X_ranked = stats.rankdata(X, axis=0)
    y_ranked = stats.rankdata(y)
    return _pearson_r_columns(X_ranked, y_ranked)


def _bootstrap_codon_corr(codon_freq_df, te, corr_columns_fn, value_col, codons, n_boot, random_state):
    """
    Shared implementation for `bootstrap_codon_pearson_r` and
    `bootstrap_codon_spearman_rho`: for each codon, compute its correlation
    (via `corr_columns_fn`) with `te` (point estimate on the full data),
    plus a bootstrap SD obtained by resampling genes (rows) with
    replacement `n_boot` times and recomputing every codon's correlation
    each time. Returns a DataFrame with columns: codon, `value_col`,
    boot_sd, p_value (precise, scientific-notation string), n.
    """
    X = codon_freq_df[list(codons)].to_numpy(dtype=float)
    y = np.asarray(te, dtype=float)
    n = len(y)

    point = corr_columns_fn(X, y)

    rng = np.random.default_rng(random_state)
    boot = np.empty((n_boot, len(codons)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = corr_columns_fn(X[idx], y[idx])
    boot_sd = boot.std(axis=0, ddof=1)

    p_values = [precise_pvalue_str(v, n) for v in point]

    return pd.DataFrame({
        'codon': list(codons),
        value_col: point,
        'boot_sd': boot_sd,
        'p_value': p_values,
        'n': n,
    })


def bootstrap_codon_pearson_r(codon_freq_df, te, codons=SENSE_CODONS, n_boot=2000, random_state=0):
    """
    For each codon, compute its Pearson r with `te` (point estimate on the
    full data), plus a bootstrap SD of that r obtained by resampling genes
    (rows) with replacement `n_boot` times and recomputing every codon's r
    each time.

    Returns a DataFrame with columns: codon, pearson_r, boot_sd, p_value
    (precise, scientific-notation string), n.
    """
    return _bootstrap_codon_corr(codon_freq_df, te, _pearson_r_columns, 'pearson_r', codons, n_boot, random_state)


def bootstrap_codon_spearman_rho(codon_freq_df, te, codons=SENSE_CODONS, n_boot=2000, random_state=0):
    """
    Spearman-rho analog of `bootstrap_codon_pearson_r`: for each codon,
    compute its Spearman rank correlation with `te` (point estimate on the
    full data), plus a bootstrap SD of that rho obtained by resampling
    genes (rows) with replacement `n_boot` times and recomputing every
    codon's rho each time.

    For speed, `X`/`te` are rank-transformed ONCE up front (over the full,
    un-resampled data), and every bootstrap resample's rho is then computed
    as the (fast, vectorized) Pearson r of the resampled ranks -- this is
    the standard "bootstrap the ranks" approach for estimating a Spearman
    correlation's bootstrap SD, and is far cheaper than re-ranking every
    one of the `n_boot` resamples from scratch (which, for tens of
    thousands of genes x 61 codons x thousands of resamples, is
    prohibitively slow). The point estimate (rho on the full, un-resampled
    data) is exact either way, since ranking then correlating equals
    correlating the ranks.

    Returns a DataFrame with columns: codon, spearman_rho, boot_sd, p_value
    (precise, scientific-notation string; computed via the same t-distribution
    approximation as for Pearson r, applied to the rank correlation -- the
    standard approach for Spearman significance at moderate/large n), n.
    """
    X = codon_freq_df[list(codons)].to_numpy(dtype=float)
    y = np.asarray(te, dtype=float)
    X_ranked = stats.rankdata(X, axis=0)
    y_ranked = stats.rankdata(y)
    ranked_df = pd.DataFrame(X_ranked, columns=list(codons))
    return _bootstrap_codon_corr(ranked_df, y_ranked, _pearson_r_columns, 'spearman_rho', codons, n_boot,
                                  random_state)


def _plot_codon_corr_barplot(r_df, value_col, title, output_file, ylabel, error_kind):
    """
    Shared implementation for `plot_codon_pearson_r_barplot` and
    `plot_codon_spearman_rho_barplot`: bar plot of each codon's correlation
    (`value_col`) with TE, sorted in descending order, with bootstrap-SD
    error bars.
    """
    df_sorted = r_df.sort_values(value_col, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(18, 6))
    colors = ['#377EB8' if v >= 0 else '#FF7F00' for v in df_sorted[value_col]]
    ax.bar(df_sorted['codon'], df_sorted[value_col], yerr=df_sorted['boot_sd'],
           color=colors, alpha=0.8, capsize=2, error_kw=dict(elinewidth=0.8, capthick=0.8))
    ax.axhline(0, color='black', linewidth=0.8)

    ax.set_xlabel('Codon')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['codon'], rotation=90, fontsize=7)
    ax.annotate(f"n = {int(df_sorted['n'].iloc[0]):,}\nerror bars = {error_kind} ({len(df_sorted):,} codons)",
                xy=(0.99, 0.98), xycoords='axes fraction', va='top', ha='right', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")


def plot_codon_pearson_r_barplot(r_df, title, output_file):
    """
    Bar plot of each codon's Pearson r with TE (from `bootstrap_codon_pearson_r`),
    sorted in descending order of r, with bootstrap-SD error bars.
    """
    _plot_codon_corr_barplot(r_df, 'pearson_r', title, output_file,
                              'Pearson R between codon frequency and TE', 'bootstrap SD')


def plot_codon_spearman_rho_barplot(r_df, title, output_file):
    """
    Spearman-rho analog of `plot_codon_pearson_r_barplot`: bar plot of each
    codon's Spearman rho with TE (from `bootstrap_codon_spearman_rho`),
    sorted in descending order of rho, with bootstrap-SD error bars.
    """
    _plot_codon_corr_barplot(r_df, 'spearman_rho', title, output_file,
                              'Spearman rho between codon frequency and TE', 'bootstrap SD')


# ── Plot: per-codon Pearson r comparison between two species/tissues ──────
def plot_pearson_r_comparison_scatter(cmp_df, y_label, x_label, r_corr, p_corr, rho_corr, p_rho, output_file,
                                       yerr=None, error_label=None):
    """
    Scatter of one species/tissue's per-codon Pearson r (`cmp_df['pearson_r_y']`)
    vs. another's (`cmp_df['pearson_r_x']`), one point per codon (labeled),
    with a y = x reference line and the Pearson/Spearman correlation between
    the two annotated. Optionally draws asymmetric error bars (`yerr`) on the
    y-axis values, e.g. a confidence interval, labeled by `error_label`.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    if yerr is not None:
        ax.errorbar(cmp_df['pearson_r_x'], cmp_df['pearson_r_y'], yerr=yerr,
                     fmt='o', markersize=5, color='#377EB8', ecolor='#377EB8',
                     elinewidth=0.8, capsize=2, alpha=0.85)
    else:
        ax.scatter(cmp_df['pearson_r_x'], cmp_df['pearson_r_y'], s=25, color='#377EB8')
    for _, row in cmp_df.iterrows():
        ax.annotate(row['codon'], (row['pearson_r_x'], row['pearson_r_y']), fontsize=6,
                    xytext=(2, 2), textcoords='offset points')

    lo = min(cmp_df['pearson_r_x'].min(), cmp_df['pearson_r_y'].min()) - 0.05
    hi = max(cmp_df['pearson_r_x'].max(), cmp_df['pearson_r_y'].max()) + 0.05
    lims = [lo, hi]
    ax.plot(lims, lims, color='gray', linestyle='--', linewidth=1, label='y = x')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(f'Pearson R between codon frequency and TE ({x_label})')
    ax.set_ylabel(f'Pearson R between codon frequency and TE ({y_label})')
    ax.set_title(f'Per-codon Pearson R: {y_label} vs. {x_label}', fontweight='bold')
    annotation = (f"Pearson R = {r_corr:.4f} (p = {p_corr:.3g})\n"
                  f"Spearman rho = {rho_corr:.4f} (p = {p_rho:.3g})\nn = {len(cmp_df)} codons")
    if error_label is not None:
        annotation += f"\nerror bars = {error_label}"
    ax.annotate(
        annotation,
        xy=(0.05, 0.95), xycoords='axes fraction', va='top', ha='left', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")


# ── Per-codon Pearson r / Spearman rho vs. TE, ordinary (analytic Fisher-z CI) ──
def _ordinary_codon_corr(codon_freq_df, te, corr_columns_fn, value_col, codons, confidence):
    """
    Shared implementation for `ordinary_codon_pearson_r` and
    `ordinary_codon_spearman_rho`: for each codon, compute its correlation
    (via `corr_columns_fn`) with `te`, plus an analytic confidence interval
    via the Fisher z transformation (z = arctanh(corr), SE_z =
    1/sqrt(n-3)). Returns a DataFrame with columns: codon, `value_col`,
    ci_lower, ci_upper, p_value (precise, scientific-notation string), n.
    """
    X = codon_freq_df[list(codons)].to_numpy(dtype=float)
    y = np.asarray(te, dtype=float)
    n = len(y)

    corr = corr_columns_fn(X, y)
    z = np.arctanh(corr)
    se_z = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(0.5 + confidence / 2)
    ci_lower = np.tanh(z - z_crit * se_z)
    ci_upper = np.tanh(z + z_crit * se_z)

    p_values = [precise_pvalue_str(v, n) for v in corr]

    return pd.DataFrame({
        'codon': list(codons),
        value_col: corr,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_values,
        'n': n,
    })


def ordinary_codon_pearson_r(codon_freq_df, te, codons=SENSE_CODONS, confidence=0.95):
    """
    For each codon, compute its ordinary (non-bootstrap) Pearson r with `te`,
    plus an analytic confidence interval obtained via the Fisher z
    transformation (z = arctanh(r), SE_z = 1/sqrt(n-3)), rather than a
    bootstrap SD.

    Returns a DataFrame with columns: codon, pearson_r, ci_lower, ci_upper,
    p_value (precise, scientific-notation string), n.
    """
    return _ordinary_codon_corr(codon_freq_df, te, _pearson_r_columns, 'pearson_r', codons, confidence)


def ordinary_codon_spearman_rho(codon_freq_df, te, codons=SENSE_CODONS, confidence=0.95):
    """
    Spearman-rho analog of `ordinary_codon_pearson_r`: for each codon,
    compute its ordinary (non-bootstrap) Spearman rho with `te`, plus an
    analytic confidence interval via the Fisher z transformation applied
    to rho (the standard approximation for Spearman confidence intervals
    at moderate/large n).

    Returns a DataFrame with columns: codon, spearman_rho, ci_lower,
    ci_upper, p_value (precise, scientific-notation string), n.
    """
    return _ordinary_codon_corr(codon_freq_df, te, _spearman_rho_columns, 'spearman_rho', codons, confidence)


def _plot_codon_corr_barplot_ci(r_df, value_col, title, output_file, ylabel, confidence):
    """
    Shared implementation for `plot_codon_pearson_r_barplot_ci` and
    `plot_codon_spearman_rho_barplot_ci`: bar plot of each codon's
    correlation (`value_col`) with TE, sorted in descending order, with
    analytic Fisher-z confidence-interval error bars.
    """
    df_sorted = r_df.sort_values(value_col, ascending=False).reset_index(drop=True)
    lower_err = (df_sorted[value_col] - df_sorted['ci_lower']).to_numpy()
    upper_err = (df_sorted['ci_upper'] - df_sorted[value_col]).to_numpy()

    fig, ax = plt.subplots(figsize=(18, 6))
    colors = ['#377EB8' if v >= 0 else '#FF7F00' for v in df_sorted[value_col]]
    ax.bar(df_sorted['codon'], df_sorted[value_col], yerr=[lower_err, upper_err],
           color=colors, alpha=0.8, capsize=2, error_kw=dict(elinewidth=0.8, capthick=0.8))
    ax.axhline(0, color='black', linewidth=0.8)

    ax.set_xlabel('Codon')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['codon'], rotation=90, fontsize=7)
    ax.annotate(f"n = {int(df_sorted['n'].iloc[0]):,}\n"
                f"error bars = {confidence:.0%} Fisher-z CI ({len(df_sorted):,} codons)",
                xy=(0.99, 0.98), xycoords='axes fraction', va='top', ha='right', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")


def plot_codon_pearson_r_barplot_ci(r_df, title, output_file, confidence=0.95):
    """
    Bar plot of each codon's Pearson r with TE (from `ordinary_codon_pearson_r`),
    sorted in descending order of r, with analytic Fisher-z confidence-interval
    error bars.
    """
    _plot_codon_corr_barplot_ci(r_df, 'pearson_r', title, output_file,
                                 'Pearson R between codon frequency and TE', confidence)


def plot_codon_spearman_rho_barplot_ci(r_df, title, output_file, confidence=0.95):
    """
    Spearman-rho analog of `plot_codon_pearson_r_barplot_ci`: bar plot of
    each codon's Spearman rho with TE (from `ordinary_codon_spearman_rho`),
    sorted in descending order of rho, with analytic Fisher-z
    confidence-interval error bars.
    """
    _plot_codon_corr_barplot_ci(r_df, 'spearman_rho', title, output_file,
                                 'Spearman rho between codon frequency and TE', confidence)
