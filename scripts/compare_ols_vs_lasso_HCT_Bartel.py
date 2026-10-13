#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
How does the lasso CO_Mega model differ from the ordinary least squares (OLS) model when the human TE is
the Bartel HCT TE?

Compares the outputs of
  - human_co_mega_analysis_HCT_Bartel.py       (OLS)    -> tables/human_HCT_Bartel/
  - human_co_mega_test_lasso_HCT_Bartel.py     (lasso)  -> tables/human_HCT_Bartel_lasso/
(run both first). Both fit  TE_mean (log2 Bartel TE) ~ 61 codon frequencies  on autosomal genes and are applied
to all genes.

Comparisons:
  1. Coefficients: which codons the lasso sets to exactly 0 and how significant they are in the OLS model;
     how much the surviving coefficients shrink (slope of lasso on OLS coefficients) and how well the two
     coefficient vectors agree (Pearson / Spearman).
  2. Fit on the autosomal training genes: in-sample R^2 (OLS can never be lower) and 10-fold
     cross-validated R^2 (out-of-sample; the lasso penalty is re-chosen inside every training fold), which
     is the fair comparison.
  3. Held-out genes (X-linked; the models are trained on autosomes only): Pearson / Spearman / R^2 of
     CO_Mega vs TE.
  4. Per-gene agreement of the two CO_Mega scores.

Outputs: tables/human_HCT_Bartel_lasso/ols_vs_lasso_*.csv and figures/human_HCT_Bartel_lasso/ols_vs_lasso_*.png.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from co_mega import SENSE_CODONS

BASE_DIR = "/lab/solexa_page/xueqi/analysis_codon"
OLS_TABLES = os.path.join(BASE_DIR, "tables", "human_HCT_Bartel")
LASSO_TABLES = os.path.join(BASE_DIR, "tables", "human_HCT_Bartel_lasso")
OUT_TABLES = LASSO_TABLES
OUT_FIGS = os.path.join(BASE_DIR, "figures", "human_HCT_Bartel_lasso")
AUTOSOMES = [str(i) for i in range(1, 23)]
N_FOLDS = 10
SEED = 0


def load_models():
    ols = pd.read_csv(os.path.join(OLS_TABLES, "co_mega_model_coefficients_human_HCT_Bartel.csv"))
    lasso = pd.read_csv(os.path.join(LASSO_TABLES, "co_mega_model_coefficients_human_HCT_Bartel_lasso.csv"))
    coef = ols.rename(columns={'coefficient': 'ols_coefficient', 'p_value': 'ols_p_value',
                               'std_err': 'ols_std_err'})[['term', 'ols_coefficient', 'ols_std_err', 'ols_p_value']].merge(
        lasso.rename(columns={'coefficient': 'lasso_coefficient'})[['term', 'lasso_coefficient', 'selected']], on='term')
    alpha = float(lasso['lasso_alpha'].iloc[0])
    scores_ols = pd.read_csv(os.path.join(OLS_TABLES, "co_mega_all_genes_human_HCT_Bartel.csv"))
    scores_lasso = pd.read_csv(os.path.join(LASSO_TABLES, "co_mega_all_genes_human_HCT_Bartel_lasso.csv"))
    scores = scores_ols[['ensembl_gene_id', 'chromosome', 'TE_mean', 'CO_Mega']].rename(columns={'CO_Mega': 'CO_Mega_ols'}).merge(
        scores_lasso[['ensembl_gene_id', 'CO_Mega']].rename(columns={'CO_Mega': 'CO_Mega_lasso'}), on='ensembl_gene_id')
    scores['chromosome'] = scores['chromosome'].astype(str)
    return coef, alpha, scores


def compare_coefficients(coef, alpha):
    codons = coef[coef['term'] != 'intercept'].copy()
    zero = codons[~codons['selected']]
    kept = codons[codons['selected']]
    r, _ = stats.pearsonr(codons['ols_coefficient'], codons['lasso_coefficient'])
    rho, _ = stats.spearmanr(codons['ols_coefficient'], codons['lasso_coefficient'])
    slope = np.polyfit(kept['ols_coefficient'], kept['lasso_coefficient'], 1)[0]
    sig = codons['ols_p_value'] < 0.05
    print("\n[1] Coefficients")
    print(f"  lasso alpha (10-fold CV, standardized predictors) = {alpha:.3g}")
    print(f"  lasso keeps {len(kept)} of {len(codons)} codons; sets {len(zero)} to exactly 0: "
          f"{', '.join(zero['term'])}")
    print(f"  of those {len(zero)}, {int((zero['ols_p_value'] < 0.05).sum())} are nominally significant "
          f"(p < 0.05) in the OLS model; OLS has {int(sig.sum())} nominally significant codons overall, "
          f"{int(codons.loc[sig, 'selected'].sum())} of them kept by the lasso")
    print(f"  OLS vs lasso coefficients: Pearson r = {r:.4f}, Spearman rho = {rho:.4f}; "
          f"slope of lasso on OLS among kept codons = {slope:.3f} (shrinkage)")
    print(f"  coefficient mean / SD across the 61 codons: OLS {codons['ols_coefficient'].mean():+.3f} / "
          f"{codons['ols_coefficient'].std():.3f}, lasso {codons['lasso_coefficient'].mean():+.3f} / "
          f"{codons['lasso_coefficient'].std():.3f}")
    print("  NOTE: the 61 sense-codon frequencies of a gene sum to ~1 (see [2]), so together with the intercept the OLS\n"
          "  coefficients are only identified up to a common shift (add c to every coefficient, subtract c from the\n"
          "  intercept): the huge OLS intercept / mean coefficient are that arbitrary shift, not an effect. Only the\n"
          "  differences between codons (SD, correlations, slope above, all shift-invariant) are meaningful; the lasso\n"
          "  penalty picks the sparse solution, which is why its intercept and mean coefficient stay small.")
    codons = codons.assign(ols_coefficient_centered=codons['ols_coefficient'] - codons['ols_coefficient'].mean(),
                           lasso_coefficient_centered=codons['lasso_coefficient'] - codons['lasso_coefficient'].mean())
    ic = coef[coef['term'] == 'intercept'].iloc[0]
    print(f"  intercept: OLS {ic['ols_coefficient']:.4f}, lasso {ic['lasso_coefficient']:.4f}")
    coef.merge(codons[['term', 'ols_coefficient_centered', 'lasso_coefficient_centered']], on='term', how='left').assign(
        zeroed_by_lasso=lambda d: ~d['selected']).to_csv(os.path.join(OUT_TABLES, "ols_vs_lasso_coefficients.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(kept['ols_coefficient'], kept['lasso_coefficient'], s=22, color='#4D4D4D', label=f'kept by lasso ({len(kept)})')
    ax.scatter(zero['ols_coefficient'], zero['lasso_coefficient'], s=30, color='#E41A1C', marker='x',
               label=f'set to 0 by lasso ({len(zero)})')
    for _, row in codons.iterrows():
        ax.annotate(row['term'], (row['ols_coefficient'], row['lasso_coefficient']), fontsize=6,
                    xytext=(2, 2), textcoords='offset points')
    lim = np.abs(codons[['ols_coefficient', 'lasso_coefficient']].to_numpy()).max() * 1.05
    ax.plot([-lim, lim], [-lim, lim], color='gray', linestyle='--', linewidth=1, label='y = x')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('OLS coefficient')
    ax.set_ylabel('Lasso coefficient')
    ax.set_title('Codon coefficients, OLS vs lasso\n(Bartel HCT TE, raw codon-frequency scale)', fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.text(0.97, 0.05, f"r = {r:.3f}\nrho = {rho:.3f}\nshrinkage slope = {slope:.2f}", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIGS, "ols_vs_lasso_coefficients.png"), dpi=150)
    plt.close(fig)
    return zero


def r2(y, pred):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def compare_fit(scores):
    freq = pd.read_csv(os.path.join(OLS_TABLES, "codon_frequencies_human_HCT_Bartel.csv"), low_memory=False)
    train = scores[scores['chromosome'].isin(AUTOSOMES) & scores['TE_mean'].notna()].merge(
        freq[['ensembl_gene_id'] + SENSE_CODONS], on='ensembl_gene_id')
    X = train[SENSE_CODONS].to_numpy()
    y = train['TE_mean'].to_numpy()
    sums = freq[SENSE_CODONS].sum(axis=1)
    print(f"\n[2] Fit on {len(train):,} autosomal training genes")
    print(f"  sum of the 61 sense-codon frequencies per gene: mean {sums.mean():.6f}, SD {sums.std():.6f}, "
          f"min {sums.min():.4f} ({(sums > 0.9999).mean():.1%} of genes have sum = 1)")
    rows = {}
    for name, col in [('OLS', 'CO_Mega_ols'), ('lasso', 'CO_Mega_lasso')]:
        rows[name] = {'model': name, 'in_sample_R2': r2(y, train[col].to_numpy())}
    # 10-fold cross-validation; lasso penalty re-chosen (5-fold) inside every training fold
    makers = {'OLS': lambda: LinearRegression(),
              'lasso': lambda: make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=SEED, max_iter=100000))}
    oof = {k: np.zeros(len(y)) for k in makers}
    fold_r2 = {k: [] for k in makers}
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X):
        for k, mk in makers.items():
            m = mk().fit(X[tr], y[tr])
            oof[k][te] = m.predict(X[te])
            fold_r2[k].append(r2(y[te], oof[k][te]))
    for k in makers:
        rows[k]['cv_R2_pooled'] = r2(y, oof[k])
        rows[k]['cv_R2_fold_mean'] = np.mean(fold_r2[k])
        rows[k]['cv_R2_fold_sd'] = np.std(fold_r2[k], ddof=1)
        rows[k]['cv_pearson_r'] = stats.pearsonr(y, oof[k])[0]
        rows[k]['cv_spearman_rho'] = stats.spearmanr(y, oof[k])[0]
        rows[k]['cv_rmse'] = np.sqrt(np.mean((y - oof[k]) ** 2))
    diffs = np.array(fold_r2['OLS']) - np.array(fold_r2['lasso'])
    t_p = stats.ttest_rel(fold_r2['OLS'], fold_r2['lasso']).pvalue
    fit = pd.DataFrame(rows.values())
    print(fit.round(4).to_string(index=False))
    print(f"  per-fold CV R^2, OLS minus lasso: mean {diffs.mean():+.5f} (sd {diffs.std(ddof=1):.5f}; "
          f"paired t-test p = {t_p:.3g}; lasso better in {int((diffs < 0).sum())} of {N_FOLDS} folds)")
    fit['cv_R2_ols_minus_lasso_mean'] = diffs.mean()
    fit['cv_R2_ols_minus_lasso_paired_t_p'] = t_p
    fit.to_csv(os.path.join(OUT_TABLES, "ols_vs_lasso_fit_train_and_cv.csv"), index=False)


def compare_heldout(scores):
    print("\n[3] CO_Mega vs TE by gene set (trained on autosomes only)")
    sets = [('Autosomal (training)', scores['chromosome'].isin(AUTOSOMES)), ('X-linked (held out)', scores['chromosome'] == 'X'),
            ('Y-linked (held out)', scores['chromosome'] == 'Y'), ('All genes', scores['chromosome'].notna())]
    rows = []
    for label, mask in sets:
        sub = scores[mask & scores['TE_mean'].notna()]
        if len(sub) < 3:
            continue
        row = {'gene_set': label, 'n': len(sub)}
        for name, col in [('ols', 'CO_Mega_ols'), ('lasso', 'CO_Mega_lasso')]:
            row[f'pearson_r_{name}'] = stats.pearsonr(sub[col], sub['TE_mean'])[0]
            row[f'spearman_rho_{name}'] = stats.spearmanr(sub[col], sub['TE_mean'])[0]
            row[f'R2_{name}'] = r2(sub['TE_mean'].to_numpy(), sub[col].to_numpy())
        rows.append(row)
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    out.to_csv(os.path.join(OUT_TABLES, "ols_vs_lasso_correlation_by_gene_set.csv"), index=False)

    print("\n[4] Per-gene agreement of the two CO_Mega scores")
    r, _ = stats.pearsonr(scores['CO_Mega_ols'], scores['CO_Mega_lasso'])
    rho, _ = stats.spearmanr(scores['CO_Mega_ols'], scores['CO_Mega_lasso'])
    print(f"  {len(scores):,} genes: Pearson r = {r:.5f}, Spearman rho = {rho:.5f}; "
          f"SD of OLS CO_Mega = {scores['CO_Mega_ols'].std():.4f}, lasso = {scores['CO_Mega_lasso'].std():.4f}")
    pd.DataFrame([{'n_genes': len(scores), 'pearson_r': r, 'spearman_rho': rho,
                   'sd_ols': scores['CO_Mega_ols'].std(), 'sd_lasso': scores['CO_Mega_lasso'].std()}]).to_csv(
        os.path.join(OUT_TABLES, "ols_vs_lasso_per_gene_score_agreement.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(scores['CO_Mega_ols'], scores['CO_Mega_lasso'], s=4, alpha=0.25, color='#4D4D4D', edgecolor='none')
    lims = [min(scores['CO_Mega_ols'].min(), scores['CO_Mega_lasso'].min()),
            max(scores['CO_Mega_ols'].max(), scores['CO_Mega_lasso'].max())]
    ax.plot(lims, lims, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('CO_Mega, OLS')
    ax.set_ylabel('CO_Mega, lasso')
    ax.set_title('Per-gene CO_Mega: OLS vs lasso\n(Bartel HCT TE)', fontweight='bold')
    ax.text(0.05, 0.95, f"r = {r:.4f}\nn = {len(scores):,}", transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIGS, "ols_vs_lasso_per_gene_co_mega.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_TABLES, exist_ok=True)
    os.makedirs(OUT_FIGS, exist_ok=True)
    print("=== OLS vs lasso CO_Mega model, Bartel HCT TE ===")
    coef, alpha, scores = load_models()
    compare_coefficients(coef, alpha)
    compare_fit(scores)
    compare_heldout(scores)
    print("\n=== Done ===")


if __name__ == '__main__':
    main()
