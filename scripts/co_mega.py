#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
CO_Mega: a single-variable summary of a gene's codon composition's
predicted contribution to translation efficiency (TE).

Mirrors the R `MegaFeatures()` procedure used to build FE_Mega / FP_Mega /
CO_Mega in ModelConstruction20211023Current.ipynb, applied to the
codon-frequency features:

  1. Per-gene codon frequencies are NOT computed in that notebook -- they are
     loaded directly from a precomputed table (each of the 64 codons' share
     of the gene's total codon count; rows sum to ~1).
  2. The 3 stop codons (TAA, TAG, TGA) are dropped, leaving 61 sense-codon
     frequencies as predictors.
  3. A single OLS model is fit once: TE ~ codon_freq_1 + ... + codon_freq_61,
     across a reference gene set with known TE.
  4. For every gene, CO_Mega = intercept + sum_i(coef_i * codon_freq_i), i.e.
     the fitted value from that mini regression -- one number per gene that
     summarizes how much its codon composition is predicted to contribute to
     TE, using the codon weights learned in step 3.

In the notebook's full multi-feature model, CO_Mega has a large, positive,
and highly significant coefficient in every tissue/cell type examined (e.g.
cortical culture: coef = +0.723, p < 2e-16), so genes whose codon
composition is predicted (by the mini-model) to have higher TE do indeed
have higher observed TE -- CO_Mega contributes positively to TE.

This script implements steps 3-4 above (fitting the mini-model and computing
CO_Mega from codon frequencies). Turning a CDS sequence into codon
frequencies is not yet implemented here.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

STOP_CODONS = ('TAA', 'TAG', 'TGA')

_BASES = 'TCAG'
ALL_CODONS = [b1 + b2 + b3 for b1 in _BASES for b2 in _BASES for b3 in _BASES]
SENSE_CODONS = [c for c in ALL_CODONS if c not in STOP_CODONS]  # 61 codons


def fit_co_mega_model(codon_freq_df, te, codons=SENSE_CODONS):
    """
    Fit the codon-usage mini-model: TE ~ codon frequencies (OLS), matching
    the notebook's `MegaFeatures()` applied to *_CO (e.g. CORCULT_CO).

    codon_freq_df : DataFrame, one row per gene, one column per codon in
                    `codons` (that codon's fraction of the gene's total
                    codon count).
    te            : array-like of TE values, same length/order as
                    codon_freq_df.
    codons        : which codon columns to use as predictors (default: all
                    61 sense codons, stop codons excluded as in the
                    notebook).

    Returns a pandas Series of fitted coefficients, indexed by
    ['intercept'] + codons.
    """
    X = sm.add_constant(codon_freq_df[list(codons)].values)
    model = sm.OLS(np.asarray(te, dtype=float), X, missing='raise').fit()
    return pd.Series(model.params, index=['intercept'] + list(codons))


def compute_co_mega(codon_freq_df, coef, codons=SENSE_CODONS):
    """
    Apply a fitted mini-model (from `fit_co_mega_model`) to per-gene codon
    frequencies to get CO_Mega for each gene:

        CO_Mega = intercept + sum_i(coef_i * codon_freq_i)

    codon_freq_df : DataFrame, one row per gene, one column per codon.
    coef          : Series/dict of fitted coefficients, indexed/keyed by
                    'intercept' plus each codon in `codons`.
    codons        : codons used as predictors (must match what `coef` was
                    fit with).

    Returns a pandas Series of CO_Mega values, indexed like codon_freq_df.
    """
    weights = np.array([coef[c] for c in codons])
    co_mega = codon_freq_df[list(codons)].to_numpy() @ weights + coef['intercept']
    return pd.Series(co_mega, index=codon_freq_df.index, name='CO_Mega')


if __name__ == '__main__':
    # Small self-test on synthetic data: random codon frequencies (each
    # gene's 61 sense-codon frequencies drawn from a Dirichlet, so they sum
    # to 1 per gene) and TE values with a known linear relationship to two
    # of the codons, to check that fitting + applying the model recovers it.
    rng = np.random.default_rng(0)
    n_genes = 200
    freqs = rng.dirichlet(np.ones(len(SENSE_CODONS)), size=n_genes)
    freq_df = pd.DataFrame(freqs, columns=SENSE_CODONS)

    true_coef = pd.Series(0.0, index=['intercept'] + SENSE_CODONS)
    true_coef['intercept'] = 1.0
    true_coef['GCC'] = 5.0
    true_coef['ATA'] = -5.0
    te = (sm.add_constant(freq_df.to_numpy()) @ true_coef.to_numpy()
          + rng.normal(scale=0.05, size=n_genes))

    fitted_coef = fit_co_mega_model(freq_df, te)
    co_mega = compute_co_mega(freq_df, fitted_coef)

    print("Recovered intercept:", round(fitted_coef['intercept'], 3), "(true 1.0)")
    print("Recovered GCC coef :", round(fitted_coef['GCC'], 3), "(true 5.0)")
    print("Recovered ATA coef :", round(fitted_coef['ATA'], 3), "(true -5.0)")
    print("corr(CO_Mega, TE)  :", round(np.corrcoef(co_mega, te)[0, 1], 4))
