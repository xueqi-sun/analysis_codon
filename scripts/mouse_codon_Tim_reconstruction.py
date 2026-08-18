#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Reconstruct a codon-usage -> TE model from Tim's own data (mouse cortical
culture, and/or mouse liver) and compare its per-codon Pearson r's with the
ones obtained from our own BioMart-derived CO_Mega models (human liver,
chicken liver, mouse liver).

Pipeline:
  1. Load Tim's codon frequencies from
     analysis_codon/data/LightStimPalAnnotations20211019_CDS_CodonFrequencies.csv
     (same as mouse_codon_frequency_comparison.py: each row's "Gene_name" is
     "<old RefSeq mRNA ID>::<locus>") and convert the old RefSeq IDs to new
     Ensembl gene IDs via
     analysis_codon/data/refseq_to_ensemble_mouse_mart_export20200909.csv
     (reusing mouse_codon_frequency_comparison.load_tim_codon_frequencies /
     convert_old_to_new_id).
  2. Load one of Tim's TE tables (e.g.
     analysis_codon/data/Tim_data_TE_mouse_corticalculture.csv or
     Tim_data_TE_mouse_liver.csv; first column is the -- already
     Ensembl-style -- gene ID; "log10.TR" is the TE value). These tables
     also have their own codon-frequency columns, which are NOT used here.
  3. Intersect: keep genes with both a converted codon-frequency row (step
     1) and a TE value (step 2).
  4. Fit TE ~ 61 sense-codon frequencies (OLS) on all intersected genes (no
     autosome/X split -- these tables don't carry chromosome info), save
     the coefficients, and compute each codon's Pearson r with TE two ways:
       - bootstrap: point estimate + bootstrap SD (as in
         co_mega_codon_pearson_r_barplot_mouse.png), suffixed `_bootstrap`.
       - ordinary: the same point estimate + an analytic Fisher-z
         confidence interval (no resampling), suffixed `_ordinary`.
  5. For each of our own CO_Mega fits (human liver, chicken liver, mouse
     liver; analysis_codon/tables/co_mega_codon_pearson_r_*.csv), scatter
     that fit's per-codon Pearson r against Tim's (both the bootstrap and
     ordinary versions), with a y = x reference line, and report how well
     the two agree (Pearson R, Spearman rho).

This is run per Tim TE table (see `main()` -- currently only mouse liver is
active; the mouse cortical-culture run is commented out). All outputs are
saved under a dedicated `testing_Tim` subfolder of `tables/` and `figures/`,
suffixed `_Tim_mouse_corticalculture` or `_Tim_mouse_liver` (+ `_bootstrap` /
`_ordinary` for the per-codon-r-dependent outputs).
"""

import os

import pandas as pd
from scipy import stats

from co_mega import SENSE_CODONS
from co_mega_common import (
    fit_co_mega_te_model, save_coefficient_table, bootstrap_codon_pearson_r,
    plot_codon_pearson_r_barplot, ordinary_codon_pearson_r, plot_codon_pearson_r_barplot_ci,
    plot_pearson_r_comparison_scatter,
)
from mouse_codon_frequency_comparison import load_tim_codon_frequencies, convert_old_to_new_id

BASE_DIR       = "/lab/solexa_page/xueqi/analysis_codon"
TIM_CODON_FILE = os.path.join(BASE_DIR, "data/LightStimPalAnnotations20211019_CDS_CodonFrequencies.csv")
MART_FILE      = os.path.join(BASE_DIR, "data/refseq_to_ensemble_mouse_mart_export20200909.csv")
TABLE_DIR      = os.path.join(BASE_DIR, "tables/testing_Tim")
FIG_DIR        = os.path.join(BASE_DIR, "figures/testing_Tim")

# -- Tim's mouse cortical-culture TE table (disabled; see main()) --
# TIM_TE_FILE_CORTICAL = os.path.join(BASE_DIR, "data/Tim_data_TE_mouse_corticalculture.csv")
# SUFFIX_CORTICAL       = "Tim_mouse_corticalculture"
# TISSUE_LABEL_CORTICAL = "Tim's mouse cortical culture"

# -- Tim's mouse liver TE table (active) --
TIM_TE_FILE_LIVER = os.path.join(BASE_DIR, "data/Tim_data_TE_mouse_liver.csv")
SUFFIX_LIVER       = "Tim_mouse_liver"
TISSUE_LABEL_LIVER = "Tim's mouse liver"

# Other species/tissues' per-codon Pearson r tables to compare against.
OTHER_PEARSON_R_FILES = {
    'Human liver':   os.path.join(BASE_DIR, "tables/co_mega_codon_pearson_r_human_liver.csv"),
    'Chicken liver': os.path.join(BASE_DIR, "tables/co_mega_codon_pearson_r_chicken_liver.csv"),
    'Mouse liver':   os.path.join(BASE_DIR, "tables/co_mega_codon_pearson_r_mouse.csv"),
}


def tbl(name):
    return os.path.join(TABLE_DIR, f"{name}.csv")


def fig(name):
    return os.path.join(FIG_DIR, f"{name}.png")


# ── Step 2: load one of Tim's TE tables ─────────────────────────────────────
def load_tim_te_table(te_file):
    """
    Load one of Tim's TE tables, keeping only the (unnamed, first) gene-ID
    column and "log10.TR" (the TE value). The table's own codon-frequency
    columns are intentionally ignored.
    """
    df = pd.read_csv(te_file)
    id_col = df.columns[0]
    df = df[[id_col, 'log10.TR']].rename(columns={id_col: 'new_id', 'log10.TR': 'TE_mean'})
    df = df.dropna(subset=['TE_mean'])
    print(f"  Loaded {len(df):,} genes with a TE value from {te_file}")
    return df


# ── Steps 2-6, for a single Tim TE table ────────────────────────────────────
def run_te_analysis(tim_codon_df, te_file, suffix, tissue_label):
    print(f"\n=== {tissue_label}: codon frequency -> TE reconstruction ===")

    print(f"\n[2] Loading {tissue_label} TE data (log10.TR) from {te_file}...")
    te_df = load_tim_te_table(te_file)

    print("\n[3] Intersecting codon frequencies with TE data...")
    merged = tim_codon_df.merge(te_df, on='new_id', how='inner')
    print(f"  {len(merged):,} genes with both codon frequencies and TE.")

    print("\n[4] Fitting TE ~ codon frequencies (OLS) on all intersected genes...")
    coef, model = fit_co_mega_te_model(merged[SENSE_CODONS], merged['TE_mean'])
    print(f"  R^2 = {model.rsquared:.4f}   Adjusted R^2 = {model.rsquared_adj:.4f}   n = {int(model.nobs)}")
    out_coef = tbl(f"co_mega_model_coefficients_{suffix}")
    save_coefficient_table(coef, model, SENSE_CODONS, out_coef)

    print("\n[5a] Per-codon Pearson r with TE (bootstrap SD)...")
    r_df_boot = bootstrap_codon_pearson_r(merged[SENSE_CODONS], merged['TE_mean'], SENSE_CODONS)
    out_codon_r_boot = tbl(f"co_mega_codon_pearson_r_{suffix}_bootstrap")
    r_df_boot.to_csv(out_codon_r_boot, index=False)
    print(f"  Saved per-codon Pearson r (+ bootstrap SD) to {out_codon_r_boot}")
    plot_codon_pearson_r_barplot(
        r_df_boot, f'Pearson R between codon frequency and TE, per codon\n({tissue_label}, bootstrap SD)',
        fig(f"co_mega_codon_pearson_r_barplot_{suffix}_bootstrap")
    )

    print("\n[5b] Per-codon Pearson r with TE (ordinary, analytic Fisher-z CI)...")
    r_df_ord = ordinary_codon_pearson_r(merged[SENSE_CODONS], merged['TE_mean'], SENSE_CODONS)
    out_codon_r_ord = tbl(f"co_mega_codon_pearson_r_{suffix}_ordinary")
    r_df_ord.to_csv(out_codon_r_ord, index=False)
    print(f"  Saved per-codon Pearson r (+ 95% Fisher-z CI) to {out_codon_r_ord}")
    plot_codon_pearson_r_barplot_ci(
        r_df_ord, f'Pearson R between codon frequency and TE, per codon\n({tissue_label}, ordinary, 95% CI)',
        fig(f"co_mega_codon_pearson_r_barplot_{suffix}_ordinary")
    )

    print(f"\n[6] Comparing {tissue_label}'s per-codon Pearson r with other species/tissues...")
    for variant, r_df, kwargs_fn in [
        ('bootstrap', r_df_boot, lambda cmp_df: {}),
        ('ordinary', r_df_ord, lambda cmp_df: {
            'yerr': [
                (cmp_df['pearson_r_y'] - cmp_df['ci_lower']).to_numpy(),
                (cmp_df['ci_upper'] - cmp_df['pearson_r_y']).to_numpy(),
            ],
            'error_label': '95% Fisher-z CI (Tim only)',
        }),
    ]:
        print(f"\n  --- {variant} ---")
        cols = ['codon', 'pearson_r'] + (['ci_lower', 'ci_upper'] if variant == 'ordinary' else [])
        summary_rows = []
        for label, path in OTHER_PEARSON_R_FILES.items():
            other_df = pd.read_csv(path)[['codon', 'pearson_r']].rename(columns={'pearson_r': 'pearson_r_x'})
            cmp_df = (r_df[cols]
                      .rename(columns={'pearson_r': 'pearson_r_y'})
                      .merge(other_df, on='codon'))

            r_corr, p_corr = stats.pearsonr(cmp_df['pearson_r_x'], cmp_df['pearson_r_y'])
            rho_corr, p_rho = stats.spearmanr(cmp_df['pearson_r_x'], cmp_df['pearson_r_y'])
            print(f"  {tissue_label} vs. {label} (n={len(cmp_df)} codons): "
                  f"Pearson R = {r_corr:.4f} (p = {p_corr:.3g}), Spearman rho = {rho_corr:.4f} (p = {p_rho:.3g})")
            summary_rows.append({
                'comparison': f'{tissue_label} vs. {label}', 'n_codons': len(cmp_df),
                'pearson_r': r_corr, 'pearson_p_value': p_corr,
                'spearman_rho': rho_corr, 'spearman_p_value': p_rho,
            })

            slug = label.lower().replace(' ', '_')
            plot_pearson_r_comparison_scatter(
                cmp_df, tissue_label, label, r_corr, p_corr, rho_corr, p_rho,
                fig(f"codon_pearson_r_{suffix}_vs_{slug}_{variant}"),
                **kwargs_fn(cmp_df)
            )

        out_summary = tbl(f"codon_pearson_r_comparison_summary_{suffix}_{variant}")
        pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
        print(f"  Saved comparison summary (Pearson/Spearman R, p) to {out_summary}")


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("\n[1] Loading Tim's codon frequencies and converting gene IDs...")
    tim_codon_df = load_tim_codon_frequencies(TIM_CODON_FILE)
    tim_codon_df = convert_old_to_new_id(tim_codon_df, MART_FILE)

    # -- Tim's mouse cortical-culture TE data (disabled) --
    # run_te_analysis(tim_codon_df, TIM_TE_FILE_CORTICAL, SUFFIX_CORTICAL, TISSUE_LABEL_CORTICAL)

    # -- Tim's mouse liver TE data (active) --
    run_te_analysis(tim_codon_df, TIM_TE_FILE_LIVER, SUFFIX_LIVER, TISSUE_LABEL_LIVER)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
