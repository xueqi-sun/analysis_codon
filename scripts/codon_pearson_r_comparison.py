#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Compare our own species/tissues' per-codon Pearson r (codon frequency vs.
TE) against each other, one point per codon, in the same scatter format
used to compare Tim's mouse data against these same tables
(mouse_codon_Tim_reconstruction.py).

Loads:
  analysis_codon/tables/co_mega_codon_pearson_r_human_liver.csv
  analysis_codon/tables/co_mega_codon_pearson_r_chicken_liver.csv
  analysis_codon/tables/co_mega_codon_pearson_r_mouse.csv
(all produced by human_co_mega_analysis.py / chicken_co_mega_analysis.py /
mouse_codon_frequency_comparison.py, via
co_mega_common.bootstrap_codon_pearson_r).

Saves, for every pair of species/tissues:
  analysis_codon/figures/codon_pearson_r_human_liver_vs_chicken_liver_bootstrap.png
  analysis_codon/figures/codon_pearson_r_human_liver_vs_mouse_liver_bootstrap.png
  analysis_codon/figures/codon_pearson_r_chicken_liver_vs_mouse_liver_bootstrap.png
"""

import os
from itertools import combinations

import pandas as pd
from scipy import stats

from co_mega_common import plot_pearson_r_comparison_scatter

BASE_DIR = "/lab/solexa_page/xueqi/analysis_codon"
PEARSON_R_FILES = {
    'Human liver':   os.path.join(BASE_DIR, "tables/co_mega_codon_pearson_r_human_liver.csv"),
    'Chicken liver': os.path.join(BASE_DIR, "tables/co_mega_codon_pearson_r_chicken_liver.csv"),
    'Mouse liver':   os.path.join(BASE_DIR, "tables/co_mega_codon_pearson_r_mouse.csv"),
}


def main():
    for y_label, x_label in combinations(PEARSON_R_FILES, 2):
        y_df = pd.read_csv(PEARSON_R_FILES[y_label])[['codon', 'pearson_r']].rename(columns={'pearson_r': 'pearson_r_y'})
        x_df = pd.read_csv(PEARSON_R_FILES[x_label])[['codon', 'pearson_r']].rename(columns={'pearson_r': 'pearson_r_x'})
        cmp_df = y_df.merge(x_df, on='codon')

        r_corr, p_corr = stats.pearsonr(cmp_df['pearson_r_x'], cmp_df['pearson_r_y'])
        rho_corr, p_rho = stats.spearmanr(cmp_df['pearson_r_x'], cmp_df['pearson_r_y'])
        print(f"{y_label} vs. {x_label} (n={len(cmp_df)} codons): "
              f"Pearson R = {r_corr:.4f} (p = {p_corr:.3g}), Spearman rho = {rho_corr:.4f} (p = {p_rho:.3g})")

        y_slug = y_label.lower().replace(' ', '_')
        x_slug = x_label.lower().replace(' ', '_')
        out_fig = os.path.join(BASE_DIR, f"figures/codon_pearson_r_{y_slug}_vs_{x_slug}_bootstrap.png")
        plot_pearson_r_comparison_scatter(
            cmp_df, y_label, x_label, r_corr, p_corr, rho_corr, p_rho, out_fig
        )


if __name__ == '__main__':
    main()
