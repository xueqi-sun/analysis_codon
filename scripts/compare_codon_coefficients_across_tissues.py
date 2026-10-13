#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Compare the codon regression coefficients (TE ~ 61 codon frequencies, OLS, autosomal genes) between human datasets:
  - liver_Wang        the original human liver data              (tables/human/)
  - HCT_Bartel        the Bartel HCT TE                          (tables/human_HCT_Bartel/)
  - <tissue>_ribonn   each tissue of the ribonn compendium       (tables/ribonn/<tissue>/)
(the circ fibroblast data set is not included.)

Outputs (tables/coefficient_comparison/, figures/coefficient_comparison/):
  - the 61 x 12 coefficient tables, raw / centered / normalized, and the dataset x dataset correlation matrices;
  - hierarchical-clustering heatmaps of the 61 codons across the datasets (rows = codons, columns = datasets; both
    clustered, average linkage on Euclidean distance), for
        raw coefficients,
        normalized coefficients (z-score across the 61 codons within each dataset),
        and, as a third view, centered coefficients (raw minus each dataset's mean over codons, scale kept);
    a colored strip marks codons ending in A/T (blue) or G/C (yellow);
  - a clustered heatmap of the dataset x dataset Pearson correlation of the normalized coefficients.

NOTE on 'raw': the 61 sense-codon frequencies of a gene sum to ~1, so together with the intercept the OLS coefficients
are only identified up to a common shift per dataset (add c to all coefficients and subtract c from the intercept
without changing any prediction). The raw heatmap therefore mostly shows each dataset's arbitrary offset and the
column clustering of the raw values reflects that; the differences between codons are in the centered and normalized
views (normalizing also removes differences in overall scale, i.e. in how strongly the codons predict TE).
"""

import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from co_mega import SENSE_CODONS

BASE_DIR = "/lab/solexa_page/xueqi/analysis_codon"
TABLE_DIR = os.path.join(BASE_DIR, "tables", "coefficient_comparison")
FIG_DIR = os.path.join(BASE_DIR, "figures", "coefficient_comparison")
RIBONN_TISSUES = ['HCT', 'HeLa', 'LCL', 'epithelial', 'cardiac_fibroblast', 'fibroblast', 'muscle', 'brain', 'neurons',
                  'skeletal_muscle']
AT_COLOR, GC_COLOR = '#0019FC', '#FFFF00'
GROUP_COLORS = {'original (Wang)': '#4D4D4D', 'Bartel': '#E69F00', 'ribonn': '#56B4E9'}


def dataset_files():
    files = {'liver_Wang': os.path.join(BASE_DIR, "tables/human/co_mega_model_coefficients_human_liver.csv"),
             'HCT_Bartel': os.path.join(BASE_DIR, "tables/human_HCT_Bartel/co_mega_model_coefficients_human_HCT_Bartel.csv")}
    for t in RIBONN_TISSUES:
        files[f'{t}_ribonn'] = os.path.join(BASE_DIR, f"tables/ribonn/{t}/co_mega_model_coefficients_ribonn_{t}.csv")
    return files


def load_coefficients():
    cols = {}
    for label, path in dataset_files().items():
        df = pd.read_csv(path).set_index('term')
        cols[label] = df.loc[SENSE_CODONS, 'coefficient']
    return pd.DataFrame(cols)


def group_of(label):
    return 'original (Wang)' if label == 'liver_Wang' else 'Bartel' if label == 'HCT_Bartel' else 'ribonn'


def clustermap(mat, title, out_file, cmap, center=None, cbar_label='coefficient', vmin=None, vmax=None):
    row_colors = pd.Series({c: AT_COLOR if c[-1] in 'AT' else GC_COLOR for c in mat.index}, name='third base')
    col_colors = pd.Series({d: GROUP_COLORS[group_of(d)] for d in mat.columns}, name='data set')
    # explicit linkage so the dendrograms are exactly average-linkage / Euclidean on the plotted values
    row_link = linkage(pdist(mat.to_numpy(), 'euclidean'), 'average')
    col_link = linkage(pdist(mat.to_numpy().T, 'euclidean'), 'average')
    g = sns.clustermap(mat, row_linkage=row_link, col_linkage=col_link, row_colors=row_colors, col_colors=col_colors,
                       cmap=cmap, center=center, vmin=vmin, vmax=vmax, linewidths=0.3, linecolor='white',
                       xticklabels=True, yticklabels=True, figsize=(9.5, 15),
                       dendrogram_ratio=(0.12, 0.08), colors_ratio=(0.03, 0.02),
                       cbar_pos=(0.02, 0.88, 0.02, 0.09), cbar_kws={'label': cbar_label})
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=60, ha='right', fontsize=9)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
    g.ax_heatmap.set_xlabel('')
    g.ax_row_colors.set_xticklabels([])  # the strip is explained by the legend; its label overlaps the first column label
    g.ax_heatmap.set_ylabel('Codon')
    handles = ([Patch(facecolor=AT_COLOR, edgecolor='gray', label='codon ends in A/T'),
                Patch(facecolor=GC_COLOR, edgecolor='gray', label='codon ends in G/C')] +
               [Patch(facecolor=c, edgecolor='gray', label=k) for k, c in GROUP_COLORS.items()])
    g.ax_heatmap.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.12, -0.12), fontsize=8, frameon=False)
    g.fig.suptitle(title, fontsize=11, fontweight='bold', y=1.0)
    g.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(g.fig)
    print(f"  Saved: {out_file}")
    return g


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=== Codon regression coefficients across human data sets ===")
    raw = load_coefficients()
    print(f"  {raw.shape[0]} codons x {raw.shape[1]} data sets: {', '.join(raw.columns)}")
    centered = raw - raw.mean(axis=0)
    normalized = centered / raw.std(axis=0, ddof=1)
    raw.to_csv(os.path.join(TABLE_DIR, "codon_coefficients_raw.csv"))
    centered.to_csv(os.path.join(TABLE_DIR, "codon_coefficients_centered.csv"))
    normalized.to_csv(os.path.join(TABLE_DIR, "codon_coefficients_normalized.csv"))
    print("  mean / SD of the 61 coefficients per data set (raw):")
    print(pd.DataFrame({'mean': raw.mean(), 'sd': raw.std(ddof=1)}).round(3).to_string())

    pearson = normalized.corr(method='pearson')
    spearman = raw.corr(method='spearman')
    pearson.to_csv(os.path.join(TABLE_DIR, "codon_coefficients_dataset_correlation_pearson.csv"))
    spearman.to_csv(os.path.join(TABLE_DIR, "codon_coefficients_dataset_correlation_spearman.csv"))
    off = pearson.where(~np.eye(len(pearson), dtype=bool))
    print(f"\n  Pearson r between data sets (coefficient vectors): median {np.nanmedian(off.to_numpy()):.3f}, "
          f"range {np.nanmin(off.to_numpy()):.3f} to {np.nanmax(off.to_numpy()):.3f}")
    print("  correlation of each data set with the others (mean r):")
    print(off.mean().sort_values(ascending=False).round(3).to_string())

    print("\n  Heatmaps...")
    vmax_n = np.nanpercentile(np.abs(normalized.to_numpy()), 99)
    clustermap(raw, 'Codon regression coefficients across human data sets: RAW coefficients\n'
                    '(hierarchical clustering, average linkage, Euclidean)',
               os.path.join(FIG_DIR, "codon_coefficients_heatmap_raw.png"), cmap='viridis',
               cbar_label='raw coefficient')
    vmax_c = np.nanpercentile(np.abs(centered.to_numpy()), 99)
    clustermap(centered, 'Codon regression coefficients across human data sets: CENTERED coefficients\n'
                         '(raw minus the data set mean over codons; hierarchical clustering, average linkage, Euclidean)',
               os.path.join(FIG_DIR, "codon_coefficients_heatmap_centered.png"), cmap='RdBu_r', center=0,
               vmin=-vmax_c, vmax=vmax_c, cbar_label='centered coefficient')
    clustermap(normalized, 'Codon regression coefficients across human data sets: NORMALIZED coefficients\n'
                           '(z-score over the 61 codons within each data set; hierarchical clustering, average linkage, Euclidean)',
               os.path.join(FIG_DIR, "codon_coefficients_heatmap_normalized.png"), cmap='RdBu_r', center=0,
               vmin=-vmax_n, vmax=vmax_n, cbar_label='z-scored coefficient')

    # dataset x dataset correlation of the (normalized) coefficient vectors
    col_colors = pd.Series({d: GROUP_COLORS[group_of(d)] for d in pearson.columns}, name='data set')
    dist = 1 - pearson.to_numpy()
    link = linkage(dist[np.triu_indices_from(dist, k=1)], 'average')
    g = sns.clustermap(pearson, row_linkage=link, col_linkage=link, row_colors=col_colors, col_colors=col_colors,
                       cmap='RdBu_r', center=0, vmin=-1, vmax=1, annot=True, fmt='.2f', annot_kws={'size': 6},
                       linewidths=0.3, linecolor='white', figsize=(10, 10), cbar_kws={'label': 'Pearson r'})
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=60, ha='right', fontsize=9)
    g.fig.suptitle('Similarity of codon coefficients between data sets\n(Pearson r of the 61 coefficients; clustering on 1 - r, average linkage)',
                   fontsize=11, fontweight='bold', y=1.0)
    out = os.path.join(FIG_DIR, "codon_coefficients_dataset_correlation_heatmap.png")
    g.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(g.fig)
    print(f"  Saved: {out}")
    print("\n=== Done ===")


if __name__ == '__main__':
    main()
