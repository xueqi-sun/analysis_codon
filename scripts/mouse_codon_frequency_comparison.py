#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Compare mouse codon frequencies computed from Ensembl BioMart CDS sequences
against Tim's precomputed codon frequencies.

Pipeline:
  1. Parse mouse CDS sequences from analysis_codon/data/mouse_CDS_sequence.txt
     (FASTA, same header format as chicken_co_mega_analysis.py:
     ">gene_id|gene_id.version|transcript_id|transcript_id.version|gene_name|chromosome")
     and save as analysis_codon/data/mouse_CDS_sequence.csv.
  2. For each gene, split the CDS into codons, drop the last codon (the stop
     codon), and compute each of the 61 sense codons' frequency, exactly as
     in chicken/human_co_mega_analysis.py.
  3. Load Tim's codon frequencies from
     analysis_codon/data/LightStimPalAnnotations20211019_CDS_CodonFrequencies.csv.
     Each row's "Gene_name" column looks like "<old RefSeq mRNA ID>::<locus>";
     the part before "::" is the old (RefSeq) ID.
  4. Convert old (RefSeq mRNA) IDs to new (Ensembl gene stable) IDs using
     analysis_codon/data/refseq_to_ensemble_mouse_mart_export20200909.csv
     ("RefSeq mRNA ID" <-> "Gene stable ID"). An old ID is kept only if it
     maps to exactly one distinct new ID in that table; among the old IDs
     that survive that filter, if more than one of Tim's rows ends up
     mapping to the same new ID (i.e. Tim's data has >1 transcript for one
     gene), all of those rows are also dropped. Counts of each kind of
     dropped gene are reported.
  5. Merge BioMart-derived codon frequencies with Tim's (converted) codon
     frequencies on gene ID and compare the two: per-codon correlation and
     mean absolute difference, per-gene overall (summed absolute)
     difference across all 61 codons, and whether the mean per-codon usage
     ranking is preserved between the two sources. Save comparison tables
     and figures.
  6. Build the mouse CO_Mega model (ignoring Tim's data -- BioMart codon
     frequencies only): load per-gene TE_mean (mouse liver) from
     analysis_riboseq/tables/mouse_liver/mouse_te.csv, fit
     TE_mean ~ 61 codon frequencies (OLS) using autosomal (chr 1-19) genes,
     apply the fitted model to all genes (autosomes + X) to get CO_Mega,
     report its correlation with TE_mean, and compare CO_Mega between
     autosomal and X-linked genes -- mirroring
     chicken/human_co_mega_analysis.py. All CO_Mega outputs are suffixed
     `_mouse`.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from co_mega import SENSE_CODONS, compute_co_mega
from co_mega_common import (
    compute_codon_frequencies, parse_cds_fasta, precise_pvalue_str,
    fit_co_mega_te_model, save_coefficient_table, plot_actual_vs_predicted,
    plot_co_mega_boxplot, bootstrap_codon_pearson_r, plot_codon_pearson_r_barplot,
)

BASE_DIR  = "/lab/solexa_page/xueqi/analysis_codon"
CDS_FASTA = os.path.join(BASE_DIR, "data/mouse_CDS_sequence.txt")
CDS_CSV   = os.path.join(BASE_DIR, "data/mouse_CDS_sequence.csv")
TIM_FILE  = os.path.join(BASE_DIR, "data/LightStimPalAnnotations20211019_CDS_CodonFrequencies.csv")
MART_FILE = os.path.join(BASE_DIR, "data/refseq_to_ensemble_mouse_mart_export20200909.csv")
TE_FILE   = "/lab/solexa_page/xueqi/analysis_riboseq/tables/mouse_liver/mouse_te.csv"
TABLE_DIR = os.path.join(BASE_DIR, "tables")
FIG_DIR   = os.path.join(BASE_DIR, "figures")

AUTOSOMES = [str(i) for i in range(1, 20)]  # mouse autosomes: chr 1-19

SUFFIX = "mouse"
TISSUE_LABEL = "Mouse liver"


def tbl(name):
    return os.path.join(TABLE_DIR, f"{name}_{SUFFIX}.csv")


def fig(name):
    return os.path.join(FIG_DIR, f"{name}_{SUFFIX}.png")


# ── Step 2: BioMart codon frequencies ───────────────────────────────────────
def load_biomart_codon_frequencies(cds_csv):
    df = pd.read_csv(cds_csv, usecols=['gene_id', 'gene_name', 'chromosome', 'CDS_sequence'])
    print(f"  Loaded {len(df):,} genes from {cds_csv}")

    freq_records = df['CDS_sequence'].apply(compute_codon_frequencies)
    n_unusable = freq_records.isna().sum()
    df = df.loc[freq_records.notna()].copy()
    freq_df = pd.DataFrame(list(freq_records.dropna()), index=df.index)
    print(f"  Dropped {n_unusable:,} genes with missing/unusable CDS sequences "
          f"(not a multiple of 3, or too short); {len(df):,} genes remain.")

    return pd.concat([df[['gene_id', 'gene_name', 'chromosome']], freq_df], axis=1)


# ── Step 3-4: Tim's codon frequencies + RefSeq -> Ensembl ID conversion ─────
def load_tim_codon_frequencies(tim_file):
    df = pd.read_csv(tim_file)
    df['old_id'] = df['Gene_name'].str.split('::').str[0]
    print(f"  Loaded {len(df):,} genes from {tim_file} ({df['old_id'].nunique():,} unique old IDs)")
    return df


def convert_old_to_new_id(tim_df, mart_file):
    """
    Map Tim's old (RefSeq mRNA) IDs to new (Ensembl gene stable) IDs via
    `mart_file`'s "RefSeq mRNA ID" <-> "Gene stable ID" columns, keeping only
    unique 1:1 mappings (see module docstring for the exact dropping rules).
    Returns `tim_df` with a `new_id` column, restricted to the successfully
    (unambiguously) converted rows.
    """
    mart = pd.read_csv(mart_file)[['RefSeq mRNA ID', 'Gene stable ID']].dropna().drop_duplicates()
    n_new_per_old = mart.groupby('RefSeq mRNA ID')['Gene stable ID'].nunique()
    ambiguous_old_ids = set(n_new_per_old[n_new_per_old > 1].index)
    id_map = (mart[~mart['RefSeq mRNA ID'].isin(ambiguous_old_ids)]
              .drop_duplicates(subset=['RefSeq mRNA ID'])
              .set_index('RefSeq mRNA ID')['Gene stable ID'])

    tim_df = tim_df.copy()
    is_ambiguous = tim_df['old_id'].isin(ambiguous_old_ids)
    tim_df['new_id'] = tim_df['old_id'].map(id_map)
    is_not_found = tim_df['new_id'].isna() & ~is_ambiguous

    print(f"  {is_ambiguous.sum():,} of Tim's genes dropped: old ID maps to >1 distinct new gene ID.")
    print(f"  {is_not_found.sum():,} of Tim's genes dropped: old ID not found in the mapping table.")

    mapped = tim_df[tim_df['new_id'].notna()]
    dup_new_id = set(mapped.loc[mapped['new_id'].duplicated(keep=False), 'new_id'])
    n_collision_rows = mapped['new_id'].isin(dup_new_id).sum()
    print(f"  {n_collision_rows:,} of Tim's genes dropped: >1 of Tim's rows map to the same new "
          f"gene ID ({len(dup_new_id):,} gene IDs affected).")

    converted = mapped[~mapped['new_id'].isin(dup_new_id)]
    print(f"  {len(converted):,} of Tim's {len(tim_df):,} genes successfully converted to a unique new gene ID.")
    return converted


# ── Step 5: compare BioMart vs. Tim's codon frequencies ─────────────────────
def compare_frequencies(biomart_df, tim_df, codons=SENSE_CODONS):
    bio = biomart_df.rename(columns={c: f'{c}_biomart' for c in codons})
    tim = tim_df.rename(columns={c: f'{c}_tim' for c in codons})[['new_id'] + [f'{c}_tim' for c in codons]]
    merged = bio.merge(tim, left_on='gene_id', right_on='new_id', how='inner')
    print(f"  {len(merged):,} genes with both BioMart and Tim codon frequencies.")

    bio_vals = merged[[f'{c}_biomart' for c in codons]].to_numpy(dtype=float)
    tim_vals = merged[[f'{c}_tim' for c in codons]].to_numpy(dtype=float)
    diff = bio_vals - tim_vals

    n_genes = len(merged)
    per_codon = []
    for i, c in enumerate(codons):
        r, _ = stats.pearsonr(bio_vals[:, i], tim_vals[:, i])
        per_codon.append({
            'codon': c,
            'mean_freq_biomart': bio_vals[:, i].mean(),
            'mean_freq_tim': tim_vals[:, i].mean(),
            'pearson_r': r,
            'p_value': precise_pvalue_str(r, n_genes),
            'mean_abs_diff': np.abs(diff[:, i]).mean(),
            'rmse': np.sqrt((diff[:, i] ** 2).mean()),
        })
    per_codon_df = pd.DataFrame(per_codon)

    merged['l1_diff'] = np.abs(diff).sum(axis=1)
    merged['rmse_diff'] = np.sqrt((diff ** 2).mean(axis=1))
    per_gene_df = merged[['gene_id', 'gene_name', 'l1_diff', 'rmse_diff']].copy()

    r_overall, _ = stats.pearsonr(bio_vals.flatten(), tim_vals.flatten())
    p_overall_str = precise_pvalue_str(r_overall, bio_vals.size)

    return merged, per_codon_df, per_gene_df, r_overall, p_overall_str, bio_vals, tim_vals


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_overall_scatter(bio_vals, tim_vals, r_overall, p_overall_str, output_file):
    fig_, ax = plt.subplots(figsize=(7, 7))
    hb = ax.hexbin(bio_vals.flatten(), tim_vals.flatten(), gridsize=80, cmap='viridis',
                    bins='log', mincnt=1)
    lims = [0, max(bio_vals.max(), tim_vals.max()) * 1.05]
    ax.plot(lims, lims, color='red', linestyle='--', linewidth=1, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Codon frequency (BioMart CDS)')
    ax.set_ylabel("Codon frequency (Tim's data)")
    ax.set_title('Mouse codon frequency: BioMart vs. Tim\n(all genes x all 61 sense codons)', fontweight='bold')
    ax.annotate(f"R = {r_overall:.4f}\np = {p_overall_str}\nn = {bio_vals.size:,}",
                xy=(0.05, 0.95), xycoords='axes fraction', va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    ax.legend(loc='lower right', fontsize=8)
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('log10(count)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")


def plot_per_codon_r_barplot(per_codon_df, n_genes, output_file):
    df_sorted = per_codon_df.sort_values('pearson_r', ascending=False).reset_index(drop=True)
    fig_, ax = plt.subplots(figsize=(18, 6))
    colors = ['#4DAF4A' if r >= 0 else '#E41A1C' for r in df_sorted['pearson_r']]
    ax.bar(df_sorted['codon'], df_sorted['pearson_r'], color=colors, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Codon')
    ax.set_ylabel("Pearson R between BioMart and Tim's codon frequency")
    ax.set_title("Mouse: agreement between BioMart and Tim's codon frequencies, per codon", fontweight='bold')
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['codon'], rotation=90, fontsize=7)
    ax.annotate(f"n = {n_genes:,} genes", xy=(0.99, 0.98), xycoords='axes fraction',
                va='top', ha='right', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")


def plot_mean_codon_scatter(per_codon_df, output_file):
    r, p = stats.pearsonr(per_codon_df['mean_freq_biomart'], per_codon_df['mean_freq_tim'])
    rho, p_rho = stats.spearmanr(per_codon_df['mean_freq_biomart'], per_codon_df['mean_freq_tim'])

    fig_, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(per_codon_df['mean_freq_biomart'], per_codon_df['mean_freq_tim'], s=25, color='#377EB8')
    for _, row in per_codon_df.iterrows():
        ax.annotate(row['codon'], (row['mean_freq_biomart'], row['mean_freq_tim']), fontsize=6,
                    xytext=(2, 2), textcoords='offset points')
    lims = [0, max(per_codon_df['mean_freq_biomart'].max(), per_codon_df['mean_freq_tim'].max()) * 1.1]
    ax.plot(lims, lims, color='gray', linestyle='--', linewidth=1, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Mean codon frequency across genes (BioMart CDS)')
    ax.set_ylabel("Mean codon frequency across genes (Tim's data)")
    ax.set_title('Mouse: mean per-codon frequency, BioMart vs. Tim\n(rank preservation of codon-usage bias)',
                 fontweight='bold')
    ax.annotate(f"Pearson R = {r:.4f} (p = {p:.3g})\nSpearman rho = {rho:.4f} (p = {p_rho:.3g})\nn = {len(per_codon_df)} codons",
                xy=(0.05, 0.95), xycoords='axes fraction', va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")


def plot_per_gene_diff_hist(per_gene_df, output_file):
    mean_l1 = per_gene_df['l1_diff'].mean()
    median_l1 = per_gene_df['l1_diff'].median()

    fig_, ax = plt.subplots(figsize=(8, 6))
    ax.hist(per_gene_df['l1_diff'], bins=60, color='#377EB8', alpha=0.8, edgecolor='none')
    ax.axvline(mean_l1, color='red', linestyle='--', linewidth=1.2, label=f'mean = {mean_l1:.3f}')
    ax.axvline(median_l1, color='black', linestyle=':', linewidth=1.2, label=f'median = {median_l1:.3f}')
    ax.set_xlabel('Per-gene total |difference| summed across all 61 sense codons\n(sum_i |freq_biomart_i - freq_tim_i|)')
    ax.set_ylabel('Number of genes')
    ax.set_title('Mouse: per-gene codon-frequency difference (BioMart vs. Tim)', fontweight='bold')
    ax.annotate(f"n = {len(per_gene_df):,} genes", xy=(0.95, 0.72), xycoords='axes fraction',
                va='top', ha='right', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file}")


# ── Step 6: CO_Mega model (BioMart codon frequencies only, ignoring Tim's data) ──
def run_co_mega_te_analysis(biomart_df):
    """
    Build the mouse CO_Mega model from BioMart codon frequencies (Tim's data
    is not used here) and apply it to compare CO_Mega between autosomal and
    X-linked genes, mirroring chicken/human_co_mega_analysis.py.
    """
    biomart_df = biomart_df.copy()
    biomart_df['chromosome'] = biomart_df['chromosome'].astype(str)
    biomart_df['is_X'] = (biomart_df['chromosome'] == 'X').astype(int)

    print("\n[6] Loading mouse liver TE data...")
    te_df = pd.read_csv(TE_FILE)[['Gene_ID', 'TE']].rename(columns={'Gene_ID': 'gene_id', 'TE': 'TE_mean'})
    te_df = te_df.dropna(subset=['TE_mean'])
    print(f"  {len(te_df):,} genes with TE_mean in {TE_FILE}")

    with_te = biomart_df.merge(te_df, on='gene_id', how='inner')
    print(f"  {len(with_te):,} genes with both BioMart codon frequencies and TE_mean.")

    print("\n[7] Fitting TE_mean ~ codon frequencies (BioMart) on autosomal genes...")
    train = with_te[with_te['chromosome'].isin(AUTOSOMES)]
    print(f"  Training on {len(train):,} autosomal genes, {len(SENSE_CODONS)} codon predictors.")
    coef, model = fit_co_mega_te_model(train[SENSE_CODONS], train['TE_mean'])
    print(f"  R^2 = {model.rsquared:.4f}   Adjusted R^2 = {model.rsquared_adj:.4f}   n = {int(model.nobs)}")

    out_coef = tbl("co_mega_model_coefficients")
    save_coefficient_table(coef, model, SENSE_CODONS, out_coef)

    r_train, _ = stats.pearsonr(train['TE_mean'], model.fittedvalues)
    p_train_str = precise_pvalue_str(r_train, len(train))
    plot_actual_vs_predicted(
        train['TE_mean'], model.fittedvalues,
        'Actual TE_mean (autosomal training genes)', 'Fitted TE_mean (OLS model)',
        f'Original regression: TE ~ codon frequencies\n(autosomal training genes, {TISSUE_LABEL})',
        r_train, p_train_str, fig("co_mega_regression_fit_train")
    )

    print("\n[8] Computing CO_Mega for all genes (autosomes + X) and comparing to TE...")
    biomart_df['CO_Mega'] = compute_co_mega(biomart_df[SENSE_CODONS], coef)

    with_te = biomart_df.merge(te_df, on='gene_id', how='inner')
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
    out_corr = tbl("co_mega_te_correlation_stats")
    corr_stats.to_csv(out_corr, index=False)
    print(f"  Saved CO_Mega vs. TE correlation stats (n, Pearson r, p) to {out_corr}")

    plot_actual_vs_predicted(with_te['TE_mean'], with_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                              f'Actual TE vs. CO_Mega: All genes ({TISSUE_LABEL})', r_all, p_all_str,
                              fig("co_mega_vs_te_all_genes"))
    plot_actual_vs_predicted(auto_te['TE_mean'], auto_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                              f'Actual TE vs. CO_Mega: Autosomal genes ({TISSUE_LABEL})', r_auto, p_auto_str,
                              fig("co_mega_vs_te_autosomal"))
    if len(x_te) > 1:
        plot_actual_vs_predicted(x_te['TE_mean'], x_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                                  f'Actual TE vs. CO_Mega: X-linked genes ({TISSUE_LABEL})', r_x, p_x_str,
                                  fig("co_mega_vs_te_X_linked"))

    out_all = tbl("co_mega_all_genes")
    biomart_df[['gene_id', 'gene_name', 'chromosome', 'is_X', 'CO_Mega']].merge(
        te_df, on='gene_id', how='left'
    ).to_csv(out_all, index=False)
    print(f"  Saved CO_Mega for all {len(biomart_df):,} genes to {out_all}")

    print("\n[9] Boxplot: CO_Mega, autosomal vs X-linked genes...")
    plot_co_mega_boxplot(biomart_df, 'CO_Mega', 'is_X', 'X chromosome',
                          f'CO_Mega: X-linked vs Autosomal genes ({TISSUE_LABEL})',
                          fig("co_mega_boxplot_autosome_vs_X"))

    print("\n[10] Per-codon Pearson r with TE (bootstrap SD, autosomal training genes)...")
    r_df = bootstrap_codon_pearson_r(train[SENSE_CODONS], train['TE_mean'], SENSE_CODONS)
    out_codon_r = tbl("co_mega_codon_pearson_r")
    r_df.to_csv(out_codon_r, index=False)
    print(f"  Saved per-codon Pearson r (+ bootstrap SD) to {out_codon_r}")
    plot_codon_pearson_r_barplot(
        r_df, f'Pearson R between codon frequency and TE, per codon\n(autosomal training genes, {TISSUE_LABEL})',
        fig("co_mega_codon_pearson_r_barplot")
    )


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=== Mouse codon frequency comparison: BioMart CDS vs. Tim's data ===")

    print("\n[1] Parsing mouse CDS FASTA into a CSV table...")
    parse_cds_fasta(CDS_FASTA, CDS_CSV)

    print("\n[2] Computing codon frequencies from BioMart CDS sequences...")
    biomart_df = load_biomart_codon_frequencies(CDS_CSV)
    out_biomart = tbl("codon_frequencies_biomart")
    biomart_df.to_csv(out_biomart, index=False)
    print(f"  Saved BioMart codon frequencies to {out_biomart}")

    print("\n[3] Loading Tim's codon frequencies and converting gene IDs...")
    tim_df = load_tim_codon_frequencies(TIM_FILE)
    tim_df = convert_old_to_new_id(tim_df, MART_FILE)

    print("\n[4] Comparing BioMart vs. Tim's codon frequencies...")
    merged, per_codon_df, per_gene_df, r_overall, p_overall_str, bio_vals, tim_vals = \
        compare_frequencies(biomart_df, tim_df)

    out_per_codon = tbl("codon_frequency_comparison_per_codon")
    per_codon_df.to_csv(out_per_codon, index=False)
    print(f"  Saved per-codon comparison stats to {out_per_codon}")

    out_per_gene = tbl("codon_frequency_comparison_per_gene")
    per_gene_df.to_csv(out_per_gene, index=False)
    print(f"  Saved per-gene comparison stats to {out_per_gene}")

    worst = per_codon_df.loc[per_codon_df['pearson_r'].idxmin()]
    best = per_codon_df.loc[per_codon_df['pearson_r'].idxmax()]
    print(f"\n  Overall (all genes x all 61 codons, n={bio_vals.size:,}): "
          f"Pearson r = {r_overall:.4f} (p = {p_overall_str})")
    print(f"  Per-codon Pearson r: min = {worst['pearson_r']:.4f} ({worst['codon']}), "
          f"median = {per_codon_df['pearson_r'].median():.4f}, "
          f"max = {best['pearson_r']:.4f} ({best['codon']})")
    print(f"  Per-gene total |difference| (summed over 61 codons): "
          f"mean = {per_gene_df['l1_diff'].mean():.4f}, median = {per_gene_df['l1_diff'].median():.4f}, "
          f"95th pct = {per_gene_df['l1_diff'].quantile(0.95):.4f}, max = {per_gene_df['l1_diff'].max():.4f}")

    print("\n[5] Plotting comparison figures...")
    plot_overall_scatter(bio_vals, tim_vals, r_overall, p_overall_str,
                          fig("codon_freq_biomart_vs_tim_overall_scatter"))
    plot_per_codon_r_barplot(per_codon_df, len(merged),
                              fig("codon_freq_biomart_vs_tim_per_codon_pearson_r"))
    plot_mean_codon_scatter(per_codon_df, fig("codon_freq_biomart_vs_tim_mean_codon_scatter"))
    plot_per_gene_diff_hist(per_gene_df, fig("codon_freq_biomart_vs_tim_per_gene_diff_hist"))

    run_co_mega_te_analysis(biomart_df)

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
