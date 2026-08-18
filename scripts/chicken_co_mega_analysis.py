#!/home/xueqisun/miniforge3/envs/xi_app/bin/python3
"""
Build our own chicken CO_Mega model (codon composition -> TE) and use it to
compare CO_Mega between autosomal and Z-linked genes. Mirrors
human_co_mega_analysis.py.

Pipeline:
  1. Parse chicken CDS sequences (autosomes + Z) from
     analysis_codon/data/chicken_CDS_sequence.txt (FASTA, header
     ">gene_id|gene_id.version|transcript_id|transcript_id.version|gene_name|chromosome")
     and save as analysis_codon/data/chicken_CDS_sequence.csv.
  2. For each gene, split the CDS into codons, drop the last codon (the stop
     codon), and compute each of the 61 sense codons' frequency as
     (# that codon) / (# total codons remaining).
  3. Load per-gene TE_mean (chicken liver) from
     analysis_riboseq/tables/chicken_liver/translational_efficiency_data.csv.
     Its gene IDs use a different (GRCg6a) ID scheme than the CDS file
     (EnsemblBiomart), so IDs are converted via
     general/genes_1to1_ortholog_human_chicken/data/GEGA_chicken_gene_id_conversion.csv
     ("gnId_eqGRCg6a" <-> TE file's Gene_ID; "gnId_eqEnsemblBiomart" /
     "gnName_eqEnsemblBiomart" <-> CDS file's gene_id / gene_name), keeping
     only unique (1:1) ID mappings and falling back to a name-based mapping
     when the ID mapping is missing/ambiguous.
  4. Fit  TE_mean ~ 61 codon frequencies  (OLS) using only autosomal
     (chr 1-39) genes. Report R^2 and save the intercept + 61 coefficients.
  5. Apply the fitted model to ALL genes (autosomes + Z) to get each gene's
     CO_Mega, report its correlation with TE_mean, and compare CO_Mega
     between autosomal and Z-linked genes with a boxplot + Mann-Whitney U.

All output tables/figures are suffixed with `_chicken_liver`.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from co_mega import SENSE_CODONS, compute_co_mega
from co_mega_common import (
    precise_pvalue_str, compute_codon_frequencies, fit_co_mega_te_model,
    save_coefficient_table, plot_actual_vs_predicted, plot_co_mega_boxplot,
    bootstrap_codon_pearson_r, plot_codon_pearson_r_barplot, parse_cds_fasta,
)

BASE_DIR    = "/lab/solexa_page/xueqi/analysis_codon"
CDS_FASTA   = os.path.join(BASE_DIR, "data/chicken_CDS_sequence.txt")
CDS_CSV     = os.path.join(BASE_DIR, "data/chicken_CDS_sequence.csv")
TE_FILE     = "/lab/solexa_page/xueqi/analysis_riboseq/tables/chicken_liver/translational_efficiency_data.csv"
GEGA_FILE   = "/lab/solexa_page/xueqi/general/genes_1to1_ortholog_human_chicken/data/GEGA_chicken_gene_id_conversion.csv"
TABLE_DIR   = os.path.join(BASE_DIR, "tables")
FIG_DIR     = os.path.join(BASE_DIR, "figures")

AUTOSOMES = [str(i) for i in range(1, 40)]  # chicken autosomes: chr 1-39

SUFFIX = "chicken_liver"
TISSUE_LABEL = "Chicken liver"


def tbl(name):
    return os.path.join(TABLE_DIR, f"{name}_{SUFFIX}.csv")


def fig(name):
    return os.path.join(FIG_DIR, f"{name}_{SUFFIX}.png")


# ── Step 2: codon frequencies ───────────────────────────────────────────────
def load_codon_frequencies(cds_csv):
    df = pd.read_csv(cds_csv, usecols=['gene_id', 'gene_name', 'chromosome', 'CDS_sequence'])
    print(f"  Loaded {len(df):,} genes from {cds_csv}")

    freq_records = df['CDS_sequence'].apply(compute_codon_frequencies)
    n_unusable = freq_records.isna().sum()
    df = df.loc[freq_records.notna()].copy()
    freq_df = pd.DataFrame(list(freq_records.dropna()), index=df.index)
    print(f"  Dropped {n_unusable:,} genes with missing/unusable CDS sequences "
          f"(not a multiple of 3, or too short); {len(df):,} genes remain.")

    out = pd.concat([df[['gene_id', 'gene_name', 'chromosome']], freq_df], axis=1)
    out['chromosome'] = out['chromosome'].astype(str)
    out['is_Z'] = (out['chromosome'] == 'Z').astype(int)
    n_auto = (out['chromosome'].isin(AUTOSOMES)).sum()
    n_z = (out['chromosome'] == 'Z').sum()
    print(f"  {n_auto:,} autosomal (chr 1-39) genes, {n_z:,} Z-linked genes.")
    return out


# ── Step 3: chicken gene-ID conversion (GEGA table) ─────────────────────────
def _unique_1to1_series(df, key_col, val_col):
    """Keep only key -> value pairs where the key maps to exactly one value."""
    sub = df[[key_col, val_col]].dropna().drop_duplicates()
    counts = sub.groupby(key_col)[val_col].nunique()
    unique_keys = set(counts[counts == 1].index)
    sub = sub[sub[key_col].isin(unique_keys)].drop_duplicates(subset=[key_col])
    return sub.set_index(key_col)[val_col]


def build_id_conversion(gega_file):
    """
    Build gene_id -> TE-file gene ID and gene_name -> TE-file gene ID lookup
    tables from the GEGA chicken gene-ID conversion file, keeping only
    unique (1:1) mappings (ambiguous many:1 or 1:many mappings are dropped).
    """
    df = pd.read_csv(
        gega_file, sep=';', quotechar='"', low_memory=False,
        usecols=['gnId_eqEnsemblBiomart', 'gnName_eqEnsemblBiomart', 'gnId_eqGRCg6a']
    )
    df = df.dropna(subset=['gnId_eqGRCg6a'])
    id_map = _unique_1to1_series(df, 'gnId_eqEnsemblBiomart', 'gnId_eqGRCg6a')
    name_map = _unique_1to1_series(df, 'gnName_eqEnsemblBiomart', 'gnId_eqGRCg6a')
    print(f"  Loaded {len(df):,} rows with a GRCg6a ID from {gega_file}")
    print(f"  {len(id_map):,} unique ID-based and {len(name_map):,} unique name-based mappings to TE-file gene IDs.")
    return id_map, name_map


def map_te_gene_id(codon_df, id_map, name_map):
    """
    Map each CDS-file gene to the TE file's gene-ID scheme: try the
    ID-based mapping first (gene_id -> gnId_eqEnsemblBiomart -> gnId_eqGRCg6a),
    then fall back to the name-based mapping (gene_name -> gnName_eqEnsemblBiomart
    -> gnId_eqGRCg6a) when the ID mapping is missing.
    """
    by_id = codon_df['gene_id'].map(id_map)
    by_name = codon_df['gene_name'].map(name_map)
    te_gene_id = by_id.combine_first(by_name)

    n_by_id = by_id.notna().sum()
    n_by_name_only = (te_gene_id.notna() & by_id.isna()).sum()
    print(f"  Mapped {n_by_id:,} genes to a TE-file gene ID by ID, "
          f"{n_by_name_only:,} more by gene-name fallback; "
          f"{te_gene_id.notna().sum():,} of {len(codon_df):,} genes mapped in total.")
    return te_gene_id


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=== Chicken CO_Mega model (chicken liver) ===")

    print("\n[1] Parsing chicken CDS FASTA into a CSV table...")
    parse_cds_fasta(CDS_FASTA, CDS_CSV)

    print("\n[2] Computing codon frequencies from CDS sequences...")
    codon_df = load_codon_frequencies(CDS_CSV)
    out_codon = tbl("codon_frequencies")
    codon_df.to_csv(out_codon, index=False)
    print(f"  Saved codon frequencies to {out_codon}")

    print("\n[3] Loading chicken liver TE data and converting gene IDs...")
    te_df = pd.read_csv(TE_FILE)[['Gene_ID', 'TE_mean']].rename(columns={'Gene_ID': 'te_gene_id'})
    te_df = te_df.dropna(subset=['TE_mean'])
    print(f"  {len(te_df):,} genes with TE_mean in {TE_FILE}")

    id_map, name_map = build_id_conversion(GEGA_FILE)
    codon_df['te_gene_id'] = map_te_gene_id(codon_df, id_map, name_map)

    merged = codon_df.merge(te_df, on='te_gene_id', how='inner')
    print(f"  {len(merged):,} genes with both codon frequencies and TE_mean.")

    print("\n[4] Fitting TE_mean ~ codon frequencies on autosomal genes...")
    train = merged[merged['chromosome'].isin(AUTOSOMES)]
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

    print("\n[5] Computing CO_Mega for all genes (autosomes + Z) and comparing to TE...")
    codon_df['CO_Mega'] = compute_co_mega(codon_df[SENSE_CODONS], coef)

    with_te = codon_df.merge(te_df, on='te_gene_id', how='inner')
    r_all, p_all = stats.pearsonr(with_te['CO_Mega'], with_te['TE_mean'])
    rho_all, p_rho_all = stats.spearmanr(with_te['CO_Mega'], with_te['TE_mean'])
    print(f"  All genes with TE (n={len(with_te):,}): "
          f"Pearson r = {r_all:.4f} (p = {p_all:.3g}), Spearman rho = {rho_all:.4f} (p = {p_rho_all:.3g})")

    auto_te = with_te[with_te['chromosome'].isin(AUTOSOMES)]
    r_auto, p_auto = stats.pearsonr(auto_te['CO_Mega'], auto_te['TE_mean'])
    print(f"  Autosomal genes only (n={len(auto_te):,}, training set): Pearson r = {r_auto:.4f} (p = {p_auto:.3g})")

    z_te = with_te[with_te['chromosome'] == 'Z']
    r_z, p_z = stats.pearsonr(z_te['CO_Mega'], z_te['TE_mean']) if len(z_te) > 1 else (np.nan, np.nan)
    if len(z_te) > 1:
        print(f"  Z-linked genes only (n={len(z_te):,}, held out): Pearson r = {r_z:.4f} (p = {p_z:.3g})")

    p_all_str = precise_pvalue_str(r_all, len(with_te))
    p_auto_str = precise_pvalue_str(r_auto, len(auto_te))
    p_z_str = precise_pvalue_str(r_z, len(z_te)) if len(z_te) > 1 else np.nan

    corr_stats = pd.DataFrame([
        {'gene_set': 'All genes', 'n': len(with_te), 'pearson_r': r_all, 'p_value': p_all_str},
        {'gene_set': 'Autosomal genes', 'n': len(auto_te), 'pearson_r': r_auto, 'p_value': p_auto_str},
        {'gene_set': 'Z-linked genes', 'n': len(z_te), 'pearson_r': r_z, 'p_value': p_z_str},
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
    if len(z_te) > 1:
        plot_actual_vs_predicted(z_te['TE_mean'], z_te['CO_Mega'], 'Actual TE_mean', 'CO_Mega',
                                  f'Actual TE vs. CO_Mega: Z-linked genes ({TISSUE_LABEL})', r_z, p_z_str,
                                  fig("co_mega_vs_te_Z_linked"))

    out_all = tbl("co_mega_all_genes")
    codon_df[['gene_id', 'gene_name', 'chromosome', 'is_Z', 'te_gene_id', 'CO_Mega']].merge(
        te_df, on='te_gene_id', how='left'
    ).to_csv(out_all, index=False)
    print(f"  Saved CO_Mega for all {len(codon_df):,} genes to {out_all}")

    print("\n[6] Boxplot: CO_Mega, autosomal vs Z-linked genes...")
    plot_co_mega_boxplot(codon_df, 'CO_Mega', 'is_Z', 'Z chromosome',
                          f'CO_Mega: Z-linked vs Autosomal genes ({TISSUE_LABEL})',
                          fig("co_mega_boxplot_autosome_vs_Z"))

    print("\n[7] Per-codon Pearson r with TE (bootstrap SD, autosomal training genes)...")
    r_df = bootstrap_codon_pearson_r(train[SENSE_CODONS], train['TE_mean'], SENSE_CODONS)
    out_codon_r = tbl("co_mega_codon_pearson_r")
    r_df.to_csv(out_codon_r, index=False)
    print(f"  Saved per-codon Pearson r (+ bootstrap SD) to {out_codon_r}")
    plot_codon_pearson_r_barplot(
        r_df, f'Pearson R between codon frequency and TE, per codon\n(autosomal training genes, {TISSUE_LABEL})',
        fig("co_mega_codon_pearson_r_barplot")
    )

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
