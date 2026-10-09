.libPaths(c("/n/groups/walsh/indData/Maya/RLibs", .libPaths()))

library(rtracklayer)
library(dplyr)
library(tidyr)
library(tibble)
library(pheatmap)

####################
# INPUT
####################
input_path <- "/n/groups/walsh/indData/Maya/Translational_Efficiency_Analysis"
df <- read.delim(file.path(input_path, "prediction_input.txt"), header = TRUE, stringsAsFactors = FALSE)

####################
# GTF
####################
gtf <- import(file.path(input_path, "MANE.GRCh38.v1.5.ensembl_genomic.gtf.gz"))
gtf$transcript_id <- sub("\\..*$", "", gtf$transcript_id)
gtf$gene_id <- sub("\\..*$", "", gtf$gene_id)
gtf <- gtf %>% as.data.frame() %>% filter(gene_type == "protein_coding")

####################
# SAN ROMAN
####################
san_roman_data <- read.table(file.path(input_path, "san_roman_bioRxiv_v1_2023_table_s1_Xi_expressed_or_Xi_modulated.txt"), sep = "\t", header = TRUE)
xi_expressed_genes <- san_roman_data %>% filter(is_Xi_expressed_San_Roman_2023) %>% pull(gene_id) %>% sub("\\..*$", "", .)
xi_silent_genes <- san_roman_data %>% filter(!is_Xi_expressed_San_Roman_2023) %>% pull(gene_id) %>% sub("\\..*$", "", .)

####################
# TRANSCRIPT ANNOTATION
####################
tx_to_info <- as.data.frame(gtf) %>%
  filter(type == "transcript", transcript_id %in% df$tx_id) %>%
  dplyr::select(tx_id = transcript_id, gene_id, chromosome = seqnames) %>%
  mutate(
    chromosome = sub("^chr", "", chromosome),
    group = case_when(
      chromosome %in% as.character(1:22) ~ "Autosome",
      chromosome == "X" & gene_id %in% xi_expressed_genes ~ "Xi_expressed",
      chromosome == "X" & gene_id %in% xi_silent_genes ~ "Xi_silent",
      TRUE ~ NA_character_
    )
  ) %>%
  distinct(tx_id, .keep_all = TRUE) %>%
  filter(chromosome %in% c(as.character(1:22), "X", "Y"))

write.table(
  tx_to_info,
  file.path(input_path, "tx_to_info_Xi_classification.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

cat("transcripts in input:", length(unique(df$tx_id)), "\n")
cat("transcripts matched:", length(unique(tx_to_info$tx_id)), "\n")
print(table(tx_to_info$group, useNA = "ifany"))

####################
# TRIPLETS
####################
get_triplets <- function(x) 
{
  x <- toupper(x)
  n <- floor(nchar(x) / 3)
  if (n == 0) return(character(0))
  substring(x, seq(1, n * 3, 3), seq(3, n * 3, 3))
}

get_triplet_freq <- function(df, sequence_col, region) {
  df %>%
    dplyr::select(tx_id, chromosome, group, sequence = all_of(sequence_col)) %>%
    filter(!is.na(group), !is.na(sequence), sequence != "") %>%
    mutate(triplets = lapply(sequence, get_triplets)) %>%
    unnest(triplets) %>%
    filter(
      nchar(triplets) == 3,
      grepl("^[ATCG]{3}$", triplets),
      !triplets %in% c("TAA", "TAG", "TGA")
    ) %>%
    count(tx_id, chromosome, group, triplets, name = "count") %>%
    group_by(tx_id) %>%
    mutate(freq = count / sum(count)) %>%
    ungroup() %>%
    mutate(region = region)
}

####################
# TRIPLET FREQUENCY
####################
triplet_freq <- bind_rows(
  df %>% left_join(tx_to_info, by = "tx_id") %>% get_triplet_freq("cds_sequence", "CDS"),
  df %>% left_join(tx_to_info, by = "tx_id") %>% get_triplet_freq("utr5_sequence", "5UTR"),
  df %>% left_join(tx_to_info, by = "tx_id") %>% get_triplet_freq("utr3_sequence", "3UTR")
) %>%
  filter(!triplets %in% c("TAA", "TAG", "TGA"))

cat("triplets retained:", length(unique(triplet_freq$triplets)), "\n")
cat("stop codons present:", sum(unique(triplet_freq$triplets) %in% c("TAA", "TAG", "TGA")), "\n")

####################
# XI VS AUTOSOME
####################
xi_vs_auto <- triplet_freq %>%
  filter(
    group %in% c("Autosome", "Xi_expressed", "Xi_silent"),
    !triplets %in% c("TAA", "TAG", "TGA")
  ) %>%
  group_by(region, triplets) %>%
  summarise(
    auto_n = sum(group == "Autosome"),
    xi_expressed_n = sum(group == "Xi_expressed"),
    xi_silent_n = sum(group == "Xi_silent"),
    auto_mean = mean(freq[group == "Autosome"]),
    xi_expressed_mean = mean(freq[group == "Xi_expressed"]),
    xi_silent_mean = mean(freq[group == "Xi_silent"]),
    xi_expressed_log2_ratio = log2(xi_expressed_mean / auto_mean),
    xi_silent_log2_ratio = log2(xi_silent_mean / auto_mean),
    xi_expressed_p_value = wilcox.test(freq[group == "Xi_expressed"], freq[group == "Autosome"])$p.value,
    xi_silent_p_value = wilcox.test(freq[group == "Xi_silent"], freq[group == "Autosome"])$p.value,
    .groups = "drop"
  ) %>%
  group_by(region) %>%
  mutate(
    xi_expressed_fdr = p.adjust(xi_expressed_p_value, "BH"),
    xi_silent_fdr = p.adjust(xi_silent_p_value, "BH")
  ) %>%
  ungroup()

write.table(
  xi_vs_auto,
  file.path(input_path, "Xi_expressed_silent_vs_autosome_triplet_usage_CDS_UTRs.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

####################
# HEATMAP MATRIX
####################
heatmap_mat <- xi_vs_auto %>%
  filter(!triplets %in% c("TAA", "TAG", "TGA")) %>%
  dplyr::select(region, triplets, xi_expressed_log2_ratio, xi_silent_log2_ratio) %>%
  pivot_longer(
    c(xi_expressed_log2_ratio, xi_silent_log2_ratio),
    names_to = "group",
    values_to = "log2_ratio"
  ) %>%
  mutate(row = paste(sub("_log2_ratio", "", group), region, sep = " - ")) %>%
  dplyr::select(row, triplets, log2_ratio) %>%
  pivot_wider(names_from = triplets, values_from = log2_ratio) %>%
  column_to_rownames("row") %>%
  as.matrix()

write.table(
  heatmap_mat,
  file.path(input_path, "Xi_expressed_silent_vs_autosome_triplet_log2_ratio_CDS_UTRs.tsv"),
  sep = "\t",
  quote = FALSE,
  col.names = NA
)

####################
# TRIPLET ANNOTATION
####################
triplet_annotation <- data.frame(
  End = ifelse(grepl("[GC]$", colnames(heatmap_mat)), "G/C", "A/T")
)
rownames(triplet_annotation) <- colnames(heatmap_mat)

annotation_colors <- list(
  End = c(
    "G/C" = "yellow",
    "A/T" = "blue"
  )
)

####################
# CDS HEATMAP
####################
cds_mat <- heatmap_mat[grepl("CDS$", rownames(heatmap_mat)), , drop = FALSE]

pheatmap(
  cds_mat,
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  scale = "none",
  border_color = NA,
  fontsize_row = 9,
  fontsize_col = 7,
  angle_col = 90,
  annotation_col = triplet_annotation,
  annotation_colors = annotation_colors,
  main = "Xi expressed/silent triplet usage - CDS\nlog2(Xi / autosome)",
  filename = file.path(input_path, "Xi_expressed_silent_vs_autosome_triplet_usage_CDS.pdf"),
  width = 12,
  height = 6
)

####################
# ALL REGIONS HEATMAP
####################
pheatmap(
  heatmap_mat,
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  scale = "column",
  border_color = NA,
  fontsize_row = 9,
  fontsize_col = 7,
  angle_col = 90,
  annotation_col = triplet_annotation,
  annotation_colors = annotation_colors,
  main = "Xi expressed/silent triplet usage - CDS + UTRs\nlog2(Xi / autosome)",
  filename = file.path(input_path, "Xi_expressed_silent_vs_autosome_triplet_usage_CDS_UTRs.pdf"),
  width = 12,
  height = 12
)

####################
# X VS AUTOSOME HEATMAP
####################
x_vs_auto_mat <- triplet_freq %>%
  filter(
    group %in% c("Autosome", "Xi_expressed", "Xi_silent"),
    !triplets %in% c("TAA", "TAG", "TGA")
  ) %>%
  group_by(region, triplets) %>%
  summarise(
    auto_mean = mean(freq[group == "Autosome"]),
    x_mean = mean(freq[group %in% c("Xi_expressed", "Xi_silent")]),
    log2_ratio = log2(x_mean / auto_mean),
    .groups = "drop"
  ) %>%
  dplyr::select(region, triplets, log2_ratio) %>%
  pivot_wider(names_from = triplets, values_from = log2_ratio) %>%
  column_to_rownames("region") %>%
  as.matrix()

write.table(
  x_vs_auto_mat,
  file.path(input_path, "X_vs_autosome_triplet_log2_ratio_CDS_UTRs.tsv"),
  sep = "\t",
  quote = FALSE,
  col.names = NA
)

####################
# X VS AUTOSOME CDS HEATMAP
####################
x_vs_auto_cds_mat <- x_vs_auto_mat["CDS", , drop = FALSE]

pheatmap(
  x_vs_auto_cds_mat,
  cluster_rows = FALSE,
  cluster_cols = TRUE,
  scale = "none",
  border_color = NA,
  fontsize_row = 10,
  fontsize_col = 7,
  angle_col = 90,
  annotation_col = triplet_annotation,
  annotation_colors = annotation_colors,
  main = "X chromosome triplet usage - CDS\nlog2(X / autosome)",
  filename = file.path(input_path, "X_vs_autosome_triplet_usage_CDS.pdf"),
  width = 12,
  height = 4
)

####################
# X VS AUTOSOME ALL REGIONS HEATMAP
####################
pheatmap(
  x_vs_auto_mat,
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  scale = "column",
  border_color = NA,
  fontsize_row = 10,
  fontsize_col = 7,
  angle_col = 90,
  annotation_col = triplet_annotation,
  annotation_colors = annotation_colors,
  main = "X chromosome triplet usage - CDS + UTRs\nlog2(X / autosome)",
  filename = file.path(input_path, "X_vs_autosome_triplet_usage_CDS_UTRs.pdf"),
  width = 12,
  height = 12
)

####################
# CHI-SQUARED TESTS: XI CLASSES VS AUTOSOMES
####################
chi_results <- triplet_freq %>%
  filter(
    group %in% c("Autosome", "Xi_expressed", "Xi_silent"),
    !triplets %in% c("TAA", "TAG", "TGA")
  ) %>%
  group_by(region, group, triplets) %>%
  summarise(count = sum(count), .groups = "drop")

run_chi <- function(region_name, test_group) 
{
  x <- chi_results %>%
    filter(region == region_name, group %in% c("Autosome", test_group)) %>%
    pivot_wider(names_from = group, values_from = count, values_fill = 0)

  tab <- as.matrix(x[, c("Autosome", test_group)])
  rownames(tab) <- x$triplets

  test <- chisq.test(tab)

  data.frame(
    Region = region_name,
    Comparison = paste(test_group, "vs autosome"),
    Chi_squared = unname(test$statistic),
    df = unname(test$parameter),
    P_value = test$p.value
  )
}

chi_results_final <- bind_rows(
  run_chi("CDS", "Xi_expressed"),
  run_chi("CDS", "Xi_silent"),
  run_chi("5UTR", "Xi_expressed"),
  run_chi("5UTR", "Xi_silent"),
  run_chi("3UTR", "Xi_expressed"),
  run_chi("3UTR", "Xi_silent")
) %>%
  mutate(FDR = p.adjust(P_value, method = "BH"))

####################
# PRINT XI CHI-SQUARED RESULTS
####################
cat("\n")
cat("============================================================\n")
cat("CHI-SQUARED TESTS: XI CLASSES VS AUTOSOMES\n")
cat("============================================================\n\n")

for (region in c("CDS", "5UTR", "3UTR")) {

  cat("------------------------------------------------------------\n")
  cat(region, "\n")
  cat("------------------------------------------------------------\n")

  x <- chi_results_final %>%
    filter(Region == region) %>%
    mutate(
      Chi_squared = sprintf("%.2f", Chi_squared),
      P_value = format.pval(P_value, digits = 3, eps = 1e-300),
      FDR = format.pval(FDR, digits = 3, eps = 1e-300)
    ) %>%
    dplyr::select(
      Comparison,
      Chi_squared,
      df,
      P_value,
      FDR
    )

  print(x, row.names = FALSE)
  cat("\n")
}

####################
# SAVE XI CHI-SQUARED RESULTS
####################
write.table(
  chi_results_final,
  file.path(input_path, "Xi_expressed_silent_vs_autosome_chi_squared.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

####################
# CHI-SQUARED TESTS: X VS AUTOSOMES
####################
x_chi_results <- triplet_freq %>%
  filter(
    group %in% c("Autosome", "Xi_expressed", "Xi_silent"),
    !triplets %in% c("TAA", "TAG", "TGA")
  ) %>%
  mutate(group = ifelse(group %in% c("Xi_expressed", "Xi_silent"), "X", "Autosome")) %>%
  group_by(region, group, triplets) %>%
  summarise(count = sum(count), .groups = "drop")

run_x_chi <- function(region_name) 
{
  x <- x_chi_results %>%
    filter(region == region_name) %>%
    pivot_wider(names_from = group, values_from = count, values_fill = 0)

  tab <- as.matrix(x[, c("Autosome", "X")])
  rownames(tab) <- x$triplets

  test <- chisq.test(tab)

  data.frame(
    Region = region_name,
    Comparison = "X vs autosome",
    Chi_squared = unname(test$statistic),
    df = unname(test$parameter),
    P_value = test$p.value
  )
}

x_chi_results_final <- bind_rows(
  run_x_chi("CDS"),
  run_x_chi("5UTR"),
  run_x_chi("3UTR")
) %>%
  mutate(FDR = p.adjust(P_value, method = "BH"))

####################
# PRINT X CHI-SQUARED RESULTS
####################
cat("\n")
cat("============================================================\n")
cat("CHI-SQUARED TESTS: X VS AUTOSOMES\n")
cat("============================================================\n\n")

x <- x_chi_results_final %>%
  mutate(
    Chi_squared = sprintf("%.2f", Chi_squared),
    P_value = format.pval(P_value, digits = 3, eps = 1e-300),
    FDR = format.pval(FDR, digits = 3, eps = 1e-300)
  ) %>%
  dplyr::select(
    Region,
    Comparison,
    Chi_squared,
    df,
    P_value,
    FDR
  )

print(x, row.names = FALSE)
cat("\n")

####################
# SAVE X CHI-SQUARED RESULTS
####################
write.table(
  x_chi_results_final,
  file.path(input_path, "X_vs_autosome_chi_squared.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)