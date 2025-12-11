library(edgeR)
library(preprocessCore)
library(RcppCNPy)

print.matrix <- function(m, quote=FALSE, right=TRUE, max=FALSE){
  write.table(format(m, justify="right"),
              row.names=FALSE, col.names=FALSE, quote=FALSE)
}

## ---- Paths ----
data_dir <- '/data1/chanj3/LUAS.multiome.results/epigenetic/rna_atac_pseudotime/data/SEACells'
out_dir <- '/data1/chanj3/LUAS.multiome.results/epigenetic/TCGA_modeling/out'

## ---- Load ATAC counts ----
rna_counts <- npyLoad(file.path(data_dir, 'combined_tumor_metacells_cell_cycle_regressed_rna_counts.npy'))
rna_genes <- read.table(file.path(data_dir, 'combined_tumor_metacells_cell_cycle_regressed_rna_genes.txt'))[, 1]
metacell_labels <- read.table(file.path(data_dir, 'combined_tumor_metacells_cell_cycle_regressed_labels.txt'))[, 1]
rownames(rna_counts) <- metacell_labels
colnames(rna_counts) <- rna_genes
rna_counts <- t(rna_counts)

y <- DGEList(counts = rna_counts)
## TMMwsp is robust for sparse counts (good for ATAC)
y <- calcNormFactors(y, method = "TMM")

## Choose a small prior.count appropriate for sparse ATAC
## (edgeR scales this by library size internally, so a single value is fine)
prior <- 1  # you can try 0.25 if you want less smoothing of zeros

rna_logcpm <- cpm(y, log = TRUE, prior.count = prior)  # normalized.lib.sizes=TRUE by default for DGEList

library(tidyverse)

## ---- Pick 10 samples from each cohort (set seed for reproducibility) ----
set.seed(42)
n_luas <- min(10, ncol(rna_logcpm))
luas_sel <- sample(colnames(rna_logcpm), n_luas)

X <- rna_logcpm[, luas_sel, drop = FALSE]

df_long <- as_tibble(X, rownames = "peak") |>
  pivot_longer(-peak, names_to = "sample", values_to = "log2cpm") |>
  mutate(
    cohort = 'LUAS'
  )

## ---- Medians per sample (optional dashed reference lines) ----
meds <- df_long |>
  group_by(sample, cohort) |>
  summarize(median_log2cpm = median(log2cpm), .groups = "drop")

## ---- Plot 1: overlay all 20 samples; color by cohort ----
p_overlay <- ggplot(df_long, aes(x = log2cpm, group = sample, color = cohort)) +
  geom_density(size = 0.6, alpha = 0.9, adjust=3) +
  geom_vline(data = meds, aes(xintercept = median_log2cpm, color = cohort),
             linetype = "dashed", alpha = 0.5, show.legend = FALSE) +
  labs(title = "Per-sample density of log2CPM (TMM)",
       x = "log2CPM", y = "Density") +
  theme_classic() +
  theme(legend.position = "top")

print(p_overlay)

new_out_dir = '/data1/chanj3/LUAS.multiome.results/epigenetic/TCGA_modeling/out'

## ---- Save LUAS (and TCGA if helpful) ----
saveRDS(rna_logcpm, file.path(new_out_dir, 'metacells_cell_cycle_regressed_rna_log2cpm_TMM.rds'))


## ======================================================================
## -------------- construct normalized ATAC pseudobulks --------------
## ======================================================================
library(scran)
library(SingleCellExperiment)

## ---- Load ATAC counts ----
atac_counts <- npyLoad(file.path(data_dir, 'combined_tumor_metacells_cell_cycle_regressed_atac_counts.npy'))
atac_peaks <- read.table(file.path(data_dir, 'combined_tumor_metacells_cell_cycle_regressed_atac_peaks.txt'))[, 1]
metacell_labels <- read.table(file.path(data_dir, 'combined_tumor_metacells_cell_cycle_regressed_labels.txt'))[, 1]
rownames(atac_counts) <- metacell_labels
colnames(atac_counts) <- atac_peaks
atac_counts <- t(atac_counts)

sce <- SingleCellExperiment(assays = list(counts = atac_counts))
clusters <- quickCluster(sce)
sce <- computeSumFactors(sce, clusters=clusters)
summary(sizeFactors(sce))
sce <- logNormCounts(sce)

## ---- Load TCGA raw counts ----
new_out_dir <- '/data1/chanj3/LUAS.multiome.results/epigenetic/TCGA_modeling/out'
tcga_dir <- '/data1/chanj3/LUAS.multiome.results/epigenetic/gorces_2018_cancer_ATAC_data'
pancan_raw <- readRDS(file.path(tcga_dir, 'TCGA-ATAC_PanCan_Raw_Counts.rds'))
pancan_peaks_meta <- pancan_raw[, 1:7]
pancan_raw_counts <- as.matrix(pancan_raw[, 8:ncol(pancan_raw)])
## Build peak IDs for TCGA (assumes first three meta columns are chr, start, end)
tcga_peak_id <- paste0(pancan_peaks_meta[[1]], ":", pancan_peaks_meta[[2]], "-", pancan_peaks_meta[[3]])
rownames(pancan_raw_counts) <- tcga_peak_id

## ---- Align to common peak set ----
common_peaks <- intersect(tcga_peak_id, rownames(atac_counts))
message("Common peaks: ", length(common_peaks))

tcga_counts_common <- pancan_raw_counts[common_peaks, , drop=FALSE]
sce <- sce[common_peaks, , drop=FALSE]
tcga <- readRDS(paste0(new_out_dir, '/tcga_log2cpm_jointTMMwsp.rds'))

## ---- Split back out if you want separate outputs ----
luas <- logcounts(sce)
library(tidyverse)

## ---- Pick 10 samples from each cohort (set seed for reproducibility) ----
set.seed(42)
n_tcga <- min(10, ncol(tcga))
n_luas <- min(10, ncol(luas))
tcga_sel <- sample(colnames(tcga), n_tcga)
luas_sel <- sample(colnames(luas), n_luas)

## ---- Optional: subsample peaks for speed if very large ----
n_peaks_for_plot <- min(200000, nrow(tcga))  # tweak this if you want more/less resolution
row_idx <- if (nrow(tcga) > n_peaks_for_plot) sample(seq_len(nrow(tcga)), n_peaks_for_plot) else seq_len(nrow(tcga))

## ---- Build long dataframe: value per (peak, sample) ----
X <- cbind(tcga[row_idx, tcga_sel, drop = FALSE],
           luas[row_idx, luas_sel, drop = FALSE])

df_long <- as_tibble(X, rownames = "peak") |>
  pivot_longer(-peak, names_to = "sample", values_to = "log2cpm") |>
  mutate(
    cohort = if_else(sample %in% tcga_sel, "TCGA", "LUAS")
  )

## ---- Medians per sample (optional dashed reference lines) ----
meds <- df_long |>
  group_by(sample, cohort) |>
  summarize(median_log2cpm = median(log2cpm), .groups = "drop")

## ---- Plot 1: overlay all 20 samples; color by cohort ----
p_overlay <- ggplot(df_long, aes(x = log2cpm, group = sample, color = cohort)) +
  geom_density(size = 0.6, alpha = 0.9, adjust=3) +
  geom_vline(data = meds, aes(xintercept = median_log2cpm, color = cohort),
             linetype = "dashed", alpha = 0.5, show.legend = FALSE) +
  labs(title = "Per-sample density of log2CPM (scran / TMMwsp)",
       x = "log2CPM", y = "Density") +
  theme_classic() +
  theme(legend.position = "top")

print(p_overlay)

## ---- Save LUAS (and TCGA if helpful) ----
saveRDS(luas, file.path(new_out_dir, 'metacells_cell_cycle_regressed_atac_log2cpm_scran.rds'))

# ## ---- Load ATAC counts ----
# atac_counts <- npyLoad(file.path(data_dir, 'combined_tumor_metacells_cell_cycle_regressed_atac_counts.npy'))
# atac_peaks <- read.table(file.path(data_dir, 'combined_tumor_metacells_cell_cycle_regressed_atac_peaks.txt'))[, 1]
# metacell_labels <- read.table(file.path(data_dir, 'combined_tumor_metacells_cell_cycle_regressed_labels.txt'))[, 1]
# rownames(atac_counts) <- metacell_labels
# colnames(atac_counts) <- atac_peaks
# atac_counts <- t(atac_counts)
# 
# ## ---- Load TCGA raw counts ----
# tcga_dir <- '/data1/chanj3/LUAS.multiome.results/epigenetic/gorces_2018_cancer_ATAC_data'
# pancan_raw <- readRDS(file.path(tcga_dir, 'TCGA-ATAC_PanCan_Raw_Counts.rds'))
# pancan_peaks_meta <- pancan_raw[, 1:7]
# pancan_raw_counts <- as.matrix(pancan_raw[, 8:ncol(pancan_raw)])
# ## Build peak IDs for TCGA (assumes first three meta columns are chr, start, end)
# tcga_peak_id <- paste0(pancan_peaks_meta[[1]], ":", pancan_peaks_meta[[2]], "-", pancan_peaks_meta[[3]])
# rownames(pancan_raw_counts) <- tcga_peak_id
# 
# ## ---- Align to common peak set ----
# common_peaks <- intersect(rownames(pancan_raw_counts), rownames(atac_counts))
# message("Common peaks: ", length(common_peaks))
# 
# tcga_counts_common <- pancan_raw_counts[common_peaks, , drop=FALSE]
# luas_counts_common <- atac_counts[common_peaks, , drop=FALSE]
# 
# ## ---- Concatenate and normalize jointly ----
# all_counts <- cbind(tcga_counts_common, luas_counts_common)
# 
# y <- DGEList(counts = all_counts)
# ## TMMwsp is robust for sparse counts (good for ATAC)
# y <- calcNormFactors(y, method = "TMMwsp")
# 
# ## Choose a small prior.count appropriate for sparse ATAC
# ## (edgeR scales this by library size internally, so a single value is fine)
# prior <- 5  # you can try 0.25 if you want less smoothing of zeros
# 
# logCPM_joint <- cpm(y, log = TRUE, prior.count = prior)  # normalized.lib.sizes=TRUE by default for DGEList
# 
# ## ---- (Optional) Quantile normalize AFTER joint logCPM if you must match a QN pipeline ----
# ## logCPM_joint <- normalize.quantiles(logCPM_joint)
# ## colnames(logCPM_joint) <- colnames(all_counts)
# ## rownames(logCPM_joint) <- rownames(all_counts)
# 
# ## ---- Split back out if you want separate outputs ----
# tcga_cols <- colnames(tcga_counts_common)
# luas_cols <- colnames(luas_counts_common)
# 
# tcga <- logCPM_joint[, tcga_cols, drop=FALSE]
# luas<- logCPM_joint[, luas_cols, drop=FALSE]
# 
# library(tidyverse)
# 
# ## ---- Pick 10 samples from each cohort (set seed for reproducibility) ----
# set.seed(42)
# n_tcga <- min(10, ncol(tcga))
# n_luas <- min(10, ncol(luas))
# tcga_sel <- sample(colnames(tcga), n_tcga)
# luas_sel <- sample(colnames(luas), n_luas)
# 
# ## ---- Optional: subsample peaks for speed if very large ----
# n_peaks_for_plot <- min(200000, nrow(tcga))  # tweak this if you want more/less resolution
# row_idx <- if (nrow(tcga) > n_peaks_for_plot) sample(seq_len(nrow(tcga)), n_peaks_for_plot) else seq_len(nrow(tcga))
# 
# ## ---- Build long dataframe: value per (peak, sample) ----
# X <- cbind(tcga[row_idx, tcga_sel, drop = FALSE],
#            luas[row_idx, luas_sel, drop = FALSE])
# 
# df_long <- as_tibble(X, rownames = "peak") |>
#   pivot_longer(-peak, names_to = "sample", values_to = "log2cpm") |>
#   mutate(
#     cohort = if_else(sample %in% tcga_sel, "TCGA", "LUAS")
#   )
# 
# ## ---- Medians per sample (optional dashed reference lines) ----
# meds <- df_long |>
#   group_by(sample, cohort) |>
#   summarize(median_log2cpm = median(log2cpm), .groups = "drop")
# 
# ## ---- Plot 1: overlay all 20 samples; color by cohort ----
# p_overlay <- ggplot(df_long, aes(x = log2cpm, group = sample, color = cohort)) +
#   geom_density(size = 0.6, alpha = 0.9, adjust=3) +
#   geom_vline(data = meds, aes(xintercept = median_log2cpm, color = cohort),
#              linetype = "dashed", alpha = 0.5, show.legend = FALSE) +
#   labs(title = "Per-sample density of log2CPM (joint TMMwsp)",
#        x = "log2CPM", y = "Density") +
#   theme_classic() +
#   theme(legend.position = "top")
# 
# print(p_overlay)
# 
# new_out_dir = '/data1/chanj3/LUAS.multiome.results/epigenetic/TCGA_modeling/out'
# 
# ## ---- Save LUAS (and TCGA if helpful) ----
# saveRDS(luas, file.path(new_out_dir, 'metacells_cell_cycle_regressed_atac_log2cpm_jointTMMwsp.rds'))
# 
