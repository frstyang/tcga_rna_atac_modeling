library(edgeR)
library(preprocessCore)
library(RcppCNPy)
library(scran)
library(SingleCellExperiment)

print.matrix <- function(m, quote=FALSE, right=TRUE, max=FALSE){
  write.table(format(m, justify="right"),
              row.names=FALSE, col.names=FALSE, quote=FALSE)
}

## ---- Paths ----
data_dir <- '/data1/chanj3/LUAS.multiome.results/epigenetic/rna_atac_pseudotime/data/SEACells'
out_dir <- '/data1/chanj3/LUAS.multiome.results/epigenetic/TCGA_modeling/out'

## ---- Load ATAC counts ----
atac_counts <- npyLoad(file.path(data_dir, 'combined_tumor_metacells_atac_counts.npy'))
atac_peaks <- read.table(file.path(data_dir, 'combined_tumor_metacells_atac_peaks.txt'))[, 1]
metacell_labels <- read.table(file.path(data_dir, 'combined_tumor_metacells_labels.txt'))[, 1]
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
pancan_peaks_meta <- read.table(paste0(new_out_dir, '/tcga_peaks_metadata.tsv'))
tcga_peak_id <- paste0(pancan_peaks_meta[[1]], ":", pancan_peaks_meta[[2]], "-", pancan_peaks_meta[[3]])

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
saveRDS(luas, file.path(new_out_dir, 'metacells_atac_log2cpm_scran.rds'))
