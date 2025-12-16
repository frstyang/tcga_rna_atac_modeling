library(argparse)
library(edgeR)
library(ggplot2)
library(Matrix)
library(preprocessCore)

print.matrix <- function(m, quote=FALSE, right=TRUE, max=FALSE){
  write.table(format(m, justify="right"),
              row.names=FALSE, col.names=FALSE, quote=FALSE)
}
parser <- ArgumentParser(description = 'log-normalize reference TCGA and query pseudobulks')
parser$add_argument('query_pseudobulks_path')
parser$add_argument('query_pseudobulks_names_path')
parser$add_argument('query_peaks_path')
parser$add_argument('TCGA_PanCan_Raw_path')
parser$add_argument('output_dir')
parser$add_argument('--n_top_peaks', type = 'integer', default = NULL)
args <- parser$parse_args()

## ---- Load objects ----
query_counts <- readMM(args$query_pseudobulks_path)
query_counts <- as(query_counts, "CsparseMatrix")
query_labels <- read.csv(args$query_pseudobulks_names_path, header = FALSE)[,1]
query_peaks <- read.csv(args$query_peaks_path, header = FALSE)[,1]
rownames(query_counts) <- query_labels
colnames(query_counts) <- query_peaks
query_counts <- t(query_counts)

frac_nnz <- function(X) nnzero(X) / prod(dim(X))
sprintf('fraction nonzero of query counts: %f', frac_nnz(query_counts))

pancan_raw <- readRDS(args$TCGA_PanCan_Raw_path)
pancan_peaks_metadata <- pancan_raw[, 1:7]
write.csv(pancan_peaks_metadata, file = file.path(args$output_dir, 'peaks_metadata.csv'))
pancan_raw <- as.matrix(pancan_raw[, 8:ncol(pancan_raw)])
pancan_peaks <- paste0(
    pancan_peaks_metadata[[1]], ":",
    pancan_peaks_metadata[[2]], "-",
    pancan_peaks_metadata[[3]]
)
rownames(pancan_raw) <- pancan_peaks
sprintf('fraction nonzero of pancan counts: %f', frac_nnz(pancan_raw))

## ---- Align to common peak set ----
common_peaks <- intersect(rownames(pancan_raw), rownames(query_counts))
sprintf("Common peaks: %d", length(common_peaks))
query_counts_per_peak <- rowSums(query_counts)
query_counts_per_peak_sorted <- sort(query_counts_per_peak, decreasing=T)
df <- data.frame(
  x = seq_along(query_counts_per_peak_sorted),
  y = query_counts_per_peak_sorted
)
if (!is.null(args$n_top_peaks)) {
  p <- ggplot(df, aes(x, y)) +
    geom_line() +
    geom_vline(xintercept = args$n_top_peaks, color = "red", linetype = "dashed")
  common_peaks <- names(query_counts_per_peak_sorted[1:args$n_top_peaks])
} else {
  p <- ggplot(df, aes(x, y)) +
    geom_line()
}
ggsave(
  filename = paste0(args$output_dir, '/query_counts_per_peak_sorted.png'),
  plot = p, width = 5, height = 4, dpi = 150
)
sprintf('Common peaks after filtering %d', length(common_peaks))
pancan_raw <- pancan_raw[common_peaks, , drop=FALSE]
query_counts <- query_counts[common_peaks, , drop=FALSE]

## ---- Concatenate and normalize jointly ----
all_counts <- cbind(pancan_raw, query_counts)

y <- DGEList(counts = all_counts)
## TMMwsp is robust for sparse counts (good for ATAC)
y <- calcNormFactors(y, method = "TMMwsp")
sprintf('fraction nonzero of query counts after peak filtering: %f', frac_nnz(query_counts))
sprintf('fraction nonzero of pancan counts after peak filtering: %f', frac_nnz(pancan_raw))

## Choose a small prior.count appropriate for sparse ATAC
## (edgeR scales this by library size internally, so a single value is fine)
prior <- 1  # you can try 0.25 if you want less smoothing of zeros
logCPM_joint <- cpm(y, log = TRUE, prior.count = prior)  # normalized.lib.sizes=TRUE by default for DGEList

## ---- Split back out if you want separate outputs ----
ref_cols <- colnames(pancan_raw)
query_cols <- colnames(query_counts)

pancan_logcpm <- logCPM_joint[, ref_cols, drop=FALSE]
query_logcpm <- logCPM_joint[, query_cols, drop=FALSE]
## ---- Save LUAS (and TCGA if helpful) ----
saveRDS(as.data.frame(query_logcpm), file.path(args$output_dir, 'query_log2cpm.rds'))
saveRDS(pancan_logcpm, file.path(args$output_dir, 'tcga_log2cpm.rds'))

# Plot the density of reference and query log2cpms
library(tidyverse)
## ---- Pick 30 samples from each cohort (set seed for reproducibility) ----
set.seed(42)
n_ref <- min(30, ncol(pancan_logcpm))
n_query <- min(30, ncol(query_logcpm))
ref_sel <- sample(colnames(pancan_logcpm), n_ref)
query_sel <- sample(colnames(query_logcpm), n_query)

## ---- Optional: subsample peaks for speed if very large ----
n_peaks_for_plot <- min(200000, nrow(pancan_logcpm))  # tweak this if you want more/less resolution
if (nrow(pancan_logcpm) > n_peaks_for_plot) {
  row_idx <- sample(seq_len(nrow(pancan_logcpm)), n_peaks_for_plot)
} else {
  row_idx <- seq_len(nrow(pancan_logcpm))
}

## ---- Build long dataframe: value per (peak, sample) ----
X <- cbind(pancan_logcpm[row_idx, ref_sel, drop = FALSE],
           query_logcpm[row_idx, query_sel, drop = FALSE])

df_long <- as_tibble(X, rownames = "peak") |>
  pivot_longer(-peak, names_to = "sample", values_to = "log2cpm") |>
  mutate(
    cohort = if_else(sample %in% ref_sel, "TCGA", "query")
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
  labs(title = "Per-sample density of log2CPM (joint TMMwsp)",
       x = "log2CPM", y = "Density") +
  theme_classic() +
  theme(legend.position = "top")
ggsave(
  filename = paste0(args$output_dir, '/log2cpm_densities.png'),
  plot = p_overlay, width = 5, height = 4, dpi = 150
)
