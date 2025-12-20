library(argparse)
library(edgeR)
library(Matrix)
library(preprocessCore) 
library(tidyverse)

parser <- ArgumentParser(description = 'log-normalize query RNA pseudobulks')
parser$add_argument('query_pseudobulks_path')
parser$add_argument('query_pseudobulks_names_path')
parser$add_argument('query_genes_path')
parser$add_argument('output_dir')
parser$add_argument('--scran_norm', action = 'store_true')
args <- parser$parse_args()

## ---- Load counts ----
query_counts <- readMM(args$query_pseudobulks_path)
query_counts <- as(query_counts, "CsparseMatrix")
query_labels <- read.table(args$query_pseudobulks_names_path)[, 1]
query_genes <- read.table(args$query_genes_path)[, 1]
rownames(query_counts) <- query_labels
colnames(query_counts) <- query_genes
query_counts <- t(query_counts)

frac_nnz <- function(X) nnzero(X) / prod(dim(X))
sprintf('fraction nonzero of query counts: %f', frac_nnz(query_counts))

if (!args$scran_norm) {
  y <- DGEList(counts = query_counts)
  ## TMMwsp is robust for sparse counts
  y <- calcNormFactors(y, method = "TMMwsp")

  ## Choose a small prior.count
  ## (edgeR scales this by library size internally, so a single value is fine)
  prior <- 1  # you can try 0.25 if you want less smoothing of zeros
  query_logcpm <- cpm(y, log = TRUE, prior.count = prior)  # normalized.lib.sizes=TRUE by default for DGEList
} else {
  library(scran)
  sce <- SingleCellExperiment(assays = list(counts = query_counts))
  clusters <- quickCluster(sce)
  sce <- computeSumFactors(sce, clusters=clusters)
  summary(sizeFactors(sce))
  query_logcpm_sce <- logNormCounts(sce)
  query_logcpm <- assay(query_logcpm_sce, "logcounts")
  query_logcpm <- as.matrix(query_logcpm)
}

## ---- Pick 10 samples from each cohort (set seed for reproducibility) ----
set.seed(42)
n_query <- min(20, ncol(query_logcpm))
query_sel <- sample(colnames(query_logcpm), n_query)

X <- query_logcpm[, query_sel, drop = FALSE]

df_long <- as_tibble(X, rownames = "gene") |>
  pivot_longer(-gene, names_to = "sample", values_to = "log2cpm") |>
  mutate(cohort = 'query')

## ---- Medians per sample (optional dashed reference lines) ----
meds <- df_long |>
  group_by(sample, cohort) |>
  summarize(median_log2cpm = median(log2cpm), .groups = "drop")

## ---- Plot 1: overlay all 20 samples; color by cohort ----
p_overlay <- ggplot(df_long, aes(x = log2cpm, group = sample, color = cohort)) +
  geom_density(linewidth = 0.6, alpha = 0.9, adjust=3) +
  geom_vline(data = meds, aes(xintercept = median_log2cpm, color = cohort),
             linetype = "dashed", alpha = 0.5, show.legend = FALSE) +
  labs(title = "Per-sample density of log2CPM (TMMwsp)",
       x = "log2CPM", y = "Density") +
  theme_classic() +
  theme(legend.position = "top")

ggsave(
  filename = file.path(args$output_dir, 'log2cpm_densities.png'),
  plot = p_overlay, width = 5, height = 4, dpi = 150
)

## ---- Save LUAS (and TCGA if helpful) ----
saveRDS(query_logcpm, file.path(args$output_dir, 'query_log2cpm.rds'))
