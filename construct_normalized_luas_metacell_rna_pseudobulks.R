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
rna_counts <- npyLoad(file.path(data_dir, 'combined_tumor_metacells_rna_counts.npy'))
rna_genes <- read.table(file.path(data_dir, 'combined_tumor_metacells_rna_genes.txt'))[, 1]
metacell_labels <- read.table(file.path(data_dir, 'combined_tumor_metacells_labels.txt'))[, 1]
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
saveRDS(rna_logcpm, file.path(new_out_dir, 'metacells_rna_log2cpm_TMM.rds'))
