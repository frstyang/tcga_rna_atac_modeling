library(RcppCNPy)
library(edgeR)
library(mclust)
library(ggplot2)
library(ggrepel)
library(data.table)
library(ggtext)
library(patchwork)

pseudobulks_dir = '/data1/chanj3/LUAS.multiome.results/epigenetic/accessibility_landscape/v4/out/compare_vs_gorces'
mat <- npyLoad(paste0(pseudobulks_dir, '/rna_pseudobulks_raw.npy'))
n <- dim(mat)[1]
g <- dim(mat)[2]
vec <- as.vector(t(mat))
mat <- matrix(vec, nrow=n, ncol=g)
genes <- read.csv(paste0(pseudobulks_dir, '/rna_pseudobulk_genes.txt'), header=F)[,1]
labels <- read.csv(paste0(pseudobulks_dir, '/rna_pseudobulk_labels.txt'), header=F)[,1]
mat <- t(mat)
colnames(mat) <- labels
rownames(mat) <- genes


# Create DGEList
dge <- DGEList(counts = mat)
dge <- calcNormFactors(dge, method = "TMM")  # TMM normalization
logcpm <- cpm(dge, log = TRUE, prior.count = 1)  # log2 CPM

# Define gene sets
adeno_genes <- c('NAPSA', 'NKX2-1', 'LMO3', 'SFTA3', 'TMC5', 'MUC1', 'KRT7')
squam_genes <- c('KRT5', 'KRT6A', 'CLCA2', 'DSG3', 'TP63', 'JAG1')

# Compute average score for each sample
adeno_scores <- colMeans(logcpm[adeno_genes, , drop = FALSE])
squam_scores <- colMeans(logcpm[squam_genes, , drop = FALSE])

# Add to output
results <- data.frame(
  cluster = colnames(logcpm),
  adeno_score = adeno_scores,
  squam_score = squam_scores
)

ggplot(results, aes(x = squam_score, y = adeno_score)) +
  geom_smooth(method = "lm", se = FALSE, color = "darkgray", linetype = "solid") +
  geom_point(size = 2, color='blue') +
  theme_minimal() +
  labs(title = "Gene set scores", x = "Squam score", y = "Adeno score") +
  geom_text_repel(aes(label=cluster))

fit <- lm(adeno_score ~ squam_score, data = results)

slope <- coef(fit)[['squam_score']]
intercept <- coef(fit)[['(Intercept)']]

# Get unit direction vector of the regression line
# Regression line goes from (0, intercept) to (1, intercept + slope)
dx <- 1
dy <- slope
mag <- sqrt(dx^2 + dy^2)
ux <- dx / mag
uy <- dy / mag

origin_idx <- which.max(results$adeno_score)
origin <- c(results[origin_idx, 'squam_score'], results[origin_idx, 'adeno_score'])

# Compute vector from origin to each point
vx <- results$squam_score - origin[1]
vy <- results$adeno_score - origin[2]

results$proj_dist <- vx*ux + vy*uy
min_val <- min(results$proj_dist)
max_val <- max(results$proj_dist)
results$proj_param <- (results$proj_dist - min_val) / (max_val - min_val)

out_dir = '/data1/chanj3/LUAS.multiome.results/epigenetic/TCGA_modeling/out'
write.csv(results, paste0(out_dir, '/ADC_SCC_scores.csv'), row.names = FALSE)

ggplot(results, aes(x = squam_score, y = adeno_score, color = proj_param)) +
  geom_smooth(method = "lm", se = FALSE, color = "darkgray", linetype = "solid") +
  geom_point(size = 2.25) +
  theme_minimal() +
  labs(title = "ADC vs SCC gene set scores",
       x = "SCC score", y = "ADC score", color = "ADC→SCC\nscore") +
  geom_text_repel(aes(label = cluster))
