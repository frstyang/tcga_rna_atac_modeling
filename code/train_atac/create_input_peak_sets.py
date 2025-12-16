import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyreadr
import scanpy as sc
import snapatac2 as snap

parser = argparse.ArgumentParser()
parser.add_argument('tcga_logcpm_path')
parser.add_argument('peaks_metadata_path')
parser.add_argument('output_dir') # where output peak sets are saved
parser.add_argument('--n_top_peaks', type=int)
parser.add_argument('--query_logcpm_path')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
tcga_data = pyreadr.read_r(args.tcga_logcpm_path)[None]
tcga_peaks_metadata = pd.read_csv(args.peaks_metadata_path)
tcga_peaks_metadata.index = (
    tcga_peaks_metadata['seqnames'] + ':'
    + tcga_peaks_metadata['start'].astype(str) + '-'
    + tcga_peaks_metadata['end'].astype(str)
)

adata = sc.AnnData(tcga_data.T)
adata.var[['score', 'annotation', 'GC']] = \
        tcga_peaks_metadata.loc[adata.var_names, ['score', 'annotation', 'GC']]

if args.n_top_peaks:
    def calc_sparsity(M):
        if isinstance(M, pd.DataFrame):
            M = M.to_numpy()
        zeros = (M == M.min())
        return np.mean(zeros)
    # peaks x metacells
    query_data = pyreadr.read_r(args.query_logcpm_path)[None]
    query_mean_accs = query_data.mean(axis=1)
    query_mean_accs_sorted = query_mean_accs.sort_values(
        ascending=False)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=150)
    n = len(query_mean_accs_sorted)
    ax.plot(range(1, n+1), query_mean_accs_sorted)
    ax.set_ylabel('Accessibility (logCPM)')
    ax.set_xlabel('Peak rank')
    ax.axvline(args.n_top_peaks, color='red', linestyle='--')
    ax.set_title('Peak accessibility, mean over query data')
    fig.savefig(f'{args.output_dir}/query_accessibility_per_peak_sorted.png')
    print(f'Subsetting to {args.n_top_peaks} top peaks')
    peaks = query_mean_accs_sorted.index[:args.n_top_peaks]
    print(f'TCGA sparsity prior to subsetting: {calc_sparsity(adata.X)}')
    print(f'Query sparsity prior to subsetting: {calc_sparsity(query_data)}')
    adata = adata[:, peaks]
    query_data = query_data.loc[peaks]
    print(f'TCGA sparsity after subsetting: {calc_sparsity(adata.X)}')
    print(f'Query sparsity after subsetting: {calc_sparsity(query_data)}')

print('adata.shape:', adata.shape)
adata = adata[:, adata.var['annotation'] != 'Promoter']
print('adata.shape after restricting to peaks not annotated as Promoter:', adata.shape)
adata = adata[adata.obs_names.str.startswith('LUAD') | adata.obs_names.str.startswith('LUSC')]
print('adata.shape after restricting to LUAD or LUSC samples:', adata.shape)

def write_peaks(peaks, filename):
    with open(filename, 'w') as f:
        for peak in peaks:
            f.write(f'{peak}\n')

n_hvps = [50000, 20000, 10000, 5000]
for n in n_hvps:
    sc.pp.highly_variable_genes(adata, n_top_genes=n)
    print(f'Number of HVPs: {np.sum(adata.var["highly_variable"])}')
    peaks = adata.var_names[adata.var['highly_variable']]
    write_peaks(peaks, f'{args.output_dir}/hvp_{n}.txt')

adata.obs['cancer_type'] = adata.obs_names.str.split('_').str[0]
print(adata.obs['cancer_type'].value_counts())

# sc.tl.rank_genes_groups "Expects logarithmized data"
sc.tl.rank_genes_groups(adata, 'cancer_type', method='wilcoxon')
luad_daps_df = sc.get.rank_genes_groups_df(adata, group='LUAD')
luad_daps_df = luad_daps_df[(luad_daps_df['pvals_adj'] < 1e-6) & (np.abs(luad_daps_df['logfoldchanges']) > 1)]
print("number of differential peaks", luad_daps_df.shape)

n_top_per = [2500, 1500]
for n in n_top_per:
    pos_daps_df = luad_daps_df[luad_daps_df['logfoldchanges'] > 0]
    neg_daps_df = luad_daps_df[luad_daps_df['logfoldchanges'] < 0]
    top_peaks = pd.concat([
        pos_daps_df.sort_values('logfoldchanges', ascending=False).head(n),
        neg_daps_df.sort_values('logfoldchanges', ascending=True).head(n)
    ])
    print(f"diff peak set size: {len(top_peaks)}")
    write_peaks(top_peaks['names'], f'{args.output_dir}/fdr_1e-6_top_{n}_per.txt')
