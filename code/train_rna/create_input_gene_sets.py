import argparse
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyreadr
import re
import scanpy as sc
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('tcga_xena_path')
parser.add_argument('tcga_clindata_path')
parser.add_argument('query_genes_path')
parser.add_argument('output_dir') # where output gene sets are saved
parser.add_argument('--genes_to_remove', nargs='*', default=["XIST"])
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

exp_data = pd.read_csv(args.tcga_xena_path, sep='\t')
exp_data = exp_data[~exp_data.iloc[:, 0].str[0].str.isdigit()]
exp_data = exp_data.fillna(0)
exp_data = exp_data.set_index('sample')

query_genes = pd.read_csv(args.query_genes_path, header=None).iloc[:, 0].values
exp_data = exp_data.loc[exp_data.index.isin(query_genes)]

clinical_data = pd.read_csv(args.tcga_clindata_path, sep='\t')
clinical_data.head()

sample_to_cancertype = clinical_data[['sample', 'cancer type abbreviation']].set_index('sample').iloc[:, 0].to_dict()
exp_data = exp_data.iloc[:, exp_data.columns.isin(list(sample_to_cancertype.keys()))]
print('TCGA gene x sample', exp_data.shape)
cancertype_labels = [sample_to_cancertype[sample] for sample in exp_data.columns]

adata = sc.AnnData(exp_data.T)
adata = adata[:, np.unique(adata.var_names, return_index=True)[1]]
adata.obs['cancer_type'] = cancertype_labels
cancer_types = ['LUAD', 'LUSC']
adata = adata[adata.obs['cancer_type'].isin(cancer_types)]
print('adata.shape after restricting to LUAD or LUSC samples:', adata.shape)
print(adata.obs['cancer_type'].value_counts())

# Remove unwanted genes
if args.genes_to_remove:
    combined = re.compile("|".join(args.genes_to_remove))
    genes = [g for g in adata.var_names if not combined.fullmatch(g)]
    adata = adata[:, genes]

genes = adata.var_names
results = pd.DataFrame(index=genes, columns=['max_FDR', 'min_log2FC'])
all_pairs_fdrs = {}
all_pairs_logfcs = {}
pairs = list(combinations(cancer_types, 2))
for ct1, ct2 in tqdm(pairs, total=len(pairs)):
    sc.tl.rank_genes_groups(adata, 'cancer_type', method='wilcoxon', groups=[ct1], reference=ct2)
    df = sc.get.rank_genes_groups_df(adata, group=ct1).set_index('names')
    pair_name = f'{ct1}_vs_{ct2}'
    df = df.loc[adata.var_names]
    all_pairs_fdrs[pair_name] = df['pvals_adj'].values
    all_pairs_logfcs[pair_name] = df['logfoldchanges'].values

all_pairs_fdrs = pd.DataFrame(all_pairs_fdrs, index=adata.var_names)
all_pairs_logfcs = pd.DataFrame(all_pairs_logfcs, index=adata.var_names)
all_pairs_logfcs = all_pairs_logfcs.fillna(0)

ct_to_results = {}
FDR_threshold = 0.000001
logFC_threshold = 2
for ct in tqdm(cancer_types, total=len(cancer_types)):
    ct1_cols = [col for col in all_pairs_fdrs.columns if col.startswith(f'{ct}_vs_')]
    ct2_cols = [col for col in all_pairs_fdrs.columns if col.endswith(f'_vs_{ct}')]
    results = pd.DataFrame()
    ct_vs_rest_fdrs = all_pairs_fdrs[ct1_cols + ct2_cols]
    ct_vs_rest_logfcs = pd.concat((all_pairs_logfcs[ct1_cols], -all_pairs_logfcs[ct2_cols]), axis=1)
    results['max_FDR'] = ct_vs_rest_fdrs.max(axis=1)
    results['min_log2FC'] = ct_vs_rest_logfcs.min(axis=1)
    results = results[results['max_FDR'] < FDR_threshold]
    results = results[results['min_log2FC'] > logFC_threshold]
    print(f'{ct}: {results.shape}')
    ct_to_results[ct] = results

def save_genes(genes, path):
    with open(path, 'w') as f:
        for g in genes:
            f.write(f'{g}\n')
genes = [g for res in ct_to_results.values()
         for g in res.sort_values('min_log2FC', ascending=False).iloc[:200].index]
assert len(genes) == len(np.unique(genes))
save_genes(genes, f'{args.output_dir}/max_fdr_1e-6_min_logfc_2_top_200_per.txt')

genes = [g for res in ct_to_results.values()
         for g in res.sort_values('min_log2FC', ascending=False).index]
assert len(genes) == len(np.unique(genes))
save_genes(genes, f'{args.output_dir}/max_fdr_1e-6_min_logfc_2.txt')

# create hvg lists
n_hvgs_list = [5000, 3000, 1000, 500]
for n_hvgs in n_hvgs_list:
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs)
    save_genes(adata.var_names[adata.var['highly_variable']], f'{args.output_dir}/hvg_{n_hvgs}.txt')
