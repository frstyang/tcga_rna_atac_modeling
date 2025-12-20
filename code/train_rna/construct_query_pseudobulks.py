import argparse
from functools import reduce
import numpy as np
import os
import pandas as pd
import scanpy as sc
import re
from scipy import sparse
from scipy.io import mmwrite
import snapatac2 as snap
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('query_samplesheet')
# Format of query_samplesheet should be a csv with columns:
# name,rna_h5ad_path,assignments_path
# where rna_h5ad_path is path to an anndata with "counts" layer
# and assignments_path is path to a csv where the first column is the
# barcode and the second column is the assignment to a group.
parser.add_argument('output_dir') # where pseudobulks are output
parser.add_argument('--genes_to_remove', nargs='*', default=["XIST"])
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

query_samplesheet_df = pd.read_csv(args.query_samplesheet)
all_pseudobulks = []
all_group_names = []
all_gene_names = []
for _, row in tqdm(query_samplesheet_df.iterrows(), total=len(query_samplesheet_df)):
    name = row['name']
    rna_h5ad_path = row['rna_h5ad_path']
    assignments = pd.read_csv(row['assignments_path'], sep=',', index_col=0)
    assignments = assignments.iloc[:, 0] # convert to pandas Series

    adata = sc.read_h5ad(rna_h5ad_path)
    adata = adata[assignments.index]
    counts = adata.X
    print('counts.shape:', counts.shape)
    
   # Ensure CSR for fast row slicing & summation
    if not sparse.isspmatrix_csr(counts):
        counts = counts.tocsr()

    # Get unique groups and create mapping
    group_names, inverse = np.unique(assignments.values, return_inverse=True)
    group_names = [f'{name}---{gn}' for gn in group_names]
    n_groups = len(group_names)

    # Build group membership matrix (n_cells × n_groups)
    G = sparse.csr_matrix(
        (np.ones_like(inverse), (np.arange(len(inverse)), inverse)),
        shape=(counts.shape[0], n_groups)
    )

    # Do: (Gᵀ × counts) = aggregation
    agg_mat = G.T @ counts 

    all_pseudobulks.append(agg_mat)
    all_group_names.append(group_names)
    all_gene_names.append(adata.var_names)

gene_names_union = reduce(np.union1d, all_gene_names)
for i in tqdm(range(len(all_pseudobulks)), total=len(all_pseudobulks)):
    old_pseudobulks = all_pseudobulks[i]
    old_genes = all_gene_names[i]
    N = old_pseudobulks.shape[0]
    G = len(gene_names_union)
    old_gene_inds = np.searchsorted(gene_names_union, old_genes)
    assert np.all(gene_names_union[old_gene_inds] == old_genes)
    coo = old_pseudobulks.tocoo()
    new_pseudobulks = sparse.coo_matrix(
        (coo.data, (coo.row, old_gene_inds[coo.col])),
        shape=(N, G),
        dtype=np.int32
    )
    all_pseudobulks[i] = new_pseudobulks
all_pseudobulks = sparse.vstack(all_pseudobulks).tocsr()
all_group_names = [gn for gns in all_group_names for gn in gns]

# Remove unwanted genes
if args.genes_to_remove:
    combined = re.compile("|".join(f"{p}" for p in args.genes_to_remove))
    mask = ~np.array([bool(combined.fullmatch(g)) for g in gene_names_union])
    gene_names_union = gene_names_union[mask]
    all_pseudobulks = all_pseudobulks[:, mask]

mmwrite(f'{args.output_dir}/query_pseudobulks.mtx', all_pseudobulks)
def write_lst(lst, path):
    with open(path, 'w') as f:
        for x in lst:
            f.write(f'{x}\n')
write_lst(all_group_names, f'{args.output_dir}/query_group_names.txt')
write_lst(gene_names_union, f'{args.output_dir}/query_genes.txt')
