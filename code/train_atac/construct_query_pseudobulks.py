import argparse
import numpy as np
import os
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmwrite
import snapatac2 as snap
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('peaks_bedfile') 
parser.add_argument('query_samplesheet')
# Format of query_samplesheet should be a csv with columns:
# name,atac_h5ad_path,assignments_path
# where atac_h5ad_path is path to a SnapATAC2 h5ad file
# and assignments_path is path to a csv where the first column is the
# barcode and the second column is the assignment to a group.
parser.add_argument('output_dir') # where pseudobulks are output
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

query_samplesheet_df = pd.read_csv(args.query_samplesheet)
all_pseudobulks = []
all_group_names = []
for _, row in tqdm(query_samplesheet_df.iterrows(), total=len(query_samplesheet_df)):
    name = row['name']
    atac_h5ad_path = row['atac_h5ad_path']
    assignments = pd.read_csv(row['assignments_path'], sep=',', index_col=0)
    assignments = assignments.iloc[:, 0] # convert to pandas Series

    data = snap.read(atac_h5ad_path, backed=None)
    data = data[assignments.index]
    peak_mat = snap.pp.make_peak_matrix(data, peak_file=args.peaks_bedfile).X
    print('peak_mat.shape:', peak_mat.shape)
    
   # Ensure CSR for fast row slicing & summation
    if not sparse.isspmatrix_csr(peak_mat):
        peak_mat = peak_mat.tocsr()

    # Get unique groups and create mapping
    group_names, inverse = np.unique(assignments.values, return_inverse=True)
    group_names = [f'{name}---{gn}' for gn in group_names]
    n_groups = len(group_names)

    # Build group membership matrix (n_cells × n_groups)
    G = sparse.csr_matrix(
        (np.ones_like(inverse), (np.arange(len(inverse)), inverse)),
        shape=(peak_mat.shape[0], n_groups)
    )

    # Do: (Gᵀ × peak_mat) = aggregation
    agg_mat = G.T @ peak_mat

    all_pseudobulks.append(agg_mat)
    all_group_names.append(group_names)

peaks_df = pd.read_csv(args.peaks_bedfile, sep='\t', header=None)
peaks = peaks_df[0] + ':' + peaks_df[1].astype(str) + '-' + peaks_df[2].astype(str)
all_pseudobulks = sparse.vstack(all_pseudobulks).tocsr()
all_group_names = [gn for gns in all_group_names for gn in gns]
mmwrite(f'{args.output_dir}/query_pseudobulks.mtx', all_pseudobulks)
def write_lst(lst, path):
    with open(path, 'w') as f:
        for x in lst:
            f.write(f'{x}\n')
write_lst(all_group_names, f'{args.output_dir}/query_group_names.txt')
write_lst(peaks, f'{args.output_dir}/query_peaks.txt')
