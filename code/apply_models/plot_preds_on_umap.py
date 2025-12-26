import argparse
import matplotlib.pyplot as plt
import matplotlib
import mudata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from tqdm import tqdm
import warnings

def aggregate(X, assignments, groups):
    """Computes the mean of rows of X mapping to the same assignment.

    Args:
        X (np.ndarray): (n x d) matrix, num samples by num features
        assignments (pd.Series): (n,) group assignment for each sample
        groups (np.ndarray): (g,) groups

    Returns:
        np.ndarray: (g x d) matrix, g is number of groups
    """    
    n = len(assignments)
    group_to_ind = {g: i for i, g in enumerate(groups)}
    group_inds = assignments.map(group_to_ind)
    g = len(groups)
    A = sp.coo_matrix(
        (np.ones(n), (group_inds, np.arange(n))),
        shape=(g, n)
    ).toarray()
    return (A @ X) / A.sum(axis=1)[:, None]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('samplesheet_path')
    parser.add_argument('scc_probs_path')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    
    samplesheet = pd.read_csv(args.samplesheet_path)
    samples = samplesheet['name'].values
    
    sample_to_mdata_assignments = {}
    for _, row in tqdm(samplesheet.iterrows(), total=len(samplesheet)):
        sample = row['name']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdata = mudata.read(row['rna_atac_h5mu_path'])
        assignments = pd.read_csv(row['assignments_path'], index_col=0)['SEACell']
        sample_to_mdata_assignments[sample] = (mdata[assignments.index], assignments)

    scc_probs = pd.read_csv(args.scc_probs_path, index_col=0)
    sample_to_agg_adata = {}
    for sample in tqdm(samples):
        mdata, assignments = sample_to_mdata_assignments[sample]
        scc_probs_sample = scc_probs[scc_probs.index.str.split('---').str[0] == sample]
        scc_probs_sample.index = scc_probs_sample.index.str.split('---').str[1]
        groups = np.unique(assignments)
        agg_umap = aggregate(mdata.obsm['X_umap'], assignments, groups)
        agg_joint_embedding = aggregate(mdata.obsm['joint_embedding'], assignments, groups)
        agg_adata = sc.AnnData(
            X=None,
            obs=pd.DataFrame([], index=groups),
            obsm={'X_umap': agg_umap, 'joint_embedding': agg_joint_embedding}
        )
        agg_adata.obs[scc_probs_sample.columns] = scc_probs_sample
        sample_to_agg_adata[sample] = agg_adata

    # plot
    vmin = 0.1
    vmax = 0.9
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'puor', colors=['purple', 'white', 'orange']
    )
    sc_s = 60
    gp_s = 180
    ncols = 5
    
    def plot_key(key, save_path):
        nrows = int(np.ceil(len(samples) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 4.2, nrows * 4.2),
            gridspec_kw={'wspace': 0.1, 'hspace': 0.2},
            dpi=150
        )
        for i, sample in tqdm(enumerate(samples), total=len(samples)):
            ax = axes.flat[i]
            mdata, assignments = sample_to_mdata_assignments[sample]
            sc.pl.umap(mdata, ax=ax, show=False, s=sc_s)
            agg_adata = sample_to_agg_adata[sample]
            sc.pl.umap(
                agg_adata,
                color=[key],
                ax=ax,
                show=False,
                s=gp_s,
                colorbar_loc=None,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                frameon=False
            )
            ax.set_title(sample, fontsize=20)
        for j in range(i+1, nrows*ncols):
            axes.flat[j].axis('off')
        fig.subplots_adjust(right=0.96)
        cbar_ax = fig.add_axes([0.975, 0.1, 0.0125, 0.8])
        cbar_mappable = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
            cmap=cmap
        )
        fig.colorbar(cbar_mappable, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=20)
        fig.savefig(save_path, bbox_inches='tight')

    print("Plotting predictions on UMAP")
    plot_key('RNA_SCC_prob', f'{args.output_dir}/SCC_RNA_probs_on_umap.png')
    plot_key('ATAC_SCC_prob', f'{args.output_dir}/SCC_ATAC_probs_on_umap.png')

    # Calculate knn deviations
    def knn_deviation(k, key):
        devs = []
        for sample in samples:
            agg_adata = sample_to_agg_adata[sample]
            sc.pp.neighbors(agg_adata, n_neighbors=k, use_rep='joint_embedding')
            sample_n = len(agg_adata)
            dist_mtx = agg_adata.obsp['distances']
            indices = dist_mtx.indices
            indptr = dist_mtx.indptr
            devs_this_sample = []
            for i in range(sample_n):
                nn_inds = indices[indptr[i]:indptr[i+1]]
                i_val = agg_adata.obs[key].iloc[i]
                nns_vals = agg_adata.obs[key].iloc[nn_inds]
                devs_this_sample.append(np.abs(i_val - nns_vals))
            mean_dev = np.concatenate(devs_this_sample).mean()
            devs.append(mean_dev)
        return np.mean(devs)

    print("Calculating knn deviations")
    k_list = [3, 5, 10]
    dev_data = {}
    for k in k_list:
        rna_dev = knn_deviation(k, 'RNA_SCC_prob')
        atac_dev = knn_deviation(k, 'ATAC_SCC_prob')
        dev_data[k] = [rna_dev, atac_dev]
    dev_df = pd.DataFrame.from_dict(
        dev_data,
        orient='index',
        columns=['RNA_prob_knn_dev', 'ATAC_prob_knn_dev']
    )
    dev_df.to_csv(f"{args.output_dir}/SCC_probs_knn_devs.csv")
