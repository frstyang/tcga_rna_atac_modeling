import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pyreadr
from scipy import stats
from scipy.signal import find_peaks
import shap

parser = argparse.ArgumentParser()
parser.add_argument('output_dir')
parser.add_argument('atac_lognorm_path')
parser.add_argument('rna_lognorm_path')
parser.add_argument('atac_model_dir')
parser.add_argument('atac_C')
parser.add_argument('rna_model_dir')
parser.add_argument('rna_C')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

with open(f'{args.atac_model_dir}/models.pkl', 'rb') as f:
    models = pickle.load(f)
atac_model = models[float(args.atac_C)]

with open(f'{args.rna_model_dir}/models.pkl', 'rb') as f:
    models = pickle.load(f)
rna_model = models[float(args.rna_C)]

with open(f'{args.output_dir}/models_info.txt', 'w') as f:
    f.write(f'rna: {os.path.basename(args.rna_model_dir)}, {args.rna_C}\n')
    f.write(f'atac: {os.path.basename(args.atac_model_dir)}, {args.atac_C}')

print("Loading data")
def subset_feats(exp_df, feats, fill_value=0):
    return exp_df.reindex(columns=feats, fill_value=fill_value)

# Load lognormed ATAC data
atac_lognorm = pyreadr.read_r(args.atac_lognorm_path)[None].T
atac_X = subset_feats(atac_lognorm, atac_model.feature_names_in_)

# Load lognormed RNA data
rna_lognorm = pyreadr.read_r(args.rna_lognorm_path)[None].T
rna_X = subset_feats(rna_lognorm, rna_model.feature_names_in_)

print('atac_lognorm_path:', args.atac_lognorm_path)
print('rna_lognorm_path:', args.rna_lognorm_path)
print('atac_X.shape:', atac_X.shape)
print('rna_X.shape:', rna_X.shape)

assert np.all(atac_X.index.isin(rna_X.index)) & (len(atac_X) == len(rna_X))
atac_X = atac_X.loc[rna_X.index]

print("Predicting logits")
# Predict logits (sklearn's decision function returns logits)
atac_logits = atac_model.decision_function(atac_X)
rna_logits = rna_model.decision_function(rna_X)

def plot_kde(logits, ax=None, label='ATAC', show=True, showpeak=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # Create density estimate
    density = stats.gaussian_kde(logits)
    x_range = np.linspace(logits.min(), logits.max(), 100)
    density_vals = density(x_range)

    if showpeak:
        # Find local minima in density
        peaks, _ = find_peaks(-density_vals)  # negative to find valleys instead of peaks
        if len(peaks) > 0:
            # Get the most prominent valley
            valley_x = x_range[peaks[np.argmax(density_vals[peaks])]]
        else:
            # If no clear valley, use median
            valley_x = np.median(logits)
        median_x = np.median(logits)
        ax.axvline(valley_x, color='r', linestyle='--')
        ax.axvline(median_x, color='b', linestyle='--')
        def get_y(frac):
            ylim = ax.get_ylim()
            return ylim[0] + frac*(ylim[1] - ylim[0])
        def get_xd(frac):
            xlim = ax.get_xlim()
            return frac*(xlim[1] - xlim[0])
        ax.text(valley_x + get_xd(0.03), get_y(0.6), f'{valley_x:.2f}', transform=ax.transAxes, color='r')
        ax.text(median_x + get_xd(0.03), get_y(0.6), f'{median_x:.2f}', transform=ax.transAxes, color='b')
    
    # Plot
    ax.hist(logits, bins=50, density=True, alpha=0.5, label=label)
    ax.plot(x_range, density_vals, 'k-', lw=2)
    ax.set_xlabel('Logits')
    ax.set_ylabel('Density')
    if show:
        plt.show()
    if showpeak:
        return valley_x, median_x

print("Plotting logits densities")
plt.rcParams['font.size'] = 20
fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
atac_valley, atac_median = plot_kde(atac_logits, ax=ax, label='RNA', show=False)
rna_valley, rna_median = plot_kde(rna_logits, ax=ax, label='ATAC', show=False)

print('atac_valley:', atac_valley, 'atac_median:', atac_median)
print('rna_valley:', rna_valley, 'rna_median:', rna_median)

ax.legend(fontsize=15.5)
fig.savefig(f"{args.output_dir}/logits_density_plot.png", bbox_inches='tight')

sample_nums = rna_X.index.str.split('---').str[0].str.split('_').str[1]
sample_means = pd.Series(rna_logits).groupby(sample_nums).mean()
sample_nums_order = sample_means.sort_values(ascending=False).index
def plot_logits(sample_nums, logits, ax=None, modality='ATAC', sample_nums_order=None, show=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    # Assign y levels to snums
    unique_sample_nums = sorted(set(sample_nums))
    if sample_nums_order is not None:
        sample_num_to_y = {sample_num: i for i, sample_num in enumerate(sample_nums_order)}
    else:
        sample_num_to_y = {sample_num: i for i, sample_num in enumerate(unique_sample_nums)}
    
    # Scatter plot
    for i, sample_num in enumerate(unique_sample_nums):
        mask = sample_nums == sample_num
        ax.scatter(
            logits[mask],
            [sample_num_to_y[sample_num]] * mask.sum(),
            color=plt.cm.tab20(i),
            label=sample_num
        )
    yrange = range(len(unique_sample_nums))
    y_to_sample_num = {y: sample_num for sample_num, y in sample_num_to_y.items()}
    ax.set_yticks(yrange, [y_to_sample_num[y] for y in yrange])
    ax.set_xlabel(f"{modality} logits")
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=16, labelspacing=0.2)
    ax.set_title(f'Metacells {modality} logits')
    if show:
        plt.show()

print("Plotting logits scatterplot")
plt.rcParams['font.size'] = 18
n_samples = len(np.unique(sample_nums))
fig, axes = plt.subplots(1, 2, figsize=(12, 0.35*n_samples), dpi=150, gridspec_kw={'wspace': 0.16})
plot_logits(sample_nums, rna_logits, ax=axes[0], modality='RNA', sample_nums_order=sample_nums_order, show=False)
axes[0].axvline(rna_valley, color='r', linestyle='--')
axes[0].axvline(rna_median, color='b', linestyle='--')
axes[0].get_legend().remove()

plot_logits(sample_nums, atac_logits, ax=axes[1], sample_nums_order=sample_nums_order, show=False)
axes[1].axvline(atac_valley, color='r', linestyle='--')
axes[1].axvline(atac_median, color='b', linestyle='--')

fig.savefig(f"{args.output_dir}/logits_scatter_plot.png", bbox_inches='tight')

print("Adjusting logits")
def qscale(x, q=0.95, qrange=5):
    q = max(q, 1-q)
    qrange = np.quantile(x, q) - np.quantile(x, 1-q)
    return x * (5 / qrange)

rna_logits_valley_adj = qscale(rna_logits - rna_valley)
rna_logits_median_adj = qscale(rna_logits - rna_median)
atac_logits_valley_adj = qscale(atac_logits - atac_valley)
atac_logits_median_adj = qscale(atac_logits - atac_median)

plt.rcParams['font.size'] = 20
fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
plot_kde(rna_logits_valley_adj, ax=ax, label='RNA', show=False, showpeak=False)
plot_kde(atac_logits_valley_adj, ax=ax, label='ATAC', show=False, showpeak=False)
ax.legend(fontsize=15.5)
ax.set_xlabel('Valley-adjusted logits')
fig.savefig(f"{args.output_dir}/logits_density_plot_valley_adjusted.png", bbox_inches='tight')

plt.rcParams['font.size'] = 20
fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
plot_kde(rna_logits_median_adj, ax=ax, label='RNA', show=False, showpeak=False)
plot_kde(atac_logits_median_adj, ax=ax, label='ATAC', show=False, showpeak=False)
ax.legend(fontsize=15.5)
ax.set_xlabel('Median-adjusted logits')
fig.savefig(f"{args.output_dir}/logits_density_plot_median_adjusted.png", bbox_inches='tight')

print("Saving adjusted probabilities")
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

probs_valley_adj_df = pd.DataFrame(
    {'RNA_SCC_prob': sigmoid(rna_logits_valley_adj),
     'ATAC_SCC_prob': sigmoid(atac_logits_valley_adj)},
    index=rna_X.index
)
probs_valley_adj_df.to_csv(f"{args.output_dir}/SCC_probs_valley_adjusted.csv")
probs_median_adj_df = pd.DataFrame(
    {'RNA_SCC_prob': sigmoid(rna_logits_median_adj),
     'ATAC_SCC_prob': sigmoid(atac_logits_median_adj)},
    index=rna_X.index
)
probs_median_adj_df.to_csv(f"{args.output_dir}/SCC_probs_median_adjusted.csv")

print("====================================================================")
print("Plotting")
print("====================================================================")

def plot_stacked_probs(probs, class_labels=None, sample_labels=None,
                       n_rows=1, figsize=(12, 6), save_path=None, title=None,
                       xlabel='Sample_cluster', sort_by_y=None, ax=None,
                       cmapping=None, renaming=None, width=0.8):
    """
    probs: (n_samples, n_classes) numpy array
    class_labels: list of class names (len = n_classes)
    sample_labels: list of sample names (len = n_samples)
    n_rows: number of rows in the subplot grid
    """
    n_samples, n_classes = probs.shape
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(n_classes)]
    if sample_labels is None:
        sample_labels = [f"S{i}" for i in range(n_samples)]
    if sort_by_y is not None:
        order = np.argsort(probs[:, sort_by_y])
        probs = probs[order]
        sample_labels = [sample_labels[i] for i in order]

    if renaming is not None:
        class_labels = [renaming.get(cls, cls) for cls in class_labels]
    
    # Compute grid
    if ax is None:
        n_cols = int(np.ceil(n_samples / n_rows))
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize, gridspec_kw={'hspace':0.3}, sharey=True)
        if n_rows == 1:
            axes = np.array([axes])
        axes = axes.ravel()
    else:
        axes = np.array([ax])
        fig = plt.gcf()
        n_rows = 1
        n_cols = int(np.ceil(n_samples / n_rows))

    # Colors
    if cmapping is not None:
        colors = [cmapping[cls] for cls in class_labels]
    else:
        cmap = plt.cm.get_cmap("tab20", n_classes)
        colors = [cmap(i) for i in range(n_classes)]

    # Split samples evenly among rows
    for i in range(n_rows):
        start = i * n_cols
        end = min((i + 1) * n_cols, n_samples)
        ax = axes[i]
        bottom = np.zeros(end - start)

        for j in range(n_classes):
            ax.bar(range(start, end), probs[start:end, j],
                   bottom=bottom, color=colors[j], label=class_labels[j], width=width)
            bottom += probs[start:end, j]

        ax.set_xticks(range(start, end))
        ax.set_xticklabels(sample_labels[start:end], rotation=90, fontsize=8)
        if i == 0:  # only put legend once
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)

        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_ylabel('Predicted probability', fontsize=14)
    axes[-1].set_xlabel(xlabel, fontsize=14)
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    if title is not None:
        axes[0].set_title(title, fontsize=14)
    return fig, axes

def sorted_barplot(probs_df, save_path, outer_sort_by='rna_prob', inner_sort_by='atac_prob'):
    index = probs_df.index
    data = pd.DataFrame({
        'sample': index.str.split('---').str[0],
        'cellgroup': index.str.split('---').str[1],
        'rna_prob': probs_df['RNA_SCC_prob'],
        'atac_prob': probs_df['ATAC_SCC_prob']
    })
    sample_means = data.groupby('sample')[outer_sort_by].mean()
    sample_order = sample_means.sort_values(ascending=True).index

    sorted_data = []
    for sample in sample_order:
        sample_data = data[data['sample'] == sample].copy()
        order = sample_data[inner_sort_by].argsort()
        sorted_data.append(sample_data.iloc[order])

    sorted_data = pd.concat(sorted_data)
    sorted_data_rna = pd.concat((sorted_data['rna_prob'], 1 - sorted_data['rna_prob']), axis=1).values
    sorted_data_atac = pd.concat((sorted_data['atac_prob'], 1 - sorted_data['atac_prob']), axis=1).values

    # Create the plot
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), gridspec_kw={'hspace': 0.18}, sharex=True)
    
    # Define colors and labels
    cmapping = {
        'SCC': 'goldenrod',
        'ADC': 'C4',
    }
    class_labels = list(cmapping.keys())
    barwidth = 1
    # Plot stacked bars
    plot_stacked_probs(sorted_data_atac, class_labels=class_labels,
                       sample_labels=sorted_data.index,
                       save_path=None, xlabel='', 
                       title='ATAC predictions',
                       ax=axes[0], cmapping=cmapping, width=barwidth)
    axes[0].set_ylabel('Pred. probability', fontsize=24)
    axes[0].set_xticks([])
    axes[0].get_legend().remove()
    axes[0].set_title(axes[0].get_title(), fontsize=26)

    plot_stacked_probs(sorted_data_rna, class_labels=class_labels,
                       sample_labels=sorted_data.index,
                       save_path=None, xlabel='',
                       title='RNA predictions',
                       ax=axes[1], cmapping=cmapping, width=barwidth)
    axes[1].set_xticks([])
    axes[1].set_title(axes[1].get_title(), fontsize=26)
    axes[1].set_ylabel('Pred. probability', fontsize=24)
    axes[1].legend(bbox_to_anchor=(1.01, 1.02), loc="center left", fontsize=22)

    # Add sample boundaries and labels
    prev_sample = None
    for i, sample in enumerate(sorted_data['sample']):
        if sample != prev_sample:
            for ax in axes:
                ax.axvline(i-0.5, color='k', linestyle='--', alpha=0.6)
                if ax == axes[1]:
                    ax.text(i + 0.03, -0.05, sample, rotation=45, ha='right', fontsize=16, rotation_mode='anchor')
        prev_sample = sample
    
    # Formatting
    for ax in axes:
        ax.tick_params(axis='both', labelsize=22)
        ax.set_xlim(-1, len(sorted_data))

    fig.savefig(save_path, bbox_inches='tight')

sorted_barplot(probs_valley_adj_df, f'{args.output_dir}/sorted_barplot_valley_adjusted.png')
sorted_barplot(probs_median_adj_df, f'{args.output_dir}/sorted_barplot_median_adjusted.png')

print("Plot partitioned predictions")
def partition_probs(probs, coef, shap_vals):
    """
    Partition predicted probabilities based on SHAP values and coefficient signs.
    
    Args:
        probs: (n,) array of predicted probabilities
        coef: (p,) array of model coefficients 
        shap_vals: (n,p) array of SHAP values
    
    Returns:
        partitioned: (n,4) array where columns represent:
            1. Negative SHAP values where coef < 0 (contributes to 1-probs)
            2. Negative SHAP values where coef > 0 (contributes to 1-probs)
            3. Positive SHAP values where coef > 0 (contributes to probs)
            4. Positive SHAP values where coef < 0 (contributes to probs)
    """
    # Initialize output array
    n = len(probs)
    partitioned = np.zeros((n, 4))
    
    # Get coefficient signs
    coef_neg = coef < 0
    coef_pos = coef >= 0
    
    # For each sample
    for i in range(n):
        shap_row = shap_vals[i]
        
        # Get SHAP value signs
        shap_neg = shap_row < 0
        shap_pos = shap_row >= 0
        
        # Calculate sums for each partition
        neg_coef_neg_shap = np.sum(shap_row[coef_neg & shap_neg])
        pos_coef_neg_shap = np.sum(shap_row[coef_pos & shap_neg])
        pos_coef_pos_shap = np.sum(shap_row[coef_pos & shap_pos])
        neg_coef_pos_shap = np.sum(shap_row[coef_neg & shap_pos])
        
        # Scale negative contributions to sum to 1-prob
        neg_total = neg_coef_neg_shap + pos_coef_neg_shap
        if neg_total != 0:
            partitioned[i,0] = (1 - probs[i]) * (neg_coef_neg_shap / neg_total)
            partitioned[i,1] = (1 - probs[i]) * (pos_coef_neg_shap / neg_total)
        else:
            partitioned[i,:2] = 0.5 * (1 - probs[i])
        
        # Scale positive contributions to sum to prob
        pos_total = pos_coef_pos_shap + neg_coef_pos_shap
        if pos_total != 0:
            partitioned[i,2] = probs[i] * (pos_coef_pos_shap / pos_total)
            partitioned[i,3] = probs[i] * (neg_coef_pos_shap / pos_total)
        else:
            partitioned[i,2:] = 0.5 * probs[i]
            
    return partitioned
    
atac_coef = atac_model.coef_[0]
rna_coef = rna_model.coef_[0]
atac_intercept = atac_model.intercept_[0]
rna_intercept = rna_model.intercept_[0]

# compute ATAC shap vals
le = shap.LinearExplainer((atac_coef, atac_intercept), masker=atac_X)
atac_shap_vals = le.shap_values(atac_X)

# compute RNA shap vals
le = shap.LinearExplainer((rna_coef, rna_intercept), masker=rna_X)
rna_shap_vals = le.shap_values(rna_X)

os.makedirs(f'{args.output_dir}/shap', exist_ok=True)
pd.DataFrame(atac_shap_vals, index=atac_X.index, columns=atac_model.feature_names_in_).to_csv(
    f'{args.output_dir}/shap/atac_shap_vals.csv')
pd.DataFrame(rna_shap_vals, index=rna_X.index, columns=rna_model.feature_names_in_).to_csv(
    f'{args.output_dir}/shap/rna_shap_vals.csv')
pd.Series(atac_coef, index=atac_model.feature_names_in_).to_csv(
    f'{args.output_dir}/shap/atac_coefs.csv')
pd.Series(rna_coef, index=rna_model.feature_names_in_).to_csv(
    f'{args.output_dir}/shap/rna_coefs.csv')

def partitioned_sorted_barplot(
    probs_df,
    rna_shap_vals,
    atac_shap_vals,
    save_path,
    outer_sort_by='rna_prob',
    inner_sort_by='atac_prob'
):
    # get metacell-level partitioned probs
    atac_partitioned_probs = partition_probs(
        probs_df['ATAC_SCC_prob'], atac_coef, atac_shap_vals 
    )
    rna_partitioned_probs = partition_probs(
        probs_df['RNA_SCC_prob'], rna_coef, rna_shap_vals 
    )

    # Get sample and metacell info into a DataFrame
    index = probs_df.index
    data = pd.DataFrame({
        'sample': index.str.split('---').str[0],
        'cellgroup': index.str.split('---').str[1],
        'rna_prob': probs_df['RNA_SCC_prob'],
        'atac_prob': probs_df['ATAC_SCC_prob']
    })
    
    # Sort samples by mean RNA prob
    sample_means = data.groupby('sample')[outer_sort_by].mean()
    sample_order = sample_means.sort_values(ascending=True).index
    
    # Sort metacells within each sample by prob
    sorted_data = []
    sorted_atac_probs = []
    sorted_rna_probs = []
    for sample in sample_order:
        mask = data['sample'] == sample
        sample_data = data[mask].copy()
        order = sample_data[inner_sort_by].argsort()
        sorted_data.append(sample_data.iloc[order])
        
        # Get indices for the partitioned probabilities
        original_indices = np.where(mask)[0][order]
        sorted_atac_probs.append(atac_partitioned_probs[original_indices])
        sorted_rna_probs.append(rna_partitioned_probs[original_indices])
    
    sorted_data = pd.concat(sorted_data)
    sorted_atac_probs = np.vstack(sorted_atac_probs)
    sorted_rna_probs = np.vstack(sorted_rna_probs)
    
    # Create the plot
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), gridspec_kw={'hspace': 0.18}, sharex=True)
    
    # Define colors and labels
    cmapping = {
        'Active ADC\nfeature': 'C4',
        'Inactive SCC\nfeature': '#a384bf',
        'Active SCC\nfeature': 'goldenrod',
        'Inactive ADC\nfeature': '#e0c379'
    }
    class_labels = list(cmapping.keys())
    reorder = [2, 3, 0, 1]
    sorted_atac_probs = sorted_atac_probs[:, reorder]
    sorted_rna_probs = sorted_rna_probs[:, reorder]
    class_labels = [class_labels[i] for i in reorder]
    barwidth = 1
    # Plot stacked bars
    plot_stacked_probs(sorted_atac_probs, class_labels=class_labels,
                       sample_labels=sorted_data.index,
                       save_path=None, xlabel='', 
                       title='ATAC predictions partitioned by contribution type',
                       ax=axes[0], cmapping=cmapping, width=barwidth)
    axes[0].set_ylabel('Pred. probability', fontsize=24)
    axes[0].set_xticks([])
    axes[0].get_legend().remove()
    axes[0].set_title(axes[0].get_title(), fontsize=26)
    
    plot_stacked_probs(sorted_rna_probs, class_labels=class_labels,
                       sample_labels=sorted_data.index,
                       save_path=None, xlabel='',
                       title='RNA predictions partitioned by contribution type',
                       ax=axes[1], cmapping=cmapping, width=barwidth)
    axes[1].set_xticks([])
    axes[1].set_title(axes[1].get_title(), fontsize=26)
    axes[1].set_ylabel('Pred. probability', fontsize=24)
    axes[1].legend(bbox_to_anchor=(1.01, 1.02), loc="center left", fontsize=22)
    
    # Add sample boundaries and labels
    prev_sample = None
    for i, sample in enumerate(sorted_data['sample']):
        if sample != prev_sample:
            for ax in axes:
                ax.axvline(i-0.5, color='k', linestyle='--', alpha=0.6)
                if ax == axes[1]:
                    ax.text(i + 0.03, -0.05, sample, rotation=45, ha='right', fontsize=16, rotation_mode='anchor')
        prev_sample = sample
    
    # Formatting
    for ax in axes:
        ax.tick_params(axis='both', labelsize=22)
        ax.set_xlim(-1, len(sorted_data))
    
    fig.savefig(save_path, bbox_inches='tight')

partitioned_sorted_barplot(
    probs_valley_adj_df,
    rna_shap_vals,
    atac_shap_vals,
    f'{args.output_dir}/partitioned_sorted_barplot_valley_adjusted.png'
)
partitioned_sorted_barplot(
    probs_median_adj_df,
    rna_shap_vals,
    atac_shap_vals,
    f'{args.output_dir}/partitioned_sorted_barplot_median_adjusted.png'
)