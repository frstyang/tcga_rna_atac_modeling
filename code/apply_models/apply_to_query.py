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
parser.add_argument('--even_barplot', action='store_true')
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
    x_range = np.linspace(logits.min(), logits.max(), 100)

    try:
        density = stats.gaussian_kde(logits)
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
            ax.axvline(valley_x, color='r', linestyle='-')
            ax.axvline(median_x, color='b', linestyle='--')
        
        # Plot
        ax.hist(logits, bins=50, density=True, alpha=0.5, label=label)
        ax.plot(x_range, density_vals, 'k-', lw=2)
        ax.set_xlabel('Logits')
        ax.set_ylabel('Density')
        if show:
            plt.show()
        if showpeak:
            def get_y(frac):
                ylim = ax.get_ylim()
                return ylim[0] + frac*(ylim[1] - ylim[0])
            def get_xd(frac):
                xlim = ax.get_xlim()
                return frac*(xlim[1] - xlim[0])
            if label == 'ATAC':
                y_frac = 0.8
            else:
                y_frac = 0.6
            ax.text(valley_x + get_xd(0.02), get_y(y_frac - 0.4), f'{valley_x:.2f}', color='r', fontsize=15)
            ax.text(median_x + get_xd(0.02), get_y(y_frac), f'{median_x:.2f}', color='b', fontsize=15)
            return valley_x, median_x
    except:
        print("for some reason, kde failed")
        return np.median(logits), np.median(logits)

print("Plotting logits densities")
is_luas = rna_X.index.str.split('---').str[0].str.startswith('LUAS')
print(f"Num is_luas: {sum(is_luas)}")
plt.rcParams['font.size'] = 20
fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)
rna_valley, rna_median = plot_kde(rna_logits[is_luas], ax=ax, label='RNA', show=False)
atac_valley, atac_median = plot_kde(atac_logits[is_luas], ax=ax, label='ATAC', show=False)

print('atac_valley:', atac_valley, 'atac_median:', atac_median)
print('rna_valley:', rna_valley, 'rna_median:', rna_median)

ax.legend(fontsize=15.5)
fig.savefig(f"{args.output_dir}/logits_density_plot.png", bbox_inches='tight')

def get_sample_num(instance_name):
    sample_name = instance_name.split('---')[0]
    if sample_name.startswith('LUSC'):
        return 'S' + sample_name.split('_')[1]
    return sample_name.split('_')[1]
sample_nums = np.array([get_sample_num(instance_name) for instance_name in rna_X.index])
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
def qscale(x, q=0.95, new_qrange=6):
    q = max(q, 1-q)
    qrange = np.quantile(x, q) - np.quantile(x, 1-q)
    factor = new_qrange / qrange
    return x * factor, factor

rna_logits_valley_adj, rna_valley_f = qscale(rna_logits - rna_valley)
rna_logits_median_adj, rna_median_f = qscale(rna_logits - rna_median)
atac_logits_valley_adj, atac_valley_f = qscale(atac_logits - atac_valley)
atac_logits_median_adj, atac_median_f = qscale(atac_logits - atac_median)
adjustment_params = pd.DataFrame.from_dict(
    {
        'rna_valley': [rna_valley, rna_valley_f],
        'atac_valley': [atac_valley, atac_valley_f],
        'rna_median': [rna_median, rna_median_f],
        'atac_median': [atac_median, atac_median_f]
    },
    orient='index',
    columns=['shift', 'scale']
)
adjustment_params.to_csv(f"{args.output_dir}/adjustment_params.csv")

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
print("Calculating noise (prediction variance on perturbed data)")
print("====================================================================")
perturb_fracs = [0.05, 0.1, 0.2, 0.3, 0.5, 1]
n_trials = 100

def perturb(X, perturb_frac):
    pert_n = int(round(X.shape[1] * perturb_frac))
    inds_to_pert = np.random.permutation(X.shape[1])[:pert_n]
    new_inds = np.arange(X.shape[1])
    new_inds[np.isin(new_inds, inds_to_pert)] = inds_to_pert
    X_perturbed = X.copy()
    X_perturbed.values[:] = X_perturbed.values[:, new_inds]
    return X_perturbed

def predict_proba_adj(model, X, shift, scale):
    logits = model.decision_function(X)
    logits_adj = (logits - shift) * scale
    return sigmoid(logits_adj)

def calc_noise(X, model, perturb_frac, shift, scale, n_trials=n_trials):
    all_pprobs = []
    for i in range(n_trials):
        X_perturbed = perturb(X, perturb_frac)
        pprobs = predict_proba_adj(model, X_perturbed, shift, scale)
        all_pprobs.append(pprobs)
    all_pprobs = np.stack(all_pprobs)
    mean_var = np.mean(np.var(all_pprobs, axis=0))
    return np.sqrt(mean_var)

noise_dict = {}
for perturb_frac in perturb_fracs:
    rna_shift, rna_scale = adjustment_params.loc['rna_median']
    rna_mean_var = calc_noise(
        rna_X, rna_model, perturb_frac, rna_shift, rna_scale
    )
    atac_shift, atac_scale = adjustment_params.loc['atac_median']
    atac_mean_var = calc_noise(
        atac_X, atac_model, perturb_frac, atac_shift, atac_scale
    )
    noise_dict[perturb_frac] = [rna_mean_var, atac_mean_var]
noise_df = pd.DataFrame.from_dict(
    noise_dict, orient='index', columns=['rna_noise', 'atac_noise']
)
noise_df.to_csv(f"{args.output_dir}/noise_levels.csv")

print("====================================================================")
print("Plotting")
print("====================================================================")

def plot_stacked_probs(probs, class_labels=None, instance_labels=None,
                       n_rows=1, figsize=(12, 6), save_path=None, title=None,
                       xlabel='Sample_cluster', ax=None, cmapping=None,
                       width=0.8):
    """
    probs: (n_instances, n_classes) numpy array
    class_labels: list of class names (len = n_classes)
    instance_labels: list of sample names (len = n_instances)
    n_rows: number of rows in the subplot grid
    """
    n_instances, n_classes = probs.shape
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(n_classes)]
    if instance_labels is None:
        instance_labels = [f"S{i}" for i in range(n_instances)]
    
    # Compute grid
    if ax is None:
        n_cols = int(np.ceil(n_instances / n_rows))
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize, gridspec_kw={'hspace':0.3}, sharey=True)
        if n_rows == 1:
            axes = np.array([axes])
        axes = axes.ravel()
    else:
        axes = np.array([ax])
        fig = plt.gcf()
        n_rows = 1
        n_cols = int(np.ceil(n_instances / n_rows))

    # Colors
    if cmapping is not None:
        colors = [cmapping[cls] for cls in class_labels]
    else:
        cmap = plt.cm.get_cmap("tab20", n_classes)
        colors = [cmap(i) for i in range(n_classes)]

    # Split instances evenly among rows
    for i in range(n_rows):
        start = i * n_cols
        end = min((i + 1) * n_cols, n_instances)
        if hasattr(width, '__len__'):
            width_i = width[start:end]
            x_pos = 0.5*(np.cumsum(np.append(0, width_i))[:-1] + np.cumsum(width_i))
        else:
            width_i = width
            x_pos = range(start, end)
        ax = axes[i]
        bottom = np.zeros(end - start)

        for j in range(n_classes):
            ax.bar(x_pos, probs[start:end, j],
                   bottom=bottom, color=colors[j], label=class_labels[j], width=width_i)
            bottom += probs[start:end, j]

        ax.set_xticks(x_pos)
        ax.set_xticklabels(instance_labels[start:end], rotation=90, fontsize=8)
        if i == 0:  # only put legend once
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)

        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_ylabel('Predicted probability', fontsize=14)
        x_span = x_pos[-1] - x_pos[0]
        ax.set_xlim(x_pos[0] - 0.015*x_span, x_pos[-1] + 0.015*x_span)
    axes[-1].set_xlabel(xlabel, fontsize=14)
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
    if title is not None:
        axes[0].set_title(title, fontsize=14)
    return fig, axes

def sorted_barplot(probs_df, save_path, outer_sort_by='rna_prob', inner_sort_by='atac_prob',
                   even_alloc_per_sample=False, figsize=(20, 8)):
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
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'hspace': 0.2}, sharex=True)
    
    # Define colors and labels
    cmapping = {
        'SCC': 'goldenrod',
        'ADC': 'C4',
    }
    class_labels = list(cmapping.keys())
    if even_alloc_per_sample:
        barwidth = []
        sample_counts = data['sample'].value_counts()
        for sample in sample_order:
            count = sample_counts.loc[sample]
            barwidth.extend([1./count]*count)
    else:
        barwidth = 1
    # Plot stacked bars
    plot_stacked_probs(sorted_data_atac, class_labels=class_labels,
                       instance_labels=sorted_data.index,
                       save_path=None, xlabel='', 
                       title='ATAC classifier predictions',
                       ax=axes[0], cmapping=cmapping, width=barwidth)
    axes[0].set_ylabel('Pred. probability', fontsize=24)
    axes[0].set_xticks([])
    axes[0].get_legend().remove()
    axes[0].set_title(axes[0].get_title(), fontsize=30)

    plot_stacked_probs(sorted_data_rna, class_labels=class_labels,
                       instance_labels=sorted_data.index,
                       save_path=None, xlabel='',
                       title='RNA classifier predictions',
                       ax=axes[1], cmapping=cmapping, width=barwidth)
    axes[1].set_xticks([])
    axes[1].set_title(axes[1].get_title(), fontsize=30)
    axes[1].set_ylabel('Pred. probability', fontsize=24)
    axes[1].legend(bbox_to_anchor=(1.01, 1.02), loc="center left", fontsize=24)

    # Add sample boundaries and labels
    sample_arr = sorted_data['sample'].values
    sample_arr = np.append('None', sample_arr)
    breakpoints = np.where(sample_arr[:-1] != sample_arr[1:])[0]
    breakpoints = np.append(breakpoints, len(sample_arr))
    barwidth_cumsum = np.cumsum(np.append(0, barwidth))
    for i in range(len(breakpoints) - 1):
        bp_curr = breakpoints[i]
        bp_next = breakpoints[i+1]
        sample = sample_arr[bp_curr+1]
        if even_alloc_per_sample:
            line_x = barwidth_cumsum[bp_curr]
            label_x = 0.35*barwidth_cumsum[bp_curr] + 0.65*barwidth_cumsum[bp_next-1]
        else:
            line_x = bp_curr
            label_x = 0.35*bp_curr + 0.65*(bp_next - 1)
        for ax in axes:
            ax.axvline(line_x, color='k', linestyle='--', alpha=0.6)
            if ax == axes[1]:
                ax.text(label_x, -0.07, sample, rotation=45, ha='right', fontsize=24, rotation_mode='anchor')
    
    # Formatting
    for ax in axes:
        ax.tick_params(axis='both', labelsize=22)
        #ax.set_xlim(-1, len(sorted_data))

    fig.savefig(save_path, bbox_inches='tight')

sorted_barplot(probs_valley_adj_df, f'{args.output_dir}/sorted_barplot_valley_adjusted.png',
               even_alloc_per_sample=args.even_barplot, figsize=(26, 8))
sorted_barplot(probs_median_adj_df, f'{args.output_dir}/sorted_barplot_median_adjusted.png',
               even_alloc_per_sample=args.even_barplot, figsize=(26, 8))

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
    inner_sort_by='atac_prob',
    even_alloc_per_sample=False,
    figsize=(20, 8),
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
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'hspace': 0.2}, sharex=True)
    
    # Define colors and labels
    cmapping = {
        'Gain of ADC\nfeature': 'C4',
        'Loss of SCC\nfeature': '#a384bf',
        'Gain of SCC\nfeature': 'goldenrod',
        'Loss of ADC\nfeature': '#e0c379'
    }
    class_labels = list(cmapping.keys())
    reorder = [2, 3, 0, 1]
    sorted_atac_probs = sorted_atac_probs[:, reorder]
    sorted_rna_probs = sorted_rna_probs[:, reorder]
    class_labels = [class_labels[i] for i in reorder]
    if even_alloc_per_sample:
        barwidth = []
        sample_counts = data['sample'].value_counts()
        for sample in sample_order:
            count = sample_counts.loc[sample]
            barwidth.extend([1./count]*count)
    else:
        barwidth = 1
    # Plot stacked bars
    plot_stacked_probs(sorted_atac_probs, class_labels=class_labels,
                       instance_labels=sorted_data.index,
                       save_path=None, xlabel='', 
                       title='ATAC predictions partitioned by feature gain/loss',
                       ax=axes[0], cmapping=cmapping, width=barwidth)
    axes[0].set_ylabel('Pred. probability', fontsize=24)
    axes[0].set_xticks([])
    axes[0].get_legend().remove()
    axes[0].set_title(axes[0].get_title(), fontsize=30)
    
    plot_stacked_probs(sorted_rna_probs, class_labels=class_labels,
                       instance_labels=sorted_data.index,
                       save_path=None, xlabel='',
                       title='RNA predictions partitioned by feature gain/loss',
                       ax=axes[1], cmapping=cmapping, width=barwidth)
    axes[1].set_xticks([])
    axes[1].set_title(axes[1].get_title(), fontsize=30)
    axes[1].set_ylabel('Pred. probability', fontsize=24)
    axes[1].legend(bbox_to_anchor=(1.01, 1.02), loc="center left", fontsize=24)
    
    # Add sample boundaries and labels
    sample_arr = sorted_data['sample'].values
    sample_arr = np.append('None', sample_arr)
    breakpoints = np.where(sample_arr[:-1] != sample_arr[1:])[0]
    breakpoints = np.append(breakpoints, len(sample_arr))
    barwidth_cumsum = np.cumsum(np.append(0, barwidth))
    for i in range(len(breakpoints) - 1):
        bp_curr = breakpoints[i]
        bp_next = breakpoints[i+1]
        sample = sample_arr[bp_curr+1]
        if even_alloc_per_sample:
            line_x = barwidth_cumsum[bp_curr]
            label_x = 0.35*barwidth_cumsum[bp_curr] + 0.65*(barwidth_cumsum[bp_next-1])
        else:
            line_x = bp_curr
            label_x = 0.35*bp_curr + 0.65*(bp_next - 1)
        for ax in axes:
            ax.axvline(line_x, color='k', linestyle='--', alpha=0.6)
            if ax == axes[1]:
                ax.text(label_x, -0.07, sample, rotation=45, ha='right', fontsize=24, rotation_mode='anchor')
    
    # Formatting
    for ax in axes:
        ax.tick_params(axis='both', labelsize=22)
        #ax.set_xlim(-1, len(sorted_data))
    
    fig.savefig(save_path, bbox_inches='tight')

partitioned_sorted_barplot(
    probs_valley_adj_df,
    rna_shap_vals,
    atac_shap_vals,
    f'{args.output_dir}/partitioned_sorted_barplot_valley_adjusted.png',
    even_alloc_per_sample=args.even_barplot,
    figsize=(26,8)
)
partitioned_sorted_barplot(
    probs_median_adj_df,
    rna_shap_vals,
    atac_shap_vals,
    f'{args.output_dir}/partitioned_sorted_barplot_median_adjusted.png',
    even_alloc_per_sample=args.even_barplot,
    figsize=(26,8)
)
