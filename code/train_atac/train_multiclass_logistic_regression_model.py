import argparse
import numpy as np
import os
import pandas as pd
import pickle
import pyreadr
import scanpy as sc
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument('tcga_logcpm_path')
parser.add_argument('peak_set_path')
parser.add_argument('output_dir') # where output model, metrics are saved
parser.add_argument('--luad_vs_lusc', action='store_true')
args = parser.parse_args()
input_path = args.tcga_logcpm_path
peaks_path = args.peak_set_path
output_dir = args.output_dir
with open(peaks_path, 'r') as f:
    peaks = [line.strip() for line in f.readlines()]
print(f'Using {len(peaks)} peaks from {peaks_path}')

print(f"Making directory {output_dir}")
os.makedirs(output_dir, exist_ok=True)

tcga_data = pyreadr.read_r(input_path)[None]
adata = sc.AnnData(tcga_data.T)
adata.obs['cancer_type'] = [s.split('_')[0] for s in adata.obs_names]
if args.luad_vs_lusc:
    cancer_types = ['LUAD', 'LUSC']
adata = adata[adata.obs['cancer_type'].isin(cancer_types)]
print('adata.shape', adata.shape)

scoring = ['accuracy', 'balanced_accuracy', 'f1_macro']
X = adata[:, np.intersect1d(adata.var_names, peaks)].to_df()
y = adata.obs['cancer_type'].values

print('X.shape', X.shape)

C_vals = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
print(f"Running cross validation with C values: {C_vals}")
C_to_scores = {}
for C in tqdm(C_vals):
    lr = LogisticRegression(C=C, class_weight='balanced', max_iter=1000)
    scores = cross_validate(lr, X, y, cv=5, scoring=scoring)
    C_to_scores[C] = scores

def plot_cv_boxplots(results, metrics=None, figsize=(12, 4), cmap=plt.cm.Blues_r, save_path=None):
    """
    results: dict like {
        0.001: {"test_accuracy": [...], "test_balanced_accuracy": [...], "test_f1_macro": [...]},
        0.01:  {...},
        ...
    }
    metrics: list of metric names to plot (default: all found in the first item)
    figsize: figure size
    cmap: colormap mapping C -> color (default Blues_r: larger C = lighter)
    """
    # Sort C values for a consistent left-to-right order
    Cs = np.array(sorted(results.keys(), key=float))
    
    # Determine which metrics to plot
    if metrics is None:
        # union across all Cs in case one C is missing a metric
        all_metrics = set()
        for d in results.values():
            all_metrics.update(d.keys())
        metrics = sorted(all_metrics)
    
    # Prepare color mapping: dark -> light as C increases (using Blues_r)
    c_to_rank = {c: i for i, c in enumerate(Cs)}
    norm = Normalize(vmin=-1, vmax=len(Cs))
    c_to_color = {c: cmap(norm(c_to_rank[c])) for c in Cs}
    
    n_panels = len(metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    
    # Collect handles for a color legend keyed by C
    legend_handles = [Patch(facecolor=c_to_color[c], edgecolor='black', label=str(c)) for c in Cs]

    for ax, metric in zip(axes, metrics):
        # For this metric, gather lists in the order of Cs
        data_for_metric = []
        colors_for_metric = []
        xlabels = []
        for c in Cs:
            vals = results[c].get(metric, None)
            if vals is None:
                # if missing, use empty list so the boxplot slot still exists
                vals = []
            data_for_metric.append(vals)
            colors_for_metric.append(c_to_color[c])
            xlabels.append(str(c))
        
        # Draw boxplots
        bp = ax.boxplot(
            data_for_metric,
            patch_artist=True,   # needed to color the boxes
            widths=0.7,
            manage_ticks=False
        )
        
        # Color each box to match its C
        for box, fc in zip(bp['boxes'], colors_for_metric):
            box.set(facecolor=fc, edgecolor='black', linewidth=1.0)
        # Make whiskers, caps, medians readable
        for elem in ('whiskers', 'caps', 'medians'):
            for artist in bp[elem]:
                artist.set(color='black', linewidth=1.0)
        for flier in bp.get('fliers', []):
            flier.set(marker='o', alpha=0.5)
        
        ax.set_title(metric, fontsize=12)
        ax.set_xticks(range(1, len(Cs) + 1))
        ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('C')
        ax.set_ylabel(metric)
        ax.grid(axis='y', linestyle=':', alpha=0.5)

    # One shared legend for C colors
    fig.legend(handles=legend_handles, title='C', loc='upper center', ncol=min(len(Cs), 8), frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.show()

# --- Example usage ---
# results = {
#     0.001: {"test_accuracy":[0.81,0.82,0.80], "test_balanced_accuracy":[0.79,0.80,0.78], "test_f1_macro":[0.77,0.76,0.78]},
#     0.01:  {"test_accuracy":[0.83,0.84,0.83], "test_balanced_accuracy":[0.81,0.82,0.80], "test_f1_macro":[0.79,0.80,0.79]},
#     0.1:   {"test_accuracy":[0.85,0.86,0.84], "test_balanced_accuracy":[0.83,0.84,0.82], "test_f1_macro":[0.81,0.82,0.80]},
#     1.0:   {"test_accuracy":[0.84,0.85,0.85], "test_balanced_accuracy":[0.82,0.83,0.83], "test_f1_macro":[0.80,0.81,0.81]},
# }
print("Plotting cross validation results...")
plot_cv_boxplots(C_to_scores, metrics=["test_accuracy","test_balanced_accuracy","test_f1_macro"],
                 figsize=(12,4.5), save_path=f'{output_dir}/cv_boxplots.png')

with open(f'{output_dir}/cv_metrics.pkl', 'wb') as f:
    pickle.dump(C_to_scores, f)

print("Training models on full data...")
C_to_model = {}
for C in C_vals:
    lr = LogisticRegression(C=C, class_weight='balanced', max_iter=1000)
    lr.fit(X, y)
    C_to_model[C] = lr
with open(f'{output_dir}/models.pkl', 'wb') as f:
    pickle.dump(C_to_model, f)
with open(f'{output_dir}/peaks.txt', 'w') as f:
    for g in peaks:
        f.write(g + '\n')
