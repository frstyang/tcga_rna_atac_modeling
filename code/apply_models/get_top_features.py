import argparse
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def get_feature_contributions_of_type(probs, shap_vals, coefs, shap_sign, coef_sign):
    """Gets overall feature contributions of a given type, defined by sign of
    Shapley value and sign of coefficient. E.g. shap_sign=1, coef_sign=1 corresponds
    to features that contribute positively by being present, while shap_sign=1,
    coef_sign=-1 corresponds to features that contribute positively by being absent.

    Args:
        probs (pd.Series): (n_features,) probabilities
        shap_vals (pd.DataFrame): (n_instances, n_features) Shapley values
        coefs (pd.Series): (n_features,) feature coefficients
        shap_sign (int): 1 or -1
        coef_sign (int): 1 or -1

    Returns:
        pd.Series: (n_features,) feature contributions of provided type, averaged
            across instances
    """
    if shap_sign == -1:
        probs = 1 - probs
    shap_vals = shap_vals * shap_sign
    shap_vals = shap_vals.clip(0, None)
    shap_fracs = shap_vals.div(shap_vals.sum(axis=1), axis=0)
    shap_contribs = shap_fracs.multiply(probs, axis=0)

    coefs = coefs * coef_sign
    coefs = coefs.clip(0, None)
    shap_contribs = shap_contribs.multiply(coefs > 0, axis=1)
    shap_contribs = shap_contribs.mean(axis=0)
    shap_contribs.name = f'contrib_{shap_sign}_{coef_sign}'
    return shap_contribs

def get_top_feature_contributions(feat_contribs, S=3):
    """Gets top feature contributions using an elbow point method.

    Args:
        feat_contribs (pd.Series): (n_features,) contributions
        S (float): how liberal elbow position is (higher = more clearance)

    Returns:
        pd.Series: (n_top_features,) top contributions in descending order
    """
    feat_contribs_sorted = feat_contribs.sort_values(ascending=False)
    x = np.arange(1, len(feat_contribs) + 1)
    kl = KneeLocator(
        x,
        feat_contribs_sorted,
        direction='decreasing',
        curve='convex',
        S=S
    )
    return feat_contribs_sorted.iloc[:kl.knee]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('shap_dir')
    parser.add_argument('probs_file')
    parser.add_argument('--S', help='KneeLocator parameter', type=float, default=3)
    args = parser.parse_args()
    
    rna_coefs = pd.read_csv(f'{args.shap_dir}/rna_coefs.csv', index_col=0).iloc[:, 0]
    rna_shap_vals = pd.read_csv(f'{args.shap_dir}/rna_shap_vals.csv', index_col=0)
    atac_coefs = pd.read_csv(f'{args.shap_dir}/atac_coefs.csv', index_col=0).iloc[:, 0]
    atac_shap_vals = pd.read_csv(f'{args.shap_dir}/atac_shap_vals.csv', index_col=0)
    rna_atac_probs = pd.read_csv(args.probs_file, index_col=0)
    rna_atac_probs.rename(
        {'RNA_SCC_prob': 'rna', 'ATAC_SCC_prob': 'atac'},
        inplace=True,
        axis='columns'
    )
    assert np.all(rna_atac_probs.index == atac_shap_vals.index)
    
    signs = {
        (1, 1): 'active_SCC',
        (1, -1): 'inactive_ADC',
        (-1, 1): 'inactive_SCC',
        (-1, -1): 'active_ADC',
    }

    def get_sample_instances(shap_vals, sample):
        sample_labels = shap_vals.index.str.split('---').str[0]
        return shap_vals.loc[sample_labels == sample]

    across_samples_contribs_dict = {
        sign: {'rna': [], 'atac': []} for sign in signs.keys()
    }
    samples = rna_shap_vals.index.str.split('---').str[0].unique()
    for sample in tqdm(samples, total=len(samples)):
        sample_dir = f'{args.output_dir}/{sample}'
        os.makedirs(sample_dir, exist_ok=True)
        sample_rna_shap_vals = get_sample_instances(rna_shap_vals, sample)
        sample_atac_shap_vals = get_sample_instances(atac_shap_vals, sample)
        shap_vals_dict = {'rna': sample_rna_shap_vals, 'atac': sample_atac_shap_vals}
        coefs_dict = {'rna': rna_coefs, 'atac': atac_coefs}
        for sign, contrib_name in signs.items():
            for mod, shap_vals in shap_vals_dict.items():
                feat_contribs = get_feature_contributions_of_type(
                    rna_atac_probs[mod],
                    shap_vals,
                    coefs_dict[mod],
                    sign[0],
                    sign[1]
                )
                top_feat_contribs = get_top_feature_contributions(
                    feat_contribs,
                    S=args.S
                )
                top_feat_contribs.to_csv(f'{sample_dir}/{mod}_{contrib_name}_features.csv')
                across_samples_contribs_dict[sign][mod].append(feat_contribs)

    print("Saving top average contributions across samples")
    for sign, contrib_name in signs.items():
        for mod, contribs_across_samples in across_samples_contribs_dict[sign].items():
            contribs_across_samples = pd.concat(contribs_across_samples, axis=1)
            mean_contribs = contribs_across_samples.mean(axis=1)
            top_contribs = get_top_feature_contributions(mean_contribs, S=args.S)
            top_contribs.to_csv(f'{args.output_dir}/{mod}_{contrib_name}_features.csv')

    print("Saving recurrent contributions across samples")
    for sign, contrib_name in signs.items():
        for mod in ['rna', 'atac']:
            top_contribs_across_samples = [
                pd.read_csv(
                    f"{args.output_dir}/{sample}/{mod}_{contrib_name}_features.csv",
                    index_col=0
                ).iloc[:, 0] for sample in samples
            ]
            from collections import defaultdict
            feat_counts = defaultdict(int)
            for top_contribs in top_contribs_across_samples:
                for feat in top_contribs.index:
                    feat_counts[feat] += 1
            feat_counts = pd.Series(feat_counts).sort_values(ascending=False)
            feat_counts.name = "num_samples"
            feat_counts.to_csv(f"{args.output_dir}/recurrent_{mod}_{contrib_name}_features.csv")
