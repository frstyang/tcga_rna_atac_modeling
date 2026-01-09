import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from pyfaidx import Fasta
import re
import scipy.stats
import snapatac2 as snap
from tqdm import tqdm

def hypergeom_test(region_motif_mat, region_set):
    """Quantifies motif enrichment with the hypergeometric test. A hyper-
    geometric test is run for each motif, comparing its occurrence frequency
    in an input set of regions vs. in all regions, and the resulting p-values
    are FDR corrected to account for multiple testing.

    Args:
        region_motif_mat (pd.DataFrame): (n_regions, n_motifs) binary matrix
            containing whether motif j occurs in region i.
        region_set (iterable): a subset of regions to test motif enrichment.

    Return:
        pd.DataFrame: motif enrichment results (logFCs, p-values, q-values, 
            score = -sign(logFC)*log(p-value))
    """
    assert all([region in region_motif_mat.index for region in region_set])
    assert 1 <= len(region_set) < len(region_motif_mat)
    M = len(region_motif_mat)
    n = region_motif_mat.sum(axis=0).values # (n_motifs,)
    N = len(region_set)
    k = region_motif_mat.loc[region_set].sum(axis=0).values # (n_motifs,)
    fc = (k / N) / (n / M)
    fc[np.isnan(fc)] = 1
    # accept that some fc values will be 0 -> -inf logfc
    logfc = np.log2(fc)
    pval = np.exp(np.minimum(
        scipy.stats.hypergeom.logcdf(k, M, n, N),
        scipy.stats.hypergeom.logsf(k - 1, M, n, N)
    ))
    qval = scipy.stats.false_discovery_control(pval)
    score = -np.sign(logfc) * np.log2(pval)
    return pd.DataFrame(
        {'log2FC': logfc, 'p-value': pval, 'q-value': qval, 'score': score},
        index=region_motif_mat.columns
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("top_features_dir")
    parser.add_argument("peak_set_path")
    parser.add_argument("all_peaks_path")
    parser.add_argument("--reduce_motifs", action="store_true")
    parser.add_argument("--n_total_peaks", type=int, default=100000)
    args = parser.parse_args()

    def read_peaks(peak_file):
        with open(peak_file, 'r') as f:
            return [x.strip() for x in f.readlines()]
    peak_set = read_peaks(args.peak_set_path)
    all_peaks = read_peaks(args.all_peaks_path)

    rename_map = {
        'BC11A': 'BCL11A',
        'BC11B': 'BCL11B',
        'PO5F1': 'POU5F1',
        'P53': 'TP53',
        'P63': 'TP63',
        'P73': 'TP73',
        'NF2L2': 'NFE2L2',
    }
    def renamer(motif):
        if '+' in motif:
            return '+'.join([renamer(part) for part in motif.split('+')])
        motif = motif.upper()
        motif = motif.replace('.MOUSE', '')
        if re.match('ZN\d', motif):
            return motif.replace('ZN', 'ZNF')
        if motif in rename_map:
            return rename_map[motif]
        return motif

    motifs = snap.datasets.Meuleman_2020()
    np.random.seed(262)
    other_peaks = list(set(all_peaks) - set(peak_set))
    n_total_peaks = args.n_total_peaks
    background_peaks = np.random.choice(other_peaks, n_total_peaks - len(peak_set), replace=False)
    peaks = peak_set + list(background_peaks)

    pvalue = 1e-5
    genome_fasta = snap.genome.hg38
    genome = genome_fasta.fasta
    genome = Fasta(genome, one_based_attributes=False)
    print(f"Fetching {len(peaks)} regions")
    sequences = [snap._utils.fetch_seq(genome, region) for region in peaks]
    def compute_gc(seq):
        seq = seq.upper()
        gc = (seq.count('G') + seq.count('C')) / len(seq) if len(seq) > 0 else 0
        return gc
    peak_gcs = pd.Series([compute_gc(seq) for seq in sequences], index=peaks)
    peak_to_motifs = {peak: [] for peak in peaks}
    
    print("Searching for the binding sites of {} motifs ...".format(len(motifs)))
    for motif in tqdm(motifs):
        bound = motif.with_nucl_prob().exists(sequences, pvalue=pvalue)
        if any(bound):
            for peak in itertools.compress(peaks, bound):
                peak_to_motifs[peak].append(motif.id)

    motif_id_to_motif = {m.id: m for m in motifs}
    if args.reduce_motifs:
        def get_name(motif_id):
            motif = motif_id_to_motif[motif_id]
            return renamer(motif.name)
    else:
        def get_name(motif_id):
            return motif_id

    motif_names = sorted(list(set([get_name(motif.id) for motif in motifs])))
    region_motif_mat = pd.DataFrame(
        0,
        index=peaks,
        columns=motif_names
    )

    print("Filling out region x motif matrix")
    for peak, peak_motifs in tqdm(peak_to_motifs.items(), total=len(peak_to_motifs)):
        for motif_id in peak_motifs:
            motif_name = get_name(motif_id)
            region_motif_mat.loc[peak, motif_name] = 1

    def gc_match_subset(region_motif_mat, peak_gcs, subset):
        """Gets subset of region_motif_mat with matched gc content
        distribution to input subset

        Args:
            region_motif_mat (pd.DataFrame): (n_peaks, n_motifs)
            peak_gcs (pd.Series): (n_peaks,)
            subset (pd.Index): (n_subset_peaks,)

        Returns:
            pd.DataFrame (n_gc_matched_peaks, n_motifs)
        """
        assert all(np.isin(subset, region_motif_mat.index))
        non_subset_peaks = np.setdiff1d(peak_gcs.index, subset)
        non_subset_gcs = peak_gcs.loc[non_subset_peaks].sort_values()
        
        tol = 0.05
        matched_peaks = []
        counter = 0
        while True:
            failed = False
            for peak in subset:
                if len(non_subset_gcs) == 0:
                    failed = True
                    break
                gc = peak_gcs[peak]

                i = np.searchsorted(non_subset_gcs.values, gc)
                i = np.clip(i, 0, len(non_subset_gcs) - 1)
                
                # Check if neighbor is closer
                if i > 0 and np.abs(non_subset_gcs.values[i - 1] - gc) < np.abs(non_subset_gcs.values[i] - gc):
                    i = i - 1
                
                if (np.abs(non_subset_gcs.values[i] - gc) < tol) or (counter < 3):
                    matched_peak = non_subset_gcs.index[i]
                    matched_peaks.append(matched_peak)
                    non_subset_gcs = non_subset_gcs.drop(labels=[matched_peak])
                else:
                    failed = True
                    break
            if failed:
                break
            counter += 1
        matched_peaks = np.unique(matched_peaks)
        new_peak_set = np.concatenate((matched_peaks, subset))
        region_motif_mat = region_motif_mat.loc[new_peak_set]
        return region_motif_mat

    def write_motif_occ_info(region_motif_mat, dir_path, peak_gcs):
        scc_feats = pd.read_csv(f'{dir_path}/atac_active_SCC_features_linked.csv', index_col=0)
        adc_feats = pd.read_csv(f'{dir_path}/atac_active_ADC_features_linked.csv', index_col=0)
        scc_peaks = scc_feats.index.intersection(region_motif_mat.index)
        adc_peaks = adc_feats.index.intersection(region_motif_mat.index)

        # gc match background
        assert all(region_motif_mat.index == peak_gcs.index)
        region_motif_mat_adc = gc_match_subset(region_motif_mat, peak_gcs, adc_peaks)
        region_motif_mat_scc = gc_match_subset(region_motif_mat, peak_gcs, scc_peaks)

        adc_bg_peaks = np.setdiff1d(region_motif_mat_adc.index, adc_peaks)
        scc_bg_peaks = np.setdiff1d(region_motif_mat_scc.index, scc_peaks)
        plt.figure(figsize=(5, 4))
        plt.hist(peak_gcs[adc_peaks], 20, alpha=0.5, density=True, label='adc')
        plt.hist(peak_gcs[adc_bg_peaks], 20, alpha=0.5, density=True, label='adc bg')
        plt.legend()
        plt.savefig(f"{dir_path}/ADC_peak_gc_dists.png")
        plt.figure(figsize=(5, 4))
        plt.hist(peak_gcs[scc_peaks], 20, alpha=0.5, density=True, label='scc')
        plt.hist(peak_gcs[scc_bg_peaks], 20, alpha=0.5, density=True, label='scc bg')
        plt.legend()
        plt.savefig(f"{dir_path}/SCC_peak_gc_dists.png")
        
        scc_counts = region_motif_mat.loc[scc_peaks].mean(axis=0)
        adc_counts = region_motif_mat.loc[adc_peaks].mean(axis=0)        
        scc_bg_counts = region_motif_mat_scc.loc[scc_bg_peaks].mean(axis=0)
        adc_bg_counts = region_motif_mat_adc.loc[adc_bg_peaks].mean(axis=0)
        
        counts_df = pd.DataFrame(
            {
                f'scc_{len(scc_peaks)}': scc_counts,
                f'adc_{len(adc_peaks)}': adc_counts,
                f'scc_bg_{len(scc_bg_peaks)}': scc_bg_counts,
                f'adc_bg_{len(adc_bg_peaks)}': adc_bg_counts,
            }
        )
        
        scc_hypergeom_res = hypergeom_test(region_motif_mat_scc, scc_peaks)
        adc_hypergeom_res = hypergeom_test(region_motif_mat_adc, adc_peaks)
        
        counts_df['scc_'+scc_hypergeom_res.columns.astype(str)] = scc_hypergeom_res
        counts_df['adc_'+adc_hypergeom_res.columns.astype(str)] = adc_hypergeom_res
    
        counts_df.to_csv(f'{dir_path}/motif_occ_info.csv')
        counts_df.sort_values('scc_score', ascending=False).head(100).to_csv(f'{dir_path}/motif_occ_info_top_scc.csv')
        counts_df.sort_values('adc_score', ascending=False).head(100).to_csv(f'{dir_path}/motif_occ_info_top_adc.csv')

    print("Writing motif occurrence info for each sample")
    samples = [
        x for x in os.listdir(args.top_features_dir)
        if (
            x.startswith('LU') and
            os.path.isdir(f"{args.top_features_dir}/{x}")
        )
    ]
    for sample in tqdm(samples):
        sample_dir = f"{args.top_features_dir}/{sample}"
        write_motif_occ_info(region_motif_mat, sample_dir, peak_gcs)
    write_motif_occ_info(region_motif_mat, args.top_features_dir, peak_gcs)
