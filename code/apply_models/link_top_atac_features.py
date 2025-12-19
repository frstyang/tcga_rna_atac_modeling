import argparse
import numpy as np
import os
import pandas as pd
import pickle
import pyreadr
import scanpy as sc
import snapatac2 as snap
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("atac_lognorm_path")
    parser.add_argument("rna_lognorm_path")
    parser.add_argument("peak_set_path")
    parser.add_argument("top_features_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--cor_threshold", type=float, default=0.1)
    parser.add_argument("--tss_dist_threshold", type=int, default=150000)
    args = parser.parse_args()

    print("Loading data")
    atac_lognorm = pyreadr.read_r(args.atac_lognorm_path)[None].T
    peak_set = pd.read_csv(args.peak_set_path, header=None).iloc[:, 0]
    atac_lognorm = atac_lognorm.loc[:, peak_set]
    rna_lognorm = pyreadr.read_r(args.rna_lognorm_path)[None].T

    print("Construct peak_mat and gene_mat")
    peak_mat = sc.AnnData(atac_lognorm)
    gene_mat = sc.AnnData(rna_lognorm)

    print("Construct regulatory network")
    network = snap.tl.init_network_from_annotation(
        peak_set,
        snap.genome.hg38,
        upstream=args.tss_dist_threshold,
        downstream=args.tss_dist_threshold
    )
    snap.tl.add_cor_scores(
        network,
        gene_mat=gene_mat,
        peak_mat=peak_mat
    )
    def edge_filter(u, v, e):
        if e.cor_score:
            return e.cor_score > args.cor_threshold
        return False
    network = snap.tl.prune_network(
        network,
        edge_filter=edge_filter,
        remove_isolates=True
    )
    snap.tl.add_tf_binding(
        network,
        motifs=snap.datasets.Meuleman_2020(),
        genome_fasta=snap.genome.hg38,
        pvalue=1e-05
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    print("Save regulatory network")
    with open(f"{args.output_dir}/GRN.pkl", "wb") as f:
        pickle.dump(network, f)

    print("Write gene-linked atac feature lists")
    region_to_ind = {n.id: i for i, n in enumerate(network.nodes())
                     if n.type == 'region'}
    def write_gene_linked_top_atac_features(dir_path):
        for fname in os.listdir(dir_path):
            if not re.fullmatch("atac.*_features\.csv", fname):
                continue
            feats_df = pd.read_csv(f"{dir_path}/{fname}", index_col=0)
            linked_genes = []
            for region in feats_df.index:
                if region in region_to_ind:
                    targets = network.successors(region_to_ind[region])
                    linked_genes.append(";".join([x.id for x in targets]))
                else:
                    linked_genes.append("")
            feats_df["linked_genes"] = linked_genes
            new_fname = fname.replace(".csv", "_linked.csv")
            feats_df.to_csv(f"{dir_path}/{new_fname}")
    samples = np.unique(rna_lognorm.index.str.split('---').str[0])
    for sample in samples:
        sample_dir = f"{args.top_features_dir}/{sample}"
        write_gene_linked_top_atac_features(sample_dir)
    write_gene_linked_top_atac_features(args.top_features_dir)

