#!/usr/bin/bash

out="clusters_samplesheet.csv"
echo "name,rna_h5ad_path,assignments_path" > "$out"

for f in /data1/chanj3/LUAS.multiome.results/data/rna_atac_v5/*_cancer.h5mu; do
	name=$(basename "$f" _cancer.h5mu)
	rna_h5ad="/data1/chanj3/LUAS.multiome.results/data/rna_clean_v5/${name}_cancer.h5ad"
	assignments="/data1/chanj3/LUAS.multiome.results/annotations/v5/rna_atac_clusters/${name}_clustering.csv"
	echo "${name},${rna_h5ad},${assignments}" >> "$out"
done
