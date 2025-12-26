#!/usr/bin/bash

out="clusters_samplesheet.csv"
echo "name,rna_atac_h5mu_path,assignments_path" > "$out"

for f in /data1/chanj3/LUAS.multiome.results/data/rna_atac_v5/*_cancer.h5mu; do
	name=$(basename "$f" _cancer.h5mu)
	assignments="/data1/chanj3/LUAS.multiome.results/annotations/v5/rna_atac_clusters/${name}_clustering.csv"
	echo "${name},${f},${assignments}" >> "$out"
done
