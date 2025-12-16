#!/usr/bin/bash

out="clusters_samplesheet.csv"
echo "name,atac_h5ad_path,assignments_path" > "$out"

for f in /data1/chanj3/LUAS.multiome.results/data/rna_atac_v5/*_cancer.h5mu; do
	base=$(basename "$f" .h5mu)
	name=${base%_cancer}
	atac_h5ad="/data1/chanj3/LUAS.multiome.results/nf_outs/atac/atac_joint_embedding/${name}/${name}_atac.h5ad"
	assignments="/data1/chanj3/LUAS.multiome.results/annotations/v5/rna_atac_clusters/${name}_clustering.csv"
	echo "${name},${atac_h5ad},${assignments}" >> "$out"
done
