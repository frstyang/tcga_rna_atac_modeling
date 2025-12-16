conda activate /home/yangf4/envs/snakemake
snakemake --profile profile --configfile metacells_config.yaml --use-conda 2>&1 | tee snakemake_metacells.log
snakemake --profile profile --configfile clusters_config.yaml --use-conda 2>&1 | tee snakemake_clusters.log
