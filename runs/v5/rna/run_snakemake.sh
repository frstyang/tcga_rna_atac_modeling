conda activate /home/yangf4/envs/snakemake
snakemake --profile profile --use-conda 2>&1 | tee snakemake.log
