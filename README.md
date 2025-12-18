Step 1: train ATAC models (Snakemake pipeline in runs/v5/atac)

Step 2: train RNA models (Snakemake pipeline in runs/v5/rna)

Step 3: apply RNA and ATAC models to query data (runs/v5/apply)

One script submitter: runs/v5/run_all.sh
