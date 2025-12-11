#!/bin/bash
#SBATCH --array=0-5
#SBATCH --job-name="tcga_luad_vs_lusc_atac_logreg"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=3:00:00
#SBATCH --output="logs/tcga_luad_vs_lusc_atac_logreg-%a.out"

# Make a sorted list of files
FILES=( $(ls -1 luad_vs_lusc_input_peaks | sort) )

# Pick the i-th file based on the task ID
INPUT_FILE="luad_vs_lusc_input_peaks/${FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Running on file: $INPUT_FILE"

python -u train_multiclass_logistic_regression_model.py "$INPUT_FILE" --luad_vs_lusc
