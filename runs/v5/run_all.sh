#!/bin/bash
# Master submitter script for train_atac, train_rna, and apply_models pipelines
#
# Usage: ./run_all.sh [--clean] [pipeline_names...]
#
# Options:
#   --clean    Delete output directories before running (fresh rerun)
#
# Examples:
#   ./run_all.sh                        # Run all pipelines (incremental)
#   ./run_all.sh --clean                # Clean and rerun all pipelines
#   ./run_all.sh metacells              # Run only metacells pipeline
#   ./run_all.sh --clean clusters metacells  # Clean and rerun clusters + metacells

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="/home/yangf4/envs/snakemake"

# ============================================
# PIPELINE CONFIGURATION MAPPING
# ============================================
# Define which config file to use for each pipeline step
# Format: PIPELINE_NAME -> (atac_config, rna_config, apply_config)
#
# To add a new pipeline or modify mappings, edit the get_config function below

get_config() {
    local pipeline_name="$1"
    local step="$2"  # atac, rna, or apply

    case "$pipeline_name" in
        clusters)
            case "$step" in
                atac)  echo "clusters" ;;
                rna)   echo "clusters" ;;
                apply) echo "clusters" ;;
            esac
            ;;
        clusters_lasso)
            case "$step" in
                atac)  echo "clusters_lasso" ;;
                rna)   echo "clusters_lasso" ;;
                apply) echo "clusters_lasso" ;;
            esac
            ;;
        metacells)
            case "$step" in
                atac)  echo "metacells" ;;
                rna)   echo "metacells" ;;
                apply) echo "metacells" ;;
            esac
            ;;
        metacells_scran)
            case "$step" in
                atac)  echo "metacells_scran" ;;
                rna)   echo "metacells" ;;
                apply) echo "metacells_scran" ;;
            esac
            ;;
        *)
            echo ""
            ;;
    esac
}

# Get output directory from a config file by parsing out_dir field
get_out_dir_from_config() {
    local step="$1"       # atac, rna, or apply
    local config_name="$2"
    local config_file="${SCRIPT_DIR}/${step}/${config_name}_config.yaml"

    if [ -f "$config_file" ]; then
        # Parse out_dir from YAML (handles both quoted and unquoted values)
        local out_dir=$(grep -E "^out_dir:" "$config_file" | sed 's/^out_dir:[[:space:]]*//' | tr -d '"' | tr -d "'")
        # Resolve relative path from the step directory
        if [ -n "$out_dir" ]; then
            echo "$(cd "${SCRIPT_DIR}/${step}" && realpath -m "$out_dir")"
        fi
    fi
}

# All available pipeline names
ALL_PIPELINES=("clusters" "clusters_lasso" "metacells" "metacells_scran")

# ============================================
# PARSE ARGUMENTS
# ============================================
CLEAN_MODE=false
SELECTED_PIPELINES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN_MODE=true
            shift
            ;;
        *)
            SELECTED_PIPELINES+=("$1")
            shift
            ;;
    esac
done

# Default to all pipelines if none specified
if [ ${#SELECTED_PIPELINES[@]} -eq 0 ]; then
    SELECTED_PIPELINES=("${ALL_PIPELINES[@]}")
fi

echo "=============================================="
echo "Master Pipeline Submitter"
echo "=============================================="
echo "Selected pipelines: ${SELECTED_PIPELINES[*]}"
echo "Clean mode: $CLEAN_MODE"
echo "Start time: $(date)"
echo "=============================================="

# ============================================
# CLEAN OUTPUT DIRECTORIES (if --clean)
# ============================================
if [ "$CLEAN_MODE" = true ]; then
    echo ""
    echo "=============================================="
    echo "Cleaning output directories..."
    echo "=============================================="

    # Track unique directories to avoid deleting same dir twice
    declare -A CLEANED_DIRS

    for pipeline in "${SELECTED_PIPELINES[@]}"; do
        for step in atac rna apply; do
            config=$(get_config "$pipeline" "$step")
            if [ -n "$config" ]; then
                out_dir=$(get_out_dir_from_config "$step" "$config")
                if [ -n "$out_dir" ] && [ -z "${CLEANED_DIRS[$out_dir]}" ]; then
                    if [ -d "$out_dir" ]; then
                        echo "Removing: $out_dir"
                        rm -rf "$out_dir"
                    else
                        echo "Skipping (not found): $out_dir"
                    fi
                    CLEANED_DIRS[$out_dir]=1
                fi
            fi
        done
    done

    echo "Clean complete."
fi

# ============================================
# PIPELINE EXECUTION FUNCTIONS
# ============================================

# Function to run snakemake for a given pipeline step
run_pipeline_step() {
    local step_dir="$1"      # atac, rna, or apply
    local config_name="$2"   # e.g., metacells, clusters_lasso
    local pipeline_name="$3" # Master pipeline name for logging
    local config_file="${config_name}_config.yaml"
    local log_file="snakemake_${pipeline_name}.log"

    cd "${SCRIPT_DIR}/${step_dir}"

    if [ -f "$config_file" ]; then
        echo "[${step_dir}/${pipeline_name}] Running with ${config_file}..."
        snakemake --profile profile --configfile "$config_file" --use-conda 2>&1 | tee "$log_file"
        echo "[${step_dir}/${pipeline_name}] Completed ${config_file}"
    else
        echo "[${step_dir}/${pipeline_name}] ERROR: ${config_file} not found!"
        return 1
    fi
}

# ============================================
# ACTIVATE CONDA
# ============================================
echo ""
echo "Activating conda environment: ${CONDA_ENV}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ============================================
# PHASE 1: ATAC and RNA in parallel (all pipelines concurrent)
# ============================================
echo ""
echo "=============================================="
echo "Phase 1: Running ATAC and RNA pipelines in parallel"
echo "=============================================="

declare -A PHASE1_PIDS
declare -A PHASE1_EXITS

# Launch all ATAC and RNA pipelines concurrently
for pipeline in "${SELECTED_PIPELINES[@]}"; do
    # ATAC
    atac_config=$(get_config "$pipeline" "atac")
    if [ -n "$atac_config" ]; then
        (
            run_pipeline_step "atac" "$atac_config" "$pipeline"
        ) 2>&1 | tee "${SCRIPT_DIR}/atac/snakemake_${pipeline}.log" &
        PHASE1_PIDS["atac_${pipeline}"]=$!
        echo "Launched: atac/${pipeline} (PID: $!)"
    fi

    # RNA
    rna_config=$(get_config "$pipeline" "rna")
    if [ -n "$rna_config" ]; then
        (
            run_pipeline_step "rna" "$rna_config" "$pipeline"
        ) 2>&1 | tee "${SCRIPT_DIR}/rna/snakemake_${pipeline}.log" &
        PHASE1_PIDS["rna_${pipeline}"]=$!
        echo "Launched: rna/${pipeline} (PID: $!)"
    fi
done

# Wait for all Phase 1 jobs
echo ""
echo "Waiting for all Phase 1 jobs to complete..."
PHASE1_FAILED=false

for key in "${!PHASE1_PIDS[@]}"; do
    pid=${PHASE1_PIDS[$key]}
    wait $pid
    exit_code=$?
    PHASE1_EXITS[$key]=$exit_code
    if [ $exit_code -ne 0 ]; then
        echo "FAILED: $key (exit code: $exit_code)"
        PHASE1_FAILED=true
    else
        echo "Completed: $key"
    fi
done

echo ""
echo "=============================================="
echo "Phase 1 Complete"
echo "=============================================="

# Check if any Phase 1 pipeline failed
if [ "$PHASE1_FAILED" = true ]; then
    echo "ERROR: One or more training pipelines failed. Aborting apply step."
    echo "Failed jobs:"
    for key in "${!PHASE1_EXITS[@]}"; do
        if [ ${PHASE1_EXITS[$key]} -ne 0 ]; then
            echo "  - $key (exit code: ${PHASE1_EXITS[$key]})"
        fi
    done
    exit 1
fi

# ============================================
# PHASE 2: Apply (all pipelines concurrent)
# ============================================
echo ""
echo "=============================================="
echo "Phase 2: Running Apply pipelines in parallel"
echo "=============================================="

declare -A PHASE2_PIDS
declare -A PHASE2_EXITS

# Launch all Apply pipelines concurrently
for pipeline in "${SELECTED_PIPELINES[@]}"; do
    apply_config=$(get_config "$pipeline" "apply")
    if [ -n "$apply_config" ]; then
        (
            run_pipeline_step "apply" "$apply_config" "$pipeline"
        ) 2>&1 | tee "${SCRIPT_DIR}/apply/snakemake_${pipeline}.log" &
        PHASE2_PIDS["apply_${pipeline}"]=$!
        echo "Launched: apply/${pipeline} (PID: $!)"
    fi
done

# Wait for all Phase 2 jobs
echo ""
echo "Waiting for all Phase 2 jobs to complete..."
PHASE2_FAILED=false

for key in "${!PHASE2_PIDS[@]}"; do
    pid=${PHASE2_PIDS[$key]}
    wait $pid
    exit_code=$?
    PHASE2_EXITS[$key]=$exit_code
    if [ $exit_code -ne 0 ]; then
        echo "FAILED: $key (exit code: $exit_code)"
        PHASE2_FAILED=true
    else
        echo "Completed: $key"
    fi
done

echo ""
echo "=============================================="
if [ "$PHASE2_FAILED" = true ]; then
    echo "Some apply pipelines failed:"
    for key in "${!PHASE2_EXITS[@]}"; do
        if [ ${PHASE2_EXITS[$key]} -ne 0 ]; then
            echo "  - $key (exit code: ${PHASE2_EXITS[$key]})"
        fi
    done
    exit 1
else
    echo "All pipelines completed successfully!"
fi
echo "End time: $(date)"
echo "=============================================="
