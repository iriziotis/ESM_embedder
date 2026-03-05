#!/bin/bash
#
# SLURM Job Submission Script for ESM-2 Embeddings Generation
#
# Usage:
#   sbatch generate_embeddings.slurm
#
# Configure the parameters below before submission

# ============================================================================
# SLURM Configuration
# ============================================================================

# Job name
#SBATCH --job-name=esm2_embeddings

# Partition (gpu or ncpu)
#SBATCH --partition=gpu

# Number of nodes
#SBATCH --nodes=1

# CPUs per node
#SBATCH --cpus-per-task=8

# GPU configuration (only for gpu partition)
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest

# Memory allocation
#SBATCH --mem=40G

# Time limit (format: HH:MM:SS)
#SBATCH --time=24:00:00

# Output files
#SBATCH --output=logs/esm2_embeddings_%j.log
#SBATCH --error=logs/esm2_embeddings_%j.err

# Email notifications (optional)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@example.com

# ============================================================================
# Job Configuration
# ============================================================================

# Input and output paths
SEQ_FILE="path/to/input/sequences.csv"
OUTPUT_FILE="path/to/output/embeddings.csv"
CACHE_DIR="path/to/model/cache"

# Model configuration
MODEL="esm2_t33_650M_UR50D"  # Options: esm2_t48_15B_UR50D, esm2_t36_3B_UR50D, esm2_t33_650M_UR50D, esm2_t30_150M_UR50D, esm2_t12_35M_UR50D, esm2_t6_8M_UR50D
TARGET="sequence"            # Options: sequence (whole sequence mean), residue (specific residue), both (concatenated)
RESIDUE_TYPE="C"             # Residue type (default: C for cysteine)
ALL_RESIDUES="true"          # Default: "true" - processes all residue instances. Set to "false" for specific residue positions only
MAX_SEQ_LENGTH=1024
MIN_SEQ_LENGTH=0

# ============================================================================
# Environment Setup
# ============================================================================

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules if needed (adjust based on your HPC system)
# module load cuda/11.8
# module load cudnn/8.6

# Activate conda environment
# Change 'esm2-embeddings' to your environment name
source activate esm2-embeddings

# Print job information
echo "=========================================="
echo "Starting ESM-2 Embeddings Generation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "GPU Available: $(which nvidia-smi > /dev/null && echo 'Yes' || echo 'No')"
echo "=========================================="

# ============================================================================
# Determine Partition-Specific Settings
# ============================================================================

if [ "${SLURM_JOB_PARTITION}" == "gpu" ]; then
    echo "Running on GPU partition"
    USE_FSDP=false
    if [ "${MODEL}" == "esm2_t48_15B_UR50D" ]; then
        USE_FSDP=true
        echo "Using FSDP for large model"
    fi
else
    echo "Running on CPU partition"
    USE_FSDP=false
fi

# ============================================================================
# Execute Script
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/generate_esm2_embeddings.py"

# Build command
CMD="python ${PYTHON_SCRIPT} \
    --seq_file ${SEQ_FILE} \
    --model_name ${MODEL} \
    --target ${TARGET} \
    --residue_type ${RESIDUE_TYPE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --min_seq_length ${MIN_SEQ_LENGTH} \
    --output_file ${OUTPUT_FILE} \
    --cache_dir ${CACHE_DIR}"

# Disable all_residues if needed (it's on by default)
if [ "${ALL_RESIDUES}" == "false" ]; then
    CMD="${CMD} --no-all_residues"
fi

# Add FSDP flag if needed
if [ "${USE_FSDP}" == "true" ]; then
    CMD="${CMD} --use_fsdp"
fi

echo "Command: ${CMD}"
echo "=========================================="

# Run the script
${CMD}

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Job completed successfully"
    echo "Output: ${OUTPUT_FILE}"
    echo "=========================================="
else
    echo "=========================================="
    echo "Job failed with exit code $?"
    echo "=========================================="
    exit 1
fi
