#!/bin/bash

#SBATCH --job-name=merge_embeddings
#SBATCH --output=logs/merge_embeddings_%j.log
#SBATCH --error=logs/merge_embeddings_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# ============================================================================
# SLURM Submission Script for Merging ESM-2 Embeddings
# ============================================================================
#
# This script submits the embedding merge job to a SLURM cluster.
#
# Usage:
#     sbatch submit_merge_embeddings.slurm
#
# To customize parameters:
#     sbatch --time=02:00:00 --mem=32G submit_merge_embeddings.slurm
#
# ============================================================================

set -e  # Exit on any error

# Print job info
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Time Limit: $SLURM_TIME_LIMIT"
echo "=========================================="
echo ""

# ============================================================================
# Configuration - MODIFY THESE PATHS
# ============================================================================

# Base directories
WORK_DIR="/Users/rizioti/Library/CloudStorage/OneDrive-GSK/research/prosperity/ml_detectability/4-detectability_ml"
SCRIPT_DIR="${WORK_DIR}/esm_embedder/src"
DATA_DIR="${WORK_DIR}/data/features_v13"
OUTPUT_DIR="${WORK_DIR}/results/merged_embeddings"

# Input embeddings and mapping
CYSTEINE_EMBEDDINGS="${DATA_DIR}/cysteine_embeddings.npy"
PEPTIDE_EMBEDDINGS="${DATA_DIR}/peptide_embeddings.npy"
PROTEIN_EMBEDDINGS="${DATA_DIR}/protein_embeddings.npy"
MAPPING_FILE="${DATA_DIR}/mapping.csv"

# Script parameters
AGGREGATION_METHOD="shortest"  # 'shortest' or 'average'
MERGE_STRATEGY="concat"        # 'concat' or 'average'
OUTPUT_FORMAT="parquet"         # 'parquet', 'csv', or 'both'
TABLES="all"                    # e.g. 'cys_only cys_pep cys_prot' or 'all'
LOG_LEVEL="INFO"                # 'DEBUG', 'INFO', 'WARNING', 'ERROR'

# ============================================================================
# Environment Setup
# ============================================================================

echo "Setting up environment..."

# Load modules if available (uncomment if needed)
# module load cuda/11.8
# module load python/3.11

# Create logs directory
mkdir -p logs

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Print configuration
echo "Configuration:"
echo "  WORK_DIR: $WORK_DIR"
echo "  SCRIPT_DIR: $SCRIPT_DIR"
echo "  DATA_DIR: $DATA_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  Aggregation Method: $AGGREGATION_METHOD"
echo "  Merge Strategy: $MERGE_STRATEGY"
echo "  Output Format: $OUTPUT_FORMAT"
echo "  Tables: $TABLES"
echo ""

# Verify files exist
echo "Verifying input files..."
for file in "$SCRIPT_DIR/merge_embeddings.py" "$CYSTEINE_EMBEDDINGS" "$PEPTIDE_EMBEDDINGS" "$PROTEIN_EMBEDDINGS" "$MAPPING_FILE"; do
    if [ ! -f "$file" ]; then
        echo "✗ ERROR: File not found: $file"
        exit 1
    fi
done
echo "✓ All input files found"
echo ""

# ============================================================================
# Run the Script
# ============================================================================

echo "Starting embedding merge..."
echo "Start time: $(date)"
echo ""

python3 "${SCRIPT_DIR}/merge_embeddings.py" \
    --cysteine-embeddings "$CYSTEINE_EMBEDDINGS" \
    --peptide-embeddings "$PEPTIDE_EMBEDDINGS" \
    --protein-embeddings "$PROTEIN_EMBEDDINGS" \
    --mapping "$MAPPING_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --aggregation-method "$AGGREGATION_METHOD" \
    --merge-strategy "$MERGE_STRATEGY" \
    --output-format "$OUTPUT_FORMAT" \
    --tables $TABLES \
    --log-level "$LOG_LEVEL"

exit_code=$?

echo ""
echo "End time: $(date)"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "✓ Job completed successfully!"
    echo ""
    echo "Output files:"
    ls -lh "${OUTPUT_DIR}"
else
    echo "✗ Job failed with exit code: $exit_code"
    exit $exit_code
fi

# ============================================================================
# Post-Processing (Optional)
# ============================================================================

# Uncomment to add summary statistics
# echo ""
# echo "Summary statistics:"
# for file in "${OUTPUT_DIR}"/*.parquet; do
#     if [ -f "$file" ]; then
#         echo "File: $(basename $file)"
#         python3 -c "import pandas as pd; df = pd.read_parquet('$file'); print(f'  Rows: {len(df)}, Columns: {df.shape[1]}')"
#     fi
# done
