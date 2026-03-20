# ESM-2 Embeddings & Merge Toolkit

A comprehensive solution for generating protein sequence embeddings using Meta's ESM-2 (Evolutionary Scale Modeling) language model and efficiently merging embeddings from cysteines, peptides, and proteins into consolidated feature tables for machine learning analysis.

## Overview

This package provides a unified, well-documented implementation for:
- Loading protein sequences from FASTA or CSV formats
- Processing sequences through ESM-2 neural language models
- Aggregating embeddings at multiple levels (whole-sequence, specific residues)
- Scaling to large datasets with distributed computing support
- Tracking processing progress and resuming interrupted jobs

## Key Features

✅ **Multiple Input Formats**
- FASTA files with standard UniProt headers
- CSV files with customizable columns

✅ **Flexible Embedding Options**
- `sequence`: Whole-sequence mean embedding
- `residue`: Specific residue position embedding
- `both`: Concatenated sequence + residue embeddings
- `peptide`: Whole-sequence embedding with peptide-specific label (uniprot_id_sequence)
- **Default behavior**: Processes all residue instances automatically (ideal for peptides and FASTA files)

✅ **Scalability**
- Single GPU support for standard use cases
- FSDP (Fully Sharded Data Parallel) for very large models
- CPU-only mode for testing/development
- Automatic checkpointing and resume capability

✅ **Model Variety**
- 7 different ESM-2 models (8M to 15B parameters)
- Trade-offs between accuracy and computational requirements

✅ **Production-Ready**
- SLURM job submission scripts for HPC clusters
- Comprehensive error handling and validation
- Detailed progress logging
- Output includes UniProt IDs for downstream analysis

## Installation

### Option 1: Conda (Recommended)

```bash
# Create environment from requirements
conda create -n esm2-embeddings python=3.10
conda activate esm2-embeddings
pip install -r requirements.txt
```

### Option 2: Manual Installation

```bash
# Create virtual environment
python -m venv esm2-env
source esm2-env/bin/activate

# Install dependencies
pip install torch numpy pandas
pip install fair-esm fairscale biopython
```

### GPU Support

For GPU acceleration, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Testing with Sample Data

Sample test datasets are provided in the `data/` directory for quick verification and testing:

### Available Test Files

#### 1. `data/test_peptides.csv`
Short peptide sequences with multiple cysteines
- **Format**: CSV with `uniprot_id` and `pep_seq` columns
- **Sequences**: 10 short peptides (10-15 amino acids)
- **Cysteines**: 1-2 per peptide (~19 total)
- **Use**: Test default all_residues behavior

```bash
python src/generate_esm2_embeddings.py \
    --seq_file data/test_peptides.csv \
    --model_name esm2_t6_8M_UR50D \
    --target residue \
    --residue_type C \
    --max_seq_length 100 \
    --output_file results/test_peptides_output.csv
```

#### 2. `data/test_proteins.fasta`
Protein sequences in FASTA format
- **Format**: FASTA with standard UniProt headers (`>sp|UNIPROT_ID|PROTEIN_NAME`)
- **Sequences**: 5 proteins (60-180 amino acids)
- **Cysteines**: Multiple per protein (~60+ total)
- **Use**: Test FASTA processing with all_residues mode

```bash
python src/generate_esm2_embeddings.py \
    --seq_file data/test_proteins.fasta \
    --model_name esm2_t6_8M_UR50D \
    --target residue \
    --residue_type C \
    --max_seq_length 200 \
    --output_file results/test_proteins_output.csv
```

#### 3. `data/test_sequences_with_residues.csv`
Sequences with pre-defined cysteine positions
- **Format**: CSV with `uniprot_id`, `pep_seq`, and `residue` columns
- **Entries**: 9 sequences with specific residue positions
- **Use**: Test `--no-all_residues` mode (process only specified positions)

```bash
python src/generate_esm2_embeddings.py \
    --seq_file data/test_sequences_with_residues.csv \
    --model_name esm2_t6_8M_UR50D \
    --target residue \
    --no-all_residues \
    --max_seq_length 100 \
    --output_file results/test_specific_output.csv
```

### Quick Verification Test

Run this command to quickly verify everything is working:

```bash
python src/generate_esm2_embeddings.py \
    --seq_file data/test_peptides.csv \
    --model_name esm2_t6_8M_UR50D \
    --target sequence \
    --max_seq_length 100 \
    --output_file quick_test.csv
```

Should complete in ~1-2 minutes and generate `quick_test.csv` with whole-sequence embeddings.

## Usage

### Basic Usage

```bash
python src/generate_esm2_embeddings.py \
    --seq_file data.csv \
    --model_name esm2_t6_8M_UR50D \
    --target sequence \
    --max_seq_length 1024 \
    --output_file results/full_seq_embeddings.csv
```

### Generate Cysteine-Specific Embeddings (CSV)

```bash
python src/generate_esm2_embeddings.py \
    --seq_file cysteinome.csv \
    --model_name esm2_t33_650M_UR50D \
    --target residue \
    --residue_type C \
    --max_seq_length 1024 \
    --output_file cysteine_embeddings.csv
```

### Process FASTA File with Large Model

```bash
python src/generate_esm2_embeddings.py \
    --seq_file proteins.fasta \
    --model_name esm2_t36_3B_UR50D \
    --target sequence \
    --max_seq_length 2048 \
    --output_file embeddings.csv \
    --cache_dir ./model_cache
```

### Generate Concatenated Sequence + Residue Embeddings

```bash
python src/generate_esm2_embeddings.py \
    --seq_file sequences.csv \
    --model_name esm2_t30_150M_UR50D \
    --target both \
    --residue_type C \
    --max_seq_length 1024 \
    --output_file combined_embeddings.csv
```

### Generate Whole-Sequence Embeddings for Peptide Library (with Peptide Tracking)

```bash
# Peptide target generates one embedding per peptide with label format: uniprot_id_sequence
python src/generate_esm2_embeddings.py \
    --seq_file peptides.csv \
    --model_name esm2_t12_35M_UR50D \
    --target peptide \
    --max_seq_length 100 \
    --output_file peptide_embeddings.csv
```

### Generate Embeddings for All Residue Instances (Default Behavior)

```bash
# For peptide libraries - generates one embedding per cysteine in each peptide
# all_residues is ON by default, no flag needed
python src/generate_esm2_embeddings.py \
    --seq_file peptides.csv \
    --model_name esm2_t12_35M_UR50D \
    --target residue \
    --residue_type C \
    --max_seq_length 100 \
    --output_file peptide_cys_embeddings.csv
```

### Generate Embeddings for Specific Residue Positions Only

```bash
# Use --no-all_residues when you have specific residue positions in CSV
python src/generate_esm2_embeddings.py \
    --seq_file sequences_with_residues.csv \
    --model_name esm2_t33_650M_UR50D \
    --target residue \
    --no-all_residues \
    --max_seq_length 1024 \
    --output_file specific_residue_embeddings.csv
```

### Submit SLURM Job (HPC Clusters)

```bash
# Edit job parameters and file paths in generate_embeddings.slurm
vim generate_embeddings.slurm

# Submit job
sbatch generate_embeddings.slurm

# Monitor job
squeue -j <job_id>
tail -f logs/esm2_embeddings_<job_id>.log
```

## Input Formats

### CSV Format

**Default Mode (all residue instances - recommended)**:
Required columns:
- `uniprot_id`: UniProt protein identifier
- `pep_seq`: Protein or peptide sequence

The script will automatically find all instances of the specified `--residue_type` in each sequence.
The `residue` column is not required and will be ignored if present.

Example for peptide library:
```csv
uniprot_id,pep_seq
P12345,CAKSCPEPSVSVSVGSRQC
P67890,MKTCAYIAKCQRQISFVKSH
```

This generates separate embeddings for each C: P12345_C1, P12345_C5, P12345_C20, P67890_C4, P67890_C10, P67890_C12.

**Specific Residue Mode (with `--no-all_residues` flag)**:
Required columns:
- `uniprot_id`: UniProt protein identifier (e.g., "P12345")
- `pep_seq`: Protein or peptide sequence
- `residue`: Residue identifier (e.g., "C45" for cysteine at position 45)

Example:
```csv
uniprot_id,pep_seq,residue
P12345,MKVLWAALLVTFAGCAKSC,C45
P12345,CAKSCPEPSVSVSVGSRQ,C46
P67890,MKTAYIAKQRQISFVKSH,C78
```

This processes only the specified residue positions.

### FASTA Format

Expected header format (standard UniProt):
```
>sp|UNIPROT_ID|PROTEIN_NAME
```

Example:
```fasta
>sp|P12345|PROTEIN_HUMAN
MKVLWAALLVTFAGCAKSACAKSCPEPSVSVSVGSRQ...
>sp|P67890|OTHER_PROTEIN_HUMAN
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPIL...
```

The script will automatically extract all specified residue types (default: cysteines) and create individual entries for each.

## Output Format

CSV file with columns:
- `label`: Unique identifier combining UniProt ID and residue (e.g., "P12345_C45")
- `uniprot_id`: UniProt protein identifier (empty if not provided)
- `embedding`: Numerical representation as a comma-separated list

Example:
```csv
label,uniprot_id,embedding
P12345_C45,P12345,"[-0.123, 0.456, -0.789, ...]"
P12345_C46,P12345,"[-0.234, 0.567, -0.890, ...]"
P67890_C78,P67890,"[-0.345, 0.678, -0.901, ...]"
```

### Output Format by Target

**For `sequence` target** (labeled by uniprot_id):
```csv
label,uniprot_id,embedding
P12345,P12345,"[-0.123, 0.456, ...]"
P67890,P67890,"[-0.234, 0.567, ...]"
```

**For `peptide` target** (labeled by uniprot_id_sequence):
```csv
label,uniprot_id,embedding
P12345_MVLSGDAKSCPEP,P12345,"[-0.123, 0.456, ...]"
P67890_MCTAYIAKQRQIS,P67890,"[-0.234, 0.567, ...]"
```

**For `residue` target** (labeled by uniprot_id_residue_position):
```csv
label,uniprot_id,embedding
P12345_C7,P12345,"[-0.123, 0.456, ...]"
P12345_C12,P12345,"[-0.234, 0.567, ...]"
P67890_C3,P67890,"[-0.345, 0.678, ...]"
```

## When to Use Each Mode

### Target Options

| Target | Use Case | Output |
|--------|----------|--------|
| `sequence` | Whole protein/peptide representation | Single averaged embedding per input sequence, labeled by uniprot_id |
| `residue` | Site-specific analysis | Embedding for each residue instance (default) or specific position |
| `both` | Combined context + site info | Concatenated [sequence + residue] embedding |
| `peptide` | Peptide-specific embeddings with sequence tracking | Single averaged embedding per peptide, labeled by uniprot_id_sequence |

### All Residues Mode (Default)

| Behavior | Description |
|----------|-------------|
| **Default** | Processes ALL instances of `--residue_type` in each sequence |
| `--no-all_residues` | Processes only specific residue positions from CSV `residue` column |

**Default behavior (all_residues) is ideal for:**
- ✅ Peptide libraries with multiple reactive sites
- ✅ FASTA files (no residue position info available)
- ✅ CSV files with peptide sequences (no cleavage site info)
- ✅ Comprehensive screening of all cysteines/serines/lysines
- ✅ When you don't know which specific residue to target

**Use `--no-all_residues` only when:**
- ✅ You have specific residue positions in CSV with `residue` column
- ✅ Processing full-length proteins with known sites of interest
- ✅ You want to limit output to pre-defined positions

## Command-Line Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--seq_file` | path | Input FASTA or CSV file |
| `--model_name` | str | ESM-2 model ID |
| `--target` | str | 'sequence', 'residue', 'both', or 'peptide' |
| `--max_seq_length` | int | Maximum sequence length to process |
| `--output_file` | path | Output CSV file path |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--min_seq_length` | int | 5 | Minimum sequence length |
| `--residue_type` | str | C | Type of residue to extract |
| `--all_residues` / `--no-all_residues` | flag | True | Process ALL residue instances (default) or only specific positions |
| `--cache_dir` | path | None | Directory for cached model weights |
| `--use_fsdp` | flag | False | Use FSDP for large models |
| `--port` | int | random | Port for distributed training |

### Available ESM-2 Models

| Model | Size | Parameters | Recommended Use |
|-------|------|-----------|-----------------|
| esm2_t6_8M_UR50D | 8M | 8M | Testing, quick runs |
| esm2_t12_35M_UR50D | 35M | 35M | Development |
| esm2_t30_150M_UR50D | 150M | 150M | Standard use |
| esm2_t33_650M_UR50D | 650M | 650M | Better accuracy |
| esm2_t36_3B_UR50D | 3B | 3B | High accuracy |
| esm2_t48_15B_UR50D | 15B | 15B | Best accuracy (requires FSDP) |
| esm_msa1b_t12_100M_UR50S | 100M | 100M | MSA-based |

## Model Selection Guide

Choose based on your needs:

**Fast prototyping** → `esm2_t6_8M_UR50D`
- Fastest inference
- Lowest memory usage
- Reasonable quality

**Balanced** → `esm2_t33_650M_UR50D` (recommended)
- Good accuracy/speed trade-off
- Fits on most GPUs
- Standard choice for research

**High accuracy** → `esm2_t36_3B_UR50D`
- State-of-the-art representations
- Requires 16+ GB VRAM
- Use with single GPU

**Maximum accuracy** → `esm2_t48_15B_UR50D`
- Requires FSDP for GPU memory optimization
- CPU offloading enabled
- Slowest but most accurate

## SLURM Job Configuration

Edit `generate_embeddings.slurm` to customize:

```bash
# Partition selection
#SBATCH --partition=gpu  # or 'ncpu'

# GPU allocation
#SBATCH --gpus-per-node=1

# Memory allocation (adjust based on model)
#SBATCH --mem=40G

# Time limit
#SBATCH --time=24:00:00
```

The script automatically adjusts settings based on partition selection.

## Performance Tips

### Memory Management
- Use smaller models for limited GPU memory
- Enable `--use_fsdp` for models >3B parameters
- Set `--cache_dir` to avoid re-downloading models

### Speed Optimization
- Batch multiple jobs on single dataset
- Use smaller models for testing (t6 or t12)
- Disable checkpointing if dataset is small
- Increase `--max_seq_length` to process more diverse sequences

### Resumability
- Job automatically resumes from last checkpoint
- Already processed sequences are skipped
- Partial outputs are preserved

## Examples

### Example 1: Quick Test with Small Model

```bash
python src/generate_esm2_embeddings.py \
    --seq_file test_sequences.fasta \
    --model_name esm2_t6_8M_UR50D \
    --target sequence \
    --max_seq_length 1024 \
    --output_file test_embeddings.csv
```

### Example 2: Production Run with Medium Model

```bash
python src/generate_esm2_embeddings.py \
    --seq_file all_proteins.csv \
    --model_name esm2_t33_650M_UR50D \
    --target residue \
    --residue_type C \
    --max_seq_length 2048 \
    --min_seq_length 10 \
    --output_file protein_embeddings.csv \
    --cache_dir /scratch/model_cache
```

### Example 3: Combined Sequence + Residue Embeddings

```bash
python src/generate_esm2_embeddings.py \
    --seq_file protein_library.csv \
    --model_name esm2_t30_150M_UR50D \
    --target both \
    --residue_type C \
    --max_seq_length 1024 \
    --output_file combined_embeddings.csv
```

### Example 4: All Residue Instances in Peptides (Default Behavior)

```bash
# Generate embeddings for every cysteine in peptide sequences
# Note: --all_residues is ON by default, no flag needed
python src/generate_esm2_embeddings.py \
    --seq_file peptide_pool.csv \
    --model_name esm2_t12_35M_UR50D \
    --target residue \
    --residue_type C \
    --max_seq_length 50 \
    --output_file all_cys_embeddings.csv
```

### Example 5: Specific Residue Positions Only

```bash
# Process only pre-defined residue positions
python src/generate_esm2_embeddings.py \
    --seq_file proteins_with_sites.csv \
    --model_name esm2_t33_650M_UR50D \
    --target residue \
    --no-all_residues \
    --max_seq_length 1024 \
    --output_file specific_sites_embeddings.csv
```

## Troubleshooting

### GPU Out of Memory

**Solution**: Use a smaller model or enable FSDP
```bash
python src/generate_esm2_embeddings.py ... --use_fsdp
```

### CUDA Not Available

**Check**: 
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Solution**: Ensure PyTorch is installed with CUDA support or use CPU mode

### Missing Sequences in Output

**Cause**: Sequences outside length limits are skipped
**Solution**: Adjust `--min_seq_length` and `--max_seq_length`

### Job Timeout on SLURM

**Solution**: Increase `--time` in SLURM script
```bash
#SBATCH --time=48:00:00
```

### Model Download Errors

**Solution**: Set cache directory with sufficient space
```bash
python src/generate_esm2_embeddings.py ... --cache_dir /path/with/500GB
```

### Invalid Residue Positions

**Cause**: When using `--target residue` or `--target both` with `--no-all_residues`, you may specify an invalid residue position.

**Behavior**: When an invalid residue specification is detected:
- A warning is printed to **stderr** with details of the error
- That line is skipped (not processed)
- The script continues with remaining lines
- Check stderr output for details on which rows were skipped

**Example stderr warning**:
```
[2/10] SKIPPING protein1_C1 - Invalid residue specification: Position 1 contains 'G', not 'C'. Sequence: GACDEFGHIKLMNC
```

**Common Issues and Solutions**:

| Issue | Example | Fix |
|-------|---------|-----|
| Position 0 (BOS token) | `C0` | Use position >= 1, e.g., `C1` |
| Position exceeds sequence length | `C15` in 14-char sequence | Use valid position, e.g., `C14` |
| Position doesn't contain the residue | `C1` when position 1 is 'G' | Use correct position, e.g., `C3` |

**Example: Fixing Mismatched Positions**

Before (has errors):
```csv
uniprot_id,pep_seq,residue
P12345,GACDEFGHIKLMNC,C1
P12346,ACDEFGHIKC,C2
```

After (corrected):
```csv
uniprot_id,pep_seq,residue
P12345,GACDEFGHIKLMNC,C3
P12346,ACDEFGHIKC,C1
```

## Output Analysis


Load and analyze embeddings:

```python
import pandas as pd
import numpy as np

# Load embeddings
df = pd.read_csv('embeddings.csv')

# Parse embedding strings to arrays
embeddings = np.array([
    eval(emb) for emb in df['embedding']
])

print(f"Shape: {embeddings.shape}")  # (N_sequences, embedding_dim)

# Group by protein
for uniprot_id in df['uniprot_id'].unique():
    protein_embs = embeddings[df['uniprot_id'] == uniprot_id]
    print(f"{uniprot_id}: {protein_embs.shape}")
```

## References

- [ESM-2 Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)
- [Fair-ESM GitHub](https://github.com/facebookresearch/fair-esm)
- [UniProt Format Guide](https://www.uniprot.org/help/fasta-headers)

---

# Embedding Merge Tool

A Python script for efficiently merging ESM-2 embeddings from cysteines, peptides, and proteins into concatenated feature tables for downstream machine learning analysis.

## Merge Tool Features

- **Multiple file format support**: Load embeddings from `.npy`, `.npz`, `.parquet`, or `.csv` files
- **Flexible aggregation strategies**: 
  - `shortest`: Use the shortest peptide containing each cysteine
  - `average`: Average embeddings across all peptides containing each cysteine
- **Four merge types**:
  1. Cysteine + Peptide embeddings
  2. Cysteine + Protein embeddings
  3. Cysteine + Peptide + Protein embeddings  
  4. Peptide + Protein embeddings
- **Memory-efficient processing**: Handles large datasets without loading entire matrices into memory
- **Parallel processing ready**: Architected for scalable processing
- **Comprehensive logging**: Detailed progress tracking and error reporting

## Merge Tool Installation

Ensure you have the required dependencies:

```bash
pip install numpy pandas scipy tqdm
```

For parquet support:

```bash
pip install pyarrow
```

## Merge Tool Input Data Format

### Embeddings
Embeddings should be numpy arrays with shape `(n_samples, embedding_dim)`. Supported formats:
- `.npy`: numpy binary format
- `.npz`: numpy compressed format (looks for 'embeddings' key or first array)
- `.parquet`: parquet with numeric columns as embeddings
- `.csv`: CSV with numeric columns as embeddings

### Mapping File
A CSV or Parquet file with the following columns:
- `uniprot_id`: Protein identifier
- `pep_seq`: Peptide sequence
- `residue`: Cysteine identifier (e.g., "C45", "C127")

Example mapping file:
```csv
uniprot_id,pep_seq,residue
P12345,ACDEFTGK,C45
P12345,ACDEFGHK,C45
P67890,ACDLMN,C127
```

## Merge Tool Usage

### Basic Usage

```bash
python src/merge_embeddings.py \
    --cysteine-embeddings path/to/cys_embeddings.npy \
    --peptide-embeddings path/to/pep_embeddings.npy \
    --protein-embeddings path/to/protein_embeddings.npy \
    --mapping path/to/mapping.csv \
    --output-dir path/to/output
```

### With Options

```bash
python src/merge_embeddings.py \
    --cysteine-embeddings data/cys_embeddings.npy \
    --peptide-embeddings data/pep_embeddings.npy \
    --protein-embeddings data/protein_embeddings.npy \
    --mapping data/mapping.csv \
    --output-dir results/merged \
    --aggregation-method average \
    --output-format both \
    --tables cys_pep cys_pep_prot \
    --log-level DEBUG
```

### Test Merge Tool

Verify the merge tool works on your system with synthetic data:

```bash
cd src

# Run test with synthetic data
python test_merge_embeddings.py --output-dir ./test_embeddings

# Expected output:
# ✓ Cysteine embeddings: (50, 1280)
# ✓ Peptide embeddings: (100, 1280)
# ✓ Protein embeddings: (20, 1280)
# ✓ Mapping file: 150 rows
# ✓ All tests passed!
```

## Merge Tool Arguments

- `--cysteine-embeddings` (required): Path to cysteine embeddings file
- `--peptide-embeddings` (required): Path to peptide embeddings file
- `--protein-embeddings` (required): Path to protein embeddings file
- `--mapping` (required): Path to mapping CSV/parquet file
- `--output-dir` (required): Output directory for merged tables

### Merge Tool Options

- `--aggregation-method` (default: `shortest`): 
  - `shortest`: Use shortest peptide for each cysteine
  - `average`: Average embeddings across all peptides for each cysteine

- `--output-format` (default: `parquet`):
  - `parquet`: Save as compressed parquet files (recommended)
  - `csv`: Save as CSV files (larger file sizes)
  - `both`: Save in both formats

- `--tables` (default: `all`):
  - `cys_pep`: Cysteine + peptide merged table
  - `cys_prot`: Cysteine + protein merged table
  - `cys_pep_prot`: Cysteine + peptide + protein merged table
  - `pep_prot`: Peptide + protein merged table
    - `pep_unique_residue`: Peptide-only table, one embedding per `uniprot_id_residue` (uses `--aggregation-method`)
    - `all`: Create all five tables

- `--log-level` (default: `INFO`):
  - `DEBUG`: Verbose debug output
  - `INFO`: Standard informational output
  - `WARNING`: Only show warnings and errors
  - `ERROR`: Only show errors

## Merge Tool Output

The script generates parquet/CSV files with concatenated embeddings:

### cys_pep_merged
Columns:
- `uniprot_id`: Protein identifier
- `residue`: Cysteine residue identifier
- `pep_seq`: Peptide sequence (shortest for aggregation_method=shortest)
- `aggregation_method`: Method used for peptide aggregation
- `embedding`: String representation of concatenated embeddings (e.g., "[1.234, 5.678, ...]")

### cys_prot_merged
Columns:
- `uniprot_id`: Protein identifier
- `residue`: Cysteine residue identifier
- `embedding`: String representation of concatenated embeddings (e.g., "[1.234, 5.678, ...]")

### cys_pep_prot_merged
Columns:
- `uniprot_id`: Protein identifier
- `residue`: Cysteine residue identifier
- `pep_seq`: Peptide sequence
- `protein_id`: Protein identifier (from mapping)
- `embedding`: String representation of concatenated embeddings (e.g., "[1.234, 5.678, ...]")

### pep_prot_merged
Columns:
- `pep_seq`: Peptide sequence
- `protein_id`: Protein identifier
- `embedding`: String representation of concatenated embeddings (e.g., "[1.234, 5.678, ...]")

### pep_unique_residue_merged
Columns:
- `uniprot_id_residue`: Unique residue identifier (stored in the `index` column in output files)
- `embedding`: Peptide embedding only (shortest peptide or average of mapped peptides, based on `--aggregation-method`)

## SLURM Job Submission for Merge

For cluster execution, use the provided SLURM script:

### Basic Submission

```bash
sbatch src/submit_merge_embeddings.slurm
```

### Custom Resource Allocation

```bash
sbatch --time=02:00:00 --mem=32G --cpus-per-task=4 src/submit_merge_embeddings.slurm
```

### Monitor Job

```bash
# Check job status
squeue -j <job_id>

# View logs
tail -f logs/merge_embeddings_<job_id>.log
```

### Configuration

Edit these variables in `src/submit_merge_embeddings.slurm`:

```bash
# Base directories
WORK_DIR="/path/to/work"
DATA_DIR="${WORK_DIR}/data"
OUTPUT_DIR="${WORK_DIR}/results"

# Input files
CYSTEINE_EMBEDDINGS="${DATA_DIR}/cysteine_embeddings.npy"
PEPTIDE_EMBEDDINGS="${DATA_DIR}/peptide_embeddings.npy"
PROTEIN_EMBEDDINGS="${DATA_DIR}/protein_embeddings.npy"
MAPPING_FILE="${DATA_DIR}/mapping.csv"

# Script parameters
AGGREGATION_METHOD="shortest"
OUTPUT_FORMAT="parquet"
TABLES="all"
```

## Merge Tool Examples

### Example 1: Basic Workflow

```bash
# Create output directory
mkdir -p results/merged_embeddings

# Run merge with shortest peptide aggregation
python src/merge_embeddings.py \
    --cysteine-embeddings data/embeddings/cysteines.npy \
    --peptide-embeddings data/embeddings/peptides.npy \
    --protein-embeddings data/embeddings/proteins.npy \
    --mapping data/metadata/mapping.csv \
    --output-dir results/merged_embeddings \
    --aggregation-method shortest

# Output files:
# - results/merged_embeddings/cys_pep_merged.parquet
# - results/merged_embeddings/cys_prot_merged.parquet
# - results/merged_embeddings/cys_pep_prot_merged.parquet
# - results/merged_embeddings/pep_prot_merged.parquet
```

### Example 2: Using Parquet Files

```bash
python src/merge_embeddings.py \
    --cysteine-embeddings data/embeddings/cysteines.parquet \
    --peptide-embeddings data/embeddings/peptides.parquet \
    --protein-embeddings data/embeddings/proteins.parquet \
    --mapping data/metadata/mapping.parquet \
    --output-dir results/merged_embeddings \
    --output-format both
```

### Example 3: Create Only Specific Tables

```bash
# Create only cysteine-based merged tables (useful for focused analysis)
python src/merge_embeddings.py \
    --cysteine-embeddings data/embeddings/cysteines.npy \
    --peptide-embeddings data/embeddings/peptides.npy \
    --protein-embeddings data/embeddings/proteins.npy \
    --mapping data/metadata/mapping.csv \
    --output-dir results/merged \
    --tables cys_pep cys_prot cys_pep_prot
```

### Example 4: Use Results in ML Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load merged embeddings
df = pd.read_parquet('results/merged_embeddings/cys_pep_merged.parquet')

# Parse embedding strings to arrays
import ast
X = np.array([ast.literal_eval(emb) for emb in df['embedding']])

# Standardize if needed
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Load your labels (indexed by uniprot_id and residue)
y = pd.read_csv('path/to/labels.csv', index_col=['uniprot_id', 'residue'])
y = y.loc[list(zip(df['uniprot_id'], df['residue']))].values.ravel()

# Train model
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X)

print(f"Accuracy: {clf.score(X, y):.4f}")
```

## Merge Tool Performance Guidelines

### Memory Usage

The script loads embeddings sequentially:
- **Cysteine embeddings**: (n_cys, 1280) ≈ n_cys * 10 KB
- **Peptide embeddings**: (n_pep, 1280) ≈ n_pep * 10 KB
- **Protein embeddings**: (n_prot, 1280) ≈ n_prot * 10 KB
- **Mapping DataFrame**: (n_mappings, 3) ≈ n_mappings * 100 bytes

Example for 100K cysteines, 1M peptides, 20K proteins:
- Total ≈ 12 GB

### Optimization Tips

1. **Use parquet format**: 40-50% smaller than CSV, faster to read
2. **Aggregate method**: `shortest` is faster than `average`
3. **Selective table creation**: Only create tables you need
4. **SLURM allocation**: 
   - For < 1M samples: 32GB memory, 4 CPUs
   - For 1M-100M samples: 64GB memory, 8 CPUs
   - For > 100M samples: 128GB+ memory, 16 CPUs

## Merge Tool Troubleshooting

### Issue: "File not found" Error

**Solution**: Verify all input paths are correct and files exist:
```bash
ls -lh /path/to/embeddings.npy
ls -lh /path/to/mapping.csv
```

### Issue: "Missing columns" Error

**Solution**: Ensure mapping file has required columns: `uniprot_id`, `pep_seq`, `residue`
```bash
# Check columns
head -3 mapping.csv
# Or for parquet:
python -c "import pandas as pd; print(pd.read_parquet('mapping.parquet').columns)"
```

### Issue: Out of Memory

**Solution**: 
- Use parquet format for faster loading and less memory overhead
- Consider processing in chunks or splitting mapping file
- Increase SLURM memory allocation: `--mem=128G`

### Issue: Slow Processing

**Solution**:
- Use SSD storage for data files
- Consider using parquet format
- Use `shortest` aggregation method
- Increase CPU cores in SLURM: `--cpus-per-task=16`

## References

- [ESM-2 Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)
- [Fair-ESM GitHub](https://github.com/facebookresearch/fair-esm)
- [UniProt Format Guide](https://www.uniprot.org/help/fasta-headers)

## Citation

If you use this tool, please cite:

```bibtex
@article{lin2022language,
  title={Language models of protein sequences at scale},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wentao and Cotterell, Allan N and Rives, Alexander},
  journal={bioRxiv},
  year={2022}
}
```

## License

This code is provided under the MIT License. The ESM-2 models are also released under the MIT License by Meta Platforms, Inc.

## Support

For issues, questions, or suggestions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review example usage in this README
3. Check command-line help: `python src/generate_esm2_embeddings.py --help`
