# ESM-2 Protein Sequence Embeddings Generator

A comprehensive tool for generating protein sequence embeddings using Meta's ESM-2 (Evolutionary Scale Modeling) language model. This tool converts protein sequences into numerical representations (embeddings) that capture biological meaning, with support for aggregating at specific residue positions.

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

## When to Use Each Mode

### Target Options

| Target | Use Case | Output |
|--------|----------|--------|
| `sequence` | Whole protein/peptide representation | Single averaged embedding per input sequence |
| `residue` | Site-specific analysis | Embedding for each residue instance (default) or specific position |
| `both` | Combined context + site info | Concatenated [sequence + residue] embedding |

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
| `--target` | str | 'sequence' (whole-seq mean), 'residue' (specific residue), or 'both' (concatenated) |
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
