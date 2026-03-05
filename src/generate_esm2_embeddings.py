#!/usr/bin/env python3
"""
Generate protein sequence embeddings using ESM-2 (Evolutionary Scale Modeling).

This script converts protein sequences into numerical representations that capture
biological meaning, with support for aggregating embeddings at specific residue sites.

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the MIT license.
"""

import argparse
import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path
from Bio import SeqIO
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
import esm


# ============================================================================
# Data Loading and Formatting Functions
# ============================================================================

def load_sequence_csv(csv_path: str, residue_type: str = 'C', all_residues: bool = True) -> pd.DataFrame:
    """
    Load protein sequence data from a CSV file.
    
    Expected columns:
    - uniprot_id: UniProt identifier
    - pep_seq: Peptide sequence
    - residue: Residue position (e.g., 'C45' for cysteine at position 45) - only if all_residues=False
    
    Args:
        csv_path: Path to the CSV file
        residue_type: Type of residue to extract (default: 'C' for cysteine)
        all_residues: If True (default), extract all instances of residue_type from each sequence.
                      If False, use specific residue positions from 'residue' column.
        
    Returns:
        Formatted pandas DataFrame with columns: label, sequence, residue_of_interest_idx, uniprot_id
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns based on mode
    if all_residues:
        required_cols = {'uniprot_id', 'pep_seq'}
    else:
        required_cols = {'uniprot_id', 'pep_seq', 'residue'}
    
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {required_cols}. "
            f"Found: {set(df.columns)}"
        )
    
    # Filter out rows with missing sequences
    df = df.dropna(subset=['pep_seq'])
    
    if all_residues:
        # Expand to create one row per residue instance
        df_formatted = expand_to_all_residues(df, residue_type, seq_col='pep_seq')
    else:
        # Create label combining uniprot_id and residue
        df['label'] = df['uniprot_id'].astype(str) + '_' + df['residue'].astype(str)
        
        # Format for processing
        df_formatted = df[['label', 'pep_seq', 'residue', 'uniprot_id']].copy()
        df_formatted.columns = ['label', 'sequence', 'residue', 'uniprot_id']
    
    return df_formatted


def load_sequence_fasta(fasta_path: str, residue_type: str = 'C') -> pd.DataFrame:
    """
    Load protein sequence data from a FASTA file.
    
    Expected header format (standard UniProt):
        >sp|UNIPROT_ID|PROTEIN_NAME
    
    This function extracts all residues of type residue_type from each sequence.
    
    Args:
        fasta_path: Path to the FASTA file
        residue_type: Type of residue to extract (default: 'C' for cysteine)
        
    Returns:
        Formatted pandas DataFrame with columns: label, sequence, residue_of_interest_idx, uniprot_id
    """
    records = list(SeqIO.parse(fasta_path, 'fasta'))
    data = []
    
    for record in records:
        # Parse header in format: sp|UNIPROT_ID|PROTEIN_NAME
        header_parts = record.id.split('|')
        if len(header_parts) >= 2:
            uniprot_id = header_parts[1]
        else:
            uniprot_id = record.id
        
        sequence = str(record.seq)
        
        # Find all positions of the residue type
        for pos, char in enumerate(sequence):
            if char == residue_type:
                residue_num = pos + 1  # 1-based indexing
                label = f"{uniprot_id}_{residue_type}{residue_num}"
                data.append({
                    'label': label,
                    'sequence': sequence,
                    'residue_of_interest_idx': residue_num,
                    'uniprot_id': uniprot_id
                })
    
    if not data:
        raise ValueError(f"No {residue_type} residues found in FASTA sequences")
    
    return pd.DataFrame(data)


def expand_to_all_residues(df: pd.DataFrame, residue_type: str, seq_col: str = 'pep_seq') -> pd.DataFrame:
    """
    Expand dataframe to create one row per residue instance of the specified type.
    
    Args:
        df: Input dataframe with columns: uniprot_id, seq_col
        residue_type: Type of residue to extract (e.g., 'C' for cysteine)
        seq_col: Name of the column containing sequences
        
    Returns:
        Formatted dataframe with columns: label, sequence, residue_of_interest_idx, uniprot_id
    """
    data = []
    
    for _, row in df.iterrows():
        uniprot_id = row['uniprot_id']
        sequence = row[seq_col]
        
        # Find all positions of the residue type
        for pos, char in enumerate(sequence):
            if char == residue_type:
                residue_num = pos + 1  # 1-based indexing
                label = f"{uniprot_id}_{residue_type}{residue_num}"
                data.append({
                    'label': label,
                    'sequence': sequence,
                    'residue_of_interest_idx': residue_num,
                    'uniprot_id': uniprot_id
                })
    
    if not data:
        raise ValueError(f"No {residue_type} residues found in sequences")
    
    return pd.DataFrame(data)


def format_sequence_dataframe(df: pd.DataFrame, residue_type: str = 'C') -> pd.DataFrame:
    """
    Format sequence dataframe for embedding generation.
    
    Args:
        df: Input dataframe with columns: label, sequence, residue, uniprot_id
        residue_type: Type of residue (for reference, default: 'C')
        
    Returns:
        Formatted dataframe with columns: label, sequence, residue_of_interest_idx, residue_type, uniprot_id
    """
    # Extract the residue number from the residue string (e.g., 'C45' -> 45)
    df['residue_of_interest_idx'] = df['residue'].str.extract(r'(\d+)').astype(int)
    
    # Extract residue type from residue string (e.g., 'C45' -> 'C')
    df['residue_type_extracted'] = df['residue'].str.extract(r'([A-Z])')[0]
    
    # Select and rename columns
    result_df = df[['label', 'sequence', 'residue_of_interest_idx', 'residue_type_extracted', 'uniprot_id']].copy()
    result_df.columns = ['label', 'sequence', 'residue_of_interest_idx', 'residue_type', 'uniprot_id']
    
    return result_df


# ============================================================================
# Embedding Generation Functions
# ============================================================================

def process_batch(model, data_tuple, batch_converter, repr_layer_id: int) -> np.ndarray:
    """
    Generate embeddings for a sequence using ESM-2 model.
    
    Args:
        model: ESM-2 model
        data_tuple: Tuple of (label, sequence)
        batch_converter: Batch converter from ESM model
        repr_layer_id: Layer ID to extract representations from
        
    Returns:
        Token representations as numpy array
    """
    batch_labels, batch_strs, batch_tokens = batch_converter([data_tuple])
    batch_tokens = batch_tokens.cuda() if torch.cuda.is_available() else batch_tokens.cpu()
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[repr_layer_id], return_contacts=False)
    
    return results['representations'][repr_layer_id].squeeze().cpu().numpy()


def get_whole_sequence_embedding(token_representations: np.ndarray, method: str = 'mean') -> np.ndarray:
    """
    Generate whole sequence embedding from token representations.
    
    Args:
        token_representations: Token representations (excludes special tokens at positions 0 and -1)
        method: Aggregation method ('mean' or 'sum')
        
    Returns:
        Aggregated embedding as numpy array
    """
    if method not in ['mean', 'sum']:
        raise ValueError("Method must be 'mean' or 'sum'")
    
    # Skip special tokens: position 0 is BOS, position -1 is EOS
    tokens_to_aggregate = token_representations[1:-1]
    
    if method == 'mean':
        return np.mean(tokens_to_aggregate, axis=0)
    else:
        return np.sum(tokens_to_aggregate, axis=0)
    
def get_residue_embedding(residue_idx: int, token_representations: np.ndarray, sequence: str = None, 
                         residue_type: str = 'C') -> np.ndarray:
    """
    Extract embedding for a specific residue position.
    
    Args:
        residue_idx: Residue index (1-based, must be > 0 and <= sequence length)
        token_representations: Token representations (includes BOS at 0 and EOS at -1)
        sequence: Sequence string (for validation, optional). If provided, validates that
                  residue_idx is within sequence bounds AND contains the expected residue.
        residue_type: Expected residue type (e.g., 'C' for cysteine). Used for validation.
        
    Returns:
        Embedding for the residue
        
    Raises:
        ValueError: If residue_idx is invalid (<=0, beyond sequence length, or doesn't contain expected residue)
    """
    # Check for special tokens
    if residue_idx <= 0:
        raise ValueError(
            f"Invalid residue position {residue_idx}: position must be >= 1 "
            f"(position 0 is BOS special token)"
        )
    
    # Check if position exceeds sequence length
    # token_representations has: BOS (idx 0) + sequence (idx 1..N) + EOS (idx N+1)
    max_valid_idx = len(token_representations) - 2  # Leave room for BOS and EOS
    if residue_idx > max_valid_idx:
        seq_length = max_valid_idx
        raise ValueError(
            f"Invalid residue position {residue_idx}: sequence length is {seq_length}"
        )
    
    # Validate sequence bounds if provided
    if sequence is not None:
        if residue_idx > len(sequence):
            raise ValueError(
                f"Invalid residue position {residue_idx}: sequence length is {len(sequence)}"
            )
        
        # Validate that position contains the expected residue
        actual_residue = sequence[residue_idx - 1]  # Convert 1-based to 0-based
        if actual_residue != residue_type:
            raise ValueError(
                f"Position {residue_idx} contains '{actual_residue}', not '{residue_type}'. "
                f"Sequence: {sequence}"
            )
    
    return token_representations[residue_idx]


# ============================================================================
# Main Processing Function
# ============================================================================

def generate_embeddings(df: pd.DataFrame, 
                       output_file: str,
                       model: object,
                       batch_converter: object,
                       repr_layer_id: int,
                       max_seq_length: int = 1024,
                       min_seq_length: int = 0,
                       target: str = 'sequence',
                       output_format: str = 'csv') -> None:
    """
    Generate embeddings for all sequences in a dataframe and write to file.
    
    Args:
        df: DataFrame with columns: label, sequence, residue_of_interest_idx, residue_type, uniprot_id
        output_file: Path to output file (CSV or Parquet)
        model: ESM-2 model
        batch_converter: Batch converter
        repr_layer_id: Layer ID for representations
        max_seq_length: Maximum sequence length to process
        min_seq_length: Minimum sequence length to process
        target: Embedding target ('sequence', 'residue', or 'peptide')
        output_format: Output format ('csv' or 'parquet')
    """
    # Clean up old output file to ensure fresh start (no appending to previous runs)
    output_path = Path(output_file)
    if output_path.exists():
        output_path.unlink()
    
    skipped_count = 0
    processed_count = 0
    write_header = True  # Always write header since we're starting fresh
    
    # For parquet, collect all records in memory
    all_records = [] if output_format == 'parquet' else None
    
    labels = df['label'].tolist()
    sequences = df['sequence'].tolist()
    
    # Handle residue information (may not exist for sequence-only mode)
    if 'residue_of_interest_idx' in df.columns:
        residue_indices = df['residue_of_interest_idx'].tolist()
    else:
        residue_indices = [None] * len(df)
    
    # Get residue types, defaulting to 'C' if column doesn't exist
    if 'residue_type' in df.columns:
        residue_types = df['residue_type'].tolist()
    else:
        residue_types = ['C'] * len(df)
    
    uniprot_ids = df['uniprot_id'].tolist()
    
    # Track cached sequence embeddings to avoid recalculation
    cached_sequence = None
    cached_token_representations = None
    
    for i, (label, seq, residue_idx, residue_type, uniprot_id) in enumerate(
        zip(labels, sequences, residue_indices, residue_types, uniprot_ids)
    ):
        # Skip if sequence length is out of range
        if not (min_seq_length < len(seq) < max_seq_length):
            print(f"[{i+1}/{len(df)}] Skipping {label} - length {len(seq)} outside range "
                  f"[{min_seq_length}, {max_seq_length})")
            skipped_count += 1
            continue
        
        # Only calculate token representations if this is a new sequence
        if seq != cached_sequence:
            # Clear previous cache to free memory
            cached_token_representations = None
            cached_sequence = seq
            
            # Generate embeddings for the new sequence
            print(f"[{i+1}/{len(df)}] Processing {label} (length={len(seq)})")
            token_representations = process_batch(
                model, (label, seq), batch_converter, repr_layer_id
            )
            # Cache for reuse with other residues from same sequence
            cached_token_representations = token_representations
        else:
            # Reuse cached token representations for this sequence
            token_representations = cached_token_representations
            print(f"[{i+1}/{len(df)}] Processing {label} (cached - length={len(seq)})")
        
        # Generate embedding based on target
        try:
            if target == 'residue':
                embedding = get_residue_embedding(residue_idx, token_representations, sequence=seq, residue_type=residue_type)
            elif target == 'sequence':
                embedding = get_whole_sequence_embedding(token_representations, method='mean')
            elif target == 'peptide':
                embedding = get_whole_sequence_embedding(token_representations, method='mean')
            else:
                raise ValueError(f"Unknown target: {target}")
        except ValueError as e:
            print(f"[{i+1}/{len(df)}] SKIPPING {label} - Invalid residue specification: {str(e)}", file=sys.stderr)
            skipped_count += 1
            continue
        
        # Prepare output with rounded embeddings (3 decimals)
        rounded_embedding = np.round(embedding, 3)
        # Format embedding as string with exactly 3 decimal places
        embedding_str = '[' + ', '.join(f'{x:.3f}' for x in rounded_embedding) + ']'
        output_data = {
            'label': label,
            'uniprot_id': uniprot_id if pd.notna(uniprot_id) else '',
            'embedding': embedding_str
        }
        
        if output_format == 'parquet':
            # Collect records for batch writing
            all_records.append(output_data)
        else:
            # Write to CSV incrementally
            output_df = pd.DataFrame([output_data])
            mode = 'w' if write_header else 'a'
            header = write_header
            output_df.to_csv(output_file, mode=mode, header=header, index=False)
            write_header = False
        
        processed_count += 1
    
    # Write parquet file if that format was selected
    if output_format == 'parquet' and all_records:
        output_df = pd.DataFrame(all_records)
        output_df.to_parquet(output_file, index=False, engine='pyarrow')
    
    # Clean up cached embeddings
    cached_token_representations = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\nCompleted: {processed_count} sequences processed, {skipped_count} skipped")


# ============================================================================
# Model Setup Functions
# ============================================================================

def load_esm2_model(model_name: str, 
                    use_fsdp: bool = False, 
                    cache_dir: str = None,
                    port: int = None):
    """
    Load ESM-2 model with optional distributed training support.
    
    Args:
        model_name: ESM-2 model identifier
        use_fsdp: Whether to use Fully Sharded Data Parallel
        cache_dir: Directory to cache model weights
        port: Port for distributed training communication
        
    Returns:
        Tuple of (model, vocabulary, batch_converter, repr_layer_id)
    """
    # Set cache directory if provided
    if cache_dir is not None:
        torch.hub.set_dir(cache_dir)
    
    # Extract representation layer ID from model name
    repr_layer_id = int(model_name.split('_t')[1].split('_')[0])
    
    # Download model and data
    model_data, regression_data = esm.pretrained._download_model_and_regression_data(model_name)
    
    if not use_fsdp:
        # Simple single-GPU loading
        model, vocab = esm.pretrained.load_model_and_alphabet_core(
            model_name, model_data, regression_data
        )
        model = model.cuda() if torch.cuda.is_available() else model.cpu()
        batch_converter = vocab.get_batch_converter()
        model.eval()
    else:
        # Distributed loading with FSDP
        if port is None:
            port = torch.randint(10000, 48000, (1,)).item()
        
        url = f"tcp://localhost:{port}"
        torch.distributed.init_process_group(
            backend="gloo", 
            init_method=url, 
            world_size=1, 
            rank=0
        )
        
        fsdp_params = dict(
            mixed_precision=True,
            flatten_parameters=True,
            state_dict_device=torch.device("cpu"),
            cpu_offload=True,
        )
        
        with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
            model, vocab = esm.pretrained.load_model_and_alphabet_core(
                model_name, model_data, regression_data
            )
            batch_converter = vocab.get_batch_converter()
            model.eval()
            
            # Wrap each layer in FSDP
            for name, child in model.named_children():
                if name == "layers":
                    for layer_name, layer in child.named_children():
                        wrapped_layer = wrap(layer)
                        setattr(child, layer_name, wrapped_layer)
            model = wrap(model)
    
    return model, vocab, batch_converter, repr_layer_id


# ============================================================================
# Main Entry Point
# ============================================================================

def main(args):
    """Main execution function."""
    
    print("=" * 80)
    print("ESM-2 Protein Sequence Embeddings Generator")
    print("=" * 80)
    
    # Check target mode and warn if residue_type is provided but not needed
    if args.target in ['sequence', 'peptide'] and args.residue_type != 'C':
        print(f"\nWARNING: --residue_type '{args.residue_type}' is ignored when --target is '{args.target}'")
        print("Residue-level embeddings will not be calculated.")
    
    # Load and validate input data
    print(f"\nLoading sequences from {args.seq_file}...")
    
    if args.target in ['sequence', 'peptide']:
        # For sequence and peptide targets, load without residue processing
        if args.seq_file.endswith('.csv'):
            df = pd.read_csv(args.seq_file)
            # Validate required columns
            required_cols = {'uniprot_id', 'pep_seq'}
            if not required_cols.issubset(df.columns):
                raise ValueError(
                    f"CSV must contain columns: {required_cols}. "
                    f"Found: {set(df.columns)}"
                )
            # Filter out rows with missing sequences
            df = df.dropna(subset=['pep_seq'])
            # Create label based on target
            if args.target == 'peptide':
                # For peptide target, label is uniprot_id_sequence
                df['label'] = df['uniprot_id'].astype(str) + '_' + df['pep_seq'].astype(str)
            else:
                # For sequence target, label is just uniprot_id
                df['label'] = df['uniprot_id'].astype(str)
            # Rename pep_seq to sequence
            df_formatted = df[['label', 'pep_seq', 'uniprot_id']].copy()
            df_formatted.columns = ['label', 'sequence', 'uniprot_id']
        elif args.seq_file.endswith('.fasta') or args.seq_file.endswith('.fa'):
            records = list(SeqIO.parse(args.seq_file, 'fasta'))
            data = []
            for record in records:
                header_parts = record.id.split('|')
                uniprot_id = header_parts[1] if len(header_parts) >= 2 else record.id
                seq = str(record.seq)
                if args.target == 'peptide':
                    label = f"{uniprot_id}_{seq}"
                else:
                    label = uniprot_id
                data.append({
                    'label': label,
                    'sequence': seq,
                    'uniprot_id': uniprot_id
                })
            df_formatted = pd.DataFrame(data)
        else:
            raise ValueError("Input file must be .csv or .fasta")
        
        print(f"Loaded {len(df_formatted)} sequences ({args.target} mode)")
    else:
        # For residue target, use residue processing
        if args.seq_file.endswith('.csv'):
            df = load_sequence_csv(args.seq_file, residue_type=args.residue_type, all_residues=args.all_residues)
        elif args.seq_file.endswith('.fasta') or args.seq_file.endswith('.fa'):
            df = load_sequence_fasta(args.seq_file, residue_type=args.residue_type)
        else:
            raise ValueError("Input file must be .csv or .fasta")
        
        if args.all_residues:
            print(f"Loaded {len(df)} residue instances from sequences")
        else:
            print(f"Loaded {len(df)} sequences")
        
        # Format data (only needed when not using all_residues from CSV)
        if args.seq_file.endswith('.csv') and not args.all_residues:
            print(f"\nFormatting sequences for target: {args.target}...")
            df_formatted = format_sequence_dataframe(df, residue_type=args.residue_type)
        else:
            # Already formatted for FASTA or all_residues mode
            df_formatted = df
    
    # Load model
    print(f"\nLoading ESM-2 model: {args.model_name}...")
    model, vocab, batch_converter, repr_layer_id = load_esm2_model(
        args.model_name,
        use_fsdp=args.use_fsdp,
        cache_dir=args.cache_dir,
        port=args.port if hasattr(args, 'port') else None
    )
    
    # Generate embeddings
    print(f"\nGenerating embeddings with target: {args.target}...")
    print(f"Output format: {args.output_format}")
    generate_embeddings(
        df_formatted,
        output_file=args.output_file,
        model=model,
        batch_converter=batch_converter,
        repr_layer_id=repr_layer_id,
        max_seq_length=args.max_seq_length,
        min_seq_length=args.min_seq_length,
        target=args.target,
        output_format=args.output_format
    )
    
    print(f"\nEmbeddings written to: {args.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate protein sequence embeddings using ESM-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate whole-sequence embeddings from CSV
  python generate_esm2_embeddings.py \\
      --seq_file sequences.csv \\
      --model_name esm2_t6_8M_UR50D \\
      --target sequence \\
      --max_seq_length 1024

  # Generate cysteine-specific embeddings from FASTA
  python generate_esm2_embeddings.py \\
      --seq_file proteins.fasta \\
      --model_name esm2_t33_650M_UR50D \\
      --target residue \\
      --residue_type C \\
      --max_seq_length 1024

  # Generate embeddings and save to Parquet format
  python generate_esm2_embeddings.py \\
      --seq_file sequences.csv \\
      --model_name esm2_t6_8M_UR50D \\
      --target sequence \\
      --output_format parquet \\
      --max_seq_length 1024

  # Generate embeddings for peptide sequences with peptide-specific labels
  python generate_esm2_embeddings.py \\
      --seq_file peptides.csv \\
      --model_name esm2_t12_35M_UR50D \\
      --target peptide \\
      --max_seq_length 100

  # Generate embeddings for all cysteines in peptide sequences
  python generate_esm2_embeddings.py \\
      --seq_file peptides.csv \\
      --model_name esm2_t12_35M_UR50D \\
      --target residue \\
      --residue_type C \\
      --max_seq_length 100

  # Generate embeddings for specific residue positions only
  python generate_esm2_embeddings.py \\
      --seq_file sequences.csv \\
      --model_name esm2_t30_150M_UR50D \\
      --target residue \\
      --no-all_residues \\
      --max_seq_length 1024
        """
    )
    
    # ESM-2 model options
    esm2_models = [
        "esm2_t48_15B_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t6_8M_UR50D",
        "esm_msa1b_t12_100M_UR50S",
    ]
    
    # Input data arguments
    parser.add_argument(
        '--seq_file', 
        type=str, 
        required=True,
        help='Path to input FASTA or CSV file with sequences'
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True,
        choices=esm2_models,
        help='ESM-2 model identifier'
    )
    parser.add_argument(
        '--target', 
        type=str, 
        required=True,
        choices=['sequence', 'residue', 'peptide'],
        help='Embedding target: sequence (whole sequence mean), residue (specific residue), or peptide (sequence with label as uniprot_id_sequence)'
    )
    parser.add_argument(
        '--residue_type', 
        type=str, 
        default='C',
        help='Type of residue to extract (default: C for cysteine)'
    )
    parser.add_argument(
        '--all_residues',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Generate embeddings for all instances of residue_type in each sequence (default). '
             'Use --no-all_residues to process only specific residue positions from CSV residue column.'
    )
    
    # Sequence filtering arguments
    parser.add_argument(
        '--max_seq_length', 
        type=int, 
        required=True,
        default=1024,
        help='Maximum sequence length to process (default: 1024)'
    )
    parser.add_argument(
        '--min_seq_length', 
        type=int, 
        default=5,
        help='Minimum sequence length to process (default: 5)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_file', 
        type=str, 
        required=True,
        help='Path to output file with embeddings (CSV or Parquet)'
    )
    parser.add_argument(
        '--output_format',
        type=str,
        choices=['csv', 'parquet'],
        default='csv',
        help='Output file format (default: csv)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--cache_dir', 
        type=str,
        help='Directory to cache model weights (optional)'
    )
    parser.add_argument(
        '--use_fsdp', 
        action='store_true',
        help='Use Fully Sharded Data Parallel for large models'
    )
    parser.add_argument(
        '--port',
        type=int,
        help='Port for distributed training (auto-assigned if not specified)'
    )
    
    args = parser.parse_args()
    
    # Auto-assign port if using FSDP and not specified
    if args.use_fsdp and not hasattr(args, 'port'):
        args.port = torch.randint(10000, 48000, (1,)).item()
    
    main(args)
