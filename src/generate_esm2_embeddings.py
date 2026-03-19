#!/usr/bin/env python3
"""
Generate protein sequence embeddings using ESM-2 (Evolutionary Scale Modeling).

This script converts protein sequences into numerical representations that capture
biological meaning, with support for aggregating embeddings at specific residue sites.

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the MIT license.
"""

import argparse
import csv
import pandas as pd
import numpy as np
import torch
import sys
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple
from Bio import SeqIO
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
import esm

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


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

def extract_residue_metadata(residue_value: object) -> Tuple[int, str]:
    """Extract 1-based residue index and residue token from a residue label such as C45."""
    residue_str = str(residue_value)
    residue_match = re.fullmatch(r'([A-Za-z])(\d+)', residue_str)
    if residue_match is None:
        raise ValueError(
            f"Invalid residue value '{residue_value}'. Expected format like 'C45'."
        )

    residue_type, residue_idx = residue_match.groups()
    return int(residue_idx), residue_type.upper()


def iter_sequence_csv(
    csv_path: str,
    target: str,
    residue_type: str = 'C',
    all_residues: bool = True,
    chunksize: int = 1000,
) -> Iterator[Dict[str, object]]:
    """Stream formatted records from a CSV input file."""
    header_df = pd.read_csv(csv_path, nrows=0)

    if target in ['sequence', 'peptide']:
        required_cols = {'uniprot_id', 'pep_seq'}
    elif all_residues:
        required_cols = {'uniprot_id', 'pep_seq'}
    else:
        required_cols = {'uniprot_id', 'pep_seq', 'residue'}

    if not required_cols.issubset(header_df.columns):
        raise ValueError(
            f"CSV must contain columns: {required_cols}. Found: {set(header_df.columns)}"
        )

    residue_type = residue_type.upper()
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk = chunk.dropna(subset=['pep_seq'])

        for row in chunk.itertuples(index=False):
            uniprot_id = str(getattr(row, 'uniprot_id'))
            sequence = str(getattr(row, 'pep_seq'))

            if target == 'sequence':
                yield {
                    'label': uniprot_id,
                    'sequence': sequence,
                    'uniprot_id': uniprot_id,
                }
                continue

            if target == 'peptide':
                yield {
                    'label': f"{uniprot_id}_{sequence}",
                    'sequence': sequence,
                    'uniprot_id': uniprot_id,
                }
                continue

            if all_residues:
                found_residue = False
                for pos, char in enumerate(sequence, start=1):
                    if char == residue_type:
                        found_residue = True
                        yield {
                            'label': f"{uniprot_id}_{residue_type}{pos}",
                            'sequence': sequence,
                            'residue_of_interest_idx': pos,
                            'residue_type': residue_type,
                            'uniprot_id': uniprot_id,
                        }

                if not found_residue:
                    continue
                continue

            residue_idx, row_residue_type = extract_residue_metadata(getattr(row, 'residue'))
            yield {
                'label': f"{uniprot_id}_{getattr(row, 'residue')}",
                'sequence': sequence,
                'residue_of_interest_idx': residue_idx,
                'residue_type': row_residue_type,
                'uniprot_id': uniprot_id,
            }


def iter_sequence_fasta(
    fasta_path: str,
    target: str,
    residue_type: str = 'C',
) -> Iterator[Dict[str, object]]:
    """Stream formatted records from a FASTA input file."""
    residue_type = residue_type.upper()

    for record in SeqIO.parse(fasta_path, 'fasta'):
        header_parts = record.id.split('|')
        uniprot_id = header_parts[1] if len(header_parts) >= 2 else record.id
        sequence = str(record.seq)

        if target == 'sequence':
            yield {
                'label': uniprot_id,
                'sequence': sequence,
                'uniprot_id': uniprot_id,
            }
            continue

        if target == 'peptide':
            yield {
                'label': f"{uniprot_id}_{sequence}",
                'sequence': sequence,
                'uniprot_id': uniprot_id,
            }
            continue

        for pos, char in enumerate(sequence, start=1):
            if char == residue_type:
                yield {
                    'label': f"{uniprot_id}_{residue_type}{pos}",
                    'sequence': sequence,
                    'residue_of_interest_idx': pos,
                    'residue_type': residue_type,
                    'uniprot_id': uniprot_id,
                }


def iter_input_records(
    seq_file: str,
    target: str,
    residue_type: str = 'C',
    all_residues: bool = True,
    csv_chunksize: int = 1000,
) -> Iterator[Dict[str, object]]:
    """Create a streaming iterator over formatted input records."""
    if seq_file.endswith('.csv'):
        return iter_sequence_csv(
            seq_file,
            target=target,
            residue_type=residue_type,
            all_residues=all_residues,
            chunksize=csv_chunksize,
        )

    if seq_file.endswith('.fasta') or seq_file.endswith('.fa'):
        return iter_sequence_fasta(seq_file, target=target, residue_type=residue_type)

    raise ValueError("Input file must be .csv or .fasta")


def flush_parquet_records(
    parquet_writer: Optional[object],
    records: List[Dict[str, object]],
    output_path: Path,
) -> Optional[object]:
    """Flush buffered records to parquet without holding the full output in memory."""
    if not records:
        return parquet_writer

    if pa is None or pq is None:
        raise ImportError(
            "pyarrow is required for parquet output. Install it with `pip install pyarrow`."
        )

    table = pa.Table.from_pylist(records)
    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')

    parquet_writer.write_table(table)
    records.clear()
    return parquet_writer

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

def generate_embeddings(records: Iterable[Dict[str, object]], 
                       output_file: str,
                       model: object,
                       batch_converter: object,
                       repr_layer_id: int,
                       max_seq_length: int = 1024,
                       min_seq_length: int = 0,
                       target: str = 'sequence',
                       output_format: str = 'csv',
                       parquet_chunk_size: int = 256) -> None:
    """
    Generate embeddings for all sequences in a dataframe and write to file.
    
    Args:
        records: Iterable of formatted sequence records
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    
    skipped_count = 0
    processed_count = 0
    csv_file = None
    csv_writer = None
    parquet_writer = None
    parquet_records: List[Dict[str, object]] = []
    
    # Track cached sequence embeddings to avoid recalculation
    cached_sequence = None
    cached_token_representations = None

    try:
        for i, record in enumerate(records, start=1):
            label = str(record['label'])
            seq = str(record['sequence'])
            residue_idx = record.get('residue_of_interest_idx')
            residue_type = str(record.get('residue_type', 'C')).upper()
            uniprot_id = record.get('uniprot_id', '')

            if not (min_seq_length < len(seq) < max_seq_length):
                print(
                    f"[{i}] Skipping {label} - length {len(seq)} outside range "
                    f"[{min_seq_length}, {max_seq_length})"
                )
                skipped_count += 1
                continue

            if seq != cached_sequence:
                cached_token_representations = None
                cached_sequence = seq

                print(f"[{i}] Processing {label} (length={len(seq)})")
                token_representations = process_batch(
                    model, (label, seq), batch_converter, repr_layer_id
                )
                cached_token_representations = token_representations
            else:
                token_representations = cached_token_representations
                print(f"[{i}] Processing {label} (cached - length={len(seq)})")

            try:
                if target == 'residue':
                    embedding = get_residue_embedding(
                        residue_idx,
                        token_representations,
                        sequence=seq,
                        residue_type=residue_type,
                    )
                elif target in ['sequence', 'peptide']:
                    embedding = get_whole_sequence_embedding(token_representations, method='mean')
                else:
                    raise ValueError(f"Unknown target: {target}")
            except ValueError as e:
                print(
                    f"[{i}] SKIPPING {label} - Invalid residue specification: {str(e)}",
                    file=sys.stderr,
                )
                skipped_count += 1
                continue

            output_data = {'index': label}
            output_data.update(embedding_to_feature_dict(embedding))

            if output_format == 'parquet':
                parquet_records.append(output_data)
                if len(parquet_records) >= parquet_chunk_size:
                    parquet_writer = flush_parquet_records(
                        parquet_writer,
                        parquet_records,
                        output_path,
                    )
            else:
                if csv_writer is None:
                    csv_file = output_path.open('w', newline='')
                    csv_writer = csv.DictWriter(csv_file, fieldnames=list(output_data.keys()))
                    csv_writer.writeheader()
                csv_writer.writerow(output_data)

            processed_count += 1

            del embedding

        if output_format == 'parquet':
            parquet_writer = flush_parquet_records(parquet_writer, parquet_records, output_path)
    finally:
        cached_token_representations = None
        if parquet_writer is not None:
            parquet_writer.close()
        if csv_file is not None:
            csv_file.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nCompleted: {processed_count} sequences processed, {skipped_count} skipped")


def embedding_to_feature_dict(embedding: np.ndarray, prefix: str = 'feat_') -> dict:
    """Convert embedding vector to expanded feature columns (feat_1, feat_2, ...)."""
    rounded_embedding = np.round(embedding, 3)
    return {
        f'{prefix}{i + 1}': float(value)
        for i, value in enumerate(rounded_embedding)
    }


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
    
    print(f"\nStreaming sequences from {args.seq_file}...")
    if args.target == 'residue' and args.all_residues:
        print(f"Residue mode enabled; expanding {args.residue_type.upper()} positions lazily during processing")

    input_records = iter_input_records(
        args.seq_file,
        target=args.target,
        residue_type=args.residue_type,
        all_residues=args.all_residues,
    )
    
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
        input_records,
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
