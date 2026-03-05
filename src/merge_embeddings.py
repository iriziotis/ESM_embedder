#!/usr/bin/env python3
"""
Merge and concatenate ESM-2 embeddings for cysteines, peptides, and proteins.

This script efficiently combines embeddings from different sequence types (cysteines,
peptides, proteins) into merged feature tables suitable for downstream analysis.
Supports batch processing and parallelization for large datasets.

Features:
- Efficient loading of embeddings (supports .npy, .npz, parquet, CSV)
- Four types of merged tables: cys+pep, cys+pep+protein, cys+protein, pep+protein
- Flexible aggregation strategies (shortest peptide or average embeddings)
- Parallel processing with configurable worker count
- Handles large files using chunking and memory-efficient I/O

Usage:
    python merge_embeddings.py \
        --cysteine-embeddings path/to/cys_embeddings.csv \
        --peptide-embeddings path/to/pep_embeddings.csv \
        --protein-embeddings path/to/protein_embeddings.csv \
        --mapping path/to/mapping.csv \
        --output-dir path/to/output \
        --aggregation-method shortest \
        --tables all
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# Setup Logging
# ============================================================================

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Configure logging for the script."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


logger = setup_logging()


# ============================================================================
# Embedding Loading Functions
# ============================================================================

class EmbeddingLoader:
    """Load embeddings from various file formats."""
    
    @staticmethod
    def load_embedding(file_path: Union[str, Path], indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Load embeddings from various formats.
        
        Args:
            file_path: Path to embedding file (.npy, .npz, .parquet, .csv)
            indices: Optional list of indices to load (for selective loading)
            
        Returns:
            Embeddings as numpy array
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.npy':
            return EmbeddingLoader._load_npy(file_path, indices)
        elif file_path.suffix == '.npz':
            return EmbeddingLoader._load_npz(file_path, indices)
        elif file_path.suffix == '.parquet':
            return EmbeddingLoader._load_parquet(file_path, indices)
        elif file_path.suffix == '.csv':
            return EmbeddingLoader._load_csv(file_path, indices)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    @staticmethod
    def _load_npy(file_path: Path, indices: Optional[List[int]] = None) -> np.ndarray:
        """Load from numpy binary format."""
        embeddings = np.load(file_path, allow_pickle=False)
        if indices is not None:
            embeddings = embeddings[indices]
        return embeddings
    
    @staticmethod
    def _load_npz(file_path: Path, indices: Optional[List[int]] = None) -> np.ndarray:
        """Load from numpy compressed format."""
        data = np.load(file_path, allow_pickle=False)
        # Assume main embedding data is in 'embeddings' key or first array
        if 'embeddings' in data:
            embeddings = data['embeddings']
        else:
            embeddings = data[list(data.keys())[0]]
        
        if indices is not None:
            embeddings = embeddings[indices]
        return embeddings
    
    @staticmethod
    def _load_parquet(file_path: Path, indices: Optional[List[int]] = None) -> np.ndarray:
        """Load from parquet format (assumes embedding columns are numeric)."""
        df = pd.read_parquet(file_path)
        # Filter to numeric columns (assumes these are embeddings)
        numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
        embeddings = df[numeric_cols].values
        
        if indices is not None:
            embeddings = embeddings[indices]
        return embeddings
    
    @staticmethod
    def _load_csv(file_path: Path, indices: Optional[List[int]] = None) -> np.ndarray:
        """Load from CSV format (handles both numeric columns and string embeddings)."""
        df = pd.read_csv(file_path)
        
        # Check if there's an 'embedding' column with string representations
        if 'embedding' in df.columns:
            # Parse string representations like "[1.0, 2.0, 3.0]" to arrays
            import ast
            embeddings = np.array([ast.literal_eval(emb) for emb in df['embedding']], dtype=np.float32)
        else:
            # Old format: index column + numeric columns
            if df.columns[0] in ['label', 'id'] or df.iloc[:, 0].dtype == 'object':
                # First column is labels, rest are embeddings
                embeddings = df.iloc[:, 1:].values
            else:
                embeddings = df.values
        
        if indices is not None:
            embeddings = embeddings[indices]
        return embeddings
    
    @staticmethod
    def load_with_metadata(
        file_path: Union[str, Path],
        indices: Optional[List[int]] = None,
        key_column: str = 'id'
    ) -> Tuple[np.ndarray, pd.Series]:
        """
        Load embeddings and associated metadata.
        
        Args:
            file_path: Path to embedding file
            indices: Optional indices to load
            key_column: Name of column containing identifiers
            
        Returns:
            Tuple of (embeddings, identifiers)
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
            if key_column in df.columns:
                ids = df[key_column]
                numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
                embeddings = df[numeric_cols].values
            else:
                ids = pd.Index(range(len(df)))
                numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
                embeddings = df[numeric_cols].values
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            
            # Handle different CSV formats
            if 'embedding' in df.columns:
                # New format: 'label' and 'embedding' columns
                import ast
                embeddings = np.array([ast.literal_eval(emb) for emb in df['embedding']], dtype=np.float32)
                if 'label' in df.columns:
                    ids = df['label']
                else:
                    ids = pd.Index(range(len(df)))
            else:
                # Old format: index column + numeric columns
                if df.columns[0] in ['label', 'id']:
                    ids = df[df.columns[0]]
                    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
                    embeddings = df[numeric_cols].values
                else:
                    # First column is index
                    ids = df.iloc[:, 0]
                    numeric_cols = df.iloc[:, 1:].select_dtypes(include=['float64', 'float32']).columns
                    embeddings = df[numeric_cols].values
        else:
            embeddings = EmbeddingLoader.load_embedding(file_path)
            ids = pd.Index(range(len(embeddings)))
        
        if indices is not None:
            embeddings = embeddings[indices]
            ids = ids[indices]
        
        return embeddings, ids


# ============================================================================
# Mapping and Aggregation Functions
# ============================================================================

class MappingHandler:
    """Handle mapping between cysteines, peptides, and proteins."""
    
    def __init__(self, mapping_df: pd.DataFrame):
        """
        Initialize with mapping dataframe.
        
        Args:
            mapping_df: DataFrame with columns: uniprot_id, pep_seq, residue
                       (residue column contains cysteine identifier)
        """
        self.mapping = mapping_df.copy()
        self._validate_columns()
        self._create_indices()
    
    def _validate_columns(self) -> None:
        """Validate required columns exist."""
        required = {'uniprot_id', 'pep_seq', 'residue'}
        missing = required - set(self.mapping.columns)
        if missing:
            raise ValueError(f"Missing columns in mapping: {missing}")
    
    def _create_indices(self) -> None:
        """Create index mappings for efficient lookup."""
        # Create full cysteine ID (uniprot_id_residue) for each row
        self.mapping['cys_full_id'] = self.mapping['uniprot_id'] + '_' + self.mapping['residue']
        
        # Cysteine (full ID) to peptides mapping
        self.cys_to_peps = self.mapping.groupby('cys_full_id')['pep_seq'].apply(list).to_dict()
        
        # Cysteine (full ID) to protein mapping (though it's redundant, keeps consistency)
        self.cys_to_proteins = self.mapping.groupby('cys_full_id')['uniprot_id'].apply(
            lambda x: list(set(x))
        ).to_dict()
        
        # Also keep residue-based lookups for backward compatibility
        self.residue_to_peps = self.mapping.groupby('residue')['pep_seq'].apply(list).to_dict()
        self.residue_to_proteins = self.mapping.groupby('residue')['uniprot_id'].apply(
            lambda x: list(set(x))
        ).to_dict()
        
        # Peptide to proteins mapping
        self.pep_to_proteins = self.mapping.groupby('pep_seq')['uniprot_id'].apply(
            lambda x: list(set(x))
        ).to_dict()
        
        # Cysteine to (peptide, protein) pairs - using full ID
        self.cys_to_pep_prot = self.mapping.groupby('cys_full_id', group_keys=False).apply(
            lambda x: list(zip(x['pep_seq'], x['uniprot_id'])), include_groups=False
        ).to_dict()
    
    def get_peptides_for_cysteine(self, cys_id: str) -> List[str]:
        """Get all peptides containing a cysteine."""
        return self.cys_to_peps.get(cys_id, [])
    
    def get_proteins_for_cysteine(self, cys_id: str) -> List[str]:
        """Get all proteins containing a cysteine."""
        return self.cys_to_proteins.get(cys_id, [])
    
    def get_shortest_peptide(self, cys_id: str) -> Optional[str]:
        """Get shortest peptide containing cysteine."""
        peps = self.get_peptides_for_cysteine(cys_id)
        if not peps:
            return None
        return min(peps, key=len)
    
    def get_proteins_for_peptide(self, pep_seq: str) -> List[str]:
        """Get all proteins containing a peptide."""
        return self.pep_to_proteins.get(pep_seq, [])


class EmbeddingAggregator:
    """Aggregate embeddings using various strategies."""
    
    @staticmethod
    def average_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """Average multiple embeddings."""
        if len(embeddings) == 0:
            return None
        return np.nanmean(embeddings, axis=0)
    
    @staticmethod
    def concatenate_embeddings(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
        """Concatenate embeddings along feature axis."""
        if not embeddings or any(e is None for e in embeddings):
            return None
        return np.concatenate(embeddings, axis=0)


# ============================================================================
# Merge Table Creation Functions
# ============================================================================

class MergeTablesCreator:
    """Create merged embedding tables."""
    
    def __init__(
        self,
        cys_embeddings: np.ndarray,
        pep_embeddings: np.ndarray,
        protein_embeddings: np.ndarray,
        cys_ids: List[str],
        pep_ids: List[str],
        protein_ids: List[str],
        mapping: MappingHandler,
        aggregation_method: str = 'shortest'
    ):
        """
        Initialize merger.
        
        Args:
            cys_embeddings: Cysteine embeddings array
            pep_embeddings: Peptide embeddings array
            protein_embeddings: Protein embeddings array
            cys_ids: List of cysteine identifiers
            pep_ids: List of peptide identifiers
            protein_ids: List of protein identifiers
            mapping: MappingHandler instance
            aggregation_method: 'shortest' or 'average' for peptide aggregation
        """
        self.cys_emb = cys_embeddings
        self.pep_emb = pep_embeddings
        self.prot_emb = protein_embeddings
        
        self.cys_ids = np.array(cys_ids)
        self.pep_ids = np.array(pep_ids)
        self.prot_ids = np.array(protein_ids)
        
        # Create mappings for index lookup
        self.cys_idx_map = {cid: idx for idx, cid in enumerate(cys_ids)}
        self.pep_idx_map = {pid: idx for idx, pid in enumerate(pep_ids)}
        self.prot_idx_map = {pid: idx for idx, pid in enumerate(protein_ids)}
        
        self.mapping = mapping
        self.aggregation_method = aggregation_method
        
        logger.info(f"Initialized MergeTablesCreator with:")
        logger.info(f"  Cysteine embeddings: {self.cys_emb.shape}")
        logger.info(f"  Peptide embeddings: {self.pep_emb.shape}")
        logger.info(f"  Protein embeddings: {self.prot_emb.shape}")
        logger.info(f"  Aggregation method: {self.aggregation_method}")
    
    def create_cys_pep_table(self) -> pd.DataFrame:
        """
        Create table of cysteine + peptide embeddings.
        
        Returns:
            DataFrame with columns: label, aggregation_method, embedding
        """
        logger.info("Creating cysteine + peptide merged table...")
        
        rows = []
        for cys_id in tqdm(self.cys_ids, desc="Processing cysteines"):
            pep_seqs = self.mapping.get_peptides_for_cysteine(cys_id)
            
            # Get cysteine embedding
            cys_idx = self.cys_idx_map.get(cys_id)
            if cys_idx is None:
                continue
            cys_emb = self.cys_emb[cys_idx]
            
            # Select peptide(s) based on aggregation method
            if self.aggregation_method == 'shortest':
                pep_seq = self.mapping.get_shortest_peptide(cys_id)
                if pep_seq is None:
                    continue
                pep_seqs_to_use = [pep_seq]
            else:  # average
                pep_seqs_to_use = pep_seqs
            
            # Get peptide embeddings and aggregate
            pep_embs = []
            for pep_seq in pep_seqs_to_use:
                pep_idx = self.pep_idx_map.get(pep_seq)
                if pep_idx is not None:
                    pep_embs.append(self.pep_emb[pep_idx])
            
            if not pep_embs:
                continue
            
            pep_emb = EmbeddingAggregator.average_embeddings(np.array(pep_embs))
            if pep_emb is None:
                continue
            
            # Concatenate embeddings
            merged_emb = EmbeddingAggregator.concatenate_embeddings([cys_emb, pep_emb])
            if merged_emb is None:
                continue
            
            rows.append({
                'label': cys_id,
                'aggregation_method': self.aggregation_method,
                'embedding': merged_emb
            })
        
        # Convert to DataFrame
        if not rows:
            logger.warning("No valid rows created for cys+pep table")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Expand embedding columns
        embedding_df = pd.DataFrame(
            df['embedding'].tolist(),
            index=df.index,
            columns=[f'feat_{i}' for i in range(df['embedding'].iloc[0].shape[0])]
        )
        
        result = pd.concat(
            [df[['label', 'aggregation_method']], embedding_df],
            axis=1
        )
        
        logger.info(f"Created cys+pep table with {len(result)} rows and {result.shape[1]} columns")
        return result
    
    def create_cys_pep_prot_table(self) -> pd.DataFrame:
        """
        Create table of cysteine + peptide + protein embeddings.
        
        Returns:
            DataFrame with columns: cysteine_label, peptide_label, uniprot_id, embedding
        """
        logger.info("Creating cysteine + peptide + protein merged table...")
        
        rows = []
        for cys_id in tqdm(self.cys_ids, desc="Processing cysteines"):
            cys_pep_prot = self.mapping.cys_to_pep_prot.get(cys_id, [])
            
            # Get cysteine embedding
            cys_idx = self.cys_idx_map.get(cys_id)
            if cys_idx is None:
                continue
            cys_emb = self.cys_emb[cys_idx]
            
            # Process each (peptide, protein) pair
            for pep_seq, prot_id in cys_pep_prot:
                pep_idx = self.pep_idx_map.get(pep_seq)
                prot_idx = self.prot_idx_map.get(prot_id)
                
                if pep_idx is None or prot_idx is None:
                    continue
                
                pep_emb = self.pep_emb[pep_idx]
                prot_emb = self.prot_emb[prot_idx]
                
                # Concatenate embeddings
                merged_emb = EmbeddingAggregator.concatenate_embeddings(
                    [cys_emb, pep_emb, prot_emb]
                )
                if merged_emb is None:
                    continue
                
                rows.append({
                    'cysteine_label': cys_id,
                    'peptide_label': pep_seq,
                    'uniprot_id': prot_id,
                    'embedding': merged_emb
                })
        
        # Convert to DataFrame
        if not rows:
            logger.warning("No valid rows created for cys+pep+prot table")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Expand embedding columns
        embedding_df = pd.DataFrame(
            df['embedding'].tolist(),
            index=df.index,
            columns=[f'feat_{i}' for i in range(df['embedding'].iloc[0].shape[0])]
        )
        
        result = pd.concat(
            [df[['cysteine_label', 'peptide_label', 'uniprot_id']], embedding_df],
            axis=1
        )
        
        logger.info(f"Created cys+pep+prot table with {len(result)} rows and {result.shape[1]} columns")
        return result
    
    def create_cys_prot_table(self) -> pd.DataFrame:
        """
        Create table of cysteine + protein embeddings.
        
        Returns:
            DataFrame with columns: label, uniprot_id, embedding
        """
        logger.info("Creating cysteine + protein merged table...")
        
        rows = []
        # Get unique cysteine-protein pairs from mapping
        cys_prot_pairs = self.mapping.mapping[['cys_full_id', 'uniprot_id']].drop_duplicates()
        
        for _, row in tqdm(cys_prot_pairs.iterrows(), total=len(cys_prot_pairs), 
                          desc="Processing cysteine-protein pairs"):
            cys_id = row['cys_full_id']
            prot_id = row['uniprot_id']
            
            cys_idx = self.cys_idx_map.get(cys_id)
            prot_idx = self.prot_idx_map.get(prot_id)
            
            if cys_idx is None or prot_idx is None:
                continue
            
            cys_emb = self.cys_emb[cys_idx]
            prot_emb = self.prot_emb[prot_idx]
            
            # Concatenate embeddings
            merged_emb = EmbeddingAggregator.concatenate_embeddings([cys_emb, prot_emb])
            if merged_emb is None:
                continue
            
            rows.append({
                'label': cys_id,
                'uniprot_id': prot_id,
                'embedding': merged_emb
            })
        
        # Convert to DataFrame
        if not rows:
            logger.warning("No valid rows created for cys+prot table")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Expand embedding columns
        embedding_df = pd.DataFrame(
            df['embedding'].tolist(),
            index=df.index,
            columns=[f'feat_{i}' for i in range(df['embedding'].iloc[0].shape[0])]
        )
        
        result = pd.concat(
            [df[['label', 'uniprot_id']], embedding_df],
            axis=1
        )
        
        logger.info(f"Created cys+prot table with {len(result)} rows and {result.shape[1]} columns")
        return result
    
    def create_pep_prot_table(self) -> pd.DataFrame:
        """
        Create table of peptide + protein embeddings.
        
        Returns:
            DataFrame with columns: label, uniprot_id, embedding
        """
        logger.info("Creating peptide + protein merged table...")
        
        rows = []
        # Get unique peptide-protein pairs from mapping
        pep_prot_pairs = self.mapping.mapping.drop_duplicates(
            subset=['pep_seq', 'uniprot_id']
        )
        
        for _, row in tqdm(pep_prot_pairs.iterrows(), total=len(pep_prot_pairs), 
                          desc="Processing peptide-protein pairs"):
            pep_seq = row['pep_seq']
            prot_id = row['uniprot_id']
            
            pep_idx = self.pep_idx_map.get(pep_seq)
            prot_idx = self.prot_idx_map.get(prot_id)
            
            if pep_idx is None or prot_idx is None:
                continue
            
            pep_emb = self.pep_emb[pep_idx]
            prot_emb = self.prot_emb[prot_idx]
            
            # Concatenate embeddings
            merged_emb = EmbeddingAggregator.concatenate_embeddings([pep_emb, prot_emb])
            if merged_emb is None:
                continue
            
            rows.append({
                'label': pep_seq,
                'uniprot_id': prot_id,
                'embedding': merged_emb
            })
        
        # Convert to DataFrame
        if not rows:
            logger.warning("No valid rows created for pep+prot table")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Expand embedding columns
        embedding_df = pd.DataFrame(
            df['embedding'].tolist(),
            index=df.index,
            columns=[f'feat_{i}' for i in range(df['embedding'].iloc[0].shape[0])]
        )
        
        result = pd.concat(
            [df[['label', 'uniprot_id']], embedding_df],
            axis=1
        )
        
        logger.info(f"Created pep+prot table with {len(result)} rows and {result.shape[1]} columns")
        return result


# ============================================================================
# Main Processing Functions
# ============================================================================

def load_embeddings_and_mapping(
    cys_embeddings_path: str,
    pep_embeddings_path: str,
    prot_embeddings_path: str,
    mapping_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str], pd.DataFrame]:
    """
    Load all embeddings and mapping file.
    
    Args:
        cys_embeddings_path: Path to cysteine embeddings
        pep_embeddings_path: Path to peptide embeddings
        prot_embeddings_path: Path to protein embeddings
        mapping_path: Path to mapping CSV
        
    Returns:
        Tuple of (cys_emb, pep_emb, prot_emb, cys_ids, pep_ids, prot_ids, mapping_df)
    """
    logger.info("Loading embeddings and mapping...")
    
    # Load embeddings
    logger.info(f"Loading cysteine embeddings from {cys_embeddings_path}")
    cys_emb, cys_ids = EmbeddingLoader.load_with_metadata(cys_embeddings_path)
    logger.info(f"Loaded {len(cys_ids)} cysteine embeddings with shape {cys_emb[0].shape if len(cys_emb) > 0 else 0}")
    
    logger.info(f"Loading peptide embeddings from {pep_embeddings_path}")
    pep_emb, pep_ids = EmbeddingLoader.load_with_metadata(pep_embeddings_path)
    logger.info(f"Loaded {len(pep_ids)} peptide embeddings with shape {pep_emb[0].shape if len(pep_emb) > 0 else 0}")
    
    logger.info(f"Loading protein embeddings from {prot_embeddings_path}")
    prot_emb, prot_ids = EmbeddingLoader.load_with_metadata(prot_embeddings_path)
    logger.info(f"Loaded {len(prot_ids)} protein embeddings with shape {prot_emb[0].shape if len(prot_emb) > 0 else 0}")
    
    # Load mapping
    logger.info(f"Loading mapping from {mapping_path}")
    if mapping_path.endswith('.parquet'):
        mapping_df = pd.read_parquet(mapping_path)
    else:
        mapping_df = pd.read_csv(mapping_path)
    logger.info(f"Loaded mapping with {len(mapping_df)} rows")
    
    return cys_emb, pep_emb, prot_emb, list(cys_ids), list(pep_ids), list(prot_ids), mapping_df


def convert_embeddings_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert expanded embedding columns (feat_0, feat_1, ...) back to string format.
    
    Args:
        df: DataFrame with expanded embedding columns
        
    Returns:
        DataFrame with 'embedding' column containing string representation
    """
    # Find all feat_* columns
    feat_cols = [col for col in df.columns if col.startswith('feat_')]
    
    if not feat_cols:
        # No embedding columns to convert
        return df
    
    # Get non-feature columns
    non_feat_cols = [col for col in df.columns if not col.startswith('feat_')]
    
    # Convert feature columns back to string format
    embedding_strs = []
    for idx, row in df[feat_cols].iterrows():
        embedding_list = row[feat_cols].tolist()
        embedding_str = str(embedding_list)
        embedding_strs.append(embedding_str)
    
    # Create new DataFrame with embeddings as strings
    result_df = df[non_feat_cols].copy()
    result_df['embedding'] = embedding_strs
    
    return result_df


def save_merged_tables(
    tables: Dict[str, pd.DataFrame],
    output_dir: Path,
    format: str = 'parquet'
) -> None:
    """
    Save merged tables to disk.
    
    Args:
        tables: Dictionary of table name -> DataFrame
        output_dir: Output directory
        format: Output format ('parquet', 'csv', 'both')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving merged tables to {output_dir}")
    
    for table_name, df in tables.items():
        if df.empty:
            logger.warning(f"Skipping empty table: {table_name}")
            continue
        
        # Convert embeddings to string format before saving
        df_to_save = convert_embeddings_to_string(df)
        
        if format in ['parquet', 'both']:
            output_path = output_dir / f"{table_name}.parquet"
            logger.info(f"Saving {table_name} to {output_path}")
            df_to_save.to_parquet(output_path, index=False, compression='snappy')
        
        if format in ['csv', 'both']:
            output_path = output_dir / f"{table_name}.csv"
            logger.info(f"Saving {table_name} to {output_path}")
            df_to_save.to_csv(output_path, index=False)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--cysteine-embeddings',
        required=True,
        help='Path to cysteine embeddings (.npy, .npz, .parquet, or .csv)'
    )
    parser.add_argument(
        '--peptide-embeddings',
        required=True,
        help='Path to peptide embeddings (.npy, .npz, .parquet, or .csv)'
    )
    parser.add_argument(
        '--protein-embeddings',
        required=True,
        help='Path to protein embeddings (.npy, .npz, .parquet, or .csv)'
    )
    parser.add_argument(
        '--mapping',
        required=True,
        help='Path to mapping CSV/parquet with columns: uniprot_id, pep_seq, residue'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for merged tables'
    )
    parser.add_argument(
        '--aggregation-method',
        choices=['shortest', 'average'],
        default='shortest',
        help='Method for aggregating peptides: shortest peptide length or average embeddings'
    )
    parser.add_argument(
        '--output-format',
        choices=['parquet', 'csv', 'both'],
        default='parquet',
        help='Output file format'
    )
    parser.add_argument(
        '--tables',
        nargs='+',
        choices=['cys_pep', 'cys_pep_prot', 'cys_prot', 'pep_prot', 'all'],
        default=['all'],
        help='Which tables to create'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Update logger level
    global logger
    logger = setup_logging(args.log_level)
    
    try:
        # Determine which tables to create
        tables_to_create = args.tables
        if 'all' in tables_to_create:
            tables_to_create = ['cys_pep', 'cys_pep_prot', 'cys_prot', 'pep_prot']
        
        # Load embeddings and mapping
        cys_emb, pep_emb, prot_emb, cys_ids, pep_ids, prot_ids, mapping_df = load_embeddings_and_mapping(
            args.cysteine_embeddings,
            args.peptide_embeddings,
            args.protein_embeddings,
            args.mapping
        )
        
        # Create mapping handler
        mapping = MappingHandler(mapping_df)
        
        # Create merger
        merger = MergeTablesCreator(
            cys_emb, pep_emb, prot_emb,
            cys_ids, pep_ids, prot_ids,
            mapping,
            aggregation_method=args.aggregation_method
        )
        
        # Create tables
        tables = {}
        
        if 'cys_pep' in tables_to_create:
            tables['cys_pep_merged'] = merger.create_cys_pep_table()
        
        if 'cys_pep_prot' in tables_to_create:
            tables['cys_pep_prot_merged'] = merger.create_cys_pep_prot_table()
        
        if 'cys_prot' in tables_to_create:
            tables['cys_prot_merged'] = merger.create_cys_prot_table()
        
        if 'pep_prot' in tables_to_create:
            tables['pep_prot_merged'] = merger.create_pep_prot_table()
        
        # Save tables
        save_merged_tables(tables, args.output_dir, format=args.output_format)
        
        logger.info("✓ Successfully completed embedding merge!")
        
    except Exception as e:
        logger.error(f"✗ Error during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
