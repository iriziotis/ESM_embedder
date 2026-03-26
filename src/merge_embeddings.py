#!/usr/bin/env python3
"""
Merge and concatenate ESM-2 embeddings for cysteines, peptides, and proteins.

This script efficiently combines embeddings from different sequence types (cysteines,
peptides, proteins) into merged feature tables suitable for downstream analysis.
Supports batch processing and parallelization for large datasets.

Features:
- Efficient loading of embeddings (supports .npy, .npz, parquet, CSV)
- Seven types of merged tables: cys+pep, cys+pep+protein (all combos), cys+pep+protein
  (aggregated), cys+protein, pep+protein (all combos), pep+protein (aggregated), and
  peptide-only (aggregated, one row per cysteine)
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
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Optional, Union
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


EMBEDDING_DECIMALS = 3
PARQUET_COMPRESSION = 'zstd'
PARQUET_COMPRESSION_LEVEL = 9

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def optimize_parquet_table(table: object) -> object:
    """Cast Parquet feature columns to compact types before writing."""
    if pa is None:
        return table

    optimized_fields = []
    for field in table.schema:
        if field.name.startswith('feat_'):
            optimized_fields.append(pa.field(field.name, pa.float32()))
        else:
            optimized_fields.append(field)

    optimized_schema = pa.schema(optimized_fields)
    return table.cast(optimized_schema, safe=False)

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
    def _sorted_feature_columns(columns: List[str]) -> List[str]:
        """Return feature columns sorted by numeric suffix (feat_1, feat_2, ...)."""
        feat_cols = [col for col in columns if str(col).startswith('feat_')]
        if not feat_cols:
            return []

        def feat_key(col: str) -> Tuple[int, str]:
            suffix = str(col).split('_', 1)[1] if '_' in str(col) else ''
            return (int(suffix), str(col)) if suffix.isdigit() else (10**9, str(col))

        return sorted(feat_cols, key=feat_key)

    @staticmethod
    def _parse_embedding_series(series: pd.Series) -> np.ndarray:
        """Parse a Series of embeddings that may be strings or iterables."""
        import ast
        import json

        n = len(series)
        if n == 0:
            return np.empty((0, 0), dtype=np.float32)

        parsed_embeddings: List[np.ndarray] = []
        for emb in series:
            if isinstance(emb, str):
                # Prefer JSON for speed; fallback to literal_eval for legacy formatting.
                try:
                    parsed = json.loads(emb)
                except json.JSONDecodeError:
                    parsed = ast.literal_eval(emb)
            else:
                parsed = emb

            parsed_embeddings.append(np.asarray(parsed, dtype=np.float32))

        return np.vstack(parsed_embeddings).astype(np.float32, copy=False)

    @staticmethod
    def _extract_embeddings_from_df(df: pd.DataFrame) -> np.ndarray:
        """Extract embedding matrix from a dataframe across supported layouts.
        
        Optimized for large parquet files with string-encoded embeddings.
        Uses json.loads() which is ~10x faster than eval() or ast.literal_eval.
        """
        feat_cols = EmbeddingLoader._sorted_feature_columns(list(df.columns))
        if feat_cols:
            return df[feat_cols].to_numpy(dtype=np.float32, copy=False)

        if 'embedding' in df.columns:
            return EmbeddingLoader._parse_embedding_series(df['embedding'])

        # Fallback to all numeric columns if no explicit embedding layout is found.
        numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        return df[numeric_cols].to_numpy(dtype=np.float32, copy=False)
    
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
        """Load from parquet format (supports `embedding` column or numeric features)."""
        df = pd.read_parquet(file_path)
        embeddings = EmbeddingLoader._extract_embeddings_from_df(df)

        if indices is not None:
            embeddings = embeddings[indices]
        return embeddings
    
    @staticmethod
    def _load_csv(file_path: Path, indices: Optional[List[int]] = None) -> np.ndarray:
        """Load from CSV format (handles both numeric columns and string embeddings)."""
        df = pd.read_csv(file_path)

        embeddings = EmbeddingLoader._extract_embeddings_from_df(df)
        
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
            elif 'index' in df.columns:
                ids = df['index']
            elif 'label' in df.columns:
                ids = df['label']
            elif 'id' in df.columns:
                ids = df['id']
            else:
                ids = pd.Index(range(len(df)))

            embeddings = EmbeddingLoader._extract_embeddings_from_df(df)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)

            if key_column in df.columns:
                ids = df[key_column]
            elif 'index' in df.columns:
                ids = df['index']
            elif 'label' in df.columns:
                ids = df['label']
            elif 'id' in df.columns:
                ids = df['id']
            elif len(df.columns) > 0 and df.iloc[:, 0].dtype == 'object':
                ids = df.iloc[:, 0]
            else:
                ids = pd.Index(range(len(df)))

            embeddings = EmbeddingLoader._extract_embeddings_from_df(df)
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
        # Normalize peptide identifiers so mapping uses peptide sequence only,
        # regardless of whether input values contain a leading uniprot prefix.
        self.mapping['pep_seq'] = self.mapping['pep_seq'].astype(str).apply(
            MergeTablesCreator.normalize_peptide_identifier
        )

        # Create full cysteine ID (uniprot_id_residue) for each row
        self.mapping['cys_full_id'] = self.mapping['uniprot_id'] + '_' + self.mapping['residue']

        mapping_pairs = self.mapping[['cys_full_id', 'pep_seq', 'uniprot_id']].drop_duplicates()

        self.cys_to_peps = {
            cys_id: group['pep_seq'].tolist()
            for cys_id, group in mapping_pairs.groupby('cys_full_id', sort=False)
        }
        self.cys_to_shortest_pep = {
            cys_id: min(peptides, key=len)
            for cys_id, peptides in self.cys_to_peps.items()
            if peptides
        }
        self.cys_to_proteins = {
            cys_id: group['uniprot_id'].tolist()
            for cys_id, group in mapping_pairs.groupby('cys_full_id', sort=False)
        }
        self.cys_to_pep_prot = {
            cys_id: list(group[['pep_seq', 'uniprot_id']].itertuples(index=False, name=None))
            for cys_id, group in mapping_pairs.groupby('cys_full_id', sort=False)
        }
        self.cys_prot_pairs = list(
            mapping_pairs[['cys_full_id', 'uniprot_id']].drop_duplicates().itertuples(index=False, name=None)
        )
        self.pep_prot_pairs = list(
            mapping_pairs[['pep_seq', 'uniprot_id']].drop_duplicates().itertuples(index=False, name=None)
        )
    
    def get_peptides_for_cysteine(self, cys_id: str) -> List[str]:
        """Get all peptides containing a cysteine."""
        return self.cys_to_peps.get(cys_id, [])
    
    def get_proteins_for_cysteine(self, cys_id: str) -> List[str]:
        """Get all proteins containing a cysteine."""
        return self.cys_to_proteins.get(cys_id, [])
    
    def get_shortest_peptide(self, cys_id: str) -> Optional[str]:
        """Get shortest peptide containing cysteine."""
        return self.cys_to_shortest_pep.get(cys_id)


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

        # Build normalized peptide lookup so mapping pep_seq (sequence-only)
        # can match peptide embedding labels with or without uniprot prefix.
        self.pep_lookup = {}
        for idx, pid in enumerate(pep_ids):
            normalized_id = self.normalize_peptide_identifier(pid)
            if normalized_id not in self.pep_lookup:
                self.pep_lookup[normalized_id] = []
            self.pep_lookup[normalized_id].append(idx)
        
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

    @staticmethod
    def normalize_peptide_identifier(peptide_id: str) -> str:
        """Normalize peptide IDs to sequence-only values.

        Accepts either `PEPTIDESEQ` or `UNIPROT_PEPTIDESEQ` and returns
        the peptide sequence component used by mapping `pep_seq`.
        """
        peptide_id = str(peptide_id)
        if '_' in peptide_id:
            return peptide_id.split('_', 1)[1]
        return peptide_id

    @staticmethod
    def _extract_residue_from_cys_id(cys_id: str) -> str:
        """Extract residue token from cysteine id formatted as UNIPROT_RESIDUE."""
        cys_id = str(cys_id)
        if '_' in cys_id:
            return cys_id.split('_', 1)[1]
        return cys_id

    @staticmethod
    def _rows_to_dataframe(rows: Iterable[Tuple[str, np.ndarray]]) -> pd.DataFrame:
        """Materialize streamed rows into a dataframe for compatibility with the public API."""
        materialized_rows = list(rows)
        if not materialized_rows:
            return pd.DataFrame()

        labels = [label for label, _ in materialized_rows]
        embeddings = np.vstack([
            np.asarray(embedding, dtype=np.float32) for _, embedding in materialized_rows
        ])
        embedding_df = pd.DataFrame(
            embeddings,
            columns=[f'feat_{i + 1}' for i in range(embeddings.shape[1])],
        )
        embedding_df.insert(0, 'index', labels)
        return embedding_df

    def _collect_peptide_embeddings(self, peptide_ids: List[str]) -> Optional[np.ndarray]:
        """Resolve peptide IDs to embeddings and aggregate them when needed."""
        peptide_embeddings: List[np.ndarray] = []
        for peptide_id in peptide_ids:
            normalized_peptide = self.normalize_peptide_identifier(peptide_id)
            for pep_idx in self.pep_lookup.get(normalized_peptide, []):
                peptide_embeddings.append(self.pep_emb[pep_idx])

        if not peptide_embeddings:
            return None

        if len(peptide_embeddings) == 1:
            return peptide_embeddings[0]

        return EmbeddingAggregator.average_embeddings(
            np.asarray(peptide_embeddings, dtype=np.float32)
        )

    def iter_cys_pep_rows(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Yield cysteine + peptide merged rows one at a time."""
        logger.info("Creating cysteine + peptide merged table...")

        for cys_id in tqdm(self.cys_ids, desc="Processing cysteines"):
            cys_idx = self.cys_idx_map.get(cys_id)
            if cys_idx is None:
                continue

            if self.aggregation_method == 'shortest':
                shortest_peptide = self.mapping.get_shortest_peptide(cys_id)
                if shortest_peptide is None:
                    continue
                peptide_ids = [shortest_peptide]
            else:
                peptide_ids = self.mapping.get_peptides_for_cysteine(cys_id)
                if not peptide_ids:
                    continue

            pep_emb = self._collect_peptide_embeddings(peptide_ids)
            if pep_emb is None:
                continue

            merged_emb = EmbeddingAggregator.concatenate_embeddings([self.cys_emb[cys_idx], pep_emb])
            if merged_emb is None:
                continue

            first_peptide = self.normalize_peptide_identifier(peptide_ids[0])
            yield f"{cys_id}_{first_peptide}", merged_emb

    def iter_cys_pep_prot_rows(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Yield cysteine + peptide + protein merged rows one at a time."""
        logger.info("Creating cysteine + peptide + protein merged table...")

        for cys_id in tqdm(self.cys_ids, desc="Processing cysteines"):
            cys_idx = self.cys_idx_map.get(cys_id)
            if cys_idx is None:
                continue

            cys_emb = self.cys_emb[cys_idx]
            for pep_seq, prot_id in self.mapping.cys_to_pep_prot.get(cys_id, []):
                prot_idx = self.prot_idx_map.get(prot_id)
                pep_indices = self.pep_lookup.get(self.normalize_peptide_identifier(pep_seq), [])

                if prot_idx is None or not pep_indices:
                    continue

                prot_emb = self.prot_emb[prot_idx]
                for pep_idx in pep_indices:
                    merged_emb = EmbeddingAggregator.concatenate_embeddings(
                        [cys_emb, self.pep_emb[pep_idx], prot_emb]
                    )
                    if merged_emb is None:
                        continue

                    yield (
                        f"{prot_id}_{self._extract_residue_from_cys_id(cys_id)}_"
                        f"{self.normalize_peptide_identifier(pep_seq)}",
                        merged_emb,
                    )

    def iter_cys_pep_prot_agg_rows(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Yield cysteine + peptide (aggregated) + protein merged rows, one per cysteine.

        Uses aggregation_method to collapse multiple peptides to a single embedding
        per cysteine, producing exactly one row per cysteine (same cardinality as
        iter_cys_pep_rows but with the protein embedding appended).
        """
        logger.info("Creating cysteine + peptide (aggregated) + protein merged table...")

        for cys_id in tqdm(self.cys_ids, desc="Processing cysteines"):
            cys_idx = self.cys_idx_map.get(cys_id)
            if cys_idx is None:
                continue

            # Protein is uniquely encoded in cys_id (uniprot_id_residue)
            proteins = self.mapping.get_proteins_for_cysteine(cys_id)
            if not proteins:
                continue
            prot_idx = self.prot_idx_map.get(proteins[0])
            if prot_idx is None:
                continue

            if self.aggregation_method == 'shortest':
                shortest_peptide = self.mapping.get_shortest_peptide(cys_id)
                if shortest_peptide is None:
                    continue
                peptide_ids = [shortest_peptide]
            else:
                peptide_ids = self.mapping.get_peptides_for_cysteine(cys_id)
                if not peptide_ids:
                    continue

            pep_emb = self._collect_peptide_embeddings(peptide_ids)
            if pep_emb is None:
                continue

            merged_emb = EmbeddingAggregator.concatenate_embeddings(
                [self.cys_emb[cys_idx], pep_emb, self.prot_emb[prot_idx]]
            )
            if merged_emb is None:
                continue

            yield f"{proteins[0]}_{self._extract_residue_from_cys_id(cys_id)}", merged_emb

    def iter_cys_prot_rows(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Yield cysteine + protein merged rows one at a time."""
        logger.info("Creating cysteine + protein merged table...")

        for cys_id, prot_id in tqdm(
            self.mapping.cys_prot_pairs,
            total=len(self.mapping.cys_prot_pairs),
            desc="Processing cysteine-protein pairs",
        ):
            cys_idx = self.cys_idx_map.get(cys_id)
            prot_idx = self.prot_idx_map.get(prot_id)
            if cys_idx is None or prot_idx is None:
                continue

            merged_emb = EmbeddingAggregator.concatenate_embeddings(
                [self.cys_emb[cys_idx], self.prot_emb[prot_idx]]
            )
            if merged_emb is None:
                continue

            yield f"{prot_id}_{self._extract_residue_from_cys_id(cys_id)}", merged_emb

    def iter_pep_prot_rows(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Yield peptide + protein merged rows one at a time."""
        logger.info("Creating peptide + protein merged table...")

        for pep_seq, prot_id in tqdm(
            self.mapping.pep_prot_pairs,
            total=len(self.mapping.pep_prot_pairs),
            desc="Processing peptide-protein pairs",
        ):
            prot_idx = self.prot_idx_map.get(prot_id)
            pep_indices = self.pep_lookup.get(self.normalize_peptide_identifier(pep_seq), [])
            if prot_idx is None or not pep_indices:
                continue

            prot_emb = self.prot_emb[prot_idx]
            for pep_idx in pep_indices:
                merged_emb = EmbeddingAggregator.concatenate_embeddings(
                    [self.pep_emb[pep_idx], prot_emb]
                )
                if merged_emb is None:
                    continue

                yield f"{prot_id}_{self.normalize_peptide_identifier(pep_seq)}", merged_emb

    def iter_pep_prot_agg_rows(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Yield peptide (aggregated) + protein merged rows, one per cysteine.

        Uses aggregation_method to collapse multiple peptides to a single embedding
        per cysteine, then concatenates with the protein embedding.
        """
        logger.info("Creating peptide (aggregated) + protein merged table...")

        seen_cys_ids: set = set()
        for cys_id in tqdm(self.cys_ids, desc="Processing cysteines"):
            if cys_id in seen_cys_ids:
                continue
            seen_cys_ids.add(cys_id)

            proteins = self.mapping.get_proteins_for_cysteine(cys_id)
            if not proteins:
                continue
            prot_idx = self.prot_idx_map.get(proteins[0])
            if prot_idx is None:
                continue

            if self.aggregation_method == 'shortest':
                shortest_peptide = self.mapping.get_shortest_peptide(cys_id)
                if shortest_peptide is None:
                    continue
                peptide_ids = [shortest_peptide]
            else:
                peptide_ids = self.mapping.get_peptides_for_cysteine(cys_id)
                if not peptide_ids:
                    continue

            pep_emb = self._collect_peptide_embeddings(peptide_ids)
            if pep_emb is None:
                continue

            merged_emb = EmbeddingAggregator.concatenate_embeddings(
                [pep_emb, self.prot_emb[prot_idx]]
            )
            if merged_emb is None:
                continue

            yield f"{proteins[0]}_{self._extract_residue_from_cys_id(cys_id)}", merged_emb

    def iter_pep_agg_rows(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Yield peptide-only embeddings aggregated to one row per unique cysteine id."""
        logger.info("Creating peptide (aggregated) table...")

        seen_cys_ids = set()
        for cys_id in tqdm(self.cys_ids, desc="Processing unique residues"):
            # Defensive de-duplication in case cysteine embedding labels repeat.
            if cys_id in seen_cys_ids:
                continue
            seen_cys_ids.add(cys_id)

            if self.aggregation_method == 'shortest':
                shortest_peptide = self.mapping.get_shortest_peptide(cys_id)
                if shortest_peptide is None:
                    continue
                peptide_ids = [shortest_peptide]
            else:
                peptide_ids = self.mapping.get_peptides_for_cysteine(cys_id)
                if not peptide_ids:
                    continue

            pep_emb = self._collect_peptide_embeddings(peptide_ids)
            if pep_emb is None:
                continue

            yield str(cys_id), pep_emb
    
    def create_cys_pep_table(self) -> pd.DataFrame:
        """
        Create table of cysteine + peptide embeddings.
        
        Returns:
            DataFrame with columns: label, feat_1..feat_N
        """
        result = self._rows_to_dataframe(self.iter_cys_pep_rows())
        if result.empty:
            logger.warning("No valid rows created for cys+pep table")
        else:
            logger.info(f"Created cys+pep table with {len(result)} rows and {result.shape[1]} columns")
        return result
    
    def create_cys_pep_prot_table(self) -> pd.DataFrame:
        """
        Create table of cysteine + peptide + protein embeddings.
        
        Returns:
            DataFrame with columns: label, feat_1..feat_N
        """
        result = self._rows_to_dataframe(self.iter_cys_pep_prot_rows())
        if result.empty:
            logger.warning("No valid rows created for cys+pep+prot table")
        else:
            logger.info(f"Created cys+pep+prot table with {len(result)} rows and {result.shape[1]} columns")
        return result
    
    def create_cys_pep_prot_agg_table(self) -> pd.DataFrame:
        """
        Create table of cysteine + peptide (aggregated) + protein embeddings.

        One row per cysteine; peptides are collapsed using aggregation_method.

        Returns:
            DataFrame with columns: index, feat_1..feat_N
        """
        result = self._rows_to_dataframe(self.iter_cys_pep_prot_agg_rows())
        if result.empty:
            logger.warning("No valid rows created for cys+pep+prot (aggregated) table")
        else:
            logger.info(
                f"Created cys+pep+prot (aggregated) table with {len(result)} rows "
                f"and {result.shape[1]} columns"
            )
        return result

    def create_cys_prot_table(self) -> pd.DataFrame:
        """
        Create table of cysteine + protein embeddings.
        
        Returns:
            DataFrame with columns: label, feat_1..feat_N
        """
        result = self._rows_to_dataframe(self.iter_cys_prot_rows())
        if result.empty:
            logger.warning("No valid rows created for cys+prot table")
        else:
            logger.info(f"Created cys+prot table with {len(result)} rows and {result.shape[1]} columns")
        return result
    
    def create_pep_prot_table(self) -> pd.DataFrame:
        """
        Create table of peptide + protein embeddings.
        
        Returns:
            DataFrame with columns: label, feat_1..feat_N
        """
        result = self._rows_to_dataframe(self.iter_pep_prot_rows())
        if result.empty:
            logger.warning("No valid rows created for pep+prot table")
        else:
            logger.info(f"Created pep+prot table with {len(result)} rows and {result.shape[1]} columns")
        return result

    def create_pep_prot_agg_table(self) -> pd.DataFrame:
        """
        Create table of peptide (aggregated) + protein embeddings.

        One row per cysteine; peptides are collapsed using aggregation_method.

        Returns:
            DataFrame with columns: index, feat_1..feat_N
        """
        result = self._rows_to_dataframe(self.iter_pep_prot_agg_rows())
        if result.empty:
            logger.warning("No valid rows created for pep+prot (aggregated) table")
        else:
            logger.info(
                f"Created pep+prot (aggregated) table with {len(result)} rows "
                f"and {result.shape[1]} columns"
            )
        return result

    def create_pep_agg_table(self) -> pd.DataFrame:
        """
        Create table of peptide-only embeddings per unique residue id.

        Returns:
            DataFrame with columns: index (uniprot_id_residue), feat_1..feat_N
        """
        result = self._rows_to_dataframe(self.iter_pep_agg_rows())
        if result.empty:
            logger.warning("No valid rows created for peptide (aggregated) table")
        else:
            logger.info(
                "Created peptide (aggregated) table with "
                f"{len(result)} rows and {result.shape[1]} columns"
            )
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


def save_merged_table(
    table_name: str,
    rows: Iterable[Tuple[str, np.ndarray]],
    output_dir: Path,
    format: str = 'parquet',
    chunk_size: int = 1024,
) -> int:
    """Save a streamed merged table to disk in bounded memory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row_iterator = iter(rows)
    first_row = next(row_iterator, None)
    if first_row is None:
        logger.warning(f"Skipping empty table: {table_name}")
        return 0

    if format in ['parquet', 'both'] and (pa is None or pq is None):
        raise ImportError(
            "pyarrow is required for parquet output. Install it with `pip install pyarrow`."
        )

    _, first_embedding = first_row
    feature_count = int(np.asarray(first_embedding).shape[0])
    feature_columns = [f'feat_{i + 1}' for i in range(feature_count)]
    csv_path = output_dir / f"{table_name}.csv"
    parquet_path = output_dir / f"{table_name}.parquet"
    csv_header_written = False
    parquet_writer = None
    buffered_labels: List[str] = []
    buffered_embeddings: List[np.ndarray] = []
    written_rows = 0

    def flush_buffer() -> None:
        nonlocal csv_header_written, parquet_writer
        if not buffered_labels:
            return

        embedding_matrix = np.round(
            np.vstack(buffered_embeddings).astype(np.float32, copy=False),
            EMBEDDING_DECIMALS,
        )
        chunk_df = pd.DataFrame(embedding_matrix, columns=feature_columns)
        chunk_df.insert(0, 'index', buffered_labels)

        if format in ['csv', 'both']:
            chunk_df.to_csv(
                csv_path,
                mode='a' if csv_header_written else 'w',
                header=not csv_header_written,
                index=False,
            )
            csv_header_written = True

        if format in ['parquet', 'both']:
            table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            table = optimize_parquet_table(table)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                    parquet_path,
                    table.schema,
                    compression=PARQUET_COMPRESSION,
                    compression_level=PARQUET_COMPRESSION_LEVEL,
                )
            parquet_writer.write_table(table)

        buffered_labels.clear()
        buffered_embeddings.clear()

    try:
        for label, embedding in chain([first_row], row_iterator):
            buffered_labels.append(str(label))
            buffered_embeddings.append(np.asarray(embedding, dtype=np.float32))
            written_rows += 1

            if len(buffered_labels) >= chunk_size:
                flush_buffer()

        flush_buffer()
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    logger.info(
        f"Saved {table_name} with {written_rows} rows and {feature_count + 1} columns to {output_dir}"
    )
    return written_rows


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
        choices=['cys_pep', 'cys_pep_prot', 'cys_pep_prot_agg', 'cys_prot', 'pep_prot', 'pep_prot_agg', 'pep_agg', 'all'],
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
            tables_to_create = [
                'cys_pep',
                'cys_pep_prot',
                'cys_pep_prot_agg',
                'cys_prot',
                'pep_prot',
                'pep_prot_agg',
                'pep_agg',
            ]
        
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
        
        table_builders = {
            'cys_pep': ('cys_pep_merged', merger.iter_cys_pep_rows),
            'cys_pep_prot': ('cys_pep_prot_merged', merger.iter_cys_pep_prot_rows),
            'cys_pep_prot_agg': ('cys_pep_prot_agg_merged', merger.iter_cys_pep_prot_agg_rows),
            'cys_prot': ('cys_prot_merged', merger.iter_cys_prot_rows),
            'pep_prot': ('pep_prot_merged', merger.iter_pep_prot_rows),
            'pep_prot_agg': ('pep_prot_agg_merged', merger.iter_pep_prot_agg_rows),
            'pep_agg': (
                'pep_agg_merged',
                merger.iter_pep_agg_rows,
            ),
        }

        for table_key in tables_to_create:
            table_name, row_builder = table_builders[table_key]
            save_merged_table(
                table_name,
                row_builder(),
                args.output_dir,
                format=args.output_format,
            )
        
        logger.info("✓ Successfully completed embedding merge!")
        
    except Exception as e:
        logger.error(f"✗ Error during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
