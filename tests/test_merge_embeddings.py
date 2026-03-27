#!/usr/bin/env python3
"""
Test script to validate the merge_embeddings setup and generate example data.

This script:
1. Creates sample embeddings and mapping files
2. Runs a small example merge
3. Validates output files
4. Reports any issues

Usage:
    python test_merge_embeddings.py [--output-dir ./test_data]
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_test_embeddings(
    output_dir: Path,
    n_cys: int = 50,
    n_pep: int = 100,
    n_prot: int = 20,
    embedding_dim: int = 1280
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Create test embedding files with proper IDs."""
    logger.info(f"Creating test embeddings (dim={embedding_dim})...")
    
    # Create random embeddings rounded to 3 decimals
    cys_emb = np.round(np.random.randn(n_cys, embedding_dim).astype(np.float32), 3)
    pep_emb = np.round(np.random.randn(n_pep, embedding_dim).astype(np.float32), 3)
    prot_emb = np.round(np.random.randn(n_prot, embedding_dim).astype(np.float32), 3)
    
    # Create IDs
    cys_residues = [f"C{i}" for i in range(1, n_cys + 1)]
    pep_seq_only = [f"ACDEFTGK{i}" for i in range(n_pep)]
    prot_ids = [f"P{str(i).zfill(5)}" for i in range(1, n_prot + 1)]
    
    # For cysteines, we'll create labels with uniprot_id_residue format
    # We need to return the residues separately for mapping creation
    # For now, use placeholder protein - we'll update labels after creating mapping
    cys_ids_temp = cys_residues  # Will be updated with uniprot_id prefix
    
    # Create peptide labels with uniprot_id_PEPTIDESEQ format
    # Each peptide is paired with a protein in a cyclic manner
    pep_labels = [f"{prot_ids[i % len(prot_ids)]}_{seq}" 
                  for i, seq in enumerate(pep_seq_only)]
    
    def embeddings_to_feature_df(labels: List[str], embeddings: np.ndarray) -> pd.DataFrame:
        """Convert embedding matrix to label + feat_1..feat_N columns."""
        feat_data = {f'feat_{i + 1}': embeddings[:, i] for i in range(embeddings.shape[1])}
        return pd.DataFrame({'label': labels, **feat_data})

    # Save embeddings as CSV with expanded feature columns.
    cys_df = embeddings_to_feature_df(cys_ids_temp, cys_emb)
    cys_df.to_csv(output_dir / 'cysteine_embeddings.csv', index=False)
    
    pep_df = embeddings_to_feature_df(pep_labels, pep_emb)
    pep_df.to_csv(output_dir / 'peptide_embeddings.csv', index=False)
    
    prot_df = embeddings_to_feature_df(prot_ids, prot_emb)
    prot_df.to_csv(output_dir / 'protein_embeddings.csv', index=False)
    
    logger.info(f"  ✓ Cysteine embeddings: {cys_emb.shape}")
    logger.info(f"  ✓ Peptide embeddings: {pep_emb.shape}")
    logger.info(f"  ✓ Protein embeddings: {prot_emb.shape}")
    
    return cys_residues, pep_labels, prot_ids, cys_ids_temp

def create_test_mapping(
    output_dir: Path,
    cys_residues: List[str],
    pep_ids: List[str],
    prot_ids: List[str]
) -> Dict[str, str]:
    """Create test mapping file and return cysteine label mapping."""
    logger.info("Creating test mapping file...")
    
    # Create mapping relationships
    rows = []
    cys_label_map = {}  # Maps residue -> uniprot_id_residue
    
    # Use provided IDs; pep_ids are labels like P00001_ACDEFTGK0.
    # Mapping pep_seq must contain the raw sequence only (ACDEFTGK0).
    cysteines = cys_residues
    peptides = pep_ids
    proteins = prot_ids
    
    n_pep = len(peptides)
    
    # Create mappings (each cysteine -> 1-3 peptides, each peptide -> 1-2 proteins)
    for cys in cysteines:
        # Randomly select 1-3 peptides for this cysteine
        n_peps_for_cys = np.random.randint(1, min(4, n_pep))
        selected_peptides = np.random.choice(peptides, size=n_peps_for_cys, replace=False)
        
        for pep in selected_peptides:
            pep_seq = pep.split('_', 1)[1] if '_' in pep else pep

            # Randomly select 1-2 proteins for this peptide
            n_prots_for_pep = np.random.randint(1, 3)
            selected_proteins = np.random.choice(proteins, size=n_prots_for_pep, replace=False)
            
            for prot in selected_proteins:
                rows.append({
                    'uniprot_id': prot,
                    'pep_seq': pep_seq,
                    'residue': cys
                })
                # Track cysteine label format for first occurrence
                if cys not in cys_label_map:
                    cys_label_map[cys] = f"{prot}_{cys}"
    
    # Create DataFrame and save
    mapping_df = pd.DataFrame(rows)
    mapping_df.to_csv(output_dir / 'mapping.csv', index=False)
    
    logger.info(f"  ✓ Mapping file: {len(mapping_df)} rows")
    logger.info(f"  ✓ Unique cysteines: {mapping_df['residue'].nunique()}")
    logger.info(f"  ✓ Unique peptides: {mapping_df['pep_seq'].nunique()}")
    logger.info(f"  ✓ Unique proteins: {mapping_df['uniprot_id'].nunique()}")
    
    return cys_label_map

def run_merge_test(
    script_path: Path,
    data_dir: Path,
    output_dir: Path,
    merge_strategy: str,
) -> bool:
    """Run the merge_embeddings script on test data."""
    logger.info(f"\nRunning merge_embeddings on test data (merge_strategy={merge_strategy})...")
    
    cmd = [
        'python3',
        str(script_path),
        '--cysteine-embeddings', str(data_dir / 'cysteine_embeddings.csv'),
        '--peptide-embeddings', str(data_dir / 'peptide_embeddings.csv'),
        '--protein-embeddings', str(data_dir / 'protein_embeddings.csv'),
        '--mapping', str(data_dir / 'mapping.csv'),
        '--output-dir', str(output_dir),
        '--aggregation-method', 'shortest',
        '--merge-strategy', merge_strategy,
        '--output-format', 'csv',
        '--tables', 'all',
        '--log-level', 'INFO'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            logger.error("Script failed!")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
        
        logger.info("✓ Script completed successfully")
        return True
    
    except subprocess.TimeoutExpired:
        logger.error("✗ Script timeout")
        return False
    except Exception as e:
        logger.error(f"✗ Error running script: {e}")
        return False

def validate_output(output_dir: Path, merge_strategy: str) -> bool:
    """Validate output files."""
    logger.info(f"\nValidating output files (merge_strategy={merge_strategy})...")
    
    expected_files = [
        f'cys_level-cys_only-{merge_strategy}.csv',
        f'cys_level-cys_pep-{merge_strategy}.csv',
        f'cys_level-cys_pep_prot_agg-{merge_strategy}.csv',
        f'cys_level-cys_prot-{merge_strategy}.csv',
        f'cys_level-pep_agg-{merge_strategy}.csv',
        f'cys_level-pep_prot_agg-{merge_strategy}.csv',
        f'peptide_level-cys_pep_prot-{merge_strategy}.csv',
        f'peptide_level-pep_prot-{merge_strategy}.csv'
    ]
    
    all_exist = True
    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            # Load and check contents
            try:
                df = pd.read_csv(filepath, index_col=0)
                logger.info(f"  ✓ {filename}: {len(df)} rows, {df.shape[1]} columns")
            except Exception as e:
                logger.error(f"  ✗ {filename}: Error reading file - {e}")
                all_exist = False
        else:
            logger.error(f"  ✗ {filename}: File not found")
            all_exist = False
    
    return all_exist

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description='Test merge_embeddings installation and functionality'
    )
    parser.add_argument(
        '--output-dir',
        default='./test_merge_embeddings',
        help='Directory for test data and results'
    )
    parser.add_argument(
        '--script-path',
        help='Path to merge_embeddings.py (auto-detected if not provided)'
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Keep test files after completion'
    )
    
    args = parser.parse_args()
    setup_logging()
    
    # Detect script path
    if args.script_path:
        script_path = Path(args.script_path)
    else:
        # Try to find in same directory or parent
        candidate_paths = [
            Path(__file__).parent / 'merge_embeddings.py',
            Path(__file__).parent.parent / 'src' / 'merge_embeddings.py',
            Path('/Users/rizioti/Library/CloudStorage/OneDrive-GSK/research/prosperity/ml_detectability/4-detectability_ml/esm_embedder/src/merge_embeddings.py')
        ]
        
        script_path = None
        for p in candidate_paths:
            if p.exists():
                script_path = p
                break
        
        if not script_path or not script_path.exists():
            logger.error(f"Could not find merge_embeddings.py")
            logger.error(f"Searched: {[str(p) for p in candidate_paths]}")
            sys.exit(1)
    
    logger.info(f"Using script: {script_path}")
    
    # Create test directories
    test_dir = Path(args.output_dir)
    data_dir = test_dir / 'data'
    output_dir = test_dir / 'output'
    
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nTest directory: {test_dir}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Create test data
        cys_residues, pep_ids, prot_ids, cys_ids_temp = create_test_embeddings(data_dir)
        cys_label_map = create_test_mapping(data_dir, cys_residues, pep_ids, prot_ids)
        
        # Step 1.5: Update cysteine embeddings with proper labels (uniprot_id_residue)
        cys_emb_path = data_dir / 'cysteine_embeddings.csv'
        cys_df = pd.read_csv(cys_emb_path)
        cys_df['label'] = cys_df['label'].map(cys_label_map)
        cys_df.to_csv(cys_emb_path, index=False)
        
        # Step 2 + 3: Run merge and validate for both merge strategies.
        for merge_strategy in ['concat', 'average']:
            if not run_merge_test(script_path, data_dir, output_dir, merge_strategy):
                logger.error(f"\n✗ Test failed at merge step (merge_strategy={merge_strategy})")
                sys.exit(1)

            if not validate_output(output_dir, merge_strategy):
                logger.error(f"\n✗ Test failed at validation step (merge_strategy={merge_strategy})")
                sys.exit(1)
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ All tests passed!")
        logger.info(f"Test data saved to: {test_dir}")
        
        if not args.no_cleanup:
            logger.info("\nCleaning up test data...")
            import shutil
            try:
                shutil.rmtree(test_dir)
                logger.info("✓ Cleanup complete")
            except Exception as e:
                logger.warning(f"Could not clean up test files: {e}")
        
        return 0
    
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
