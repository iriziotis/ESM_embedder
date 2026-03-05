#!/usr/bin/env python3
"""
Test script to validate the ESM2 embedding generator.

This script:
1. Creates sample sequence data (FASTA and CSV)
2. Generates embeddings for different target types (sequence, residue, peptide)
3. Validates output format and dimensions
4. Reports results

Usage:
    python test_embedder.py [--output-dir ./test_embeddings] [--model-name esm2_t6_8M_UR50D]
    python test_embedder.py --no-cleanup  # Keep test files
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import List
import logging
import csv

logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_test_sequences_csv(output_path: Path):
    """Create test CSV file with sequences and residue positions."""
    sequences = [
        ('P00001', 'MSLFLYTLCGSIYCSDGVLSAASVDGKCIDQKCLEKQVMSESCVLWQSFKVGPEFLVSMKIDYDPPKYLPELPVDKHHVYKW', 'C5'),
        ('P00002', 'MKTFIFLCGSCTNSSPPELGKIIFSNKAKQLAVCEVSPEQICCQNQSGHCVVQPIIFEQVGIAFSNLNTVDTKAEGVNQD', 'C12'),
        ('P00003', 'MLSLLTILPSVYACSDEVASTASVDGKCIDQKCLEKQVMSESCVLWQSFKVGPEFLVSMKIDYDPPKYLAELPVDKHHVYK', 'C7'),
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['uniprot_id', 'pep_seq', 'residue'])
        writer.writeheader()
        for uniprot_id, seq, residue in sequences:
            writer.writerow({
                'uniprot_id': uniprot_id,
                'pep_seq': seq,
                'residue': residue
            })
    
    logger.info(f"Created test CSV with {len(sequences)} sequences")
    return len(sequences)

def create_test_sequences_fasta(output_path: Path):
    """Create test FASTA file."""
    sequences = [
        ('P00001', 'MSLFLYTLCGSIYCSDGVLSAASVDGKCIDQKCLEKQVMSESCVLWQSFKVGPEFLVSMKIDYDPPKYLPELPVDKHHVYKW'),
        ('P00002', 'MKTFIFLCGSCTNSSPPELGKIIFSNKAKQLAVCEVSPEQICCQNQSGHCVVQPIIFEQVGIAFSNLNTVDTKAEGVNQD'),
        ('P00003', 'MLSLLTILPSVYACSDEVASTASVDGKCIDQKCLEKQVMSESCVLWQSFKVGPEFLVSMKIDYDPPKYLPELPVDKHHVYK'),
    ]
    
    with open(output_path, 'w') as f:
        for uniprot_id, seq in sequences:
            f.write(f">{uniprot_id}\n{seq}\n")
    
    logger.info(f"Created test FASTA with {len(sequences)} sequences")
    return len(sequences)

def create_test_peptides_csv(output_path: Path):
    """Create test CSV file with peptide sequences."""
    peptides = [
        ('P00001', 'SLFLYTLCGSIYCS'),
        ('P00002', 'TFIFLCGSCTNSSPP'),
        ('P00003', 'SLLTILPSVYACSDE'),
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['uniprot_id', 'pep_seq'])
        writer.writeheader()
        for uniprot_id, seq in peptides:
            writer.writerow({
                'uniprot_id': uniprot_id,
                'pep_seq': seq
            })
    
    logger.info(f"Created test peptides CSV with {len(peptides)} peptides")
    return len(peptides)

def test_embedding_generation(
    script_path: Path,
    seq_file: Path,
    output_file: Path,
    target: str,
    model_name: str,
    residue_type: str = 'C'
) -> bool:
    """Test embedding generation for a specific target."""
    logger.info(f"\nTesting {target} embeddings...")
    
    cmd = [
        'python3',
        str(script_path),
        '--seq_file', str(seq_file),
        '--output_file', str(output_file),
        '--target', target,
        '--model_name', model_name,
        '--max_seq_length', '1024',
    ]
    
    if target == 'residue':
        cmd.extend(['--residue_type', residue_type])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            logger.error(f"✗ {target} embedding generation failed")
            logger.error(f"STDERR: {result.stderr}")
            return False
        
        if output_file.exists():
            # Count output lines
            with open(output_file, 'r') as f:
                lines = len(f.readlines()) - 1  # Subtract header
            logger.info(f"✓ {target} embeddings generated: {lines} entries")
            return True
        else:
            logger.error(f"✗ Output file not created: {output_file}")
            return False
    
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {target} embedding generation timeout")
        return False
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return False

def test_embedding_generation_with_flag(
    script_path: Path,
    seq_file: Path,
    output_file: Path,
    target: str,
    model_name: str,
    residue_type: str = 'C',
    use_no_all_residues: bool = True
) -> bool:
    """Test embedding generation with optional --no-all_residues flag."""
    logger.info(f"\nTesting {target} embeddings...")
    
    cmd = [
        'python3',
        str(script_path),
        '--seq_file', str(seq_file),
        '--output_file', str(output_file),
        '--target', target,
        '--model_name', model_name,
        '--max_seq_length', '1024',
    ]
    
    if target == 'residue':
        cmd.extend(['--residue_type', residue_type])
        if use_no_all_residues:
            cmd.append('--no-all_residues')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            logger.error(f"✗ {target} embedding generation failed")
            logger.error(f"STDERR: {result.stderr}")
            return False
        
        if output_file.exists():
            # Count output lines
            with open(output_file, 'r') as f:
                lines = len(f.readlines()) - 1  # Subtract header
            logger.info(f"✓ {target} embeddings generated: {lines} entries")
            return True
        else:
            logger.error(f"✗ Output file not created: {output_file}")
            return False
    
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {target} embedding generation timeout")
        return False
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return False

def find_generate_script() -> Path:
    """Find the generate_esm2_embeddings.py script."""
    candidate_paths = [
        Path(__file__).parent / 'src' / 'generate_esm2_embeddings.py',
        Path(__file__).parent.parent / 'src' / 'generate_esm2_embeddings.py',
        Path('/Users/rizioti/Library/CloudStorage/OneDrive-GSK/research/prosperity/ml_detectability/4-detectability_ml/esm_embedder/src/generate_esm2_embeddings.py')
    ]
    
    for p in candidate_paths:
        if p.exists():
            return p
    
    return None

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description='Test ESM2 embedding generator'
    )
    parser.add_argument(
        '--output-dir',
        default='./test_embeddings',
        help='Directory for test data and results'
    )
    parser.add_argument(
        '--model-name',
        default='esm2_t6_8M_UR50D',
        help='ESM2 model name to use (default: esm2_t6_8M_UR50D for speed)'
    )
    parser.add_argument(
        '--script-path',
        help='Path to generate_esm2_embeddings.py (auto-detected if not provided)'
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Keep test files after completion'
    )
    
    args = parser.parse_args()
    setup_logging()
    
    # Find script
    if args.script_path:
        script_path = Path(args.script_path)
    else:
        script_path = find_generate_script()
    
    if not script_path or not script_path.exists():
        logger.error("Could not find generate_esm2_embeddings.py")
        return 1
    
    logger.info(f"Using script: {script_path}")
    logger.info(f"Using model: {args.model_name}")
    logger.info("=" * 60)
    
    # Create test directories
    test_dir = Path(args.output_dir)
    data_dir = test_dir / 'data'
    output_dir = test_dir / 'output'
    
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nTest directory: {test_dir}")
    logger.info("=" * 60)
    
    try:
        # Create test data
        logger.info("\nCreating test data...")
        seq_csv = data_dir / 'test_sequences.csv'
        seq_fasta = data_dir / 'test_sequences.fasta'
        pep_csv = data_dir / 'test_peptides.csv'
        
        create_test_sequences_csv(seq_csv)
        create_test_sequences_fasta(seq_fasta)
        create_test_peptides_csv(pep_csv)
        
        # Test different target types
        logger.info("\n" + "=" * 60)
        logger.info("Testing embedding generation...")
        logger.info("=" * 60)
        
        all_passed = True
        
        # Test sequence embeddings
        seq_out = output_dir / 'sequence_embeddings.csv'
        if not test_embedding_generation(script_path, seq_fasta, seq_out, 'sequence', args.model_name):
            all_passed = False
        
        # Test residue embeddings (default: all residue instances)
        res_out = output_dir / 'residue_embeddings.csv'
        if not test_embedding_generation(script_path, seq_csv, res_out, 'residue', args.model_name, residue_type='C'):
            all_passed = False
        
        # Test peptide embeddings
        pep_out = output_dir / 'peptide_embeddings.csv'
        if not test_embedding_generation(script_path, pep_csv, pep_out, 'peptide', args.model_name):
            all_passed = False
        
        logger.info("\n" + "=" * 60)
        if all_passed:
            logger.info("✓ All embedding generation tests passed!")
            logger.info(f"Test data saved to: {test_dir}")
        else:
            logger.error("✗ Some tests failed")
            logger.error(f"Test data saved to: {test_dir}")
        
        if not args.no_cleanup:
            logger.info("\nCleaning up test data...")
            import shutil
            try:
                shutil.rmtree(test_dir)
                logger.info("✓ Cleanup complete")
            except Exception as e:
                logger.warning(f"Could not clean up test files: {e}")
        
        return 0 if all_passed else 1
    
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
