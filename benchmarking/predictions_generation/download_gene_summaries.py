#!/usr/bin/env python3
"""
Download gene summaries from NCBI.

This script provides two approaches:
1. Bulk download from FTP (gene_info.gz) - Contains Symbol, Full Name, but NOT Summary
2. Entrez API - Contains full Summary text, but requires batched requests

For the full Summary field, we must use the Entrez API since it's not in the FTP bulk files.
"""

import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import os
import gzip
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET
from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# NCBI FTP URLs
GENE_INFO_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz"
GENE_INFO_HUMAN_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"

# Entrez API base URL
ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def download_gene_info(output_dir: str, human_only: bool = True) -> pd.DataFrame:
    """
    Download gene_info from NCBI FTP.
    
    This contains: GeneID, Symbol, Full_name, description, etc.
    But does NOT contain the full Summary text.
    
    Args:
        output_dir: Directory to save downloaded files
        human_only: If True, download only human genes (much smaller file)
    
    Returns:
        DataFrame with gene information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = GENE_INFO_HUMAN_URL if human_only else GENE_INFO_URL
    filename = "Homo_sapiens.gene_info.gz" if human_only else "gene_info.gz"
    filepath = output_dir / filename
    
    print(f"Downloading gene_info from {url}...")
    if not filepath.exists():
        urlretrieve(url, filepath)
        print(f"Downloaded to {filepath}")
    else:
        print(f"File already exists: {filepath}")
    
    # Parse the gene_info file
    print("Parsing gene_info file...")
    df = pd.read_csv(
        filepath,
        sep='\t',
        compression='gzip',
        dtype=str,
        na_values=['-', '']
    )
    
    # Rename columns for clarity
    # Columns: #tax_id, GeneID, Symbol, LocusTag, Synonyms, dbXrefs, chromosome,
    # map_location, description, type_of_gene, Symbol_from_nomenclature_authority,
    # Full_name_from_nomenclature_authority, Nomenclature_status, Other_designations,
    # Modification_date, Feature_type
    
    print(f"Loaded {len(df)} genes")
    return df


def fetch_gene_summaries_batch(
    gene_ids: list,
    output_file: str,
    batch_size: int = 200,
    delay: float = 0.34,
    api_key: Optional[str] = None,
    resume: bool = True
) -> dict:
    """
    Fetch gene summaries using NCBI Entrez API with incremental saving.
    
    The Summary field is only available via the API, not in bulk FTP files.
    Saves progress incrementally to allow resuming if interrupted.
    
    Args:
        gene_ids: List of NCBI Gene IDs
        output_file: Path to save results incrementally
        batch_size: Number of genes per request (max 500 without API key)
        delay: Delay between requests (0.34s without API key, can be lower with key)
        api_key: NCBI API key for higher rate limits
        resume: If True, skip gene IDs already in output file
    
    Returns:
        Dictionary mapping gene_id -> {symbol, full_name, summary}
    """
    results = {}
    
    # Load existing results if resuming
    output_path = Path(output_file)
    if resume and output_path.exists():
        print(f"Resuming from existing file: {output_file}")
        existing_df = pd.read_csv(output_file, sep='\t', dtype=str)
        for _, row in existing_df.iterrows():
            results[str(row['gene_id'])] = {
                "symbol": row.get('symbol'),
                "full_name": row.get('full_name'),
                "summary": row.get('summary')
            }
        print(f"Loaded {len(results)} existing genes, skipping those...")
        gene_ids = [gid for gid in gene_ids if str(gid) not in results]
    
    if not gene_ids:
        print("All genes already fetched!")
        return results
    
    # Adjust rate limit based on API key
    if api_key:
        delay = max(delay, 0.1)  # 10 requests/second with API key
        batch_size = min(batch_size, 500)
    else:
        delay = max(delay, 0.34)  # 3 requests/second without API key
        batch_size = min(batch_size, 200)
    
    total_batches = (len(gene_ids) + batch_size - 1) // batch_size
    
    print(f"Fetching summaries for {len(gene_ids)} genes in {total_batches} batches...")
    print(f"Progress will be saved to: {output_file}")
    
    # Prepare for incremental writing
    write_header = not output_path.exists()
    
    batch_num = 0
    for i in range(0, len(gene_ids), batch_size):
        batch_num += 1
        batch = gene_ids[i:i + batch_size]
        batch_str = ",".join(str(gid) for gid in batch)
        
        params = {
            "db": "gene",
            "id": batch_str,
            "rettype": "xml",
            "retmode": "xml"
        }
        if api_key:
            params["api_key"] = api_key
        
        batch_results = []
        retries = 3
        
        for attempt in range(retries):
            try:
                print(f"Batch {batch_num}/{total_batches}: fetching {len(batch)} genes...", end=" ", flush=True)
                response = requests.get(f"{ENTREZ_BASE}/efetch.fcgi", params=params, timeout=30)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                
                for entrezgene in root.findall(".//Entrezgene"):
                    gene_id = None
                    symbol = None
                    full_name = None
                    summary = None
                    
                    # Get Gene ID
                    gene_track = entrezgene.find(".//Entrezgene_track-info/Gene-track/Gene-track_geneid")
                    if gene_track is not None:
                        gene_id = gene_track.text
                    
                    # Get Symbol and Full Name from Gene-ref
                    gene_ref = entrezgene.find(".//Entrezgene_gene/Gene-ref")
                    if gene_ref is not None:
                        symbol_elem = gene_ref.find("Gene-ref_locus")
                        if symbol_elem is not None:
                            symbol = symbol_elem.text
                        
                        desc_elem = gene_ref.find("Gene-ref_desc")
                        if desc_elem is not None:
                            full_name = desc_elem.text
                    
                    # Get Summary
                    summary_elem = entrezgene.find(".//Entrezgene_summary")
                    if summary_elem is not None:
                        summary = summary_elem.text
                    
                    if gene_id:
                        results[gene_id] = {
                            "symbol": symbol,
                            "full_name": full_name,
                            "summary": summary
                        }
                        batch_results.append({
                            "gene_id": gene_id,
                            "symbol": symbol or "",
                            "full_name": full_name or "",
                            "summary": summary or ""
                        })
                
                print(f"got {len(batch_results)} genes")
                break  # Success, exit retry loop
                
            except requests.exceptions.Timeout:
                print(f"TIMEOUT (attempt {attempt+1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(5)  # Wait before retry
                continue
            except Exception as e:
                print(f"ERROR: {e} (attempt {attempt+1}/{retries})")
                if attempt < retries - 1:
                    time.sleep(5)
                continue
        
        # Save batch results incrementally
        if batch_results:
            batch_df = pd.DataFrame(batch_results)
            batch_df.to_csv(output_file, sep='\t', index=False, mode='a', header=write_header)
            write_header = False  # Only write header once
        
        # Rate limiting
        time.sleep(delay)
    
    print(f"\nDone! Total genes fetched: {len(results)}")
    return results


def download_all_gene_summaries(
    output_dir: str,
    human_only: bool = True,
    gene_types: Optional[list] = None,
    api_key: Optional[str] = None,
    max_genes: Optional[int] = None,
    resume: bool = True
) -> pd.DataFrame:
    """
    Download all gene summaries by combining FTP data with Entrez API.
    
    Saves progress incrementally, so you can resume if interrupted.
    
    Args:
        output_dir: Directory to save output files
        human_only: If True, only download human genes
        gene_types: Filter by gene type (e.g., ['protein-coding'])
        api_key: NCBI API key for higher rate limits
        max_genes: Maximum number of genes to fetch (for testing)
        resume: If True, resume from existing progress file
    
    Returns:
        DataFrame with gene summaries
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "gene_summaries.tsv"
    
    # Step 1: Download gene_info for basic info and gene IDs
    gene_info = download_gene_info(output_dir, human_only=human_only)
    
    # Filter by gene type if specified
    if gene_types:
        gene_info = gene_info[gene_info['type_of_gene'].isin(gene_types)]
        print(f"Filtered to {len(gene_info)} genes of types: {gene_types}")
    
    # Get gene IDs
    gene_ids = gene_info['GeneID'].tolist()
    
    if max_genes:
        gene_ids = gene_ids[:max_genes]
        print(f"Limited to {max_genes} genes for testing")
    
    # Step 2: Fetch summaries via Entrez API (saves incrementally)
    summaries = fetch_gene_summaries_batch(
        gene_ids, 
        output_file=str(output_file),
        api_key=api_key,
        resume=resume
    )
    
    # Step 3: Load final results
    print("Loading final results...")
    if output_file.exists():
        df = pd.read_csv(output_file, sep='\t', dtype=str)
        print(f"Final file has {len(df)} gene summaries: {output_file}")
        print(f"Genes with summaries: {df['summary'].notna().sum()}")
    else:
        df = pd.DataFrame()
        print("No results file created")
    
    return df


def fetch_single_gene(gene_id: int, api_key: Optional[str] = None) -> dict:
    """
    Fetch a single gene's information.
    
    Args:
        gene_id: NCBI Gene ID
        api_key: Optional NCBI API key
    
    Returns:
        Dictionary with gene information
    """
    params = {
        "db": "gene",
        "id": str(gene_id),
        "rettype": "xml",
        "retmode": "xml"
    }
    if api_key:
        params["api_key"] = api_key
    
    response = requests.get(f"{ENTREZ_BASE}/efetch.fcgi", params=params, timeout=30)
    response.raise_for_status()
    
    root = ET.fromstring(response.content)
    
    result = {
        "gene_id": gene_id,
        "symbol": None,
        "full_name": None,
        "summary": None
    }
    
    entrezgene = root.find(".//Entrezgene")
    if entrezgene is not None:
        # Get Symbol and Full Name
        gene_ref = entrezgene.find(".//Entrezgene_gene/Gene-ref")
        if gene_ref is not None:
            symbol_elem = gene_ref.find("Gene-ref_locus")
            if symbol_elem is not None:
                result["symbol"] = symbol_elem.text
            
            desc_elem = gene_ref.find("Gene-ref_desc")
            if desc_elem is not None:
                result["full_name"] = desc_elem.text
        
        # Get Summary
        summary_elem = entrezgene.find(".//Entrezgene_summary")
        if summary_elem is not None:
            result["summary"] = summary_elem.text
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Download gene summaries from NCBI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with a single gene (PPP2R1A)
  python download_gene_summaries.py --gene-id 5518
  
  # Download all human protein-coding gene summaries
  python download_gene_summaries.py --output-dir ./gene_data --protein-coding
  
  # Download with API key for faster rate limits
  python download_gene_summaries.py --output-dir ./gene_data --api-key YOUR_KEY
  
  # Test with first 100 genes
  python download_gene_summaries.py --output-dir ./gene_data --max-genes 100
        """
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./gene_data",
        help="Output directory for downloaded files"
    )
    parser.add_argument(
        "--gene-id", "-g",
        type=int,
        help="Fetch a single gene by ID (e.g., 5518 for PPP2R1A)"
    )
    parser.add_argument(
        "--human-only",
        action="store_true",
        default=True,
        help="Download only human genes (default: True)"
    )
    parser.add_argument(
        "--all-species",
        action="store_true",
        help="Download genes for all species (warning: very large)"
    )
    parser.add_argument(
        "--protein-coding",
        action="store_true",
        help="Filter to protein-coding genes only"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="NCBI API key for higher rate limits (get from ncbi.nlm.nih.gov/account)"
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        help="Maximum number of genes to fetch (for testing)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from existing progress"
    )
    
    args = parser.parse_args()
    
    # Handle single gene fetch
    if args.gene_id:
        print(f"Fetching gene {args.gene_id}...")
        result = fetch_single_gene(args.gene_id, api_key=args.api_key)
        print("\n" + "="*60)
        print(f"Gene ID: {result['gene_id']}")
        print(f"Official Symbol: {result['symbol']}")
        print(f"Official Full Name: {result['full_name']}")
        print(f"Summary: {result['summary']}")
        print("="*60)
        return
    
    # Download all genes
    gene_types = ["protein-coding"] if args.protein_coding else None
    human_only = not args.all_species
    
    df = download_all_gene_summaries(
        output_dir=args.output_dir,
        human_only=human_only,
        gene_types=gene_types,
        api_key=args.api_key,
        max_genes=args.max_genes,
        resume=not args.no_resume
    )
    
    print(f"\nDone! Downloaded {len(df)} gene summaries.")
    print(f"Genes with summaries: {df['summary'].notna().sum()}")


if __name__ == "__main__":
    main()

