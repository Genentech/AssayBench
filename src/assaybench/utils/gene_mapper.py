"""
Gene Mapper - Maps gene symbols/aliases to official HGNC gene symbols.

Uses the HGNC complete dataset to perform mappings including:
- Approved symbols
- Previous symbols (aliases)
- Synonyms
- Manual mappings for common variants/typos
- UniProt protein-to-gene mappings
- Smart normalization rules for common LLM output patterns
"""

import csv
import io
import json
import re
import requests
from importlib.resources import files
from pathlib import Path
from typing import Optional


# HGNC custom download URL for TSV format with symbol, previous symbols, and aliases
HGNC_DOWNLOAD_URL = (
    "https://www.genenames.org/cgi-bin/download/custom?"
    "col=gd_hgnc_id&col=gd_app_sym&col=gd_prev_sym&col=gd_aliases&"
    "status=Approved&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit"
)

# Default paths using package resources
def _get_package_data_dir():
    """Get the package data directory as a Traversable."""
    return files("assaybench.data.hgnc")

def _get_default_cache_path() -> Path:
    """Get default path for HGNC cache file."""
    # For cache file, we need a real filesystem path since we may write to it
    data_dir = _get_package_data_dir()
    # Use joinpath for proper resource path resolution
    resource = data_dir.joinpath("hgnc_symbols_cache.tsv")
    # Convert to Path if it's a real file (for editable installs)
    if hasattr(resource, '__fspath__'):
        return Path(resource)
    return Path(str(resource))

def _get_default_manual_mappings_path():
    """Get default path for manual mappings file."""
    return _get_package_data_dir().joinpath("manual_mappings.json")

def _get_default_uniprot_mappings_path():
    """Get default path for UniProt mappings file."""
    return _get_package_data_dir().joinpath("uniprot_protein_to_gene.json")


class GeneMapper:
    """Maps gene symbols/aliases to official HGNC symbols."""

    def __init__(
        self,
        cache_file: Optional[Path] = None,
        manual_mappings_file = None,
        uniprot_mappings_file = None,
        verbose: bool = False,
    ):
        """
        Initialize the gene mapper.

        Args:
            cache_file: Path to cached HGNC data. If None, uses package default.
            manual_mappings_file: Path or Traversable to manual mappings JSON. If None, uses package default.
            uniprot_mappings_file: Path or Traversable to UniProt mappings JSON. If None, uses package default.
            verbose: If True, print loading messages.
        """
        self.cache_file = cache_file or _get_default_cache_path()
        self.manual_mappings_file = manual_mappings_file or _get_default_manual_mappings_path()
        self.uniprot_mappings_file = uniprot_mappings_file or _get_default_uniprot_mappings_path()
        self.verbose = verbose
        self._symbol_to_hgnc: dict[str, str] = {}
        self._explicitly_unmappable: set[str] = set()  # Genes marked as null in manual mappings
        self._loaded = False

    def _download_hgnc_data(self) -> str:
        """Download HGNC symbol data as TSV."""
        if self.verbose:
            print("Downloading HGNC data...")
        response = requests.get(HGNC_DOWNLOAD_URL, timeout=120)
        response.raise_for_status()
        data = response.text

        # Cache the data
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            f.write(data)
        if self.verbose:
            print(f"HGNC data cached to {self.cache_file}")

        return data

    def _load_hgnc_data(self) -> str:
        """Load HGNC data from cache or download if not available."""
        if self.cache_file.exists():
            if self.verbose:
                print(f"Loading HGNC data from cache: {self.cache_file}")
            with open(self.cache_file) as f:
                return f.read()
        else:
            return self._download_hgnc_data()

    def _load_manual_mappings(self) -> dict[str, Optional[str]]:
        """Load manual mappings from JSON file or Traversable resource."""
        try:
            # Handle both Path and Traversable (package resources)
            if hasattr(self.manual_mappings_file, 'read_text'):
                # It's a Traversable (package resource)
                text = self.manual_mappings_file.read_text()
                data = json.loads(text)
            elif hasattr(self.manual_mappings_file, 'exists') and self.manual_mappings_file.exists():
                # It's a Path
                with open(self.manual_mappings_file) as f:
                    data = json.load(f)
            else:
                return {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

        return data.get("mappings", {})

    def _load_uniprot_mappings(self) -> dict[str, Optional[str]]:
        """Load UniProt protein-to-gene mappings from JSON file or Traversable resource."""
        try:
            # Handle both Path and Traversable (package resources)
            if hasattr(self.uniprot_mappings_file, 'read_text'):
                # It's a Traversable (package resource)
                text = self.uniprot_mappings_file.read_text()
                return json.loads(text)
            elif hasattr(self.uniprot_mappings_file, 'exists') and self.uniprot_mappings_file.exists():
                # It's a Path
                with open(self.uniprot_mappings_file) as f:
                    return json.load(f)
            else:
                return {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _build_lookup_tables(self):
        """Build lookup tables from HGNC data."""
        if self._loaded:
            return

        data = self._load_hgnc_data()
        reader = csv.DictReader(io.StringIO(data), delimiter="\t")

        for row in reader:
            approved_symbol = row.get("Approved symbol", "").strip()
            if not approved_symbol:
                continue

            # Map approved symbol (case-insensitive)
            self._symbol_to_hgnc[approved_symbol.upper()] = approved_symbol

            # Map previous symbols (comma-separated)
            prev_symbols = row.get("Previous symbols", "")
            if prev_symbols:
                for prev in prev_symbols.split(", "):
                    prev = prev.strip()
                    if prev:
                        key = prev.upper()
                        # Don't overwrite if already exists (prefer approved symbols)
                        if key not in self._symbol_to_hgnc:
                            self._symbol_to_hgnc[key] = approved_symbol

            # Map aliases/synonyms (comma-separated)
            aliases = row.get("Alias symbols", "")
            if aliases:
                for alias in aliases.split(", "):
                    alias = alias.strip()
                    if alias:
                        key = alias.upper()
                        if key not in self._symbol_to_hgnc:
                            self._symbol_to_hgnc[key] = approved_symbol

        if self.verbose:
            print(f"Loaded {len(self._symbol_to_hgnc)} gene symbol mappings from HGNC")

        # Load UniProt mappings (lower priority than HGNC, but fills gaps)
        uniprot_mappings = self._load_uniprot_mappings()
        uniprot_count = 0
        uniprot_null_count = 0
        for symbol, hgnc_symbol in uniprot_mappings.items():
            key = symbol.upper()
            if hgnc_symbol is None:
                # UniProt returned no result - mark as explicitly unmappable
                self._explicitly_unmappable.add(key)
                uniprot_null_count += 1
            elif key not in self._symbol_to_hgnc:
                # Only add if not already in HGNC
                self._symbol_to_hgnc[key] = hgnc_symbol
                uniprot_count += 1

        if self.verbose and (uniprot_count > 0 or uniprot_null_count > 0):
            print(f"Loaded {uniprot_count} UniProt mappings, {uniprot_null_count} marked unmappable")

        # Load manual mappings (these override HGNC and UniProt mappings)
        manual_mappings = self._load_manual_mappings()
        manual_count = 0
        for symbol, hgnc_symbol in manual_mappings.items():
            key = symbol.upper()
            if hgnc_symbol is None:
                # Explicitly unmappable (e.g., pathway names, categories)
                self._explicitly_unmappable.add(key)
            else:
                self._symbol_to_hgnc[key] = hgnc_symbol
                manual_count += 1

        if self.verbose and (manual_count > 0 or self._explicitly_unmappable):
            print(f"Loaded {manual_count} manual mappings, {len(self._explicitly_unmappable)} total explicitly unmappable")

        self._loaded = True

    def _extract_candidates(self, gene: str) -> list[str]:
        """
        Extract candidate gene symbols from a potentially messy input string.
        
        Handles common LLM output patterns like:
        - "BBC3 (PUMA)" -> ["BBC3", "PUMA"]
        - "HUWE1 (duplicate removed)" -> ["HUWE1"]
        - "RPS15 [[ ## completed ## ]]" -> ["RPS15"]
        - "VPS11-like(VPS18 already listed)" -> ["VPS11", "VPS18"]
        - "PARK2(Parkin)" -> ["PARK2", "Parkin"]
        - "BRCA1-associated RING domain 1 (BARD1)" -> ["BARD1"]
        - "p62/SQSTM1" -> ["SQSTM1", "p62"]
        - "eIF2α" -> ["EIF2A", "EIF2S1"]
        
        Returns list of candidates to try, in priority order.
        """
        candidates = []
        original = gene.strip()
        
        # Remove common noise patterns
        cleaned = original
        
        # Take only the first line (Kimi outputs reasoning after newlines)
        if '\n' in cleaned:
            cleaned = cleaned.split('\n')[0].strip()
        
        # Remove leading "- " bullet points
        cleaned = re.sub(r'^-\s+', '', cleaned).strip()
        
        # Remove leading numbers like "10." or "42." (numbered lists)
        cleaned = re.sub(r'^\d+\.\s*', '', cleaned).strip()
        
        # Remove [[ ## completed ## ]] or similar bracket markers
        cleaned = re.sub(r'\[\[.*?\]\]', '', cleaned).strip()
        
        # Remove trailing parenthetical annotations that are just comments
        noise_patterns = [
            r'\s*\(duplicate removed\)',
            r'\s*\(duplicates avoided\)',
            r'\s*\(already present\)',
            r'\s*\(already listed\)',
            r'\s*\(listed earlier\)',
            r'\s*\(duplicate\)',
        ]
        for pattern in noise_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        
        # If cleaned version is different, add it as first candidate
        if cleaned and cleaned != original:
            candidates.append(cleaned)
        
        working = cleaned or original
        
        # Pattern: "Full gene description (SYMBOL)" - extract gene from end parenthesis
        # e.g., "BRCA1-associated RING domain 1 (BARD1)" -> BARD1
        # e.g., "DNA polymerase alpha 1 (POLA1)" -> POLA1
        long_desc_match = re.search(r'\(([A-Z][A-Z0-9]{1,10})\)\s*$', working)
        if long_desc_match:
            candidates.append(long_desc_match.group(1))
        
        # Pattern: "complex/pathway (GENE1" - partial parens, extract gene
        # e.g., "GBPs (GBP1" -> GBP1, "HOPS complex (VPS11" -> VPS11
        partial_paren_match = re.search(r'\(([A-Z][A-Z0-9]{1,10})$', working)
        if partial_paren_match:
            candidates.append(partial_paren_match.group(1))
        
        # Pattern: trailing ) with gene before - "GBP5)" or "VPS41)"
        trailing_paren_match = re.match(r'^([A-Z][A-Z0-9]{1,10})\)$', working)
        if trailing_paren_match:
            candidates.append(trailing_paren_match.group(1))
        
        # Pattern: "p62/SQSTM1" or "tBID" - slash separated, try both parts
        if '/' in working:
            parts = working.split('/')
            for part in parts:
                part = part.strip()
                if part and re.match(r'^[A-Za-z][A-Za-z0-9]*$', part):
                    candidates.append(part)
        
        # Pattern: Greek letters - convert to ASCII equivalents
        greek_map = {
            'α': 'A', 'β': 'B', 'γ': 'G', 'δ': 'D', 'ε': 'E',
            'κ': 'K', 'λ': 'L', 'μ': 'M', 'π': 'P', 'σ': 'S',
        }
        ascii_version = working
        for greek, ascii_char in greek_map.items():
            ascii_version = ascii_version.replace(greek, ascii_char)
        if ascii_version != working:
            candidates.append(ascii_version)
            # Also try uppercase version
            candidates.append(ascii_version.upper())
        
        # Pattern: "SYMBOL (ALIAS)" or "SYMBOL(ALIAS)" - extract both parts
        # Match: text before parens, and text inside parens
        paren_match = re.match(r'^([A-Za-z0-9_/-]+)\s*\(([^)]+)\)$', working)
        if paren_match:
            before_paren = paren_match.group(1).strip()
            inside_paren = paren_match.group(2).strip()
            
            # Add before-paren part (e.g., "BBC3" from "BBC3 (PUMA)")
            if before_paren:
                # Handle "-like" suffix: VPS11-like -> VPS11
                if before_paren.endswith('-like'):
                    candidates.append(before_paren[:-5])
                # Handle "-BP" or similar suffixes
                elif '-' in before_paren:
                    base = before_paren.split('-')[0]
                    candidates.append(base)
                    candidates.append(before_paren)
                else:
                    candidates.append(before_paren)
            
            # Try to extract gene symbol from inside parens
            # Skip if it's clearly a description (too long, has spaces and lowercase)
            if inside_paren and len(inside_paren) < 20:
                # Remove common prefixes like "AKA_"
                inside_clean = re.sub(r'^AKA_', '', inside_paren)
                # Take first word if there are spaces (e.g., "VPS retromer" -> "VPS")
                first_word = inside_clean.split()[0] if ' ' in inside_clean else inside_clean
                
                # Only add if it looks like a gene symbol (alphanumeric, possibly with numbers)
                if re.match(r'^[A-Za-z][A-Za-z0-9]*$', first_word):
                    candidates.append(first_word)
                    if inside_clean != first_word:
                        candidates.append(inside_clean)
        
        # Pattern: "SYMBOL-like" without parens
        like_match = re.match(r'^([A-Za-z0-9]+)-like$', working, re.IGNORECASE)
        if like_match:
            candidates.append(like_match.group(1))
        
        # Pattern: "description: GENE" - colon separator
        if ':' in working:
            after_colon = working.split(':')[-1].strip()
            if re.match(r'^[A-Z][A-Z0-9]{1,10}$', after_colon):
                candidates.append(after_colon)
        
        # Pattern: starts with lowercase (like "p53", "tBID", "eIF2") - try uppercase
        if working and working[0].islower():
            candidates.append(working.upper())
            # Also try removing the lowercase prefix
            upper_start = re.search(r'[A-Z]', working)
            if upper_start:
                candidates.append(working[upper_start.start():])
        
        # Pattern: letters followed by numbers without hyphen (ABIN3 -> ABIN-3)
        # Try inserting hyphen before trailing numbers
        hyphen_match = re.match(r'^([A-Za-z]+)(\d+)$', working)
        if hyphen_match:
            hyphenated = f"{hyphen_match.group(1)}-{hyphen_match.group(2)}"
            candidates.append(hyphenated)
        
        # Always include the original/cleaned as last resort
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
        if original not in candidates:
            candidates.append(original)
        
        return candidates

    def map_gene(self, gene: str) -> Optional[str]:
        """
        Map a gene symbol to its official HGNC symbol.
        
        Applies smart normalization to handle common LLM output patterns.

        Args:
            gene: Gene symbol to map (case-insensitive)

        Returns:
            Official HGNC symbol if found, None otherwise
        """
        self._build_lookup_tables()
        # First try exact match
        result = self._symbol_to_hgnc.get(gene.upper())
        if result:
            return result
        
        # Try normalized candidates
        candidates = self._extract_candidates(gene)
        for candidate in candidates:
            result = self._symbol_to_hgnc.get(candidate.upper())
            if result:
                return result
        
        return None

    def is_valid_gene(self, gene: str) -> bool:
        """
        Check if a gene symbol maps to a valid HGNC symbol.

        Args:
            gene: Gene symbol to check

        Returns:
            True if the gene maps to a valid HGNC symbol
        """
        return self.map_gene(gene) is not None

    def is_explicitly_unmappable(self, gene: str) -> bool:
        """
        Check if a gene is explicitly marked as unmappable (e.g., pathway names).

        Args:
            gene: Gene symbol to check

        Returns:
            True if the gene is explicitly marked as not a valid gene symbol
        """
        self._build_lookup_tables()
        
        # Check original
        if gene.upper() in self._explicitly_unmappable:
            return True
        
        # Check normalized candidates
        candidates = self._extract_candidates(gene)
        for candidate in candidates:
            if candidate.upper() in self._explicitly_unmappable:
                return True
        
        return False

    def is_known(self, gene: str) -> bool:
        """
        Check if a gene is either valid or explicitly unmappable.

        Args:
            gene: Gene symbol to check

        Returns:
            True if we have information about this symbol (valid or explicitly unmappable)
        """
        return self.is_valid_gene(gene) or self.is_explicitly_unmappable(gene)

    def map_genes(self, genes: list[str]) -> dict[str, Optional[str]]:
        """
        Map multiple genes to their HGNC symbols.

        Args:
            genes: List of gene symbols

        Returns:
            Dictionary mapping input symbols to HGNC symbols (or None if not found)
        """
        return {gene: self.map_gene(gene) for gene in genes}

    def get_unmapped_genes(self, genes: list[str]) -> list[str]:
        """
        Get list of genes that don't map to HGNC symbols.

        Args:
            genes: List of gene symbols

        Returns:
            List of genes that couldn't be mapped
        """
        return [gene for gene in genes if not self.is_valid_gene(gene)]

    def get_stats(self, genes: list[str]) -> dict:
        """
        Get mapping statistics for a list of genes.

        Args:
            genes: List of gene symbols

        Returns:
            Dictionary with mapping statistics
        """
        mapped = self.map_genes(genes)
        valid = [g for g, m in mapped.items() if m is not None]
        invalid = [g for g, m in mapped.items() if m is None]

        return {
            "total": len(genes),
            "mapped": len(valid),
            "unmapped": len(invalid),
            "mapped_pct": len(valid) / len(genes) * 100 if genes else 0,
            "unmapped_genes": invalid,
        }


def get_mapper(verbose: bool = False) -> GeneMapper:
    """Get a singleton gene mapper instance."""
    if not hasattr(get_mapper, "_instance"):
        get_mapper._instance = GeneMapper(verbose=verbose)
    return get_mapper._instance
