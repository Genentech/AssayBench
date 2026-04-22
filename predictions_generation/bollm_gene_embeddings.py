"""
BOLLM Gene Embeddings Utility

Provides a drop-in replacement for PertGeneEmbeddings that uses the BOLLM embeddings
from gene_embeddings.pkl and genes_list.txt (BOLLM embedding files)

The BOLLM embeddings have shape (10278 genes, 128 dimensions, 42 embedding options).

Available embedding options (0-41):
    0: periscope_HeLa_DMEM
    1: periscope_A549_CP186
    2: periscope_HeLa_HPLM
    3: CRISPRGeneEffectDepMap
    4: cellprofiler
    5: GenePT_ada
    6: GenePT_protein
    7: biogpt
    8: esm
    9: stringdb_highest
    10: stringdb_high
    11: stringdb_medium
    12-41: MSigDB pathway collections (c1-c8, hallmarks)
"""

import rootutils

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Union


class BOLLMGeneEmbeddings:
    """
    Gene embeddings from BOLLM with flexible option selection.
    
    The embeddings have shape (10278, 128, 42) where:
    - 10278 genes
    - 128-dimensional embeddings
    - 42 different embedding options (layers, contexts, or methods)
    
    You can select embeddings in multiple ways:
    1. Single option: embedding_option=0 (returns 128-dim embedding)
    2. Multiple options with mean: embedding_option=[0,1,2], combine='mean' (returns 128-dim)
    3. Multiple options with concatenation: embedding_option=[0,1,2], combine='concat' (returns 384-dim)
    4. All options with mean: embedding_option='all', combine='mean' (returns 128-dim)
    5. All options with concatenation: embedding_option='all', combine='concat' (returns 5376-dim)
    6. By name: embedding_option='GenePT_protein' (returns 128-dim)
    """
    
    # Mapping of short names to indices (parsed from embeddings_names.txt)
    EMBEDDING_NAMES = {
        'periscope_HeLa_DMEM': 0,
        'periscope_A549_CP186': 1,
        'periscope_HeLa_HPLM': 2,
        'CRISPRGeneEffectDepMap': 3,
        'cellprofiler': 4,
        'GenePT_ada': 5,
        'GenePT_protein': 6,
        'biogpt': 7,
        'esm': 8,
        'stringdb_highest': 9,
        'stringdb_high': 10,
        'stringdb_medium': 11,
        'msigdb_c1': 12,
        'msigdb_c2_all': 13,
        'msigdb_c2_cgp': 14,
        'msigdb_c2_cp_pid': 15,
        'msigdb_c2_cp_reactome': 16,
        'msigdb_c2_cp': 17,
        'msigdb_c2_cp_wikipathways': 18,
        'msigdb_c3_all': 19,
        'msigdb_c3_mir_mirdb': 20,
        'msigdb_c3_mir_legacy': 21,
        'msigdb_c3_mir': 22,
        'msigdb_c3_tft_gtrd': 23,
        'msigdb_c3_tft_legacy': 24,
        'msigdb_c3_tft': 25,
        'msigdb_c4_3ca': 26,
        'msigdb_c4_all': 27,
        'msigdb_c4_cgn': 28,
        'msigdb_c4_cm': 29,
        'msigdb_c5_all': 30,
        'msigdb_c5_go_bp': 31,
        'msigdb_c5_go_cc': 32,
        'msigdb_c5_go_mf': 33,
        'msigdb_c5_go': 34,
        'msigdb_c5_hpo': 35,
        'msigdb_c6': 36,
        'msigdb_c7_all': 37,
        'msigdb_c7_immunesigdb': 38,
        'msigdb_c7_vax': 39,
        'msigdb_c8': 40,
        'msigdb_hallmarks': 41,
    }
    
    def __init__(
        self,
        embeddings_path: str = "./data/bollm/gene_embeddings.pkl",
        genes_list_path: str = "./data/bollm/genes_list.txt",
        embedding_names_path: str = "./data/bollm/embeddings_names.txt",
        embedding_option: Union[int, List[int], str] = 0,
        combine: str = 'mean'
    ):
        """
        Initialize BOLLM gene embeddings.
        
        Args:
            embeddings_path: Path to gene_embeddings.pkl
            genes_list_path: Path to genes_list.txt
            embedding_names_path: Path to embeddings_names.txt
            embedding_option: Which embedding option(s) to use
                - int: Single option index (0-41)
                - List[int]: Multiple option indices (e.g., [0, 1, 2])
                - str (name): Single embedding by name (e.g., 'GenePT_protein')
                - List[str]: Multiple embeddings by name
                - 'all': Use all 42 options
            combine: How to combine multiple options
                - 'mean': Average across options (maintains 128 dims)
                - 'concat': Concatenate options (increases dims)
        """
        self.embeddings_path = Path(embeddings_path)
        self.genes_list_path = Path(genes_list_path)
        self.embedding_names_path = Path(embedding_names_path)
        
        # Convert string names to indices
        self.original_embedding_option = embedding_option
        self.embedding_option = self._parse_embedding_option(embedding_option)
        self.combine = combine
        
        # Load embeddings
        print(f"Loading BOLLM embeddings from {self.embeddings_path}")
        with open(self.embeddings_path, 'rb') as f:
            self.embeddings = pickle.load(f)
        
        # Load genes list
        with open(self.genes_list_path, 'r') as f:
            self.genes_list = [line.strip() for line in f if line.strip()]
        
        # Create gene to index mapping
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.genes_list)}
        
        # Validate
        assert self.embeddings.shape[0] == len(self.genes_list), \
            f"Mismatch: {self.embeddings.shape[0]} embeddings but {len(self.genes_list)} genes"
        
        print(f"Loaded {len(self.genes_list)} genes with embeddings shape {self.embeddings.shape}")
        print(f"Embedding option: {self.original_embedding_option}, combine: {combine}")
        
        # Determine output dimension
        self._compute_output_dim()
    
    def _parse_embedding_option(self, option: Union[int, List[int], str, List[str]]) -> Union[int, List[int], str]:
        """
        Parse embedding option, converting names to indices.
        
        Args:
            option: Embedding option (int, list of ints, name, list of names, or 'all')
        
        Returns:
            Parsed option (int, list of ints, or 'all')
        """
        if option == 'all':
            return 'all'
        elif isinstance(option, int):
            return option
        elif isinstance(option, list):
            # Check if it's a list of strings (names) or ints
            if all(isinstance(x, str) for x in option):
                # Convert names to indices
                indices = []
                for name in option:
                    if name not in self.EMBEDDING_NAMES:
                        raise ValueError(f"Unknown embedding name: {name}. "
                                       f"Available names: {list(self.EMBEDDING_NAMES.keys())}")
                    indices.append(self.EMBEDDING_NAMES[name])
                return indices
            elif all(isinstance(x, int) for x in option):
                return option
            else:
                raise ValueError("Embedding option list must contain all strings or all ints")
        elif isinstance(option, str):
            # Single name
            if option not in self.EMBEDDING_NAMES:
                raise ValueError(f"Unknown embedding name: {option}. "
                               f"Available names: {list(self.EMBEDDING_NAMES.keys())}")
            return self.EMBEDDING_NAMES[option]
        else:
            raise ValueError(f"Unknown embedding option type: {type(option)}")
    
    def _compute_output_dim(self):
        """Compute the output embedding dimension based on options."""
        if isinstance(self.embedding_option, int):
            self.output_dim = 128
        elif isinstance(self.embedding_option, list):
            if self.combine == 'mean':
                self.output_dim = 128
            elif self.combine == 'concat':
                self.output_dim = 128 * len(self.embedding_option)
            else:
                raise ValueError(f"Unknown combine method: {self.combine}")
        elif self.embedding_option == 'all':
            if self.combine == 'mean':
                self.output_dim = 128
            elif self.combine == 'concat':
                self.output_dim = 128 * 42
            else:
                raise ValueError(f"Unknown combine method: {self.combine}")
        else:
            raise ValueError(f"Unknown embedding_option type: {type(self.embedding_option)}")
        
        print(f"Output embedding dimension: {self.output_dim}")
    
    def get_embedding(self, gene: str) -> np.ndarray:
        """
        Get embedding for a single gene.
        
        Args:
            gene: Gene name (e.g., 'TP53')
        
        Returns:
            Embedding vector (shape depends on options)
        """
        if gene not in self.gene_to_idx:
            # Return zero embedding for unknown genes
            return np.zeros(self.output_dim, dtype=np.float32)
        
        gene_idx = self.gene_to_idx[gene]
        
        # Get embedding(s) based on option
        if isinstance(self.embedding_option, int):
            # Single option
            emb = self.embeddings[gene_idx, :, self.embedding_option]
        elif isinstance(self.embedding_option, list):
            # Multiple specific options
            # Note: advanced indexing with list returns shape (num_options, 128) not (128, num_options)
            embs = self.embeddings[gene_idx, :, self.embedding_option]  # Shape: (num_options, 128)
            if self.combine == 'mean':
                # Average across options (axis=0) to get (128,)
                emb = np.mean(embs, axis=0)
            elif self.combine == 'concat':
                # Flatten to (num_options * 128,)
                emb = embs.flatten()
            else:
                raise ValueError(f"Unknown combine method: {self.combine}")
        elif self.embedding_option == 'all':
            # All options
            # Shape: (128, 42) - standard slicing preserves order
            embs = self.embeddings[gene_idx, :, :]
            if self.combine == 'mean':
                # Average across options (axis=1) to get (128,)
                emb = np.mean(embs, axis=1)
            elif self.combine == 'concat':
                # Flatten to (128 * 42,)
                emb = embs.T.flatten()  # Transpose to get options as first dim, then flatten
            else:
                raise ValueError(f"Unknown combine method: {self.combine}")
        else:
            raise ValueError(f"Unknown embedding_option: {self.embedding_option}")
        
        return emb.astype(np.float32)
    
    def get_embeddings(self, genes: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple genes.
        
        Args:
            genes: List of gene names
        
        Returns:
            Embeddings array (N x embedding_dim)
        """
        return np.array([self.get_embedding(gene) for gene in genes])
    
    def get_gene_dim(self) -> int:
        """
        Get the output embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.output_dim
    
    @classmethod
    def get_available_embeddings(cls) -> List[str]:
        """
        Get list of all available embedding names.
        
        Returns:
            List of embedding names
        """
        return sorted(cls.EMBEDDING_NAMES.keys())
    
    @classmethod
    def print_available_embeddings(cls):
        """Print all available embeddings with their indices."""
        print("Available BOLLM embeddings:")
        print("="*80)
        for name, idx in sorted(cls.EMBEDDING_NAMES.items(), key=lambda x: x[1]):
            print(f"  [{idx:2d}] {name}")
        print("="*80)
    
    def __repr__(self):
        return (f"BOLLMGeneEmbeddings(genes={len(self.genes_list)}, "
                f"option={self.original_embedding_option}, combine={self.combine}, "
                f"dim={self.output_dim})")


def create_bollm_embedder(
    embedding_option: Union[int, List[int], str] = 0,
    combine: str = 'mean'
) -> BOLLMGeneEmbeddings:
    """
    Factory function to create a BOLLM gene embedder.
    
    Args:
        embedding_option: Which embedding option(s) to use
            - int: Single option index (0-41)
            - List[int]: Multiple option indices (e.g., [0, 1, 2])
            - 'all': Use all 42 options
        combine: How to combine multiple options
            - 'mean': Average across options (maintains 128 dims)
            - 'concat': Concatenate options (increases dims)
    
    Returns:
        BOLLMGeneEmbeddings instance
    
    Examples:
        >>> # Single option
        >>> embedder = create_bollm_embedder(embedding_option=0)
        >>> emb = embedder.get_embedding('TP53')  # Returns 128-dim
        
        >>> # Mean of first 3 options
        >>> embedder = create_bollm_embedder(embedding_option=[0, 1, 2], combine='mean')
        >>> emb = embedder.get_embedding('TP53')  # Returns 128-dim
        
        >>> # Concatenate all options
        >>> embedder = create_bollm_embedder(embedding_option='all', combine='concat')
        >>> emb = embedder.get_embedding('TP53')  # Returns 5376-dim
    """
    return BOLLMGeneEmbeddings(
        embedding_option=embedding_option,
        combine=combine
    )


if __name__ == "__main__":
    # Test the embedder
    print("="*80)
    print("Testing BOLLMGeneEmbeddings")
    print("="*80)
    
    # Print available embeddings
    print("\n")
    BOLLMGeneEmbeddings.print_available_embeddings()
    
    # Test 1: Single option by index
    print("\nTest 1: Single option by index (option=0)")
    embedder = create_bollm_embedder(embedding_option=0)
    emb = embedder.get_embedding('TP53')
    print(f"TP53 embedding shape: {emb.shape}")
    print(f"TP53 embedding (first 10): {emb[:10]}")
    
    # Test 2: Single option by name
    print("\nTest 2: Single option by name (option='GenePT_protein')")
    embedder = create_bollm_embedder(embedding_option='GenePT_protein')
    emb = embedder.get_embedding('TP53')
    print(f"TP53 embedding shape: {emb.shape}")
    print(f"TP53 embedding (first 10): {emb[:10]}")
    
    # Test 3: Mean of multiple options by names
    print("\nTest 3: Mean of multiple options by names (['GenePT_ada', 'GenePT_protein', 'biogpt'])")
    embedder = create_bollm_embedder(
        embedding_option=['GenePT_ada', 'GenePT_protein', 'biogpt'], 
        combine='mean'
    )
    emb = embedder.get_embedding('TP53')
    print(f"TP53 embedding shape: {emb.shape}")
    print(f"TP53 embedding (first 10): {emb[:10]}")
    
    # Test 4: Concatenate multiple options
    print("\nTest 4: Concatenate options [0, 1, 2]")
    embedder = create_bollm_embedder(embedding_option=[0, 1, 2], combine='concat')
    emb = embedder.get_embedding('TP53')
    print(f"TP53 embedding shape: {emb.shape}")
    print(f"TP53 embedding (first 10): {emb[:10]}")
    
    # Test 5: Mean of all options
    print("\nTest 5: Mean of all options")
    embedder = create_bollm_embedder(embedding_option='all', combine='mean')
    emb = embedder.get_embedding('TP53')
    print(f"TP53 embedding shape: {emb.shape}")
    print(f"TP53 embedding (first 10): {emb[:10]}")
    
    # Test 6: Unknown gene
    print("\nTest 6: Unknown gene")
    embedder = create_bollm_embedder(embedding_option=0)
    emb = embedder.get_embedding('UNKNOWN_GENE_XYZ')
    print(f"Unknown gene embedding shape: {emb.shape}")
    print(f"Unknown gene embedding all zeros: {np.allclose(emb, 0)}")
    
    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)

