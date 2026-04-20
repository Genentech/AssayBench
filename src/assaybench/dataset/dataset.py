from datasets import Dataset
from datasets import load_from_disk
import os
import dotenv
from typing import Literal
dotenv.load_dotenv()

import os
from typing import List, Dict, Any, Optional
import numpy as np


#from screensqa.prompt_generation.datasets import StudentTeacherDataset, InMemoryDSPackage
from assaybench.utils.prompt_loaders import load_objective_prompt
from assaybench.utils.screen_processing import _extract_screen_ids_from_dataset_name


class AssayBenchDataset:
    """
    Loader for AssayBench Screen Ranking dataset.
    
    This dataset contains BioGRID CRISPR screen data where the task is to rank genes
    by their relevance scores (thresholded percentile scores based on hit status).
    
    The dataset format includes:
    - dataset_name: Screen ID
    - relevance_genes: List of gene names
    - relevance_scores: Thresholded percentile scores
    - hit: Boolean hit labels
    - cell_line, cell_type, phenotype, screen_rationale, library_methodology: Metadata
    """
    
    def __init__(
        self,
        dataset_path: str = "/cv/data/braid/gnesys/datasets/screensQA/biogrid_v0.4_combined",
        split_type: str = "year",
        fold: int = 0,
        max_examples: Optional[int] = None,
        prompt_template: Optional[str] = None,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_seed: int = 42,
        display_library_genes: bool = False,
        use_existing_prompt: bool = False,
        internal_ds_path: bool = None,
    ):
        """
        Initialize the BioGRID dataset loader.
        
        Args:
            dataset_path: Path to the BioGRID HuggingFace dataset on disk.
            split_type: Type of split to use (default: "random_split")
            fold: Fold number for cross-validation (default: 0)
            max_examples: Maximum number of examples to load (default: None = all)
            prompt_template: Custom prompt template to use (default: None = load biogrid_ranking_prompt from YAML)
            test_size: Fraction of data for test set (default: 0.1)
            val_size: Fraction of data for validation set (default: 0.1)
            random_seed: Random seed for reproducibility (default: 42)
            display_library_genes: Whether to include the list of genes considered in the screen in the prompt - only done when the number of genes is small (<2000) (default: False)
            use_existing_prompt: If True, use the ``"prompt"`` field from each
                dataset entry as the question (bypassing template formatting).
                Falls back to template formatting if the field is absent.
            internal_ds_path: Path to the internal dataset (default: None) - this effectively adds an additional test set of more recent screens that have not been included in the training data, to evaluate generalization to new screens
        """
        self.dataset_path = dataset_path
        self.split_type = split_type
        self.fold = fold
        self.max_examples = max_examples
        self.dataset = None
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed
        self.display_library_genes = display_library_genes
        self.use_existing_prompt = use_existing_prompt
        
        # Use custom prompt template or load from YAML
        if prompt_template is not None:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = load_objective_prompt("biogrid_ranking_prompt")

        self.internal_ds_path = internal_ds_path
    
    def load(self) -> Dataset:
        """
        Load the dataset from disk.
        
        Returns:
            HuggingFace Dataset object
        """
        dataset = load_from_disk(self.dataset_path)
        
        # Limit number of examples if specified
        if self.max_examples is not None:
            dataset = dataset.select(range(min(self.max_examples, len(dataset))))

        # Remove the merged directional screens if specified
        if self.internal_ds_path is not None:
            if isinstance(self.internal_ds_path, list):
                self.internal_ds = [load_from_disk(path) for path in self.internal_ds_path]
            else:
                self.internal_ds = [load_from_disk(self.internal_ds_path)]
        else:
            self.internal_ds = None
        
        self.dataset = dataset
        return dataset
    
    def get_list_examples(self) -> List[Dict[str, Any]]:
        """
        Convert dataset to list of examples.
        
        Each example contains:
        - question: The ranking task prompt
        - relevance_genes: List of gene names
        - relevance_scores: Relevance scores for each gene
        - hit: Boolean hit labels
        - dataset_name: Screen ID
        - phenotype, cell_type, cell_line: Metadata
        - answer: Top 10 genes as comma-separated string
        
        Returns:
            List of dictionaries suitable for DSPy
        """
        if self.dataset is None:
            self.load()
        
        examples = []


        for item in self.dataset:
            # Get top 10 genes by relevance score
            top10_genes_args = np.argsort(item['relevance_scores'])[::-1]
            top10_genes = np.array(item['relevance_genes'])[top10_genes_args][:10].tolist()
            
            #Removing trailing period from phenotype if it exists for better prompt formatting
            if item["phenotype"][-1] == ".":
                item["phenotype"] = item["phenotype"][:-1] # remove trailing period for better prompt formatting
            if self.use_existing_prompt and "prompt" in item:
                prompt = item["prompt"]
            else:
                prompt = self.prompt_template.format(**item)
            if self.display_library_genes:
                if len(item['relevance_genes']) < 2000:
                    prompt += "\n\Only the following genes were considered in this screen. Only output genes from this list: " + ', '.join(item['relevance_genes'])

 
            example = {
                'question': prompt,
                'relevance_genes': item['relevance_genes'],
                'relevance_scores': item['relevance_scores'],
                'hit': item['hit'],
                'dataset_name': item['dataset_name'],
                'screen_ids': _extract_screen_ids_from_dataset_name(item["dataset_name"]),
                'phenotype': item.get('phenotype', 'Not specified'),
                'condition_clause': item.get('condition_clause', 'Not specified'),
                'cell_type': item.get('cell_type', 'Not specified'),
                'cell_line': item.get('cell_line', 'Not specified'),
                'screen_type': item.get('screen_type', 'Not specified'),
                'public_screen': item.get('public_screen', 'Not specified'),
                'library_methodology': item.get('library_methodology', 'Not specified'),
                'screen_rationale': item.get('screen_rationale', 'Not specified'),
                'num_genes': len(item['relevance_genes']),
                'screen_category': item.get('screen_category', 'Not specified'),
                'split': item[self.split_type+"fold"+str(self.fold)] if self.split_type is not None else "unknown",
                'cleaned_phenotype': item.get('cleaned_phenotype', 'Not specified'),
                # Create answer as top 10 genes for reference
                'answer': ', '.join(top10_genes)
            }
            examples.append(example)
        if self.internal_ds is not None:
            for internal_ds in self.internal_ds:
                for item in internal_ds:
                    prompt = self.prompt_template.format(**item)
                    top10_genes_args = np.argsort(item['relevance_scores'])[::-1]
                    top10_genes = np.array(item['relevance_genes'])[top10_genes_args][:10].tolist()
                    example = {
                        'question': prompt,
                        'relevance_genes': item['relevance_genes'],
                        'relevance_scores': item['relevance_scores'],
                        'hit': item['hit'],
                        'dataset_name': item['dataset_name'],
                        'screen_ids': _extract_screen_ids_from_dataset_name(item["dataset_name"]),
                        'phenotype': item.get('phenotype', 'Not specified'),
                        'condition_clause': item.get('condition_clause', 'Not specified'),
                        'cell_type': item.get('cell_type', 'Not specified'),
                        'cell_line': item.get('cell_line', 'Not specified'),
                        'screen_type': item.get('screen_type', 'Not specified'),
                        'public_screen': item.get('public_screen', True),
                        'library_methodology': item.get('library_methodology', 'Not specified'),
                        'screen_rationale': item.get('screen_rationale', 'Not specified'),
                        'num_genes': len(item['relevance_genes']),
                        'screen_category': item.get('screen_category', 'Not specified'),
                        'split': "internal_test",
                        # Create answer as top 10 genes for reference
                        'answer': ', '.join(top10_genes)
                    }
                    examples.append(example)
        
        return examples
    
    def get_train_test_split(
        self,
        include_internal_test: bool = False
    ) -> tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset into train, validation, and test sets.
        
        Returns:
            Tuple of (train_examples, val_examples, test_examples, internal_test_examples) - internal_test_examples is only returned if include_internal_test is True and internal_ds is not None
        
        """
        
        examples = self.get_list_examples()

        if self.split_type is None:
            train_examples = []
            val_examples = []
            test_examples = examples
        else:
            train_examples = [example for example in examples if example['split'] == 'train']
            val_examples = [example for example in examples if example['split'] == 'validation']
            test_examples = [example for example in examples if example['split'] == 'test']
        
        if include_internal_test and self.internal_ds is not None:
            internal_test_examples = [example for example in examples if example['split'] == 'internal_test']
            return train_examples, val_examples, test_examples, internal_test_examples
        return train_examples, val_examples, test_examples
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        if self.dataset is None:
            self.load()
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example by index."""
        if self.dataset is None:
            self.load()
        return self.dataset[idx]