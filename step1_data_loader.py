"""
Step 1: Collect News Dataset
Loads the 20newsgroups dataset from sklearn for demonstration
"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from typing import Tuple, List


class NewsDataLoader:
    """Handles loading and basic info about the news dataset"""
    
    def __init__(self, subset: str = 'train', categories: List[str] = None, 
                 remove: Tuple = ('headers', 'footers', 'quotes')):
        """
        Initialize the data loader
        
        Args:
            subset: 'train', 'test', or 'all'
            categories: List of categories to load (None = all categories)
            remove: Tuple of parts to remove from the text
        """
        self.subset = subset
        self.categories = categories
        self.remove = remove
        self.dataset = None
        
    def load_data(self, num_samples: int = None) -> Tuple[List[str], np.ndarray, List[str]]:
        """
        Load the 20newsgroups dataset
        
        Args:
            num_samples: Number of samples to load (None = all)
            
        Returns:
            texts: List of document texts
            labels: Array of numeric labels
            target_names: List of category names
        """
        print(f"Loading 20newsgroups dataset (subset={self.subset})...")
        self.dataset = fetch_20newsgroups(
            subset=self.subset,
            categories=self.categories,
            remove=self.remove,
            shuffle=True,
            random_state=42
        )
        
        texts = self.dataset.data
        labels = self.dataset.target
        target_names = self.dataset.target_names
        
        if num_samples is not None:
            texts = texts[:num_samples]
            labels = labels[:num_samples]
        
        print(f"Loaded {len(texts)} documents from {len(target_names)} categories")
        print(f"Categories: {', '.join(target_names)}")
        
        return texts, labels, target_names
    
    def get_sample_docs(self, n: int = 5) -> List[str]:
        """Get first n documents as samples"""
        if self.dataset is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")
        return self.dataset.data[:n]


if __name__ == "__main__":
    # Example usage
    loader = NewsDataLoader(subset='train')
    texts, labels, categories = loader.load_data(num_samples=500)
    
    print(f"\nSample document (first 200 chars):")
    print(texts[0][:200])
    print(f"\nLabel: {labels[0]} ({categories[labels[0]]})")
