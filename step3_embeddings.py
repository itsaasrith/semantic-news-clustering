"""
Step 3: Generate Semantic Embeddings
Uses sentence transformers to create semantic embeddings of text
"""
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class SemanticEmbedder:
    """Generates semantic embeddings using pre-trained transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedder
        
        Args:
            model_name: Name of the sentence transformer model
                       'all-MiniLM-L6-v2': Fast, good performance (default)
                       'all-mpnet-base-v2': Better quality, slower
                       'paraphrase-MiniLM-L6-v2': Good for paraphrase detection
        """
        print(f"Loading sentence transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, 
                   show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text documents
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        print(f"Generating embeddings for {len(texts)} documents...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()
    
    def compute_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity matrix between embeddings
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            
        Returns:
            Similarity matrix (n_samples, n_samples)
        """
        # Since embeddings are normalized, dot product = cosine similarity
        similarity_matrix = np.dot(embeddings, embeddings.T)
        return similarity_matrix


if __name__ == "__main__":
    # Example usage
    from step1_data_loader import NewsDataLoader
    from step2_preprocessor import TextPreprocessor
    
    # Load and preprocess data
    loader = NewsDataLoader(subset='train')
    texts, labels, categories = loader.load_data(num_samples=100)
    
    preprocessor = TextPreprocessor()
    cleaned_texts = preprocessor.preprocess_batch(texts)
    
    # Generate embeddings
    embedder = SemanticEmbedder()
    embeddings = embedder.embed_texts(cleaned_texts)
    
    # Compute similarity
    similarity_matrix = embedder.compute_similarity(embeddings)
    
    print(f"\nSimilarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    print(f"\nSample similarities (first doc with others):")
    print(similarity_matrix[0, :5])
