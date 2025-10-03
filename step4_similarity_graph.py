"""
Step 4: Build Similarity Graph
Constructs a graph where nodes are documents and edges represent semantic similarity
"""
import numpy as np
import networkx as nx
from typing import Tuple, Optional


class SimilarityGraph:
    """Builds and manages the document similarity graph"""
    
    def __init__(self, similarity_threshold: float = 0.3, 
                 top_k_neighbors: Optional[int] = None):
        """
        Initialize the graph builder
        
        Args:
            similarity_threshold: Minimum similarity for edge creation
            top_k_neighbors: If set, keep only top-k most similar neighbors per node
        """
        self.similarity_threshold = similarity_threshold
        self.top_k_neighbors = top_k_neighbors
        self.graph = None
    
    def build_graph(self, similarity_matrix: np.ndarray, 
                   labels: Optional[np.ndarray] = None,
                   texts: Optional[list] = None) -> nx.Graph:
        """
        Build similarity graph from similarity matrix
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            labels: Optional true labels for evaluation
            texts: Optional text content for nodes
            
        Returns:
            NetworkX graph
        """
        n_nodes = similarity_matrix.shape[0]
        print(f"Building similarity graph with {n_nodes} nodes...")
        
        # Create graph
        self.graph = nx.Graph()
        
        # Add nodes with attributes
        for i in range(n_nodes):
            node_attrs = {'id': i}
            if labels is not None:
                node_attrs['true_label'] = int(labels[i])
            if texts is not None:
                node_attrs['text'] = texts[i][:100]  # Store truncated text
            self.graph.add_node(i, **node_attrs)
        
        # Add edges based on similarity
        edge_count = 0
        
        for i in range(n_nodes):
            # Get similarities for node i
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1  # Ignore self-similarity
            
            if self.top_k_neighbors is not None:
                # Keep only top-k neighbors
                top_k_indices = np.argsort(similarities)[-self.top_k_neighbors:]
                for j in top_k_indices:
                    if similarities[j] >= self.similarity_threshold and j > i:
                        self.graph.add_edge(i, j, weight=float(similarities[j]))
                        edge_count += 1
            else:
                # Use threshold only
                for j in range(i + 1, n_nodes):
                    if similarities[j] >= self.similarity_threshold:
                        self.graph.add_edge(i, j, weight=float(similarities[j]))
                        edge_count += 1
        
        print(f"Graph created with {edge_count} edges")
        print(f"Average degree: {sum(dict(self.graph.degree()).values()) / n_nodes:.2f}")
        
        # Graph statistics
        if nx.is_connected(self.graph):
            print("Graph is connected")
        else:
            components = list(nx.connected_components(self.graph))
            print(f"Graph has {len(components)} connected components")
            print(f"Largest component size: {len(max(components, key=len))}")
        
        return self.graph
    
    def get_graph_stats(self) -> dict:
        """Get statistics about the graph"""
        if self.graph is None:
            raise ValueError("Graph not built yet")
        
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'num_components': nx.number_connected_components(self.graph),
        }
        
        # Add clustering coefficient if graph is not too large
        if self.graph.number_of_nodes() < 1000:
            stats['avg_clustering'] = nx.average_clustering(self.graph)
        
        return stats


if __name__ == "__main__":
    # Example usage
    from step1_data_loader import NewsDataLoader
    from step2_preprocessor import TextPreprocessor
    from step3_embeddings import SemanticEmbedder
    
    # Load, preprocess, and embed data
    loader = NewsDataLoader(subset='train')
    texts, labels, categories = loader.load_data(num_samples=200)
    
    preprocessor = TextPreprocessor()
    cleaned_texts = preprocessor.preprocess_batch(texts)
    
    embedder = SemanticEmbedder()
    embeddings = embedder.embed_texts(cleaned_texts)
    similarity_matrix = embedder.compute_similarity(embeddings)
    
    # Build graph
    graph_builder = SimilarityGraph(similarity_threshold=0.4, top_k_neighbors=10)
    graph = graph_builder.build_graph(similarity_matrix, labels=labels)
    
    stats = graph_builder.get_graph_stats()
    print(f"\nGraph statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
