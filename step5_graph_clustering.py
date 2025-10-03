"""
Step 5: Apply Graph Clustering
Applies various graph-based clustering algorithms
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from sklearn.cluster import SpectralClustering
import community as community_louvain  # python-louvain


class GraphClusterer:
    """Applies graph-based clustering algorithms"""
    
    def __init__(self, n_clusters: int = 10):
        """
        Initialize the clusterer
        
        Args:
            n_clusters: Number of clusters (for algorithms that require it)
        """
        self.n_clusters = n_clusters
        self.results = {}
    
    def spectral_clustering(self, graph: nx.Graph, 
                           similarity_matrix: np.ndarray = None) -> np.ndarray:
        """
        Apply Spectral Clustering on the graph
        
        Args:
            graph: NetworkX graph
            similarity_matrix: Optional precomputed similarity matrix
            
        Returns:
            Array of cluster labels
        """
        print(f"\nApplying Spectral Clustering (k={self.n_clusters})...")
        
        if similarity_matrix is not None:
            # Use precomputed similarity matrix
            clusterer = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity='precomputed',
                random_state=42,
                assign_labels='kmeans'
            )
            labels = clusterer.fit_predict(similarity_matrix)
        else:
            # Use graph adjacency matrix
            adj_matrix = nx.to_numpy_array(graph)
            clusterer = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity='precomputed',
                random_state=42,
                assign_labels='kmeans'
            )
            labels = clusterer.fit_predict(adj_matrix)
        
        self.results['spectral'] = labels
        self._print_cluster_sizes('Spectral Clustering', labels)
        return labels
    
    def louvain_clustering(self, graph: nx.Graph) -> np.ndarray:
        """
        Apply Louvain community detection
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Array of cluster labels
        """
        print(f"\nApplying Louvain Community Detection...")
        
        # Detect communities
        partition = community_louvain.best_partition(graph, random_state=42)
        
        # Convert to array
        n_nodes = graph.number_of_nodes()
        labels = np.array([partition[i] for i in range(n_nodes)])
        
        self.results['louvain'] = labels
        self._print_cluster_sizes('Louvain', labels)
        return labels
    
    def label_propagation(self, graph: nx.Graph) -> np.ndarray:
        """
        Apply Label Propagation algorithm
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Array of cluster labels
        """
        print(f"\nApplying Label Propagation...")
        
        communities = nx.community.label_propagation_communities(graph)
        
        # Convert to labels
        n_nodes = graph.number_of_nodes()
        labels = np.zeros(n_nodes, dtype=int)
        for cluster_id, community in enumerate(communities):
            for node in community:
                labels[node] = cluster_id
        
        self.results['label_propagation'] = labels
        self._print_cluster_sizes('Label Propagation', labels)
        return labels
    
    def greedy_modularity(self, graph: nx.Graph) -> np.ndarray:
        """
        Apply Greedy Modularity Maximization
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Array of cluster labels
        """
        print(f"\nApplying Greedy Modularity Maximization...")
        
        communities = nx.community.greedy_modularity_communities(graph)
        
        # Convert to labels
        n_nodes = graph.number_of_nodes()
        labels = np.zeros(n_nodes, dtype=int)
        for cluster_id, community in enumerate(communities):
            for node in community:
                labels[node] = cluster_id
        
        self.results['greedy_modularity'] = labels
        self._print_cluster_sizes('Greedy Modularity', labels)
        return labels
    
    def _print_cluster_sizes(self, method_name: str, labels: np.ndarray):
        """Print cluster size distribution"""
        unique, counts = np.unique(labels, return_counts=True)
        print(f"{method_name} - Found {len(unique)} clusters")
        print(f"  Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    
    def get_all_results(self) -> Dict[str, np.ndarray]:
        """Get all clustering results"""
        return self.results


if __name__ == "__main__":
    # Example usage
    from step1_data_loader import NewsDataLoader
    from step2_preprocessor import TextPreprocessor
    from step3_embeddings import SemanticEmbedder
    from step4_similarity_graph import SimilarityGraph
    
    # Load, preprocess, and embed data
    loader = NewsDataLoader(subset='train')
    texts, labels, categories = loader.load_data(num_samples=300)
    
    preprocessor = TextPreprocessor()
    cleaned_texts = preprocessor.preprocess_batch(texts)
    
    embedder = SemanticEmbedder()
    embeddings = embedder.embed_texts(cleaned_texts)
    similarity_matrix = embedder.compute_similarity(embeddings)
    
    # Build graph
    graph_builder = SimilarityGraph(similarity_threshold=0.3, top_k_neighbors=15)
    graph = graph_builder.build_graph(similarity_matrix, labels=labels)
    
    # Apply clustering
    n_true_clusters = len(np.unique(labels))
    clusterer = GraphClusterer(n_clusters=n_true_clusters)
    
    spectral_labels = clusterer.spectral_clustering(graph, similarity_matrix)
    louvain_labels = clusterer.louvain_clustering(graph)
    lp_labels = clusterer.label_propagation(graph)
    gm_labels = clusterer.greedy_modularity(graph)
