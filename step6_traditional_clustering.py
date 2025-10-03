"""
Step 6: Compare with Traditional Clustering
Compares graph-based clustering with traditional methods
"""
import numpy as np
from typing import Dict, Tuple
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    adjusted_rand_score, 
    normalized_mutual_info_score,
    silhouette_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)


class TraditionalClusterer:
    """Applies traditional clustering algorithms for comparison"""
    
    def __init__(self, n_clusters: int = 10):
        """
        Initialize the clusterer
        
        Args:
            n_clusters: Number of clusters
        """
        self.n_clusters = n_clusters
        self.results = {}
    
    def kmeans_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply K-Means clustering
        
        Args:
            embeddings: Document embeddings
            
        Returns:
            Array of cluster labels
        """
        print(f"\nApplying K-Means Clustering (k={self.n_clusters})...")
        
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        labels = kmeans.fit_predict(embeddings)
        
        self.results['kmeans'] = labels
        self._print_cluster_sizes('K-Means', labels)
        return labels
    
    def hierarchical_clustering(self, embeddings: np.ndarray, 
                               linkage: str = 'ward') -> np.ndarray:
        """
        Apply Hierarchical Clustering
        
        Args:
            embeddings: Document embeddings
            linkage: Linkage criterion ('ward', 'complete', 'average')
            
        Returns:
            Array of cluster labels
        """
        print(f"\nApplying Hierarchical Clustering (k={self.n_clusters}, linkage={linkage})...")
        
        hierarchical = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=linkage
        )
        labels = hierarchical.fit_predict(embeddings)
        
        self.results[f'hierarchical_{linkage}'] = labels
        self._print_cluster_sizes(f'Hierarchical ({linkage})', labels)
        return labels
    
    def dbscan_clustering(self, embeddings: np.ndarray, 
                         eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """
        Apply DBSCAN clustering
        
        Args:
            embeddings: Document embeddings
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
            
        Returns:
            Array of cluster labels
        """
        print(f"\nApplying DBSCAN (eps={eps}, min_samples={min_samples})...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(embeddings)
        
        self.results['dbscan'] = labels
        self._print_cluster_sizes('DBSCAN', labels)
        return labels
    
    def _print_cluster_sizes(self, method_name: str, labels: np.ndarray):
        """Print cluster size distribution"""
        unique, counts = np.unique(labels, return_counts=True)
        print(f"{method_name} - Found {len(unique)} clusters")
        if len(unique) > 0:
            # Filter out noise (-1 label in DBSCAN)
            valid_counts = counts[unique >= 0]
            if len(valid_counts) > 0:
                print(f"  Cluster sizes: min={valid_counts.min()}, "
                      f"max={valid_counts.max()}, mean={valid_counts.mean():.1f}")
    
    def get_all_results(self) -> Dict[str, np.ndarray]:
        """Get all clustering results"""
        return self.results


class ClusteringEvaluator:
    """Evaluates and compares clustering results"""
    
    @staticmethod
    def evaluate_clustering(true_labels: np.ndarray, 
                          pred_labels: np.ndarray,
                          embeddings: np.ndarray = None,
                          method_name: str = "Method") -> Dict[str, float]:
        """
        Evaluate clustering results
        
        Args:
            true_labels: Ground truth labels
            pred_labels: Predicted cluster labels
            embeddings: Optional embeddings for silhouette score
            method_name: Name of the clustering method
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Filter out noise points (label -1 from DBSCAN)
        mask = pred_labels >= 0
        true_labels_filtered = true_labels[mask]
        pred_labels_filtered = pred_labels[mask]
        
        if len(np.unique(pred_labels_filtered)) < 2:
            print(f"{method_name}: Too few clusters for evaluation")
            return {}
        
        metrics = {
            'ARI': adjusted_rand_score(true_labels_filtered, pred_labels_filtered),
            'NMI': normalized_mutual_info_score(true_labels_filtered, pred_labels_filtered),
            'Homogeneity': homogeneity_score(true_labels_filtered, pred_labels_filtered),
            'Completeness': completeness_score(true_labels_filtered, pred_labels_filtered),
            'V-Measure': v_measure_score(true_labels_filtered, pred_labels_filtered),
        }
        
        # Add silhouette score if embeddings provided
        if embeddings is not None and len(np.unique(pred_labels_filtered)) > 1:
            try:
                embeddings_filtered = embeddings[mask]
                metrics['Silhouette'] = silhouette_score(
                    embeddings_filtered, pred_labels_filtered
                )
            except Exception as e:
                print(f"Could not compute silhouette score: {e}")
        
        return metrics
    
    @staticmethod
    def compare_methods(results: Dict[str, np.ndarray], 
                       true_labels: np.ndarray,
                       embeddings: np.ndarray = None) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple clustering methods
        
        Args:
            results: Dictionary of method_name -> cluster_labels
            true_labels: Ground truth labels
            embeddings: Optional embeddings
            
        Returns:
            Dictionary of method_name -> metrics
        """
        print("\n" + "="*70)
        print("CLUSTERING EVALUATION RESULTS")
        print("="*70)
        
        all_metrics = {}
        
        for method_name, pred_labels in results.items():
            metrics = ClusteringEvaluator.evaluate_clustering(
                true_labels, pred_labels, embeddings, method_name
            )
            all_metrics[method_name] = metrics
            
            if metrics:
                print(f"\n{method_name}:")
                for metric_name, value in metrics.items():
                    print(f"  {metric_name:15s}: {value:.4f}")
        
        print("\n" + "="*70)
        return all_metrics


if __name__ == "__main__":
    # Example usage
    from step1_data_loader import NewsDataLoader
    from step2_preprocessor import TextPreprocessor
    from step3_embeddings import SemanticEmbedder
    from step4_similarity_graph import SimilarityGraph
    from step5_graph_clustering import GraphClusterer
    
    # Load, preprocess, and embed data
    loader = NewsDataLoader(subset='train')
    texts, labels, categories = loader.load_data(num_samples=300)
    
    preprocessor = TextPreprocessor()
    cleaned_texts = preprocessor.preprocess_batch(texts)
    
    embedder = SemanticEmbedder()
    embeddings = embedder.embed_texts(cleaned_texts)
    similarity_matrix = embedder.compute_similarity(embeddings)
    
    # Build graph and apply graph clustering
    graph_builder = SimilarityGraph(similarity_threshold=0.3, top_k_neighbors=15)
    graph = graph_builder.build_graph(similarity_matrix, labels=labels)
    
    n_true_clusters = len(np.unique(labels))
    graph_clusterer = GraphClusterer(n_clusters=n_true_clusters)
    spectral_labels = graph_clusterer.spectral_clustering(graph, similarity_matrix)
    louvain_labels = graph_clusterer.louvain_clustering(graph)
    
    # Apply traditional clustering
    trad_clusterer = TraditionalClusterer(n_clusters=n_true_clusters)
    kmeans_labels = trad_clusterer.kmeans_clustering(embeddings)
    hierarchical_labels = trad_clusterer.hierarchical_clustering(embeddings)
    
    # Combine all results
    all_results = {
        'Spectral': spectral_labels,
        'Louvain': louvain_labels,
        'K-Means': kmeans_labels,
        'Hierarchical': hierarchical_labels,
    }
    
    # Evaluate and compare
    evaluator = ClusteringEvaluator()
    metrics = evaluator.compare_methods(all_results, labels, embeddings)
