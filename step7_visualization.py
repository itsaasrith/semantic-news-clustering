"""
Step 7: Visualization
Visualizes clustering results using various techniques
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ClusteringVisualizer:
    """Visualizes clustering results"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize the visualizer
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        sns.set_style("whitegrid")
        self.colors = sns.color_palette("husl", 20)
    
    def plot_embeddings_2d(self, embeddings: np.ndarray, 
                          labels: np.ndarray,
                          true_labels: Optional[np.ndarray] = None,
                          method: str = 'tsne',
                          title: str = "Clustering Visualization",
                          save_path: Optional[str] = None):
        """
        Plot embeddings in 2D using dimensionality reduction
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Cluster labels
            true_labels: Optional ground truth labels
            method: Reduction method ('tsne' or 'pca')
            title: Plot title
            save_path: Path to save the figure
        """
        print(f"Reducing dimensions using {method.upper()}...")
        
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create subplots
        n_plots = 2 if true_labels is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(self.figsize[0], self.figsize[1]//2))
        if n_plots == 1:
            axes = [axes]
        
        # Plot predicted clusters
        scatter1 = axes[0].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=labels, cmap='tab20', alpha=0.6, s=50
        )
        axes[0].set_title(f"{title} - Predicted Clusters")
        axes[0].set_xlabel(f"{method.upper()} Component 1")
        axes[0].set_ylabel(f"{method.upper()} Component 2")
        plt.colorbar(scatter1, ax=axes[0], label='Cluster')
        
        # Plot true labels if available
        if true_labels is not None:
            scatter2 = axes[1].scatter(
                embeddings_2d[:, 0], embeddings_2d[:, 1],
                c=true_labels, cmap='tab20', alpha=0.6, s=50
            )
            axes[1].set_title(f"{title} - True Labels")
            axes[1].set_xlabel(f"{method.upper()} Component 1")
            axes[1].set_ylabel(f"{method.upper()} Component 2")
            plt.colorbar(scatter2, ax=axes[1], label='True Category')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_graph_network(self, graph: nx.Graph, 
                          labels: np.ndarray,
                          true_labels: Optional[np.ndarray] = None,
                          layout: str = 'spring',
                          max_nodes: int = 200,
                          save_path: Optional[str] = None):
        """
        Plot the similarity graph network
        
        Args:
            graph: NetworkX graph
            labels: Cluster labels
            true_labels: Optional ground truth labels
            layout: Layout algorithm ('spring', 'kamada_kawai', 'circular')
            max_nodes: Maximum nodes to plot (for performance)
            save_path: Path to save the figure
        """
        print(f"Plotting graph network with {graph.number_of_nodes()} nodes...")
        
        # Sample nodes if graph is too large
        if graph.number_of_nodes() > max_nodes:
            print(f"Sampling {max_nodes} nodes for visualization...")
            nodes = list(graph.nodes())[:max_nodes]
            subgraph = graph.subgraph(nodes).copy()
            labels_subset = labels[:max_nodes]
            true_labels_subset = true_labels[:max_nodes] if true_labels is not None else None
        else:
            subgraph = graph
            labels_subset = labels
            true_labels_subset = true_labels
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(subgraph)
        elif layout == 'circular':
            pos = nx.circular_layout(subgraph)
        else:
            pos = nx.spring_layout(subgraph, seed=42)
        
        # Create figure
        n_plots = 2 if true_labels_subset is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=self.figsize)
        if n_plots == 1:
            axes = [axes]
        
        # Plot with predicted clusters
        node_colors1 = [self.colors[labels_subset[node] % len(self.colors)] 
                       for node in subgraph.nodes()]
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors1, 
                              node_size=100, alpha=0.7, ax=axes[0])
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5, ax=axes[0])
        axes[0].set_title("Graph Network - Predicted Clusters")
        axes[0].axis('off')
        
        # Plot with true labels if available
        if true_labels_subset is not None:
            node_colors2 = [self.colors[true_labels_subset[node] % len(self.colors)] 
                           for node in subgraph.nodes()]
            nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors2, 
                                  node_size=100, alpha=0.7, ax=axes[1])
            nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5, ax=axes[1])
            axes[1].set_title("Graph Network - True Labels")
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_confusion_matrix(self, true_labels: np.ndarray,
                             pred_labels: np.ndarray,
                             title: str = "Confusion Matrix",
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix between true and predicted labels
        
        Args:
            true_labels: Ground truth labels
            pred_labels: Predicted cluster labels
            title: Plot title
            save_path: Path to save the figure
        """
        from sklearn.metrics import confusion_matrix
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Normalize by row (true label)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Proportion'})
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Cluster')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_cluster_sizes(self, results: dict, save_path: Optional[str] = None):
        """
        Plot cluster size distributions for different methods
        
        Args:
            results: Dictionary of method_name -> cluster_labels
            save_path: Path to save the figure
        """
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method_name, labels) in enumerate(results.items()):
            unique, counts = np.unique(labels, return_counts=True)
            axes[idx].bar(unique, counts, color=self.colors[idx % len(self.colors)])
            axes[idx].set_title(f"{method_name}")
            axes[idx].set_xlabel("Cluster ID")
            axes[idx].set_ylabel("Number of Documents")
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
        plt.close()


if __name__ == "__main__":
    # Example usage
    from step1_data_loader import NewsDataLoader
    from step2_preprocessor import TextPreprocessor
    from step3_embeddings import SemanticEmbedder
    from step4_similarity_graph import SimilarityGraph
    from step5_graph_clustering import GraphClusterer
    from step6_traditional_clustering import TraditionalClusterer
    
    # Load, preprocess, and embed data
    loader = NewsDataLoader(subset='train')
    texts, labels, categories = loader.load_data(num_samples=300)
    
    preprocessor = TextPreprocessor()
    cleaned_texts = preprocessor.preprocess_batch(texts)
    
    embedder = SemanticEmbedder()
    embeddings = embedder.embed_texts(cleaned_texts)
    similarity_matrix = embedder.compute_similarity(embeddings)
    
    # Build graph and cluster
    graph_builder = SimilarityGraph(similarity_threshold=0.3, top_k_neighbors=15)
    graph = graph_builder.build_graph(similarity_matrix, labels=labels)
    
    n_true_clusters = len(np.unique(labels))
    graph_clusterer = GraphClusterer(n_clusters=n_true_clusters)
    spectral_labels = graph_clusterer.spectral_clustering(graph, similarity_matrix)
    
    # Visualize
    visualizer = ClusteringVisualizer()
    visualizer.plot_embeddings_2d(embeddings, spectral_labels, labels, 
                                 method='tsne', title='Spectral Clustering')
    visualizer.plot_graph_network(graph, spectral_labels, labels, max_nodes=150)
