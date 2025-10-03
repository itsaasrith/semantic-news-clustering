"""
Step 8: Demonstration / Presentation
Main script that demonstrates the complete semantic news clustering pipeline
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from step1_data_loader import NewsDataLoader
from step2_preprocessor import TextPreprocessor
from step3_embeddings import SemanticEmbedder
from step4_similarity_graph import SimilarityGraph
from step5_graph_clustering import GraphClusterer
from step6_traditional_clustering import TraditionalClusterer, ClusteringEvaluator
from step7_visualization import ClusteringVisualizer


def print_section_header(title: str):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def main():
    """Main demonstration pipeline"""
    
    print_section_header("SEMANTIC NEWS CLUSTERING DEMONSTRATION")
    print("This demo shows how to cluster news articles by meaning (semantics),")
    print("not just shared keywords, using graph-based and traditional methods.")
    
    # Configuration
    NUM_SAMPLES = 500  # Number of documents to cluster
    SIMILARITY_THRESHOLD = 0.35  # Threshold for creating edges in graph
    TOP_K_NEIGHBORS = 10  # Keep top-k similar neighbors per document
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print_section_header("STEP 1: LOAD NEWS DATASET")
    
    loader = NewsDataLoader(subset='train')
    texts, true_labels, categories = loader.load_data(num_samples=NUM_SAMPLES)
    
    print(f"\nDataset statistics:")
    print(f"  Total documents: {len(texts)}")
    print(f"  Number of categories: {len(categories)}")
    print(f"  Categories: {', '.join(categories[:5])}...")
    
    # Show sample
    print(f"\nSample document (category: {categories[true_labels[0]]}):")
    print(texts[0][:300] + "...")
    
    # =========================================================================
    # Step 2: Preprocess Text
    # =========================================================================
    print_section_header("STEP 2: PREPROCESS TEXT")
    
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        lowercase=True,
        min_token_length=2
    )
    cleaned_texts = preprocessor.preprocess_batch(texts, verbose=True)
    
    print(f"\nSample cleaned text:")
    print(cleaned_texts[0][:300] + "...")
    
    # =========================================================================
    # Step 3: Generate Semantic Embeddings
    # =========================================================================
    print_section_header("STEP 3: GENERATE SEMANTIC EMBEDDINGS")
    
    embedder = SemanticEmbedder(model_name='all-MiniLM-L6-v2')
    embeddings = embedder.embed_texts(cleaned_texts, batch_size=32)
    similarity_matrix = embedder.compute_similarity(embeddings)
    
    print(f"\nEmbedding statistics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    
    # =========================================================================
    # Step 4: Build Similarity Graph
    # =========================================================================
    print_section_header("STEP 4: BUILD SIMILARITY GRAPH")
    
    graph_builder = SimilarityGraph(
        similarity_threshold=SIMILARITY_THRESHOLD,
        top_k_neighbors=TOP_K_NEIGHBORS
    )
    graph = graph_builder.build_graph(similarity_matrix, labels=true_labels, texts=cleaned_texts)
    
    stats = graph_builder.get_graph_stats()
    print(f"\nGraph statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # =========================================================================
    # Step 5: Apply Graph-Based Clustering
    # =========================================================================
    print_section_header("STEP 5: APPLY GRAPH-BASED CLUSTERING")
    
    n_true_clusters = len(np.unique(true_labels))
    print(f"Number of true categories: {n_true_clusters}")
    
    graph_clusterer = GraphClusterer(n_clusters=n_true_clusters)
    
    # Apply different graph clustering algorithms
    spectral_labels = graph_clusterer.spectral_clustering(graph, similarity_matrix)
    louvain_labels = graph_clusterer.louvain_clustering(graph)
    lp_labels = graph_clusterer.label_propagation(graph)
    gm_labels = graph_clusterer.greedy_modularity(graph)
    
    # =========================================================================
    # Step 6: Apply Traditional Clustering for Comparison
    # =========================================================================
    print_section_header("STEP 6: APPLY TRADITIONAL CLUSTERING")
    
    trad_clusterer = TraditionalClusterer(n_clusters=n_true_clusters)
    
    kmeans_labels = trad_clusterer.kmeans_clustering(embeddings)
    hierarchical_labels = trad_clusterer.hierarchical_clustering(embeddings, linkage='ward')
    
    # =========================================================================
    # Step 7: Compare and Evaluate All Methods
    # =========================================================================
    print_section_header("STEP 7: EVALUATION AND COMPARISON")
    
    all_results = {
        'Spectral (Graph)': spectral_labels,
        'Louvain (Graph)': louvain_labels,
        'Label Propagation (Graph)': lp_labels,
        'Greedy Modularity (Graph)': gm_labels,
        'K-Means (Traditional)': kmeans_labels,
        'Hierarchical (Traditional)': hierarchical_labels,
    }
    
    evaluator = ClusteringEvaluator()
    metrics = evaluator.compare_methods(all_results, true_labels, embeddings)
    
    # Find best method
    best_method = None
    best_nmi = -1
    for method_name, method_metrics in metrics.items():
        if 'NMI' in method_metrics and method_metrics['NMI'] > best_nmi:
            best_nmi = method_metrics['NMI']
            best_method = method_name
    
    print(f"\n🏆 Best performing method: {best_method} (NMI: {best_nmi:.4f})")
    
    # =========================================================================
    # Step 8: Visualization
    # =========================================================================
    print_section_header("STEP 8: VISUALIZATION")
    
    visualizer = ClusteringVisualizer(figsize=(16, 10))
    
    print("\nGenerating visualizations...")
    
    # Plot cluster size distributions
    print("1. Plotting cluster size distributions...")
    visualizer.plot_cluster_sizes({
        'Spectral': spectral_labels,
        'Louvain': louvain_labels,
        'K-Means': kmeans_labels,
    })
    
    # Plot 2D embedding visualization (t-SNE)
    print("2. Plotting 2D t-SNE visualization...")
    visualizer.plot_embeddings_2d(
        embeddings, 
        spectral_labels, 
        true_labels,
        method='tsne',
        title='Spectral Clustering (Graph-based)'
    )
    
    # Plot comparison with traditional method
    print("3. Plotting K-Means comparison...")
    visualizer.plot_embeddings_2d(
        embeddings,
        kmeans_labels,
        true_labels,
        method='tsne',
        title='K-Means Clustering (Traditional)'
    )
    
    # Plot graph network
    print("4. Plotting graph network structure...")
    visualizer.plot_graph_network(
        graph,
        spectral_labels,
        true_labels,
        layout='spring',
        max_nodes=200
    )
    
    # =========================================================================
    # Summary and Insights
    # =========================================================================
    print_section_header("SUMMARY AND INSIGHTS")
    
    print("\n📊 Key Findings:")
    print(f"1. Processed {len(texts)} news articles from {len(categories)} categories")
    print(f"2. Built similarity graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    print(f"3. Tested {len(all_results)} clustering methods (4 graph-based, 2 traditional)")
    print(f"4. Best method: {best_method} with NMI score of {best_nmi:.4f}")
    
    print("\n💡 Insights:")
    print("• Graph-based methods capture semantic relationships between documents")
    print("• Community detection algorithms (Louvain, Greedy Modularity) find natural clusters")
    print("• Spectral clustering leverages graph structure for better semantic grouping")
    print("• Traditional methods (K-Means) rely only on embedding distances")
    print("• Semantic embeddings provide better representation than keyword-based methods")
    
    print("\n✅ Demonstration completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
