"""
Test script to verify the clustering pipeline works correctly
Uses sample data instead of downloading from internet
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sample_data_generator import generate_sample_news_data
from step2_preprocessor import TextPreprocessor
from step3_embeddings import SemanticEmbedder
from step4_similarity_graph import SimilarityGraph
from step5_graph_clustering import GraphClusterer
from step6_traditional_clustering import TraditionalClusterer, ClusteringEvaluator


def test_pipeline():
    """Test the complete clustering pipeline"""
    
    print("="*70)
    print("TESTING SEMANTIC NEWS CLUSTERING PIPELINE")
    print("="*70)
    
    # Generate sample data
    print("\n1. Generating sample news data...")
    texts, true_labels, categories = generate_sample_news_data(
        n_samples=50, 
        n_categories=5
    )
    print(f"✓ Generated {len(texts)} documents from {len(categories)} categories")
    print(f"  Categories: {', '.join(categories)}")
    
    # Preprocess
    print("\n2. Preprocessing text...")
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        lowercase=True,
        min_token_length=2
    )
    cleaned_texts = preprocessor.preprocess_batch(texts, verbose=False)
    print(f"✓ Preprocessed {len(cleaned_texts)} documents")
    
    # Generate embeddings
    print("\n3. Generating semantic embeddings...")
    embedder = SemanticEmbedder(model_name='all-MiniLM-L6-v2')
    embeddings = embedder.embed_texts(cleaned_texts, batch_size=16, show_progress=False)
    similarity_matrix = embedder.compute_similarity(embeddings)
    print(f"✓ Generated embeddings with shape {embeddings.shape}")
    print(f"  Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
    
    # Build graph
    print("\n4. Building similarity graph...")
    graph_builder = SimilarityGraph(
        similarity_threshold=0.3,
        top_k_neighbors=8
    )
    graph = graph_builder.build_graph(similarity_matrix, labels=true_labels)
    stats = graph_builder.get_graph_stats()
    print(f"✓ Graph created:")
    print(f"  Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
    print(f"  Density: {stats['density']:.4f}")
    
    # Apply graph clustering
    print("\n5. Applying graph-based clustering...")
    n_clusters = len(categories)
    graph_clusterer = GraphClusterer(n_clusters=n_clusters)
    
    spectral_labels = graph_clusterer.spectral_clustering(graph, similarity_matrix)
    louvain_labels = graph_clusterer.louvain_clustering(graph)
    print(f"✓ Applied 2 graph-based clustering algorithms")
    
    # Apply traditional clustering
    print("\n6. Applying traditional clustering...")
    trad_clusterer = TraditionalClusterer(n_clusters=n_clusters)
    kmeans_labels = trad_clusterer.kmeans_clustering(embeddings)
    print(f"✓ Applied 1 traditional clustering algorithm")
    
    # Evaluate
    print("\n7. Evaluating clustering results...")
    all_results = {
        'Spectral (Graph)': spectral_labels,
        'Louvain (Graph)': louvain_labels,
        'K-Means (Traditional)': kmeans_labels,
    }
    
    evaluator = ClusteringEvaluator()
    metrics = evaluator.compare_methods(all_results, true_labels, embeddings)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    best_nmi = -1
    best_method = None
    for method_name, method_metrics in metrics.items():
        if 'NMI' in method_metrics and method_metrics['NMI'] > best_nmi:
            best_nmi = method_metrics['NMI']
            best_method = method_name
    
    print(f"\n✅ Pipeline test completed successfully!")
    print(f"🏆 Best method: {best_method} (NMI: {best_nmi:.4f})")
    print(f"\n📊 All components working correctly:")
    print(f"   ✓ Data loading/generation")
    print(f"   ✓ Text preprocessing")
    print(f"   ✓ Semantic embeddings")
    print(f"   ✓ Graph construction")
    print(f"   ✓ Graph-based clustering")
    print(f"   ✓ Traditional clustering")
    print(f"   ✓ Evaluation metrics")
    print("="*70)


if __name__ == "__main__":
    test_pipeline()
