#!/usr/bin/env python3
"""
Command-line interface for semantic news clustering
"""
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

from step1_data_loader import NewsDataLoader
from step2_preprocessor import TextPreprocessor
from step3_embeddings import SemanticEmbedder
from step4_similarity_graph import SimilarityGraph
from step5_graph_clustering import GraphClusterer
from step6_traditional_clustering import TraditionalClusterer, ClusteringEvaluator
from step7_visualization import ClusteringVisualizer
from sample_data_generator import generate_sample_news_data


def main():
    parser = argparse.ArgumentParser(
        description='Semantic News Clustering - Cluster news articles by meaning, not keywords',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with sample data (no internet required)
  python cli.py --sample --n-samples 100
  
  # Run with 20newsgroups dataset
  python cli.py --n-samples 500 --method spectral
  
  # Compare multiple methods
  python cli.py --n-samples 300 --compare
  
  # Save visualizations
  python cli.py --sample --n-samples 200 --visualize --output-dir ./results/
        """
    )
    
    # Data options
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--sample', action='store_true',
                           help='Use sample data generator (no internet required)')
    data_group.add_argument('--n-samples', type=int, default=500,
                           help='Number of documents to process (default: 500)')
    data_group.add_argument('--n-categories', type=int, default=5,
                           help='Number of categories for sample data (default: 5)')
    
    # Clustering options
    cluster_group = parser.add_argument_group('Clustering Options')
    cluster_group.add_argument('--method', type=str, default='spectral',
                              choices=['spectral', 'louvain', 'label_prop', 'greedy_mod', 'kmeans', 'hierarchical'],
                              help='Clustering method to use (default: spectral)')
    cluster_group.add_argument('--compare', action='store_true',
                              help='Compare multiple clustering methods')
    cluster_group.add_argument('--n-clusters', type=int, default=None,
                              help='Number of clusters (default: auto-detect from data)')
    
    # Graph options
    graph_group = parser.add_argument_group('Graph Options')
    graph_group.add_argument('--similarity-threshold', type=float, default=0.35,
                            help='Similarity threshold for graph edges (default: 0.35)')
    graph_group.add_argument('--top-k', type=int, default=10,
                            help='Number of top similar neighbors to keep (default: 10)')
    
    # Model options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                            help='Sentence transformer model (default: all-MiniLM-L6-v2)')
    
    # Visualization options
    viz_group = parser.add_argument_group('Visualization Options')
    viz_group.add_argument('--visualize', action='store_true',
                          help='Generate visualizations')
    viz_group.add_argument('--no-plot', action='store_true',
                          help='Do not show plots (only save)')
    viz_group.add_argument('--output-dir', type=str, default='.',
                          help='Directory to save visualizations (default: current directory)')
    
    # Other options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output messages')
    parser.add_argument('--version', action='version', version='Semantic News Clustering v1.0')
    
    args = parser.parse_args()
    
    # Validate
    if args.n_samples < 10:
        print("Error: --n-samples must be at least 10")
        sys.exit(1)
    
    # Main processing
    try:
        run_clustering(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_clustering(args):
    """Run the clustering pipeline with given arguments"""
    
    verbose = not args.quiet
    
    # Step 1: Load data
    if verbose:
        print("\n" + "="*70)
        print("STEP 1: Loading data...")
    
    if args.sample:
        if verbose:
            print("Using sample data generator...")
        texts, labels, categories = generate_sample_news_data(
            n_samples=args.n_samples,
            n_categories=args.n_categories
        )
    else:
        if verbose:
            print("Loading 20newsgroups dataset...")
        loader = NewsDataLoader(subset='train')
        texts, labels, categories = loader.load_data(num_samples=args.n_samples)
    
    if verbose:
        print(f"✓ Loaded {len(texts)} documents from {len(categories)} categories")
    
    # Step 2: Preprocess
    if verbose:
        print("\n" + "="*70)
        print("STEP 2: Preprocessing text...")
    
    preprocessor = TextPreprocessor()
    cleaned_texts = preprocessor.preprocess_batch(texts, verbose=verbose)
    
    if verbose:
        print(f"✓ Preprocessed {len(cleaned_texts)} documents")
    
    # Step 3: Generate embeddings
    if verbose:
        print("\n" + "="*70)
        print("STEP 3: Generating semantic embeddings...")
    
    embedder = SemanticEmbedder(model_name=args.model)
    embeddings = embedder.embed_texts(cleaned_texts, show_progress=verbose)
    similarity_matrix = embedder.compute_similarity(embeddings)
    
    if verbose:
        print(f"✓ Generated embeddings with shape {embeddings.shape}")
    
    # Step 4: Build graph
    if verbose:
        print("\n" + "="*70)
        print("STEP 4: Building similarity graph...")
    
    graph_builder = SimilarityGraph(
        similarity_threshold=args.similarity_threshold,
        top_k_neighbors=args.top_k
    )
    graph = graph_builder.build_graph(similarity_matrix, labels=labels)
    
    if verbose:
        stats = graph_builder.get_graph_stats()
        print(f"✓ Graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
    
    # Step 5: Clustering
    if verbose:
        print("\n" + "="*70)
        print("STEP 5: Applying clustering...")
    
    n_clusters = args.n_clusters or len(categories)
    
    results = {}
    
    if args.compare:
        # Compare multiple methods
        graph_clusterer = GraphClusterer(n_clusters=n_clusters)
        trad_clusterer = TraditionalClusterer(n_clusters=n_clusters)
        
        results['Spectral'] = graph_clusterer.spectral_clustering(graph, similarity_matrix)
        results['Louvain'] = graph_clusterer.louvain_clustering(graph)
        results['K-Means'] = trad_clusterer.kmeans_clustering(embeddings)
        results['Hierarchical'] = trad_clusterer.hierarchical_clustering(embeddings)
    else:
        # Single method
        if args.method in ['spectral', 'louvain', 'label_prop', 'greedy_mod']:
            clusterer = GraphClusterer(n_clusters=n_clusters)
            if args.method == 'spectral':
                cluster_labels = clusterer.spectral_clustering(graph, similarity_matrix)
            elif args.method == 'louvain':
                cluster_labels = clusterer.louvain_clustering(graph)
            elif args.method == 'label_prop':
                cluster_labels = clusterer.label_propagation(graph)
            else:  # greedy_mod
                cluster_labels = clusterer.greedy_modularity(graph)
        else:
            clusterer = TraditionalClusterer(n_clusters=n_clusters)
            if args.method == 'kmeans':
                cluster_labels = clusterer.kmeans_clustering(embeddings)
            else:  # hierarchical
                cluster_labels = clusterer.hierarchical_clustering(embeddings)
        
        results[args.method.capitalize()] = cluster_labels
    
    # Step 6: Evaluation
    if verbose:
        print("\n" + "="*70)
        print("STEP 6: Evaluation...")
    
    evaluator = ClusteringEvaluator()
    metrics = evaluator.compare_methods(results, labels, embeddings)
    
    # Step 7: Visualization
    if args.visualize:
        if verbose:
            print("\n" + "="*70)
            print("STEP 7: Generating visualizations...")
        
        visualizer = ClusteringVisualizer()
        
        # Get first result for visualization
        method_name = list(results.keys())[0]
        cluster_labels = results[method_name]
        
        import matplotlib.pyplot as plt
        if args.no_plot:
            plt.ioff()
        
        # t-SNE plot
        save_path = f"{args.output_dir}/tsne_{method_name.lower()}.png" if args.output_dir != '.' else None
        visualizer.plot_embeddings_2d(
            embeddings, cluster_labels, labels,
            method='tsne', title=f'{method_name} Clustering',
            save_path=save_path
        )
        
        # Network graph
        save_path = f"{args.output_dir}/network_{method_name.lower()}.png" if args.output_dir != '.' else None
        visualizer.plot_graph_network(
            graph, cluster_labels, labels,
            max_nodes=min(200, len(texts)),
            save_path=save_path
        )
        
        if verbose:
            print("✓ Visualizations generated")
    
    # Summary
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        best_nmi = -1
        best_method = None
        for method_name, method_metrics in metrics.items():
            if 'NMI' in method_metrics and method_metrics['NMI'] > best_nmi:
                best_nmi = method_metrics['NMI']
                best_method = method_name
        
        print(f"\n✅ Clustering completed successfully!")
        print(f"🏆 Best method: {best_method} (NMI: {best_nmi:.4f})")
        print("="*70)


if __name__ == "__main__":
    main()
