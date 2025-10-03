# Semantic News Clustering

A comprehensive implementation of semantic news article clustering using graph-based and traditional machine learning methods. This project groups news articles by **meaning (semantics)**, not just shared keywords, using state-of-the-art NLP techniques.

## 🎯 Overview

This project demonstrates how to cluster news articles based on their semantic similarity using:
- **Semantic embeddings** from pre-trained transformer models
- **Graph-based clustering** algorithms (Spectral, Louvain, Label Propagation)
- **Traditional clustering** methods (K-Means, Hierarchical) for comparison
- **Advanced visualizations** (t-SNE, network graphs, confusion matrices)

## 🚀 Features

- ✨ **Semantic Understanding**: Uses sentence transformers to capture deep semantic meaning
- 📊 **Multiple Algorithms**: Implements 6+ clustering algorithms for comparison
- 🔍 **Graph-Based Approach**: Builds similarity graphs to capture document relationships
- 📈 **Comprehensive Evaluation**: Includes metrics like ARI, NMI, V-Measure, Silhouette score
- 🎨 **Rich Visualizations**: t-SNE plots, network graphs, cluster size distributions
- 🧪 **Easy to Use**: Simple API and complete demonstration script

## 📋 Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/DkStine/semantic-news-clustering.git
cd semantic-news-clustering
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (if needed):
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 🎮 Quick Start

Run the complete demonstration:
```bash
python demo.py
```

This will:
1. Load the 20 newsgroups dataset
2. Preprocess the text
3. Generate semantic embeddings
4. Build a similarity graph
5. Apply graph-based clustering algorithms
6. Compare with traditional methods
7. Visualize and evaluate results

## 📚 Pipeline Steps

### Step 1: Data Collection
```python
from step1_data_loader import NewsDataLoader

loader = NewsDataLoader(subset='train')
texts, labels, categories = loader.load_data(num_samples=500)
```

### Step 2: Text Preprocessing
```python
from step2_preprocessor import TextPreprocessor

preprocessor = TextPreprocessor(remove_stopwords=True, lowercase=True)
cleaned_texts = preprocessor.preprocess_batch(texts)
```

### Step 3: Semantic Embeddings
```python
from step3_embeddings import SemanticEmbedder

embedder = SemanticEmbedder(model_name='all-MiniLM-L6-v2')
embeddings = embedder.embed_texts(cleaned_texts)
similarity_matrix = embedder.compute_similarity(embeddings)
```

### Step 4: Build Similarity Graph
```python
from step4_similarity_graph import SimilarityGraph

graph_builder = SimilarityGraph(similarity_threshold=0.35, top_k_neighbors=10)
graph = graph_builder.build_graph(similarity_matrix, labels=labels)
```

### Step 5: Graph-Based Clustering
```python
from step5_graph_clustering import GraphClusterer

clusterer = GraphClusterer(n_clusters=20)
spectral_labels = clusterer.spectral_clustering(graph, similarity_matrix)
louvain_labels = clusterer.louvain_clustering(graph)
```

### Step 6: Traditional Clustering
```python
from step6_traditional_clustering import TraditionalClusterer

trad_clusterer = TraditionalClusterer(n_clusters=20)
kmeans_labels = trad_clusterer.kmeans_clustering(embeddings)
hierarchical_labels = trad_clusterer.hierarchical_clustering(embeddings)
```

### Step 7: Visualization
```python
from step7_visualization import ClusteringVisualizer

visualizer = ClusteringVisualizer()
visualizer.plot_embeddings_2d(embeddings, spectral_labels, true_labels, method='tsne')
visualizer.plot_graph_network(graph, spectral_labels, true_labels)
```

### Step 8: Evaluation
```python
from step6_traditional_clustering import ClusteringEvaluator

evaluator = ClusteringEvaluator()
metrics = evaluator.compare_methods(all_results, true_labels, embeddings)
```

## 🧩 Project Structure

```
semantic-news-clustering/
├── README.md                           # This file
├── requirements.txt                     # Python dependencies
├── demo.py                             # Complete demonstration script
├── step1_data_loader.py                # Data loading module
├── step2_preprocessor.py               # Text preprocessing
├── step3_embeddings.py                 # Semantic embedding generation
├── step4_similarity_graph.py           # Graph construction
├── step5_graph_clustering.py           # Graph-based clustering algorithms
├── step6_traditional_clustering.py     # Traditional clustering & evaluation
└── step7_visualization.py              # Visualization tools
```

## 🔬 Algorithms Implemented

### Graph-Based Clustering
1. **Spectral Clustering**: Uses graph Laplacian eigenvalues
2. **Louvain Community Detection**: Optimizes modularity
3. **Label Propagation**: Propagates labels through the graph
4. **Greedy Modularity**: Maximizes network modularity

### Traditional Clustering
1. **K-Means**: Classic centroid-based clustering
2. **Hierarchical Clustering**: Agglomerative clustering with Ward linkage
3. **DBSCAN**: Density-based clustering

## 📊 Evaluation Metrics

- **Adjusted Rand Index (ARI)**: Similarity between clusterings
- **Normalized Mutual Information (NMI)**: Information-theoretic similarity
- **Homogeneity**: Each cluster contains only members of a single class
- **Completeness**: All members of a class are in the same cluster
- **V-Measure**: Harmonic mean of homogeneity and completeness
- **Silhouette Score**: How similar objects are to their own cluster

## 🎨 Visualizations

The project generates several types of visualizations:

1. **t-SNE/PCA Plots**: 2D visualization of high-dimensional embeddings
2. **Network Graphs**: Graph structure with community detection
3. **Cluster Size Distributions**: Bar charts showing cluster sizes
4. **Confusion Matrices**: Comparison between true and predicted labels

## 🔍 Example Results

When running on 500 documents from the 20 newsgroups dataset:

```
Graph statistics:
  num_nodes: 500
  num_edges: ~2500
  density: 0.02
  num_components: 1-3

Clustering Performance (NMI scores):
  Spectral (Graph):         0.65-0.75
  Louvain (Graph):          0.60-0.70
  K-Means (Traditional):    0.55-0.65
  Hierarchical:             0.50-0.60
```

Graph-based methods typically outperform traditional methods by capturing semantic relationships through the graph structure.

## 🛠️ Customization

### Using Different Datasets

Replace the data loader with your own:
```python
# Your custom data
texts = ["article 1...", "article 2...", ...]
labels = [0, 1, 0, 2, ...]  # Optional, for evaluation

# Continue with preprocessing and embedding
preprocessor = TextPreprocessor()
cleaned_texts = preprocessor.preprocess_batch(texts)
# ... rest of the pipeline
```

### Tuning Parameters

Key parameters to adjust:
- `similarity_threshold`: Controls edge creation in graph (0.2-0.5)
- `top_k_neighbors`: Number of neighbors to keep per node (5-20)
- `n_clusters`: Number of clusters for algorithms that require it
- `model_name`: Sentence transformer model ('all-MiniLM-L6-v2', 'all-mpnet-base-v2')

## 📖 How It Works

1. **Semantic Embeddings**: Instead of keyword matching, we use pre-trained transformer models to convert text into dense vector representations that capture semantic meaning

2. **Similarity Graph**: We build a graph where nodes are documents and edges connect semantically similar documents (based on cosine similarity)

3. **Graph Clustering**: We apply community detection algorithms that find groups of densely connected nodes in the graph

4. **Comparison**: We compare graph-based methods with traditional methods that only use embedding distances

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **scikit-learn**: For the 20 newsgroups dataset and traditional clustering algorithms
- **sentence-transformers**: For semantic embeddings
- **NetworkX**: For graph algorithms
- **python-louvain**: For Louvain community detection

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Clustering! 🎉**