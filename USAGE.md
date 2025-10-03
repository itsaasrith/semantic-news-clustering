# Usage Guide

This guide shows you how to use the semantic news clustering system.

## Quick Start (5 minutes)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/DkStine/semantic-news-clustering.git
cd semantic-news-clustering

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Demo

The easiest way to get started is to run the complete demonstration:

```bash
python demo.py
```

This will:
- Load 500 news articles from the 20newsgroups dataset
- Preprocess the text
- Generate semantic embeddings
- Build a similarity graph
- Apply 4 graph-based clustering algorithms
- Apply 2 traditional clustering algorithms for comparison
- Generate visualizations
- Display evaluation metrics

**Expected output:**
- Console output showing progress through each step
- Multiple visualization windows (t-SNE plots, network graphs, cluster distributions)
- Evaluation metrics comparing all methods (ARI, NMI, V-Measure, etc.)

## Step-by-Step Usage

### Example 1: Basic Clustering

```python
from step1_data_loader import NewsDataLoader
from step2_preprocessor import TextPreprocessor
from step3_embeddings import SemanticEmbedder
from step5_graph_clustering import GraphClusterer
from step4_similarity_graph import SimilarityGraph

# Load data
loader = NewsDataLoader(subset='train')
texts, labels, categories = loader.load_data(num_samples=200)

# Preprocess
preprocessor = TextPreprocessor()
cleaned_texts = preprocessor.preprocess_batch(texts)

# Generate embeddings
embedder = SemanticEmbedder(model_name='all-MiniLM-L6-v2')
embeddings = embedder.embed_texts(cleaned_texts)
similarity_matrix = embedder.compute_similarity(embeddings)

# Build graph and cluster
graph_builder = SimilarityGraph(similarity_threshold=0.35)
graph = graph_builder.build_graph(similarity_matrix)

clusterer = GraphClusterer(n_clusters=20)
cluster_labels = clusterer.spectral_clustering(graph, similarity_matrix)

print(f"Found {len(set(cluster_labels))} clusters")
```

### Example 2: Using Your Own Data

```python
from step2_preprocessor import TextPreprocessor
from step3_embeddings import SemanticEmbedder
from step4_similarity_graph import SimilarityGraph
from step5_graph_clustering import GraphClusterer

# Your own news articles
texts = [
    "Apple releases new iPhone with advanced AI features...",
    "Scientists discover new planet in distant solar system...",
    "Local football team wins championship game...",
    # ... more articles
]

# Preprocess
preprocessor = TextPreprocessor(
    remove_stopwords=True,
    lowercase=True,
    min_token_length=2
)
cleaned_texts = preprocessor.preprocess_batch(texts)

# Generate embeddings
embedder = SemanticEmbedder(model_name='all-MiniLM-L6-v2')
embeddings = embedder.embed_texts(cleaned_texts)
similarity_matrix = embedder.compute_similarity(embeddings)

# Build graph and cluster
graph_builder = SimilarityGraph(
    similarity_threshold=0.3,
    top_k_neighbors=10
)
graph = graph_builder.build_graph(similarity_matrix)

# Apply clustering
clusterer = GraphClusterer(n_clusters=5)  # Adjust based on your data
labels = clusterer.louvain_clustering(graph)

# Check cluster assignments
for i, (text, label) in enumerate(zip(texts[:5], labels[:5])):
    print(f"Article {i} -> Cluster {label}: {text[:50]}...")
```

### Example 3: Comparison with Traditional Methods

```python
from step6_traditional_clustering import TraditionalClusterer, ClusteringEvaluator

# ... (after loading and embedding data as above)

# Apply graph-based clustering
graph_clusterer = GraphClusterer(n_clusters=10)
spectral_labels = graph_clusterer.spectral_clustering(graph, similarity_matrix)
louvain_labels = graph_clusterer.louvain_clustering(graph)

# Apply traditional clustering
trad_clusterer = TraditionalClusterer(n_clusters=10)
kmeans_labels = trad_clusterer.kmeans_clustering(embeddings)
hierarchical_labels = trad_clusterer.hierarchical_clustering(embeddings)

# Compare all methods
all_results = {
    'Spectral': spectral_labels,
    'Louvain': louvain_labels,
    'K-Means': kmeans_labels,
    'Hierarchical': hierarchical_labels,
}

# Evaluate (if you have ground truth labels)
evaluator = ClusteringEvaluator()
metrics = evaluator.compare_methods(all_results, true_labels, embeddings)
```

### Example 4: Visualization

```python
from step7_visualization import ClusteringVisualizer

# ... (after clustering)

visualizer = ClusteringVisualizer(figsize=(16, 10))

# Plot 2D t-SNE visualization
visualizer.plot_embeddings_2d(
    embeddings, 
    cluster_labels,
    true_labels=labels,  # optional
    method='tsne',
    title='Semantic News Clustering',
    save_path='clustering_tsne.png'  # optional
)

# Plot network graph
visualizer.plot_graph_network(
    graph,
    cluster_labels,
    true_labels=labels,  # optional
    layout='spring',
    max_nodes=200,
    save_path='network_graph.png'  # optional
)

# Plot cluster size distribution
visualizer.plot_cluster_sizes(
    {'Spectral': spectral_labels, 'K-Means': kmeans_labels},
    save_path='cluster_sizes.png'  # optional
)
```

## Parameter Tuning Guide

### Preprocessing Parameters

```python
TextPreprocessor(
    remove_stopwords=True,   # Remove common words like "the", "a", "is"
    lowercase=True,           # Convert all text to lowercase
    min_token_length=2        # Minimum word length to keep
)
```

### Embedding Model Selection

```python
# Fast, good performance (recommended for most cases)
embedder = SemanticEmbedder(model_name='all-MiniLM-L6-v2')

# Better quality, slower
embedder = SemanticEmbedder(model_name='all-mpnet-base-v2')

# Specialized for paraphrase detection
embedder = SemanticEmbedder(model_name='paraphrase-MiniLM-L6-v2')
```

### Graph Construction Parameters

```python
SimilarityGraph(
    similarity_threshold=0.35,  # 0.2-0.5 typical range
    top_k_neighbors=10          # 5-20 typical range
)
```

**Guidelines:**
- **Higher threshold** (0.4-0.5): Fewer, stronger connections; more isolated clusters
- **Lower threshold** (0.2-0.3): More connections; larger, merged clusters
- **More neighbors** (15-20): Denser graph; better connectivity
- **Fewer neighbors** (5-10): Sparser graph; clearer cluster boundaries

### Clustering Algorithm Selection

**When to use each algorithm:**

1. **Spectral Clustering**: 
   - Best when you know the number of clusters
   - Good for balanced cluster sizes
   - Computationally expensive for large datasets

2. **Louvain Community Detection**:
   - Best for discovering natural communities
   - No need to specify number of clusters
   - Fast, scales well to large graphs

3. **Label Propagation**:
   - Very fast
   - Good for large datasets
   - May produce unbalanced clusters

4. **K-Means** (traditional):
   - Baseline comparison
   - Fast, scalable
   - Works only on embedding distances, not graph structure

## Common Use Cases

### Use Case 1: News Article Organization

Automatically organize news articles into topic-based clusters:

```python
# Load your news articles
texts = load_your_news_articles()

# Process and cluster
# ... (see Example 2)

# Use Louvain for automatic topic discovery
labels = clusterer.louvain_clustering(graph)

# Assign articles to topics
topics = {}
for i, label in enumerate(labels):
    if label not in topics:
        topics[label] = []
    topics[label].append(texts[i])

# Show articles by topic
for topic_id, articles in topics.items():
    print(f"\nTopic {topic_id}:")
    for article in articles[:3]:  # Show first 3
        print(f"  - {article[:100]}...")
```

### Use Case 2: Duplicate Detection

Find similar or duplicate articles:

```python
# Build graph with high similarity threshold
graph_builder = SimilarityGraph(similarity_threshold=0.7)
graph = graph_builder.build_graph(similarity_matrix, texts=texts)

# Find connected components (groups of similar articles)
import networkx as nx
components = list(nx.connected_components(graph))

# Show potential duplicates
for component in components:
    if len(component) > 1:
        print(f"\nPotential duplicates:")
        for node_id in component:
            print(f"  - {texts[node_id][:100]}...")
```

### Use Case 3: Content Recommendation

Find related articles for recommendations:

```python
# Get embeddings
embeddings = embedder.embed_texts(cleaned_texts)

# Function to find similar articles
def find_similar(article_index, top_n=5):
    # Compute similarities
    query_embedding = embeddings[article_index:article_index+1]
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    
    # Get top similar (excluding the article itself)
    similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    
    print(f"Articles similar to: {texts[article_index][:100]}...\n")
    for idx in similar_indices:
        print(f"  Similarity: {similarities[idx]:.3f}")
        print(f"  {texts[idx][:100]}...\n")

# Example: find articles similar to article 0
find_similar(0, top_n=5)
```

## Troubleshooting

### Issue: Out of Memory

**Solution:** Process data in smaller batches

```python
# Reduce number of samples
texts, labels, _ = loader.load_data(num_samples=100)

# Reduce embedding batch size
embeddings = embedder.embed_texts(cleaned_texts, batch_size=16)

# Use sparser graph
graph_builder = SimilarityGraph(
    similarity_threshold=0.5,  # Higher threshold
    top_k_neighbors=5          # Fewer neighbors
)
```

### Issue: Slow Performance

**Solution:** Use faster alternatives

```python
# Use faster embedding model
embedder = SemanticEmbedder(model_name='all-MiniLM-L6-v2')

# Use faster clustering algorithm
labels = clusterer.louvain_clustering(graph)  # Instead of spectral

# Limit graph size
graph_builder = SimilarityGraph(
    similarity_threshold=0.4,
    top_k_neighbors=8
)
```

### Issue: Poor Clustering Quality

**Solution:** Tune parameters

```python
# Try different similarity thresholds
for threshold in [0.3, 0.35, 0.4, 0.45]:
    graph_builder = SimilarityGraph(similarity_threshold=threshold)
    graph = graph_builder.build_graph(similarity_matrix)
    labels = clusterer.louvain_clustering(graph)
    # Evaluate...

# Try different number of clusters
for n_clusters in [5, 10, 15, 20]:
    clusterer = GraphClusterer(n_clusters=n_clusters)
    labels = clusterer.spectral_clustering(graph, similarity_matrix)
    # Evaluate...
```

## Best Practices

1. **Data Preprocessing**: Always clean your text data for better embedding quality
2. **Model Selection**: Use 'all-MiniLM-L6-v2' for speed, 'all-mpnet-base-v2' for quality
3. **Graph Tuning**: Start with threshold=0.35 and k=10, then adjust based on results
4. **Algorithm Choice**: Use Louvain for discovery, Spectral when you know cluster count
5. **Evaluation**: Always compare multiple methods to find the best for your data
6. **Visualization**: Use t-SNE plots to visually verify clustering quality

## Advanced Topics

### Custom Similarity Metrics

You can modify the similarity computation:

```python
# In step3_embeddings.py, modify compute_similarity method
def compute_similarity(self, embeddings, metric='cosine'):
    if metric == 'cosine':
        # Normalized dot product
        return np.dot(embeddings, embeddings.T)
    elif metric == 'euclidean':
        # Euclidean distance converted to similarity
        from scipy.spatial.distance import cdist
        distances = cdist(embeddings, embeddings, metric='euclidean')
        return 1 / (1 + distances)
```

### Incremental Clustering

For continuously incoming articles:

```python
# Initial clustering
initial_labels = clusterer.louvain_clustering(graph)

# New article arrives
new_text = "Breaking news article..."
new_cleaned = preprocessor.clean_text(new_text)
new_embedding = embedder.embed_texts([new_cleaned])

# Find most similar cluster
similarities = np.dot(embeddings, new_embedding.T).flatten()
assigned_cluster = initial_labels[np.argmax(similarities)]
```

## Support

For questions or issues:
1. Check this guide
2. Read the code comments in each module
3. Open an issue on GitHub

Happy clustering! 🎉
