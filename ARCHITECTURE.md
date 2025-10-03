# Project Structure and Architecture

This document explains the architecture and design of the semantic news clustering system.

## Overview

The system follows a modular pipeline architecture where each step is encapsulated in its own module. This design allows for:
- Easy testing and debugging of individual components
- Flexibility to swap implementations
- Clear separation of concerns
- Reusability of components

## Directory Structure

```
semantic-news-clustering/
├── README.md                       # Main project documentation
├── USAGE.md                        # Detailed usage guide
├── EXAMPLE_OUTPUT.md               # Expected outputs and examples
├── ARCHITECTURE.md                 # This file
├── requirements.txt                # Python dependencies
│
├── demo.py                         # Complete demonstration script
├── cli.py                          # Command-line interface
├── test_pipeline.py                # Test script with sample data
├── quickstart.ipynb                # Jupyter notebook tutorial
│
├── step1_data_loader.py            # Data loading module
├── step2_preprocessor.py           # Text preprocessing
├── step3_embeddings.py             # Semantic embedding generation
├── step4_similarity_graph.py       # Graph construction
├── step5_graph_clustering.py       # Graph-based clustering
├── step6_traditional_clustering.py # Traditional methods & evaluation
├── step7_visualization.py          # Visualization tools
└── sample_data_generator.py        # Sample data for testing
```

## Module Descriptions

### Core Pipeline Modules

#### 1. `step1_data_loader.py` - Data Loading
**Purpose**: Load news datasets for clustering

**Key Components**:
- `NewsDataLoader`: Class for loading 20newsgroups dataset
  - Methods: `load_data()`, `get_sample_docs()`
  - Configurable: subset, categories, filtering

**Design Decisions**:
- Uses sklearn's built-in dataset for convenience
- Supports filtering (headers, footers, quotes) for cleaner text
- Returns texts, labels, and category names separately

**Usage**:
```python
loader = NewsDataLoader(subset='train')
texts, labels, categories = loader.load_data(num_samples=500)
```

#### 2. `step2_preprocessor.py` - Text Preprocessing
**Purpose**: Clean and normalize text for better embedding quality

**Key Components**:
- `TextPreprocessor`: Handles text cleaning
  - Removes URLs, emails, numbers
  - Lowercasing, stopword removal
  - Tokenization and filtering

**Design Decisions**:
- Uses NLTK for robust tokenization
- Configurable preprocessing steps
- Batch processing for efficiency
- Auto-downloads required NLTK data

**Pipeline**:
```
Raw Text → URL/Email Removal → Number Removal → Lowercasing → 
Punctuation Removal → Tokenization → Stopword Filtering → Clean Text
```

#### 3. `step3_embeddings.py` - Semantic Embeddings
**Purpose**: Convert text to semantic vector representations

**Key Components**:
- `SemanticEmbedder`: Generates embeddings using transformers
  - Uses sentence-transformers library
  - Supports multiple pre-trained models
  - Computes similarity matrices

**Design Decisions**:
- Uses pre-trained transformer models (no training needed)
- L2 normalization for cosine similarity
- Batch processing for GPU efficiency
- Default model: all-MiniLM-L6-v2 (good balance of speed/quality)

**Model Options**:
- `all-MiniLM-L6-v2`: Fast, 384 dimensions (default)
- `all-mpnet-base-v2`: Better quality, 768 dimensions
- `paraphrase-MiniLM-L6-v2`: Specialized for paraphrases

#### 4. `step4_similarity_graph.py` - Graph Construction
**Purpose**: Build a graph where edges represent semantic similarity

**Key Components**:
- `SimilarityGraph`: Constructs and manages the graph
  - Uses NetworkX for graph representation
  - Threshold-based edge creation
  - Top-k neighbor selection

**Design Decisions**:
- Undirected weighted graph
- Edges only above similarity threshold
- Optional top-k filtering to control graph density
- Stores graph statistics for analysis

**Graph Properties**:
- Nodes: Documents
- Edges: Semantic similarity (cosine)
- Weights: Similarity scores
- Optional node attributes: true_label, text snippet

#### 5. `step5_graph_clustering.py` - Graph-based Clustering
**Purpose**: Apply graph algorithms to find document clusters

**Key Components**:
- `GraphClusterer`: Implements multiple graph clustering algorithms
  - Spectral Clustering
  - Louvain Community Detection
  - Label Propagation
  - Greedy Modularity

**Design Decisions**:
- Multiple algorithms for comparison
- Some require k (number of clusters), others discover automatically
- Leverages NetworkX and scikit-learn implementations
- Stores all results for comparison

**Algorithm Comparison**:
| Algorithm | Requires k? | Speed | Quality | Best For |
|-----------|-------------|-------|---------|----------|
| Spectral | Yes | Slow | High | Known k |
| Louvain | No | Fast | High | Discovery |
| Label Propagation | No | Very Fast | Medium | Large graphs |
| Greedy Modularity | No | Medium | Medium | Modularity optimization |

#### 6. `step6_traditional_clustering.py` - Traditional Methods & Evaluation
**Purpose**: Apply traditional clustering for comparison and evaluate all methods

**Key Components**:
- `TraditionalClusterer`: Implements traditional algorithms
  - K-Means
  - Hierarchical Clustering
  - DBSCAN
- `ClusteringEvaluator`: Computes evaluation metrics
  - ARI, NMI, Homogeneity, Completeness, V-Measure
  - Silhouette score

**Design Decisions**:
- Traditional methods work on embeddings only (not graph)
- Comprehensive evaluation suite
- Comparison framework for all methods
- Filtering of noise points (DBSCAN)

**Evaluation Metrics**:
- **ARI**: Similarity to ground truth (adjusted for chance)
- **NMI**: Information-theoretic measure
- **Homogeneity**: Cluster purity
- **Completeness**: Class coverage
- **V-Measure**: Harmonic mean of homogeneity & completeness
- **Silhouette**: Cluster cohesion & separation

#### 7. `step7_visualization.py` - Visualization
**Purpose**: Visualize clustering results for analysis

**Key Components**:
- `ClusteringVisualizer`: Creates multiple visualization types
  - 2D embeddings (t-SNE/PCA)
  - Network graphs
  - Cluster size distributions
  - Confusion matrices

**Design Decisions**:
- Uses matplotlib, seaborn for plots
- NetworkX for graph visualization
- Dimensionality reduction for 2D projections
- Configurable figure sizes and layouts
- Optional saving to files

**Visualization Types**:
1. **t-SNE/PCA plots**: Show semantic structure in 2D
2. **Network graphs**: Display graph connectivity
3. **Cluster distributions**: Compare cluster sizes
4. **Confusion matrices**: Compare true vs predicted

### Supporting Modules

#### `sample_data_generator.py` - Sample Data Generation
**Purpose**: Generate synthetic news data for testing without internet

**Key Components**:
- `generate_sample_news_data()`: Creates realistic sample articles
  - Uses category-specific keywords
  - Template-based sentence generation
  - Controllable sample size and categories

**Design Decisions**:
- Simple keyword-based generation
- Fast and deterministic (no downloads)
- Good for testing and development
- 5 default categories: Technology, Sports, Politics, Science, Business

#### `demo.py` - Complete Demonstration
**Purpose**: End-to-end demonstration of the entire pipeline

**Design**:
- Follows all 8 steps sequentially
- Comprehensive output and logging
- Multiple clustering algorithms
- Full evaluation and visualization
- Configurable parameters at top

#### `cli.py` - Command-line Interface
**Purpose**: Provide easy command-line access to the system

**Features**:
- Argument parsing for all major options
- Multiple modes: single method, comparison, visualization
- Sample data or real dataset
- Output directory for saving results
- Quiet mode for automation

#### `test_pipeline.py` - Testing Script
**Purpose**: Quick validation that all components work

**Design**:
- Uses sample data (no internet required)
- Tests all major components
- Minimal output
- Returns success/failure status

#### `quickstart.ipynb` - Jupyter Notebook
**Purpose**: Interactive tutorial and exploration

**Structure**:
1. Setup and imports
2. Data loading
3. Preprocessing
4. Embeddings
5. Graph building
6. Clustering
7. Evaluation
8. Visualization
9. Analysis examples
10. Summary

## Data Flow

```
┌─────────────────┐
│  Raw Text Data  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │
└────────┬────────┘
         │
         ├───────────────┐
         ▼               ▼
┌──────────────┐  ┌─────────────────┐
│  Similarity  │  │   Traditional   │
│    Graph     │  │   Clustering    │
└──────┬───────┘  └────────┬────────┘
       │                   │
       ▼                   │
┌─────────────┐            │
│  Graph      │            │
│  Clustering │            │
└──────┬──────┘            │
       │                   │
       └──────────┬────────┘
                  ▼
         ┌────────────────┐
         │   Evaluation   │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │ Visualization  │
         └────────────────┘
```

## Key Design Patterns

### 1. Pipeline Pattern
Each module is a step in the pipeline, with clear inputs and outputs.

### 2. Strategy Pattern
Multiple clustering algorithms implement the same interface, allowing easy comparison.

### 3. Builder Pattern
Graph construction uses configurable parameters to build complex structures.

### 4. Facade Pattern
High-level scripts (demo, CLI) provide simple interfaces to complex subsystems.

## Extension Points

### Adding New Clustering Algorithms

1. **Graph-based**: Add method to `GraphClusterer` class
```python
def new_algorithm(self, graph: nx.Graph) -> np.ndarray:
    # Implementation
    labels = ...
    return labels
```

2. **Traditional**: Add method to `TraditionalClusterer` class
```python
def new_method(self, embeddings: np.ndarray) -> np.ndarray:
    # Implementation
    labels = ...
    return labels
```

### Adding New Visualizations

Add method to `ClusteringVisualizer` class:
```python
def plot_new_viz(self, data, labels, **kwargs):
    # Create visualization
    plt.figure()
    # ... plotting code
    plt.show()
```

### Adding New Datasets

Create new loader class following same interface:
```python
class CustomDataLoader:
    def load_data(self, **kwargs) -> Tuple[List[str], np.ndarray, List[str]]:
        # Load your data
        return texts, labels, categories
```

### Adding New Preprocessing Steps

Extend `TextPreprocessor`:
```python
def custom_preprocessing(self, text: str) -> str:
    # Add custom logic
    return processed_text
```

## Performance Considerations

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Preprocessing | O(n*m) | n=docs, m=avg length |
| Embedding | O(n*m) | Batched, GPU-accelerated |
| Similarity matrix | O(n²*d) | n=docs, d=dimensions |
| Graph construction | O(n²) | With threshold/top-k filtering |
| Spectral clustering | O(n³) | Eigendecomposition |
| Louvain | O(n*log(n)) | Approximate |
| K-Means | O(n*k*i) | k=clusters, i=iterations |
| t-SNE | O(n²) | Dimensionality reduction |

### Memory Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Raw texts | ~1KB/doc | Variable |
| Embeddings | 4*n*d bytes | d=384 for default model |
| Similarity matrix | 4*n² bytes | Symmetric |
| Graph (sparse) | ~8*e bytes | e=number of edges |

**Example**: 1000 documents with default settings:
- Embeddings: ~1.5 MB
- Similarity matrix: ~4 MB
- Graph: ~0.5-2 MB (depends on threshold)
- Total: ~6-8 MB

### Optimization Strategies

1. **Batch Processing**: Process large datasets in chunks
2. **GPU Acceleration**: Use GPU for embedding generation
3. **Sparse Matrices**: Use scipy.sparse for large similarity matrices
4. **Graph Pruning**: Use top-k neighbors to limit graph size
5. **Parallel Processing**: Use joblib for parallel preprocessing

## Testing Strategy

### Unit Tests (Future Work)
- Test each module independently
- Mock external dependencies
- Test edge cases and error conditions

### Integration Tests
- `test_pipeline.py`: End-to-end pipeline test
- Uses sample data for reproducibility
- Validates all components work together

### Performance Tests
- Benchmark on different dataset sizes
- Profile memory usage
- Identify bottlenecks

## Future Enhancements

### Potential Improvements

1. **Incremental Clustering**: Add new documents without recomputing
2. **Hierarchical Visualization**: Show cluster hierarchies
3. **Interactive Visualization**: Use Plotly for interactive plots
4. **Custom Embeddings**: Support fine-tuning on domain-specific data
5. **Streaming Processing**: Handle continuous data streams
6. **Distributed Processing**: Scale to millions of documents
7. **API Server**: REST API for clustering service
8. **Web Interface**: User-friendly web UI

### Advanced Features

1. **Multi-modal Clustering**: Combine text, images, metadata
2. **Temporal Analysis**: Track cluster evolution over time
3. **Cross-lingual Clustering**: Multi-language support
4. **Active Learning**: Human-in-the-loop cluster refinement
5. **Explainability**: Show why documents are clustered together

## Best Practices

### Code Quality
- Type hints for all function parameters
- Comprehensive docstrings
- Consistent naming conventions
- Error handling and validation
- Progress indicators for long operations

### Configuration
- Sensible defaults
- Clear parameter documentation
- Easy customization
- Validation of parameter combinations

### Documentation
- README for quick start
- USAGE for detailed examples
- EXAMPLE_OUTPUT for expected results
- ARCHITECTURE (this file) for understanding design

### Reproducibility
- Fixed random seeds (seed=42)
- Version pinning in requirements.txt
- Deterministic algorithms where possible
- Sample data for testing

## Conclusion

The semantic news clustering system is designed to be:
- **Modular**: Easy to understand and modify
- **Flexible**: Supports multiple algorithms and configurations
- **Extensible**: Simple to add new features
- **User-friendly**: Multiple interfaces (demo, CLI, notebook)
- **Well-documented**: Comprehensive documentation at all levels

This architecture supports both research/experimentation and production use cases.
