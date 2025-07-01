# SimpleTextEmbedder Class

## Overview

The `SimpleTextEmbedder` class provides an easy-to-use interface for converting text into high-dimensional vector embeddings using state-of-the-art sentence transformer models. This class handles the complex process of text encoding, making semantic search accessible through simple method calls.

## Class Definition

```python
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder

class SimpleTextEmbedder:
    """
    Text-to-vector embedding using sentence transformers.
    
    Provides efficient text embedding with automatic model management,
    caching, and batch processing capabilities.
    """
```

## Key Features

- **State-of-the-Art Models**: Uses sentence transformer models optimized for semantic understanding
- **Automatic Model Management**: Downloads and caches models automatically
- **Batch Processing**: Efficient processing of multiple texts
- **GPU Acceleration**: Automatic GPU usage when available
- **Consistent Outputs**: Deterministic embeddings for the same input
- **Flexible Model Selection**: Support for various embedding models

## Constructor

```python
def __init__(
    self,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    cache_folder: Optional[str] = None
):
    """
    Initialize the text embedder.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to use ("cpu", "cuda", "auto")
        cache_folder: Custom cache directory for models
    """
```

### Model Selection

```python
# Default model - balanced performance and quality
embedder = SimpleTextEmbedder()

# High-quality model (larger, slower)
embedder = SimpleTextEmbedder("sentence-transformers/all-mpnet-base-v2")

# Fast model (smaller, faster)
embedder = SimpleTextEmbedder("sentence-transformers/all-MiniLM-L12-v2")

# Multilingual model
embedder = SimpleTextEmbedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Domain-specific models
embedder = SimpleTextEmbedder("sentence-transformers/all-distilroberta-v1")  # General
embedder = SimpleTextEmbedder("sentence-transformers/msmarco-distilbert-base-v4")  # Search
```

### Device Configuration

```python
# Automatic device selection (recommended)
embedder = SimpleTextEmbedder(device="auto")

# Force CPU usage
embedder = SimpleTextEmbedder(device="cpu")

# Force GPU usage (if available)
embedder = SimpleTextEmbedder(device="cuda")

# Specific GPU device
embedder = SimpleTextEmbedder(device="cuda:0")
```

## Core Methods

### Single Text Embedding

#### embed_text()

```python
def embed_text(self, text: str) -> List[float]:
    """
    Generate embedding for a single text.
    
    Args:
        text: Input text to embed
        
    Returns:
        List[float]: Vector embedding of the text
        
    Raises:
        EmbeddingError: If embedding generation fails
    """
```

**Usage Example:**
```python
# Simple text embedding
text = "Dogs are loyal companions who bring joy to our lives."
embedding = embedder.embed_text(text)

print(f"Text: {text}")
print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

# Output:
# Text: Dogs are loyal companions who bring joy to our lives.
# Embedding dimensions: 384
# First 5 values: [0.123, -0.456, 0.789, 0.321, -0.654]
```

### Batch Text Embedding

#### embed_batch()

```python
def embed_batch(
    self,
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = False
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts efficiently.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts to process at once
        show_progress: Whether to show progress bar
        
    Returns:
        List[List[float]]: List of vector embeddings
    """
```

**Usage Example:**
```python
# Batch embedding for efficiency
texts = [
    "Cats are independent creatures with mysterious personalities.",
    "Birds teach us about freedom and the beauty of flight.",
    "Elephants demonstrate wisdom and strong family bonds.",
    "Dolphins show intelligence and playful social behavior."
]

# Process all texts in batches
embeddings = embedder.embed_batch(texts, show_progress=True)

print(f"Processed {len(texts)} texts")
print(f"Each embedding has {len(embeddings[0])} dimensions")

# Use the embeddings
for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    print(f"Text {i+1}: {text[:50]}...")
    print(f"Embedding magnitude: {sum(x*x for x in embedding)**0.5:.3f}")
```

### Model Information

#### get_vector_size()

```python
def get_vector_size(self) -> int:
    """
    Get the dimensionality of the embeddings.
    
    Returns:
        int: Number of dimensions in the embedding vectors
    """
```

#### get_distance_metric()

```python
def get_distance_metric(self) -> str:
    """
    Get the recommended distance metric for this model.
    
    Returns:
        str: Distance metric ("Cosine", "Euclid", "Dot")
    """
```

#### get_model_info()

```python
def get_model_info(self) -> Dict[str, Any]:
    """
    Get comprehensive model information.
    
    Returns:
        dict: Model metadata including size, capabilities, performance
    """
```

**Usage Example:**
```python
# Get model specifications
print(f"Vector size: {embedder.get_vector_size()}")
print(f"Distance metric: {embedder.get_distance_metric()}")

model_info = embedder.get_model_info()
print(f"Model name: {model_info['name']}")
print(f"Max sequence length: {model_info['max_seq_length']}")
print(f"Model size: {model_info['model_size_mb']} MB")
```

## Advanced Features

### Text Preprocessing

```python
def embed_text_with_preprocessing(
    self,
    text: str,
    normalize: bool = True,
    remove_special_chars: bool = False,
    max_length: Optional[int] = None
) -> List[float]:
    """
    Embed text with preprocessing options.
    
    Args:
        normalize: Whether to normalize text casing and whitespace
        remove_special_chars: Remove special characters
        max_length: Truncate text to maximum length
    """
```

**Usage Example:**
```python
# Text with preprocessing
raw_text = "  Hello!!! This is a TEST with WEIRD formatting...  "

# Clean and embed
embedding = embedder.embed_text_with_preprocessing(
    raw_text,
    normalize=True,
    remove_special_chars=True,
    max_length=100
)
```

### Similarity Calculation

```python
def calculate_similarity(
    self,
    embedding1: List[float],
    embedding2: List[float],
    metric: str = "cosine"
) -> float:
    """
    Calculate similarity between two embeddings.
    
    Args:
        embedding1, embedding2: Vector embeddings to compare
        metric: Similarity metric ("cosine", "euclidean", "dot")
        
    Returns:
        float: Similarity score
    """
```

**Usage Example:**
```python
# Compare text similarity
text1 = "Dogs are loyal friends"
text2 = "Canines are faithful companions"
text3 = "I like pizza"

emb1 = embedder.embed_text(text1)
emb2 = embedder.embed_text(text2)
emb3 = embedder.embed_text(text3)

# Calculate similarities
sim_1_2 = embedder.calculate_similarity(emb1, emb2)
sim_1_3 = embedder.calculate_similarity(emb1, emb3)

print(f"Similarity between '{text1}' and '{text2}': {sim_1_2:.3f}")
print(f"Similarity between '{text1}' and '{text3}': {sim_1_3:.3f}")

# Output:
# Similarity between 'Dogs are loyal friends' and 'Canines are faithful companions': 0.847
# Similarity between 'Dogs are loyal friends' and 'I like pizza': 0.123
```

## Performance Optimization

### Caching

```python
# Enable embedding caching for repeated texts
embedder.enable_cache(max_size=1000)

# First call - computes embedding
embedding1 = embedder.embed_text("This text will be cached")

# Second call - retrieves from cache (much faster)
embedding2 = embedder.embed_text("This text will be cached")

# Cache statistics
cache_stats = embedder.get_cache_stats()
print(f"Cache hits: {cache_stats['hits']}")
print(f"Cache size: {cache_stats['size']}")
```

### Batch Size Optimization

```python
# Optimize batch size for your hardware
texts = ["Text {}".format(i) for i in range(1000)]

# Small batches for limited memory
embeddings = embedder.embed_batch(texts, batch_size=16)

# Large batches for high-memory systems
embeddings = embedder.embed_batch(texts, batch_size=128)

# Automatic batch size selection
embeddings = embedder.embed_batch(texts, batch_size="auto")
```

### Memory Management

```python
# For large-scale processing
def process_large_text_collection(texts, embedder):
    """Process large collections efficiently."""
    
    embeddings = []
    batch_size = 64
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedder.embed_batch(batch)
        embeddings.extend(batch_embeddings)
        
        # Clear GPU cache periodically
        if i % 1000 == 0:
            embedder.clear_gpu_cache()
    
    return embeddings
```

## Integration Examples

### With EmbeddedVectorDB

```python
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB

# Complete embedding and storage pipeline
embedder = SimpleTextEmbedder()
vector_db = EmbeddedVectorDB()

# Create collection with correct dimensions
collection_name = "documents"
vector_db.create_collection(
    collection_name,
    vector_size=embedder.get_vector_size(),
    distance=embedder.get_distance_metric()
)

# Embed and store documents
documents = ["Document 1", "Document 2", "Document 3"]
embeddings = embedder.embed_batch(documents)

# Store in vector database
# (see EmbeddedVectorDB documentation for storage details)
```

### With Search Operations

```python
from rag_to_riches.search.semantic_search import SemanticSearch

# Initialize search with embedder
search_engine = SemanticSearch(
    vector_db=vector_db,
    embedder=embedder,
    collection_name="documents"
)

# The search engine uses the embedder automatically
results = search_engine.search("query text")
```

### With Corpus Classes

```python
from rag_to_riches.corpus.animals import Animals

# Custom embedder for animals corpus
custom_embedder = SimpleTextEmbedder("sentence-transformers/all-mpnet-base-v2")

animals = Animals(
    vector_db,
    collection_name="animal_wisdom",
    embedder=custom_embedder
)

# The corpus uses the custom embedder for all operations
```

## Model Comparison

### Performance vs Quality Trade-offs

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 80MB | Fast | Good | General purpose, development |
| `all-MiniLM-L12-v2` | 120MB | Medium | Better | Balanced performance/quality |
| `all-mpnet-base-v2` | 420MB | Slow | Best | High-quality production |
| `all-distilroberta-v1` | 290MB | Medium | Very Good | General text understanding |

### Specialized Models

```python
# For specific domains
scientific_embedder = SimpleTextEmbedder("allenai/scibert_scivocab_uncased")
legal_embedder = SimpleTextEmbedder("nlpaueb/legal-bert-base-uncased")

# For multilingual content
multilingual_embedder = SimpleTextEmbedder(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# For code and technical content
code_embedder = SimpleTextEmbedder("microsoft/codebert-base")
```

## Error Handling

```python
from rag_to_riches.exceptions import EmbeddingError

try:
    embedding = embedder.embed_text("Some text")
except EmbeddingError as e:
    print(f"Embedding failed: {e}")
    # Handle the error (retry, fallback model, etc.)
```

## Configuration and Environment

### Environment Variables

```bash
# Set cache directory
export SENTENCE_TRANSFORMERS_HOME=/custom/cache/path

# Set device preference
export CUDA_VISIBLE_DEVICES=0

# Control memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Production Configuration

```python
# Production embedder setup
production_embedder = SimpleTextEmbedder(
    model_name="sentence-transformers/all-mpnet-base-v2",
    device="auto",  # Use best available device
    cache_folder="/data/model_cache"
)

# Enable optimizations
production_embedder.enable_cache(max_size=10000)
production_embedder.set_batch_size_auto()
```

## Troubleshooting

### Common Issues

1. **Model Download Issues**
   ```python
   # Manual model download
   try:
       embedder = SimpleTextEmbedder("model-name")
   except Exception as e:
       print(f"Download failed: {e}")
       # Use offline mode or alternative model
   ```

2. **Memory Issues**
   ```python
   # Reduce batch size for large models
   embedder = SimpleTextEmbedder(
       "sentence-transformers/all-mpnet-base-v2"
   )
   
   # Process in smaller batches
   embeddings = embedder.embed_batch(texts, batch_size=8)
   ```

3. **GPU Out of Memory**
   ```python
   # Force CPU usage
   embedder = SimpleTextEmbedder(device="cpu")
   
   # Or clear GPU cache
   embedder.clear_gpu_cache()
   ```

### Performance Monitoring

```python
import time

# Measure embedding performance
start_time = time.time()
embeddings = embedder.embed_batch(test_texts)
duration = time.time() - start_time

print(f"Processed {len(test_texts)} texts in {duration:.2f} seconds")
print(f"Rate: {len(test_texts)/duration:.1f} texts/second")
```

## Related Components

- [`EmbeddedVectorDB`](EmbeddedVectorDB.md): Vector storage component that uses embeddings
- [`SemanticSearch`](../search/index.md): High-level search interface using embeddings
- [`Animals`](../corpus/Animals.md): Domain-specific corpus that uses text embeddings
- [`Exceptions`](../exceptions/index.md): Error handling for embedding operations

---

*Part of the RAG to Riches framework - transforming text into intelligent vector representations.* 