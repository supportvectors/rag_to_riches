# SimpleTextEmbedder

Usage Documentation: [SimpleTextEmbedder Class Guide](../vectordb/SimpleTextEmbedder.md)

Text-only embedder using sentence transformers for semantic text embeddings.

This class provides an easy-to-use interface for converting text into high-dimensional vector embeddings using state-of-the-art sentence transformer models. It handles the complex process of text encoding, making semantic search accessible through simple method calls.

## rag_to_riches.vectordb.embedder.SimpleTextEmbedder

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `model_name` | `str` | HuggingFace model identifier for text embeddings |
| `tokenizer` | `AutoTokenizer` | Tokenizer for text preprocessing |
| `model` | `AutoModel` | Transformer model for generating embeddings |
| `_vector_size` | `int` | Dimensionality of generated embedding vectors |

### __init__

```python
def __init__(
    self,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> None:
```

Initialize the text embedder with a specified model.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_name` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | HuggingFace model identifier for text embeddings |

**Raises:**

| Type | Description |
|------|-------------|
| `InvalidPointsError` | If model cannot be loaded or initialized |

**Example:**
```python
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder

# Default model (balanced performance/quality)
embedder = SimpleTextEmbedder()

# High-quality model (larger, slower)
embedder = SimpleTextEmbedder("sentence-transformers/all-mpnet-base-v2")

# Fast model (smaller, faster)
embedder = SimpleTextEmbedder("sentence-transformers/all-MiniLM-L12-v2")

# Multilingual model
embedder = SimpleTextEmbedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

### embed

```python
def embed(
    self,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    point_id: Optional[str] = None
) -> models.PointStruct:
```

Convert text content to a vector embedding.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `content` | `str` | *required* | Text string to embed (must be non-empty) |
| `metadata` | `Optional[Dict[str, Any]]` | `None` | Optional metadata to store with the point |
| `point_id` | `Optional[str]` | `None` | Optional custom ID for the point (UUID generated if None) |

**Returns:**

| Type | Description |
|------|-------------|
| `models.PointStruct` | Point containing the text embedding, content, and metadata |

**Raises:**

| Type | Description |
|------|-------------|
| `InvalidPointsError` | If text cannot be embedded or is empty |

**Example:**
```python
# Basic text embedding
point = embedder.embed("The quick brown fox jumps over the lazy dog")
print(f"Vector dimensions: {len(point.vector)}")
print(f"Content: {point.payload['content']}")

# With metadata
point = embedder.embed(
    "Machine learning is transforming technology",
    metadata={"topic": "AI", "difficulty": "intermediate"},
    point_id="doc_001"
)
```

### get_vector_size

```python
def get_vector_size(self) -> int:
```

Get the dimensionality of text embedding vectors.

**Returns:**

| Type | Description |
|------|-------------|
| `int` | Number of dimensions in embedding vectors (e.g., 384 for all-MiniLM-L6-v2) |

**Example:**
```python
vector_size = embedder.get_vector_size()
print(f"Embedding dimensions: {vector_size}")
```

### get_distance_metric

```python
def get_distance_metric(self) -> str:
```

Get the recommended distance metric for text embeddings.

**Returns:**

| Type | Description |
|------|-------------|
| `str` | Recommended distance metric (always "Cosine" for text embeddings) |

**Example:**
```python
distance = embedder.get_distance_metric()
print(f"Recommended distance: {distance}")  # Output: "Cosine"
```

## Usage Patterns

### Single Text Embedding

```python
embedder = SimpleTextEmbedder()

# Embed a single text
text = "Artificial intelligence is revolutionizing many industries"
point = embedder.embed(text)

# Access the embedding vector
vector = point.vector  # List[float] with embedding values
content = point.payload["content"]  # Original text
point_id = point.id  # UUID string
```

### Batch Text Processing

```python
texts = [
    "Machine learning enables computers to learn",
    "Deep learning uses neural networks",
    "Natural language processing understands text"
]

# Process multiple texts
points = []
for i, text in enumerate(texts):
    point = embedder.embed(
        text,
        metadata={"index": i, "topic": "AI"},
        point_id=f"text_{i}"
    )
    points.append(point)

print(f"Processed {len(points)} texts")
```

### Integration with Vector Database

```python
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB

# Initialize components
embedder = SimpleTextEmbedder()
vector_db = EmbeddedVectorDB()

# Create collection with correct dimensions
collection_name = "text_documents"
vector_db.create_collection(
    collection_name,
    vector_size=embedder.get_vector_size(),
    distance=embedder.get_distance_metric()
)

# Embed and store text
text = "Example document content"
point = embedder.embed(text, metadata={"source": "example.txt"})
vector_db.upsert_points(collection_name, [point])
```

### Search and Similarity

```python
# Embed query text
query = "information about machine learning"
query_point = embedder.embed(query)
query_vector = query_point.vector

# Search for similar content
results = vector_db.search_points(
    collection_name="text_documents",
    query_vector=query_vector,
    limit=5
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.payload['content']}")
```

## Model Selection Guide

### Default Model (all-MiniLM-L6-v2)
- **Vector Size**: 384 dimensions
- **Performance**: Fast inference, good quality
- **Use Case**: General-purpose text embedding
- **Languages**: English (primary)

### High-Quality Model (all-mpnet-base-v2)
- **Vector Size**: 768 dimensions
- **Performance**: Slower inference, excellent quality
- **Use Case**: Applications requiring highest accuracy
- **Languages**: English

### Fast Model (all-MiniLM-L12-v2)
- **Vector Size**: 384 dimensions
- **Performance**: Very fast inference, good quality
- **Use Case**: Real-time applications, large-scale processing
- **Languages**: English

### Multilingual Model (paraphrase-multilingual-MiniLM-L12-v2)
- **Vector Size**: 384 dimensions
- **Performance**: Moderate speed, good cross-language quality
- **Use Case**: Multi-language applications
- **Languages**: 50+ languages

## Performance Considerations

### Memory Usage
```python
# Monitor model memory usage
import psutil
import os

process = psutil.Process(os.getpid())
memory_before = process.memory_info().rss / 1024 / 1024  # MB

embedder = SimpleTextEmbedder("sentence-transformers/all-mpnet-base-v2")

memory_after = process.memory_info().rss / 1024 / 1024  # MB
print(f"Model loaded, memory increase: {memory_after - memory_before:.1f} MB")
```

### Batch Processing Optimization
```python
# For processing many texts, consider batching at the application level
def process_large_text_collection(texts, embedder, batch_size=32):
    """Process large text collections efficiently."""
    points = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Process batch
        for text in batch:
            point = embedder.embed(text)
            points.append(point)
            
        # Optional: Clear GPU cache periodically
        if i % 1000 == 0:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return points
```

### GPU Acceleration
```python
# Check if GPU is being used
import torch

embedder = SimpleTextEmbedder()
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {next(embedder.model.parameters()).device}")

# The model automatically uses GPU if available
```

## Error Handling

### Common Issues and Solutions

```python
try:
    # Attempt to embed text
    point = embedder.embed(user_input)
except InvalidPointsError as e:
    if "empty" in str(e).lower():
        print("Error: Text cannot be empty")
    else:
        print(f"Embedding error: {e}")

# Validate input before embedding
def safe_embed(embedder, text, metadata=None):
    """Safely embed text with validation."""
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    
    if len(text.strip()) == 0:
        raise ValueError("Text cannot be empty or whitespace only")
    
    if len(text) > 10000:  # Arbitrary limit
        print(f"Warning: Very long text ({len(text)} chars), truncating")
        text = text[:10000]
    
    return embedder.embed(text, metadata)
```

## Source Code

??? abstract "View Complete Source Code"
    Click to expand and view the complete source implementation of the SimpleTextEmbedder class.

    ```python title="src/rag_to_riches/vectordb/embedder.py"
    # =============================================================================
    #  Filename: embedder.py
    #
    #  Short Description: Embedder class hierarchy for converting text and images to vector embeddings.
    #
    #  Creation date: 2025-01-06
    #  Author: Asif Qamar
    # =============================================================================

    from abc import ABC, abstractmethod
    from pathlib import Path
    from typing import Union, List, Optional, Dict, Any
    from uuid import uuid4
    from PIL import Image
    from icontract import require, ensure
    from qdrant_client import models
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
    from loguru import logger
    from rag_to_riches.exceptions import InvalidPointsError

    class Embedder(ABC):
        """Abstract base class for all embedders.
        
        Defines the interface for converting various input types (text, images)
        into vector embeddings as PointStruct objects for vector database storage.
        """
        
        @abstractmethod
        def embed(self, content: Union[str, Image.Image], metadata: Optional[Dict[str, Any]] = None,
                  point_id: Optional[str] = None) -> models.PointStruct:
            """Convert content to a vector embedding as a PointStruct."""
            pass

        @abstractmethod
        def get_vector_size(self) -> int:
            """Get the dimensionality of the embedding vectors."""
            pass

        @abstractmethod
        def get_distance_metric(self) -> str:
            """Get the recommended distance metric for this embedder."""
            pass

    class SimpleTextEmbedder(Embedder):
        """Text-only embedder using sentence transformers for semantic text embeddings."""
        
        @require(lambda model_name: isinstance(model_name, str) and len(model_name.strip()) > 0,
                 "Model name must be a non-empty string")
        def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
            """Initialize the text embedder with a specified model.
            
            Args:
                model_name: HuggingFace model identifier for text embeddings.
                
            Raises:
                InvalidPointsError: If model cannot be loaded.
            """
            try:
                self.model_name = model_name
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.eval()
                
                # Determine vector size by running a test embedding
                with torch.no_grad():
                    test_input = self.tokenizer("test", return_tensors="pt", 
                                              padding=True, truncation=True)
                    test_output = self.model(**test_input)
                    # Use mean pooling of last hidden states
                    self._vector_size = test_output.last_hidden_state.mean(dim=1).shape[1]
                    
                logger.info(f"Initialized SimpleTextEmbedder with model '{model_name}', "
                           f"vector size: {self._vector_size}")
                           
            except Exception as e:
                raise InvalidPointsError(
                    issue=f"Failed to initialize text embedder with model '{model_name}': {str(e)}"
                )

        @require(lambda content: isinstance(content, str) and len(content.strip()) > 0,
                 "Content must be a non-empty string")
        @ensure(lambda result: isinstance(result, models.PointStruct), 
                "Must return a valid PointStruct")
        def embed(self, content: str, metadata: Optional[Dict[str, Any]] = None,
                  point_id: Optional[str] = None) -> models.PointStruct:
            """Convert text content to a vector embedding.
            
            Args:
                content: Text string to embed.
                metadata: Optional metadata to store with the point.
                point_id: Optional custom ID for the point.
                
            Returns:
                PointStruct containing the text embedding and metadata.
                
            Raises:
                InvalidPointsError: If text cannot be embedded.
            """
            try:
                # Generate ID if not provided
                if point_id is None:
                    point_id = str(uuid4())
                    
                # Tokenize and embed the text
                with torch.no_grad():
                    inputs = self.tokenizer(content, return_tensors="pt", 
                                          padding=True, truncation=True, max_length=512)
                    outputs = self.model(**inputs)
                    
                    # Use mean pooling over token embeddings
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    # Normalize the embeddings for better similarity search
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                    vector = embeddings.squeeze().numpy()
                    
                # Prepare payload with content and metadata
                payload = {"content": content, "content_type": "text"}
                if metadata:
                    payload.update(metadata)
                    
                point = models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                
                logger.debug(f"Embedded text content (length: {len(content)}) to vector "
                            f"(dimension: {len(vector)})")
                return point
                
            except Exception as e:
                raise InvalidPointsError(
                    issue=f"Failed to embed text content: {str(e)}",
                    points_count=1
                )

        def get_vector_size(self) -> int:
            """Get the dimensionality of text embedding vectors."""
            return self._vector_size

        def get_distance_metric(self) -> str:
            """Get the recommended distance metric for text embeddings."""
            return "Cosine"

    # Additional classes: MultimodalEmbedder, create_embedder function, etc.
    ```

---

*The SimpleTextEmbedder class provides state-of-the-art text embedding capabilities with automatic model management and optimization for semantic search applications.* 