# EmbeddedVectorDB Class

## Overview

The `EmbeddedVectorDB` class provides a comprehensive interface to an embedded Qdrant vector database. This class handles all vector database operations including collection management, point storage and retrieval, and search functionality for semantic search applications.

## Class Definition

```python
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB

class EmbeddedVectorDB:
    """
    Embedded Qdrant vector database for efficient similarity search.
    
    Provides a complete interface for vector storage, retrieval, and search
    operations using a local Qdrant instance.
    """
```

## Key Features

- **Local Vector Storage**: Embedded Qdrant database for production and development
- **Collection Management**: Create, configure, and manage vector collections
- **CRUD Operations**: Complete create, read, update, delete functionality
- **Similarity Search**: High-performance vector similarity search
- **Health Monitoring**: Database health checks and statistics
- **Flexible Configuration**: Customizable distance metrics and collection settings

## Constructor

```python
def __init__(self, config: Optional[Dict[str, Any]] = None):
    """
    Initialize the embedded vector database.
    
    Args:
        config: Optional configuration dictionary
                Default uses local Qdrant with standard settings
    """
```

### Configuration Options

```python
# Default configuration
vector_db = EmbeddedVectorDB()

# Custom configuration
config = {
    "host": "localhost",
    "port": 6333,
    "grpc_port": 6334,
    "prefer_grpc": True,
    "https": False,
    "api_key": None,
    "path": "./qdrant_db"  # Local storage path
}
vector_db = EmbeddedVectorDB(config)
```

## Core Methods

### Collection Management

#### create_collection()

```python
def create_collection(
    self,
    collection_name: str,
    vector_size: int,
    distance: str = "Cosine"
) -> bool:
    """
    Create a new vector collection.
    
    Args:
        collection_name: Name of the collection to create
        vector_size: Dimension of vectors to store
        distance: Distance metric ("Cosine", "Euclid", "Dot")
        
    Returns:
        bool: True if collection created successfully
        
    Raises:
        DatabaseError: If collection creation fails
    """
```

**Usage Example:**
```python
# Create a collection for 384-dimensional embeddings
success = vector_db.create_collection(
    collection_name="document_embeddings",
    vector_size=384,
    distance="Cosine"
)

if success:
    print("Collection created successfully")
```

#### delete_collection()

```python
def delete_collection(self, collection_name: str) -> bool:
    """
    Delete an existing collection.
    
    Args:
        collection_name: Name of collection to delete
        
    Returns:
        bool: True if deletion successful
    """
```

#### get_collection_info()

```python
def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a collection.
    
    Returns:
        dict: Collection metadata and statistics
    """
```

### Point Operations

#### upsert_points()

```python
def upsert_points(
    self,
    collection_name: str,
    points: List[models.PointStruct]
) -> None:
    """
    Insert or update points in the collection.
    
    Args:
        collection_name: Target collection name
        points: List of PointStruct objects to upsert
        
    Raises:
        DatabaseError: If upsert operation fails
    """
```

**Usage Example:**
```python
from qdrant_client import models
from uuid import uuid4

# Create points
points = [
    models.PointStruct(
        id=str(uuid4()),
        vector=[0.1, 0.2, 0.3, ...],  # 384-dimensional vector
        payload={
            "content": "Example text content",
            "category": "example",
            "timestamp": "2024-01-15T10:30:00Z"
        }
    )
]

# Upsert to collection
vector_db.upsert_points("document_embeddings", points)
```

#### search_points()

```python
def search_points(
    self,
    collection_name: str,
    query_vector: List[float],
    limit: int = 10,
    score_threshold: Optional[float] = None,
    filter_conditions: Optional[models.Filter] = None
) -> List[models.ScoredPoint]:
    """
    Search for similar vectors in the collection.
    
    Args:
        collection_name: Collection to search in
        query_vector: Vector to find similarities for
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score
        filter_conditions: Optional metadata filters
        
    Returns:
        List of ScoredPoint objects with similarity scores
    """
```

**Usage Example:**
```python
# Search for similar vectors
query_vector = embedder.embed_text("search query")
results = vector_db.search_points(
    collection_name="document_embeddings",
    query_vector=query_vector,
    limit=5,
    score_threshold=0.7
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.payload['content']}")
```

#### count_points()

```python
def count_points(self, collection_name: str) -> int:
    """
    Count the number of points in a collection.
    
    Returns:
        int: Number of points in the collection
    """
```

### Advanced Search

#### search_with_filters()

```python
def search_with_filters(
    self,
    collection_name: str,
    query_vector: List[float],
    metadata_filter: Dict[str, Any],
    limit: int = 10
) -> List[models.ScoredPoint]:
    """
    Search with metadata filtering.
    
    Args:
        metadata_filter: Dictionary of filter conditions
        
    Example filters:
        {"category": "science"}
        {"rating": {"gte": 4.0}}
        {"tags": {"any": ["python", "ml"]}}
    """
```

**Usage Example:**
```python
# Search with category filter
results = vector_db.search_with_filters(
    collection_name="document_embeddings",
    query_vector=query_vector,
    metadata_filter={"category": "research"},
    limit=10
)

# Complex filter example
complex_filter = {
    "must": [
        {"key": "category", "match": {"value": "science"}},
        {"key": "rating", "range": {"gte": 4.0}}
    ]
}
```

### Health and Monitoring

#### consistency_check()

```python
def consistency_check(self, collection_name: str) -> Dict[str, Any]:
    """
    Perform consistency check on collection.
    
    Returns:
        dict: Health status and any issues found
    """
```

#### get_database_stats()

```python
def get_database_stats(self) -> Dict[str, Any]:
    """
    Get overall database statistics.
    
    Returns:
        dict: Database health, memory usage, collection stats
    """
```

## Error Handling

The `EmbeddedVectorDB` class uses the project's exception hierarchy:

```python
from rag_to_riches.exceptions import DatabaseError, CollectionNotFoundError

try:
    vector_db.create_collection("test", 384)
except DatabaseError as e:
    print(f"Database error: {e}")
except CollectionNotFoundError as e:
    print(f"Collection not found: {e}")
```

## Performance Considerations

### Batch Operations

```python
# Efficient batch upsert
large_points_batch = [...]  # Many points

# Process in chunks for memory efficiency
chunk_size = 1000
for i in range(0, len(large_points_batch), chunk_size):
    chunk = large_points_batch[i:i + chunk_size]
    vector_db.upsert_points(collection_name, chunk)
```

### Connection Management

```python
# The class automatically manages connections
# Multiple operations reuse the same connection
vector_db.create_collection("col1", 384)
vector_db.upsert_points("col1", points1)
vector_db.search_points("col1", query_vector1)
# No need to manually manage connections
```

### Memory Optimization

```python
# For large collections, consider:
# 1. Batch processing
# 2. Appropriate vector dimensions
# 3. Regular cleanup of unused collections

# Monitor memory usage
stats = vector_db.get_database_stats()
print(f"Memory usage: {stats['memory_usage_mb']} MB")
```

## Integration Examples

### With SimpleTextEmbedder

```python
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder

# Complete pipeline
embedder = SimpleTextEmbedder()
vector_db = EmbeddedVectorDB()

# Setup
collection_name = "documents"
vector_db.create_collection(
    collection_name,
    embedder.get_vector_size(),
    embedder.get_distance_metric()
)

# Add documents
texts = ["Document 1", "Document 2", "Document 3"]
vectors = embedder.embed_batch(texts)

points = []
for i, (text, vector) in enumerate(zip(texts, vectors)):
    points.append(models.PointStruct(
        id=f"doc_{i}",
        vector=vector,
        payload={"content": text}
    ))

vector_db.upsert_points(collection_name, points)

# Search
query = "search terms"
query_vector = embedder.embed_text(query)
results = vector_db.search_points(collection_name, query_vector)
```

### With Corpus Classes

```python
from rag_to_riches.corpus.animals import Animals

# Used automatically by corpus classes
animals = Animals(vector_db, collection_name="animal_wisdom")

# The corpus class handles all vector operations internally
# vector_db methods are called automatically during:
# - Data loading and indexing
# - Search operations
# - Collection management
```

## Configuration and Environment

### Environment Variables

```bash
# Optional environment configuration
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export QDRANT_API_KEY=your_api_key  # For cloud instances
```

### Production Configuration

```python
# Production setup with persistence
production_config = {
    "path": "/data/qdrant_storage",  # Persistent storage
    "host": "localhost",
    "port": 6333,
    "grpc_port": 6334,
    "prefer_grpc": True,  # Better performance
    "timeout": 30.0
}

vector_db = EmbeddedVectorDB(production_config)
```

## Troubleshooting

### Common Issues

1. **Collection Already Exists**
   ```python
   try:
       vector_db.create_collection("existing", 384)
   except DatabaseError as e:
       if "already exists" in str(e):
           print("Collection exists, using existing collection")
   ```

2. **Dimension Mismatch**
   ```python
   # Ensure vector dimensions match collection
   collection_info = vector_db.get_collection_info("my_collection")
   expected_size = collection_info["config"]["params"]["vectors"]["size"]
   
   if len(query_vector) != expected_size:
       raise ValueError(f"Vector size mismatch: {len(query_vector)} != {expected_size}")
   ```

3. **Database Connection Issues**
   ```python
   # Check database health
   try:
       stats = vector_db.get_database_stats()
       print("Database is healthy")
   except Exception as e:
       print(f"Database connection issue: {e}")
   ```

## Related Components

- [`SimpleTextEmbedder`](SimpleTextEmbedder.md): Text embedding component
- [`SemanticSearch`](../search/index.md): High-level search interface
- [`Animals`](../corpus/Animals.md): Domain-specific corpus using this database
- [`Exceptions`](../exceptions/index.md): Error handling for database operations

---

*Part of the RAG to Riches framework - reliable vector storage for intelligent applications.* 