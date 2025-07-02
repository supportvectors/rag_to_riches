# EmbeddedVectorDB

Usage Documentation: [EmbeddedVectorDB Class Guide](../vectordb/EmbeddedVectorDB.md)

Embedded vector database client using Qdrant for local storage.

This class provides a comprehensive interface to an embedded Qdrant vector database, including collection management, point storage and retrieval, and search functionality for semantic search applications.

## rag_to_riches.vectordb.embedded_vectordb.EmbeddedVectorDB

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `client` | `QdrantClient` | Qdrant client instance for database operations |
| `config` | `Dict[str, Any]` | Database configuration settings |

### __init__

```python
def __init__(
    self,
    config: Optional[Dict[str, Any]] = None
) -> None:
```

Initialize the embedded vector database client.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `config` | `Optional[Dict[str, Any]]` | `None` | Optional configuration dictionary. If None, uses default local Qdrant settings |

**Raises:**

| Type | Description |
|------|-------------|
| `VectorDatabasePathNotFoundError` | If specified database path doesn't exist |

**Example:**
```python
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB

# Default configuration
vector_db = EmbeddedVectorDB()

# Custom configuration
config = {
    "host": "localhost",
    "port": 6333,
    "path": "./custom_qdrant_db"
}
vector_db = EmbeddedVectorDB(config)
```

### create_collection

```python
def create_collection(
    self,
    collection_name: str,
    vector_size: int,
    distance: str = "Cosine"
) -> bool:
```

Create a new vector collection with specified parameters.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `collection_name` | `str` | *required* | Name for the new collection |
| `vector_size` | `int` | *required* | Dimensionality of vectors to store |
| `distance` | `str` | `"Cosine"` | Distance metric: "Cosine", "Euclidean", or "Dot" |

**Returns:**

| Type | Description |
|------|-------------|
| `bool` | True if collection created successfully |

**Raises:**

| Type | Description |
|------|-------------|
| `CollectionAlreadyExistsError` | If collection with this name already exists |
| `InvalidVectorSizeError` | If vector_size is not a positive integer |
| `InvalidDistanceMetricError` | If distance metric is not supported |

**Example:**
```python
# Create collection for text embeddings
success = vector_db.create_collection(
    collection_name="documents",
    vector_size=384,
    distance="Cosine"
)
print(f"Collection created: {success}")
```

### ensure_collection

```python
def ensure_collection(
    self,
    collection_name: str,
    vector_size: int,
    distance: str = "Cosine"
) -> bool:
```

Ensure collection exists with correct parameters, create if necessary.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `collection_name` | `str` | *required* | Name of the collection |
| `vector_size` | `int` | *required* | Required vector dimensionality |
| `distance` | `str` | `"Cosine"` | Required distance metric |

**Returns:**

| Type | Description |
|------|-------------|
| `bool` | True if collection exists or was created with correct parameters |

**Raises:**

| Type | Description |
|------|-------------|
| `CollectionParameterMismatchError` | If existing collection has different parameters |

### collection_exists

```python
def collection_exists(
    self,
    collection_name: str
) -> bool:
```

Check if a collection exists in the database.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `collection_name` | `str` | *required* | Name of the collection to check |

**Returns:**

| Type | Description |
|------|-------------|
| `bool` | True if collection exists |

### get_collection_info

```python
def get_collection_info(
    self,
    collection_name: str
) -> models.CollectionInfo:
```

Get detailed information about a collection.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `collection_name` | `str` | *required* | Name of the collection |

**Returns:**

| Type | Description |
|------|-------------|
| `models.CollectionInfo` | Collection configuration and statistics |

**Raises:**

| Type | Description |
|------|-------------|
| `CollectionNotFoundError` | If collection doesn't exist |

**Example:**
```python
info = vector_db.get_collection_info("documents")
print(f"Vector size: {info.config.params.vectors.size}")
print(f"Distance metric: {info.config.params.vectors.distance}")
print(f"Points count: {info.points_count}")
```

### upsert_points

```python
def upsert_points(
    self,
    collection_name: str,
    points: List[models.PointStruct]
) -> None:
```

Insert or update points in a collection.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `collection_name` | `str` | *required* | Target collection name |
| `points` | `List[models.PointStruct]` | *required* | List of points to upsert |

**Raises:**

| Type | Description |
|------|-------------|
| `CollectionNotFoundError` | If collection doesn't exist |
| `InvalidPointsError` | If points have invalid format or dimensions |

**Example:**
```python
from qdrant_client import models
from uuid import uuid4

points = [
    models.PointStruct(
        id=str(uuid4()),
        vector=[0.1, 0.2, 0.3],  # Must match collection vector_size
        payload={
            "content": "Example text",
            "category": "example"
        }
    )
]

vector_db.upsert_points("documents", points)
```

### search_points

```python
def search_points(
    self,
    collection_name: str,
    query_vector: List[float],
    limit: int = 10,
    score_threshold: Optional[float] = None,
    filter_conditions: Optional[models.Filter] = None
) -> List[models.ScoredPoint]:
```

Search for similar vectors in the collection.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `collection_name` | `str` | *required* | Collection to search in |
| `query_vector` | `List[float]` | *required* | Vector to find similarities for |
| `limit` | `int` | `10` | Maximum number of results |
| `score_threshold` | `Optional[float]` | `None` | Minimum similarity score |
| `filter_conditions` | `Optional[models.Filter]` | `None` | Metadata filters |

**Returns:**

| Type | Description |
|------|-------------|
| `List[models.ScoredPoint]` | List of scored points sorted by similarity |

**Raises:**

| Type | Description |
|------|-------------|
| `CollectionNotFoundError` | If collection doesn't exist |
| `InvalidPointsError` | If query vector has wrong dimensions |

**Example:**
```python
# Search for similar vectors
query_vector = [0.1, 0.2, 0.3]  # Must match collection dimensions
results = vector_db.search_points(
    collection_name="documents",
    query_vector=query_vector,
    limit=5,
    score_threshold=0.7
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.payload['content']}")
```

### count_points

```python
def count_points(
    self,
    collection_name: str
) -> int:
```

Count the number of points in a collection.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `collection_name` | `str` | *required* | Collection to count |

**Returns:**

| Type | Description |
|------|-------------|
| `int` | Number of points in the collection |

**Raises:**

| Type | Description |
|------|-------------|
| `CollectionNotFoundError` | If collection doesn't exist |

### delete_collection

```python
def delete_collection(
    self,
    collection_name: str
) -> bool:
```

Delete a collection and all its data.

!!! warning
    This permanently deletes all data in the collection.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `collection_name` | `str` | *required* | Collection to delete |

**Returns:**

| Type | Description |
|------|-------------|
| `bool` | True if collection was deleted successfully |

**Raises:**

| Type | Description |
|------|-------------|
| `CollectionNotFoundError` | If collection doesn't exist |

### list_collections

```python
def list_collections(self) -> List[str]:
```

Get list of all collection names in the database.

**Returns:**

| Type | Description |
|------|-------------|
| `List[str]` | List of collection names |

**Example:**
```python
collections = vector_db.list_collections()
print(f"Available collections: {collections}")
```

### get_database_stats

```python
def get_database_stats(self) -> Dict[str, Any]:
```

Get comprehensive database statistics and health information.

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Database statistics including memory usage and collection info |

**Example:**
```python
stats = vector_db.get_database_stats()
print(f"Total collections: {stats['collections_count']}")
print(f"Memory usage: {stats['memory_usage_mb']} MB")
```

### get_point

```python
def get_point(
    self,
    collection_name: str,
    point_id: str
) -> Optional[models.Record]:
```

Retrieve a specific point by ID.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `collection_name` | `str` | *required* | Collection containing the point |
| `point_id` | `str` | *required* | ID of the point to retrieve |

**Returns:**

| Type | Description |
|------|-------------|
| `Optional[models.Record]` | Point record if found, None otherwise |

### delete_points

```python
def delete_points(
    self,
    collection_name: str,
    point_ids: List[str]
) -> None:
```

Delete specific points from a collection.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `collection_name` | `str` | *required* | Collection to delete from |
| `point_ids` | `List[str]` | *required* | List of point IDs to delete |

**Raises:**

| Type | Description |
|------|-------------|
| `CollectionNotFoundError` | If collection doesn't exist |

### update_collection

```python
def update_collection(
    self,
    collection_name: str,
    optimizer_config: Optional[models.OptimizersConfig] = None,
    params: Optional[models.CollectionParams] = None
) -> bool:
```

Update collection configuration and optimization settings.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `collection_name` | `str` | *required* | Collection to update |
| `optimizer_config` | `Optional[models.OptimizersConfig]` | `None` | New optimizer settings |
| `params` | `Optional[models.CollectionParams]` | `None` | New collection parameters |

**Returns:**

| Type | Description |
|------|-------------|
| `bool` | True if update was successful |

### batch_upsert

```python
def batch_upsert(
    self,
    collection_name: str,
    points: List[models.PointStruct],
    batch_size: int = 100
) -> None:
```

Upsert points in batches for better performance with large datasets.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `collection_name` | `str` | *required* | Target collection |
| `points` | `List[models.PointStruct]` | *required* | Points to upsert |
| `batch_size` | `int` | `100` | Number of points per batch |

## Source Code

??? abstract "View Complete Source Code"
    Click to expand and view the complete source implementation of the EmbeddedVectorDB class.

    ```python title="src/rag_to_riches/vectordb/embedded_vectordb.py"
    # =============================================================================
    #  Filename: embedded_vectordb.py
    #
    #  Short Description: Embedded vector database client using Qdrant for local storage.
    #
    #  Creation date: 2025-01-06
    #  Author: Asif Qamar
    # =============================================================================

    from pathlib import Path
    from typing import List, Optional
    from icontract import require, ensure, invariant
    from qdrant_client import QdrantClient, models
    from rag_to_riches import config
    from rag_to_riches.exceptions import (
        VectorDatabasePathNotFoundError,
        CollectionAlreadyExistsError,
        CollectionNotFoundError,
        InvalidVectorSizeError,
        InvalidDistanceMetricError,
        InvalidPointsError,
    )
    from loguru import logger

    @invariant(lambda self: hasattr(self, 'client') and self.client is not None, 
               "Client must be initialized and not None")
    class EmbeddedVectorDB:
        """Embedded vector database client using Qdrant for local storage.
        
        This class provides a simple interface to interact with a local Qdrant
        vector database, including collection management and point operations.
        """
        
        @require(lambda: "vector_db" in config, "Config must contain vector_db section")
        @require(lambda: "path" in config["vector_db"], "Config must contain vector_db.path")
        @ensure(lambda self: hasattr(self, 'client'), "Client must be initialized after construction")
        def __init__(self) -> None:
            """Initialize the embedded vector database client.
            
            Raises:
                VectorDatabasePathNotFoundError: If the vector database path doesn't exist.
            """
            path = config["vector_db"]["path"]
            
            if not Path(path).exists():
                raise VectorDatabasePathNotFoundError(
                    path=path,
                    suggestion="Create the directory or update your configuration"
                )
                
            self.client = QdrantClient(path=path)
            logger.info(f"Connected to embedded vector database at {path}")

        def collection_exists(self, collection_name: str) -> bool:
            """Check if a collection exists.
            
            Args:
                collection_name: Name of the collection to check.
                
            Returns:
                True if collection exists, False otherwise.
            """
            return self.client.collection_exists(collection_name)

        @require(lambda collection_name: isinstance(collection_name, str) and 
                 len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
        @require(lambda vector_size: isinstance(vector_size, int) and vector_size > 0,
                 "Vector size must be a positive integer")
        @require(lambda distance: isinstance(distance, str) and distance.strip() != "",
                 "Distance must be a non-empty string")
        @ensure(lambda result: isinstance(result, bool), "Must return boolean")
        def create_collection(self, collection_name: str, vector_size: int, 
                             distance: str) -> bool:
            """Create a new collection in the vector database.
            
            Args:
                collection_name: Name of the collection to create.
                vector_size: Size of the vectors to be stored.
                distance: Distance metric to use (e.g., 'Cosine', 'Euclidean', 'Dot').
                
            Returns:
                True if collection was created successfully.
                
            Raises:
                CollectionAlreadyExistsError: If collection already exists.
                InvalidVectorSizeError: If vector size is invalid.
                InvalidDistanceMetricError: If distance metric is invalid.
            """
            # Additional validation with custom exceptions
            if not isinstance(vector_size, int) or vector_size <= 0:
                raise InvalidVectorSizeError(vector_size)
                
            valid_distances = ["Cosine", "Euclid", "Dot"]
            if distance not in valid_distances:
                raise InvalidDistanceMetricError(distance, valid_distances)
            
            if self.client.collection_exists(collection_name):
                raise CollectionAlreadyExistsError(
                    collection_name=collection_name,
                    vector_size=vector_size,
                    distance=distance
                )
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=distance)
            )
            logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")
            return self.client.collection_exists(collection_name)

        @require(lambda collection_name: isinstance(collection_name, str) and 
                 len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
        @require(lambda query_vector: isinstance(query_vector, list) and len(query_vector) > 0,
                 "Query vector must be a non-empty list")
        @require(lambda query_vector: all(isinstance(x, (int, float)) for x in query_vector),
                 "Query vector must contain only numbers")
        @require(lambda limit: isinstance(limit, int) and limit > 0,
                 "Limit must be a positive integer")
        @ensure(lambda result: isinstance(result, list), "Must return a list")
        def search_points(self, collection_name: str, query_vector: List[float], 
                         limit: int = 10, score_threshold: Optional[float] = None) -> List[models.ScoredPoint]:
            """Search for similar points in a collection using vector similarity.
            
            Args:
                collection_name: Name of the collection to search in.
                query_vector: Vector to search for similar points.
                limit: Maximum number of results to return.
                score_threshold: Minimum similarity score threshold.
                
            Returns:
                List of scored points sorted by similarity.
                
            Raises:
                CollectionNotFoundError: If collection doesn't exist.
                InvalidPointsError: If query vector is invalid.
            """
            if not self.client.collection_exists(collection_name):
                try:
                    available_collections = [col.name for col in self.client.get_collections().collections]
                except Exception:
                    available_collections = None
                    
                raise CollectionNotFoundError(
                    collection_name=collection_name,
                    operation="search_points",
                    available_collections=available_collections
                )
            
            search_params = {
                "collection_name": collection_name,
                "query_vector": query_vector,
                "limit": limit,
            }

            if score_threshold is not None:
                search_params["score_threshold"] = score_threshold

            results = self.client.search(**search_params)
            logger.info(f"Found {len(results)} points in collection '{collection_name}'")
            return results

        @require(lambda collection_name: isinstance(collection_name, str) and 
                 len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
        @require(lambda points: isinstance(points, list) and len(points) > 0,
                 "Points must be a non-empty list")
        @require(lambda points: all(isinstance(p, models.PointStruct) for p in points),
                 "All points must be PointStruct instances")
        def upsert_points(self, collection_name: str, 
                         points: List[models.PointStruct]) -> None:
            """Insert or update points in a collection.
            
            Args:
                collection_name: Name of the collection to upsert points into.
                points: List of point structures to upsert.
                
            Raises:
                CollectionNotFoundError: If collection doesn't exist.
            """
            if not self.client.collection_exists(collection_name):
                try:
                    available_collections = [col.name for col in self.client.get_collections().collections]
                except Exception:
                    available_collections = None
                    
                raise CollectionNotFoundError(
                    collection_name=collection_name,
                    operation="upsert_points",
                    available_collections=available_collections
                )
                
            self.client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Upserted {len(points)} points to collection '{collection_name}'")

        # ... Additional methods: count_points, delete_collection, ensure_collection,
        # get_collection_info, get_points, list_collections, delete_points, etc.
    ```

---

*The EmbeddedVectorDB class provides reliable, high-performance vector storage for intelligent applications using local Qdrant database.* 