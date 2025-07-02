# SemanticSearch

Usage Documentation: [Search Package Guide](../search/index.md)

High-level semantic search interface combining embedders and vector database.

This class provides a unified interface for indexing and searching text and images using embeddings stored in a Qdrant vector database. It automatically handles collection setup, parameter validation, and provides both individual and batch operations for efficient content management.

## rag_to_riches.search.semantic_search.SemanticSearch

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `embedder` | `Embedder` | Embedder instance for converting content to vectors |
| `vector_db` | `EmbeddedVectorDB` | Vector database instance for storage and retrieval |
| `collection_name` | `str` | Name of the collection to work with |

### __init__

```python
def __init__(
    self,
    embedder: Embedder,
    vector_db: EmbeddedVectorDB,
    collection_name: str
) -> None:
```

Initialize the semantic search system.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `embedder` | `Embedder` | *required* | Embedder instance for converting content to vectors |
| `vector_db` | `EmbeddedVectorDB` | *required* | Vector database instance for storage and retrieval |
| `collection_name` | `str` | *required* | Name of the collection to work with |

**Raises:**

| Type | Description |
|------|-------------|
| `CollectionParameterMismatchError` | If collection parameters don't match embedder requirements |

**Example:**
```python
from rag_to_riches.search.semantic_search import SemanticSearch
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder

# Initialize components
embedder = SimpleTextEmbedder()
vector_db = EmbeddedVectorDB()
search = SemanticSearch(embedder, vector_db, "documents")
```

### index_text

```python
def index_text(
    self,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    point_id: Optional[str] = None
) -> str:
```

Index a text document into the collection.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | *required* | Text content to index (must be non-empty) |
| `metadata` | `Optional[Dict[str, Any]]` | `None` | Optional metadata to store with the text |
| `point_id` | `Optional[str]` | `None` | Optional custom ID for the point |

**Returns:**

| Type | Description |
|------|-------------|
| `str` | The ID of the indexed point |

**Raises:**

| Type | Description |
|------|-------------|
| `InvalidPointsError` | If text cannot be embedded or indexed |

**Example:**
```python
# Index a document
doc_id = search.index_text(
    "The quick brown fox jumps over the lazy dog",
    metadata={"source": "example.txt", "category": "animals"}
)
print(f"Indexed document with ID: {doc_id}")
```

### index_image

```python
def index_image(
    self,
    image: Image.Image,
    metadata: Optional[Dict[str, Any]] = None,
    point_id: Optional[str] = None
) -> str:
```

Index an image into the collection.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | `Image.Image` | *required* | PIL Image to index |
| `metadata` | `Optional[Dict[str, Any]]` | `None` | Optional metadata to store with the image |
| `point_id` | `Optional[str]` | `None` | Optional custom ID for the point |

**Returns:**

| Type | Description |
|------|-------------|
| `str` | The ID of the indexed point |

**Raises:**

| Type | Description |
|------|-------------|
| `InvalidPointsError` | If image cannot be embedded or indexed |

**Example:**
```python
from PIL import Image

# Index an image
image = Image.open("photo.jpg")
image_id = search.index_image(
    image,
    metadata={"filename": "photo.jpg", "category": "nature"}
)
```

### search_with_text

```python
def search_with_text(
    self,
    query_text: str,
    limit: int = 10,
    score_threshold: Optional[float] = None
) -> List[models.ScoredPoint]:
```

Search for similar content using text query.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query_text` | `str` | *required* | Text query to search for (must be non-empty) |
| `limit` | `int` | `10` | Maximum number of results to return |
| `score_threshold` | `Optional[float]` | `None` | Minimum similarity score threshold (0.0-1.0) |

**Returns:**

| Type | Description |
|------|-------------|
| `List[models.ScoredPoint]` | List of scored points sorted by similarity |

**Raises:**

| Type | Description |
|------|-------------|
| `InvalidPointsError` | If query cannot be processed |

**Example:**
```python
# Search for similar content
results = search.search_with_text(
    "fast animal jumping",
    limit=5,
    score_threshold=0.7
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.payload['content']}")
```

### search_with_image

```python
def search_with_image(
    self,
    query_image: Image.Image,
    limit: int = 10,
    score_threshold: Optional[float] = None
) -> List[models.ScoredPoint]:
```

Search for similar content using image query.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query_image` | `Image.Image` | *required* | PIL Image to search for |
| `limit` | `int` | `10` | Maximum number of results to return |
| `score_threshold` | `Optional[float]` | `None` | Minimum similarity score threshold (0.0-1.0) |

**Returns:**

| Type | Description |
|------|-------------|
| `List[models.ScoredPoint]` | List of scored points sorted by similarity |

**Raises:**

| Type | Description |
|------|-------------|
| `InvalidPointsError` | If query cannot be processed |

**Example:**
```python
# Search with image query
query_image = Image.open("query.jpg")
results = search.search_with_image(query_image, limit=10)

for result in results:
    content_type = result.payload.get('content_type', 'unknown')
    if content_type == 'text':
        print(f"Text: {result.payload['content']}")
    elif content_type == 'image':
        print(f"Image: {result.payload.get('filename', 'unnamed')}")
```

### index_all_text

```python
def index_all_text(
    self,
    texts: List[str],
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    point_ids: Optional[List[str]] = None
) -> List[str]:
```

Index multiple text documents into the collection.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `texts` | `List[str]` | *required* | List of text content to index (all must be non-empty) |
| `metadata_list` | `Optional[List[Dict[str, Any]]]` | `None` | Optional list of metadata for each text |
| `point_ids` | `Optional[List[str]]` | `None` | Optional list of custom IDs for the points |

**Returns:**

| Type | Description |
|------|-------------|
| `List[str]` | List of IDs of the indexed points |

**Raises:**

| Type | Description |
|------|-------------|
| `InvalidPointsError` | If texts cannot be embedded, indexed, or list lengths don't match |

**Example:**
```python
# Batch index multiple texts
texts = [
    "Machine learning is transforming technology",
    "Deep learning models require large datasets", 
    "Natural language processing enables text understanding"
]

metadata_list = [
    {"topic": "ML", "difficulty": "beginner"},
    {"topic": "DL", "difficulty": "intermediate"},
    {"topic": "NLP", "difficulty": "advanced"}
]

ids = search.index_all_text(texts, metadata_list=metadata_list)
print(f"Indexed {len(ids)} documents")
```

### index_all_images

```python
def index_all_images(
    self,
    images: List[Image.Image],
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    point_ids: Optional[List[str]] = None
) -> List[str]:
```

Index multiple images into the collection.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `images` | `List[Image.Image]` | *required* | List of PIL Images to index |
| `metadata_list` | `Optional[List[Dict[str, Any]]]` | `None` | Optional list of metadata for each image |
| `point_ids` | `Optional[List[str]]` | `None` | Optional list of custom IDs for the points |

**Returns:**

| Type | Description |
|------|-------------|
| `List[str]` | List of IDs of the indexed points |

**Raises:**

| Type | Description |
|------|-------------|
| `InvalidPointsError` | If images cannot be embedded, indexed, or list lengths don't match |

**Example:**
```python
# Batch index multiple images
images = [Image.open(f"image_{i}.jpg") for i in range(5)]
image_metadata = [{"batch": "photos", "index": i} for i in range(5)]

image_ids = search.index_all_images(images, metadata_list=image_metadata)
print(f"Indexed {len(image_ids)} images")
```

### consistency_check

```python
def consistency_check(self) -> bool:
```

Ensure that the vector size and distance metric of the collection is compatible with the embedder.

**Returns:**

| Type | Description |
|------|-------------|
| `bool` | True if collection parameters match embedder requirements |

**Raises:**

| Type | Description |
|------|-------------|
| `CollectionNotFoundError` | If collection doesn't exist |
| `CollectionParameterMismatchError` | If parameters don't match |

**Example:**
```python
# Verify collection compatibility
try:
    is_consistent = search.consistency_check()
    print(f"Collection consistency: {is_consistent}")
except CollectionParameterMismatchError as e:
    print(f"Collection mismatch: {e}")
```

## Usage Patterns

### Text-Only Semantic Search

```python
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder

# Text-only setup
embedder = SimpleTextEmbedder()
vector_db = EmbeddedVectorDB()
search = SemanticSearch(embedder, vector_db, "text_collection")

# Index documents
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Artificial intelligence is revolutionizing technology",
    "Machine learning algorithms learn from data"
]

for i, text in enumerate(texts):
    search.index_text(text, metadata={"doc_id": i})

# Search for similar content
results = search.search_with_text("AI and technology", limit=5)
```

### Multimodal Search

```python
from rag_to_riches.vectordb.embedder import create_embedder

# Multimodal setup
embedder = create_embedder("multimodal")
vector_db = EmbeddedVectorDB()
search = SemanticSearch(embedder, vector_db, "multimodal_collection")

# Index text and images
search.index_text(
    "A beautiful sunset over the ocean",
    metadata={"type": "description", "location": "beach"}
)

image = Image.open("sunset.jpg")
search.index_image(
    image,
    metadata={"type": "photo", "location": "beach", "filename": "sunset.jpg"}
)

# Search with text (finds both text and images)
text_results = search.search_with_text("ocean sunset", limit=10)

# Search with image (finds similar images and related text)
query_image = Image.open("query_sunset.jpg")
image_results = search.search_with_image(query_image, limit=10)
```

### Advanced Search with Score Thresholds

```python
# Only return high-quality results
high_quality_results = search.search_with_text(
    "artificial intelligence applications",
    limit=20,
    score_threshold=0.8  # Only results with >80% similarity
)

# Process results by content type
for result in high_quality_results:
    content_type = result.payload.get('content_type', 'unknown')
    print(f"[{content_type}] Score: {result.score:.3f}")
    
    if content_type == 'text':
        print(f"Text: {result.payload['content'][:100]}...")
    elif content_type == 'image':
        print(f"Image: {result.payload.get('filename', 'unnamed')}")
```

### Error Handling and Validation

```python
try:
    # Verify collection compatibility before operations
    search.consistency_check()
    
    # Safe indexing with validation
    if len(user_text.strip()) > 0:
        doc_id = search.index_text(user_text, metadata={"source": "user"})
        print(f"Indexed: {doc_id}")
    else:
        print("Error: Empty text provided")
        
except CollectionParameterMismatchError as e:
    print(f"Collection parameters don't match: {e}")
except InvalidPointsError as e:
    print(f"Indexing failed: {e}")
```

### Custom Point IDs for Better Tracking

```python
# Use meaningful IDs for better tracking
custom_id = search.index_text(
    "Important document content",
    metadata={"priority": "high"},
    point_id="doc_2024_001"
)
print(f"Indexed with custom ID: {custom_id}")

# Batch operations with custom IDs
texts = ["Document 1", "Document 2", "Document 3"]
custom_ids = ["doc_001", "doc_002", "doc_003"]
indexed_ids = search.index_all_text(texts, point_ids=custom_ids)
print(f"Indexed with custom IDs: {indexed_ids}")
```

## Source Code

??? abstract "View Complete Source Code"
    Click to expand and view the complete source implementation of the SemanticSearch class.

    ```python title="src/rag_to_riches/search/semantic_search.py"
    # =============================================================================
    #  Filename: semantic_search.py
    #
    #  Short Description: High-level semantic search interface combining embedders and vector database.
    #
    #  Creation date: 2025-01-06
    #  Author: Asif Qamar
    # =============================================================================

    from typing import List, Dict, Any, Union, Optional
    from PIL import Image
    from icontract import require, ensure
    from qdrant_client import models
    from loguru import logger

    from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
    from rag_to_riches.vectordb.embedder import Embedder, create_embedder
    from rag_to_riches.exceptions import (
        CollectionParameterMismatchError,
        InvalidPointsError,
        CollectionNotFoundError
    )

    class SemanticSearch:
        """High-level semantic search interface combining embedders and vector database.
        
        This class provides a unified interface for indexing and searching text and images
        using embeddings stored in a Qdrant vector database. It automatically handles
        collection setup, parameter validation, and provides both individual and batch
        operations for efficient content management.
        """
        
        @require(lambda embedder: isinstance(embedder, Embedder), "Embedder must be an Embedder instance")
        @require(lambda vector_db: isinstance(vector_db, EmbeddedVectorDB), 
                 "Vector DB must be an EmbeddedVectorDB instance")
        @require(lambda collection_name: isinstance(collection_name, str) and len(collection_name.strip()) > 0,
                 "Collection name must be a non-empty string")
        def __init__(self, embedder: Embedder, vector_db: EmbeddedVectorDB, 
                     collection_name: str) -> None:
            """Initialize the semantic search system.
            
            Args:
                embedder: Embedder instance for converting content to vectors.
                vector_db: Vector database instance for storage and retrieval.
                collection_name: Name of the collection to work with.
            """
            self.embedder = embedder
            self.vector_db = vector_db
            self.collection_name = collection_name
            
            # Ensure collection exists with correct parameters
            self._ensure_collection_setup()
            
            logger.info(f"Initialized SemanticSearch for collection '{collection_name}' "
                       f"with {type(embedder).__name__}")

        @require(lambda text: isinstance(text, str) and len(text.strip()) > 0,
                 "Text must be a non-empty string")
        def index_text(self, text: str, metadata: Optional[Dict[str, Any]] = None,
                       point_id: Optional[str] = None) -> str:
            """Index a text document into the collection.
            
            Args:
                text: Text content to index (must be non-empty).
                metadata: Optional metadata to store with the text.
                point_id: Optional custom ID for the point.
                
            Returns:
                The ID of the indexed point.
                
            Raises:
                InvalidPointsError: If text cannot be embedded or indexed.
            """
            try:
                # Use embedder to create a point with vector and metadata
                point = self.embedder.embed(content=text, metadata=metadata, point_id=point_id)
                
                # Upsert the point to the vector database
                self.vector_db.upsert_points(self.collection_name, [point])
                
                logger.debug(f"Indexed text document with ID: {point.id}")
                return point.id
                
            except Exception as e:
                raise InvalidPointsError(
                    issue=f"Failed to index text: {str(e)}",
                    points_count=1
                )

        @require(lambda query_text: isinstance(query_text, str) and len(query_text.strip()) > 0,
                 "Query text must be a non-empty string")
        @require(lambda limit: isinstance(limit, int) and limit > 0,
                 "Limit must be a positive integer")
        @ensure(lambda result: isinstance(result, list), "Must return a list")
        def search_with_text(self, query_text: str, limit: int = 10,
                            score_threshold: Optional[float] = None) -> List[models.ScoredPoint]:
            """Search for similar content using text query.
            
            Args:
                query_text: Text query to search for (must be non-empty).
                limit: Maximum number of results to return.
                score_threshold: Minimum similarity score threshold (0.0-1.0).
                
            Returns:
                List of scored points sorted by similarity.
                
            Raises:
                InvalidPointsError: If query cannot be processed.
            """
            try:
                # Embed the query text
                query_point = self.embedder.embed(content=query_text.strip())
                query_vector = query_point.vector
                
                # Perform vector search
                results = self.vector_db.search_points(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold
                )
                
                logger.debug(f"Text search for '{query_text[:50]}...' returned {len(results)} results")
                return results
                
            except Exception as e:
                raise InvalidPointsError(
                    issue=f"Failed to search with text: {str(e)}",
                    points_count=1
                )

        def consistency_check(self) -> bool:
            """Ensure that the vector size and distance metric of the collection is compatible with the embedder.
            
            Returns:
                True if collection parameters match embedder requirements.
                
            Raises:
                CollectionNotFoundError: If collection doesn't exist.
                CollectionParameterMismatchError: If parameters don't match.
            """
            if not self.vector_db.collection_exists(self.collection_name):
                raise CollectionNotFoundError(
                    collection_name=self.collection_name,
                    operation="consistency_check",
                    available_collections=self.vector_db.list_collections()
                )
            
            # Get collection info and check parameters
            collection_info = self.vector_db.get_collection_info(self.collection_name)
            expected_vector_size = self.embedder.get_vector_size()
            expected_distance = self.embedder.get_distance_metric()
            
            actual_vector_size = collection_info.config.params.vectors.size
            actual_distance = collection_info.config.params.vectors.distance
            
            if (actual_vector_size != expected_vector_size or 
                actual_distance != expected_distance):
                raise CollectionParameterMismatchError(
                    collection_name=self.collection_name,
                    expected_vector_size=expected_vector_size,
                    actual_vector_size=actual_vector_size,
                    expected_distance=expected_distance,
                    actual_distance=actual_distance
                )
            
            return True

        def _ensure_collection_setup(self) -> None:
            """Ensure collection exists with correct parameters."""
            required_vector_size = self.embedder.get_vector_size()
            required_distance = self.embedder.get_distance_metric()
            
            self.vector_db.ensure_collection(
                collection_name=self.collection_name,
                vector_size=required_vector_size,
                distance=required_distance
            )

        # ... Additional methods: index_image, search_with_image, index_all_text, 
        # index_all_images, and other batch operations
    ```

---

*The SemanticSearch class provides a high-level interface for semantic search operations, abstracting the complexity of vector embeddings and database operations into simple, intuitive methods.* 