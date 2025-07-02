# API Reference

The complete technical reference for the RAG to Riches framework. This documentation follows the traditional API reference style, providing detailed method signatures, parameters, return types, exceptions, and comprehensive examples for each class and function.

!!! tip "Documentation Style"
    This API reference follows the traditional Python library documentation format (similar to [Pydantic](https://docs.pydantic.dev/latest/api/base_model/)), with complete technical details for developers and library maintainers.

    For conceptual guides and usage tutorials, see the [User Guide](../index.md).

## Core Classes

### Vector Database Operations

#### [`EmbeddedVectorDB`](embedded_vectordb.md)
Embedded vector database client using Qdrant for local storage.

```python
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
```

#### [`SimpleTextEmbedder`](simple_text_embedder.md)  
Text-only embedder using sentence transformers for semantic text embeddings.

```python
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder
```

### Search Operations

#### [`SemanticSearch`](semantic_search.md)
High-level semantic search interface combining embedders and vector database.

```python
from rag_to_riches.search.semantic_search import SemanticSearch
```

### Corpus Management

#### [`Animals`](animals.md)
Powerful, intelligent corpus loader for animal quotes with RAG capabilities.

```python
from rag_to_riches.corpus.animals import Animals
```

---

## Package Guides

For conceptual guides and tutorials, see the package documentation:

- **[Vector Database Guide](../vectordb/index.md)** - Working with embeddings and vector storage
- **[Search Guide](../search/index.md)** - Advanced semantic search patterns

- **[Corpus Guide](../corpus/index.md)** - Data models and content management
- **[Exceptions Guide](../exceptions/index.md)** - Error handling system
- **[Examples Guide](../examples/index.md)** - Usage examples and patterns
- **[Utilities Guide](../utils/index.md)** - Helper functions and utilities

## Quick Reference

### Class Overview

| Class | Purpose | Import |
|-------|---------|---------|
| [`EmbeddedVectorDB`](embedded_vectordb.md) | Vector database operations | `rag_to_riches.vectordb.embedded_vectordb` |
| [`SimpleTextEmbedder`](simple_text_embedder.md) | Text embedding generation | `rag_to_riches.vectordb.embedder` |
| [`SemanticSearch`](semantic_search.md) | High-level search interface | `rag_to_riches.search.semantic_search` |
| [`Animals`](animals.md) | RAG with animal quotes corpus | `rag_to_riches.corpus.animals` |

### Common Patterns

#### Basic Vector Database Setup
```python
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder

# Initialize components
vector_db = EmbeddedVectorDB()
embedder = SimpleTextEmbedder()

# Create collection
vector_db.create_collection(
    "documents", 
    vector_size=embedder.get_vector_size(),
    distance="Cosine"
)
```

#### Text Embedding and Search
```python
# Embed and store text
text = "Machine learning is transforming technology"
point = embedder.embed(text)
vector_db.upsert_points("documents", [point])

# Search for similar content
query_point = embedder.embed("AI technology applications")
results = vector_db.search_points(
    "documents", 
    query_point.vector, 
    limit=5
)
```

#### High-Level Semantic Search
```python
from rag_to_riches.search.semantic_search import SemanticSearch

# Initialize search system
search = SemanticSearch(embedder, vector_db, "documents")

# Index content
search.index_text("Document content")

# Search
results = search.search_with_text("query text")
```

---

## External Links

- **[Project Repository](https://github.com/asifqamar/rag_to_riches)** - Source code and contributions
- **[Interactive Notebooks](../notebooks/examples/index.md)** - Hands-on tutorials
- **[User Guide](../index.md)** - Conceptual documentation and getting started

---

*This API reference provides technical details for all public classes and methods in the RAG to Riches framework. For tutorials and conceptual guides, see the [User Guide](../index.md).* 