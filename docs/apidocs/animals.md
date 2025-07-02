# Animals

Usage Documentation: [Animals Class Guide](../corpus/Animals.md)

A powerful, intelligent corpus loader for animal quotes with RAG capabilities.

This class provides a complete solution for working with collections of animal quotes, offering semantic search, AI-powered question answering, and beautiful result display. Perfect for educational applications, research, or building chatbots that need access to animal wisdom and quotes.

## rag_to_riches.corpus.animals.Animals

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `embedder` | `SimpleTextEmbedder` | Text embedding model for vector representations |
| `collection_name` | `str` | Name of the Qdrant collection storing the quotes |
| `wisdom` | `Optional[AnimalWisdom]` | Loaded collection of animal quotes (None until loaded) |
| `semantic_search` | `SemanticSearch` | Underlying search engine for similarity queries |
| `ANIMALS_RAG_SYSTEM_PROMPT` | `str` | Comprehensive system prompt for RAG operations |
| `SIMPLE_ANIMALS_PROMPT` | `str` | Alternative shorter prompt for quick use |

### __init__

```python
def __init__(
    self,
    vector_db: EmbeddedVectorDB,
    embedder: Optional[SimpleTextEmbedder] = None,
    collection_name: str = "animals"
) -> None:
```

Initialize the Animals corpus loader with vector database and embedding capabilities.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `vector_db` | `EmbeddedVectorDB` | *required* | Vector database instance for storage and retrieval |
| `embedder` | `Optional[SimpleTextEmbedder]` | `None` | Text embedder for vector generation. If None, uses default model |
| `collection_name` | `str` | `"animals"` | Name of the collection to work with |

**Raises:**

| Type | Description |
|------|-------------|
| `InvalidPointsError` | If vector database initialization fails |

**Example:**
```python
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
from rag_to_riches.corpus.animals import Animals

vector_db = EmbeddedVectorDB()
animals = Animals(vector_db, collection_name="my_quotes")
```

### load_from_jsonl

```python
def load_from_jsonl(
    self,
    jsonl_path: Union[str, Path]
) -> AnimalWisdom:
```

Load animal quotes from a JSONL file with automatic validation and error handling.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `jsonl_path` | `Union[str, Path]` | *required* | Path to the JSONL file containing animal quotes |

**Returns:**

| Type | Description |
|------|-------------|
| `AnimalWisdom` | Validated collection of animal quotes |

**Raises:**

| Type | Description |
|------|-------------|
| `FileNotFoundError` | If the JSONL file doesn't exist |
| `ValidationError` | If quote data doesn't match expected schema |
| `InvalidPointsError` | If file cannot be parsed or contains invalid data |

**Example:**
```python
from pathlib import Path

quotes_file = Path("data/animal_quotes.jsonl")
wisdom = animals.load_from_jsonl(quotes_file)
print(f"Loaded {len(wisdom)} quotes")
```

### load_and_index

```python
def load_and_index(
    self,
    jsonl_path: Union[str, Path]
) -> Tuple[AnimalWisdom, List[str]]:
```

Load quotes from JSONL file and automatically index them for semantic search.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `jsonl_path` | `Union[str, Path]` | *required* | Path to the JSONL file containing animal quotes |

**Returns:**

| Type | Description |
|------|-------------|
| `Tuple[AnimalWisdom, List[str]]` | Tuple of (loaded wisdom collection, list of indexed point IDs) |

**Raises:**

| Type | Description |
|------|-------------|
| `FileNotFoundError` | If the JSONL file doesn't exist |
| `ValidationError` | If quote data doesn't match expected schema |
| `InvalidPointsError` | If indexing fails |

**Example:**
```python
wisdom, point_ids = animals.load_and_index("data/animal_quotes.jsonl")
print(f"Loaded and indexed {len(wisdom)} quotes with {len(point_ids)} vectors")
```

### search

```python
def search(
    self,
    query: str,
    limit: int = 10,
    score_threshold: Optional[float] = None,
    author: Optional[str] = None,
    category: Optional[str] = None
) -> List[models.ScoredPoint]:
```

Search for animal quotes using semantic similarity with optional filtering.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | `str` | *required* | Search query (natural language) |
| `limit` | `int` | `10` | Maximum number of results to return |
| `score_threshold` | `Optional[float]` | `None` | Minimum similarity score threshold (0.0-1.0) |
| `author` | `Optional[str]` | `None` | Filter results by author name |
| `category` | `Optional[str]` | `None` | Filter results by category |

**Returns:**

| Type | Description |
|------|-------------|
| `List[models.ScoredPoint]` | List of scored search results with similarity scores and metadata |

**Raises:**

| Type | Description |
|------|-------------|
| `InvalidPointsError` | If search query cannot be processed |

**Example:**
```python
# Basic search
results = animals.search("wisdom about loyalty", limit=5)

# Search with filters
results = animals.search(
    "friendship", 
    limit=10, 
    score_threshold=0.7,
    author="Mahatma Gandhi"
)

# Display results
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Quote: {result.payload['content']}")
    print(f"Author: {result.payload['author']}")
```

### ask_llm

```python
def ask_llm(
    self,
    user_query: str,
    limit: int = 5,
    score_threshold: Optional[float] = None,
    author: Optional[str] = None,
    category: Optional[str] = None,
    model: str = "gpt-4o"
) -> AnimalWisdomResponse:
```

Get structured AI response about animals using the complete RAG pipeline.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `user_query` | `str` | *required* | Question about animals or animal wisdom |
| `limit` | `int` | `5` | Number of quotes to use as context |
| `score_threshold` | `Optional[float]` | `None` | Minimum relevance score for quotes |
| `author` | `Optional[str]` | `None` | Limit context to quotes from specific author |
| `category` | `Optional[str]` | `None` | Limit context to quotes from specific category |
| `model` | `str` | `"gpt-4o"` | OpenAI model to use for generation |

**Returns:**

| Type | Description |
|------|-------------|
| `AnimalWisdomResponse` | Structured response with answer, insights, quotes, and follow-ups |

**Raises:**

| Type | Description |
|------|-------------|
| `InvalidPointsError` | If LLM query fails or API error occurs |

**Example:**
```python
response = animals.ask_llm(
    "What do animals teach us about unconditional love?",
    limit=7,
    score_threshold=0.6
)

print("Answer:", response.answer)
print("Key Insights:", response.key_insights)
print("Recommended Quotes:", response.recommended_quotes)
```

### rag

```python
def rag(
    self,
    user_query: str,
    limit: int = 5,
    score_threshold: Optional[float] = None,
    author: Optional[str] = None,
    category: Optional[str] = None,
    model: str = "gpt-4o",
    response_type: str = "structured"
) -> Dict[str, Any]:
```

Complete AI-powered question answering in one powerful method call.

This is the flagship method that provides the complete RAG pipeline: semantic search, context generation, and AI reasoning in a single call.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `user_query` | `str` | *required* | Question about animals (natural language) |
| `limit` | `int` | `5` | Number of quotes to use as context |
| `score_threshold` | `Optional[float]` | `None` | Minimum relevance score for quotes (0.0-1.0) |
| `author` | `Optional[str]` | `None` | Limit context to quotes from specific author |
| `category` | `Optional[str]` | `None` | Limit context to quotes from specific category |
| `model` | `str` | `"gpt-4o"` | OpenAI model for AI responses |
| `response_type` | `str` | `"structured"` | Format of AI response: "structured" or "simple" |

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Complete results dictionary with llm_response, search_results, rag_context, and query_info |

**Raises:**

| Type | Description |
|------|-------------|
| `InvalidPointsError` | If any step in the RAG pipeline fails |
| `ValueError` | If response_type is not "structured" or "simple" |

**Example:**
```python
# Complete RAG in one call
result = animals.rag(
    "What do animals teach us about unconditional love?",
    limit=7,
    score_threshold=0.6,
    response_type="structured"
)

# Access components
ai_answer = result["llm_response"]
quotes_used = result["search_results"]
full_prompt = result["rag_context"]
query_info = result["query_info"]

print(f"Answer: {ai_answer.answer}")
print(f"Based on {len(quotes_used)} relevant quotes")
```

### display_search_results

```python
def display_search_results(
    self,
    results: List[models.ScoredPoint],
    search_description: str,
    max_text_length: int = 120
) -> None:
```

Present search results in a beautiful, easy-to-read table format using Rich library.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `results` | `List[models.ScoredPoint]` | *required* | Search results from the search() method |
| `search_description` | `str` | *required* | Descriptive title for the search display |
| `max_text_length` | `int` | `120` | Maximum characters per quote before truncation |

**Example:**
```python
results = animals.search("courage and bravery", limit=5)
animals.display_search_results(
    results, 
    "Quotes about Courage and Bravery",
    max_text_length=100
)
```

### get_collection_stats

```python
def get_collection_stats(self) -> Dict[str, Any]:
```

Get comprehensive statistics about the collection and loaded data.

**Returns:**

| Type | Description |
|------|-------------|
| `Dict[str, Any]` | Statistics including collection info, point count, categories, and authors |

**Example:**
```python
stats = animals.get_collection_stats()
print(f"Collection: {stats['collection_name']}")
print(f"Total quotes: {stats['point_count']}")
print(f"Categories: {stats['categories']}")
print(f"Authors: {stats['authors']}")
```

### consistency_check

```python
def consistency_check(self) -> bool:
```

Verify that the vector database collection is compatible with the current embedder.

**Returns:**

| Type | Description |
|------|-------------|
| `bool` | True if collection parameters match embedder requirements |

**Raises:**

| Type | Description |
|------|-------------|
| `CollectionNotFoundError` | If collection doesn't exist |
| `CollectionParameterMismatchError` | If vector size or distance metric don't match |

### recreate_collection

```python
def recreate_collection(self) -> None:
```

Delete and recreate the collection with current embedder parameters.

!!! warning
    This permanently deletes all indexed data in the collection.

**Raises:**

| Type | Description |
|------|-------------|
| `DatabaseError` | If collection operations fail |

### Helper Methods

#### format_search_results_for_rag

```python
def format_search_results_for_rag(
    self,
    search_results: List[models.ScoredPoint],
    max_results: int = 5
) -> str:
```

Format search results for RAG system prompt context.

#### create_rag_context

```python
def create_rag_context(
    self,
    user_query: str,
    search_results: List[models.ScoredPoint],
    system_prompt: Optional[str] = None
) -> str:
```

Create complete RAG context with system prompt and formatted results.

#### search_and_create_rag_context

```python
def search_and_create_rag_context(
    self,
    user_query: str,
    limit: int = 5,
    score_threshold: Optional[float] = None,
    author: Optional[str] = None,
    category: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> str:
```

Search for quotes and create RAG context in one convenient method.

## Source Code

??? abstract "View Complete Source Code"
    Click to expand and view the complete source implementation of the Animals class.

    ```python title="src/rag_to_riches/corpus/animals.py"
    # =============================================================================
    #  Filename: animals.py
    #
    #  Short Description: Animal quotes corpus loader with Pydantic models for Qdrant vectorization.
    #
    #  Creation date: 2025-01-06
    #  Author: Asif Qamar
    # =============================================================================

    import json
    from pathlib import Path
    from typing import List, Optional, Dict, Any, Union
    from uuid import uuid4
    from icontract import require, ensure
    from pydantic import BaseModel, Field, ConfigDict, model_validator
    from qdrant_client import models
    from loguru import logger

    from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
    from rag_to_riches.vectordb.embedder import SimpleTextEmbedder
    from rag_to_riches import config
    from rag_to_riches.exceptions import DataLoadingError, ValidationError

    class AnimalQuote(BaseModel):
        """Individual animal quote data model with validation and vector conversion."""
        
        model_config = ConfigDict(
            str_strip_whitespace=True,
            validate_assignment=True,
            extra="forbid"
        )
        
        animal: str = Field(..., min_length=1, max_length=100, 
                           description="Animal name (1-100 characters)")
        quote: str = Field(..., min_length=10, max_length=1000,
                          description="Wisdom quote (10-1000 characters)")
        category: Optional[str] = Field(None, max_length=50,
                                      description="Quote category (optional, max 50 chars)")
        
        @model_validator(mode='after')
        def validate_content_quality(self) -> 'AnimalQuote':
            """Ensure quote has meaningful content."""
            if len(self.quote.split()) < 3:
                raise ValueError("Quote must contain at least 3 words")
            return self

        def to_point_struct(self, embedder: SimpleTextEmbedder) -> models.PointStruct:
            """Convert to Qdrant PointStruct with embedding.
            
            Args:
                embedder: Text embedder for vector generation.
                
            Returns:
                PointStruct ready for vector database storage.
            """
            # Create searchable content by combining animal and quote
            content = f"{self.animal}: {self.quote}"
            
            # Prepare metadata
            metadata = {
                "animal": self.animal,
                "quote": self.quote,
                "content_type": "animal_quote"
            }
            if self.category:
                metadata["category"] = self.category
                
            return embedder.embed(content=content, metadata=metadata)

    class AnimalWisdom(BaseModel):
        """Collection of animal quotes with validation and loading capabilities."""
        
        model_config = ConfigDict(validate_assignment=True)
        
        quotes: List[AnimalQuote] = Field(default_factory=list,
                                        description="List of animal quotes")
        
        @classmethod
        def load_from_jsonl(cls, file_path: Union[str, Path]) -> 'AnimalWisdom':
            """Load animal quotes from JSONL file with validation.
            
            Args:
                file_path: Path to JSONL file containing quotes.
                
            Returns:
                AnimalWisdom instance with loaded and validated quotes.
                
            Raises:
                DataLoadingError: If file cannot be loaded or data is invalid.
            """
            try:
                path = Path(file_path)
                if not path.exists():
                    raise DataLoadingError(
                        file_path=str(path),
                        issue="File not found",
                        suggestion="Check the file path"
                    )
                
                quotes = []
                with open(path, 'r', encoding='utf-8') as file:
                    for line_num, line in enumerate(file, 1):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                quote = AnimalQuote(**data)
                                quotes.append(quote)
                            except (json.JSONDecodeError, ValueError) as e:
                                raise DataLoadingError(
                                    file_path=str(path),
                                    issue=f"Invalid data on line {line_num}: {str(e)}",
                                    suggestion="Check JSON format and data validation"
                                )
                
                if not quotes:
                    raise DataLoadingError(
                        file_path=str(path),
                        issue="No valid quotes found in file",
                        suggestion="Ensure the file contains valid JSONL data"
                    )
                    
                logger.info(f"Loaded {len(quotes)} animal quotes from {path}")
                return cls(quotes=quotes)
                
            except DataLoadingError:
                raise
            except Exception as e:
                raise DataLoadingError(
                    file_path=str(file_path),
                    issue=f"Unexpected error: {str(e)}",
                    suggestion="Check file accessibility and format"
                )

        def to_point_structs(self, embedder: SimpleTextEmbedder) -> List[models.PointStruct]:
            """Convert all quotes to PointStruct objects for vector storage.
            
            Args:
                embedder: Text embedder for vector generation.
                
            Returns:
                List of PointStruct objects ready for vector database.
            """
            return [quote.to_point_struct(embedder) for quote in self.quotes]

    class Animals:
        """Powerful, intelligent corpus loader for animal quotes with RAG capabilities."""
        
        @require(lambda vector_db: isinstance(vector_db, EmbeddedVectorDB),
                 "Vector DB must be an EmbeddedVectorDB instance")
        @require(lambda collection_name: isinstance(collection_name, str) and len(collection_name.strip()) > 0,
                 "Collection name must be a non-empty string")
        def __init__(self, vector_db: EmbeddedVectorDB, collection_name: str,
                     embedder: Optional[SimpleTextEmbedder] = None) -> None:
            """Initialize the Animals corpus loader.
            
            Args:
                vector_db: Vector database instance for storage.
                collection_name: Name of the collection for animal quotes.
                embedder: Optional text embedder (default: all-MiniLM-L6-v2).
            """
            self.vector_db = vector_db
            self.collection_name = collection_name
            self.embedder = embedder or SimpleTextEmbedder()
            self.wisdom = None
            
            # Ensure collection exists with proper parameters
            self._setup_collection()
            
            logger.info(f"Initialized Animals corpus for collection '{collection_name}'")

        @require(lambda file_path: isinstance(file_path, (str, Path)),
                 "File path must be a string or Path object")
        def load_corpus(self, file_path: Union[str, Path] = None) -> None:
            """Load animal quotes from JSONL file into the vector database.
            
            Args:
                file_path: Path to JSONL file. If None, uses default from config.
                
            Raises:
                DataLoadingError: If corpus cannot be loaded.
            """
            if file_path is None:
                # Use default path from configuration
                file_path = Path(config["data"]["corpus_path"]) / "animals" / "animals.jsonl"
            
            # Load and validate the wisdom corpus
            self.wisdom = AnimalWisdom.load_from_jsonl(file_path)
            
            # Convert to point structures and upsert to vector database
            points = self.wisdom.to_point_structs(self.embedder)
            self.vector_db.upsert_points(self.collection_name, points)
            
            logger.info(f"Loaded {len(points)} animal quotes into collection '{self.collection_name}'")

        @require(lambda query: isinstance(query, str) and len(query.strip()) > 0,
                 "Query must be a non-empty string")
        @require(lambda limit: isinstance(limit, int) and limit > 0,
                 "Limit must be a positive integer")
        @ensure(lambda result: isinstance(result, list), "Must return a list")
        def search(self, query: str, limit: int = 5, 
                  score_threshold: Optional[float] = None) -> List[models.ScoredPoint]:
            """Search for animal quotes using semantic similarity.
            
            Args:
                query: Search query text.
                limit: Maximum number of results to return.
                score_threshold: Minimum similarity score threshold.
                
            Returns:
                List of scored points containing similar animal quotes.
            """
            # Embed the query
            query_point = self.embedder.embed(content=query.strip())
            
            # Search in vector database
            results = self.vector_db.search_points(
                collection_name=self.collection_name,
                query_vector=query_point.vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            logger.debug(f"Search for '{query[:50]}...' returned {len(results)} results")
            return results

        def _setup_collection(self) -> None:
            """Ensure collection exists with correct parameters."""
            required_vector_size = self.embedder.get_vector_size()
            required_distance = self.embedder.get_distance_metric()
            
            self.vector_db.ensure_collection(
                collection_name=self.collection_name,
                vector_size=required_vector_size,
                distance=required_distance
            )

        # ... Additional methods: get_random_quote, display_search_results,
        # count_quotes, get_statistics, etc.
    ```

---

*The Animals class represents the culmination of modern RAG system design - combining semantic search, AI reasoning, and beautiful presentation into a single, powerful interface.* 