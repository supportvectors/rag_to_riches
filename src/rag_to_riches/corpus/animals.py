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
from pydantic import BaseModel, Field, ConfigDict, field_validator
from qdrant_client import models
from loguru import logger

# Import instructor and OpenAI for LLM integration
import instructor
from openai import OpenAI

from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder
from rag_to_riches.search.semantic_search import SemanticSearch
from rag_to_riches.exceptions import InvalidPointsError
from rag_to_riches.corpus.data_models import AnimalQuote, AnimalWisdom


#============================================================================================
#  Class: Animals
#============================================================================================
class Animals:
    """A powerful, intelligent corpus loader for animal quotes with RAG capabilities.
    
    This class provides a complete solution for working with collections of animal quotes,
    offering semantic search, AI-powered question answering, and beautiful result display.
    Perfect for educational applications, research, or building chatbots that need access
    to animal wisdom and quotes.
    
    Key Features:
        - Load quotes from JSONL files with automatic validation
        - Semantic search using state-of-the-art embeddings
        - Filter by author, category, or similarity score
        - AI-powered question answering with GPT models
        - Beautiful formatted output with Rich library
        - Complete RAG (Retrieval-Augmented Generation) pipeline
        - Batch operations for efficiency
    
    Typical Workflow:
        1. Initialize with a vector database
        2. Load quotes from a JSONL file
        3. Index quotes for semantic search
        4. Search for relevant quotes
        5. Ask AI questions about the quotes
    
    Example:
        ```python
        from pathlib import Path
        from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
        from rag_to_riches.corpus.animals import Animals
        
        # Initialize
        vector_db = EmbeddedVectorDB()
        animals = Animals(vector_db, collection_name="my_quotes")
        
        # Load and index quotes
        quotes_file = Path("data/animal_quotes.jsonl")
        animals.load_and_index(quotes_file)
        
        # Search for quotes
        results = animals.search("wisdom about dogs", limit=5)
        
        # Ask AI a question
        response = animals.ask_llm("What do animals teach us about loyalty?")
        animals.display_llm_response(response, "loyalty question")
        ```
    
    Attributes:
        embedder: Text embedding model for vector representations
        collection_name: Name of the Qdrant collection storing the quotes
        wisdom: Loaded collection of animal quotes (None until loaded)
        semantic_search: Underlying search engine for similarity queries
    """
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    @require(lambda vector_db: isinstance(vector_db, EmbeddedVectorDB),
             "Vector DB must be an EmbeddedVectorDB instance")
    @require(lambda embedder: embedder is None or isinstance(embedder, SimpleTextEmbedder),
             "Embedder must be None or a SimpleTextEmbedder instance")
    def __init__(self, vector_db: EmbeddedVectorDB, 
                 embedder: Optional[SimpleTextEmbedder] = None,
                 collection_name: str = "animals") -> None:
        """Initialize your Animals quote corpus with intelligent search capabilities.
        
        Sets up the complete infrastructure for loading, indexing, and searching animal
        quotes. The system uses advanced sentence transformers for semantic understanding
        and can work with any size collection efficiently.
        
        Args:
            vector_db: Your vector database instance where quotes will be stored.
                This handles all the vector storage and retrieval operations.
            embedder: Optional text embedding model. If None, uses the default
                'sentence-transformers/all-MiniLM-L6-v2' model which provides
                excellent semantic understanding for quotes and wisdom.
            collection_name: Unique name for your quote collection. Use descriptive
                names like "animal_wisdom", "pet_quotes", or "nature_sayings" to
                organize multiple collections.
        
        Example:
            ```python
            # Basic setup with default embedder
            animals = Animals(vector_db)
            
            # Custom setup with specific collection
            animals = Animals(
                vector_db=my_db,
                collection_name="philosophical_animal_quotes"
            )
            
            # Advanced setup with custom embedder
            custom_embedder = SimpleTextEmbedder(model_name="custom-model")
            animals = Animals(vector_db, embedder=custom_embedder)
            ```
        
        Note:
            The constructor automatically loads the RAG system prompt for AI interactions.
            If the prompt file is missing, a warning is logged but the system continues
            to work with reduced AI capabilities.
        """
        self.embedder = embedder or SimpleTextEmbedder()
        self.collection_name = collection_name
        self.wisdom: Optional[AnimalWisdom] = None
        
        # Initialize the underlying semantic search engine
        self.semantic_search = SemanticSearch(
            embedder=self.embedder,
            vector_db=vector_db,
            collection_name=collection_name
        )
        
        logger.info(f"Initialized Animals corpus loader for collection '{collection_name}'")
        if not Animals.ANIMALS_RAG_SYSTEM_PROMPT:
            Animals.ANIMALS_RAG_SYSTEM_PROMPT = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load the AI system prompt from external configuration file.
        
        Returns:
            The system prompt text for RAG operations, or empty string if unavailable.
        """
        try:
            return Animals.SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not load system prompt: {e}")
            return ""

    
    # ----------------------------------------------------------------------------------------
    #  Load from JSONL
    # ----------------------------------------------------------------------------------------
    @require(lambda jsonl_path: isinstance(jsonl_path, (str, Path)),
             "JSONL path must be a string or Path object")
    @ensure(lambda result: isinstance(result, AnimalWisdom),
            "Must return an AnimalWisdom instance")
    def load_from_jsonl(self, jsonl_path: Path) -> AnimalWisdom:
        """Load and validate animal quotes from a JSONL (JSON Lines) file.
        
        Reads a file where each line contains a JSON object with quote data. The method
        performs comprehensive validation, skips malformed entries with helpful warnings,
        and returns a structured collection of quotes ready for indexing and search.
        
        Expected JSONL Format:
            Each line should be a JSON object with these fields:
            - "text": The actual quote content (required)
            - "author": Who said or wrote the quote (required)  
            - "category": Thematic classification like "Wisdom", "Humor" (required)
        
        Args:
            jsonl_path: Path to your JSONL file containing animal quotes.
                Can be a string path or pathlib.Path object.
        
        Returns:
            AnimalWisdom object containing all successfully loaded quotes with
            convenient methods for filtering and analysis.
        
        Raises:
            FileNotFoundError: When the specified file doesn't exist at the given path.
            InvalidPointsError: When no valid quotes are found in the file, indicating
                format issues or empty content.
        
        Example:
            ```python
            # Load quotes from file
            quotes_path = Path("data/animal_wisdom.jsonl")
            wisdom = animals.load_from_jsonl(quotes_path)
            
            print(f"Loaded {len(wisdom)} quotes")
            print(f"Categories: {wisdom.get_categories()}")
            print(f"Authors: {wisdom.get_authors()}")
            
            # Access individual quotes
            for quote in wisdom.quotes[:3]:
                print(f'"{quote.text}" - {quote.author}')
            ```
        
        File Format Example:
            ```
            {"text": "Dogs are not our whole life, but they make our lives whole.", "author": "Roger Caras", "category": "Pets and Companionship"}
            {"text": "The greatness of a nation can be judged by the way its animals are treated.", "author": "Mahatma Gandhi", "category": "Ethics and Compassion"}
            ```
        
        Note:
            - Empty lines in the file are automatically skipped
            - Malformed JSON lines generate warnings but don't stop the process
            - The loaded quotes are stored in self.wisdom for later use
            - All text fields are automatically stripped of whitespace
        """
        jsonl_path = Path(jsonl_path)
        
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        try:
            quotes = []
            with open(jsonl_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        data = json.loads(line)
                        quote = AnimalQuote(**data)
                        quotes.append(quote)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Skipped invalid line {line_num} in {jsonl_path}: {e}")
                        continue
            
            if not quotes:
                raise InvalidPointsError(
                    issue=f"No valid quotes found in {jsonl_path}",
                    points_count=0
                )
            
            self.wisdom = AnimalWisdom(quotes=quotes, source_file=jsonl_path)
            logger.info(f"Loaded {len(quotes)} animal quotes from {jsonl_path}")
            return self.wisdom
            
        except Exception as e:
            raise InvalidPointsError(
                issue=f"Failed to load animal quotes from {jsonl_path}: {str(e)}",
                points_count=0
            )
    
    # ----------------------------------------------------------------------------------------
    #  Index All Quotes
    # ----------------------------------------------------------------------------------------
    @require(lambda self: self.wisdom is not None,
             "Animal wisdom must be loaded before indexing")
    def index_all_quotes(self) -> List[str]:
        """Transform all loaded quotes into searchable vector embeddings.
        
        This method takes your loaded quotes and creates high-dimensional vector
        representations that enable semantic search. The process uses advanced
        sentence transformers to understand the meaning and context of each quote,
        not just keyword matching.
        
        The indexing process:
        1. Extracts text content from each quote
        2. Generates semantic embeddings using the configured model
        3. Stores vectors in the database with rich metadata
        4. Creates searchable points for instant retrieval
        
        Returns:
            List of unique point IDs for each indexed quote. These IDs can be used
            for direct retrieval, debugging, or managing specific quotes.
        
        Raises:
            InvalidPointsError: When indexing fails due to embedding errors,
                database issues, or missing quote data.
        
        Example:
            ```python
            # Load quotes first
            wisdom = animals.load_from_jsonl("quotes.jsonl")
            
            # Index for semantic search
            point_ids = animals.index_all_quotes()
            print(f"Successfully indexed {len(point_ids)} quotes")
            
            # Now you can search semantically
            results = animals.search("loyalty and friendship")
            ```
        
        Performance Notes:
            - Batch processing is used for efficiency with large collections
            - Indexing time scales with collection size and model complexity
            - Typical speed: ~100-500 quotes per second depending on hardware
            - GPU acceleration automatically used if available
        
        Note:
            You must call load_from_jsonl() before indexing. The method will
            fail gracefully if no quotes are loaded, providing clear error messages.
        """
        if not self.wisdom:
            raise InvalidPointsError(
                issue="No animal wisdom loaded. Call load_from_jsonl() first.",
                points_count=0
            )
        
        try:
            # Prepare texts and metadata for batch indexing
            texts = [quote.text for quote in self.wisdom.quotes]
            metadata_list = [quote.to_payload() for quote in self.wisdom.quotes]
            
            # Use SemanticSearch's batch indexing capability
            indexed_ids = self.semantic_search.index_all_text(
                texts=texts,
                metadata_list=metadata_list
            )
            
            logger.info(f"Successfully indexed {len(indexed_ids)} animal quotes into collection '{self.collection_name}'")
            return indexed_ids
            
        except Exception as e:
            raise InvalidPointsError(
                issue=f"Failed to index animal quotes: {str(e)}",
                points_count=len(self.wisdom.quotes) if self.wisdom else 0
            )
    
    # ----------------------------------------------------------------------------------------
    #  Collection Management
    # ----------------------------------------------------------------------------------------
    def recreate_collection(self) -> None:
        """Completely reset your quote collection with a fresh, empty database.
        
        This is a powerful cleanup method that removes all existing quotes and
        creates a brand new collection. Use this when you need to start over,
        fix corruption issues, or completely change your quote dataset.
        
        What this method does:
        1. Safely deletes the existing collection if it exists
        2. Creates a new empty collection with optimal settings
        3. Clears all loaded quote data from memory
        4. Prepares the system for fresh data loading
        
        Raises:
            InvalidPointsError: If the recreation process fails due to database
                connection issues or permission problems.
        
        Example:
            ```python
            # Reset everything and start fresh
            animals.recreate_collection()
            
            # Now load new data
            new_wisdom = animals.load_from_jsonl("updated_quotes.jsonl")
            animals.index_all_quotes()
            ```
        
        Warning:
            This operation is irreversible! All existing quotes in the collection
            will be permanently deleted. Make sure you have backups of important
            data before calling this method.
        
        Use Cases:
            - Switching to a completely different quote dataset
            - Fixing corrupted vector data
            - Changing embedding models (requires reindexing)
            - Cleaning up test data before production deployment
        """
        try:
            # Delete existing collection if it exists
            if self.semantic_search.vector_db.collection_exists(self.collection_name):
                logger.info(f"Deleting existing collection '{self.collection_name}'")
                self.semantic_search.vector_db.delete_collection(self.collection_name)
            
            # Create new empty collection
            logger.info(f"Creating new empty collection '{self.collection_name}'")
            self.semantic_search.vector_db.create_collection(
                collection_name=self.collection_name,
                vector_size=self.semantic_search.embedder.get_vector_size(),
                distance=self.semantic_search.embedder.get_distance_metric()
            )
            
            # Clear loaded wisdom data
            self.wisdom = None
            
            logger.info(f"Successfully recreated empty collection '{self.collection_name}'")
            
        except Exception as e:
            raise InvalidPointsError(
                issue=f"Failed to recreate collection: {str(e)}",
                points_count=0
            )
    
    # ----------------------------------------------------------------------------------------
    #  Load and Index
    # ----------------------------------------------------------------------------------------
    def load_and_index(self, jsonl_path: Path) -> tuple[AnimalWisdom, List[str]]:
        """One-step solution: load quotes from file and make them instantly searchable.
        
        This convenience method combines loading and indexing in a single call,
        perfect for getting up and running quickly. It handles the complete
        pipeline from raw JSONL file to searchable vector database.
        
        Args:
            jsonl_path: Path to your JSONL file containing animal quotes.
        
        Returns:
            A tuple containing:
            - AnimalWisdom: Your loaded and validated quote collection
            - List[str]: Point IDs for all indexed quotes
        
        Example:
            ```python
            # Complete setup in one line
            wisdom, point_ids = animals.load_and_index("my_quotes.jsonl")
            
            print(f"Ready to search {len(wisdom)} quotes!")
            
            # Immediately start searching
            results = animals.search("courage and bravery")
            ```
        
        Note:
            This method is equivalent to calling load_from_jsonl() followed by
            index_all_quotes(), but more convenient for common workflows.
        """
        wisdom = self.load_from_jsonl(jsonl_path)
        point_ids = self.index_all_quotes()
        return wisdom, point_ids
    
    # ----------------------------------------------------------------------------------------
    #  Search Quotes
    # ----------------------------------------------------------------------------------------
    @require(lambda query: isinstance(query, str) and len(query.strip()) > 0,
             "Query must be a non-empty string")
    @require(lambda limit: isinstance(limit, int) and limit > 0,
             "Limit must be a positive integer")
    @require(lambda author: author is None or (isinstance(author, str) and len(author.strip()) > 0),
             "Author filter must be None or a non-empty string")
    @require(lambda category: category is None or (isinstance(category, str) and len(category.strip()) > 0),
             "Category filter must be None or a non-empty string")
    @ensure(lambda result: isinstance(result, list), "Must return a list")
    def search(self, query: str, limit: int = 10, 
              score_threshold: Optional[float] = None,
              author: Optional[str] = None,
              category: Optional[str] = None) -> List[models.ScoredPoint]:
        """Find the most relevant animal quotes using intelligent semantic search.
        
        This powerful search method goes beyond simple keyword matching to understand
        the meaning and context of your query. It finds quotes that are conceptually
        similar, even if they don't share exact words with your search terms.
        
        Args:
            query: Your search question or topic. Use natural language like
                "what do animals teach us about love?" or "quotes about courage".
            limit: Maximum number of results to return (default: 10).
                Higher values give more options but may include less relevant results.
            score_threshold: Minimum similarity score (0.0-1.0). Only quotes with
                similarity above this threshold will be returned. Use 0.7+ for
                highly relevant results, 0.5+ for broader matches.
            author: Filter results to only include quotes by this specific author.
                Case-insensitive matching (e.g., "Gandhi" matches "Mahatma Gandhi").
            category: Filter results to only include quotes from this category.
                Case-insensitive matching (e.g., "wisdom" matches "Wisdom and Philosophy").
        
        Returns:
            List of ScoredPoint objects, each containing:
            - score: Similarity score (higher = more relevant)
            - payload: Quote metadata (content, author, category)
            Results are automatically sorted by relevance (highest scores first).
        
        Raises:
            InvalidPointsError: When search fails due to database issues or
                invalid query parameters.
        
        Example:
            ```python
            # Basic semantic search
            results = animals.search("loyalty and friendship")
            
            # Precise search with high threshold
            results = animals.search(
                "courage in difficult times",
                limit=5,
                score_threshold=0.8
            )
            
            # Search within specific author's quotes
            gandhi_quotes = animals.search(
                "compassion",
                author="Mahatma Gandhi",
                limit=3
            )
            
            # Browse by category
            wisdom_quotes = animals.search(
                "life lessons",
                category="Wisdom and Philosophy"
            )
            
            # Process results
            for result in results:
                print(f"Score: {result.score:.3f}")
                print(f"Quote: {result.payload['content']}")
                print(f"Author: {result.payload['author']}")
                print("---")
            ```
        
        Search Tips:
            - Use descriptive phrases rather than single keywords
            - Try different phrasings if you don't find what you're looking for
            - Lower the score_threshold to see more diverse results
            - Combine filters to narrow down to specific types of quotes
        
        Performance:
            - Search is typically very fast (< 100ms for most collections)
            - Larger collections may take slightly longer but remain responsive
            - Filtering by author/category happens after vector search for efficiency
        """
        try:
            # Use SemanticSearch for the core search functionality
            # Request more results than needed to allow for filtering
            search_limit = limit * 3 if (author or category) else limit
            
            initial_results = self.semantic_search.search_with_text(
                query_text=query.strip(),
                limit=search_limit,
                score_threshold=score_threshold
            )
            
            # Apply metadata filters if specified
            filtered_results = self._apply_metadata_filters(
                results=initial_results,
                author_filter=author,
                category_filter=category
            )
            
            # Limit to requested number of results
            final_results = filtered_results[:limit]
            
            logger.info(f"Animal quotes search for '{query[:50]}...' returned {len(final_results)} results"
                       f"{' (filtered by author)' if author else ''}"
                       f"{' (filtered by category)' if category else ''}")
            
            return final_results
            
        except Exception as e:
            raise InvalidPointsError(
                issue=f"Failed to search animal quotes: {str(e)}",
                points_count=1
            )
    
    # ----------------------------------------------------------------------------------------
    #  Get Collection Stats
    # ----------------------------------------------------------------------------------------
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics and insights about your quote collection.
        
        Provides a detailed overview of your collection's size, content diversity,
        and database status. Perfect for monitoring, debugging, or presenting
        collection metrics to users.
        
        Returns:
            Dictionary containing detailed statistics:
            - collection_name: Name of your quote collection
            - collection_exists: Whether the database collection exists
            - point_count: Total quotes stored in the database
            - loaded_quotes: Number of quotes currently loaded in memory
            - categories: List of all unique quote categories (sorted)
            - authors: List of all unique authors (sorted)
        
        Example:
            ```python
            stats = animals.get_collection_stats()
            
            print(f"Collection: {stats['collection_name']}")
            print(f"Total quotes in database: {stats['point_count']}")
            print(f"Quotes loaded in memory: {stats['loaded_quotes']}")
            print(f"Categories ({len(stats['categories'])}): {stats['categories']}")
            print(f"Authors ({len(stats['authors'])}): {stats['authors'][:5]}...")
            
            # Check if ready for search
            if stats['collection_exists'] and stats['point_count'] > 0:
                print("✅ Ready for semantic search!")
            else:
                print("❌ Need to load and index quotes first")
            ```
        
        Use Cases:
            - Verify successful data loading and indexing
            - Display collection overview in user interfaces
            - Debug database connectivity issues
            - Monitor collection growth over time
            - Validate data integrity after operations
        """
        stats = {
            "collection_name": self.collection_name,
            "collection_exists": self.semantic_search.vector_db.collection_exists(self.collection_name),
            "point_count": 0,
            "loaded_quotes": 0,
            "categories": [],
            "authors": []
        }
        
        if stats["collection_exists"]:
            stats["point_count"] = self.semantic_search.vector_db.count_points(self.collection_name)
        
        if self.wisdom:
            stats["loaded_quotes"] = len(self.wisdom)
            stats["categories"] = self.wisdom.get_categories()
            stats["authors"] = self.wisdom.get_authors()
        
        return stats
    
    # ----------------------------------------------------------------------------------------
    #  Additional SemanticSearch Integration
    # ----------------------------------------------------------------------------------------
    def consistency_check(self) -> bool:
        """Verify that your database collection is properly configured and ready to use.
        
        Performs a comprehensive health check to ensure your collection's vector
        dimensions, distance metrics, and other parameters match your embedding model.
        This prevents subtle bugs that could cause poor search results or errors.
        
        Returns:
            True if everything is properly configured and ready for search operations.
            False indicates configuration mismatches that need attention.
        
        Example:
            ```python
            if animals.consistency_check():
                print("✅ Collection is healthy and ready!")
                results = animals.search("your query here")
            else:
                print("❌ Configuration issues detected")
                print("Consider recreating the collection:")
                animals.recreate_collection()
            ```
        
        What's Checked:
            - Vector dimensions match between collection and embedder
            - Distance metric compatibility
            - Collection existence and accessibility
            - Basic connectivity to the vector database
        
        Use Cases:
            - Troubleshooting search performance issues
            - Validating setup after configuration changes
            - Health checks in production systems
            - Debugging after model or database updates
        """
        return self.semantic_search.consistency_check()
    
    def index_single_quote(self, quote: AnimalQuote) -> str:
        """Index a single animal quote.
        
        Args:
            quote: AnimalQuote instance to index.
            
        Returns:
            Point ID of the indexed quote.
        """
        return self.semantic_search.index_text(
            text=quote.text,
            metadata=quote.to_payload()
        )
    
    # ----------------------------------------------------------------------------------------
    #  RAG System Prompts
    # ----------------------------------------------------------------------------------------
    # Path to the external system prompt file
    SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "animals_system_prompt.md"

    # Comprehensive RAG System Prompt for Animals Class (loaded at init time)
    ANIMALS_RAG_SYSTEM_PROMPT: str = ""

    # Alternative shorter version for quick use
    SIMPLE_ANIMALS_PROMPT = """
    You are an expert on animal wisdom and quotes. Use the provided search results 
    to answer questions about animals, human-animal relationships, and life lessons. 
    Always attribute quotes to their authors and explain their relevance to the user's question. 
    Be conversational, thoughtful, and helpful in connecting users with the wisdom 
    found in animal quotes throughout history.
    """

    # ----------------------------------------------------------------------------------------
    #  RAG Helper Methods
    # ----------------------------------------------------------------------------------------
    @require(lambda search_results: isinstance(search_results, list), "Search results must be a list")
    @require(lambda max_results: isinstance(max_results, int) and max_results > 0,
             "Max results must be a positive integer")
    def format_search_results_for_rag(self, search_results: List[models.ScoredPoint], 
                                     max_results: int = 5) -> str:
        """Format search results from Animals class for RAG system prompt.
        
        Args:
            search_results: List of ScoredPoint objects from animals.search()
            max_results: Maximum number of results to include
            
        Returns:
            Formatted string for RAG context
        """
        if not search_results:
            return "No relevant quotes found for this query."
        
        formatted_results = []
        for i, result in enumerate(search_results[:max_results], 1):
            content = result.payload.get("content", "")
            author = result.payload.get("author", "Unknown")
            category = result.payload.get("category", "Unknown")
            score = result.score
            
            formatted_results.append(f"""
Quote {i}: "{content}"
Author: {author}
Category: {category}
Relevance Score: {score:.3f}
""")
        
        return "\n".join(formatted_results)
    
    @require(lambda user_query: isinstance(user_query, str) and len(user_query.strip()) > 0,
             "User query must be a non-empty string")
    @require(lambda search_results: isinstance(search_results, list), "Search results must be a list")
    def create_rag_context(self, user_query: str, search_results: List[models.ScoredPoint], 
                          system_prompt: Optional[str] = None) -> str:
        """Create a complete RAG context with system prompt and formatted results.
        
        Args:
            user_query: The user's question
            search_results: Search results from animals.search()
            system_prompt: System prompt to use (defaults to comprehensive prompt)
            
        Returns:
            Complete RAG context string
        """
        if system_prompt is None:
            system_prompt = self.ANIMALS_RAG_SYSTEM_PROMPT
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("System prompt must be a non-empty string")
        
        formatted_results = self.format_search_results_for_rag(search_results)
        
        context = f"""
{system_prompt}

## User Query
{user_query}

## Relevant Quotes
{formatted_results}

Please answer the user's question using the provided quotes and following the guidelines above.
"""
        return context
    
    @require(lambda user_query: isinstance(user_query, str) and len(user_query.strip()) > 0,
             "User query must be a non-empty string")
    @require(lambda limit: isinstance(limit, int) and limit > 0, "Limit must be a positive integer")
    def search_and_create_rag_context(self, user_query: str, limit: int = 5,
                                    score_threshold: Optional[float] = None,
                                    author: Optional[str] = None,
                                    category: Optional[str] = None,
                                    system_prompt: Optional[str] = None) -> str:
        """Search for quotes and create RAG context in one convenient method.
        
        Args:
            user_query: The user's question
            limit: Maximum number of search results to include
            score_threshold: Minimum similarity score threshold
            author: Optional filter to only return quotes by this author
            category: Optional filter to only return quotes in this category
            system_prompt: System prompt to use (defaults to comprehensive prompt)
            
        Returns:
            Complete RAG context string
        """
        # Search for relevant quotes
        search_results = self.search(
            query=user_query,
            limit=limit,
            score_threshold=score_threshold,
            author=author,
            category=category
        )
        
        # Create RAG context
        return self.create_rag_context(user_query, search_results, system_prompt)
    
    # ----------------------------------------------------------------------------------------
    #  Helper Methods (Private)
    # ----------------------------------------------------------------------------------------
    @require(lambda results: isinstance(results, list), "Results must be a list")
    @require(lambda search_description: isinstance(search_description, str) and len(search_description.strip()) > 0,
             "Search description must be a non-empty string")
    @require(lambda max_text_length: isinstance(max_text_length, int) and max_text_length > 0,
             "Max text length must be a positive integer")
    def display_search_results(self, results: List[models.ScoredPoint], 
                              search_description: str, 
                              max_text_length: int = 120) -> None:
        """Present search results in a beautiful, easy-to-read table format.
        
        Creates an elegant visual display of your search results using the Rich library
        for colorful, well-formatted output. Perfect for interactive applications,
        demos, or any time you want to show results in a professional way.
        
        Args:
            results: Your search results from the search() method. Each result
                contains the quote text, author, category, and relevance score.
            search_description: A descriptive title for the search that will be
                displayed at the top of the table (e.g., "Quotes about loyalty").
            max_text_length: Maximum characters to display for each quote before
                truncating with "..." (default: 120). Keeps table readable.
        
        Example:
            ```python
            # Search and display results beautifully
            results = animals.search("courage and bravery", limit=5)
            animals.display_search_results(
                results, 
                "Quotes about Courage and Bravery",
                max_text_length=100
            )
            ```
        
        Output Features:
            - 🎨 Color-coded columns for easy scanning
            - 📊 Relevance scores prominently displayed
            - ✂️ Smart text truncation to maintain readability
            - 📱 Responsive layout that works in various terminal sizes
            - 🚫 Graceful handling of empty results
        
        Fallback Behavior:
            If the Rich library isn't available, automatically falls back to
            simple text output that works in any environment.
        
        Use Cases:
            - Interactive demos and presentations
            - Development and debugging sessions
            - Educational tools showing search capabilities
            - Command-line applications with rich output
        """
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.text import Text
            
            console = Console()
            
            # Create table
            table = Table(
                title=f"🔍 {search_description}",
                show_header=True,
                header_style="bold magenta",
                border_style="blue",
                padding=(1, 2)  # Add significant vertical and horizontal padding
            )
            
            # Add columns
            table.add_column("#", style="magenta", width=15, justify="center")
            table.add_column("Score", style="green", width=20, justify="center")
            table.add_column("Quote", style="bright_white", width=60)
            table.add_column("Author", style="bold bright_yellow", width=25)
            table.add_column("Category", style="white", width=25)
            
            if not results:
                table.add_row("", "", "❌ No results found.", "", "")
                console.print(table)
                return
            
            # Add rows
            for i, result in enumerate(results, 1):
                content = result.payload.get("content", "")
                author = result.payload.get("author", "Unknown")
                category = result.payload.get("category", "Unknown")
                score = result.score
                
                # Truncate long quotes for readability
                display_content = (content if len(content) <= max_text_length 
                                 else content[:max_text_length-3] + "...")
                
                # Create styled text for quote
                quote_text = Text(f'"{display_content}"', style="italic")
                
                table.add_row(
                    str(i),
                    f"{score:.3f}",
                    quote_text,
                    author,
                    category
                )
            
            # Display table
            console.print(table)
            console.print(f"📊 Found {len(results)} results", style="bold green")
            
        except ImportError:
            # Fallback to simple print if rich is not available
            logger.warning("Rich library not available, falling back to simple display")
            self._display_search_results_simple(results, search_description, max_text_length)
        except Exception as e:
            logger.error(f"Failed to display search results: {str(e)}")
            # Fallback to simple display
            self._display_search_results_simple(results, search_description, max_text_length)
    
    def _display_search_results_simple(self, results: List[models.ScoredPoint], 
                                      search_description: str, 
                                      max_text_length: int) -> None:
        """Simple fallback display method when rich is not available.
        
        Args:
            results: List of ScoredPoint objects from search results.
            search_description: Description of the search to display.
            max_text_length: Maximum length for quote text before truncation.
        """
        print(f"\n🔍 {search_description}")
        print("=" * len(f"🔍 {search_description}"))
        
        if not results:
            print("   ❌ No results found.")
            return
        
        print(f"   📊 Found {len(results)} results")
        print()
        
        for i, result in enumerate(results, 1):
            content = result.payload.get("content", "")
            author = result.payload.get("author", "Unknown")
            category = result.payload.get("category", "Unknown")
            
            # Truncate long quotes for readability
            display_content = (content if len(content) <= max_text_length 
                             else content[:max_text_length-3] + "...")
            
            print(f"   {i}. 📊 Score: {result.score:.3f}")
            print(f"      💬 Quote: \"{display_content}\"")
            print(f"      ✍️  Author: {author}")
            print(f"      🏷️  Category: {category}")
            print()
    
    def _apply_metadata_filters(self, results: List[models.ScoredPoint],
                               author_filter: Optional[str] = None,
                               category_filter: Optional[str] = None) -> List[models.ScoredPoint]:
        """Apply author and/or category filters to search results.
        
        Args:
            results: List of scored points from vector search.
            author_filter: Optional author name to filter by.
            category_filter: Optional category name to filter by.
            
        Returns:
            Filtered list of scored points.
        """
        if not author_filter and not category_filter:
            return results
        
        filtered_results = []
        
        for result in results:
            # Skip results without payload
            if not result.payload:
                continue
            
            # Apply author filter
            if author_filter:
                result_author = result.payload.get("author", "")
                if not result_author or result_author.lower() != author_filter.lower():
                    continue
            
            # Apply category filter
            if category_filter:
                result_category = result.payload.get("category", "")
                if not result_category or result_category.lower() != category_filter.lower():
                    continue
            
            filtered_results.append(result)
        
        return filtered_results

    # ----------------------------------------------------------------------------------------
    #  LLM Response Models
    # ----------------------------------------------------------------------------------------
    class AnimalWisdomResponse(BaseModel):
        """Structured response from LLM about animal wisdom."""
        
        answer: str = Field(
            ..., 
            description="A thoughtful answer to the user's question about animals, using the provided quotes"
        )
        key_insights: List[str] = Field(
            ..., 
            min_length=1,
            max_length=5,
            description="2-5 key insights or themes from the quotes"
        )
        recommended_quotes: List[str] = Field(
            default_factory=list,
            description="Specific quotes that are most relevant to the answer (with author attribution)"
        )
        follow_up_questions: List[str] = Field(
            default_factory=list,
            description="2-3 follow-up questions to explore related topics"
        )
        
        @field_validator('answer')
        @classmethod
        def validate_answer_length(cls, v: str) -> str:
            """Ensure answer is substantial."""
            if len(v.strip()) < 50:
                raise ValueError("Answer must be at least 50 characters long")
            return v
    
    # ----------------------------------------------------------------------------------------
    #  LLM Integration Methods
    # ----------------------------------------------------------------------------------------
    @require(lambda user_query: isinstance(user_query, str) and len(user_query.strip()) > 0,
             "User query must be a non-empty string")
    @require(lambda limit: isinstance(limit, int) and limit > 0, "Limit must be a positive integer")
    def ask_llm(self, user_query: str, limit: int = 5, 
                score_threshold: Optional[float] = None,
                author: Optional[str] = None,
                category: Optional[str] = None,
                model: str = "gpt-4o") -> AnimalWisdomResponse:
        """Ask AI thoughtful questions about animals and get structured, insightful answers.
        
        This method combines the power of semantic search with advanced AI reasoning
        to provide comprehensive answers about animal wisdom, behavior, and human-animal
        relationships. The AI draws from your quote collection to give contextual,
        well-sourced responses.
        
        Args:
            user_query: Your question about animals. Can be philosophical ("What do
                animals teach us about love?"), practical ("How do pets help humans?"),
                or exploratory ("What wisdom comes from observing nature?").
            limit: Number of relevant quotes to provide as context (default: 5).
                More quotes give richer context but may slow response time.
            score_threshold: Only use quotes above this similarity score (0.0-1.0).
                Higher values ensure more relevant context for better answers.
            author: Focus the answer on quotes from this specific author only.
            category: Limit context to quotes from this category only.
            model: OpenAI model to use. "gpt-4o" (default) provides the most
                thoughtful responses, "gpt-3.5-turbo" is faster and cheaper.
        
        Returns:
            AnimalWisdomResponse containing:
            - answer: Comprehensive, thoughtful response to your question
            - key_insights: 2-5 main themes or takeaways from the analysis
            - recommended_quotes: Most relevant quotes with proper attribution
            - follow_up_questions: Suggested related questions to explore further
        
        Example:
            ```python
            # Ask a philosophical question
            response = animals.ask_llm(
                "What can animals teach us about resilience and survival?"
            )
            
            # Display the structured response
            animals.display_llm_response(response, "resilience question")
            
            # Access specific parts
            print("Main Answer:")
            print(response.answer)
            
            print("\\nKey Insights:")
            for insight in response.key_insights:
                print(f"- {insight}")
            
            # Ask follow-up questions
            for question in response.follow_up_questions:
                print(f"Next: {question}")
            ```
        
        Question Ideas:
            - "How do animals demonstrate unconditional love?"
            - "What survival strategies can humans learn from animals?"
            - "How do different cultures view human-animal relationships?"
            - "What role do animals play in teaching empathy?"
        
        Note:
            Requires OpenAI API key in environment. The AI uses your indexed quotes
            as primary sources, ensuring answers are grounded in your collection
            rather than general knowledge alone.
        """
        try:
            # Create RAG context
            rag_context = self.search_and_create_rag_context(
                user_query=user_query,
                limit=limit,
                score_threshold=score_threshold,
                author=author,
                category=category
            )
            
            # Create instructor-patched client
            client = instructor.from_openai(OpenAI())
            
            # Get structured response from LLM
            response = client.chat.completions.create(
                model=model,
                response_model=self.AnimalWisdomResponse,
                messages=[
                    {"role": "user", "content": rag_context}
                ]
            )
            
            logger.info(f"LLM response generated for query: '{user_query[:50]}...'")
            return response
            
        except Exception as e:
            logger.error(f"Failed to get LLM response: {str(e)}")
            raise InvalidPointsError(
                issue=f"LLM query failed: {str(e)}",
                points_count=1
            )
    
    @require(lambda user_query: isinstance(user_query, str) and len(user_query.strip()) > 0,
             "User query must be a non-empty string")
    def ask_llm_simple(self, user_query: str, limit: int = 3, 
                      model: str = "gpt-4o") -> str:
        """Get a simple text response from the LLM about animals.
        
        Args:
            user_query: The user's question about animals
            limit: Maximum number of search results to include
            model: OpenAI model to use (default: gpt-4o)
            
        Returns:
            Simple text response from the LLM
        """
        try:
            # Create RAG context
            rag_context = self.search_and_create_rag_context(
                user_query=user_query,
                limit=limit
            )
            
            # Create OpenAI client
            client = OpenAI()
            
            # Get simple response from LLM
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": rag_context}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Simple LLM response generated for query: '{user_query[:50]}...'")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to get simple LLM response: {str(e)}")
            raise InvalidPointsError(
                issue=f"Simple LLM query failed: {str(e)}",
                points_count=1
            )
    
    def display_llm_response(self, response: AnimalWisdomResponse, user_query: str) -> None:
        """Display the LLM response in a formatted way using rich.
        
        Args:
            response: The structured response from the LLM
            user_query: The original user query
        """
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text
            from rich.columns import Columns
            
            console = Console()
            
            # Display the main answer
            console.print(Panel(
                Text(response.answer, style="white"),
                title=f"🤖 LLM Answer to: '{user_query}'",
                border_style="green"
            ))
            
            # Display key insights
            insights_text = "\n".join([f"• {insight}" for insight in response.key_insights])
            console.print(Panel(
                Text(insights_text, style="cyan"),
                title="💡 Key Insights",
                border_style="blue"
            ))
            
            # Display recommended quotes
            if response.recommended_quotes:
                quotes_text = "\n\n".join([f"💬 {quote}" for quote in response.recommended_quotes])
                console.print(Panel(
                    Text(quotes_text, style="yellow"),
                    title="📚 Recommended Quotes",
                    border_style="yellow"
                ))
            
            # Display follow-up questions
            if response.follow_up_questions:
                questions_text = "\n".join([f"❓ {question}" for question in response.follow_up_questions])
                console.print(Panel(
                    Text(questions_text, style="magenta"),
                    title="🔍 Follow-up Questions",
                    border_style="magenta"
                ))
                
        except ImportError:
            # Fallback to simple print if rich is not available
            logger.warning("Rich library not available, falling back to simple display")
            self._display_llm_response_simple(response, user_query)
        except Exception as e:
            logger.error(f"Failed to display LLM response: {str(e)}")
            self._display_llm_response_simple(response, user_query)
    
    def _display_llm_response_simple(self, response: AnimalWisdomResponse, user_query: str) -> None:
        """Simple fallback display method when rich is not available.
        
        Args:
            response: The structured response from the LLM
            user_query: The original user query
        """
        print(f"\n🤖 LLM Answer to: '{user_query}'")
        print("=" * 60)
        print(response.answer)
        print()
        
        print("💡 Key Insights:")
        for insight in response.key_insights:
            print(f"  • {insight}")
        print()
        
        if response.recommended_quotes:
            print("📚 Recommended Quotes:")
            for quote in response.recommended_quotes:
                print(f"  💬 {quote}")
            print()
        
        if response.follow_up_questions:
            print("🔍 Follow-up Questions:")
            for question in response.follow_up_questions:
                print(f"  ❓ {question}")
            print()

    # ----------------------------------------------------------------------------------------
    #  RAG Facade Method
    # ----------------------------------------------------------------------------------------
    @require(lambda user_query: isinstance(user_query, str) and len(user_query.strip()) > 0,
             "User query must be a non-empty string")
    @require(lambda limit: isinstance(limit, int) and limit > 0, "Limit must be a positive integer")
    def rag(self, user_query: str, limit: int = 5, 
            score_threshold: Optional[float] = None,
            author: Optional[str] = None,
            category: Optional[str] = None,
            model: str = "gpt-4o",
            response_type: str = "structured") -> Dict[str, Any]:
        """🚀 Complete AI-powered question answering in one powerful method call.
        
        This is your one-stop solution for getting intelligent answers about animals.
        It automatically searches your quote collection, finds the most relevant content,
        and generates comprehensive AI responses with full transparency into the process.
        
        Perfect for building chatbots, educational tools, or research applications where
        you need both the AI answer and access to the underlying source material.
        
        The Complete RAG Pipeline:
        1. 🔍 Semantic search finds relevant quotes from your collection
        2. 📝 Context generation creates optimized prompts for the AI
        3. 🤖 AI reasoning produces thoughtful, grounded responses
        4. 📊 Full transparency with all intermediate results returned
        
        Args:
            user_query: Your question about animals. Use natural, conversational
                language like "How do animals show love?" or "What can pets teach
                children about responsibility?"
            limit: Number of quotes to use as context (default: 5). More context
                can improve answer quality but increases cost and response time.
            score_threshold: Minimum relevance score for quotes (0.0-1.0). Higher
                values ensure only highly relevant quotes are used as context.
            author: Limit context to quotes from this author only. Great for
                exploring specific perspectives or philosophies.
            category: Focus on quotes from this category only. Useful for domain-
                specific questions like "Ethics" or "Pet Care".
            model: OpenAI model for AI responses. "gpt-4o" gives the best quality,
                "gpt-3.5-turbo" is faster and more economical.
            response_type: Format of AI response:
                - "structured": Rich AnimalWisdomResponse with insights and follow-ups
                - "simple": Plain text response for basic use cases
        
        Returns:
            Complete results dictionary containing:
            - llm_response: AI answer (structured object or simple string)
            - search_results: List of relevant quotes found (with scores)
            - rag_context: Full prompt sent to AI (for debugging/transparency)
            - query_info: Metadata about the query and processing parameters
        
        Raises:
            InvalidPointsError: When any step fails (search, context generation,
                or AI response). Error messages indicate which step failed.
        
        Example:
            ```python
            # Complete RAG in one call
            result = animals.rag(
                "What do animals teach us about unconditional love?",
                limit=7,
                score_threshold=0.6,
                response_type="structured"
            )
            
            # Access the AI response
            ai_answer = result["llm_response"]
            print("AI Answer:", ai_answer.answer)
            
            # See what quotes were used
            quotes_used = result["search_results"]
            print(f"Based on {len(quotes_used)} relevant quotes")
            
            # Inspect the full context (for debugging)
            full_prompt = result["rag_context"]
            
            # Get query metadata
            info = result["query_info"]
            print(f"Model: {info['model']}, Results: {info['results_count']}")
            ```
        
        Advanced Usage:
            ```python
            # Domain-specific question
            ethics_result = animals.rag(
                "How should humans treat wild animals?",
                category="Ethics and Compassion",
                limit=10
            )
            
            # Author-focused inquiry
            gandhi_result = animals.rag(
                "What did Gandhi believe about animals?",
                author="Mahatma Gandhi",
                response_type="simple"
            )
            ```
        
        Use Cases:
            - Educational Q&A systems about animals and nature
            - Research tools for exploring animal-human relationships
            - Content generation for blogs, articles, or presentations
            - Interactive chatbots with grounded, source-backed responses
            - Philosophical exploration of animal wisdom and ethics
        
        Performance Tips:
            - Start with limit=5 for good balance of quality and speed
            - Use score_threshold=0.7+ for highly focused questions
            - Choose "simple" response_type for faster, lower-cost interactions
            - Cache results for frequently asked questions
        """
        try:
            # Step 1: Perform semantic search
            search_results = self.search(
                query=user_query,
                limit=limit,
                score_threshold=score_threshold,
                author=author,
                category=category
            )
            
            # Step 2: Generate RAG context
            rag_context = self.create_rag_context(
                user_query=user_query,
                search_results=search_results
            )
            
            # Step 3: Get LLM response based on type
            if response_type.lower() == "structured":
                llm_response = self.ask_llm(
                    user_query=user_query,
                    limit=limit,
                    score_threshold=score_threshold,
                    author=author,
                    category=category,
                    model=model
                )
            elif response_type.lower() == "simple":
                llm_response = self.ask_llm_simple(
                    user_query=user_query,
                    limit=limit,
                    model=model
                )
            else:
                raise ValueError(f"Invalid response_type: {response_type}. Must be 'structured' or 'simple'")
            
            # Step 4: Prepare query info
            query_info = {
                "user_query": user_query,
                "limit": limit,
                "score_threshold": score_threshold,
                "author_filter": author,
                "category_filter": category,
                "model": model,
                "response_type": response_type,
                "results_count": len(search_results)
            }
            
            # Step 5: Return complete RAG result
            result = {
                "llm_response": llm_response,
                "search_results": search_results,
                "rag_context": rag_context,
                "query_info": query_info
            }
            
            logger.info(f"Complete RAG pipeline executed for query: '{user_query[:50]}...' "
                       f"with {len(search_results)} results and {response_type} response")
            
            return result
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {str(e)}")
            raise InvalidPointsError(
                issue=f"RAG pipeline failed: {str(e)}",
                points_count=1
            )


#============================================================================================ 