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


#============================================================================================
#  Pydantic Model: AnimalQuote
#============================================================================================
class AnimalQuote(BaseModel):
    """Represents a single animal quote with metadata.
    
    This model captures the structure of each line in the animals.jsonl file,
    containing the quote text, author attribution, and thematic category.
    
    Attributes:
        text: The actual quote text content
        author: Attribution of the quote to its original author
        category: Thematic categorization for the quote
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid"
    )
    
    text: str = Field(
        ..., 
        min_length=1, 
        description="The animal quote text content"
    )
    author: str = Field(
        ..., 
        min_length=1, 
        description="The author of the quote"
    )
    category: str = Field(
        ..., 
        min_length=1, 
        description="Thematic category for the quote"
    )
    
    @field_validator('text', 'author', 'category')
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        """Ensure all string fields are non-empty after stripping."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only")
        return v.strip()
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert the quote to a payload dictionary for vector storage.
        
        Returns:
            Dictionary suitable for Qdrant point payload.
        """
        return {
            "content": self.text,
            "content_type": "animal_quote",
            "author": self.author,
            "category": self.category
        }


#============================================================================================
#  Pydantic Model: AnimalWisdom
#============================================================================================
class AnimalWisdom(BaseModel):
    """Collection of animal quotes loaded from the corpus.
    
    This model represents the complete collection of animal quotes,
    providing validation and convenient access methods for the data.
    
    Attributes:
        quotes: List of AnimalQuote instances
        source_file: Optional path to the source JSONL file
    """
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    
    quotes: List[AnimalQuote] = Field(
        ..., 
        min_length=1, 
        description="Collection of animal quotes"
    )
    source_file: Optional[Path] = Field(
        default=None,
        description="Path to the source JSONL file"
    )
    
    @field_validator('quotes')
    @classmethod
    def validate_quotes_not_empty(cls, v: List[AnimalQuote]) -> List[AnimalQuote]:
        """Ensure quotes list is not empty."""
        if not v:
            raise ValueError("Quotes collection cannot be empty")
        return v
    
    def __len__(self) -> int:
        """Return the number of quotes in the collection."""
        return len(self.quotes)
    
    def get_categories(self) -> List[str]:
        """Get unique categories from all quotes.
        
        Returns:
            Sorted list of unique category names.
        """
        categories = {quote.category for quote in self.quotes}
        return sorted(categories)
    
    def get_authors(self) -> List[str]:
        """Get unique authors from all quotes.
        
        Returns:
            Sorted list of unique author names.
        """
        authors = {quote.author for quote in self.quotes}
        return sorted(authors)
    
    def filter_by_category(self, category: str) -> List[AnimalQuote]:
        """Filter quotes by category.
        
        Args:
            category: Category name to filter by.
            
        Returns:
            List of quotes matching the category.
        """
        return [quote for quote in self.quotes if quote.category == category]
    
    def filter_by_author(self, author: str) -> List[AnimalQuote]:
        """Filter quotes by author.
        
        Args:
            author: Author name to filter by.
            
        Returns:
            List of quotes by the specified author.
        """
        return [quote for quote in self.quotes if quote.author == author]


#============================================================================================
#  Class: Animals
#============================================================================================
class Animals:
    """Loader and vectorizer for animal quotes corpus.
    
    This class handles loading animal quotes from JSONL files and provides
    domain-specific search capabilities using the underlying SemanticSearch engine.
    It acts as a specialized wrapper around SemanticSearch for animal quotes.
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
        """Initialize the Animals corpus loader.
        
        Args:
            vector_db: Vector database instance for storage.
            embedder: Text embedder for creating vector representations.
            collection_name: Name of the Qdrant collection to use.
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
    
    # ----------------------------------------------------------------------------------------
    #  Load from JSONL
    # ----------------------------------------------------------------------------------------
    @require(lambda jsonl_path: isinstance(jsonl_path, (str, Path)),
             "JSONL path must be a string or Path object")
    @ensure(lambda result: isinstance(result, AnimalWisdom),
            "Must return an AnimalWisdom instance")
    def load_from_jsonl(self, jsonl_path: Path) -> AnimalWisdom:
        """Load animal quotes from a JSONL file.
        
        Args:
            jsonl_path: Path to the JSONL file containing animal quotes.
            
        Returns:
            AnimalWisdom instance containing all loaded quotes.
            
        Raises:
            FileNotFoundError: If the JSONL file doesn't exist.
            InvalidPointsError: If the file format is invalid or corrupted.
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
        """Index all loaded quotes into the vector database.
        
        Creates embeddings for each quote and stores them as points in the
        Qdrant collection with appropriate metadata.
        
        Returns:
            List of point IDs that were indexed.
            
        Raises:
            InvalidPointsError: If quotes cannot be embedded or indexed.
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
        """Delete and recreate an empty animals collection.
        
        This method will:
        1. Delete the existing collection if it exists
        2. Create a new empty collection with the same parameters
        3. Clear the loaded wisdom data
        
        This is useful for starting fresh or clearing corrupted collections.
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
        """Convenience method to load and index quotes in one call.
        
        Args:
            jsonl_path: Path to the JSONL file containing animal quotes.
            
        Returns:
            Tuple containing the loaded AnimalWisdom and list of indexed point IDs.
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
        """Search for animal quotes using semantic similarity with optional metadata filtering.
        
        Args:
            query: Text query to search for semantically similar quotes.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold.
            author: Optional filter to only return quotes by this author.
            category: Optional filter to only return quotes in this category.
            
        Returns:
            List of scored points sorted by similarity, optionally filtered by metadata.
            
        Raises:
            InvalidPointsError: If search cannot be performed.
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
        """Get statistics about the animals collection.
        
        Returns:
            Dictionary containing collection statistics.
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
        """Check if the collection parameters are consistent with the embedder.
        
        Returns:
            True if collection is consistent with embedder requirements.
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
    # Comprehensive RAG System Prompt for Animals Class
    ANIMALS_RAG_SYSTEM_PROMPT = """
You are an expert assistant specializing in animal wisdom and quotes. You have access to a curated collection of meaningful quotes about animals from famous authors, philosophers, and thinkers throughout history.

## Your Knowledge Base
You work with search results from a vector database containing animal quotes. Each search result includes:
- **Quote Content**: The actual text of the quote
- **Author**: The person who said or wrote the quote
- **Category**: The thematic category (e.g., "Famous Literary Passages", "Proverbs and Sayings", "Reflections and Lessons")
- **Similarity Score**: How well the quote matches the user's query (higher scores = better matches)

## Your Capabilities
1. **Answer Questions**: Use the provided quotes to answer questions about animals, human-animal relationships, and wisdom
2. **Provide Context**: Explain the meaning and significance of quotes when relevant
3. **Make Connections**: Draw connections between different quotes and themes
4. **Cite Sources**: Always attribute quotes to their authors
5. **Suggest Related Topics**: Recommend other aspects of animal wisdom to explore

## Response Guidelines
- **Be Conversational**: Engage naturally while being informative
- **Use Quotes Effectively**: Reference specific quotes to support your answers
- **Provide Attribution**: Always mention the author when referencing a quote
- **Explain Relevance**: Help users understand why certain quotes are relevant to their questions
- **Be Thoughtful**: Offer insights that go beyond just listing quotes
- **Stay Focused**: Keep responses relevant to the user's query and the available quotes

## When No Relevant Quotes Are Found+
- Acknowledge that no specific quotes match the query
- Suggest alternative ways to phrase the question
- Offer to search for related themes or topics
- Explain what types of animal wisdom are available in the collection

## Example Response Structure
1. **Direct Answer**: Address the user's question using relevant quotes
2. **Quote Integration**: Seamlessly incorporate quotes with proper attribution
3. **Additional Insights**: Provide context, connections, or deeper meaning
4. **Follow-up Suggestions**: Recommend related topics or questions

## Context Format
When you receive search results, they will be formatted as:
```
Quote: "[quote text]"
Author: [author name]
Category: [category]
Score: [similarity score]
```

## Usage Instructions
1. Use the search results to answer the user's question
2. Reference specific quotes with proper attribution
3. Explain the relevance and meaning of the quotes
4. Provide thoughtful insights and connections
5. Suggest follow-up questions or related topics

Remember: You are not just a quote repository—you are a thoughtful guide helping users discover and understand the wisdom about animals that has been shared throughout human history.
"""

    # Alternative shorter version for quick use
    SIMPLE_ANIMALS_PROMPT = """
You are an expert on animal wisdom and quotes. Use the provided search results to answer questions about animals, human-animal relationships, and life lessons. Always attribute quotes to their authors and explain their relevance to the user's question. Be conversational, thoughtful, and helpful in connecting users with the wisdom found in animal quotes throughout history.
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
        """Display search results in a formatted table using rich.Table.
        
        Args:
            results: List of ScoredPoint objects from search results.
            search_description: Description of the search to display.
            max_text_length: Maximum length for quote text before truncation.
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
        """Ask the LLM a question about animals using RAG.
        
        Args:
            user_query: The user's question about animals
            limit: Maximum number of search results to include
            score_threshold: Minimum similarity score threshold
            author: Optional filter to only return quotes by this author
            category: Optional filter to only return quotes in this category
            model: OpenAI model to use (default: gpt-4o)
            
        Returns:
            Structured response from the LLM about animal wisdom
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
        """Complete RAG pipeline: search + LLM response in a single method call.
        
        This is a facade method that encapsulates the entire RAG workflow:
        1. Perform semantic search for relevant quotes
        2. Generate RAG context from search results
        3. Get LLM response (structured or simple)
        4. Return both the response and search results
        
        Args:
            user_query: The user's question about animals
            limit: Maximum number of search results to include
            score_threshold: Minimum similarity score threshold
            author: Optional filter to only return quotes by this author
            category: Optional filter to only return quotes in this category
            model: OpenAI model to use (default: gpt-4o)
            response_type: Type of LLM response - "structured" or "simple"
            
        Returns:
            Dictionary containing:
            - "llm_response": The LLM response (AnimalWisdomResponse or str)
            - "search_results": List of search results (ScoredPoint objects)
            - "rag_context": The generated RAG context string
            - "query_info": Dictionary with query metadata
            
        Raises:
            InvalidPointsError: If any step in the RAG pipeline fails
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