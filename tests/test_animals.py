# =============================================================================
#  Filename: test_animals.py
#
#  Short Description: Test suite for the Animals corpus loader.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

import pytest
import tempfile
import json
from pathlib import Path
from typing import List

from rag_to_riches.corpus.animals import AnimalQuote, AnimalWisdom, Animals
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder
from rag_to_riches.exceptions import InvalidPointsError


#============================================================================================
#  Test Data
#============================================================================================
SAMPLE_ANIMAL_QUOTES = [
    {
        "text": "The greatness of a nation can be judged by the way its animals are treated.",
        "author": "Mahatma Gandhi",
        "category": "Wisdom and Philosophy"
    },
    {
        "text": "All animals are equal, but some animals are more equal than others.",
        "author": "George Orwell",
        "category": "Literary Passages"
    },
    {
        "text": "Dogs are our link to paradise.",
        "author": "Milan Kundera",
        "category": "Proverbs and Sayings"
    }
]


#============================================================================================
#  Test Classes
#============================================================================================
class TestAnimalQuote:
    """Test suite for AnimalQuote Pydantic model."""
    
    def test_valid_quote_creation(self):
        """Test creating a valid AnimalQuote."""
        quote_data = SAMPLE_ANIMAL_QUOTES[0]
        quote = AnimalQuote(**quote_data)
        
        assert quote.text == quote_data["text"]
        assert quote.author == quote_data["author"]
        assert quote.category == quote_data["category"]
    
    def test_quote_validation_empty_text(self):
        """Test validation fails for empty text."""
        with pytest.raises(ValueError):
            AnimalQuote(text="", author="Test Author", category="Test Category")
    
    def test_quote_validation_whitespace_only(self):
        """Test validation fails for whitespace-only fields."""
        with pytest.raises(ValueError):
            AnimalQuote(text="Valid text", author="   ", category="Test Category")
    
    def test_quote_string_stripping(self):
        """Test that strings are properly stripped."""
        quote = AnimalQuote(
            text="  Leading and trailing spaces  ",
            author="  Author Name  ",
            category="  Category  "
        )
        
        assert quote.text == "Leading and trailing spaces"
        assert quote.author == "Author Name"
        assert quote.category == "Category"
    
    def test_quote_frozen_model(self):
        """Test that AnimalQuote is frozen (immutable)."""
        quote = AnimalQuote(**SAMPLE_ANIMAL_QUOTES[0])
        
        with pytest.raises(ValueError):  # Pydantic ValidationError for frozen model
            quote.text = "New text"
    
    def test_to_payload_method(self):
        """Test the to_payload method."""
        quote = AnimalQuote(**SAMPLE_ANIMAL_QUOTES[0])
        payload = quote.to_payload()
        
        expected_payload = {
            "content": quote.text,
            "content_type": "animal_quote",
            "author": quote.author,
            "category": quote.category
        }
        
        assert payload == expected_payload


class TestAnimalWisdom:
    """Test suite for AnimalWisdom Pydantic model."""
    
    @pytest.fixture
    def sample_quotes(self) -> List[AnimalQuote]:
        """Create sample AnimalQuote instances."""
        return [AnimalQuote(**quote_data) for quote_data in SAMPLE_ANIMAL_QUOTES]
    
    def test_valid_wisdom_creation(self, sample_quotes: List[AnimalQuote]):
        """Test creating a valid AnimalWisdom."""
        wisdom = AnimalWisdom(quotes=sample_quotes)
        
        assert len(wisdom) == len(sample_quotes)
        assert wisdom.quotes == sample_quotes
        assert wisdom.source_file is None
    
    def test_wisdom_with_source_file(self, sample_quotes: List[AnimalQuote]):
        """Test creating AnimalWisdom with source file."""
        source_path = Path("test.jsonl")
        wisdom = AnimalWisdom(quotes=sample_quotes, source_file=source_path)
        
        assert wisdom.source_file == source_path
    
    def test_empty_quotes_validation(self):
        """Test validation fails for empty quotes list."""
        with pytest.raises(ValueError):
            AnimalWisdom(quotes=[])
    
    def test_get_categories(self, sample_quotes: List[AnimalQuote]):
        """Test getting unique categories."""
        wisdom = AnimalWisdom(quotes=sample_quotes)
        categories = wisdom.get_categories()
        
        expected_categories = sorted(["Wisdom and Philosophy", "Literary Passages", "Proverbs and Sayings"])
        assert categories == expected_categories
    
    def test_get_authors(self, sample_quotes: List[AnimalQuote]):
        """Test getting unique authors."""
        wisdom = AnimalWisdom(quotes=sample_quotes)
        authors = wisdom.get_authors()
        
        expected_authors = sorted(["Mahatma Gandhi", "George Orwell", "Milan Kundera"])
        assert authors == expected_authors
    
    def test_filter_by_category(self, sample_quotes: List[AnimalQuote]):
        """Test filtering quotes by category."""
        wisdom = AnimalWisdom(quotes=sample_quotes)
        philosophy_quotes = wisdom.filter_by_category("Wisdom and Philosophy")
        
        assert len(philosophy_quotes) == 1
        assert philosophy_quotes[0].author == "Mahatma Gandhi"
    
    def test_filter_by_author(self, sample_quotes: List[AnimalQuote]):
        """Test filtering quotes by author."""
        wisdom = AnimalWisdom(quotes=sample_quotes)
        orwell_quotes = wisdom.filter_by_author("George Orwell")
        
        assert len(orwell_quotes) == 1
        assert orwell_quotes[0].category == "Literary Passages"


class TestAnimals:
    """Test suite for Animals corpus loader."""
    
    @pytest.fixture
    def temp_jsonl_file(self) -> Path:
        """Create a temporary JSONL file with test data."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        
        for quote_data in SAMPLE_ANIMAL_QUOTES:
            json.dump(quote_data, temp_file)
            temp_file.write('\n')
        
        temp_file.close()
        yield Path(temp_file.name)
        
        # Cleanup
        Path(temp_file.name).unlink()
    
    @pytest.fixture
    def vector_db(self) -> EmbeddedVectorDB:
        """Create a test vector database."""
        return EmbeddedVectorDB()
    
    @pytest.fixture
    def embedder(self) -> SimpleTextEmbedder:
        """Create a test embedder."""
        return SimpleTextEmbedder()
    
    @pytest.fixture
    def animals_loader(self, vector_db: EmbeddedVectorDB, 
                      embedder: SimpleTextEmbedder) -> Animals:
        """Create Animals loader instance."""
        return Animals(
            vector_db=vector_db,
            embedder=embedder,
            collection_name="test_animals"
        )
    
    def test_animals_initialization(self, animals_loader: Animals):
        """Test Animals class initialization."""
        assert animals_loader.collection_name == "test_animals"
        assert animals_loader.wisdom is None
        assert isinstance(animals_loader.vector_db, EmbeddedVectorDB)
        assert isinstance(animals_loader.embedder, SimpleTextEmbedder)
    
    def test_load_from_jsonl(self, animals_loader: Animals, temp_jsonl_file: Path):
        """Test loading quotes from JSONL file."""
        wisdom = animals_loader.load_from_jsonl(temp_jsonl_file)
        
        assert isinstance(wisdom, AnimalWisdom)
        assert len(wisdom) == len(SAMPLE_ANIMAL_QUOTES)
        assert wisdom.source_file == temp_jsonl_file
        assert animals_loader.wisdom is not None
    
    def test_load_from_nonexistent_file(self, animals_loader: Animals):
        """Test loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            animals_loader.load_from_jsonl(Path("nonexistent.jsonl"))
    
    def test_index_quotes_without_loading(self, animals_loader: Animals):
        """Test indexing quotes without loading first raises error."""
        from icontract.errors import ViolationError
        with pytest.raises(ViolationError, match="Animal wisdom must be loaded before indexing"):
            animals_loader.index_all_quotes()
    
    def test_load_and_index(self, animals_loader: Animals, temp_jsonl_file: Path):
        """Test complete load and index workflow."""
        wisdom, point_ids = animals_loader.load_and_index(temp_jsonl_file)
        
        assert isinstance(wisdom, AnimalWisdom)
        assert len(wisdom) == len(SAMPLE_ANIMAL_QUOTES)
        assert len(point_ids) == len(SAMPLE_ANIMAL_QUOTES)
        assert all(isinstance(pid, str) for pid in point_ids)
    
    def test_get_collection_stats(self, animals_loader: Animals, temp_jsonl_file: Path):
        """Test getting collection statistics."""
        # Test stats before loading
        stats = animals_loader.get_collection_stats()
        assert stats["collection_name"] == "test_animals"
        assert stats["collection_exists"] is True
        assert stats["loaded_quotes"] == 0
        
        # Load and test stats after loading
        animals_loader.load_and_index(temp_jsonl_file)
        stats = animals_loader.get_collection_stats()
        
        assert stats["point_count"] == len(SAMPLE_ANIMAL_QUOTES)
        assert stats["loaded_quotes"] == len(SAMPLE_ANIMAL_QUOTES)
        assert len(stats["categories"]) > 0
        assert len(stats["authors"]) > 0
    
    def test_search_basic(self, animals_loader: Animals, temp_jsonl_file: Path):
        """Test basic search functionality."""
        # Load and index quotes first
        animals_loader.load_and_index(temp_jsonl_file)
        
        # Perform a basic search
        results = animals_loader.search("animals equality", limit=5)
        
        assert isinstance(results, list)
        assert len(results) <= 5
        # Should find the George Orwell quote about animal equality
        assert any("equal" in result.payload.get("content", "").lower() for result in results)
    
    def test_search_with_author_filter(self, animals_loader: Animals, temp_jsonl_file: Path):
        """Test search with author filtering."""
        # Load and index quotes first
        animals_loader.load_and_index(temp_jsonl_file)
        
        # Search with author filter
        results = animals_loader.search("animals", limit=10, author="George Orwell")
        
        assert isinstance(results, list)
        # All results should be by George Orwell
        for result in results:
            assert result.payload.get("author", "") == "George Orwell"
    
    def test_search_with_category_filter(self, animals_loader: Animals, temp_jsonl_file: Path):
        """Test search with category filtering."""
        # Load and index quotes first
        animals_loader.load_and_index(temp_jsonl_file)
        
        # Search with category filter
        results = animals_loader.search("wisdom", limit=10, category="Wisdom and Philosophy")
        
        assert isinstance(results, list)
        # All results should be in the specified category
        for result in results:
            assert result.payload.get("category", "") == "Wisdom and Philosophy"
    
    def test_search_with_score_threshold(self, animals_loader: Animals, temp_jsonl_file: Path):
        """Test search with score threshold."""
        # Load and index quotes first
        animals_loader.load_and_index(temp_jsonl_file)
        
        # Search with high score threshold
        results = animals_loader.search("animals", limit=10, score_threshold=0.8)
        
        assert isinstance(results, list)
        # All results should meet the score threshold
        for result in results:
            assert result.score >= 0.8
    
    def test_search_empty_query_fails(self, animals_loader: Animals):
        """Test that empty query raises contract violation."""
        from icontract.errors import ViolationError
        with pytest.raises(ViolationError, match="Query must be a non-empty string"):
            animals_loader.search("")
    
    def test_search_invalid_limit_fails(self, animals_loader: Animals):
        """Test that invalid limit raises contract violation."""
        from icontract.errors import ViolationError
        with pytest.raises(ViolationError, match="Limit must be a positive integer"):
            animals_loader.search("test query", limit=0)


#============================================================================================ 