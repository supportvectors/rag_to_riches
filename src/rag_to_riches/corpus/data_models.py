# =============================================================================
#  Filename: data_models.py
#
#  Short Description: Pydantic data models for animal quotes corpus.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


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