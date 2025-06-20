# =============================================================================
#  Filename: test_semantic_search.py
#
#  Short Description: Comprehensive tests for SemanticSearch class.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from qdrant_client import models

from rag_to_riches.search.semantic_search import SemanticSearch
from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
from rag_to_riches.vectordb.embedder import SimpleTextEmbedder, MultimodalEmbedder, create_embedder
from rag_to_riches.exceptions import (
    CollectionParameterMismatchError,
    InvalidPointsError,
    CollectionNotFoundError
)


class TestSemanticSearch:
    """Test suite for SemanticSearch class."""
    
    @pytest.fixture
    def temp_db_path(self) -> Path:
        """Create a temporary directory for test database."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup after test
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_db_path: Path):
        """Mock configuration with temporary database path."""
        return {
            "vector_db": {
                "path": str(temp_db_path)
            }
        }
    
    @pytest.fixture
    def vector_db(self, mock_config: dict) -> EmbeddedVectorDB:
        """Create EmbeddedVectorDB instance with mocked config."""
        with patch('rag_to_riches.vectordb.embedded_vectordb.config', mock_config):
            return EmbeddedVectorDB()
    
    @pytest.fixture
    def text_embedder(self) -> SimpleTextEmbedder:
        """Create text embedder for testing."""
        return SimpleTextEmbedder()
    
    @pytest.fixture
    def multimodal_embedder(self) -> MultimodalEmbedder:
        """Create multimodal embedder for testing."""
        return MultimodalEmbedder()
    
    @pytest.fixture
    def text_search(self, text_embedder: SimpleTextEmbedder, vector_db: EmbeddedVectorDB) -> SemanticSearch:
        """Create SemanticSearch instance with text embedder."""
        return SemanticSearch(text_embedder, vector_db, "text_collection")
    
    @pytest.fixture
    def multimodal_search(self, multimodal_embedder: MultimodalEmbedder, vector_db: EmbeddedVectorDB) -> SemanticSearch:
        """Create SemanticSearch instance with multimodal embedder."""
        return SemanticSearch(multimodal_embedder, vector_db, "multimodal_collection")
    
    @pytest.fixture
    def test_images(self) -> Dict[str, Image.Image]:
        """Load test images."""
        images = {}
        
        # Load car image
        car_path = Path("data/small_image_collection/car.jpg")
        if car_path.exists():
            images["car"] = Image.open(car_path)
        
        # Load dog image
        dog_path = Path("data/small_image_collection/dog.jpg")
        if dog_path.exists():
            images["dog"] = Image.open(dog_path)
        
        # Create simple test images if files don't exist
        if not images:
            images["red_car"] = Image.new('RGB', (224, 224), color='red')
            images["blue_sky"] = Image.new('RGB', (224, 224), color='blue')
        
        return images
    
    @pytest.fixture
    def sample_texts(self) -> list[str]:
        """Sample texts for testing."""
        return [
            "Machine learning algorithms are transforming technology",
            "Deep learning neural networks require massive datasets",
            "Natural language processing enables text understanding",
            "Computer vision helps machines interpret visual data",
            "The golden retriever is a friendly and intelligent dog breed",
            "Sports cars are designed for high performance and speed",
            "Artificial intelligence is revolutionizing many industries",
            "Data science combines statistics with programming skills"
        ]

    # ----------------------------------------------------------------------------------------
    #  Initialization Tests
    # ----------------------------------------------------------------------------------------
    def test_init_text_search(self, text_embedder: SimpleTextEmbedder, vector_db: EmbeddedVectorDB):
        """Test initialization with text embedder."""
        search = SemanticSearch(text_embedder, vector_db, "test_collection")
        
        assert search.embedder == text_embedder
        assert search.vector_db == vector_db
        assert search.collection_name == "test_collection"
        assert search.vector_db.collection_exists("test_collection")
    
    def test_init_multimodal_search(self, multimodal_embedder: MultimodalEmbedder, vector_db: EmbeddedVectorDB):
        """Test initialization with multimodal embedder."""
        search = SemanticSearch(multimodal_embedder, vector_db, "multimodal_test")
        
        assert search.embedder == multimodal_embedder
        assert search.vector_db == vector_db
        assert search.collection_name == "multimodal_test"
        assert search.vector_db.collection_exists("multimodal_test")
    
    def test_init_creates_collection_with_correct_parameters(self, text_embedder: SimpleTextEmbedder, vector_db: EmbeddedVectorDB):
        """Test that initialization creates collection with correct parameters."""
        collection_name = "param_test_collection"
        search = SemanticSearch(text_embedder, vector_db, collection_name)
        
        # Verify collection exists and has correct parameters
        info = vector_db.get_collection_info(collection_name)
        assert info.config.params.vectors.size == text_embedder.get_vector_size()
        assert info.config.params.vectors.distance == text_embedder.get_distance_metric()

    # ----------------------------------------------------------------------------------------
    #  Consistency Check Tests
    # ----------------------------------------------------------------------------------------
    def test_consistency_check_success(self, text_search: SemanticSearch):
        """Test successful consistency check."""
        result = text_search.consistency_check()
        assert result is True
    
    def test_consistency_check_collection_not_found(self, text_embedder: SimpleTextEmbedder, vector_db: EmbeddedVectorDB):
        """Test consistency check with non-existent collection."""
        # Create search without auto-creating collection
        search = SemanticSearch.__new__(SemanticSearch)
        search.embedder = text_embedder
        search.vector_db = vector_db
        search.collection_name = "non_existent_collection"
        
        with pytest.raises(CollectionNotFoundError) as exc_info:
            search.consistency_check()
        
        assert "non_existent_collection" in str(exc_info.value)
        assert "consistency_check" in str(exc_info.value)

    # ----------------------------------------------------------------------------------------
    #  Text Indexing Tests
    # ----------------------------------------------------------------------------------------
    def test_index_text_success(self, text_search: SemanticSearch):
        """Test successful text indexing."""
        text = "Machine learning is revolutionizing technology"
        point_id = text_search.index_text(text)
        
        assert isinstance(point_id, str)
        assert len(point_id) > 0
        
        # Verify point was stored
        count = text_search.vector_db.count_points(text_search.collection_name)
        assert count == 1
    
    def test_index_text_with_metadata(self, text_search: SemanticSearch):
        """Test text indexing with metadata."""
        text = "Deep learning requires GPU acceleration"
        metadata = {"topic": "AI", "difficulty": "advanced"}
        
        point_id = text_search.index_text(text, metadata=metadata)
        
        # Retrieve and verify metadata
        points = text_search.vector_db.get_points(text_search.collection_name, [point_id])
        assert len(points) == 1
        assert points[0].payload["topic"] == "AI"
        assert points[0].payload["difficulty"] == "advanced"
    
    def test_index_text_with_custom_id(self, text_search: SemanticSearch):
        """Test text indexing with custom ID."""
        text = "Natural language processing applications"
        custom_id = "550e8400-e29b-41d4-a716-446655440010"  # Valid UUID format
        
        returned_id = text_search.index_text(text, point_id=custom_id)
        assert returned_id == custom_id
        
        # Verify custom ID was used
        points = text_search.vector_db.get_points(text_search.collection_name, [custom_id])
        assert len(points) == 1
        assert points[0].id == custom_id
    
    def test_index_text_empty_string(self, text_search: SemanticSearch):
        """Test indexing empty text raises error."""
        with pytest.raises(Exception):  # icontract violation
            text_search.index_text("")

    # ----------------------------------------------------------------------------------------
    #  Image Indexing Tests
    # ----------------------------------------------------------------------------------------
    def test_index_image_success(self, multimodal_search: SemanticSearch, test_images: Dict[str, Image.Image]):
        """Test successful image indexing."""
        if not test_images:
            pytest.skip("No test images available")
        
        image_name, image = next(iter(test_images.items()))
        point_id = multimodal_search.index_image(image)
        
        assert isinstance(point_id, str)
        assert len(point_id) > 0
        
        # Verify point was stored
        count = multimodal_search.vector_db.count_points(multimodal_search.collection_name)
        assert count == 1
    
    def test_index_image_with_metadata(self, multimodal_search: SemanticSearch, test_images: Dict[str, Image.Image]):
        """Test image indexing with metadata."""
        if not test_images:
            pytest.skip("No test images available")
        
        image_name, image = next(iter(test_images.items()))
        metadata = {"filename": f"{image_name}.jpg", "category": "vehicle"}
        
        point_id = multimodal_search.index_image(image, metadata=metadata)
        
        # Retrieve and verify metadata
        points = multimodal_search.vector_db.get_points(multimodal_search.collection_name, [point_id])
        assert len(points) == 1
        assert points[0].payload["filename"] == f"{image_name}.jpg"
        assert points[0].payload["category"] == "vehicle"
    
    def test_index_image_with_custom_id(self, multimodal_search: SemanticSearch, test_images: Dict[str, Image.Image]):
        """Test image indexing with custom ID."""
        if not test_images:
            pytest.skip("No test images available")
        
        image_name, image = next(iter(test_images.items()))
        custom_id = "550e8400-e29b-41d4-a716-446655440011"  # Valid UUID format
        
        returned_id = multimodal_search.index_image(image, point_id=custom_id)
        assert returned_id == custom_id

    # ----------------------------------------------------------------------------------------
    #  Text Search Tests
    # ----------------------------------------------------------------------------------------
    def test_search_with_text_success(self, text_search: SemanticSearch, sample_texts: list[str]):
        """Test successful text search."""
        # Index some texts
        for text in sample_texts[:5]:
            text_search.index_text(text)
        
        # Search for similar content
        results = text_search.search_with_text("artificial intelligence machine learning", limit=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(hasattr(result, 'score') for result in results)
        assert all(hasattr(result, 'payload') for result in results)
    
    def test_search_with_text_score_threshold(self, text_search: SemanticSearch, sample_texts: list[str]):
        """Test text search with score threshold."""
        # Index texts
        for text in sample_texts[:3]:
            text_search.index_text(text)
        
        # Search with high threshold
        results = text_search.search_with_text(
            "machine learning algorithms", 
            limit=10, 
            score_threshold=0.7
        )
        
        # All results should meet threshold
        assert all(result.score >= 0.7 for result in results)
    
    def test_search_with_text_semantic_relevance(self, text_search: SemanticSearch):
        """Test that semantically similar texts rank higher."""
        # Index related and unrelated texts
        ai_text = "Artificial intelligence and machine learning"
        cooking_text = "Cooking pasta with tomato sauce"
        
        text_search.index_text(ai_text)
        text_search.index_text(cooking_text)
        
        # Search for AI-related content
        results = text_search.search_with_text("deep learning neural networks", limit=2)
        
        # AI text should rank higher than cooking text
        assert len(results) == 2
        ai_result = next(r for r in results if "intelligence" in r.payload["content"])
        cooking_result = next(r for r in results if "pasta" in r.payload["content"])
        assert ai_result.score > cooking_result.score
    
    def test_search_with_text_empty_query(self, text_search: SemanticSearch):
        """Test search with empty query raises error."""
        with pytest.raises(Exception):  # icontract violation
            text_search.search_with_text("")

    # ----------------------------------------------------------------------------------------
    #  Image Search Tests
    # ----------------------------------------------------------------------------------------
    def test_search_with_image_success(self, multimodal_search: SemanticSearch, test_images: Dict[str, Image.Image]):
        """Test successful image search."""
        if len(test_images) < 2:
            pytest.skip("Need at least 2 test images")
        
        # Index images
        for image_name, image in test_images.items():
            multimodal_search.index_image(image, metadata={"name": image_name})
        
        # Search with one of the images
        query_image = next(iter(test_images.values()))
        results = multimodal_search.search_with_image(query_image, limit=2)
        
        assert isinstance(results, list)
        assert len(results) <= 2
        assert all(hasattr(result, 'score') for result in results)
    
    def test_search_with_image_score_threshold(self, multimodal_search: SemanticSearch, test_images: Dict[str, Image.Image]):
        """Test image search with score threshold."""
        if not test_images:
            pytest.skip("No test images available")
        
        # Index an image
        image_name, image = next(iter(test_images.items()))
        multimodal_search.index_image(image)
        
        # Search with high threshold
        results = multimodal_search.search_with_image(image, limit=10, score_threshold=0.9)
        
        # Should find the exact same image with high score
        assert len(results) >= 1
        assert all(result.score >= 0.9 for result in results)

    # ----------------------------------------------------------------------------------------
    #  Batch Indexing Tests
    # ----------------------------------------------------------------------------------------
    def test_index_all_text_success(self, text_search: SemanticSearch, sample_texts: list[str]):
        """Test successful batch text indexing."""
        texts = sample_texts[:5]
        ids = text_search.index_all_text(texts)
        
        assert len(ids) == len(texts)
        assert all(isinstance(id_str, str) for id_str in ids)
        
        # Verify all texts were indexed
        count = text_search.vector_db.count_points(text_search.collection_name)
        assert count == len(texts)
    
    def test_index_all_text_with_metadata(self, text_search: SemanticSearch, sample_texts: list[str]):
        """Test batch text indexing with metadata."""
        texts = sample_texts[:3]
        metadata_list = [
            {"topic": "AI", "index": 0},
            {"topic": "ML", "index": 1},
            {"topic": "NLP", "index": 2}
        ]
        
        ids = text_search.index_all_text(texts, metadata_list=metadata_list)
        
        # Verify metadata was stored correctly
        points = text_search.vector_db.get_points(text_search.collection_name, ids)
        assert len(points) == 3
        
        topics = [point.payload["topic"] for point in points]
        assert "AI" in topics
        assert "ML" in topics
        assert "NLP" in topics
    
    def test_index_all_text_with_custom_ids(self, text_search: SemanticSearch, sample_texts: list[str]):
        """Test batch text indexing with custom IDs."""
        texts = sample_texts[:3]
        custom_ids = [
            "550e8400-e29b-41d4-a716-446655440012",
            "550e8400-e29b-41d4-a716-446655440013", 
            "550e8400-e29b-41d4-a716-446655440014"
        ]  # Valid UUID formats
        
        returned_ids = text_search.index_all_text(texts, point_ids=custom_ids)
        
        assert returned_ids == custom_ids
        
        # Verify custom IDs were used
        points = text_search.vector_db.get_points(text_search.collection_name, custom_ids)
        assert len(points) == 3
        retrieved_ids = [point.id for point in points]
        assert set(retrieved_ids) == set(custom_ids)
    
    def test_index_all_text_metadata_length_mismatch(self, text_search: SemanticSearch, sample_texts: list[str]):
        """Test batch indexing with mismatched metadata length."""
        texts = sample_texts[:3]
        metadata_list = [{"topic": "AI"}]  # Only one metadata item
        
        with pytest.raises(InvalidPointsError) as exc_info:
            text_search.index_all_text(texts, metadata_list=metadata_list)
        
        assert "Metadata list length must match texts list length" in str(exc_info.value)
    
    def test_index_all_images_success(self, multimodal_search: SemanticSearch, test_images: Dict[str, Image.Image]):
        """Test successful batch image indexing."""
        if not test_images:
            pytest.skip("No test images available")
        
        images = list(test_images.values())
        ids = multimodal_search.index_all_images(images)
        
        assert len(ids) == len(images)
        assert all(isinstance(id_str, str) for id_str in ids)
        
        # Verify all images were indexed
        count = multimodal_search.vector_db.count_points(multimodal_search.collection_name)
        assert count == len(images)
    
    def test_index_all_images_with_metadata(self, multimodal_search: SemanticSearch, test_images: Dict[str, Image.Image]):
        """Test batch image indexing with metadata."""
        if not test_images:
            pytest.skip("No test images available")
        
        images = list(test_images.values())
        image_names = list(test_images.keys())
        metadata_list = [{"name": name, "type": "test"} for name in image_names]
        
        ids = multimodal_search.index_all_images(images, metadata_list=metadata_list)
        
        # Verify metadata was stored correctly
        points = multimodal_search.vector_db.get_points(multimodal_search.collection_name, ids)
        assert len(points) == len(images)
        
        names = [point.payload["name"] for point in points]
        assert set(names) == set(image_names)

    # ----------------------------------------------------------------------------------------
    #  Cross-Modal Search Tests
    # ----------------------------------------------------------------------------------------
    def test_cross_modal_search_text_finds_images(self, multimodal_search: SemanticSearch, test_images: Dict[str, Image.Image]):
        """Test that text queries can find relevant images."""
        if "car" not in test_images:
            pytest.skip("Car image not available")
        
        # Index car image
        car_image = test_images["car"]
        multimodal_search.index_image(car_image, metadata={"type": "vehicle"})
        
        # Search with car-related text
        results = multimodal_search.search_with_text("red sports car automobile", limit=5)
        
        # Should find the car image
        assert len(results) >= 1
        vehicle_results = [r for r in results if r.payload.get("type") == "vehicle"]
        assert len(vehicle_results) >= 1
    
    def test_cross_modal_search_image_finds_text(self, multimodal_search: SemanticSearch, test_images: Dict[str, Image.Image]):
        """Test that image queries can find relevant text."""
        if not test_images:
            pytest.skip("No test images available")
        
        # Index text about cars
        multimodal_search.index_text("A fast red sports car driving on the highway")
        
        # Search with car image (if available)
        if "car" in test_images:
            car_image = test_images["car"]
            results = multimodal_search.search_with_image(car_image, limit=5)
            
            # Should find the car-related text
            assert len(results) >= 1
            text_results = [r for r in results if r.payload.get("content_type") == "text"]
            assert len(text_results) >= 1

    # ----------------------------------------------------------------------------------------
    #  Error Handling Tests
    # ----------------------------------------------------------------------------------------
    def test_design_by_contract_violations(self, text_search: SemanticSearch):
        """Test that Design-by-Contract violations raise appropriate errors."""
        # Invalid embedder type
        with pytest.raises(Exception):
            SemanticSearch("not_an_embedder", text_search.vector_db, "test")
        
        # Invalid vector_db type
        with pytest.raises(Exception):
            SemanticSearch(text_search.embedder, "not_a_vector_db", "test")
        
        # Empty collection name
        with pytest.raises(Exception):
            SemanticSearch(text_search.embedder, text_search.vector_db, "")
    
    def test_search_empty_collection(self, text_search: SemanticSearch):
        """Test search on empty collection returns empty results."""
        results = text_search.search_with_text("any query", limit=5)
        assert isinstance(results, list)
        assert len(results) == 0

    # ----------------------------------------------------------------------------------------
    #  Integration Tests
    # ----------------------------------------------------------------------------------------
    def test_full_text_workflow(self, text_search: SemanticSearch, sample_texts: list[str]):
        """Test complete text workflow."""
        # 1. Index multiple texts
        texts = sample_texts[:5]
        ids = text_search.index_all_text(texts)
        assert len(ids) == 5
        
        # 2. Search for relevant content
        results = text_search.search_with_text("machine learning technology", limit=3)
        assert len(results) <= 3
        
        # 3. Verify search quality
        ml_results = [r for r in results if "machine" in r.payload["content"].lower() or "learning" in r.payload["content"].lower()]
        assert len(ml_results) >= 1
        
        # 4. Test consistency check
        assert text_search.consistency_check() is True
    
    def test_full_multimodal_workflow(self, multimodal_search: SemanticSearch, test_images: Dict[str, Image.Image]):
        """Test complete multimodal workflow."""
        if not test_images:
            pytest.skip("No test images available")
        
        # 1. Index mixed content
        multimodal_search.index_text("A beautiful red sports car")
        
        images = list(test_images.values())[:2]  # Limit to 2 images
        multimodal_search.index_all_images(images)
        
        # 2. Cross-modal search
        text_results = multimodal_search.search_with_text("vehicle automobile", limit=5)
        assert len(text_results) >= 1
        
        # 3. Image search
        if images:
            image_results = multimodal_search.search_with_image(images[0], limit=5)
            assert len(image_results) >= 1
        
        # 4. Verify collection has mixed content
        total_count = multimodal_search.vector_db.count_points(multimodal_search.collection_name)
        assert total_count >= len(images) + 1  # At least images + 1 text

#============================================================================================ 