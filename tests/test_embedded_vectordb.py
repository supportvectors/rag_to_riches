# =============================================================================
#  Filename: test_embedded_vectordb.py
#
#  Short Description: Comprehensive tests for EmbeddedVectorDB class.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock

from qdrant_client import models

from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
from rag_to_riches.exceptions import (
    VectorDatabasePathNotFoundError,
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    InvalidVectorSizeError,
    InvalidDistanceMetricError,
    InvalidPointsError,
    CollectionParameterMismatchError
)


class TestEmbeddedVectorDB:
    """Test suite for EmbeddedVectorDB class."""
    
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
    def sample_points(self) -> List[models.PointStruct]:
        """Create sample points for testing."""
        return [
            models.PointStruct(
                id="550e8400-e29b-41d4-a716-446655440001",
                vector=[0.1, 0.2, 0.3, 0.4],
                payload={"text": "Machine learning is fascinating", "category": "tech"}
            ),
            models.PointStruct(
                id="550e8400-e29b-41d4-a716-446655440002", 
                vector=[0.2, 0.3, 0.4, 0.5],
                payload={"text": "Deep learning requires large datasets", "category": "tech"}
            ),
            models.PointStruct(
                id="550e8400-e29b-41d4-a716-446655440003",
                vector=[0.8, 0.7, 0.6, 0.5],
                payload={"text": "The golden retriever is a friendly dog", "category": "animals"}
            )
        ]

    # ----------------------------------------------------------------------------------------
    #  Initialization Tests
    # ----------------------------------------------------------------------------------------
    def test_init_success(self, mock_config: dict):
        """Test successful initialization."""
        with patch('rag_to_riches.vectordb.embedded_vectordb.config', mock_config):
            db = EmbeddedVectorDB()
            assert db.client is not None
    
    def test_init_path_not_found(self):
        """Test initialization with non-existent path."""
        mock_config = {
            "vector_db": {
                "path": "/non/existent/path"
            }
        }
        
        with patch('rag_to_riches.vectordb.embedded_vectordb.config', mock_config):
            with pytest.raises(VectorDatabasePathNotFoundError) as exc_info:
                EmbeddedVectorDB()
            
            assert "/non/existent/path" in str(exc_info.value)
            assert "Create the directory" in str(exc_info.value)

    # ----------------------------------------------------------------------------------------
    #  Collection Management Tests
    # ----------------------------------------------------------------------------------------
    def test_create_collection_success(self, vector_db: EmbeddedVectorDB):
        """Test successful collection creation."""
        result = vector_db.create_collection("test_collection", 4, "Cosine")
        assert result is True
        assert vector_db.collection_exists("test_collection")
    
    def test_create_collection_already_exists(self, vector_db: EmbeddedVectorDB):
        """Test creating collection that already exists."""
        vector_db.create_collection("test_collection", 4, "Cosine")
        
        with pytest.raises(CollectionAlreadyExistsError) as exc_info:
            vector_db.create_collection("test_collection", 4, "Cosine")
        
        assert "test_collection" in str(exc_info.value)
        assert "already exists" in str(exc_info.value)
    
    def test_create_collection_invalid_vector_size(self, vector_db: EmbeddedVectorDB):
        """Test creating collection with invalid vector size."""
        with pytest.raises(Exception):  # icontract violation for negative vector size
            vector_db.create_collection("test_collection", -1, "Cosine")
    
    def test_create_collection_invalid_distance(self, vector_db: EmbeddedVectorDB):
        """Test creating collection with invalid distance metric."""
        with pytest.raises(InvalidDistanceMetricError) as exc_info:
            vector_db.create_collection("test_collection", 4, "InvalidMetric")
        
        assert "InvalidMetric" in str(exc_info.value)
        assert "Cosine" in str(exc_info.value)
        assert "Euclid" in str(exc_info.value)
    
    def test_delete_collection_success(self, vector_db: EmbeddedVectorDB):
        """Test successful collection deletion."""
        vector_db.create_collection("test_collection", 4, "Cosine")
        vector_db.delete_collection("test_collection")
        assert not vector_db.collection_exists("test_collection")
    
    def test_delete_collection_not_found(self, vector_db: EmbeddedVectorDB):
        """Test deleting non-existent collection."""
        with pytest.raises(CollectionNotFoundError) as exc_info:
            vector_db.delete_collection("non_existent")
        
        assert "non_existent" in str(exc_info.value)
        assert "delete" in str(exc_info.value)
    
    def test_ensure_collection_new(self, vector_db: EmbeddedVectorDB):
        """Test ensure_collection creates new collection."""
        result = vector_db.ensure_collection("new_collection", 4, "Cosine")
        assert result is True
        assert vector_db.collection_exists("new_collection")
    
    def test_ensure_collection_existing_match(self, vector_db: EmbeddedVectorDB):
        """Test ensure_collection with matching existing collection."""
        vector_db.create_collection("test_collection", 4, "Cosine")
        result = vector_db.ensure_collection("test_collection", 4, "Cosine")
        assert result is True
    
    def test_ensure_collection_existing_mismatch(self, vector_db: EmbeddedVectorDB):
        """Test ensure_collection recreates collection with different parameters."""
        vector_db.create_collection("test_collection", 4, "Cosine")
        
        # This should recreate the collection with new parameters
        result = vector_db.ensure_collection("test_collection", 8, "Euclid")
        assert result is True
        
        # Verify new parameters
        info = vector_db.get_collection_info("test_collection")
        assert info.config.params.vectors.size == 8
        assert info.config.params.vectors.distance == "Euclid"

    # ----------------------------------------------------------------------------------------
    #  Point Operations Tests
    # ----------------------------------------------------------------------------------------
    def test_upsert_points_success(self, vector_db: EmbeddedVectorDB, sample_points: List[models.PointStruct]):
        """Test successful point insertion."""
        vector_db.create_collection("test_collection", 4, "Cosine")
        vector_db.upsert_points("test_collection", sample_points)
        
        count = vector_db.count_points("test_collection")
        assert count == len(sample_points)
    
    def test_upsert_points_collection_not_found(self, vector_db: EmbeddedVectorDB, sample_points: List[models.PointStruct]):
        """Test upserting points to non-existent collection."""
        with pytest.raises(CollectionNotFoundError) as exc_info:
            vector_db.upsert_points("non_existent", sample_points)
        
        assert "non_existent" in str(exc_info.value)
        assert "upsert_points" in str(exc_info.value)
    
    def test_search_points_success(self, vector_db: EmbeddedVectorDB, sample_points: List[models.PointStruct]):
        """Test successful point search."""
        vector_db.create_collection("test_collection", 4, "Cosine")
        vector_db.upsert_points("test_collection", sample_points)
        
        # Search with a query vector similar to first point
        results = vector_db.search_points("test_collection", [0.1, 0.2, 0.3, 0.4], limit=2)
        
        assert len(results) <= 2
        assert all(hasattr(result, 'score') for result in results)
        assert all(hasattr(result, 'payload') for result in results)
    
    def test_search_points_with_threshold(self, vector_db: EmbeddedVectorDB, sample_points: List[models.PointStruct]):
        """Test point search with score threshold."""
        vector_db.create_collection("test_collection", 4, "Cosine")
        vector_db.upsert_points("test_collection", sample_points)
        
        results = vector_db.search_points(
            "test_collection", 
            [0.1, 0.2, 0.3, 0.4], 
            limit=10, 
            score_threshold=0.9
        )
        
        # Should return fewer results due to high threshold
        assert all(result.score >= 0.9 for result in results)
    
    def test_get_points_success(self, vector_db: EmbeddedVectorDB, sample_points: List[models.PointStruct]):
        """Test successful point retrieval."""
        vector_db.create_collection("test_collection", 4, "Cosine")
        vector_db.upsert_points("test_collection", sample_points)
        
        results = vector_db.get_points("test_collection", ["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002"])
        
        assert len(results) == 2
        retrieved_ids = {result.id for result in results}
        assert "550e8400-e29b-41d4-a716-446655440001" in retrieved_ids
        assert "550e8400-e29b-41d4-a716-446655440002" in retrieved_ids
    
    def test_delete_points_success(self, vector_db: EmbeddedVectorDB, sample_points: List[models.PointStruct]):
        """Test successful point deletion."""
        vector_db.create_collection("test_collection", 4, "Cosine")
        vector_db.upsert_points("test_collection", sample_points)
        
        initial_count = vector_db.count_points("test_collection")
        vector_db.delete_points("test_collection", ["550e8400-e29b-41d4-a716-446655440001"])
        final_count = vector_db.count_points("test_collection")
        
        assert final_count == initial_count - 1
    
    def test_count_points_success(self, vector_db: EmbeddedVectorDB, sample_points: List[models.PointStruct]):
        """Test point counting."""
        vector_db.create_collection("test_collection", 4, "Cosine")
        
        initial_count = vector_db.count_points("test_collection")
        assert initial_count == 0
        
        vector_db.upsert_points("test_collection", sample_points)
        final_count = vector_db.count_points("test_collection")
        assert final_count == len(sample_points)

    # ----------------------------------------------------------------------------------------
    #  Information Retrieval Tests
    # ----------------------------------------------------------------------------------------
    def test_list_collections_empty(self, vector_db: EmbeddedVectorDB):
        """Test listing collections when none exist."""
        collections = vector_db.list_collections()
        assert isinstance(collections, list)
        assert len(collections) == 0
    
    def test_list_collections_with_data(self, vector_db: EmbeddedVectorDB):
        """Test listing collections with existing collections."""
        vector_db.create_collection("collection_1", 4, "Cosine")
        vector_db.create_collection("collection_2", 8, "Euclid")
        
        collections = vector_db.list_collections()
        assert len(collections) == 2
        assert "collection_1" in collections
        assert "collection_2" in collections
    
    def test_get_collection_info_success(self, vector_db: EmbeddedVectorDB):
        """Test getting collection information."""
        vector_db.create_collection("test_collection", 4, "Cosine")
        
        info = vector_db.get_collection_info("test_collection")
        assert info.config.params.vectors.size == 4
        assert info.config.params.vectors.distance == "Cosine"
    
    def test_collection_exists(self, vector_db: EmbeddedVectorDB):
        """Test collection existence check."""
        assert not vector_db.collection_exists("non_existent")
        
        vector_db.create_collection("test_collection", 4, "Cosine")
        assert vector_db.collection_exists("test_collection")

    # ----------------------------------------------------------------------------------------
    #  Edge Cases and Error Handling Tests
    # ----------------------------------------------------------------------------------------
    def test_design_by_contract_violations(self, vector_db: EmbeddedVectorDB):
        """Test that Design-by-Contract violations raise appropriate errors."""
        # Empty collection name
        with pytest.raises(Exception):  # icontract will raise ContractViolation
            vector_db.create_collection("", 4, "Cosine")
        
        # Invalid vector size
        with pytest.raises(Exception):
            vector_db.create_collection("test", 0, "Cosine")
        
        # Empty distance metric
        with pytest.raises(Exception):
            vector_db.create_collection("test", 4, "")
    
    def test_upsert_empty_points_list(self, vector_db: EmbeddedVectorDB):
        """Test upserting empty points list."""
        vector_db.create_collection("test_collection", 4, "Cosine")
        
        with pytest.raises(Exception):  # icontract violation
            vector_db.upsert_points("test_collection", [])
    
    def test_search_invalid_query_vector(self, vector_db: EmbeddedVectorDB):
        """Test search with invalid query vector."""
        vector_db.create_collection("test_collection", 4, "Cosine")
        
        # Empty vector
        with pytest.raises(Exception):
            vector_db.search_points("test_collection", [], limit=5)
        
        # Non-numeric values
        with pytest.raises(Exception):
            vector_db.search_points("test_collection", ["a", "b", "c", "d"], limit=5)

    # ----------------------------------------------------------------------------------------
    #  Integration Tests
    # ----------------------------------------------------------------------------------------
    def test_full_workflow(self, vector_db: EmbeddedVectorDB):
        """Test complete workflow from collection creation to search."""
        collection_name = "integration_test"
        
        # 1. Create collection
        vector_db.create_collection(collection_name, 4, "Cosine")
        assert vector_db.collection_exists(collection_name)
        
        # 2. Insert points
        points = [
            models.PointStruct(
                id=f"550e8400-e29b-41d4-a716-44665544000{i}",
                vector=[i*0.1, i*0.2, i*0.3, i*0.4],
                payload={"content": f"Document {i}", "index": i}
            )
            for i in range(1, 6)
        ]
        vector_db.upsert_points(collection_name, points)
        
        # 3. Verify count
        assert vector_db.count_points(collection_name) == 5
        
        # 4. Search
        results = vector_db.search_points(collection_name, [0.1, 0.2, 0.3, 0.4], limit=3)
        assert len(results) <= 3
        
        # 5. Get specific points
        retrieved = vector_db.get_points(collection_name, ["550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002"])
        assert len(retrieved) == 2
        
        # 6. Delete some points
        vector_db.delete_points(collection_name, ["550e8400-e29b-41d4-a716-446655440001"])
        assert vector_db.count_points(collection_name) == 4
        
        # 7. Clean up
        vector_db.delete_collection(collection_name)
        assert not vector_db.collection_exists(collection_name)

#============================================================================================ 