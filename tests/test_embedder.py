# =============================================================================
#  Filename: test_embedder.py
#
#  Short Description: Comprehensive tests for Embedder class hierarchy.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from qdrant_client import models

from rag_to_riches.vectordb.embedder import (
    Embedder,
    SimpleTextEmbedder, 
    MultimodalEmbedder,
    create_embedder
)
from rag_to_riches.exceptions import InvalidPointsError


class TestSimpleTextEmbedder:
    """Test suite for SimpleTextEmbedder class."""
    
    @pytest.fixture
    def embedder(self) -> SimpleTextEmbedder:
        """Create SimpleTextEmbedder instance for testing."""
        return SimpleTextEmbedder()
    
    @pytest.fixture
    def sample_texts(self) -> list[str]:
        """Sample texts for testing."""
        return [
            "Machine learning is transforming technology",
            "Deep learning models require large datasets",
            "Natural language processing enables text understanding",
            "Computer vision helps machines see and interpret images",
            "The golden retriever is a friendly and loyal dog breed"
        ]

    # ----------------------------------------------------------------------------------------
    #  Initialization Tests
    # ----------------------------------------------------------------------------------------
    def test_init_default_model(self):
        """Test initialization with default model."""
        embedder = SimpleTextEmbedder()
        assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedder.tokenizer is not None
        assert embedder.model is not None
        assert embedder.get_vector_size() > 0
    
    def test_init_custom_model(self):
        """Test initialization with custom model."""
        custom_model = "sentence-transformers/all-mpnet-base-v2"
        embedder = SimpleTextEmbedder(custom_model)
        assert embedder.model_name == custom_model
        assert embedder.tokenizer is not None
        assert embedder.model is not None
    
    def test_init_invalid_model(self):
        """Test initialization with invalid model name."""
        with pytest.raises(InvalidPointsError) as exc_info:
            SimpleTextEmbedder("non-existent-model-12345")
        
        assert "Failed to initialize text embedder" in str(exc_info.value)

    # ----------------------------------------------------------------------------------------
    #  Embedding Tests
    # ----------------------------------------------------------------------------------------
    def test_embed_single_text(self, embedder: SimpleTextEmbedder):
        """Test embedding a single text."""
        text = "Machine learning is fascinating"
        point = embedder.embed(text)
        
        assert isinstance(point, models.PointStruct)
        assert point.id is not None
        assert isinstance(point.vector, list)  # Qdrant converts numpy arrays to lists
        assert len(point.vector) == embedder.get_vector_size()
        assert point.payload["content"] == text
        assert point.payload["content_type"] == "text"
    
    def test_embed_with_metadata(self, embedder: SimpleTextEmbedder):
        """Test embedding with custom metadata."""
        text = "Deep learning requires GPU computing"
        metadata = {"topic": "AI", "difficulty": "advanced", "source": "research"}
        
        point = embedder.embed(text, metadata=metadata)
        
        assert point.payload["content"] == text
        assert point.payload["content_type"] == "text"
        assert point.payload["topic"] == "AI"
        assert point.payload["difficulty"] == "advanced"
        assert point.payload["source"] == "research"
    
    def test_embed_with_custom_id(self, embedder: SimpleTextEmbedder):
        """Test embedding with custom point ID."""
        text = "Natural language processing is evolving"
        custom_id = "nlp_doc_001"
        
        point = embedder.embed(text, point_id=custom_id)
        
        assert point.id == custom_id
        assert point.payload["content"] == text
    
    def test_embed_multiple_texts_consistency(self, embedder: SimpleTextEmbedder, sample_texts: list[str]):
        """Test that same text produces consistent embeddings."""
        text = sample_texts[0]
        
        point1 = embedder.embed(text)
        point2 = embedder.embed(text)
        
        # Vectors should be identical for same text
        assert point1.vector == point2.vector
    
    def test_embed_empty_text(self, embedder: SimpleTextEmbedder):
        """Test embedding empty text raises error."""
        with pytest.raises(Exception):  # icontract violation
            embedder.embed("")
    
    def test_embed_whitespace_only(self, embedder: SimpleTextEmbedder):
        """Test embedding whitespace-only text raises error."""
        with pytest.raises(Exception):  # icontract violation
            embedder.embed("   \n\t   ")

    # ----------------------------------------------------------------------------------------
    #  Vector Properties Tests
    # ----------------------------------------------------------------------------------------
    def test_get_vector_size(self, embedder: SimpleTextEmbedder):
        """Test vector size retrieval."""
        size = embedder.get_vector_size()
        assert isinstance(size, int)
        assert size > 0
        
        # Verify actual embedding matches reported size
        point = embedder.embed("test text")
        assert len(point.vector) == size
    
    def test_get_distance_metric(self, embedder: SimpleTextEmbedder):
        """Test distance metric retrieval."""
        metric = embedder.get_distance_metric()
        assert metric == "Cosine"
    
    def test_vector_normalization(self, embedder: SimpleTextEmbedder):
        """Test that vectors are properly normalized."""
        point = embedder.embed("Test vector normalization")
        vector_norm = np.linalg.norm(np.array(point.vector))
        
        # For normalized text embeddings, vectors should have unit norm
        assert 0.98 < vector_norm < 1.02  # Unit norm with small tolerance


class TestMultimodalEmbedder:
    """Test suite for MultimodalEmbedder class."""
    
    @pytest.fixture
    def embedder(self) -> MultimodalEmbedder:
        """Create MultimodalEmbedder instance for testing."""
        return MultimodalEmbedder()
    
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
        
        # Create a simple test image if files don't exist
        if not images:
            # Create a simple RGB image for testing
            test_image = Image.new('RGB', (224, 224), color='red')
            images["test"] = test_image
        
        return images
    
    @pytest.fixture
    def sample_texts(self) -> list[str]:
        """Sample texts for multimodal testing."""
        return [
            "A red sports car driving on the highway",
            "A golden retriever dog playing in the park",
            "Beautiful sunset over the ocean",
            "Modern architecture with glass buildings",
            "Fresh vegetables in a farmer's market"
        ]

    # ----------------------------------------------------------------------------------------
    #  Initialization Tests
    # ----------------------------------------------------------------------------------------
    def test_init_default_model(self):
        """Test initialization with default SigLIP model."""
        embedder = MultimodalEmbedder()
        assert embedder.model_name == "ViT-B-16-SigLIP2"
        assert embedder.pretrained == "webli"
        assert embedder.model is not None
        assert embedder.tokenizer is not None
        assert embedder.preprocess is not None
        assert embedder.get_vector_size() == 768
    
    def test_init_custom_model(self):
        """Test initialization with custom model parameters."""
        custom_model = "ViT-B-32-SigLIP2-256"  # Use a valid SigLIP model
        custom_pretrained = "webli"
        
        embedder = MultimodalEmbedder(custom_model, custom_pretrained)
        assert embedder.model_name == custom_model
        assert embedder.pretrained == custom_pretrained
    
    def test_init_invalid_model(self):
        """Test initialization with invalid model raises error."""
        with pytest.raises(InvalidPointsError) as exc_info:
            MultimodalEmbedder("non-existent-model-12345")
        
        assert "Failed to initialize multimodal embedder" in str(exc_info.value)

    # ----------------------------------------------------------------------------------------
    #  Text Embedding Tests
    # ----------------------------------------------------------------------------------------
    def test_embed_text(self, embedder: MultimodalEmbedder):
        """Test embedding text content."""
        text = "A beautiful red sports car"
        point = embedder.embed(text)
        
        assert isinstance(point, models.PointStruct)
        assert point.id is not None
        assert isinstance(point.vector, list)  # Qdrant converts numpy arrays to lists
        assert len(point.vector) == embedder.get_vector_size()
        assert point.payload["content"] == text
        assert point.payload["content_type"] == "text"
    
    def test_embed_text_with_metadata(self, embedder: MultimodalEmbedder):
        """Test embedding text with metadata."""
        text = "Golden retriever playing fetch"
        metadata = {"category": "animals", "activity": "playing"}
        
        point = embedder.embed(text, metadata=metadata)
        
        assert point.payload["content"] == text
        assert point.payload["content_type"] == "text"
        assert point.payload["category"] == "animals"
        assert point.payload["activity"] == "playing"

    # ----------------------------------------------------------------------------------------
    #  Image Embedding Tests
    # ----------------------------------------------------------------------------------------
    def test_embed_image(self, embedder: MultimodalEmbedder, test_images: Dict[str, Image.Image]):
        """Test embedding image content."""
        if not test_images:
            pytest.skip("No test images available")
        
        image_name, image = next(iter(test_images.items()))
        point = embedder.embed(image)
        
        assert isinstance(point, models.PointStruct)
        assert point.id is not None
        assert isinstance(point.vector, list)  # Qdrant converts numpy arrays to lists
        assert len(point.vector) == embedder.get_vector_size()
        assert point.payload["content_type"] == "image"
        assert point.payload["image_size"] == image.size
        assert point.payload["image_mode"] == image.mode
    
    def test_embed_image_with_metadata(self, embedder: MultimodalEmbedder, test_images: Dict[str, Image.Image]):
        """Test embedding image with metadata."""
        if not test_images:
            pytest.skip("No test images available")
        
        image_name, image = next(iter(test_images.items()))
        metadata = {"filename": f"{image_name}.jpg", "source": "test_collection"}
        
        point = embedder.embed(image, metadata=metadata)
        
        assert point.payload["content_type"] == "image"
        assert point.payload["filename"] == f"{image_name}.jpg"
        assert point.payload["source"] == "test_collection"
    
    def test_embed_different_images_different_vectors(self, embedder: MultimodalEmbedder, test_images: Dict[str, Image.Image]):
        """Test that different images produce different embeddings."""
        if len(test_images) < 2:
            pytest.skip("Need at least 2 test images")
        
        images = list(test_images.values())
        point1 = embedder.embed(images[0])
        point2 = embedder.embed(images[1])
        
        # Vectors should be different for different images
        assert point1.vector != point2.vector

    # ----------------------------------------------------------------------------------------
    #  Cross-Modal Tests
    # ----------------------------------------------------------------------------------------
    def test_text_and_image_same_vector_space(self, embedder: MultimodalEmbedder, test_images: Dict[str, Image.Image]):
        """Test that text and image embeddings are in the same vector space."""
        if not test_images:
            pytest.skip("No test images available")
        
        # Embed related text and image
        text_point = embedder.embed("A car on the road")
        image_name, image = next(iter(test_images.items()))
        image_point = embedder.embed(image)
        
        # Both should have same vector dimensions
        assert len(text_point.vector) == len(image_point.vector)
        assert len(text_point.vector) == embedder.get_vector_size()
        
        # Can compute similarity (convert to numpy arrays first)
        text_vec = np.array(text_point.vector)
        image_vec = np.array(image_point.vector)
        similarity = np.dot(text_vec, image_vec)
        assert isinstance(similarity, (float, np.floating))

    # ----------------------------------------------------------------------------------------
    #  Vector Properties Tests
    # ----------------------------------------------------------------------------------------
    def test_get_vector_size(self, embedder: MultimodalEmbedder):
        """Test vector size retrieval."""
        size = embedder.get_vector_size()
        assert size == 768  # SigLIP standard size
        
        # Verify actual embeddings match reported size
        text_point = embedder.embed("test text")
        assert len(text_point.vector) == size
    
    def test_get_distance_metric(self, embedder: MultimodalEmbedder):
        """Test distance metric retrieval."""
        metric = embedder.get_distance_metric()
        assert metric == "Cosine"
    
    def test_vector_normalization(self, embedder: MultimodalEmbedder):
        """Test that vectors are properly normalized."""
        text_point = embedder.embed("Test normalization")
        text_norm = np.linalg.norm(np.array(text_point.vector))
        
        # SigLIP vectors should be normalized
        assert abs(text_norm - 1.0) < 0.1  # Should be close to unit length

    # ----------------------------------------------------------------------------------------
    #  Error Handling Tests
    # ----------------------------------------------------------------------------------------
    def test_embed_invalid_content_type(self, embedder: MultimodalEmbedder):
        """Test embedding invalid content type."""
        with pytest.raises(Exception):  # icontract violation for invalid type
            embedder.embed(123)  # Invalid type
    
    def test_embed_empty_image(self, embedder: MultimodalEmbedder):
        """Test embedding empty image."""
        empty_image = Image.new('RGB', (0, 0))
        
        with pytest.raises(InvalidPointsError) as exc_info:
            embedder.embed(empty_image)
        
        assert "Image cannot be empty" in str(exc_info.value)


class TestEmbedderFactory:
    """Test suite for embedder factory function."""
    
    def test_create_text_embedder_default(self):
        """Test creating text embedder with default parameters."""
        embedder = create_embedder("text")
        
        assert isinstance(embedder, SimpleTextEmbedder)
        assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    
    def test_create_text_embedder_custom_model(self):
        """Test creating text embedder with custom model."""
        custom_model = "sentence-transformers/all-mpnet-base-v2"
        embedder = create_embedder("text", model_name=custom_model)
        
        assert isinstance(embedder, SimpleTextEmbedder)
        assert embedder.model_name == custom_model
    
    def test_create_multimodal_embedder_default(self):
        """Test creating multimodal embedder with default parameters."""
        embedder = create_embedder("multimodal")
        
        assert isinstance(embedder, MultimodalEmbedder)
        assert embedder.model_name == "ViT-B-16-SigLIP2"
        assert embedder.pretrained == "webli"
    
    def test_create_multimodal_embedder_custom(self):
        """Test creating multimodal embedder with custom parameters."""
        custom_model = "ViT-B-32-SigLIP2-256"  # Use a valid SigLIP model
        custom_pretrained = "webli"
        
        embedder = create_embedder("multimodal", model_name=custom_model, pretrained=custom_pretrained)
        
        assert isinstance(embedder, MultimodalEmbedder)
        assert embedder.model_name == custom_model
        assert embedder.pretrained == custom_pretrained
    
    def test_create_invalid_embedder_type(self):
        """Test creating embedder with invalid type."""
        with pytest.raises(InvalidPointsError) as exc_info:
            create_embedder("invalid_type")
        
        assert "Unsupported embedder type" in str(exc_info.value)
        assert "text" in str(exc_info.value)
        assert "multimodal" in str(exc_info.value)
    
    def test_create_embedder_case_insensitive(self):
        """Test that embedder type is case insensitive."""
        embedder1 = create_embedder("TEXT")
        embedder2 = create_embedder("Text")
        embedder3 = create_embedder("MULTIMODAL")
        
        assert isinstance(embedder1, SimpleTextEmbedder)
        assert isinstance(embedder2, SimpleTextEmbedder) 
        assert isinstance(embedder3, MultimodalEmbedder)


class TestEmbedderIntegration:
    """Integration tests for embedder functionality."""
    
    @pytest.fixture
    def text_embedder(self) -> SimpleTextEmbedder:
        """Text embedder for integration tests."""
        return SimpleTextEmbedder()
    
    @pytest.fixture
    def multimodal_embedder(self) -> MultimodalEmbedder:
        """Multimodal embedder for integration tests."""
        return MultimodalEmbedder()
    
    def test_embedder_compatibility(self, text_embedder: SimpleTextEmbedder, multimodal_embedder: MultimodalEmbedder):
        """Test that both embedders implement the same interface."""
        # Both should have the same abstract methods
        assert hasattr(text_embedder, 'embed')
        assert hasattr(text_embedder, 'get_vector_size')
        assert hasattr(text_embedder, 'get_distance_metric')
        
        assert hasattr(multimodal_embedder, 'embed')
        assert hasattr(multimodal_embedder, 'get_vector_size')
        assert hasattr(multimodal_embedder, 'get_distance_metric')
        
        # Distance metrics should be compatible
        assert text_embedder.get_distance_metric() == multimodal_embedder.get_distance_metric()
    
    def test_semantic_similarity_text(self, text_embedder: SimpleTextEmbedder):
        """Test that semantically similar texts have higher similarity."""
        # Related texts
        text1 = "Machine learning algorithms"
        text2 = "Artificial intelligence models"
        
        # Unrelated text
        text3 = "Cooking pasta recipes"
        
        point1 = text_embedder.embed(text1)
        point2 = text_embedder.embed(text2)
        point3 = text_embedder.embed(text3)
        
        # Compute similarities (convert to numpy arrays first)
        vec1 = np.array(point1.vector)
        vec2 = np.array(point2.vector)
        vec3 = np.array(point3.vector)
        sim_related = np.dot(vec1, vec2)
        sim_unrelated = np.dot(vec1, vec3)
        
        # Related texts should be more similar
        assert sim_related > sim_unrelated
    
    def test_batch_embedding_consistency(self, text_embedder: SimpleTextEmbedder):
        """Test that batch processing produces consistent results."""
        texts = [
            "Deep learning neural networks",
            "Computer vision applications", 
            "Natural language processing"
        ]
        
        # Embed individually
        individual_points = [text_embedder.embed(text) for text in texts]
        
        # Embed again (should be consistent)
        repeat_points = [text_embedder.embed(text) for text in texts]
        
        # Results should be identical
        for i in range(len(texts)):
            assert individual_points[i].vector == repeat_points[i].vector

#============================================================================================ 