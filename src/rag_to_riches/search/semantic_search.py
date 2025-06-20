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


#============================================================================================
#  Class: SemanticSearch
#============================================================================================
class SemanticSearch:
    """High-level semantic search interface combining embedders and vector database.
    
    This class provides a unified interface for indexing and searching text and images
    using embeddings stored in a Qdrant vector database. It automatically handles
    collection setup, parameter validation, and provides both individual and batch
    operations for efficient content management.
    
    Examples:
        Basic text-only semantic search:
        
        >>> from rag_to_riches.vectordb.embedded_vectordb import EmbeddedVectorDB
        >>> from rag_to_riches.vectordb.embedder import create_embedder
        >>> from rag_to_riches.search.semantic_search import SemanticSearch
        >>> 
        >>> # Initialize components
        >>> embedder = create_embedder("text")
        >>> vector_db = EmbeddedVectorDB()
        >>> search = SemanticSearch(embedder, vector_db, "documents")
        >>> 
        >>> # Index individual documents
        >>> doc_id = search.index_text(
        ...     "The quick brown fox jumps over the lazy dog",
        ...     metadata={"source": "example.txt", "category": "animals"}
        ... )
        >>> 
        >>> # Search for similar content
        >>> results = search.search_with_text("fast animal jumping", limit=5)
        >>> for result in results:
        ...     print(f"Score: {result.score}, Text: {result.payload['content']}")
        
        Multimodal search with text and images:
        
        >>> from PIL import Image
        >>> 
        >>> # Initialize with multimodal embedder
        >>> embedder = create_embedder("multimodal")
        >>> vector_db = EmbeddedVectorDB()
        >>> search = SemanticSearch(embedder, vector_db, "multimodal_content")
        >>> 
        >>> # Index text content
        >>> search.index_text(
        ...     "A beautiful sunset over the ocean",
        ...     metadata={"type": "description", "location": "beach"}
        ... )
        >>> 
        >>> # Index image content
        >>> image = Image.open("sunset.jpg")
        >>> search.index_image(
        ...     image,
        ...     metadata={"type": "photo", "location": "beach", "filename": "sunset.jpg"}
        ... )
        >>> 
        >>> # Search with text query (finds both text and images)
        >>> text_results = search.search_with_text("ocean sunset", limit=10)
        >>> 
        >>> # Search with image query (finds similar images and related text)
        >>> query_image = Image.open("query_sunset.jpg")
        >>> image_results = search.search_with_image(query_image, limit=10)
        
        Batch operations for efficient indexing:
        
        >>> # Batch index multiple texts
        >>> texts = [
        ...     "Machine learning is transforming technology",
        ...     "Deep learning models require large datasets",
        ...     "Natural language processing enables text understanding"
        ... ]
        >>> metadata_list = [
        ...     {"topic": "ML", "difficulty": "beginner"},
        ...     {"topic": "DL", "difficulty": "intermediate"},
        ...     {"topic": "NLP", "difficulty": "advanced"}
        ... ]
        >>> ids = search.index_all_text(texts, metadata_list=metadata_list)
        >>> print(f"Indexed {len(ids)} documents")
        >>> 
        >>> # Batch index multiple images
        >>> images = [Image.open(f"image_{i}.jpg") for i in range(5)]
        >>> image_metadata = [{"batch": "photos", "index": i} for i in range(5)]
        >>> image_ids = search.index_all_images(images, metadata_list=image_metadata)
        
        Advanced search with score thresholds:
        
        >>> # Only return results with high similarity scores
        >>> high_quality_results = search.search_with_text(
        ...     "artificial intelligence applications",
        ...     limit=20,
        ...     score_threshold=0.8  # Only results with >80% similarity
        ... )
        >>> 
        >>> # Process results
        >>> for result in high_quality_results:
        ...     content = result.payload.get('content', 'N/A')
        ...     content_type = result.payload.get('content_type', 'unknown')
        ...     print(f"[{content_type}] Score: {result.score:.3f}")
        ...     if content_type == 'text':
        ...         print(f"Text: {content[:100]}...")
        ...     elif content_type == 'image':
        ...         print(f"Image: {result.payload.get('filename', 'unnamed')}")
        
        Error handling and consistency checks:
        
        >>> try:
        ...     # Manually verify collection compatibility
        ...     is_consistent = search.consistency_check()
        ...     print(f"Collection consistency: {is_consistent}")
        ...     
        ...     # Handle indexing errors gracefully
        ...     search.index_text("", metadata={"empty": True})  # Will raise InvalidPointsError
        ... except InvalidPointsError as e:
        ...     print(f"Indexing error: {e}")
        ... except CollectionParameterMismatchError as e:
        ...     print(f"Collection mismatch: {e}")
        
        Working with custom point IDs:
        
        >>> # Use custom IDs for better tracking
        >>> custom_id = search.index_text(
        ...     "Important document content",
        ...     metadata={"priority": "high"},
        ...     point_id="doc_2024_001"
        ... )
        >>> print(f"Indexed with custom ID: {custom_id}")
        >>> 
        >>> # Batch operations with custom IDs
        >>> texts = ["Document 1", "Document 2", "Document 3"]
        >>> custom_ids = ["doc_001", "doc_002", "doc_003"]
        >>> indexed_ids = search.index_all_text(texts, point_ids=custom_ids)
    """
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------------
    #  Consistency Check
    # ----------------------------------------------------------------------------------------
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
        
        # Get collection info
        collection_info = self.vector_db.get_collection_info(self.collection_name)
        collection_vector_size = collection_info.config.params.vectors.size
        collection_distance = collection_info.config.params.vectors.distance
        
        # Get embedder requirements
        embedder_vector_size = self.embedder.get_vector_size()
        embedder_distance = self.embedder.get_distance_metric()
        
        # Check for mismatches
        if collection_vector_size != embedder_vector_size:
            raise CollectionParameterMismatchError(
                collection_name=self.collection_name,
                parameter_name="vector_size",
                expected_value=embedder_vector_size,
                actual_value=collection_vector_size,
                embedder_type=type(self.embedder).__name__
            )
        
        if collection_distance != embedder_distance:
            raise CollectionParameterMismatchError(
                collection_name=self.collection_name,
                parameter_name="distance_metric",
                expected_value=embedder_distance,
                actual_value=collection_distance,
                embedder_type=type(self.embedder).__name__
            )
        
        logger.info(f"Consistency check passed for collection '{self.collection_name}'")
        return True

    # ----------------------------------------------------------------------------------------
    #  Index Text
    # ----------------------------------------------------------------------------------------
    @require(lambda text: isinstance(text, str) and len(text.strip()) > 0,
             "Text must be a non-empty string")
    def index_text(self, text: str, metadata: Optional[Dict[str, Any]] = None,
                   point_id: Optional[str] = None) -> str:
        """Index a text document into the collection.
        
        Args:
            text: Text content to index.
            metadata: Optional metadata to store with the text.
            point_id: Optional custom ID for the point.
            
        Returns:
            The ID of the indexed point.
            
        Raises:
            InvalidPointsError: If text cannot be embedded or indexed.
        """
        try:
            # Create embedding using the embedder
            point = self.embedder.embed(text, metadata, point_id)
            
            # Store in vector database
            self.vector_db.upsert_points(self.collection_name, [point])
            
            logger.debug(f"Indexed text (length: {len(text)}) with ID: {point.id}")
            return point.id
            
        except Exception as e:
            raise InvalidPointsError(
                issue=f"Failed to index text: {str(e)}",
                points_count=1
            )

    # ----------------------------------------------------------------------------------------
    #  Index Image
    # ----------------------------------------------------------------------------------------
    @require(lambda image: isinstance(image, Image.Image), "Image must be a PIL Image instance")
    def index_image(self, image: Image.Image, metadata: Optional[Dict[str, Any]] = None,
                    point_id: Optional[str] = None) -> str:
        """Index an image into the collection.
        
        Args:
            image: PIL Image to index.
            metadata: Optional metadata to store with the image.
            point_id: Optional custom ID for the point.
            
        Returns:
            The ID of the indexed point.
            
        Raises:
            InvalidPointsError: If image cannot be embedded or indexed.
        """
        try:
            # Create embedding using the embedder
            point = self.embedder.embed(image, metadata, point_id)
            
            # Store in vector database
            self.vector_db.upsert_points(self.collection_name, [point])
            
            logger.debug(f"Indexed image (size: {image.size}) with ID: {point.id}")
            return point.id
            
        except Exception as e:
            raise InvalidPointsError(
                issue=f"Failed to index image: {str(e)}",
                points_count=1
            )

    # ----------------------------------------------------------------------------------------
    #  Search with Text
    # ----------------------------------------------------------------------------------------
    @require(lambda query_text: isinstance(query_text, str) and len(query_text.strip()) > 0,
             "Query text must be a non-empty string")
    @require(lambda limit: isinstance(limit, int) and limit > 0,
             "Limit must be a positive integer")
    @ensure(lambda result: isinstance(result, list), "Must return a list")
    def search_with_text(self, query_text: str, limit: int = 10,
                        score_threshold: Optional[float] = None) -> List[models.ScoredPoint]:
        """Search for similar content using text query.
        
        Args:
            query_text: Text query to search for.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold.
            
        Returns:
            List of scored points sorted by similarity.
            
        Raises:
            InvalidPointsError: If query cannot be processed.
        """
        try:
            # Create query embedding
            query_point = self.embedder.embed(query_text)
            query_vector = query_point.vector
            
            # Convert numpy array to list if needed
            if hasattr(query_vector, 'tolist'):
                query_vector = query_vector.tolist()
            
            # Search in vector database
            results = self.vector_db.search_points(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            logger.info(f"Text search for '{query_text[:50]}...' returned {len(results)} results")
            return results
            
        except Exception as e:
            raise InvalidPointsError(
                issue=f"Failed to search with text query: {str(e)}",
                points_count=1
            )

    # ----------------------------------------------------------------------------------------
    #  Search with Image
    # ----------------------------------------------------------------------------------------
    @require(lambda query_image: isinstance(query_image, Image.Image),
             "Query image must be a PIL Image instance")
    @require(lambda limit: isinstance(limit, int) and limit > 0,
             "Limit must be a positive integer")
    @ensure(lambda result: isinstance(result, list), "Must return a list")
    def search_with_image(self, query_image: Image.Image, limit: int = 10,
                         score_threshold: Optional[float] = None) -> List[models.ScoredPoint]:
        """Search for similar content using image query.
        
        Args:
            query_image: PIL Image to search for.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold.
            
        Returns:
            List of scored points sorted by similarity.
            
        Raises:
            InvalidPointsError: If query cannot be processed.
        """
        try:
            # Create query embedding
            query_point = self.embedder.embed(query_image)
            query_vector = query_point.vector
            
            # Convert numpy array to list if needed
            if hasattr(query_vector, 'tolist'):
                query_vector = query_vector.tolist()
            
            # Search in vector database
            results = self.vector_db.search_points(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            logger.info(f"Image search (size: {query_image.size}) returned {len(results)} results")
            return results
            
        except Exception as e:
            raise InvalidPointsError(
                issue=f"Failed to search with image query: {str(e)}",
                points_count=1
            )

    # ----------------------------------------------------------------------------------------
    #  Index All Text
    # ----------------------------------------------------------------------------------------
    @require(lambda texts: isinstance(texts, list) and len(texts) > 0,
             "Texts must be a non-empty list")
    @require(lambda texts: all(isinstance(text, str) and len(text.strip()) > 0 for text in texts),
             "All texts must be non-empty strings")
    def index_all_text(self, texts: List[str], 
                      metadata_list: Optional[List[Dict[str, Any]]] = None,
                      point_ids: Optional[List[str]] = None) -> List[str]:
        """Index multiple text documents into the collection.
        
        Args:
            texts: List of text content to index.
            metadata_list: Optional list of metadata for each text.
            point_ids: Optional list of custom IDs for the points.
            
        Returns:
            List of IDs of the indexed points.
            
        Raises:
            InvalidPointsError: If texts cannot be embedded or indexed.
        """
        try:
            # Validate input lengths
            if metadata_list and len(metadata_list) != len(texts):
                raise InvalidPointsError(
                    issue="Metadata list length must match texts list length",
                    points_count=len(texts)
                )
            
            if point_ids and len(point_ids) != len(texts):
                raise InvalidPointsError(
                    issue="Point IDs list length must match texts list length",
                    points_count=len(texts)
                )
            
            # Create embeddings for all texts
            points = []
            indexed_ids = []
            
            for i, text in enumerate(texts):
                metadata = metadata_list[i] if metadata_list else None
                point_id = point_ids[i] if point_ids else None
                
                point = self.embedder.embed(text, metadata, point_id)
                points.append(point)
                indexed_ids.append(point.id)
            
            # Batch upsert to vector database
            self.vector_db.upsert_points(self.collection_name, points)
            
            logger.info(f"Indexed {len(texts)} texts into collection '{self.collection_name}'")
            return indexed_ids
            
        except Exception as e:
            raise InvalidPointsError(
                issue=f"Failed to index texts: {str(e)}",
                points_count=len(texts)
            )

    # ----------------------------------------------------------------------------------------
    #  Index All Images
    # ----------------------------------------------------------------------------------------
    @require(lambda images: isinstance(images, list) and len(images) > 0,
             "Images must be a non-empty list")
    @require(lambda images: all(isinstance(img, Image.Image) for img in images),
             "All items must be PIL Image instances")
    def index_all_images(self, images: List[Image.Image],
                        metadata_list: Optional[List[Dict[str, Any]]] = None,
                        point_ids: Optional[List[str]] = None) -> List[str]:
        """Index multiple images into the collection.
        
        Args:
            images: List of PIL Images to index.
            metadata_list: Optional list of metadata for each image.
            point_ids: Optional list of custom IDs for the points.
            
        Returns:
            List of IDs of the indexed points.
            
        Raises:
            InvalidPointsError: If images cannot be embedded or indexed.
        """
        try:
            # Validate input lengths
            if metadata_list and len(metadata_list) != len(images):
                raise InvalidPointsError(
                    issue="Metadata list length must match images list length",
                    points_count=len(images)
                )
            
            if point_ids and len(point_ids) != len(images):
                raise InvalidPointsError(
                    issue="Point IDs list length must match images list length",
                    points_count=len(images)
                )
            
            # Create embeddings for all images
            points = []
            indexed_ids = []
            
            for i, image in enumerate(images):
                metadata = metadata_list[i] if metadata_list else None
                point_id = point_ids[i] if point_ids else None
                
                point = self.embedder.embed(image, metadata, point_id)
                points.append(point)
                indexed_ids.append(point.id)
            
            # Batch upsert to vector database
            self.vector_db.upsert_points(self.collection_name, points)
            
            logger.info(f"Indexed {len(images)} images into collection '{self.collection_name}'")
            return indexed_ids
            
        except Exception as e:
            raise InvalidPointsError(
                issue=f"Failed to index images: {str(e)}",
                points_count=len(images)
            )

    # ----------------------------------------------------------------------------------------
    #  Helper Methods (Private)
    # ----------------------------------------------------------------------------------------
    def _ensure_collection_setup(self) -> None:
        """Helper to ensure collection exists with correct parameters."""
        vector_size = self.embedder.get_vector_size()
        distance_metric = self.embedder.get_distance_metric()
        
        # Ensure collection exists with correct parameters
        self.vector_db.ensure_collection(
            collection_name=self.collection_name,
            vector_size=vector_size,
            distance=distance_metric
        )
        
        # Run consistency check to verify everything is correct
        self.consistency_check()

#============================================================================================ 