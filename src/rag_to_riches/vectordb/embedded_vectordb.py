# =============================================================================
#  Filename: embedded_vectordb.py
#
#  Short Description: Embedded vector database client using Qdrant for local storage.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

from pathlib import Path
from typing import List, Optional
from icontract import require, ensure, invariant
from qdrant_client import QdrantClient, models
from rag_to_riches import config
from rag_to_riches.exceptions import (
    VectorDatabasePathNotFoundError,
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    InvalidVectorSizeError,
    InvalidDistanceMetricError,
    InvalidPointsError,
)
from loguru import logger

#============================================================================================
#  Class: EmbeddedVectorDB
#============================================================================================
@invariant(lambda self: hasattr(self, 'client') and self.client is not None, 
           "Client must be initialized and not None")
class EmbeddedVectorDB:
    """Embedded vector database client using Qdrant for local storage.
    
    This class provides a simple interface to interact with a local Qdrant
    vector database, including collection management and point operations.
    """
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    @require(lambda: "vector_db" in config, "Config must contain vector_db section")
    @require(lambda: "path" in config["vector_db"], "Config must contain vector_db.path")
    @ensure(lambda self: hasattr(self, 'client'), "Client must be initialized after construction")
    def __init__(self) -> None:
        """Initialize the embedded vector database client.
        
        Raises:
            VectorDatabasePathNotFoundError: If the vector database path doesn't exist.
        """
        path = config["vector_db"]["path"]
        
        if not Path(path).exists():
            raise VectorDatabasePathNotFoundError(
                path=path,
                suggestion="Create the directory or update your configuration"
            )
            
        self.client = QdrantClient(path=path)
        logger.info(f"Connected to embedded vector database at {path}")

    # ----------------------------------------------------------------------------------------
    #  Collection Exists
    # ----------------------------------------------------------------------------------------
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.
        
        Args:
            collection_name: Name of the collection to check.
            
        Returns:
            True if collection exists, False otherwise.
        """
        return self.client.collection_exists(collection_name)

    # ----------------------------------------------------------------------------------------
    #  Count Points
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    def count_points(self, collection_name: str) -> int:
        """Count the number of points in a collection.
        
        Args:
            collection_name: Name of the collection.
            
        Returns:
            Number of points in the collection.
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist.
        """
        if not self.client.collection_exists(collection_name):
            try:
                available_collections = [col.name for col in self.client.get_collections().collections]
            except Exception:
                available_collections = None
                
            raise CollectionNotFoundError(
                collection_name=collection_name,
                operation="count_points",
                available_collections=available_collections
            )
        
        info = self.client.get_collection(collection_name)
        count = info.points_count
        logger.info(f"Collection '{collection_name}' contains {count} points")
        return count

    # ----------------------------------------------------------------------------------------
    #  Create Collection
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    @require(lambda vector_size: isinstance(vector_size, int) and vector_size > 0,
             "Vector size must be a positive integer")
    @require(lambda distance: isinstance(distance, str) and distance.strip() != "",
             "Distance must be a non-empty string")
    @ensure(lambda result: isinstance(result, bool), "Must return boolean")
    def create_collection(self, collection_name: str, vector_size: int, 
                         distance: str) -> bool:
        """Create a new collection in the vector database.
        
        Args:
            collection_name: Name of the collection to create.
            vector_size: Size of the vectors to be stored.
            distance: Distance metric to use (e.g., 'Cosine', 'Euclidean', 'Dot').
            
        Returns:
            True if collection was created successfully.
            
        Raises:
            CollectionAlreadyExistsError: If collection already exists.
            InvalidVectorSizeError: If vector size is invalid.
            InvalidDistanceMetricError: If distance metric is invalid.
        """
        # Additional validation with custom exceptions
        if not isinstance(vector_size, int) or vector_size <= 0:
            raise InvalidVectorSizeError(vector_size)
            
        valid_distances = ["Cosine", "Euclid", "Dot"]
        if distance not in valid_distances:
            raise InvalidDistanceMetricError(distance, valid_distances)
        
        if self.client.collection_exists(collection_name):
            raise CollectionAlreadyExistsError(
                collection_name=collection_name,
                vector_size=vector_size,
                distance=distance
            )
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance)
        )
        logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")
        return self.client.collection_exists(collection_name)

    # ----------------------------------------------------------------------------------------
    #  Delete Collection
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the vector database.
        
        Args:
            collection_name: Name of the collection to delete.
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist.
        """
        if not self.client.collection_exists(collection_name):
            # Get available collections for better error message
            try:
                available_collections = [col.name for col in self.client.get_collections().collections]
            except Exception:
                available_collections = None
                
            raise CollectionNotFoundError(
                collection_name=collection_name,
                operation="delete",
                available_collections=available_collections
            )
        
        self.client.delete_collection(collection_name)
        logger.info(f"Deleted collection '{collection_name}'")

    # ----------------------------------------------------------------------------------------
    #  Delete Points
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    @require(lambda point_ids: isinstance(point_ids, list) and len(point_ids) > 0,
             "Point IDs must be a non-empty list")
    def delete_points(self, collection_name: str, point_ids: List[str]) -> None:
        """Delete points by their IDs from a collection.
        
        Args:
            collection_name: Name of the collection.
            point_ids: List of point IDs to delete.
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist.
        """
        if not self.client.collection_exists(collection_name):
            try:
                available_collections = [col.name for col in self.client.get_collections().collections]
            except Exception:
                available_collections = None
                
            raise CollectionNotFoundError(
                collection_name=collection_name,
                operation="delete_points",
                available_collections=available_collections
            )
        
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=point_ids)
        )
        logger.info(f"Deleted {len(point_ids)} points from collection '{collection_name}'")

    # ----------------------------------------------------------------------------------------
    #  Ensure Collection
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    @require(lambda vector_size: isinstance(vector_size, int) and vector_size > 0,
             "Vector size must be a positive integer")
    @require(lambda distance: isinstance(distance, str) and distance.strip() != "",
             "Distance must be a non-empty string")
    @ensure(lambda result: isinstance(result, bool) and result is True,
            "Must return True when collection is ensured")
    def ensure_collection(self, collection_name: str, vector_size: int, 
                         distance: str) -> bool:
        """Ensure a collection exists with the specified parameters.
        
        Creates the collection if it doesn't exist, or recreates it if the
        existing collection has different vector size or distance parameters.
        
        Args:
            collection_name: Name of the collection.
            vector_size: Required vector size.
            distance: Required distance metric.
            
        Returns:
            True if collection exists with correct parameters.
            
        Raises:
            InvalidVectorSizeError: If vector size is invalid.
            InvalidDistanceMetricError: If distance metric is invalid.
        """
        # Validate parameters early
        if not isinstance(vector_size, int) or vector_size <= 0:
            raise InvalidVectorSizeError(vector_size)
            
        valid_distances = ["Cosine", "Euclid", "Dot"]
        if distance not in valid_distances:
            raise InvalidDistanceMetricError(distance, valid_distances)
        
        if not self.client.collection_exists(collection_name):
            return self._create_new_collection(collection_name, vector_size, distance)
            
        return self._ensure_existing_collection_matches(collection_name, vector_size, distance)

    # ----------------------------------------------------------------------------------------
    #  Get Collection Info
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    def get_collection_info(self, collection_name: str) -> dict:
        """Get detailed information about a collection.
        
        Args:
            collection_name: Name of the collection.
            
        Returns:
            Dictionary containing collection information.
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist.
        """
        if not self.client.collection_exists(collection_name):
            try:
                available_collections = [col.name for col in self.client.get_collections().collections]
            except Exception:
                available_collections = None
                
            raise CollectionNotFoundError(
                collection_name=collection_name,
                operation="get_collection_info",
                available_collections=available_collections
            )
        
        info = self.client.get_collection(collection_name)
        logger.info(f"Retrieved info for collection '{collection_name}'")
        return info

    # ----------------------------------------------------------------------------------------
    #  Get Points
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    @require(lambda point_ids: isinstance(point_ids, list) and len(point_ids) > 0,
             "Point IDs must be a non-empty list")
    def get_points(self, collection_name: str, point_ids: List[str]) -> List[models.Record]:
        """Retrieve points by their IDs from a collection.
        
        Args:
            collection_name: Name of the collection.
            point_ids: List of point IDs to retrieve.
            
        Returns:
            List of point records.
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist.
        """
        if not self.client.collection_exists(collection_name):
            try:
                available_collections = [col.name for col in self.client.get_collections().collections]
            except Exception:
                available_collections = None
                
            raise CollectionNotFoundError(
                collection_name=collection_name,
                operation="get_points",
                available_collections=available_collections
            )
        
        results = self.client.retrieve(
            collection_name=collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=True
        )
        logger.info(f"Retrieved {len(results)} points from collection '{collection_name}'")
        return results

    # ----------------------------------------------------------------------------------------
    #  List Collections
    # ----------------------------------------------------------------------------------------
    def list_collections(self) -> List[str]:
        """List all available collections in the database.
        
        Returns:
            List of collection names.
        """
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            logger.info(f"Found {len(collection_names)} collections")
            return collection_names
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    # ----------------------------------------------------------------------------------------
    #  Search Points
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    @require(lambda query_vector: isinstance(query_vector, list) and len(query_vector) > 0,
             "Query vector must be a non-empty list")
    @require(lambda query_vector: all(isinstance(x, (int, float)) for x in query_vector),
             "Query vector must contain only numbers")
    @require(lambda limit: isinstance(limit, int) and limit > 0,
             "Limit must be a positive integer")
    @ensure(lambda result: isinstance(result, list), "Must return a list")
    def search_points(self, collection_name: str, query_vector: List[float], 
                     limit: int = 10, score_threshold: Optional[float] = None) -> List[models.ScoredPoint]:
        """Search for similar points in a collection using vector similarity.
        
        Args:
            collection_name: Name of the collection to search in.
            query_vector: Vector to search for similar points.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold.
            
        Returns:
            List of scored points sorted by similarity.
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist.
            InvalidPointsError: If query vector is invalid.
        """
        if not self.client.collection_exists(collection_name):
            try:
                available_collections = [col.name for col in self.client.get_collections().collections]
            except Exception:
                available_collections = None
                
            raise CollectionNotFoundError(
                collection_name=collection_name,
                operation="search_points",
                available_collections=available_collections
            )
        
        search_params = {
            "collection_name": collection_name,
            "query": query_vector,
            "limit": limit
        }
        
        if score_threshold is not None:
            search_params["score_threshold"] = score_threshold
        
        results = self.client.query_points(**search_params).points
        logger.info(f"Found {len(results)} points in collection '{collection_name}'")
        return results

    # ----------------------------------------------------------------------------------------
    #  Upsert Points
    # ----------------------------------------------------------------------------------------
    @require(lambda collection_name: isinstance(collection_name, str) and 
             len(collection_name.strip()) > 0, "Collection name must be a non-empty string")
    @require(lambda points: isinstance(points, list) and len(points) > 0,
             "Points must be a non-empty list")
    @require(lambda points: all(isinstance(p, models.PointStruct) for p in points),
             "All points must be PointStruct instances")
    def upsert_points(self, collection_name: str, 
                     points: List[models.PointStruct]) -> None:
        """Insert or update points in a collection.
        
        Args:
            collection_name: Name of the collection to upsert points into.
            points: List of point structures to upsert.
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist.
        """
        if not self.client.collection_exists(collection_name):
            # Get available collections for better error message
            try:
                available_collections = [col.name for col in self.client.get_collections().collections]
            except Exception:
                available_collections = None
                
            raise CollectionNotFoundError(
                collection_name=collection_name,
                operation="upsert_points",
                available_collections=available_collections
            )
            
        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Upserted {len(points)} points to collection '{collection_name}'")

    # ----------------------------------------------------------------------------------------
    #  Helper Methods (Private)
    # ----------------------------------------------------------------------------------------
    def _create_new_collection(self, collection_name: str, vector_size: int, 
                              distance: str) -> bool:
        """Helper to create a new collection."""
        self.create_collection(collection_name, vector_size, distance)
        return True
        
    def _ensure_existing_collection_matches(self, collection_name: str, 
                                          vector_size: int, distance: str) -> bool:
        """Helper to ensure existing collection matches required parameters."""
        collection_info = self.client.get_collection(collection_name)
        existing_size = collection_info.config.params.vectors.size
        existing_distance = collection_info.config.params.vectors.distance
        
        if self._parameters_match(existing_size, existing_distance, vector_size, distance):
            logger.info(f"Collection '{collection_name}' exists with correct parameters")
            return True
            
        self._recreate_collection_with_warning(
            collection_name, vector_size, distance, existing_size, existing_distance
        )
        return True
    
    def _parameters_match(self, existing_size: int, existing_distance: str,
                         required_size: int, required_distance: str) -> bool:
        """Helper to check if collection parameters match requirements."""
        return (existing_size == required_size and 
                existing_distance == required_distance)
    
    def _recreate_collection_with_warning(self, collection_name: str, vector_size: int, 
                                        distance: str, existing_size: int, 
                                        existing_distance: str) -> None:
        """Helper to recreate collection when parameters don't match."""
        logger.warning(
            f"Collection '{collection_name}' exists but has mismatched parameters. "
            f"Expected: size={vector_size}, distance={distance}. "
            f"Found: size={existing_size}, distance={existing_distance}. "
            "Recreating collection."
        )
        self.delete_collection(collection_name)
        self.create_collection(collection_name, vector_size, distance)

#============================================================================================