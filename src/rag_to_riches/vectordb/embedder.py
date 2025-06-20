# =============================================================================
#  Filename: embedder.py
#
#  Short Description: Embedder class hierarchy for converting text and images to vector embeddings.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from uuid import uuid4
from PIL import Image
from icontract import require, ensure
from qdrant_client import models
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import open_clip
from loguru import logger
from rag_to_riches.exceptions import InvalidPointsError


#============================================================================================
#  Abstract Base Class: Embedder
#============================================================================================
class Embedder(ABC):
    """Abstract base class for all embedders.
    
    Defines the interface for converting various input types (text, images)
    into vector embeddings as PointStruct objects for vector database storage.
    """
    
    # ----------------------------------------------------------------------------------------
    #  Abstract Method: Embed Content
    # ----------------------------------------------------------------------------------------
    @abstractmethod
    def embed(self, content: Union[str, Image.Image], metadata: Optional[Dict[str, Any]] = None,
              point_id: Optional[str] = None) -> models.PointStruct:
        """Convert content to a vector embedding as a PointStruct.
        
        Args:
            content: The content to embed (text string or PIL Image).
            metadata: Optional metadata to store with the point.
            point_id: Optional custom ID for the point. If None, generates UUID.
            
        Returns:
            PointStruct containing the embedding vector and metadata.
            
        Raises:
            InvalidPointsError: If content cannot be embedded.
        """
        pass

    # ----------------------------------------------------------------------------------------
    #  Abstract Method: Get Vector Size
    # ----------------------------------------------------------------------------------------
    @abstractmethod
    def get_vector_size(self) -> int:
        """Get the dimensionality of the embedding vectors.
        
        Returns:
            Integer representing the vector dimension size.
        """
        pass

    # ----------------------------------------------------------------------------------------
    #  Abstract Method: Get Distance Metric
    # ----------------------------------------------------------------------------------------
    @abstractmethod
    def get_distance_metric(self) -> str:
        """Get the recommended distance metric for this embedder.
        
        Returns:
            String representing the distance metric ('Cosine', 'Euclidean', 'Dot').
        """
        pass


#============================================================================================
#  Class: SimpleTextEmbedder
#============================================================================================
class SimpleTextEmbedder(Embedder):
    """Text-only embedder using sentence transformers for semantic text embeddings."""
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    @require(lambda model_name: isinstance(model_name, str) and len(model_name.strip()) > 0,
             "Model name must be a non-empty string")
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Initialize the text embedder with a specified model.
        
        Args:
            model_name: HuggingFace model identifier for text embeddings.
            
        Raises:
            InvalidPointsError: If model cannot be loaded.
        """
        try:
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            
            # Determine vector size by running a test embedding
            with torch.no_grad():
                test_input = self.tokenizer("test", return_tensors="pt", 
                                          padding=True, truncation=True)
                test_output = self.model(**test_input)
                # Use mean pooling of last hidden states
                self._vector_size = test_output.last_hidden_state.mean(dim=1).shape[1]
                
            logger.info(f"Initialized SimpleTextEmbedder with model '{model_name}', "
                       f"vector size: {self._vector_size}")
                       
        except Exception as e:
            raise InvalidPointsError(
                issue=f"Failed to initialize text embedder with model '{model_name}': {str(e)}"
            )

    # ----------------------------------------------------------------------------------------
    #  Embed Text Content
    # ----------------------------------------------------------------------------------------
    @require(lambda content: isinstance(content, str) and len(content.strip()) > 0,
             "Content must be a non-empty string")
    @ensure(lambda result: isinstance(result, models.PointStruct), 
            "Must return a valid PointStruct")
    def embed(self, content: str, metadata: Optional[Dict[str, Any]] = None,
              point_id: Optional[str] = None) -> models.PointStruct:
        """Convert text content to a vector embedding.
        
        Args:
            content: Text string to embed.
            metadata: Optional metadata to store with the point.
            point_id: Optional custom ID for the point.
            
        Returns:
            PointStruct containing the text embedding and metadata.
            
        Raises:
            InvalidPointsError: If text cannot be embedded.
        """
        try:
            # Generate ID if not provided
            if point_id is None:
                point_id = str(uuid4())
                
            # Tokenize and embed the text
            with torch.no_grad():
                inputs = self.tokenizer(content, return_tensors="pt", 
                                      padding=True, truncation=True, max_length=512)
                outputs = self.model(**inputs)
                
                # Use mean pooling over token embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1)
                # Normalize the embeddings for better similarity search
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                vector = embeddings.squeeze().numpy()
                
            # Prepare payload with content and metadata
            payload = {"content": content, "content_type": "text"}
            if metadata:
                payload.update(metadata)
                
            point = models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            
            logger.debug(f"Embedded text content (length: {len(content)}) to vector "
                        f"(dimension: {len(vector)})")
            return point
            
        except Exception as e:
            raise InvalidPointsError(
                issue=f"Failed to embed text content: {str(e)}",
                points_count=1
            )

    # ----------------------------------------------------------------------------------------
    #  Get Vector Size
    # ----------------------------------------------------------------------------------------
    def get_vector_size(self) -> int:
        """Get the dimensionality of text embedding vectors."""
        return self._vector_size

    # ----------------------------------------------------------------------------------------
    #  Get Distance Metric
    # ----------------------------------------------------------------------------------------
    def get_distance_metric(self) -> str:
        """Get the recommended distance metric for text embeddings."""
        return "Cosine"


#============================================================================================
#  Class: MultimodalEmbedder
#============================================================================================
class MultimodalEmbedder(Embedder):
    """Multimodal embedder supporting both text and images using SigLIP model."""
    
    # ----------------------------------------------------------------------------------------
    #  Constructor
    # ----------------------------------------------------------------------------------------
    @require(lambda model_name: isinstance(model_name, str) and len(model_name.strip()) > 0,
             "Model name must be a non-empty string")
    def __init__(self, model_name: str = "ViT-B-16-SigLIP2", pretrained: str = "webli") -> None:
        """Initialize the multimodal embedder with a SigLIP model.
        
        Args:
            model_name: Open-CLIP model identifier for SigLIP embeddings.
            pretrained: Pretrained weights identifier.
            
        Raises:
            InvalidPointsError: If model cannot be loaded.
        """
        try:
            self.model_name = model_name
            self.pretrained = pretrained
            
            # Load SigLIP model from open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()
            
            # SigLIP models typically have 768-dimensional embeddings
            self._vector_size = 768
            
            logger.info(f"Initialized MultimodalEmbedder with SigLIP model '{model_name}' "
                       f"(pretrained: {pretrained}), vector size: {self._vector_size}")
                       
        except Exception as e:
            raise InvalidPointsError(
                issue=f"Failed to initialize multimodal embedder with model '{model_name}': {str(e)}"
            )

    # ----------------------------------------------------------------------------------------
    #  Embed Content (Text or Image)
    # ----------------------------------------------------------------------------------------
    @require(lambda content: isinstance(content, (str, Image.Image)),
             "Content must be a string or PIL Image")
    @ensure(lambda result: isinstance(result, models.PointStruct), 
            "Must return a valid PointStruct")
    def embed(self, content: Union[str, Image.Image], metadata: Optional[Dict[str, Any]] = None,
              point_id: Optional[str] = None) -> models.PointStruct:
        """Convert text or image content to a vector embedding.
        
        Args:
            content: Text string or PIL Image to embed.
            metadata: Optional metadata to store with the point.
            point_id: Optional custom ID for the point.
            
        Returns:
            PointStruct containing the embedding and metadata.
            
        Raises:
            InvalidPointsError: If content cannot be embedded.
        """
        try:
            # Generate ID if not provided
            if point_id is None:
                point_id = str(uuid4())
                
            # Determine content type and process accordingly
            if isinstance(content, str):
                return self._embed_text(content, metadata, point_id)
            elif isinstance(content, Image.Image):
                return self._embed_image(content, metadata, point_id)
            else:
                raise InvalidPointsError(
                    issue=f"Unsupported content type: {type(content)}",
                    points_count=1
                )
                
        except InvalidPointsError:
            raise
        except Exception as e:
            raise InvalidPointsError(
                issue=f"Failed to embed content: {str(e)}",
                points_count=1
            )

    # ----------------------------------------------------------------------------------------
    #  Get Vector Size
    # ----------------------------------------------------------------------------------------
    def get_vector_size(self) -> int:
        """Get the dimensionality of multimodal embedding vectors."""
        return self._vector_size

    # ----------------------------------------------------------------------------------------
    #  Get Distance Metric
    # ----------------------------------------------------------------------------------------
    def get_distance_metric(self) -> str:
        """Get the recommended distance metric for SigLIP embeddings."""
        return "Cosine"

    # ----------------------------------------------------------------------------------------
    #  Helper Methods (Private)
    # ----------------------------------------------------------------------------------------
    def _embed_text(self, text: str, metadata: Optional[Dict[str, Any]], 
                   point_id: str) -> models.PointStruct:
        """Helper method to embed text content using SigLIP."""
        if len(text.strip()) == 0:
            raise InvalidPointsError(issue="Text content cannot be empty")
            
        with torch.no_grad():
            # Tokenize text using open_clip tokenizer
            text_input = self.tokenizer([text])
            
            # Generate text features
            text_features = self.model.encode_text(text_input)
            
            # Normalize features for consistent similarity computation
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy array for Qdrant storage
            vector = text_features.squeeze().numpy()
            
        # Prepare payload
        payload = {"content": text, "content_type": "text"}
        if metadata:
            payload.update(metadata)
            
        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
        
        logger.debug(f"Embedded text content (length: {len(text)}) to vector "
                    f"(dimension: {len(vector)})")
        return point

    def _embed_image(self, image: Image.Image, metadata: Optional[Dict[str, Any]], 
                    point_id: str) -> models.PointStruct:
        """Helper method to embed image content using SigLIP."""
        if image.size == (0, 0):
            raise InvalidPointsError(issue="Image cannot be empty")
            
        with torch.no_grad():
            # Preprocess image using open_clip preprocessor
            image_input = self.preprocess(image).unsqueeze(0)
            
            # Generate image features
            image_features = self.model.encode_image(image_input)
            
            # Normalize features for consistent similarity computation
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy array for Qdrant storage
            vector = image_features.squeeze().numpy()
            
        # Prepare payload with image metadata
        payload = {
            "content_type": "image",
            "image_size": image.size,
            "image_mode": image.mode
        }
        if metadata:
            payload.update(metadata)
            
        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
        
        logger.debug(f"Embedded image (size: {image.size}) to vector "
                    f"(dimension: {len(vector)})")
        return point


#============================================================================================
#  Utility Functions
#============================================================================================

# ----------------------------------------------------------------------------------------
#  Factory Function: Create Embedder
# ----------------------------------------------------------------------------------------
def create_embedder(embedder_type: str, model_name: Optional[str] = None, 
                   pretrained: Optional[str] = None) -> Embedder:
    """Factory function to create embedder instances.
    
    Args:
        embedder_type: Type of embedder ('text' or 'multimodal').
        model_name: Optional custom model name.
        pretrained: Optional pretrained weights (for multimodal embedders).
        
    Returns:
        Configured embedder instance.
        
    Raises:
        InvalidPointsError: If embedder type is not supported.
    """
    embedder_type = embedder_type.lower()
    
    if embedder_type == "text":
        return SimpleTextEmbedder(model_name) if model_name else SimpleTextEmbedder()
    elif embedder_type == "multimodal":
        if model_name and pretrained:
            return MultimodalEmbedder(model_name, pretrained)
        elif model_name:
            return MultimodalEmbedder(model_name)
        else:
            return MultimodalEmbedder()
    else:
        raise InvalidPointsError(
            issue=f"Unsupported embedder type: '{embedder_type}'. "
                  "Supported types: 'text', 'multimodal'"
        )

#============================================================================================ 