# =============================================================================
#  Filename: exceptions.py
#
#  Short Description: Custom exception classes for rag_to_riches package.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

from typing import Optional, List, Any
from pathlib import Path


class VectorDatabaseError(Exception):
    """Base exception class for all vector database related errors.
    
    This serves as the base class for all custom exceptions in the
    rag_to_riches vector database operations.
    """
    
    def __init__(self, message: str, context: Optional[dict] = None) -> None:
        """Initialize the exception with message and optional context.
        
        Args:
            message: The error message.
            context: Optional dictionary with additional context information.
        """
        super().__init__(message)
        self.context = context or {}
        
    def __str__(self) -> str:
        """Return a detailed string representation of the error."""
        base_message = super().__str__()
        if not self.context:
            return base_message
            
        context_info = ", ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{base_message} (Context: {context_info})"


class VectorDatabasePathNotFoundError(VectorDatabaseError):
    """Raised when the vector database path does not exist."""
    
    def __init__(self, path: str, suggestion: Optional[str] = None) -> None:
        """Initialize with path and optional suggestion.
        
        Args:
            path: The path that was not found.
            suggestion: Optional suggestion for resolving the issue.
        """
        message = f"Vector database path '{path}' does not exist"
        if suggestion:
            message += f". {suggestion}"
        else:
            message += (
                ". Please ensure the path exists and is accessible, "
                "or check your configuration settings."
            )
            
        context = {
            "path": path,
            "absolute_path": str(Path(path).resolve()),
            "parent_exists": Path(path).parent.exists(),
        }
        
        super().__init__(message, context)
        self.path = path


class CollectionAlreadyExistsError(VectorDatabaseError):
    """Raised when attempting to create a collection that already exists."""
    
    def __init__(self, collection_name: str, vector_size: Optional[int] = None,
                 distance: Optional[str] = None) -> None:
        """Initialize with collection details.
        
        Args:
            collection_name: Name of the collection that already exists.
            vector_size: Vector size that was attempted to be used.
            distance: Distance metric that was attempted to be used.
        """
        message = (
            f"Collection '{collection_name}' already exists. "
            "Use 'ensure_collection()' to create or update, "
            "or 'delete_collection()' to remove the existing collection first."
        )
        
        context = {"collection_name": collection_name}
        if vector_size is not None:
            context["attempted_vector_size"] = vector_size
        if distance is not None:
            context["attempted_distance"] = distance
            
        super().__init__(message, context)
        self.collection_name = collection_name


class CollectionNotFoundError(VectorDatabaseError):
    """Raised when attempting to operate on a collection that doesn't exist."""
    
    def __init__(self, collection_name: str, operation: str,
                 available_collections: Optional[List[str]] = None) -> None:
        """Initialize with collection details and operation context.
        
        Args:
            collection_name: Name of the collection that was not found.
            operation: The operation that was being attempted.
            available_collections: List of available collections, if known.
        """
        message = (
            f"Collection '{collection_name}' does not exist. "
            f"Cannot perform operation: {operation}."
        )
        
        if available_collections:
            if available_collections:
                message += f" Available collections: {', '.join(available_collections)}"
            else:
                message += " No collections currently exist in the database."
        else:
            message += " Use 'create_collection()' or 'ensure_collection()' to create it first."
            
        context = {
            "collection_name": collection_name,
            "operation": operation,
        }
        if available_collections is not None:
            context["available_collections"] = available_collections
            
        super().__init__(message, context)
        self.collection_name = collection_name
        self.operation = operation


class InvalidCollectionParametersError(VectorDatabaseError):
    """Raised when collection parameters are invalid."""
    
    def __init__(self, parameter_name: str, parameter_value: Any,
                 expected_type: str, additional_constraints: Optional[str] = None) -> None:
        """Initialize with parameter validation details.
        
        Args:
            parameter_name: Name of the invalid parameter.
            parameter_value: The invalid value that was provided.
            expected_type: Description of the expected type/format.
            additional_constraints: Additional constraints that were violated.
        """
        message = (
            f"Invalid {parameter_name}: '{parameter_value}'. "
            f"Expected {expected_type}"
        )
        
        if additional_constraints:
            message += f" with constraints: {additional_constraints}"
            
        context = {
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "parameter_type": type(parameter_value).__name__,
            "expected_type": expected_type,
        }
        if additional_constraints:
            context["additional_constraints"] = additional_constraints
            
        super().__init__(message, context)
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value


class InvalidVectorSizeError(InvalidCollectionParametersError):
    """Raised when vector size is invalid."""
    
    def __init__(self, vector_size: Any) -> None:
        """Initialize with vector size details.
        
        Args:
            vector_size: The invalid vector size value.
        """
        super().__init__(
            parameter_name="vector_size",
            parameter_value=vector_size,
            expected_type="positive integer",
            additional_constraints="must be > 0 and typically between 1 and 4096"
        )


class InvalidDistanceMetricError(InvalidCollectionParametersError):
    """Raised when distance metric is invalid."""
    
    def __init__(self, distance: Any, valid_metrics: Optional[List[str]] = None) -> None:
        """Initialize with distance metric details.
        
        Args:
            distance: The invalid distance metric.
            valid_metrics: List of valid distance metrics.
        """
        valid_metrics = valid_metrics or ["Cosine", "Euclidean", "Dot"]
        super().__init__(
            parameter_name="distance",
            parameter_value=distance,
            expected_type="string",
            additional_constraints=f"must be one of: {', '.join(valid_metrics)}"
        )
        self.valid_metrics = valid_metrics


class InvalidPointsError(VectorDatabaseError):
    """Raised when points data is invalid for upsert operations."""
    
    def __init__(self, issue: str, points_count: Optional[int] = None,
                 invalid_indices: Optional[List[int]] = None) -> None:
        """Initialize with points validation details.
        
        Args:
            issue: Description of the validation issue.
            points_count: Total number of points, if available.
            invalid_indices: List of indices with invalid points, if applicable.
        """
        message = f"Invalid points data: {issue}"
        
        if points_count is not None:
            message += f" (Total points: {points_count})"
            
        if invalid_indices:
            if len(invalid_indices) <= 5:
                message += f" (Invalid at indices: {invalid_indices})"
            else:
                message += f" (Invalid at {len(invalid_indices)} indices, first 5: {invalid_indices[:5]})"
                
        context = {"issue": issue}
        if points_count is not None:
            context["points_count"] = points_count
        if invalid_indices is not None:
            context["invalid_indices"] = invalid_indices
            context["invalid_count"] = len(invalid_indices)
            
        super().__init__(message, context)
        self.issue = issue


class CollectionParameterMismatchError(VectorDatabaseError):
    """Raised when existing collection parameters don't match requirements."""
    
    def __init__(self, collection_name: str, existing_params: dict,
                 required_params: dict, action_taken: Optional[str] = None) -> None:
        """Initialize with parameter mismatch details.
        
        Args:
            collection_name: Name of the collection with mismatched parameters.
            existing_params: Dictionary of existing parameters.
            required_params: Dictionary of required parameters.
            action_taken: Description of action taken to resolve, if any.
        """
        mismatches = []
        for key in required_params:
            if key in existing_params and existing_params[key] != required_params[key]:
                mismatches.append(f"{key}: {existing_params[key]} → {required_params[key]}")
                
        message = (
            f"Collection '{collection_name}' parameter mismatch. "
            f"Changes needed: {', '.join(mismatches)}"
        )
        
        if action_taken:
            message += f". Action taken: {action_taken}"
        else:
            message += ". Use 'ensure_collection()' to automatically resolve mismatches."
            
        context = {
            "collection_name": collection_name,
            "existing_params": existing_params,
            "required_params": required_params,
            "mismatches": mismatches,
        }
        if action_taken:
            context["action_taken"] = action_taken
            
        super().__init__(message, context)
        self.collection_name = collection_name
        self.existing_params = existing_params
        self.required_params = required_params 