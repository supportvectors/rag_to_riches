# =============================================================================
#  Filename: __init__.py
#
#  Short Description: Exceptions package for rag_to_riches.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

from .exceptions import (
    VectorDatabaseError,
    VectorDatabasePathNotFoundError,
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    InvalidCollectionParametersError,
    InvalidVectorSizeError,
    InvalidDistanceMetricError,
    InvalidPointsError,
    CollectionParameterMismatchError,
)

__all__ = [
    "VectorDatabaseError",
    "VectorDatabasePathNotFoundError",
    "CollectionAlreadyExistsError",
    "CollectionNotFoundError",
    "InvalidCollectionParametersError",
    "InvalidVectorSizeError",
    "InvalidDistanceMetricError",
    "InvalidPointsError",
    "CollectionParameterMismatchError",
] 