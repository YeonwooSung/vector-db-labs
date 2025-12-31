"""
Database stores using Repository Pattern.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import logging

from ..models import Chunk, SearchResult

logger = logging.getLogger(__name__)


class BaseStore(ABC):
    """Abstract base class for all document stores."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the store (create collections/indices if needed)."""
        pass
    
    @abstractmethod
    def store_chunks(self, chunks: List[Chunk]) -> None:
        """Store document chunks."""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10, **kwargs) -> List[SearchResult]:
        """Search for relevant chunks."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the store."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close connections."""
        pass
