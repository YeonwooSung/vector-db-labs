"""
Data models for the document processing system.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


@dataclass
class Document:
    """Represents a PDF document."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    path: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}
        self.metadata["document_name"] = self.name
        self.metadata["document_path"] = self.path


@dataclass
class Chunk:
    """Represents a text chunk from a document."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    document_name: str = ""
    content: str = ""
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}
        self.metadata["document_id"] = self.document_id
        self.metadata["document_name"] = self.document_name
        self.metadata["chunk_index"] = self.chunk_index


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: str = ""  # e.g., "Person", "Organization", "Concept"
    properties: Dict[str, Any] = field(default_factory=dict)
    document_id: str = ""


@dataclass
class Relationship:
    """Represents a relationship between entities in the knowledge graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_entity: str = ""  # Entity name
    target_entity: str = ""  # Entity name
    relationship_type: str = ""  # e.g., "WORKS_FOR", "RELATED_TO"
    properties: Dict[str, Any] = field(default_factory=dict)
    document_id: str = ""


@dataclass
class KnowledgeGraph:
    """Represents extracted knowledge graph from a document."""
    document_id: str = ""
    document_name: str = ""
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)


@dataclass
class SearchResult:
    """Represents a search result."""
    chunk_id: str = ""
    document_name: str = ""
    content: str = ""
    score: float = 0.0
    source: str = ""  # "vector", "fulltext", "graph"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.chunk_id)
    
    def __eq__(self, other):
        if isinstance(other, SearchResult):
            return self.chunk_id == other.chunk_id
        return False
