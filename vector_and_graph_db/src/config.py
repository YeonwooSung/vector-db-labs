"""
Configuration management for the vector and graph database system.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import os


class IndexType(Enum):
    """Supported vector index types for Qdrant."""
    HNSW = "hnsw"
    FLAT = "flat"


class LLMProvider(Enum):
    """Supported LLM providers for knowledge graph extraction."""
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "documents"
    index_type: IndexType = IndexType.HNSW
    # HNSW specific parameters
    hnsw_m: int = 16  # Number of edges per node
    hnsw_ef_construct: int = 100  # Size of dynamic candidate list for construction


@dataclass
class ElasticsearchConfig:
    """Elasticsearch configuration."""
    host: str = "localhost"
    port: int = 9200
    index_name: str = "documents"
    scheme: str = "http"
    
    @property
    def url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"


@dataclass
class Neo4jConfig:
    """Neo4j graph database configuration."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384  # all-MiniLM-L6-v2 output dimension


@dataclass
class ChunkingConfig:
    """Text chunking configuration."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    separator: str = "\n"


@dataclass
class LLMConfig:
    """LLM configuration for knowledge graph extraction."""
    provider: LLMProvider = LLMProvider.GEMINI
    # Gemini
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    gemini_model: str = "gemini-2.5-flash"
    # Anthropic
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    # Common
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class SearchConfig:
    """Search configuration."""
    top_k: int = 10
    hybrid_alpha: float = 0.5  # Weight for vector search in hybrid (0-1)


@dataclass
class AppConfig:
    """Main application configuration."""
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    elasticsearch: ElasticsearchConfig = field(default_factory=ElasticsearchConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    docs_dir: str = "docs"
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv("QDRANT_HOST"):
            config.qdrant.host = os.getenv("QDRANT_HOST")
        if os.getenv("QDRANT_PORT"):
            config.qdrant.port = int(os.getenv("QDRANT_PORT"))
            
        if os.getenv("ES_HOST"):
            config.elasticsearch.host = os.getenv("ES_HOST")
        if os.getenv("ES_PORT"):
            config.elasticsearch.port = int(os.getenv("ES_PORT"))
            
        if os.getenv("NEO4J_URI"):
            config.neo4j.uri = os.getenv("NEO4J_URI")
        if os.getenv("NEO4J_USER"):
            config.neo4j.user = os.getenv("NEO4J_USER")
        if os.getenv("NEO4J_PASSWORD"):
            config.neo4j.password = os.getenv("NEO4J_PASSWORD")
            
        if os.getenv("LLM_PROVIDER"):
            config.llm.provider = LLMProvider(os.getenv("LLM_PROVIDER"))
            
        return config
