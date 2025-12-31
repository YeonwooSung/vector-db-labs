"""
Qdrant vector database store.
"""
from typing import List, Optional
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from . import BaseStore
from ..models import Chunk, SearchResult
from ..config import QdrantConfig, IndexType, EmbeddingConfig
from ..processors import ChunkEmbedder

logger = logging.getLogger(__name__)


class QdrantStore(BaseStore):
    """
    Qdrant vector database store.
    Supports HNSW and flat index types.
    """
    
    def __init__(
        self,
        config: Optional[QdrantConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        embedder: Optional[ChunkEmbedder] = None
    ):
        """
        Initialize Qdrant store.
        
        Args:
            config: Qdrant configuration
            embedding_config: Embedding configuration
            embedder: ChunkEmbedder instance for query embedding
        """
        self.config = config or QdrantConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.embedder = embedder
        self._client: Optional[QdrantClient] = None
        
    @property
    def client(self) -> QdrantClient:
        """Lazy load Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                host=self.config.host,
                port=self.config.port
            )
            logger.info(f"Connected to Qdrant at {self.config.host}:{self.config.port}")
        return self._client
    
    def initialize(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.config.collection_name in collection_names:
            logger.info(f"Collection '{self.config.collection_name}' already exists")
            return
            
        # Configure index based on type
        if self.config.index_type == IndexType.HNSW:
            hnsw_config = qmodels.HnswConfigDiff(
                m=self.config.hnsw_m,
                ef_construct=self.config.hnsw_ef_construct
            )
            vectors_config = qmodels.VectorParams(
                size=self.embedding_config.dimension,
                distance=qmodels.Distance.COSINE,
                hnsw_config=hnsw_config
            )
        else:  # FLAT
            vectors_config = qmodels.VectorParams(
                size=self.embedding_config.dimension,
                distance=qmodels.Distance.COSINE,
                on_disk=True  # Use disk for flat index
            )
        
        self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=vectors_config
        )
        logger.info(
            f"Created collection '{self.config.collection_name}' "
            f"with {self.config.index_type.value} index"
        )
    
    def store_chunks(self, chunks: List[Chunk]) -> None:
        """
        Store chunks with their embeddings in Qdrant.
        
        Args:
            chunks: List of chunks (must have embeddings)
        """
        if not chunks:
            logger.warning("No chunks to store")
            return
            
        # Verify all chunks have embeddings
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")
        
        points = []
        for chunk in chunks:
            point = qmodels.PointStruct(
                id=chunk.id,
                vector=chunk.embedding,
                payload={
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata
                }
            )
            points.append(point)
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=batch
            )
            logger.debug(f"Upserted batch {i // batch_size + 1}")
        
        logger.info(f"Stored {len(chunks)} chunks in Qdrant")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        query_embedding: Optional[List[float]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            query_embedding: Pre-computed query embedding (optional)
            
        Returns:
            List of SearchResult objects
        """
        # Get query embedding
        if query_embedding is None:
            if self.embedder is None:
                raise ValueError("No embedder provided for query embedding")
            query_embedding = self.embedder.embed_query(query)
        
        results = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        search_results = []
        for result in results:
            search_result = SearchResult(
                chunk_id=str(result.id),
                document_name=result.payload.get("document_name", ""),
                content=result.payload.get("content", ""),
                score=result.score,
                source="vector",
                metadata=result.payload.get("metadata", {})
            )
            search_results.append(search_result)
        
        return search_results
    
    def clear(self) -> None:
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(self.config.collection_name)
            logger.info(f"Deleted collection '{self.config.collection_name}'")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")
        
        self.initialize()
    
    def close(self) -> None:
        """Close the client connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Closed Qdrant connection")
    
    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        info = self.client.get_collection(self.config.collection_name)
        return {
            "name": self.config.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "index_type": self.config.index_type.value
        }
