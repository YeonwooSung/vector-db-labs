"""
Elasticsearch store for full-text search with BM25.
"""
from typing import List, Optional, Dict, Any
import logging

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from . import BaseStore
from ..models import Chunk, SearchResult
from ..config import ElasticsearchConfig

logger = logging.getLogger(__name__)


class ElasticsearchStore(BaseStore):
    """
    Elasticsearch store with BM25 indexing for full-text search.
    """
    
    # Index mapping with BM25 configuration
    INDEX_MAPPING = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "analysis": {
                "analyzer": {
                    "custom_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop", "snowball"]
                    }
                }
            },
            "similarity": {
                "bm25_custom": {
                    "type": "BM25",
                    "k1": 1.2,
                    "b": 0.75
                }
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "document_name": {"type": "keyword"},
                "content": {
                    "type": "text",
                    "analyzer": "custom_analyzer",
                    "similarity": "bm25_custom"
                },
                "chunk_index": {"type": "integer"},
                "metadata": {"type": "object", "enabled": False}
            }
        }
    }
    
    def __init__(self, config: Optional[ElasticsearchConfig] = None):
        """
        Initialize Elasticsearch store.
        
        Args:
            config: Elasticsearch configuration
        """
        self.config = config or ElasticsearchConfig()
        self._client: Optional[Elasticsearch] = None
        
    @property
    def client(self) -> Elasticsearch:
        """Lazy load Elasticsearch client."""
        if self._client is None:
            self._client = Elasticsearch(
                hosts=[self.config.url],
                request_timeout=30
            )
            # Test connection using info() which is more reliable in ES 8.x
            try:
                info = self._client.info()
                logger.info(f"Connected to Elasticsearch at {self.config.url} (version: {info['version']['number']})")
            except Exception as e:
                raise ConnectionError(
                    f"Cannot connect to Elasticsearch at {self.config.url}: {str(e)}"
                )
        return self._client
    
    def initialize(self) -> None:
        """Create index if it doesn't exist."""
        if self.client.indices.exists(index=self.config.index_name):
            logger.info(f"Index '{self.config.index_name}' already exists")
            return
            
        self.client.indices.create(
            index=self.config.index_name,
            body=self.INDEX_MAPPING
        )
        logger.info(f"Created index '{self.config.index_name}' with BM25 similarity")
    
    def store_chunks(self, chunks: List[Chunk]) -> None:
        """
        Store chunks in Elasticsearch.
        
        Args:
            chunks: List of chunks to store
        """
        if not chunks:
            logger.warning("No chunks to store")
            return
        
        actions = []
        for chunk in chunks:
            action = {
                "_index": self.config.index_name,
                "_id": chunk.id,
                "_source": {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata
                }
            }
            actions.append(action)
        
        # Bulk index
        success, errors = bulk(self.client, actions, raise_on_error=False)
        
        if errors:
            logger.error(f"Bulk indexing had {len(errors)} errors")
            for error in errors[:5]:  # Log first 5 errors
                logger.error(f"Error: {error}")
        
        # Refresh index to make documents searchable immediately
        self.client.indices.refresh(index=self.config.index_name)
        
        logger.info(f"Indexed {success} chunks in Elasticsearch")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search using BM25 full-text search.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content"],
                    "type": "best_fields"
                }
            },
            "size": top_k
        }
        
        response = self.client.search(
            index=self.config.index_name,
            body=search_body
        )
        
        search_results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            search_result = SearchResult(
                chunk_id=source.get("chunk_id", hit["_id"]),
                document_name=source.get("document_name", ""),
                content=source.get("content", ""),
                score=hit["_score"],
                source="fulltext",
                metadata=source.get("metadata", {})
            )
            search_results.append(search_result)
        
        return search_results
    
    def search_by_document(
        self,
        query: str,
        document_name: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search within a specific document.
        
        Args:
            query: Search query text
            document_name: Name of document to search within
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "content": query
                            }
                        }
                    ],
                    "filter": [
                        {
                            "term": {
                                "document_name": document_name
                            }
                        }
                    ]
                }
            },
            "size": top_k
        }
        
        response = self.client.search(
            index=self.config.index_name,
            body=search_body
        )
        
        search_results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            search_result = SearchResult(
                chunk_id=source.get("chunk_id", hit["_id"]),
                document_name=source.get("document_name", ""),
                content=source.get("content", ""),
                score=hit["_score"],
                source="fulltext",
                metadata=source.get("metadata", {})
            )
            search_results.append(search_result)
        
        return search_results
    
    def clear(self) -> None:
        """Delete and recreate the index."""
        try:
            self.client.indices.delete(index=self.config.index_name)
            logger.info(f"Deleted index '{self.config.index_name}'")
        except Exception as e:
            logger.warning(f"Could not delete index: {e}")
        
        self.initialize()
    
    def close(self) -> None:
        """Close the client connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Closed Elasticsearch connection")
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the index."""
        stats = self.client.indices.stats(index=self.config.index_name)
        index_stats = stats["indices"][self.config.index_name]["primaries"]
        
        return {
            "name": self.config.index_name,
            "docs_count": index_stats["docs"]["count"],
            "size_bytes": index_stats["store"]["size_in_bytes"]
        }
