"""
Hybrid search functionality combining vector, fulltext, and graph search.
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from ..models import SearchResult
from ..config import SearchConfig

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Result from hybrid search with combined scoring."""
    chunk_id: str
    document_name: str
    content: str
    combined_score: float
    vector_score: Optional[float] = None
    fulltext_score: Optional[float] = None
    sources: List[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.metadata is None:
            self.metadata = {}


class HybridSearchMerger:
    """
    Merges results from vector search and fulltext search using
    Reciprocal Rank Fusion (RRF) or weighted scoring.
    """
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """
        Initialize the merger.
        
        Args:
            config: Search configuration
        """
        self.config = config or SearchConfig()
    
    def merge_rrf(
        self,
        vector_results: List[SearchResult],
        fulltext_results: List[SearchResult],
        k: int = 60,
        top_k: int = 10
    ) -> List[HybridSearchResult]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(1 / (k + rank)) for each result list
        
        Args:
            vector_results: Results from vector search
            fulltext_results: Results from fulltext search
            k: RRF parameter (default 60)
            top_k: Number of results to return
            
        Returns:
            List of merged HybridSearchResult
        """
        # Build RRF scores
        scores: Dict[str, dict] = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            chunk_id = result.chunk_id
            rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed
            
            if chunk_id not in scores:
                scores[chunk_id] = {
                    "result": result,
                    "rrf_score": 0.0,
                    "vector_score": None,
                    "fulltext_score": None,
                    "sources": []
                }
            
            scores[chunk_id]["rrf_score"] += rrf_score
            scores[chunk_id]["vector_score"] = result.score
            scores[chunk_id]["sources"].append("vector")
        
        # Process fulltext results
        for rank, result in enumerate(fulltext_results):
            chunk_id = result.chunk_id
            rrf_score = 1.0 / (k + rank + 1)
            
            if chunk_id not in scores:
                scores[chunk_id] = {
                    "result": result,
                    "rrf_score": 0.0,
                    "vector_score": None,
                    "fulltext_score": None,
                    "sources": []
                }
            
            scores[chunk_id]["rrf_score"] += rrf_score
            scores[chunk_id]["fulltext_score"] = result.score
            scores[chunk_id]["sources"].append("fulltext")
        
        # Sort by RRF score and create results
        sorted_items = sorted(
            scores.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True
        )[:top_k]
        
        hybrid_results = []
        for chunk_id, data in sorted_items:
            result = data["result"]
            hybrid_result = HybridSearchResult(
                chunk_id=chunk_id,
                document_name=result.document_name,
                content=result.content,
                combined_score=data["rrf_score"],
                vector_score=data["vector_score"],
                fulltext_score=data["fulltext_score"],
                sources=list(set(data["sources"])),
                metadata=result.metadata
            )
            hybrid_results.append(hybrid_result)
        
        return hybrid_results
    
    def merge_weighted(
        self,
        vector_results: List[SearchResult],
        fulltext_results: List[SearchResult],
        alpha: Optional[float] = None,
        top_k: int = 10
    ) -> List[HybridSearchResult]:
        """
        Merge results using weighted scoring.
        
        Combined score = alpha * vector_score + (1 - alpha) * fulltext_score
        
        Args:
            vector_results: Results from vector search
            fulltext_results: Results from fulltext search
            alpha: Weight for vector search (0-1, default from config)
            top_k: Number of results to return
            
        Returns:
            List of merged HybridSearchResult
        """
        if alpha is None:
            alpha = self.config.hybrid_alpha
        
        # Normalize scores to 0-1 range
        vector_scores = self._normalize_scores(vector_results)
        fulltext_scores = self._normalize_scores(fulltext_results)
        
        # Combine scores
        scores: Dict[str, dict] = {}
        
        for result in vector_results:
            chunk_id = result.chunk_id
            normalized_score = vector_scores.get(chunk_id, 0.0)
            
            scores[chunk_id] = {
                "result": result,
                "vector_score": normalized_score,
                "fulltext_score": 0.0,
                "sources": ["vector"]
            }
        
        for result in fulltext_results:
            chunk_id = result.chunk_id
            normalized_score = fulltext_scores.get(chunk_id, 0.0)
            
            if chunk_id in scores:
                scores[chunk_id]["fulltext_score"] = normalized_score
                scores[chunk_id]["sources"].append("fulltext")
            else:
                scores[chunk_id] = {
                    "result": result,
                    "vector_score": 0.0,
                    "fulltext_score": normalized_score,
                    "sources": ["fulltext"]
                }
        
        # Calculate combined scores
        for data in scores.values():
            data["combined_score"] = (
                alpha * data["vector_score"] +
                (1 - alpha) * data["fulltext_score"]
            )
        
        # Sort and create results
        sorted_items = sorted(
            scores.items(),
            key=lambda x: x[1]["combined_score"],
            reverse=True
        )[:top_k]
        
        hybrid_results = []
        for chunk_id, data in sorted_items:
            result = data["result"]
            hybrid_result = HybridSearchResult(
                chunk_id=chunk_id,
                document_name=result.document_name,
                content=result.content,
                combined_score=data["combined_score"],
                vector_score=data["vector_score"],
                fulltext_score=data["fulltext_score"],
                sources=list(set(data["sources"])),
                metadata=result.metadata
            )
            hybrid_results.append(hybrid_result)
        
        return hybrid_results
    
    def _normalize_scores(self, results: List[SearchResult]) -> Dict[str, float]:
        """Normalize scores to 0-1 range using min-max normalization."""
        if not results:
            return {}
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return {r.chunk_id: 1.0 for r in results}
        
        return {
            r.chunk_id: (r.score - min_score) / (max_score - min_score)
            for r in results
        }
