"""
REPL (Read-Eval-Print Loop) interface for hybrid search.
"""
import logging
from typing import Optional, List
import sys

from .config import AppConfig, SearchConfig
from .models import SearchResult
from .processors import ChunkEmbedder, SentenceTransformerEmbedding
from .stores.qdrant_store import QdrantStore
from .stores.es_store import ElasticsearchStore
from .stores.neo4j_store import Neo4jStore
from .search import HybridSearchMerger, HybridSearchResult

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during REPL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Unified search engine that queries all three databases
    and provides hybrid search results.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the search engine.
        
        Args:
            config: Application configuration
        """
        self.config = config or AppConfig.from_env()
        
        # Embedding model for query encoding
        self.embedding_model = SentenceTransformerEmbedding(self.config.embedding)
        self.embedder = ChunkEmbedder(self.embedding_model)
        
        # Initialize stores
        self.qdrant_store = QdrantStore(
            config=self.config.qdrant,
            embedding_config=self.config.embedding,
            embedder=self.embedder
        )
        self.es_store = ElasticsearchStore(self.config.elasticsearch)
        self.neo4j_store = Neo4jStore(self.config.neo4j)
        
        # Hybrid merger
        self.merger = HybridSearchMerger(self.config.search)
        
        self._connected = False
    
    def connect(self) -> None:
        """Test connections to all databases."""
        print("Connecting to databases...")
        
        try:
            # Test Qdrant
            _ = self.qdrant_store.client
            print("  ✓ Qdrant connected")
        except Exception as e:
            print(f"  ✗ Qdrant connection failed: {e}")
            
        try:
            # Test Elasticsearch
            _ = self.es_store.client
            print("  ✓ Elasticsearch connected")
        except Exception as e:
            print(f"  ✗ Elasticsearch connection failed: {e}")
            
        try:
            # Test Neo4j
            _ = self.neo4j_store.driver
            print("  ✓ Neo4j connected")
        except Exception as e:
            print(f"  ✗ Neo4j connection failed: {e}")
        
        self._connected = True
        print()
    
    def search_vector(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform vector similarity search in Qdrant.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of SearchResult
        """
        top_k = top_k or self.config.search.top_k
        query_embedding = self.embedder.embed_query(query)
        return self.qdrant_store.search(
            query=query,
            top_k=top_k,
            query_embedding=query_embedding
        )
    
    def search_fulltext(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform fulltext search in Elasticsearch.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of SearchResult
        """
        top_k = top_k or self.config.search.top_k
        return self.es_store.search(query=query, top_k=top_k)
    
    def search_graph(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Search knowledge graph in Neo4j.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of SearchResult
        """
        top_k = top_k or self.config.search.top_k
        return self.neo4j_store.search(query=query, top_k=top_k)
    
    def search_hybrid(
        self,
        query: str,
        top_k: Optional[int] = None,
        method: str = "rrf"
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining vector and fulltext results.
        
        Args:
            query: Search query
            top_k: Number of results
            method: Merge method ("rrf" or "weighted")
            
        Returns:
            List of HybridSearchResult
        """
        top_k = top_k or self.config.search.top_k
        
        # Get results from both stores
        vector_results = self.search_vector(query, top_k=top_k * 2)
        fulltext_results = self.search_fulltext(query, top_k=top_k * 2)
        
        # Merge results
        if method == "rrf":
            return self.merger.merge_rrf(
                vector_results,
                fulltext_results,
                top_k=top_k
            )
        else:
            return self.merger.merge_weighted(
                vector_results,
                fulltext_results,
                top_k=top_k
            )
    
    def search_all(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> dict:
        """
        Search all databases and return combined results.
        
        Args:
            query: Search query
            top_k: Number of results per source
            
        Returns:
            Dictionary with results from each source and hybrid
        """
        top_k = top_k or self.config.search.top_k
        
        return {
            "vector": self.search_vector(query, top_k),
            "fulltext": self.search_fulltext(query, top_k),
            "graph": self.search_graph(query, top_k),
            "hybrid": self.search_hybrid(query, top_k)
        }
    
    def close(self) -> None:
        """Close all connections."""
        self.qdrant_store.close()
        self.es_store.close()
        self.neo4j_store.close()


def print_search_results(results: List[SearchResult], title: str, max_content_len: int = 200):
    """Pretty print search results."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    
    if not results:
        print("  No results found.")
        return
    
    for i, result in enumerate(results, 1):
        content_preview = result.content[:max_content_len]
        if len(result.content) > max_content_len:
            content_preview += "..."
        
        print(f"\n  [{i}] Score: {result.score:.4f}")
        print(f"      Document: {result.document_name}")
        print(f"      Content: {content_preview}")


def print_hybrid_results(results: List[HybridSearchResult], max_content_len: int = 200):
    """Pretty print hybrid search results."""
    print(f"\n{'='*60}")
    print(f" 🔀 Hybrid Search Results (Vector + Fulltext)")
    print(f"{'='*60}")
    
    if not results:
        print("  No results found.")
        return
    
    for i, result in enumerate(results, 1):
        content_preview = result.content[:max_content_len]
        if len(result.content) > max_content_len:
            content_preview += "..."
        
        scores_str = f"Combined: {result.combined_score:.4f}"
        if result.vector_score is not None:
            scores_str += f", Vector: {result.vector_score:.4f}"
        if result.fulltext_score is not None:
            scores_str += f", FTS: {result.fulltext_score:.4f}"
        
        print(f"\n  [{i}] {scores_str}")
        print(f"      Sources: {', '.join(result.sources)}")
        print(f"      Document: {result.document_name}")
        print(f"      Content: {content_preview}")


def run_repl(config: Optional[AppConfig] = None):
    """
    Run the search REPL.
    
    Args:
        config: Application configuration
    """
    engine = SearchEngine(config)
    
    print("\n" + "="*60)
    print(" 🔍 Hybrid Search REPL")
    print("="*60)
    print("\nCommands:")
    print("  <query>       - Search all databases")
    print("  /vector <q>   - Vector search only")
    print("  /fts <q>      - Fulltext search only")
    print("  /graph <q>    - Graph search only")
    print("  /hybrid <q>   - Hybrid (vector + fts) search")
    print("  /top <n>      - Set number of results (default: 10)")
    print("  /stats        - Show database statistics")
    print("  /help         - Show this help")
    print("  /quit         - Exit")
    print()
    
    # Connect to databases
    engine.connect()
    
    top_k = config.search.top_k if config else 10
    
    try:
        while True:
            try:
                user_input = input("\n🔎 Query> ").strip()
            except EOFError:
                break
                
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                print("Goodbye!")
                break
            
            if user_input.lower() == "/help":
                print("\nCommands:")
                print("  <query>       - Search all databases")
                print("  /vector <q>   - Vector search only")
                print("  /fts <q>      - Fulltext search only")
                print("  /graph <q>    - Graph search only")
                print("  /hybrid <q>   - Hybrid (vector + fts) search")
                print("  /top <n>      - Set number of results")
                print("  /stats        - Show database statistics")
                print("  /quit         - Exit")
                continue
            
            if user_input.lower().startswith("/top "):
                try:
                    top_k = int(user_input.split()[1])
                    print(f"Results per source set to: {top_k}")
                except (IndexError, ValueError):
                    print("Usage: /top <number>")
                continue
            
            if user_input.lower() == "/stats":
                print("\n📊 Database Statistics:")
                try:
                    qdrant_info = engine.qdrant_store.get_collection_info()
                    print(f"  Qdrant: {qdrant_info['points_count']} vectors ({qdrant_info['index_type']})")
                except Exception as e:
                    print(f"  Qdrant: Error - {e}")
                
                try:
                    es_info = engine.es_store.get_index_info()
                    print(f"  Elasticsearch: {es_info['docs_count']} documents")
                except Exception as e:
                    print(f"  Elasticsearch: Error - {e}")
                
                try:
                    neo4j_stats = engine.neo4j_store.get_stats()
                    print(f"  Neo4j: {neo4j_stats['entities']} entities, "
                          f"{neo4j_stats['relationships']} relationships")
                except Exception as e:
                    print(f"  Neo4j: Error - {e}")
                continue
            
            # Parse query
            if user_input.startswith("/vector "):
                query = user_input[8:].strip()
                if query:
                    results = engine.search_vector(query, top_k)
                    print_search_results(results, "🎯 Vector Search Results (Qdrant)")
                continue
            
            if user_input.startswith("/fts "):
                query = user_input[5:].strip()
                if query:
                    results = engine.search_fulltext(query, top_k)
                    print_search_results(results, "📝 Fulltext Search Results (Elasticsearch)")
                continue
            
            if user_input.startswith("/graph "):
                query = user_input[7:].strip()
                if query:
                    results = engine.search_graph(query, top_k)
                    print_search_results(results, "🕸️ Graph Search Results (Neo4j)")
                continue
            
            if user_input.startswith("/hybrid "):
                query = user_input[8:].strip()
                if query:
                    results = engine.search_hybrid(query, top_k)
                    print_hybrid_results(results)
                continue
            
            # Default: search all
            query = user_input
            print(f"\nSearching for: '{query}'")
            
            # Vector search
            try:
                vector_results = engine.search_vector(query, top_k)
                print_search_results(vector_results, "🎯 Vector Search Results (Qdrant)")
            except Exception as e:
                print(f"\n❌ Vector search error: {e}")
                vector_results = []
            
            # Fulltext search
            try:
                fulltext_results = engine.search_fulltext(query, top_k)
                print_search_results(fulltext_results, "📝 Fulltext Search Results (Elasticsearch)")
            except Exception as e:
                print(f"\n❌ Fulltext search error: {e}")
                fulltext_results = []
            
            # Graph search
            try:
                graph_results = engine.search_graph(query, top_k)
                print_search_results(graph_results, "🕸️ Graph Search Results (Neo4j)")
            except Exception as e:
                print(f"\n❌ Graph search error: {e}")
            
            # Hybrid results
            if vector_results or fulltext_results:
                try:
                    hybrid_results = engine.merger.merge_rrf(
                        vector_results,
                        fulltext_results,
                        top_k=top_k
                    )
                    print_hybrid_results(hybrid_results)
                except Exception as e:
                    print(f"\n❌ Hybrid merge error: {e}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
    finally:
        engine.close()


def main():
    """Main entry point for the search REPL."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Search REPL")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results per source (default: 10)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for vector search in hybrid (0-1, default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = AppConfig.from_env()
    config.search.top_k = args.top_k
    config.search.hybrid_alpha = args.alpha
    
    run_repl(config)


if __name__ == "__main__":
    main()
