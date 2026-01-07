"""
ETL Pipeline for processing PDF documents into vector, search, and graph databases.
"""
import logging
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

from .config import AppConfig, IndexType, LLMProvider
from .models import Document, Chunk, KnowledgeGraph
from .loaders import PDFLoader
from .processors import TextChunker, ChunkEmbedder, SentenceTransformerEmbedding
from .stores.qdrant_store import QdrantStore
from .stores.es_store import ElasticsearchStore
from .stores.neo4j_store import Neo4jStore
from .extractors import KnowledgeGraphExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ETLPipeline:
    """
    ETL Pipeline using Facade Pattern to orchestrate:
    1. PDF loading
    2. Text chunking
    3. Embedding generation
    4. Storage in Qdrant (vector), Elasticsearch (fulltext), Neo4j (graph)
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the ETL pipeline.
        
        Args:
            config: Application configuration
        """
        self.config = config or AppConfig.from_env()
        
        # Initialize components
        self.loader = PDFLoader(self.config.docs_dir)
        self.chunker = TextChunker(self.config.chunking)
        
        # Embedding
        self.embedding_model = SentenceTransformerEmbedding(self.config.embedding)
        self.embedder = ChunkEmbedder(self.embedding_model)
        
        # Stores
        self.qdrant_store = QdrantStore(
            config=self.config.qdrant,
            embedding_config=self.config.embedding,
            embedder=self.embedder
        )
        self.es_store = ElasticsearchStore(self.config.elasticsearch)
        self.neo4j_store = Neo4jStore(self.config.neo4j)
        
        # KG Extractor (lazy loaded)
        self._kg_extractor: Optional[KnowledgeGraphExtractor] = None
        
    @property
    def kg_extractor(self) -> KnowledgeGraphExtractor:
        """Lazy load KG extractor."""
        if self._kg_extractor is None:
            self._kg_extractor = KnowledgeGraphExtractor(self.config.llm)
        return self._kg_extractor
    
    def initialize_stores(self) -> None:
        """Initialize all database stores."""
        logger.info("Initializing database stores...")
        self.qdrant_store.initialize()
        self.es_store.initialize()
        self.neo4j_store.initialize()
        logger.info("All stores initialized")
    
    def clear_stores(self) -> None:
        """Clear all data from stores."""
        logger.info("Clearing all stores...")
        self.qdrant_store.clear()
        self.es_store.clear()
        self.neo4j_store.clear()
        logger.info("All stores cleared")
    
    def process_documents(
        self,
        extract_knowledge_graph: bool = True,
        clear_existing: bool = False
    ) -> dict:
        """
        Run the complete ETL pipeline.
        
        Args:
            extract_knowledge_graph: Whether to extract and store knowledge graphs
            clear_existing: Whether to clear existing data before processing
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "chunks_stored_qdrant": 0,
            "chunks_stored_es": 0,
            "knowledge_graphs_extracted": 0
        }
        
        # Initialize stores
        if clear_existing:
            self.clear_stores()
        else:
            self.initialize_stores()
        
        # Load documents
        logger.info(f"Loading PDF documents from: {self.config.docs_dir}")
        documents = self.loader.load_all()
        
        if not documents:
            logger.warning("No documents found to process")
            return stats
        
        stats["documents_processed"] = len(documents)
        logger.info(f"Loaded {len(documents)} documents")
        
        # Process each document
        all_chunks: List[Chunk] = []
        all_kgs: List[KnowledgeGraph] = []
        
        for doc in documents:
            logger.info(f"Processing document: {doc.name}")
            
            # Chunk document
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            
            # Extract knowledge graph if enabled
            if extract_knowledge_graph:
                try:
                    kg = self.kg_extractor.extract_from_document(doc)
                    all_kgs.append(kg)
                except Exception as e:
                    logger.error(f"Failed to extract KG from {doc.name}: {e}")
        
        stats["chunks_created"] = len(all_chunks)
        logger.info(f"Created {len(all_chunks)} chunks total")
        
        # Embed chunks
        logger.info("Embedding chunks...")
        embedded_chunks = self.embedder.embed_chunks(all_chunks)
        
        # Store in Qdrant
        logger.info("Storing chunks in Qdrant...")
        self.qdrant_store.store_chunks(embedded_chunks)
        stats["chunks_stored_qdrant"] = len(embedded_chunks)
        
        # Store in Elasticsearch
        logger.info("Storing chunks in Elasticsearch...")
        self.es_store.store_chunks(all_chunks)
        stats["chunks_stored_es"] = len(all_chunks)
        
        # Store in Neo4j
        logger.info("Storing document references and knowledge graphs in Neo4j...")
        self.neo4j_store.store_chunks(all_chunks)
        
        if all_kgs:
            self.neo4j_store.store_knowledge_graphs(all_kgs)
            stats["knowledge_graphs_extracted"] = len(all_kgs)
        
        logger.info("ETL pipeline completed successfully!")
        self._print_stats(stats)
        
        return stats
    
    def _print_stats(self, stats: dict) -> None:
        """Print processing statistics."""
        print("\n" + "=" * 50)
        print("ETL Pipeline Statistics")
        print("=" * 50)
        print(f"Documents processed:        {stats['documents_processed']}")
        print(f"Chunks created:             {stats['chunks_created']}")
        print(f"Chunks stored in Qdrant:    {stats['chunks_stored_qdrant']}")
        print(f"Chunks stored in ES:        {stats['chunks_stored_es']}")
        print(f"Knowledge graphs extracted: {stats['knowledge_graphs_extracted']}")
        print("=" * 50 + "\n")
    
    def close(self) -> None:
        """Close all connections."""
        self.qdrant_store.close()
        self.es_store.close()
        self.neo4j_store.close()
        logger.info("All connections closed")


def main():
    """Main entry point for the ETL pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ETL Pipeline for PDF documents")
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="docs",
        help="Directory containing PDF files (default: docs)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before processing"
    )
    parser.add_argument(
        "--no-kg",
        action="store_true",
        help="Skip knowledge graph extraction"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        choices=["hnsw", "flat"],
        default="hnsw",
        help="Qdrant index type (default: hnsw)"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["gemini", "anthropic"],
        default="gemini",
        help="LLM provider for KG extraction (default: gemini)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in characters (default: 512)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in characters (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = AppConfig.from_env()
    config.docs_dir = args.docs_dir
    config.qdrant.index_type = IndexType(args.index_type)
    config.llm.provider = LLMProvider(args.llm_provider)
    config.chunking.chunk_size = args.chunk_size
    config.chunking.chunk_overlap = args.chunk_overlap
    
    # Run pipeline
    pipeline = ETLPipeline(config)
    
    try:
        pipeline.process_documents(
            extract_knowledge_graph=not args.no_kg,
            clear_existing=args.clear
        )
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
