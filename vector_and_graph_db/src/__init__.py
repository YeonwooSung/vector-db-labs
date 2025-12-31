"""
Vector and Graph DB - PDF Document Processing & Hybrid Search

This package provides:
1. ETL Pipeline: Load PDFs, chunk, embed, and store in multiple databases
2. Hybrid Search: Query across vector, fulltext, and graph databases

Usage:
    # ETL Pipeline
    python -m src.etl --docs-dir docs --llm-provider gemini
    
    # Search REPL
    python -m src.repl

Environment Variables:
    - GOOGLE_API_KEY: Google API key for Gemini
    - ANTHROPIC_API_KEY: Anthropic API key for Claude
    - QDRANT_HOST, QDRANT_PORT: Qdrant connection
    - ES_HOST, ES_PORT: Elasticsearch connection
    - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD: Neo4j connection
"""

from .config import AppConfig, IndexType, LLMProvider
from .models import Document, Chunk, Entity, Relationship, KnowledgeGraph, SearchResult
from .etl import ETLPipeline
from .repl import SearchEngine, run_repl

__all__ = [
    "AppConfig",
    "IndexType", 
    "LLMProvider",
    "Document",
    "Chunk",
    "Entity",
    "Relationship",
    "KnowledgeGraph",
    "SearchResult",
    "ETLPipeline",
    "SearchEngine",
    "run_repl",
]
