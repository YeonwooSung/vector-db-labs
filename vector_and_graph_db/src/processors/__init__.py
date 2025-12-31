"""
Text processing utilities: chunking and embedding.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import logging

from ..models import Document, Chunk
from ..config import ChunkingConfig, EmbeddingConfig

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits documents into chunks for processing.
    Uses a simple sliding window approach with configurable overlap.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the text chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of Chunk objects
        """
        text = document.content
        chunks = []
        
        # Split by separator first if present
        if self.config.separator and self.config.separator in text:
            paragraphs = text.split(self.config.separator)
        else:
            paragraphs = [text]
        
        # Merge small paragraphs and split large ones
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph exceeds chunk size, save current and start new
            if len(current_chunk) + len(para) + 1 > self.config.chunk_size:
                if current_chunk:
                    chunk = Chunk(
                        document_id=document.id,
                        document_name=document.name,
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        metadata=document.metadata.copy()
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Handle overlap
                    if self.config.chunk_overlap > 0:
                        overlap_text = current_chunk[-self.config.chunk_overlap:]
                        current_chunk = overlap_text + " " + para
                        current_start = current_start + len(current_chunk) - len(overlap_text)
                    else:
                        current_chunk = para
                        current_start = current_start + len(current_chunk)
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += self.config.separator + para
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunk = Chunk(
                document_id=document.id,
                document_name=document.name,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata=document.metadata.copy()
            )
            chunks.append(chunk)
        
        logger.info(f"Document '{document.name}' split into {len(chunks)} chunks")
        return chunks
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of all Chunk objects
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class SentenceTransformerEmbedding(EmbeddingModel):
    """
    Embedding model using sentence-transformers.
    Default model: all-MiniLM-L6-v2
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedding model.
        
        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self._model = None
        self._dimension = self.config.dimension
        
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install it with: pip install sentence-transformers"
                )
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self._model = SentenceTransformer(self.config.model_name)
        return self._model
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension
    
    def embed(self, text: str) -> List[float]:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=True
        )
        return embeddings.tolist()


class ChunkEmbedder:
    """Embeds chunks using the specified embedding model."""
    
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        """
        Initialize the chunk embedder.
        
        Args:
            embedding_model: Embedding model to use
        """
        self.embedding_model = embedding_model or SentenceTransformerEmbedding()
        
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Add embeddings to chunks.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            Same chunks with embeddings added
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.embed_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
        return chunks
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.embedding_model.embed(query)
