"""
Knowledge Graph extractor using LLM (Gemini or Claude via LangChain).
"""
from typing import List, Optional
import logging
import json
import re

from ..models import Document, Entity, Relationship, KnowledgeGraph
from ..config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


# Prompt template for knowledge graph extraction
KG_EXTRACTION_PROMPT = """You are a knowledge graph extraction expert. Analyze the following text and extract entities and relationships to form a knowledge graph.

## Instructions:
1. Identify all important entities (people, organizations, concepts, technologies, places, events, etc.)
2. Identify relationships between these entities
3. Be specific with entity types and relationship types
4. Only extract information explicitly mentioned in the text

## Output Format:
Return a JSON object with the following structure:
```json
{{
    "entities": [
        {{"name": "Entity Name", "type": "EntityType", "properties": {{"key": "value"}}}}
    ],
    "relationships": [
        {{"source": "Source Entity Name", "target": "Target Entity Name", "type": "RELATIONSHIP_TYPE", "properties": {{"key": "value"}}}}
    ]
}}
```

## Entity Types (examples):
- Person, Organization, Company, Product, Technology
- Concept, Topic, Event, Location, Date
- Document, Process, Method, Tool

## Relationship Types (examples):
- WORKS_FOR, CREATED_BY, PART_OF, RELATED_TO
- USES, IMPLEMENTS, DEPENDS_ON, BASED_ON
- LOCATED_IN, OCCURRED_AT, AUTHORED_BY

## Text to analyze:
{text}

## Knowledge Graph (JSON only, no markdown):"""


class KnowledgeGraphExtractor:
    """
    Extracts knowledge graphs from documents using LLM.
    Supports Google Gemini and Anthropic Claude via LangChain.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the knowledge graph extractor.
        
        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        self._llm = None
        
    @property
    def llm(self):
        """Lazy load the LLM based on configuration."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm
    
    def _create_llm(self):
        """Factory method to create the appropriate LLM."""
        if self.config.provider == LLMProvider.GEMINI:
            return self._create_gemini_llm()
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return self._create_anthropic_llm()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    def _create_gemini_llm(self):
        """Create Google Gemini LLM."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for Gemini. "
                "Install it with: pip install langchain-google-genai"
            )
        
        if not self.config.gemini_api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Gemini"
            )
        
        logger.info(f"Using Gemini model: {self.config.gemini_model}")
        return ChatGoogleGenerativeAI(
            model=self.config.gemini_model,
            google_api_key=self.config.gemini_api_key,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens
        )
    
    def _create_anthropic_llm(self):
        """Create Anthropic Claude LLM."""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for Claude. "
                "Install it with: pip install langchain-anthropic"
            )
        
        if not self.config.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for Claude"
            )
        
        logger.info(f"Using Claude model: {self.config.anthropic_model}")
        return ChatAnthropic(
            model=self.config.anthropic_model,
            api_key=self.config.anthropic_api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
    
    def extract_from_document(
        self,
        document: Document,
        max_text_length: int = 8000
    ) -> KnowledgeGraph:
        """
        Extract knowledge graph from a document.
        
        Args:
            document: Document to process
            max_text_length: Maximum text length to send to LLM
            
        Returns:
            KnowledgeGraph object
        """
        logger.info(f"Extracting knowledge graph from: {document.name}")
        
        # Truncate text if too long
        text = document.content[:max_text_length]
        if len(document.content) > max_text_length:
            logger.warning(
                f"Document truncated from {len(document.content)} to {max_text_length} chars"
            )
        
        # Call LLM
        prompt = KG_EXTRACTION_PROMPT.format(text=text)
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            kg_data = self._parse_response(response_text)
            
            # Convert to model objects
            entities = []
            for entity_data in kg_data.get("entities", []):
                entity = Entity(
                    name=entity_data.get("name", ""),
                    type=entity_data.get("type", "Unknown"),
                    properties=entity_data.get("properties", {}),
                    document_id=document.id
                )
                entities.append(entity)
            
            relationships = []
            for rel_data in kg_data.get("relationships", []):
                relationship = Relationship(
                    source_entity=rel_data.get("source", ""),
                    target_entity=rel_data.get("target", ""),
                    relationship_type=rel_data.get("type", "RELATED_TO"),
                    properties=rel_data.get("properties", {}),
                    document_id=document.id
                )
                relationships.append(relationship)
            
            kg = KnowledgeGraph(
                document_id=document.id,
                document_name=document.name,
                entities=entities,
                relationships=relationships
            )
            
            logger.info(
                f"Extracted {len(entities)} entities and "
                f"{len(relationships)} relationships from {document.name}"
            )
            
            return kg
            
        except Exception as e:
            logger.error(f"Failed to extract KG from {document.name}: {e}")
            # Return empty KG on failure
            return KnowledgeGraph(
                document_id=document.id,
                document_name=document.name
            )
    
    def _parse_response(self, response_text: str) -> dict:
        """Parse LLM response to extract JSON."""
        # Try to extract JSON from the response
        # First, try to find JSON block
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try parsing the entire response as JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response as JSON")
            return {"entities": [], "relationships": []}
    
    def extract_from_documents(
        self,
        documents: List[Document],
        max_text_length: int = 8000
    ) -> List[KnowledgeGraph]:
        """
        Extract knowledge graphs from multiple documents.
        
        Args:
            documents: List of documents to process
            max_text_length: Maximum text length per document
            
        Returns:
            List of KnowledgeGraph objects
        """
        knowledge_graphs = []
        for doc in documents:
            kg = self.extract_from_document(doc, max_text_length)
            knowledge_graphs.append(kg)
        return knowledge_graphs
