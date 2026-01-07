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


    def _split_text_into_chunks(
        self,
        text: str,
        max_length: int,
        overlap: int = 200
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            max_length: Maximum length of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + max_length
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start forward, with overlap
            start = end - overlap
            
            # Break if we've reached the end
            if end >= len(text):
                break

        return chunks

    def _merge_knowledge_graphs(
        self,
        knowledge_graphs: List[KnowledgeGraph]
    ) -> KnowledgeGraph:
        """
        Merge multiple knowledge graphs, combining duplicate entities.
        
        Args:
            knowledge_graphs: List of knowledge graphs to merge
            
        Returns:
            Merged knowledge graph
        """
        if not knowledge_graphs:
            return KnowledgeGraph(
                document_id="",
                document_name=""
            )

        if len(knowledge_graphs) == 1:
            return knowledge_graphs[0]

        # Use first KG as base
        merged_kg = knowledge_graphs[0]

        # Entity map: name -> Entity (for deduplication)
        entity_map = {}
        for entity in merged_kg.entities:
            key = (entity.name.lower().strip(), entity.type.lower())
            entity_map[key] = entity

        # Merge entities from remaining KGs
        for kg in knowledge_graphs[1:]:
            for entity in kg.entities:
                key = (entity.name.lower().strip(), entity.type.lower())

                if key in entity_map:
                    # Merge properties
                    existing = entity_map[key]
                    for prop_key, prop_value in entity.properties.items():
                        if prop_key not in existing.properties:
                            existing.properties[prop_key] = prop_value
                else:
                    # Add new entity
                    entity_map[key] = entity
                    merged_kg.entities.append(entity)

        # Merge relationships (with deduplication)
        relationship_set = set()
        merged_relationships = []

        for kg in knowledge_graphs:
            for rel in kg.relationships:
                # Create a unique key for the relationship
                rel_key = (
                    rel.source_entity.lower().strip(),
                    rel.target_entity.lower().strip(),
                    rel.relationship_type.upper()
                )
                
                if rel_key not in relationship_set:
                    relationship_set.add(rel_key)
                    merged_relationships.append(rel)

        merged_kg.relationships = merged_relationships

        logger.info(
            f"Merged {len(knowledge_graphs)} KGs into 1 with "
            f"{len(merged_kg.entities)} unique entities and "
            f"{len(merged_kg.relationships)} unique relationships"
        )

        return merged_kg

    def extract_from_document(
        self,
        document: Document,
        max_text_length: int = 3000
    ) -> KnowledgeGraph:
        """
        Extract knowledge graph from a document.
        For long documents, splits into chunks and merges results.
        
        Args:
            document: Document to process
            max_text_length: Maximum text length per chunk to send to LLM
            
        Returns:
            KnowledgeGraph object
        """
        logger.info(f"Extracting knowledge graph from: {document.name}")

        # Split text into chunks if necessary
        chunks = self._split_text_into_chunks(
            document.content,
            max_text_length,
            overlap=200
        )

        if len(chunks) > 1:
            logger.info(
                f"Document split into {len(chunks)} chunks "
                f"(original length: {len(document.content)} chars)"
            )

        # Extract KG from each chunk
        chunk_kgs = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}")

            # Call LLM
            prompt = KG_EXTRACTION_PROMPT.format(text=chunk)

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

                chunk_kg = KnowledgeGraph(
                    document_id=document.id,
                    document_name=document.name,
                    entities=entities,
                    relationships=relationships
                )

                chunk_kgs.append(chunk_kg)

                logger.info(
                    f"Chunk {i}: Extracted {len(entities)} entities and "
                    f"{len(relationships)} relationships"
                )

            except Exception as e:
                logger.error(f"Failed to extract KG from chunk {i} of {document.name}: {e}")
                # Continue with other chunks
                continue

        # Merge all chunk KGs
        if chunk_kgs:
            merged_kg = self._merge_knowledge_graphs(chunk_kgs)
            logger.info(
                f"Final result for {document.name}: {len(merged_kg.entities)} entities and "
                f"{len(merged_kg.relationships)} relationships"
            )
            return merged_kg
        else:
            # Return empty KG if all chunks failed
            logger.error(f"All chunks failed for {document.name}")
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
            print("Attempting to parse entire response as JSON: ")
            print(response_text)
            print()
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
