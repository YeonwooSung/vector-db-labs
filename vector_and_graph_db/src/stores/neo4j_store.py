"""
Neo4j graph database store.
"""
from typing import List, Optional, Dict, Any
import logging

from neo4j import GraphDatabase

from . import BaseStore
from ..models import Chunk, SearchResult, KnowledgeGraph
from ..config import Neo4jConfig

logger = logging.getLogger(__name__)


class Neo4jStore(BaseStore):
    """
    Neo4j graph database store for knowledge graphs.
    """
    
    def __init__(self, config: Optional[Neo4jConfig] = None):
        """
        Initialize Neo4j store.
        
        Args:
            config: Neo4j configuration
        """
        self.config = config or Neo4jConfig()
        self._driver = None
        
    @property
    def driver(self):
        """Lazy load Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password)
            )
            # Test connection
            with self._driver.session(database=self.config.database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.config.uri}")
        return self._driver
    
    def initialize(self) -> None:
        """Create constraints and indexes."""
        with self.driver.session(database=self.config.database) as session:
            # Create constraint for Entity names (unique per type)
            try:
                session.run("""
                    CREATE CONSTRAINT entity_name IF NOT EXISTS
                    FOR (e:Entity) REQUIRE e.name IS NOT NULL
                """)
            except Exception as e:
                logger.debug(f"Constraint might already exist: {e}")
            
            # Create index for faster lookups
            try:
                session.run("""
                    CREATE INDEX entity_type_idx IF NOT EXISTS
                    FOR (e:Entity) ON (e.type)
                """)
            except Exception as e:
                logger.debug(f"Index might already exist: {e}")
            
            # Create index for document relationships
            try:
                session.run("""
                    CREATE INDEX document_idx IF NOT EXISTS
                    FOR (d:Document) ON (d.name)
                """)
            except Exception as e:
                logger.debug(f"Index might already exist: {e}")
                
            # Create fulltext index for entity search
            try:
                session.run("""
                    CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
                    FOR (e:Entity) ON EACH [e.name]
                """)
            except Exception as e:
                logger.debug(f"Fulltext index might already exist: {e}")
        
        logger.info("Neo4j indexes and constraints initialized")
    
    def store_chunks(self, chunks: List[Chunk]) -> None:
        """
        Store chunks as Document nodes (optional, for reference).
        Note: Primary storage for chunks is in ES/Qdrant.
        This creates Document nodes that can be linked to entities.
        
        Args:
            chunks: List of chunks
        """
        if not chunks:
            return
            
        # Group chunks by document
        docs = {}
        for chunk in chunks:
            if chunk.document_id not in docs:
                docs[chunk.document_id] = {
                    "id": chunk.document_id,
                    "name": chunk.document_name,
                    "chunk_count": 0
                }
            docs[chunk.document_id]["chunk_count"] += 1
        
        with self.driver.session(database=self.config.database) as session:
            for doc_info in docs.values():
                session.run("""
                    MERGE (d:Document {id: $id})
                    SET d.name = $name, d.chunk_count = $chunk_count
                """, doc_info)
        
        logger.info(f"Created {len(docs)} document nodes in Neo4j")
    
    def store_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """
        Store a knowledge graph in Neo4j.
        
        Args:
            kg: KnowledgeGraph object to store
        """
        with self.driver.session(database=self.config.database) as session:
            # Create/update document node
            session.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.name = $doc_name
            """, {"doc_id": kg.document_id, "doc_name": kg.document_name})
            
            # Create entity nodes
            for entity in kg.entities:
                # Create entity with its specific type as a label
                session.run(f"""
                    MERGE (e:Entity:{entity.type.replace(' ', '_')} {{name: $name}})
                    SET e.type = $type,
                        e.document_id = $doc_id,
                        e.properties = $properties
                    WITH e
                    MATCH (d:Document {{id: $doc_id}})
                    MERGE (e)-[:MENTIONED_IN]->(d)
                """, {
                    "name": entity.name,
                    "type": entity.type,
                    "doc_id": kg.document_id,
                    "properties": str(entity.properties)
                })
            
            # Create relationships
            for rel in kg.relationships:
                # Create relationship with dynamic type
                rel_type = rel.relationship_type.upper().replace(' ', '_')
                session.run(f"""
                    MATCH (source:Entity {{name: $source_name}})
                    MATCH (target:Entity {{name: $target_name}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    SET r.document_id = $doc_id,
                        r.properties = $properties
                """, {
                    "source_name": rel.source_entity,
                    "target_name": rel.target_entity,
                    "doc_id": kg.document_id,
                    "properties": str(rel.properties)
                })
        
        logger.info(
            f"Stored KG for '{kg.document_name}': "
            f"{len(kg.entities)} entities, {len(kg.relationships)} relationships"
        )
    
    def store_knowledge_graphs(self, kgs: List[KnowledgeGraph]) -> None:
        """Store multiple knowledge graphs."""
        for kg in kgs:
            self.store_knowledge_graph(kg)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search the knowledge graph using fulltext search on entities.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        results = []
        
        with self.driver.session(database=self.config.database) as session:
            # Use fulltext index if available, otherwise fall back to CONTAINS
            try:
                # Try fulltext search first
                result = session.run("""
                    CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
                    YIELD node, score
                    MATCH (node)-[:MENTIONED_IN]->(d:Document)
                    OPTIONAL MATCH (node)-[r]->(related:Entity)
                    WHERE type(r) <> 'MENTIONED_IN'
                    RETURN 
                        node.name as entity_name,
                        node.type as entity_type,
                        d.name as document_name,
                        d.id as document_id,
                        score,
                        collect(DISTINCT {name: related.name, type: related.type, rel: type(r)})[..5] as related_entities
                    ORDER BY score DESC
                    LIMIT $limit
                """, {"query": query, "limit": top_k})
            except Exception:
                # Fall back to CONTAINS search
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($query)
                    MATCH (e)-[:MENTIONED_IN]->(d:Document)
                    OPTIONAL MATCH (e)-[r]->(related:Entity)
                    WHERE type(r) <> 'MENTIONED_IN'
                    WITH e, d, related, r, 
                         CASE WHEN toLower(e.name) = toLower($query) THEN 2.0
                              WHEN toLower(e.name) STARTS WITH toLower($query) THEN 1.5
                              ELSE 1.0 END as score
                    RETURN 
                        e.name as entity_name,
                        e.type as entity_type,
                        d.name as document_name,
                        d.id as document_id,
                        score,
                        collect(DISTINCT {name: related.name, type: related.type, rel: type(r)})[..5] as related_entities
                    ORDER BY score DESC
                    LIMIT $limit
                """, {"query": query, "limit": top_k})
            
            for record in result:
                # Format content with entity and relationship info
                related_info = ""
                if record["related_entities"]:
                    related_items = [
                        f"{r['rel']} -> {r['name']} ({r['type']})"
                        for r in record["related_entities"]
                        if r['name']
                    ]
                    if related_items:
                        related_info = "\nRelated: " + ", ".join(related_items)
                
                content = (
                    f"Entity: {record['entity_name']} ({record['entity_type']})"
                    f"{related_info}"
                )
                
                search_result = SearchResult(
                    chunk_id=f"neo4j_{record['entity_name']}",
                    document_name=record["document_name"] or "",
                    content=content,
                    score=record["score"],
                    source="graph",
                    metadata={
                        "entity_name": record["entity_name"],
                        "entity_type": record["entity_type"],
                        "document_id": record["document_id"]
                    }
                )
                results.append(search_result)
        
        return results
    
    def search_relationships(
        self,
        entity_name: str,
        relationship_type: Optional[str] = None,
        depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Search for entity relationships.
        
        Args:
            entity_name: Name of the entity to search from
            relationship_type: Optional specific relationship type to filter
            depth: How many hops to traverse
            
        Returns:
            List of relationship paths
        """
        with self.driver.session(database=self.config.database) as session:
            if relationship_type:
                query = f"""
                    MATCH path = (e:Entity {{name: $name}})-[:{relationship_type}*1..{depth}]-(related:Entity)
                    RETURN path
                    LIMIT 50
                """
            else:
                query = f"""
                    MATCH path = (e:Entity {{name: $name}})-[*1..{depth}]-(related:Entity)
                    WHERE NONE(r IN relationships(path) WHERE type(r) = 'MENTIONED_IN')
                    RETURN path
                    LIMIT 50
                """
            
            result = session.run(query, {"name": entity_name})
            
            paths = []
            for record in result:
                path = record["path"]
                path_info = {
                    "nodes": [
                        {"name": node["name"], "type": node.get("type", "Unknown")}
                        for node in path.nodes
                    ],
                    "relationships": [
                        {"type": rel.type}
                        for rel in path.relationships
                    ]
                }
                paths.append(path_info)
            
            return paths
    
    def clear(self) -> None:
        """Delete all nodes and relationships."""
        with self.driver.session(database=self.config.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared all data from Neo4j")
        self.initialize()
    
    def close(self) -> None:
        """Close the driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Closed Neo4j connection")
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        with self.driver.session(database=self.config.database) as session:
            result = session.run("""
                MATCH (e:Entity)
                WITH count(e) as entity_count
                MATCH (d:Document)
                WITH entity_count, count(d) as doc_count
                MATCH ()-[r]->()
                WHERE type(r) <> 'MENTIONED_IN'
                RETURN entity_count, doc_count, count(r) as rel_count
            """)
            record = result.single()
            
            if record:
                return {
                    "entities": record["entity_count"],
                    "documents": record["doc_count"],
                    "relationships": record["rel_count"]
                }
            return {"entities": 0, "documents": 0, "relationships": 0}
