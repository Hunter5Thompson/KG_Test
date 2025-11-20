"kg_extractor.py"
"""
Knowledge Graph Extraction from text using LLMs
"""
import re
from typing import List
from llama_index.core import Document
from llama_index.llms.ollama import Ollama
from ollama import Client

# Handle imports for both direct execution and module import
try:
    from ..models.triplets import Triplet
    from ..embeddings.ollama_embeddings import OllamaEmbedding
    from ..storage.neo4j_store import Neo4jStore
except ImportError:
    from src.models.triplets import Triplet
    from src.embeddings.ollama_embeddings import OllamaEmbedding
    from src.storage.neo4j_store import Neo4jStore


class KnowledgeGraphExtractor:
    """
    Extract knowledge graphs from text using LLMs
    
    Features:
    - Custom prompt engineering for reliable extraction
    - Optional semantic embeddings
    - Flexible storage backends
    
    Args:
        llm: LlamaIndex LLM instance
        embed_model: Optional embedding model
        store: Optional Neo4j storage backend
    
    Example:
        >>> extractor = KnowledgeGraphExtractor(
        ...     llm=ollama_llm,
        ...     embed_model=embed_model,
        ...     store=neo4j_store
        ... )
        >>> result = extractor.extract_from_documents([doc1, doc2])
    """
    
    def __init__(
        self,
        llm: Ollama,
        embed_model: OllamaEmbedding = None,
        store: Neo4jStore = None
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.store = store
        
        print(f"✅ KG Extractor initialized")
        if embed_model:
            print(f"   Embeddings: {embed_model.model_name}")
        if store:
            print(f"   Storage: Neo4j")
    
    def extract_triplets_from_text(
        self,
        text: str,
        verbose: bool = True
    ) -> List[Triplet]:
        """
        Extract triplets from a single text
        
        Args:
            text: Input text
            verbose: Print extraction details
        
        Returns:
            List of extracted Triplet objects
        """
        prompt = self._build_extraction_prompt(text)
        response = self.llm.complete(prompt)
        triplets = self._parse_triplets(response.text)
        
        if verbose:
            print(f"📝 Extracted {len(triplets)} triplets from: '{text[:50]}...'")
            for t in triplets:
                print(f"   • {t}")
        
        return triplets
    
    def extract_from_documents(
        self,
        documents: List[Document],
        store_embeddings: bool = True,
        verbose: bool = True
    ) -> dict:
        """
        Extract knowledge graph from multiple documents
        
        Args:
            documents: List of Document objects
            store_embeddings: Compute and store embeddings
            verbose: Print progress
        
        Returns:
            Statistics dictionary
        """
        all_triplets = []
        
        if verbose:
            print(f"\n📊 Processing {len(documents)} documents...")
        
        for i, doc in enumerate(documents, 1):
            if verbose:
                print(f"\n[{i}/{len(documents)}] Processing document...")
            
            triplets = self.extract_triplets_from_text(doc.text, verbose=verbose)
            all_triplets.extend(triplets)
        
        # Compute embeddings if requested
        entity_embeddings = None
        if store_embeddings and self.embed_model:
            entity_embeddings = self._compute_embeddings(all_triplets, verbose=verbose)
        
        # Write to storage if available
        if self.store:
            if verbose:
                print(f"\n💾 Writing to Neo4j...")
            self.store.write_triplets(all_triplets, entity_embeddings)
        
        return {
            "documents_processed": len(documents),
            "triplets_extracted": len(all_triplets),
            "embeddings_computed": bool(entity_embeddings)
        }
    
    def _compute_embeddings(
        self,
        triplets: List[Triplet],
        verbose: bool = True
    ) -> dict[str, List[float]]:
        """Compute embeddings for all entities"""
        entities = set()
        for triplet in triplets:
            entities.add(triplet.subject)
            entities.add(triplet.object)
        
        if verbose:
            print(f"🧮 Computing embeddings for {len(entities)} entities...")
        
        entity_embeddings = {}
        for i, entity in enumerate(entities, 1):
            embedding = self.embed_model.get_text_embedding(entity)
            entity_embeddings[entity] = embedding
            
            if verbose and (i % 5 == 0 or i == len(entities)):
                print(f"   Progress: {i}/{len(entities)}")
        
        if verbose:
            dim = len(next(iter(entity_embeddings.values())))
            print(f"✅ Embeddings computed (dim={dim})")
        
        return entity_embeddings
    
    def _build_extraction_prompt(self, text: str) -> str:
        """Build optimized extraction prompt"""
        return f"""Extract ALL knowledge graph triplets from the following text.

RULES:
1. Return ONLY triplets in format: (subject, relation, object)
2. One triplet per line
3. Use clear, concise relation names (e.g., works_at, located_in, acquired)
4. Include ALL entities and their relationships

EXAMPLES:
(Alice, works_at, Acme Corp)
(Acme Corp, located_in, Berlin)
(Bob, is_a, engineer)

TEXT: {text}

TRIPLETS:"""
    
    def _parse_triplets(self, llm_response: str) -> List[Triplet]:
        """Parse triplets from LLM response"""
        triplets = []
        pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
        matches = re.findall(pattern, llm_response)
        
        for match in matches:
            subject = match[0].strip()
            predicate = match[1].strip()
            obj = match[2].strip()
            
            # Validation
            if all([subject, predicate, obj]) and all(len(x) < 200 for x in [subject, predicate, obj]):
                triplets.append(Triplet(subject, predicate, obj))
        
        return triplets