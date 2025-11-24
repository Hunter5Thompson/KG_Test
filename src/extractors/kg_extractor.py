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
        """Build optimized extraction prompt with focus on causal relationships"""
        return f"""Extract ALL knowledge graph triplets from the following text.

You are extracting a knowledge graph about military wargaming, strategy, and planning.

ENTITIES: Extract key concepts (nouns, processes, outcomes, technologies, organizations).

RELATIONSHIPS: Extract ALL of these types:

1. TAXONOMIC (classification):
   - IS_A, IS_PART_OF, IS_TYPE_OF, CATEGORY_OF, INCLUDES, COMPRISES

2. CAUSAL (cause and effect) - **PRIORITIZE THESE**:
   - LEADS_TO, CAUSES, RESULTS_IN, PRODUCES, GENERATES, CREATES

3. FUNCTIONAL (purpose and enablement):
   - ENABLES, SUPPORTS, IMPROVES, ENHANCES, FACILITATES, STRENGTHENS

4. USAGE (application):
   - USES, USED_FOR, APPLIED_IN, APPLIED_TO, EMPLOYS, UTILIZES

5. STRUCTURAL (composition):
   - INVOLVES, CONTAINS, HAS_COMPONENT, CONSISTS_OF

6. TEMPORAL (sequence):
   - FOLLOWED_BY, PRECEDES, OCCURS_DURING, SUCCEEDS

7. INFLUENCE (impact):
   - AFFECTS, INFLUENCES, IMPACTS, SHAPES, DETERMINES

CRITICAL EXTRACTION RULES:

✅ Extract BOTH explicit AND implicit relationships
✅ From "X in Y to achieve Z" extract: (X, APPLIED_IN, Y) AND (Y, ACHIEVES, Z)
✅ From "X improves Y" extract: (X, IMPROVES, Y)
✅ From "X enables Y which leads to Z" extract: (X, ENABLES, Y) AND (Y, LEADS_TO, Z)
✅ Aim for minimum 3-5 relationships per entity
✅ Prefer specific relationship types (ENABLES) over generic ones (RELATED_TO)
✅ Extract causal chains: if X causes Y and Y causes Z, extract both relationships

❌ Do NOT create relationships between completely unrelated concepts
❌ Do NOT duplicate relationships with different names
❌ Do NOT use generic "RELATED_TO" - use specific relationship types

EXTRACTION STRATEGY:
1. Identify all entities (people, places, concepts, technologies, processes)
2. For each entity, identify:
   - What type/category is it? → IS_A, IS_TYPE_OF
   - What does it do/enable? → ENABLES, SUPPORTS, IMPROVES
   - What uses it? → USES, EMPLOYS
   - What does it affect? → AFFECTS, INFLUENCES, LEADS_TO
   - What comes before/after? → PRECEDES, FOLLOWED_BY

EXAMPLES:

Text: "NATO uses artificial intelligence in wargaming exercises to improve coordination between allied units."

Extract Entities:
- NATO
- artificial intelligence
- wargaming exercises
- coordination
- allied units

Extract Relationships:
(NATO, USES, artificial intelligence)
(artificial intelligence, APPLIED_IN, wargaming exercises)
(wargaming exercises, IMPROVES, coordination)
(coordination, APPLIES_TO, allied units)
(NATO, CONDUCTS, wargaming exercises)

Text: "Scenario design enables realistic testing which leads to better strategy validation."

Extract Entities:
- scenario design
- realistic testing
- strategy validation

Extract Relationships:
(scenario design, ENABLES, realistic testing)
(realistic testing, LEADS_TO, strategy validation)
(realistic testing, IS_TYPE_OF, testing)

FORMAT:
Return ONLY triplets in format: (subject, relation, object)
One triplet per line.

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