"""
Knowledge Graph Extraction from text using LLMs
Features: 
- Two-Phase Extraction (Scout & Connect)
- YAML-based Configuration (Synonyms, Ontology)
- Advanced Entity Normalization
"""
import re
import yaml
import os
from typing import List, Set, Dict, Optional
from llama_index.core import Document
from llama_index.llms.ollama import Ollama

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
    Extract knowledge graphs from text using LLMs with advanced normalization and configuration.
    """
    
    def __init__(
        self,
        llm: Ollama,
        embed_model: OllamaEmbedding = None,
        store: Neo4jStore = None,
        config_path: str = "config/domain_rules.yaml"
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.store = store
        
        # Load Configuration from YAML
        self.config = self._load_config(config_path)
        
        # Parse Config into efficient structures
        self.ALLOWED_RELATIONS = set(self.config.get("allowed_relations", []))
        self.ACRONYMS = set(self.config.get("acronyms", []))
        self.SYNONYMS = self.config.get("synonyms", {})
        
        print(f"✅ KG Extractor initialized (Production Mode)")
        print(f"   Loaded Configuration: {config_path}")
        print(f"   Ontology: {len(self.ALLOWED_RELATIONS)} relations, {len(self.SYNONYMS)} synonyms defined.")
        
        if embed_model:
            print(f"   Embeddings: {embed_model.model_name}")
        if store:
            print(f"   Storage: Neo4j")

    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML file safely"""
        # Fix path if running from root
        if not os.path.exists(path):
            # Try looking one level up just in case
            alt_path = os.path.join("..", path)
            if os.path.exists(alt_path):
                path = alt_path
            else:
                print(f"⚠️  WARNING: Config file not found at '{path}'. Using empty defaults.")
                return {}
            
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                print(f"❌ Error parsing YAML: {exc}")
                return {}

    def extract_triplets_from_text(
        self,
        text: str,
        verbose: bool = True
    ) -> List[Triplet]:
        """
        Orchestrates the two-phase extraction process:
        1. Identify & Normalize Entities
        2. Extract Relationships between them
        """
        if not text or len(text.strip()) < 10:
            return []

        # PHASE 1: Entity Discovery & Normalization
        if verbose: 
            print("   🔍 Phase 1: Scouting Entities...")
        
        entities = self._extract_entities_only(text)
        
        if not entities:
            if verbose: print("   ⚠️ No entities found.")
            return []
            
        # PHASE 2: Relationship Mapping (Graph Construction)
        if verbose: 
            print(f"   🕸️  Phase 2: Connecting {len(entities)} normalized entities...")
            
        triplets = self._extract_dense_relationships(text, entities)
        
        if verbose:
            print(f"   📝 Extracted {len(triplets)} valid relationships")
            
        return triplets

    def extract_from_documents(
        self,
        documents: List[Document],
        store_embeddings: bool = True,
        verbose: bool = True
    ) -> dict:
        """
        Process multiple documents, deduplicate triplets, compute embeddings, and store.
        """
        all_triplets = []
        
        if verbose:
            print(f"\n📊 Processing {len(documents)} documents...")
        
        for i, doc in enumerate(documents, 1):
            if verbose: 
                print(f"\n[{i}/{len(documents)}] Processing document...")
            
            triplets = self.extract_triplets_from_text(doc.text, verbose=verbose)
            all_triplets.extend(triplets)
        
        # Deduplicate globally before storage
        unique_triplets = self._deduplicate_triplets(all_triplets)
        
        # Compute embeddings if requested
        entity_embeddings = None
        if store_embeddings and self.embed_model:
            entity_embeddings = self._compute_embeddings(unique_triplets, verbose=verbose)
        
        # Write to storage
        if self.store:
            if verbose:
                print(f"\n💾 Writing {len(unique_triplets)} triplets to Neo4j...")
            self.store.write_triplets(unique_triplets, entity_embeddings)
        
        return {
            "documents_processed": len(documents),
            "triplets_extracted": len(unique_triplets),
            "embeddings_computed": bool(entity_embeddings)
        }

    def _extract_entities_only(self, text: str) -> List[str]:
        """
        Phase 1: Scout entities and normalize them immediately via YAML/Heuristics.
        """
        prompt = f"""Identify the key entities (Concepts, Organizations, Technologies, Activities) in the text below.
Return ONLY a comma-separated list. Keep entities atomic (1-3 words).
Ignore generic terms like "it", "they", "features", "various".

TEXT: {text}

ENTITIES:"""
        
        response = self.llm.complete(prompt)
        raw_entities = response.text.split(',')
        
        # Clean, Normalize, and Deduplicate immediately
        clean_entities = set()
        for e in raw_entities:
            normalized = self._clean_entity(e)
            if normalized and len(normalized) > 2:
                clean_entities.add(normalized)
        
        return list(clean_entities)

    def _extract_dense_relationships(self, text: str, entities: List[str]) -> List[Triplet]:
        """
        Phase 2: Connect the nodes using the specific Allowed Relations from Config.
        """
        entity_str = ", ".join(entities)
        relations_str = ', '.join(self.ALLOWED_RELATIONS) if self.ALLOWED_RELATIONS else "RELATED_TO"
        
        prompt = f"""You are a Knowledge Graph Architect.
Task: Connect the provided entities based on the text to create a DENSE graph.

CONTEXT TEXT: 
{text}

AVAILABLE ENTITIES: 
{entity_str}

GOAL: 
Generate as many valid relationships between these entities as possible.
- **Cross-Linking**: Connect entities from the beginning of text to those at the end.
- **Density**: Every entity should ideally have 2+ connections.
- **Transitivity**: If A->B and B->C, consider if A->C (LEADS_TO) is true.

ALLOWED RELATIONS: {relations_str}

FORMAT: `(Subject | RELATION | Object)`
Use the provided entity names exactly. One triplet per line.

TRIPLETS:"""
        
        response = self.llm.complete(prompt)
        return self._parse_triplets(response.text)

    def _parse_triplets(self, llm_response: str) -> List[Triplet]:
        """
        Parses the LLM response using Pipe separator and re-cleans entities.
        """
        triplets = []
        # Regex captures ( Part1 | Part2 | Part3 )
        pattern = r'\(([^|]+)\|([^|]+)\|([^|]+)\)'
        matches = re.findall(pattern, llm_response)
        
        for match in matches:
            # Clean Subject and Object again to ensure they match Phase 1 normalization
            subj = self._clean_entity(match[0])
            pred = match[1].strip().upper().replace(" ", "_")
            obj = self._clean_entity(match[2])
            
            if self._is_valid(subj, pred, obj):
                triplets.append(Triplet(subj, pred, obj))
                
        return triplets

    def _clean_entity(self, text: str) -> str:
        """
        Standardizes entities using Config Rules and Heuristics.
        Ensures 'Docking' works across documents.
        """
        # 1. Remove noise (Pipes, quotes, bullets)
        text = text.replace("|", "").strip(' \n"\'().,;:-_*•')
        
        # 2. Normalize whitespace (remove double spaces)
        text = ' '.join(text.split())
        
        if not text or len(text) < 2:
            return ""

        text_lower = text.lower()
        
        # 3. YAML Synonym Lookup (Primary Source of Truth)
        if text_lower in self.SYNONYMS:
            return self.SYNONYMS[text_lower]
            
        # 4. Plural Normalization (Generic fallback)
        # Try converting to singular and check synonyms again
        singular = self._normalize_plural(text_lower)
        if singular != text_lower and singular in self.SYNONYMS:
             return self.SYNONYMS[singular]
        
        # 5. YAML Acronym Lookup
        if text.upper() in self.ACRONYMS:
            return text.upper()
            
        # 6. Heuristic Acronym Detection (if not in YAML)
        if self._is_acronym(text):
            return text.upper()

        # 7. Default Casing
        words = text.split()
        if len(words) == 1:
            # Single words -> lowercase (unless it looks like a specific proper noun)
            return text.lower()
        else:
            # Multi-words -> Title Case
            return text.title()

    def _normalize_plural(self, text: str) -> str:
        """Simple plural normalization helper"""
        if text.endswith('ies'):
            return text[:-3] + 'y'  # strategies -> strategy
        elif text.endswith('es') and not text.endswith('ss'):
            return text[:-2]        # exercises -> exercise
        elif text.endswith('s') and not text.endswith('ss'):
            return text[:-1]        # systems -> system
        return text

    def _is_acronym(self, text: str) -> bool:
        """Detect potential acronyms heuristically"""
        # If it's all uppercase and short (e.g. UAV, DOD)
        if text.isupper() and 2 <= len(text) <= 5:
            return True
        return False

    def _is_valid(self, subj: str, pred: str, obj: str) -> bool:
        """Validate triplet against rules"""
        if not subj or not pred or not obj: 
            return False
        # Prevent hallucinations of entire sentences
        if len(subj) > 60 or len(obj) > 60: 
            return False
        # Enforce Ontology
        if self.ALLOWED_RELATIONS and pred not in self.ALLOWED_RELATIONS:
            return False
        # Prevent self-loops
        if subj.lower() == obj.lower(): 
            return False
        return True

    def _deduplicate_triplets(self, triplets: List[Triplet]) -> List[Triplet]:
        """Removes exact duplicates from a list"""
        seen = set()
        unique = []
        for t in triplets:
            # Create a unique key based on normalized content
            key = (t.subject, t.predicate, t.object)
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

    def _compute_embeddings(
        self, 
        triplets: List[Triplet], 
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Compute embeddings for all unique entities"""
        entities = set()
        for triplet in triplets:
            entities.add(triplet.subject)
            entities.add(triplet.object)
        
        if verbose:
            print(f"🧮 Computing embeddings for {len(entities)} unique entities...")
        
        entity_embeddings = {}
        for i, entity in enumerate(entities, 1):
            entity_embeddings[entity] = self.embed_model.get_text_embedding(entity)
            if verbose and i % 10 == 0:
                print(f"   Progress: {i}/{len(entities)}")
                
        return entity_embeddings