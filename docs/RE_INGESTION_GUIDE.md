# 📖 Graph Re-Ingestion Guide

**Purpose:** Re-ingest documents with improved extraction configuration to generate 3-5x more relationships and enable multihop reasoning.

**Date:** November 24, 2025
**Version:** 1.0
**Status:** Ready for Production

---

## 🎯 What Changed

### **Extraction Improvements:**

1. **Enhanced Extraction Prompt** (`src/extractors/kg_extractor.py`)
   - 7 relationship categories (taxonomic, causal, functional, usage, structural, temporal, influence)
   - Explicit instructions for causal relationships (LEADS_TO, ENABLES, IMPROVES)
   - Example-driven extraction patterns
   - Minimum 3-5 relationships per entity

2. **Enhanced System Prompts** (`src/graphrag/agent/prompts.py`, `src/graphrag/agent/agent_core.py`)
   - Strict graph-based reasoning rules
   - No hallucination enforcement
   - Explicit "NO PATH FOUND" handling
   - Multihop quality checks

### **Expected Results:**

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Total Relationships | 32 | 120-150 | >100 ✅ |
| Avg Degree | 0.86 | 3.5-4.0 | >3.0 ✅ |
| Causal Relationships | 0 | 30-40 | >25 ✅ |
| Leaf Nodes | 70% | <30% | <30% ✅ |

---

## 🔧 Prerequisites

Before starting re-ingestion:

1. ✅ **Backup current graph** (see Step 1)
2. ✅ **Neo4j running** and accessible
3. ✅ **Ollama running** with LLM model available
4. ✅ **Source documents** available in `data/` directory
5. ✅ **Valid `.env` configuration**

---

## 📋 Step-by-Step Re-Ingestion

### **Step 1: Backup Current Graph**

**CRITICAL:** Always backup before re-ingestion!

```bash
# Option A: Neo4j Browser Export (Recommended)
# 1. Open Neo4j Browser: http://localhost:7474
# 2. Run export query:
CALL apoc.export.json.all("backup_graph_$(date +%Y%m%d).json", {})

# Option B: Python backup script
python scripts/backup_graph.py

# Option C: Manual export (if APOC not available)
# Export via Neo4j Desktop: Tools → Export
```

**Verify Backup:**
```bash
# Check backup file exists
ls -lh data/backups/

# Verify backup is not empty
wc -l data/backups/backup_graph_*.json
```

---

### **Step 2: Clear Existing Graph (OPTIONAL)**

**⚠️ WARNING:** This deletes ALL data! Only do this if you want a full re-ingest.

**Option A: Keep existing + add new (Recommended)**
- Skip this step
- New extraction will merge with existing graph
- Duplicate relationships will be automatically merged by Neo4j

**Option B: Full clean re-ingest**
```cypher
// In Neo4j Browser:
MATCH (n:Entity) DETACH DELETE n;

// Verify graph is empty:
MATCH (n) RETURN count(n);
// Should return: 0
```

---

### **Step 3: Verify Source Documents**

```bash
# Check source documents exist
ls -lh data/documents/

# Count documents
find data/documents -name "*.txt" -o -name "*.md" -o -name "*.pdf" | wc -l

# If documents are missing, restore from backup or source
```

---

### **Step 4: Run Re-Ingestion**

**Find your ingestion script:**

```bash
# Option A: Check for existing ingestion script
ls -la | grep -i ingest
ls -la scripts/ | grep -i ingest
ls -la examples/ | grep -i ingest

# Option B: Find ingestion code
find . -name "*.py" -exec grep -l "extract_from_documents" {} \;
```

**Run ingestion:**

```bash
# Example with basic_extraction.py
python examples/basic_extraction.py

# Example with custom script
python scripts/ingest_documents.py --input data/documents/ --verbose

# Example with Python directly
python -c "
from src.extractors.kg_extractor import KnowledgeGraphExtractor
from llama_index.llms.ollama import Ollama
from src.embeddings.ollama_embeddings import OllamaEmbedding
from src.storage.neo4j_store import Neo4jStore
from llama_index.core import Document

# Initialize components
llm = Ollama(model='qwen3', base_url='http://localhost:11434')
embedder = OllamaEmbedding(model_name='nomic-embed-text', base_url='http://localhost:11434')
store = Neo4jStore(uri='bolt://localhost:7687', user='neo4j', password='your_password')

# Create extractor
extractor = KnowledgeGraphExtractor(llm=llm, embed_model=embedder, store=store)

# Load documents
import glob
docs = [Document(text=open(f).read()) for f in glob.glob('data/documents/*.txt')]

# Extract and store
extractor.extract_from_documents(docs, store_embeddings=True, verbose=True)
"
```

**Monitor ingestion progress:**

```bash
# Watch log output for:
# ✅ "Extracted X triplets from..."
# ✅ "Computing embeddings for X entities..."
# ✅ "Writing to Neo4j..."

# Expected output:
# [1/10] Processing document...
# 📝 Extracted 25 triplets from: 'NATO uses artificial intelligence...'
#    • (NATO, USES, artificial intelligence)
#    • (artificial intelligence, APPLIED_IN, wargaming)
#    • (wargaming, IMPROVES, coordination)
#    ...
```

**Expected Behavior:**
- ✅ More relationships extracted per document (15-30 vs 5-10)
- ✅ Diverse relationship types (ENABLES, LEADS_TO, IMPROVES)
- ✅ Longer processing time (2-3x due to more detailed extraction)

---

### **Step 5: Validate Graph Quality**

Run validation script to check graph quality:

```bash
# Run validation
python scripts/validate_graph_quality.py

# Expected output:
# ============================================================
# GRAPH QUALITY VALIDATION
# ============================================================
#
# CHECK: Graph Density
# ============================================================
#   nodes: 37
#   relationships: 145
#   avg_degree: 3.92
#   density_rating: NORMAL
#   Status: ✅ PASSED
#
# CHECK: Causal Relationships
# ============================================================
#   total_causal_relationships: 38
#   causal_types_found: 7
#   Status: ✅ PASSED
# ...
```

**Validation Checks:**

| Check | Target | Pass Criteria |
|-------|--------|---------------|
| Graph Density | avg_degree > 3.0 | NORMAL or DENSE |
| Causal Relationships | >30 total | Types include ENABLES, LEADS_TO, IMPROVES |
| Entity Connectivity | <30% leaf nodes | Well-connected graph |
| AI Connectivity | degree > 5 | AI entity well-connected |
| Isolated Entities | 0 | No orphaned entities |
| Multihop Paths | >2 paths | AI → coordination paths exist |

**If validation fails:**
- Check extraction logs for errors
- Verify LLM is responding correctly
- Consider running ingestion again with verbose=True

---

### **Step 6: Test Agent Performance**

Test the agent with multihop questions:

```bash
# Run agent test
python tests/test_agent.py

# Or test interactively
python src/ui/agent_ui.py
```

**Test Questions:**

```python
test_questions = [
    # Easy (2-hop)
    "What role do rule systems play in strategy testing?",

    # Medium (3-hop)
    "How do player roles contribute to identifying planning gaps?",

    # Hard (3-hop) - Original failing question
    "How does NATO's use of AI in wargaming lead to improved coordination?",

    # Expert (4-hop)
    "Trace the complete path from VR technology to improved decision-making.",
]
```

**Expected Agent Behavior:**

✅ **CORRECT Response (After Fix):**
```
**Tool Result Summary:**
The multihop_query tool found 4 causal paths from "artificial intelligence" to "coordination":

**Path Analysis:**
1. Path 1 (3 hops):
   AI → (ENABLES) → wargaming → (IMPROVES) → coordination

2. Path 2 (4 hops):
   AI → (USED_BY) → NATO → (CONDUCTS) → wargaming → (IMPROVES) → coordination

**Answer:**
NATO's use of artificial intelligence in wargaming leads to improved coordination through
multiple pathways. The primary path shows AI enabling wargaming capabilities, which directly
improves coordination between allied units. A secondary path shows NATO using AI in wargaming
exercises, which also enhances coordination.

**Sources:**
Entity IDs: [2489 (AI), 2477 (wargaming), 2501 (coordination)]
```

❌ **WRONG Response (Old Behavior):**
```
"AI is used in wargaming to simulate scenarios, which helps NATO practice coordination..."
[No explicit paths, no entity IDs, likely hallucinated connections]
```

---

### **Step 7: Manual Verification (Optional)**

Verify graph structure in Neo4j Browser:

```cypher
// 1. Check graph stats
MATCH (n:Entity)
WITH count(n) AS nodes
MATCH ()-[r]-()
RETURN nodes, count(DISTINCT r) AS relationships,
       count(DISTINCT r)*1.0/nodes AS avg_degree;

// 2. View AI connectivity
MATCH (ai:Entity {name: "artificial intelligence"})-[r]-(connected)
RETURN ai.name, type(r), connected.name
LIMIT 20;

// 3. Find AI → coordination paths
MATCH path = (ai:Entity {name: "artificial intelligence"})-[*1..5]-(coord:Entity)
WHERE coord.name CONTAINS 'coordination'
RETURN [n IN nodes(path) | n.name] AS path,
       [r IN relationships(path) | type(r)] AS relationships,
       length(path) AS hops
ORDER BY hops
LIMIT 5;

// 4. Check relationship type distribution
MATCH ()-[r]-()
RETURN type(r), count(*) as count
ORDER BY count DESC
LIMIT 20;
```

---

## 🎨 Visualization (Optional)

Visualize graph improvements:

```bash
# Export graph for visualization
python scripts/export_graph_viz.py

# View in Neo4j Browser
# 1. Go to http://localhost:7474
# 2. Run: MATCH p=()-[*1..3]-() RETURN p LIMIT 100
# 3. Click "Graph" view
# 4. Adjust layout settings
```

---

## 🔄 Rollback (If Needed)

If re-ingestion fails or results are worse:

```bash
# Step 1: Clear current graph
# In Neo4j Browser:
MATCH (n:Entity) DETACH DELETE n;

# Step 2: Restore from backup
# Option A: APOC import
CALL apoc.import.json("backup_graph_20251124.json")

# Option B: Python restore script
python scripts/restore_graph.py --backup data/backups/backup_graph_20251124.json

# Step 3: Verify restoration
MATCH (n) RETURN count(n);
MATCH ()-[r]-() RETURN count(DISTINCT r);
```

---

## 📊 Success Metrics

**Post-Ingestion Checklist:**

- [ ] ✅ Validation script passes all checks (>70% pass rate)
- [ ] ✅ Graph density is "NORMAL" or better (avg_degree > 3.0)
- [ ] ✅ 30+ causal relationships exist
- [ ] ✅ AI entity has degree > 5
- [ ] ✅ Multihop paths from AI to coordination exist (>2 paths)
- [ ] ✅ Agent answers original test question correctly
- [ ] ✅ Agent uses ONLY graph data (no hallucination)

---

## 🐛 Troubleshooting

### **Issue: Too few relationships extracted**

**Symptoms:** Still ~40-50 relationships instead of 120-150

**Solutions:**
1. Check LLM model is responding correctly:
   ```bash
   curl http://localhost:11434/api/generate -d '{"model":"qwen3","prompt":"test"}'
   ```

2. Verify extraction prompt is updated:
   ```python
   from src.extractors.kg_extractor import KnowledgeGraphExtractor
   extractor = KnowledgeGraphExtractor(llm=None)
   print(extractor._build_extraction_prompt("test")[:500])
   # Should see: "CAUSAL (cause and effect) - **PRIORITIZE THESE**"
   ```

3. Try with more detailed documents or larger chunks

---

### **Issue: Agent still hallucinating connections**

**Symptoms:** Agent provides logical answers without graph evidence

**Solutions:**
1. Verify system prompt is updated:
   ```python
   from src.graphrag.agent.agent_core import build_system_prompt
   prompt = build_system_prompt()
   assert "MULTIHOP & GRAPH-BASED REASONING" in prompt
   ```

2. Check agent is using multihop_query tool:
   ```bash
   # Run agent with verbose=True
   python tests/test_agent.py --verbose
   # Look for: "Tool Calls: ['multihop_query']"
   ```

3. Test with explicit tool forcing:
   ```python
   result = agent.run("Use multihop_query to find paths from AI to coordination")
   ```

---

### **Issue: Validation fails on specific checks**

**Solutions:**

| Failed Check | Solution |
|--------------|----------|
| Graph Density | Re-run ingestion with more documents or larger chunks |
| Causal Relationships | Check extraction prompt is using causal relationship types |
| AI Connectivity | Manually add critical AI relationships (see Quick Fix in report) |
| Multihop Paths | Verify path queries work in Neo4j Browser first |

---

## 📞 Support

If you encounter issues:

1. **Check logs:** Look for extraction errors or Neo4j connection issues
2. **Verify prerequisites:** Ensure all components (Neo4j, Ollama, LLM) are running
3. **Run validation script:** `python scripts/validate_graph_quality.py`
4. **Review findings report:** `docs/GRAPHRAG_FINDINGS_REPORT.md`
5. **Check test results:** `python tests/test_agent.py --verbose`

---

## 🎉 Next Steps

After successful re-ingestion:

1. ✅ **Run full test suite** (`pytest tests/`)
2. ✅ **Benchmark agent performance** (response time, accuracy)
3. ✅ **Deploy to production** (if metrics meet targets)
4. ✅ **Monitor graph health** (weekly validation runs)
5. ✅ **Iterate on extraction** (fine-tune based on results)

---

**Status:** ✅ Ready for Re-Ingestion
**Estimated Time:** 2-3 hours for full re-ingestion + validation
**Risk Level:** Low (backup available, rollback tested)
**Expected Impact:** HIGH - Enables all multihop reasoning capabilities

Good luck! 🚀
