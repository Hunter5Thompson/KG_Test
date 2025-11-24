# 🎯 Improved Extraction Prompt - Proposal

## Problem Analysis

**Current Prompt Issues:**
1. **Too long (89 lines)** → "Lost-in-the-Middle" effect
2. **Examples too late** → LLM forgets rules before seeing examples
3. **30+ relationship types** → LLM is overwhelmed
4. **Implicit relationships not clear** → Only extracts explicit facts
5. **Model limitation** → `qwen3` needs simpler, focused prompts

**Current Results:**
- Graph has isolated entities
- Missing connections: AI → wargaming, wargaming → coordination
- User needs manual Cypher queries to add obvious relationships

---

## 🎯 Solution: Focused Prompt (Option 1)

### Key Principles:
1. ✅ **Short & Focused** (30-40 lines max)
2. ✅ **Examples FIRST** (Few-Shot Learning)
3. ✅ **5-7 Core Relationships** (not 30+)
4. ✅ **Explicit + Implicit** rules
5. ✅ **Domain-Specific** (military wargaming)

### Improved Prompt Structure:

```python
def _build_extraction_prompt(self, text: str) -> str:
    """Build optimized extraction prompt (IMPROVED VERSION)"""
    return f"""Extract knowledge graph triplets from the text below.

Domain: Military wargaming, strategy, and planning.

=== EXAMPLES (Follow These Patterns) ===

Text: "NATO uses artificial intelligence in wargaming exercises to improve coordination between allied units."

Entities: NATO, artificial intelligence, wargaming exercises, coordination, allied units

Relationships:
(NATO, USES, artificial intelligence)
(artificial intelligence, ENABLES, wargaming exercises)
(NATO, CONDUCTS, wargaming exercises)
(wargaming exercises, IMPROVES, coordination)
(coordination, APPLIES_TO, allied units)

---

Text: "Scenario design enables realistic testing which leads to better strategy validation."

Entities: scenario design, realistic testing, strategy validation

Relationships:
(scenario design, ENABLES, realistic testing)
(realistic testing, LEADS_TO, strategy validation)

---

=== YOUR TASK ===

For the text below, extract ALL entities and relationships.

**Entity Types:** technologies, processes, organizations, outcomes, concepts

**Relationship Types (Use These 7):**
1. **ENABLES** - X makes Y possible
2. **IMPROVES** - X makes Y better
3. **USES** - X applies/employs Y
4. **LEADS_TO** - X causes/results in Y
5. **IS_PART_OF** - X is component of Y
6. **APPLIES_TO** - X is relevant for Y
7. **CONDUCTS** - X performs/executes Y

**CRITICAL RULES:**

✅ Extract BOTH explicit AND implicit relationships
✅ From "NATO uses AI in wargaming" extract:
   - (NATO, USES, AI)
   - (AI, ENABLES, wargaming)      ← IMPLICIT but clear
   - (NATO, CONDUCTS, wargaming)   ← IMPLICIT but clear

✅ From "X improves Y for Z" extract:
   - (X, IMPROVES, Y)
   - (Y, APPLIES_TO, Z)

✅ Aim for 5-8 relationships per sentence
✅ Use specific relationship types (ENABLES, not RELATED_TO)
✅ Extract causal chains: X→Y→Z means extract both (X,ENABLES,Y) and (Y,LEADS_TO,Z)

❌ Do NOT create far-fetched relationships between unrelated concepts
❌ Do NOT use relationship types outside the 7 listed above

---

**FORMAT:** Return ONLY triplets in format: (subject, relation, object)
One triplet per line.

**TEXT:** {text}

**TRIPLETS:**"""
```

---

## 📊 Comparison: Old vs New

| Aspect | OLD Prompt | NEW Prompt |
|--------|-----------|------------|
| **Length** | 89 lines | ~40 lines |
| **Relationship Types** | 30+ types | 7 focused types |
| **Examples Position** | Line 54-88 | Line 6-25 (TOP) |
| **Implicit Rules** | Mentioned but unclear | Explicit examples |
| **Domain Focus** | Generic knowledge | Military wargaming |
| **Few-Shot Learning** | 2 examples at end | 2 examples at start |

---

## 🧪 Expected Improvements

**Before (Current):**
```
Text: "NATO uses AI in wargaming to improve coordination"
→ Extracts: (NATO, USES, AI)
→ Missing: AI→wargaming, wargaming→coordination
```

**After (Improved):**
```
Text: "NATO uses AI in wargaming to improve coordination"
→ Extracts:
  (NATO, USES, AI)
  (AI, ENABLES, wargaming)         ← NOW EXTRACTED
  (NATO, CONDUCTS, wargaming)       ← NOW EXTRACTED
  (wargaming, IMPROVES, coordination) ← NOW EXTRACTED
```

**Metrics:**
- **Before:** ~0.86 avg degree (very sparse)
- **Target:** >3.0 avg degree (well-connected)
- **Expected:** 3.5-4.5 avg degree with improved prompt

---

## 🔧 Alternative Options

### **Option 2: Two-Phase Extraction**

**Phase 1 Prompt:** "Extract only entities from this text"
**Phase 2 Prompt:** "Given these entities: [...], find all relationships"

**Pros:** Clearer focus per phase
**Cons:** 2x LLM calls (slower, more expensive)

---

### **Option 3: Chain-of-Thought**

**Prompt structure:**
```
Step 1: List all entities you see
Step 2: For each entity pair, ask: "Is there a relationship?"
Step 3: For each relationship, classify the type
Step 4: Output triplets
```

**Pros:** LLM "thinks" more deeply
**Cons:** Longer output, may be verbose

---

## ✅ Recommendation

**Use Option 1 (Focused Prompt)** because:
1. ✅ Fastest to implement (single file change)
2. ✅ No architecture changes needed
3. ✅ Proven pattern (Few-Shot Learning)
4. ✅ Works well with smaller LLMs like `qwen3`
5. ✅ Easy to test and iterate

**Next Steps:**
1. Implement improved prompt in `src/extractors/kg_extractor.py`
2. Test on sample documents
3. Compare extraction results
4. Re-ingest if results are better

---

## 📝 Implementation Plan

```bash
# 1. Backup current implementation
cp src/extractors/kg_extractor.py src/extractors/kg_extractor.py.backup

# 2. Apply new prompt (replace _build_extraction_prompt method)

# 3. Test with sample text
python examples/test_improved_extraction.py

# 4. Compare results
# Old: 2-3 triplets per document
# New: 8-12 triplets per document (target)

# 5. If successful: Re-ingest documents
python examples/basic_extraction.py
```

---

## 🎯 Success Criteria

After implementing improved prompt:

- [ ] Graph avg_degree > 3.0
- [ ] AI entity has >5 connections
- [ ] Paths exist: AI → wargaming → coordination
- [ ] No manual Cypher queries needed
- [ ] Agent can answer multihop questions

---

**Status:** Ready for implementation
**Risk:** Low (easy to rollback)
**Expected Impact:** HIGH (3-5x more relationships)
