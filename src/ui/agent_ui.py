"""
GraphRAG Query Interface (Streamlit Tab)
========================================

Integration des LangGraph Agents in die bestehende UI.
Kompatibel mit dem tatsächlichen LangGraph v1.0+ Agent.
"""

import streamlit as st
from datetime import datetime
from neo4j import GraphDatabase

from config.settings import config
from src.graphrag.agent import create_graphrag_agent, run_agent
from src.graphrag.langchain_ollama_auth import create_authenticated_ollama_llm
from src.embeddings.ollama_embeddings import OllamaEmbedding


def render_agent_chat():
    """
    Render GraphRAG Agent Chat Interface
    
    Features:
    - Chat History
    - Real-time Agent Execution
    - Context Display
    """
    st.header("🤖 GraphRAG Agent")
    st.markdown("*Ask questions about the knowledge graph*")
    
    # Initialize agent in session state (singleton)
    if 'graphrag_agent' not in st.session_state:
        try:
            with st.spinner("🔧 Initializing GraphRAG Agent..."):
                # Driver
                driver = GraphDatabase.driver(
                    config.neo4j.uri,
                    auth=(config.neo4j.user, config.neo4j.password)
                )
                
                # LLM
                llm = create_authenticated_ollama_llm(
                    model_name=config.ollama.llm_model,
                    base_url=config.ollama.host,
                    api_key=config.ollama.api_key,
                    temperature=0.0
                )
                
                # Embeddings
                embedder = OllamaEmbedding(
                    model_name=config.ollama.embedding_model,
                    base_url=config.ollama.host,
                    api_key=config.ollama.api_key
                )
                
                # Create Agent
                agent = create_graphrag_agent(
                    llm=llm,
                    driver=driver,
                    embed_fn=embedder.get_query_embedding,
                    max_iterations=10
                )
                
                st.session_state.graphrag_agent = agent
                st.session_state.graphrag_driver = driver
            
            st.success("✅ Agent ready!")
        
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {e}")
            return
    
    # Chat history
    if 'agent_chat_history' not in st.session_state:
        st.session_state.agent_chat_history = []
    
    # Display chat history
    for entry in st.session_state.agent_chat_history:
        with st.chat_message("user"):
            st.write(entry['question'])
        
        with st.chat_message("assistant"):
            st.write(entry['answer'])
            
            # Show tool info in expander
            if entry.get('metadata'):
                with st.expander("🔧 Agent Details"):
                    st.json(entry['metadata'])
    
    # Query input
    question = st.chat_input("Ask a question about the knowledge graph...")
    
    if question:
        # Add user message
        with st.chat_message("user"):
            st.write(question)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    agent = st.session_state.graphrag_agent
                    
                    # Run agent (captures intermediate steps)
                    response = run_agent(agent, question, verbose=False)
                    
                    # Display answer
                    st.write(response)
                    
                    # Add to history
                    st.session_state.agent_chat_history.append({
                        'question': question,
                        'answer': response,
                        'metadata': {
                            'model': config.ollama.llm_model,
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                
                except Exception as e:
                    st.error(f"❌ Query failed: {e}")
    
    # Sidebar controls
    with st.sidebar:
        st.divider()
        st.subheader("💬 Chat Controls")
        
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.agent_chat_history = []
            st.rerun()
        
        # Show agent stats
        if 'graphrag_agent' in st.session_state:
            st.metric("Total Queries", len(st.session_state.agent_chat_history))
            st.metric("Model", config.ollama.llm_model)


def render_agent_playground():
    """
    Render Agent Playground für Experimente
    
    Features:
    - Direct tool testing
    - Custom parameters
    - Raw output view
    """
    st.header("🧪 Agent Playground")
    st.markdown("*Test queries with custom parameters*")
    
    if 'graphrag_agent' not in st.session_state:
        st.warning("⚠️ Agent not initialized. Go to 'Query Graph' tab first.")
        return
    
    agent = st.session_state.graphrag_agent
    
    # Query input
    query = st.text_area(
        "Question",
        placeholder="What are the key components of wargaming?",
        height=100
    )
    
    col1, col2 = st.columns(2)
    with col1:
        max_iterations = st.slider("Max Tool Calls", 1, 20, 10)
    with col2:
        verbose = st.checkbox("Verbose Output", value=True)
    
    if st.button("🚀 Run Agent", type="primary"):
        if not query.strip():
            st.warning("Please enter a question")
            return
        
        with st.spinner("Agent is thinking..."):
            try:
                # Update max_iterations temporarily
                old_max = getattr(agent, 'max_iterations', 10)
                agent.max_iterations = max_iterations
                
                # Capture verbose output
                if verbose:
                    with st.expander("🔍 Agent Steps", expanded=True):
                        # Redirect stdout to capture prints
                        import io
                        import sys
                        old_stdout = sys.stdout
                        sys.stdout = buffer = io.StringIO()
                        
                        try:
                            response = run_agent(agent, query, verbose=True)
                            
                            # Get captured output
                            output = buffer.getvalue()
                            st.code(output, language="text")
                        finally:
                            sys.stdout = old_stdout
                else:
                    response = run_agent(agent, query, verbose=False)
                
                # Restore
                agent.max_iterations = old_max
                
                # Display result
                st.divider()
                st.subheader("📝 Agent Response:")
                st.write(response)
                
                # Metrics
                col1, col2 = st.columns(2)
                col1.metric("Max Iterations", max_iterations)
                col2.metric("Model", config.ollama.llm_model)
            
            except Exception as e:
                st.error(f"❌ Agent execution failed: {e}")
                import traceback
                with st.expander("🐛 Full Error"):
                    st.code(traceback.format_exc())
    
    # Additional Tools
    st.divider()
    st.subheader("🔧 Direct Tool Access")
    
    tool_type = st.selectbox(
        "Select Tool",
        ["Hybrid Retrieve", "Semantic Search", "Graph Query"]
    )
    
    if tool_type == "Hybrid Retrieve":
        st.markdown("**Test the hybrid retrieval tool directly**")
        
        search_query = st.text_input("Search Query", placeholder="wargaming methodologies")
        top_k = st.slider("Top K", 1, 20, 5)
        
        if st.button("🔍 Search"):
            if search_query:
                with st.spinner("Searching..."):
                    try:
                        from src.graphrag.hybrid_retriever import HybridGraphRetriever
                        from src.embeddings.ollama_embeddings import OllamaEmbedding
                        
                        embedder = OllamaEmbedding(
                            model_name=config.ollama.embedding_model,
                            base_url=config.ollama.host,
                            api_key=config.ollama.api_key
                        )
                        
                        retriever = HybridGraphRetriever(
                            st.session_state.graphrag_driver,
                            embedder.get_query_embedding
                        )
                        
                        results = retriever.retrieve(
                            query=search_query,
                            strategy="hybrid",
                            top_k=top_k
                        )
                        
                        st.subheader(f"Found {len(results)} results:")
                        for i, result in enumerate(results, 1):
                            # Handle both dict and object
                            if isinstance(result, dict):
                                entity = result.get('entity', 'Unknown')
                                score = result.get('score', 0.0)
                                desc = result.get('description', '')
                            else:
                                entity = getattr(result, 'entity', 'Unknown')
                                score = getattr(result, 'score', 0.0)
                                desc = getattr(result, 'description', '')
                            
                            with st.expander(f"{i}. {entity} (score: {score:.3f})"):
                                st.text(desc[:500])
                    
                    except Exception as e:
                        st.error(f"Search failed: {e}")
    
    elif tool_type == "Semantic Search":
        st.markdown("**Vector-only semantic search**")
        
        search_query = st.text_input("Search Query", placeholder="NATO wargaming")
        top_k = st.slider("Top K", 1, 20, 5)
        
        if st.button("🔍 Search"):
            if search_query:
                with st.spinner("Searching..."):
                    try:
                        from src.graphrag.hybrid_retriever import HybridGraphRetriever
                        from src.embeddings.ollama_embeddings import OllamaEmbedding
                        
                        embedder = OllamaEmbedding(
                            model_name=config.ollama.embedding_model,
                            base_url=config.ollama.host,
                            api_key=config.ollama.api_key
                        )
                        
                        retriever = HybridGraphRetriever(
                            st.session_state.graphrag_driver,
                            embedder.get_query_embedding
                        )
                        
                        results = retriever.retrieve(
                            query=search_query,
                            strategy="vector",
                            top_k=top_k
                        )
                        
                        st.subheader(f"Found {len(results)} results:")
                        for i, result in enumerate(results, 1):
                            if isinstance(result, dict):
                                entity = result.get('entity', 'Unknown')
                                score = result.get('score', 0.0)
                            else:
                                entity = getattr(result, 'entity', 'Unknown')
                                score = getattr(result, 'score', 0.0)
                            
                            st.text(f"{i}. {entity} (score: {score:.3f})")
                    
                    except Exception as e:
                        st.error(f"Search failed: {e}")
    
    else:  # Graph Query
        st.markdown("**Direct Neo4j query (READ-ONLY)**")
        st.warning("⚠️ Advanced users only. WRITE operations are blocked.")
        
        cypher = st.text_area(
            "Cypher Query",
            placeholder="MATCH (n:Entity) RETURN n.id, n.name LIMIT 10",
            height=150
        )
        
        if st.button("▶️ Execute"):
            if cypher:
                try:
                    # Basic safety check
                    dangerous = ['CREATE', 'DELETE', 'SET', 'REMOVE', 'MERGE', 'DROP']
                    if any(kw in cypher.upper() for kw in dangerous):
                        st.error("❌ WRITE operations not allowed!")
                    else:
                        with st.session_state.graphrag_driver.session() as session:
                            result = session.run(cypher)
                            records = [dict(r) for r in result]
                        
                        st.subheader(f"Results ({len(records)} records):")
                        st.json(records)
                
                except Exception as e:
                    st.error(f"Query failed: {e}")