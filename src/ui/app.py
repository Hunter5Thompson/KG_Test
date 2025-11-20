# File: src/ui/app.py
"""
Streamlit UI for Knowledge Graph Extraction
POC for file upload and processing with robust imports/caching and better error messages.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
from llama_index.core import Document

# ---- Robust imports (avoid brittle sys.path hacks) -------------------------
# Expectation: run from project root with `uv run streamlit run src/ui/app.py`
# If not, try to add project root/src dynamically as fallback.
try:
    from config.settings import AppConfig
    from src.embeddings.ollama_embeddings import OllamaEmbedding
    from src.extractors.kg_extractor import KnowledgeGraphExtractor
    from src.storage.neo4j_store import Neo4jStore
    from src.ui.file_processor import FileProcessor, chunk_text
    from src.graphrag.authenticated_ollama_llm import AuthenticatedOllamaLLM
    from src.ui.agent_ui import render_agent_chat, render_agent_playground
except Exception:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in os.sys.path:  # ← ROOT statt SRC!
        os.sys.path.insert(0, str(ROOT))
    from config.settings import AppConfig
    from src.embeddings.ollama_embeddings import OllamaEmbedding
    from src.extractors.kg_extractor import KnowledgeGraphExtractor
    from src.storage.neo4j_store import Neo4jStore
    from src.ui.file_processor import FileProcessor, chunk_text
    from src.graphrag.authenticated_ollama_llm import AuthenticatedOllamaLLM
    from src.ui.agent_ui import render_agent_chat, render_agent_playground

# ---------------------------------------------------------------------------

st.set_page_config(page_title="Knowledge Graph Extractor", page_icon="🧠", layout="wide")


def init_session_state() -> None:
    """Initialize session state variables."""
    st.session_state.setdefault("config_loaded", False)
    st.session_state.setdefault("extractor_ready", False)
    st.session_state.setdefault("processed_files", [])  # store only metadata here (name/size)


def load_config() -> AppConfig | None:
    """Load configuration from environment and validate."""
    try:
        config = AppConfig.from_env()

        missing = []
        if not (config.ollama and config.ollama.host):
            missing.append("OLLAMA_HOST")
        if not (config.neo4j and config.neo4j.password):
            missing.append("NEO4J_PASSWORD")
        if missing:
            st.error("❌ Missing required configuration: " + ", ".join(missing))
            return None

        return config
    except Exception as e:
        st.error(f"❌ Config error: {e}")
        return None


@st.cache_resource(show_spinner=False)
def init_components_cached(cache_key: tuple[str, str, str, str, str]) -> Dict[str, Any] | None:
    """
    Initialize KG extraction components (cached).
    cache_key forces re-init if host/model/db changes.
    """
    try:
        # Re-hydrate config from cache_key if needed (or pass via session)
        # Here we still read from session_state to get full object:
        config: AppConfig = st.session_state.config

        # Initialize LLM with Authentication
        llm = AuthenticatedOllamaLLM(
            model_name=config.ollama.llm_model,
            base_url=config.ollama.host,
            api_key=config.ollama.api_key,
            request_timeout=120.0,
            max_tokens=1000,
            temperature=0.1,
        )

        # Initialize Embeddings
        embed_model = OllamaEmbedding(
            model_name=config.ollama.embedding_model,
            base_url=config.ollama.host,
            api_key=config.ollama.api_key,
        )

        # Initialize Neo4j Store
        store = Neo4jStore(
            uri=config.neo4j.uri,
            user=config.neo4j.user,
            password=config.neo4j.password,
            database=config.neo4j.database,
        )

        # Initialize Extractor
        extractor = KnowledgeGraphExtractor(llm=llm, embed_model=embed_model, store=store)

        # File Processor
        file_processor = FileProcessor()

        return {
            "extractor": extractor,
            "store": store,
            "embed_model": embed_model,
            "file_processor": file_processor,
        }
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        return None


def render_sidebar() -> None:
    """Render sidebar with config and controls."""
    with st.sidebar:
        st.title("⚙️ Configuration")

        if st.session_state.config_loaded:
            st.success("✅ Config loaded")
        else:
            st.warning("⚠️ Config not loaded")

        if st.session_state.extractor_ready:
            st.success("✅ Extractor ready")
        else:
            st.warning("⚠️ Extractor not ready")

        st.divider()
        st.subheader("🗄️ Database")

        if st.button("Clear Neo4j Graph", type="secondary", use_container_width=True):
            if st.session_state.extractor_ready:
                components = st.session_state.components
                try:
                    components["store"].clear()
                    st.success("🧹 Graph cleared")
                except Exception as e:
                    st.error(f"❌ Clear failed: {e}")

        if st.button("Show Stats", type="secondary", use_container_width=True):
            if st.session_state.extractor_ready:
                components = st.session_state.components
                try:
                    stats = components["store"].get_stats()
                    st.metric("Nodes", stats.get("nodes", 0))
                    st.metric("Relationships", stats.get("relationships", 0))
                except Exception as e:
                    st.error(f"❌ Stats failed: {e}")

        st.divider()
        st.subheader("Cache / Runtime")

        if st.button("🔄 Reset cached components", type="secondary", use_container_width=True):
            init_components_cached.clear()
            st.success("Cache cleared. Components will re-initialize on next run.")
            st.rerun()


def render_file_upload(components: Dict[str, Any]) -> None:
    """Render file upload section."""
    st.header("📁 Upload Documents")

    file_processor: FileProcessor = components["file_processor"]

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD files", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("👆 Upload files to start extraction")
        return

    st.subheader(f"📄 {len(uploaded_files)} file(s) uploaded")
    for file in uploaded_files:
        with st.expander(f"📄 {file.name} ({file.size:,} bytes)"):
            if file_processor.is_supported(file.name):
                st.success(f"✅ Supported format: {Path(file.name).suffix}")
            else:
                st.error(f"❌ Unsupported format: {Path(file.name).suffix}")

    col1, col2 = st.columns(2)
    with col1:
        use_chunking = st.checkbox(
            "Enable text chunking (for large files)",
            value=True,
            help="Split large documents into smaller chunks",
        )
    with col2:
        chunk_size = st.number_input(
            "Chunk size (characters)",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500,
            disabled=not use_chunking,
        )

    if st.button("🚀 Extract Knowledge Graph", type="primary", use_container_width=True):
        process_files(uploaded_files, components, use_chunking, int(chunk_size))


def process_files(
    uploaded_files: List[Any],
    components: Dict[str, Any],
    use_chunking: bool,
    chunk_size: int,
) -> None:
    """Process uploaded files and extract knowledge graph."""
    file_processor: FileProcessor = components["file_processor"]
    extractor: KnowledgeGraphExtractor = components["extractor"]

    progress_bar = st.progress(0)
    status_text = st.empty()

    all_documents: List[Document] = []
    total_files = len(uploaded_files)

    # Step 1: Extract text from files
    status_text.text("📄 Extracting text from files...")

    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i / max(total_files, 1)) * 0.3  # 0-30% for file extraction
        progress_bar.progress(progress)

        try:
            file_bytes = uploaded_file.read()
            result = file_processor.process_uploaded_file(file_bytes, uploaded_file.name)

            if result.error:
                st.error(f"❌ {uploaded_file.name}: {result.error}")
                continue

            st.success(f"✅ {uploaded_file.name}: {result.char_count:,} characters")

            # Chunking
            if use_chunking and result.char_count > chunk_size:
                chunks = chunk_text(result.content, chunk_size=chunk_size)
                st.info(f"   → Split into {len(chunks)} chunks")
                for j, chunk in enumerate(chunks):
                    all_documents.append(
                        Document(
                            text=chunk,
                            metadata={"filename": uploaded_file.name, "chunk": j + 1, "total_chunks": len(chunks)},
                        )
                    )
            else:
                all_documents.append(Document(text=result.content, metadata={"filename": uploaded_file.name}))
        finally:
            # Streamlit UploadedFile: kein persistenter Handle notwendig
            pass

    if not all_documents:
        st.error("❌ No documents to process")
        return

    # Step 2: Extract knowledge graph
    status_text.text(f"🧠 Extracting knowledge graph from {len(all_documents)} document(s)...")
    progress_bar.progress(0.3)

    try:
        with st.spinner("Processing... This may take a few minutes"):
            stats = extractor.extract_from_documents(
                documents=all_documents,
                store_embeddings=True,
                verbose=False,  # UI nicht zuspammen
            )

        progress_bar.progress(1.0)
        status_text.text("✅ Extraction complete!")

        st.success("🎉 Knowledge Graph extracted successfully!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Files Processed", total_files)
        col2.metric("Documents", stats.get("documents_processed", 0))
        col3.metric("Triplets", stats.get("triplets_extracted", 0))

        # Nur Metadaten im Session-State speichern (kein UploadedFile-Objekt)
        st.session_state.processed_files.extend(
            [{"name": f.name, "size": f.size} for f in uploaded_files]
        )

    except Exception as e:
        # Häufiger Fall: LLM liefert leere Chat-Message → ValueError aus AuthenticatedOllamaLLM
        st.error(
            "❌ Extraction failed: "
            f"{e}\n\n"
            "Troubleshooting:\n"
            "• Prüfe, ob dein Ollama-Gateway in /api/chat ein nicht-leeres 'message.content' liefert.\n"
            "• Teste ein alternatives Modell oder passe die Endpoint-Reihenfolge an.\n"
            "• Nutze den 'Reset cached components'-Button, wenn du ENV/Model gewechselt hast."
        )
        progress_bar.empty()
        status_text.empty()


def render_graph_stats(components: Dict[str, Any]) -> None:
    """Render graph statistics."""
    store: Neo4jStore = components["store"]
    st.header("📊 Knowledge Graph Statistics")

    try:
        stats = store.get_stats()
        col1, col2 = st.columns(2)
        col1.metric("Total Nodes", stats.get("nodes", 0))
        col2.metric("Total Relationships", stats.get("relationships", 0))

        if stats.get("relationships", 0) > 0:
            st.subheader("🔗 Sample Triplets")
            for triplet in store.get_triplets(limit=10):
                st.text(f"• {triplet}")
    except Exception as e:
        st.error(f"❌ Failed to load stats: {e}")


def render_semantic_search(components: Dict[str, Any]) -> None:
    """Render semantic search interface."""
    st.header("🔍 Semantic Search")

    query = st.text_input("Search for entities:", placeholder="e.g., CEO, technology, location...")
    if not query:
        return

    try:
        embed_model: OllamaEmbedding = components["embed_model"]
        store: Neo4jStore = components["store"]

        query_embedding = embed_model.get_query_embedding(query)
        results = store.semantic_search(query_embedding, limit=5)

        if results:
            st.subheader(f"Top {len(results)} matches:")
            for i, result in enumerate(results, 1):
                similarity = result["similarity"]
                entity = result["entity"]

                icon = "🟢" if similarity > 0.8 else "🟡" if similarity > 0.6 else "🔴"
                st.text(f"{icon} {i}. {entity} (similarity: {similarity:.3f})")
        else:
            st.info("No results found")
    except Exception as e:
        st.error(f"❌ Search failed: {e}")


def main() -> None:
    """Main application entry point."""
    init_session_state()

    st.title("🧠 Knowledge Graph Extractor")
    st.markdown("*Extract knowledge graphs from documents using LLMs*")

    # Load config once
    if not st.session_state.config_loaded:
        config = load_config()
        if config:
            st.session_state.config = config
            st.session_state.config_loaded = True
        else:
            st.stop()

    # Build cache key for components
    cfg = st.session_state.config
    cache_key = (
        cfg.ollama.host,
        cfg.ollama.llm_model,
        cfg.ollama.embedding_model,
        cfg.neo4j.uri,
        cfg.neo4j.database,
    )

    # Initialize components via cache
    if not st.session_state.extractor_ready:
        with st.spinner("🔧 Initializing components..."):
            components = init_components_cached(cache_key)
            if components:
                st.session_state.components = components
                st.session_state.extractor_ready = True
            else:
                st.stop()

    # UI
    render_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📁 Upload", "📊 Statistics", "🔍 Search", "🤖 Query Graph", "🧪 Playground"]
    )

    with tab1:
        render_file_upload(st.session_state.components)

    with tab2:
        render_graph_stats(st.session_state.components)

    with tab3:
        render_semantic_search(st.session_state.components)

    with tab4:
        # Diese UIs nutzen interne Session Keys – sollten selbst robust sein
        render_agent_chat()

    with tab5:
        render_agent_playground()


if __name__ == "__main__":
    main()
