# ============================================================
# app.py — Streamlit Chat UI for MultiModal RAG System
# ============================================================
# Main entry point for the application. Provides:
#   - File upload sidebar (PDF, TXT, DOCX, XLSX, CSV, PNG, JPG, JPEG)
#   - LLM status indicators (Gemini & Ollama)
#   - Chat interface with conversation history
#   - Agent and LLM source labels on each response
#
# Run with: streamlit run app.py
# ============================================================

import os
import sys
import tempfile
import streamlit as st

# Ensure the project root is on the Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.document_agent import document_agent
from agents.excel_agent import excel_agent
from agents.image_agent import image_agent
from agents.coordinator import coordinator
from agents.aggregator import aggregator
from core.memory import conversation_memory
from core.vector_store import vector_store
from core.llm_router import check_llm_status


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="MultiModal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom CSS for dark-themed chat UI
# ============================================================
st.markdown(
    """
    <style>
    /* Dark background for the main content area */
    .stApp {
        background-color: #0e1117;
    }

    /* User message bubble — right-aligned, blue accent */
    .user-message {
        background: linear-gradient(135deg, #1a73e8, #1565c0);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 75%;
        margin-left: auto;
        text-align: left;
        word-wrap: break-word;
    }

    /* Assistant message bubble — left-aligned, dark grey */
    .assistant-message {
        background: #1e1e2e;
        color: #e0e0e0;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 75%;
        border: 1px solid #333;
        word-wrap: break-word;
    }

    /* Subtle metadata labels below assistant messages */
    .meta-label {
        font-size: 0.75rem;
        color: #888;
        margin-top: 2px;
        margin-bottom: 12px;
    }

    /* Status badge styling */
    .status-online {
        color: #4caf50;
        font-weight: bold;
    }
    .status-offline {
        color: #f44336;
        font-weight: bold;
    }

    /* File indexed confirmation */
    .file-indexed {
        color: #4caf50;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session State Initialization
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "last_upload_agent" not in st.session_state:
    st.session_state.last_upload_agent = None

if "llm_status" not in st.session_state:
    # Check LLM status on first load
    st.session_state.llm_status = check_llm_status()

# ---- Session-startup cleanup ----
# Wipe stale ChromaDB data from previous runs so each
# Streamlit session starts with a clean vector store.
if "session_initialized" not in st.session_state:
    vector_store.clear_all()
    conversation_memory.clear()
    st.session_state.session_initialized = True


# ============================================================
# Helper: Map file extension to the correct agent
# ============================================================
AGENT_MAP = {
    ".pdf": ("DocumentAgent", document_agent),
    ".txt": ("DocumentAgent", document_agent),
    ".docx": ("DocumentAgent", document_agent),
    ".xlsx": ("ExcelAgent", excel_agent),
    ".csv": ("ExcelAgent", excel_agent),
    ".png": ("ImageAgent", image_agent),
    ".jpg": ("ImageAgent", image_agent),
    ".jpeg": ("ImageAgent", image_agent),
}


def index_uploaded_file(uploaded_file) -> str:
    """
    Save an uploaded file with its ORIGINAL name to a temp
    directory and index it with the appropriate agent.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Status message string.
    """
    ext = os.path.splitext(uploaded_file.name)[1].lower()

    if ext not in AGENT_MAP:
        return f"❌ Unsupported file type: {ext}"

    agent_name, agent = AGENT_MAP[ext]

    # Save with the ORIGINAL filename so metadata is readable
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        result = agent.index(tmp_path)
        return result
    except Exception as e:
        return f"❌ Error indexing {uploaded_file.name}: {e}"
    finally:
        # Clean up the temp file
        try:
            os.unlink(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("🧠 MultiModal RAG")
    st.caption("Multi-Agent Retrieval-Augmented Generation")
    st.divider()

    # ---- LLM Status Indicators ----
    st.subheader("🔌 LLM Status")
    status = st.session_state.llm_status

    # Show network status
    is_online = status.get("is_online", True)
    if is_online:
        st.markdown(
            '🌐 <span style="color: #4caf50; font-weight: bold;">'
            'Online Mode</span> — using cloud LLMs',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '📡 <span style="color: #ff9800; font-weight: bold;">'
            'Offline Mode</span> — using local Ollama',
            unsafe_allow_html=True,
        )

    groq_status = status.get("groq", "unavailable")
    gemini_status = status.get("gemini", "unavailable")
    ollama_status = status.get("ollama", "unavailable")

    col1, col2, col3 = st.columns(3)
    with col1:
        if groq_status == "available":
            st.markdown(
                '<span class="status-online">Groq: ✓</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-offline">Groq: ✗</span>',
                unsafe_allow_html=True,
            )
    with col2:
        if gemini_status == "available":
            st.markdown(
                '<span class="status-online">Gemini: ✓</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-offline">Gemini: ✗</span>',
                unsafe_allow_html=True,
            )
    with col3:
        if ollama_status == "available":
            st.markdown(
                '<span class="status-online">Ollama: ✓</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-offline">Ollama: ✗</span>',
                unsafe_allow_html=True,
            )

    if st.button("🔄 Refresh LLM Status"):
        st.session_state.llm_status = check_llm_status()
        st.rerun()

    st.divider()

    # ---- File Uploader ----
    st.subheader("📂 Upload Files")
    uploaded_files = st.file_uploader(
        "Drag & drop files to index",
        type=["pdf", "txt", "docx", "xlsx", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        for uf in uploaded_files:
            # Skip files that have already been indexed
            if uf.name not in st.session_state.indexed_files:
                with st.spinner(f"Indexing {uf.name}..."):
                    result = index_uploaded_file(uf)
                    st.session_state.indexed_files.append(uf.name)
                    # Track which agent handles the latest upload
                    ext = os.path.splitext(uf.name)[1].lower()
                    agent_name, _ = AGENT_MAP.get(ext, (None, None))
                    if agent_name:
                        st.session_state.last_upload_agent = agent_name
                    st.markdown(
                        f'<span class="file-indexed">✓ {uf.name} indexed</span>',
                        unsafe_allow_html=True,
                    )

    # ---- Auto-cleanup: remove stale file data from ChromaDB ----
    # When files are removed from the uploader, delete their chunks
    current_upload_names = set()
    if uploaded_files:
        current_upload_names = {uf.name for uf in uploaded_files}

    stale_files = [
        f for f in st.session_state.indexed_files
        if f not in current_upload_names
    ]
    if stale_files:
        for fname in stale_files:
            vector_store.delete_by_source(fname)
            st.session_state.indexed_files.remove(fname)
        # Update last_upload_agent to reflect current state
        if st.session_state.indexed_files:
            last_file = st.session_state.indexed_files[-1]
            ext = os.path.splitext(last_file)[1].lower()
            agent_name, _ = AGENT_MAP.get(ext, (None, None))
            st.session_state.last_upload_agent = agent_name
        else:
            st.session_state.last_upload_agent = None

    # Show currently indexed files
    if st.session_state.indexed_files:
        st.divider()
        st.subheader("📋 Indexed Files")
        for fname in st.session_state.indexed_files:
            st.markdown(
                f'<span class="file-indexed">✓ {fname}</span>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ---- Clear Memory Button ----
    if st.button("🗑️ Clear Conversation Memory"):
        conversation_memory.clear()
        vector_store.clear_all()  # Wipe all ChromaDB collections
        st.session_state.chat_history = []
        st.session_state.indexed_files = []
        st.session_state.last_upload_agent = None  # Reset upload tracking
        st.success("Conversation memory and all indexed files cleared.")
        st.rerun()


# ============================================================
# Main Chat Area
# ============================================================
st.title("💬 Chat")

# ---- Display chat history ----
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-message">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="assistant-message">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        # Show metadata labels (agents used and LLM source)
        meta_parts = []
        if msg.get("agents_used"):
            meta_parts.append(
                f"Agents used: {', '.join(msg['agents_used'])}"
            )
        if msg.get("llm_source"):
            meta_parts.append(f"[via {msg['llm_source']}]")
        if meta_parts:
            st.markdown(
                f'<div class="meta-label">{" · ".join(meta_parts)}</div>',
                unsafe_allow_html=True,
            )

# ---- Chat input ----
user_input = st.chat_input("Ask a question about your uploaded files...")

if user_input:
    # Add user message to history
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )
    conversation_memory.add_message("user", user_input)

    # Display user message immediately
    st.markdown(
        f'<div class="user-message">{user_input}</div>',
        unsafe_allow_html=True,
    )

    try:
        # Step 1: Coordinator routes query to agents and retrieves context
        with st.spinner("Retrieving context..."):
            coord_result = coordinator.query(
                user_input,
                last_upload_agent=st.session_state.last_upload_agent,
            )
            context_chunks = coord_result.get("results", [])
            agents_used = coord_result.get("agents_used", [])
            history = conversation_memory.get_history()

        # Step 2: Get streaming response from aggregator
        stream, llm_source = aggregator.generate_answer_stream(
            query=user_input,
            context_chunks=context_chunks,
            history=history,
        )

        # Step 3: Stream tokens into the chat bubble in real-time
        response_placeholder = st.empty()
        meta_placeholder = st.empty()
        full_response = ""
        first_token = True

        # Show a spinner while waiting for the first token
        spinner_placeholder = st.empty()
        spinner_placeholder.markdown(
            '<div class="assistant-message" style="display:flex;align-items:center;gap:10px;">'
            '<div class="thinking-spinner"></div> Thinking...</div>'
            '<style>'
            '@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }'
            '.thinking-spinner { width: 16px; height: 16px; border: 2.5px solid #444;'
            ' border-top: 2.5px solid #1a73e8; border-radius: 50%;'
            ' animation: spin 0.8s linear infinite; display: inline-block; }'
            '</style>',
            unsafe_allow_html=True,
        )

        for token in stream:
            if first_token:
                spinner_placeholder.empty()  # Remove spinner on first token
                first_token = False
            full_response += token
            # Show response with a blinking cursor while streaming
            response_placeholder.markdown(
                f'<div class="assistant-message">{full_response}▌</div>',
                unsafe_allow_html=True,
            )

        # Edge case: no tokens at all
        if first_token:
            spinner_placeholder.empty()

        # Final render without cursor
        response_placeholder.markdown(
            f'<div class="assistant-message">{full_response}</div>',
            unsafe_allow_html=True,
        )

        # Show metadata labels
        meta_parts = []
        if agents_used:
            meta_parts.append(
                f"Agents used: {', '.join(agents_used)}"
            )
        meta_parts.append(f"[via {llm_source}]")
        meta_placeholder.markdown(
            f'<div class="meta-label">{" · ".join(meta_parts)}</div>',
            unsafe_allow_html=True,
        )

        # Store in chat history for persistence across reruns
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": full_response,
                "agents_used": agents_used,
                "llm_source": llm_source,
            }
        )
        conversation_memory.add_message("assistant", full_response)

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        st.markdown(
            f'<div class="assistant-message">{error_msg}</div>',
            unsafe_allow_html=True,
        )
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": error_msg,
                "agents_used": [],
                "llm_source": "Error",
            }
        )

