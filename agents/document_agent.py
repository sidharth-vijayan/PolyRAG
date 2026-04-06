# ============================================================
# agents/document_agent.py — Document Processing Agent
# ============================================================
# Handles PDF and TXT files:
#   - Extracts text from PDFs page-by-page using PyMuPDF
#   - Reads plain text files directly
#   - Chunks text with a sliding window
#   - Embeds and stores chunks in ChromaDB 'documents' collection
#   - Provides semantic search over indexed documents
# ============================================================

import os
import fitz  # PyMuPDF
from docx import Document as DocxDocument  # python-docx
from config import CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_DOCUMENTS
from core.vector_store import vector_store


class DocumentAgent:
    """Agent responsible for indexing and querying PDF/TXT files."""

    def __init__(self):
        self.name = "DocumentAgent"
        self.collection = COLLECTION_DOCUMENTS

    # ---- Indexing ----

    def index(self, file_path: str) -> str:
        """
        Index a PDF or TXT file into the vector store.

        Args:
            file_path: Absolute path to the file.

        Returns:
            A status message string.
        """
        try:
            ext = os.path.splitext(file_path)[1].lower()
            filename = os.path.basename(file_path)

            # Extract raw text based on file type
            if ext == ".pdf":
                text = self._extract_pdf(file_path)
            elif ext == ".txt":
                text = self._extract_txt(file_path)
            elif ext == ".docx":
                text = self._extract_docx(file_path)
            else:
                return f"[{self.name}] Unsupported file type: {ext}"

            if not text.strip():
                return f"[{self.name}] No text found in {filename}"

            # Chunk the text with sliding window
            chunks = self._chunk_text(text)

            # Prepare metadata and IDs for each chunk
            metadatas = [
                {"source": filename, "chunk_index": i, "type": ext}
                for i in range(len(chunks))
            ]
            ids = [f"doc_{filename}_{i}" for i in range(len(chunks))]

            # Store in ChromaDB
            vector_store.add_documents(
                collection_name=self.collection,
                texts=chunks,
                metadatas=metadatas,
                ids=ids,
            )

            return (
                f"[{self.name}] Indexed '{filename}': "
                f"{len(chunks)} chunks stored."
            )

        except Exception as e:
            return f"[{self.name}] Error indexing {file_path}: {e}"

    # ---- Querying ----

    def query(self, query_text: str) -> list[dict]:
        """
        Search the documents collection for relevant chunks.

        Args:
            query_text: The user's query string.

        Returns:
            List of result dicts with 'text' and 'metadata' keys.
        """
        try:
            results = vector_store.search(
                collection_name=self.collection,
                query_text=query_text,
            )
            return results
        except Exception as e:
            print(f"[{self.name}] Query error: {e}")
            return []

    # ---- Private helpers ----

    def _extract_pdf(self, file_path: str) -> str:
        """Extract all text from a PDF, page by page."""
        text_parts = []
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text:
                text_parts.append(page_text)
        doc.close()
        return "\n".join(text_parts)

    def _extract_txt(self, file_path: str) -> str:
        """Read the entire contents of a plain text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _extract_docx(self, file_path: str) -> str:
        """Extract all text from a Word .docx file, paragraph by paragraph."""
        doc = DocxDocument(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        Uses a sliding window of CHUNK_SIZE characters with
        CHUNK_OVERLAP character overlap between consecutive chunks.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():  # Only keep non-empty chunks
                chunks.append(chunk.strip())
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks


# ---- Module-level convenience instance ----
document_agent = DocumentAgent()
