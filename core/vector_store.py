# ============================================================
# core/vector_store.py — ChromaDB Vector Store Wrapper
# ============================================================
# Manages ChromaDB collections for documents, excel, and
import os
import logging

# Hard-disable telemetry via env var (fallback)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress the broken ChromaDB posthog logger which ignores configs
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings
from core.embeddings import embedder
from config import (
    CHROMA_PERSIST_DIR,
    TOP_K_RESULTS,
    COLLECTION_DOCUMENTS,
    COLLECTION_EXCEL,
    COLLECTION_IMAGES,
)


class VectorStore:
    """
    Wrapper around ChromaDB for persistent vector storage.

    Uses local file-based persistence so embeddings survive
    between application restarts. Maintains three separate
    collections: 'documents', 'excel', and 'images'.
    """

    def __init__(self):
        """Initialize the ChromaDB client with local persistence."""
        print(f"[VectorStore] Initializing ChromaDB at: {CHROMA_PERSIST_DIR}")
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

    def _get_collection(self, collection_name: str):
        """
        Get or create a ChromaDB collection by name.

        Args:
            collection_name: Name of the collection (e.g. 'documents').

        Returns:
            A ChromaDB Collection object.
        """
        return self.client.get_or_create_collection(name=collection_name)

    def add_documents(
        self,
        collection_name: str,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ):
        """
        Add text chunks with embeddings to a collection.

        Args:
            collection_name: Target collection name.
            texts:           List of text chunks to store.
            metadatas:       List of metadata dicts (one per chunk).
            ids:             List of unique IDs (one per chunk).
        """
        try:
            collection = self._get_collection(collection_name)

            # Generate embeddings for all text chunks
            embeddings = embedder.encode(texts)

            # Upsert into ChromaDB (insert or update if ID exists)
            collection.upsert(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            print(
                f"[VectorStore] Added {len(texts)} chunks to "
                f"'{collection_name}' collection."
            )
        except Exception as e:
            print(f"[VectorStore] Error adding documents: {e}")
            raise

    def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = TOP_K_RESULTS,
    ) -> list[dict]:
        """
        Perform semantic search in a collection.

        Args:
            collection_name: Collection to search in.
            query_text:      The user's query string.
            top_k:           Number of top results to return.

        Returns:
            List of dicts, each with keys 'text' and 'metadata'.
        """
        try:
            collection = self._get_collection(collection_name)

            # Check if collection has any documents
            if collection.count() == 0:
                return []

            # Embed the query
            query_embedding = embedder.encode([query_text])[0]

            # Query ChromaDB for nearest neighbors
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
            )

            # Format results into a clean list of dicts
            formatted = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted.append(
                        {
                            "text": doc,
                            "metadata": (
                                results["metadatas"][0][i]
                                if results["metadatas"]
                                else {}
                            ),
                        }
                    )
            return formatted

        except Exception as e:
            print(f"[VectorStore] Error during search: {e}")
            return []

    def search_with_scores(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = TOP_K_RESULTS,
    ) -> list[dict]:
        """
        Perform semantic search and return results WITH distance scores.

        Same as search(), but each result dict also includes a 'distance'
        key with the ChromaDB L2 distance (lower = more similar).

        Args:
            collection_name: Collection to search in.
            query_text:      The user's query string.
            top_k:           Number of top results to return.

        Returns:
            List of dicts with keys 'text', 'metadata', and 'distance'.
        """
        try:
            collection = self._get_collection(collection_name)

            if collection.count() == 0:
                return []

            query_embedding = embedder.encode([query_text])[0]

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            formatted = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = (
                        results["distances"][0][i]
                        if results.get("distances")
                        else None
                    )
                    formatted.append(
                        {
                            "text": doc,
                            "metadata": (
                                results["metadatas"][0][i]
                                if results["metadatas"]
                                else {}
                            ),
                            "distance": distance,
                        }
                    )
            return formatted

        except Exception as e:
            print(f"[VectorStore] Error during scored search: {e}")
            return []

    def clear_all(self):
        """
        Delete all collections and recreate them empty.

        This ensures old document/excel/image data doesn't persist
        across user sessions after the user clears their workspace.
        """
        for name in [COLLECTION_DOCUMENTS, COLLECTION_EXCEL, COLLECTION_IMAGES]:
            try:
                self.client.delete_collection(name=name)
                print(f"[VectorStore] Deleted collection '{name}'.")
            except Exception:
                pass  # Collection might not exist yet
        print("[VectorStore] All collections cleared.")

    def delete_by_source(self, source_name: str):
        """
        Delete all chunks with a given source filename from ALL collections.

        Args:
            source_name: The filename to match in metadata 'source' field.
        """
        for col_name in [COLLECTION_DOCUMENTS, COLLECTION_EXCEL, COLLECTION_IMAGES]:
            try:
                collection = self._get_collection(col_name)
                # ChromaDB supports metadata-based deletion
                results = collection.get(where={"source": source_name})
                if results and results["ids"]:
                    collection.delete(ids=results["ids"])
                    print(
                        f"[VectorStore] Removed {len(results['ids'])} chunks "
                        f"for '{source_name}' from '{col_name}'."
                    )
            except Exception:
                pass  # Collection might not exist or no matching docs


# ---- Module-level convenience instance ----
vector_store = VectorStore()
