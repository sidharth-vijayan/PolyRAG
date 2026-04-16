# ============================================================
# core/embeddings.py — Embedding Model Wrapper
# ============================================================
# Provides a singleton wrapper around the SentenceTransformer
# model (all-MiniLM-L6-v2) so the model is loaded into memory
# only once across the entire application lifetime.
# ============================================================

# Force Hugging Face to use cached models only (no network check)
import os
import warnings

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ignore the PyTorch/SentenceTransformers warning on Windows
warnings.filterwarnings(
    "ignore", 
    message=".*Examining the path of torch.classes raised.*"
)

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class EmbeddingModel:
    """
    Singleton wrapper for SentenceTransformer.

    Ensures the embedding model is loaded only once,
    regardless of how many times encode() is called
    from different agents or modules.
    """

    _instance = None   # Class-level singleton reference
    _model = None      # The loaded SentenceTransformer model

    def __new__(cls):
        """Create a new instance only if one doesn't already exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print(f"[Embeddings] Loading model: {EMBEDDING_MODEL} ...")
            cls._model = SentenceTransformer(EMBEDDING_MODEL)
            print("[Embeddings] Model loaded successfully.")
        return cls._instance

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of text strings into embedding vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each vector is a list of floats).
        """
        # SentenceTransformer returns numpy arrays; convert to plain lists
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()


# ---- Module-level convenience instance ----
# Other modules can do: from core.embeddings import embedder
embedder = EmbeddingModel()
