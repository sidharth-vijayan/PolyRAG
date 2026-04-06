# ============================================================
# agents/aggregator.py — Response Aggregator (RAG Prompt Builder)
# ============================================================
# Takes the retrieved context chunks from the coordinator,
# combines them with conversation history, builds a structured
# RAG prompt, and calls the LLM via llm_router.
#
# This is the ONLY agent that interacts with an LLM — and it
# does so exclusively through the llm_router, never directly.
# ============================================================

from core.llm_router import generate as llm_generate
from core.llm_router import generate_stream as llm_generate_stream


class Aggregator:
    """
    Builds the final RAG prompt and generates an answer.

    Combines conversation history, retrieved context chunks,
    and the user's question into a structured prompt, then
    sends it to the LLM router for generation.
    """

    def __init__(self):
        self.name = "Aggregator"

    def _build_prompt(
        self,
        query: str,
        context_chunks: list[dict],
        history: str,
    ) -> str:
        """
        Build the RAG prompt from context, history, and query.

        Args:
            query:          The user's question.
            context_chunks: List of dicts with 'text' and 'metadata'.
            history:        Formatted conversation history string.

        Returns:
            The assembled prompt string.
        """
        # ---- Build the context section ----
        if context_chunks:
            context_parts = []
            for i, chunk in enumerate(context_chunks, 1):
                source = chunk.get("metadata", {}).get("source", "Unknown")
                context_parts.append(
                    f"[{i}] (Source: {source})\n{chunk['text']}"
                )
            context_text = "\n\n".join(context_parts)
        else:
            context_text = "No relevant context was found."

        # ---- Assemble the full RAG prompt ----
        prompt = (
            "You are a helpful AI assistant for a knowledge retrieval system.\n"
            "Rules:\n"
            "1. If the Retrieved Context below answers the question, use it "
            "and end your answer with: (Found from document)\n"
            "2. If the context does NOT answer the question, use your own "
            "knowledge and end your answer with: (Based on general knowledge)\n"
            "3. Do NOT list or mention source file names in your answer.\n"
            "4. Be thorough, clear, and informative.\n\n"
            f"Conversation History:\n{history}\n\n"
            f"Retrieved Context:\n{context_text}\n\n"
            f"User Question: {query}\n\n"
            "Answer:"
        )
        return prompt

    def generate_answer(
        self,
        query: str,
        context_chunks: list[dict],
        history: str,
    ) -> tuple[str, str]:
        """
        Generate a complete answer (non-streaming).

        Returns:
            A tuple of (answer_text, llm_source).
        """
        prompt = self._build_prompt(query, context_chunks, history)

        try:
            answer, llm_source = llm_generate(prompt)
            return answer, llm_source
        except RuntimeError as e:
            return str(e), "Error"

    def generate_answer_stream(
        self,
        query: str,
        context_chunks: list[dict],
        history: str,
    ):
        """
        Streaming version of generate_answer.

        Returns:
            A tuple of (token_generator, llm_source) where
            token_generator yields string chunks.
        """
        prompt = self._build_prompt(query, context_chunks, history)

        try:
            stream, llm_source = llm_generate_stream(prompt)
            return stream, llm_source
        except RuntimeError as e:
            def _error_gen():
                yield str(e)
            return _error_gen(), "Error"


# ---- Module-level convenience instance ----
aggregator = Aggregator()
