# core/memory.py — Conversation Memory Manager
# Wraps LangChain ConversationBufferMemory to store the last N messages of the convo

from langchain.memory import ConversationBufferMemory
from config import MEMORY_MAX_MESSAGES


class ConversationMemory:
    """
    Manages conversation history for the RAG system.

    Stores alternating human/AI messages and trims to
    MEMORY_MAX_MESSAGES to keep context window manageable.
    """

    def __init__(self):
        """Initialize LangChain memory with human/AI prefixes."""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            human_prefix="User",
            ai_prefix="Assistant",
            return_messages=True,
        )
        self.max_messages = MEMORY_MAX_MESSAGES

    def add_message(self, role: str, content: str):
        """
        Add a message to conversation history.

        Args:
            role:    Either 'user' or 'assistant'.
            content: The message text.
        """
        if role == "user":
            self.memory.chat_memory.add_user_message(content)
        elif role == "assistant":
            self.memory.chat_memory.add_ai_message(content)

        # Trim to keep only the last N messages
        self._trim_history()

    def _trim_history(self):
        """Remove oldest messages if history exceeds the maximum."""
        messages = self.memory.chat_memory.messages
        if len(messages) > self.max_messages:
            # Keep only the most recent messages
            self.memory.chat_memory.messages = messages[-self.max_messages :]

    def get_history(self) -> str:
        """
        Get the full conversation history as a formatted string.

        Returns:
            A string with each message on its own line,
            prefixed by 'User:' or 'Assistant:'.
        """
        messages = self.memory.chat_memory.messages
        if not messages:
            return "No previous conversation."

        history_lines = []
        for msg in messages:
            # LangChain message types: HumanMessage, AIMessage
            role = "User" if msg.type == "human" else "Assistant"
            history_lines.append(f"{role}: {msg.content}")

        return "\n".join(history_lines)

    def clear(self):
        """Clear all conversation history."""
        self.memory.clear()
        print("[Memory] Conversation history cleared.")


# ---- Module-level convenience instance ----
conversation_memory = ConversationMemory()
