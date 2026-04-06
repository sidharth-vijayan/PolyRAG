# ============================================================
# agents/coordinator.py — Multi-Agent Coordinator
# ============================================================
# Routes user queries to the appropriate agent(s) based on:
#   1. Explicit keywords in the query (excel, image, document)
#   2. The most recently uploaded file type (when no keywords)
#   3. All agents as a final fallback
#
# This module does NOT call any LLM — it only coordinates
# which agents to query and merges their retrieval results.
# ============================================================

from agents.document_agent import document_agent
from agents.excel_agent import excel_agent
from agents.image_agent import image_agent


# ---- Keywords for explicit routing ----
DOCUMENT_KEYWORDS = [
    "document", "doc", "pdf", "text", "txt",
    "report", "paper", "article", "paragraph",
    "chapter", "word", "docx",
]

EXCEL_KEYWORDS = [
    "excel", "xlsx", "csv", "spreadsheet", "table",
    "column", "row", "cell", "sheet", "tabular",
]

IMAGE_KEYWORDS = [
    "image", "picture", "photo", "ocr", "screenshot",
    "scan", "jpg", "png", "jpeg", "visual",
]

# ---- Map file extensions to agent names ----
EXT_TO_AGENT = {
    ".pdf": "DocumentAgent",
    ".txt": "DocumentAgent",
    ".docx": "DocumentAgent",
    ".xlsx": "ExcelAgent",
    ".csv": "ExcelAgent",
    ".png": "ImageAgent",
    ".jpg": "ImageAgent",
    ".jpeg": "ImageAgent",
}


class CoordinatorAgent:
    """
    Orchestrates queries across all specialized agents.

    Routing priority:
      1. Explicit keywords in the query → specific agent
      2. No keywords → most recently uploaded file's agent
      3. No uploads at all → all agents
    """

    def __init__(self):
        self.name = "CoordinatorAgent"
        # Registry of all available agents
        self.agents = {
            "DocumentAgent": document_agent,
            "ExcelAgent": excel_agent,
            "ImageAgent": image_agent,
        }

    def query(self, query_text: str, last_upload_agent: str = None) -> dict:
        """
        Route a query to the appropriate agent(s) and merge results.

        Args:
            query_text:        The user's query string.
            last_upload_agent: Name of the agent for the most recently
                               uploaded file (e.g. 'DocumentAgent').
                               Used as default when no keywords match.

        Returns:
            A dict with:
              - 'results': list of context chunk dicts
              - 'agents_used': list of agent name strings
        """
        query_lower = query_text.lower()

        # Determine which agents to use
        agents_to_query = self._route(query_lower, last_upload_agent)

        # Collect results from selected agents
        all_results = []
        agents_used = []

        for agent_name in agents_to_query:
            agent = self.agents[agent_name]
            try:
                results = agent.query(query_text)
                if results:
                    all_results.extend(results)
                    agents_used.append(agent_name)
            except Exception as e:
                print(
                    f"[{self.name}] Error querying {agent_name}: {e}"
                )

        # If specific agents returned nothing, try all agents as fallback
        if not all_results and len(agents_to_query) < len(self.agents):
            print(
                f"[{self.name}] No results from targeted agents. "
                f"Querying all agents..."
            )
            return self._query_all(query_text)

        return {
            "results": all_results,
            "agents_used": agents_used,
        }

    def _route(
        self, query_lower: str, last_upload_agent: str = None
    ) -> list[str]:
        """
        Determine which agents to query.

        Priority:
          1. Explicit keywords → that agent
          2. No keywords, but recent upload → that upload's agent
          3. No keywords, no upload → all agents

        Args:
            query_lower:       The query in lowercase.
            last_upload_agent: Agent name of the last uploaded file.

        Returns:
            List of agent name strings to query.
        """
        # Check for Document-related keywords
        if any(kw in query_lower for kw in DOCUMENT_KEYWORDS):
            return ["DocumentAgent"]

        # Check for Excel-related keywords
        if any(kw in query_lower for kw in EXCEL_KEYWORDS):
            return ["ExcelAgent"]

        # Check for Image-related keywords
        if any(kw in query_lower for kw in IMAGE_KEYWORDS):
            return ["ImageAgent"]

        # No keyword match → use the most recently uploaded file's agent
        if last_upload_agent and last_upload_agent in self.agents:
            print(
                f"[{self.name}] No keyword match. "
                f"Routing to last upload: {last_upload_agent}"
            )
            return [last_upload_agent]

        # Final fallback: query all agents
        return list(self.agents.keys())

    def _query_all(self, query_text: str) -> dict:
        """Query all agents and merge results."""
        all_results = []
        agents_used = []

        for agent_name, agent in self.agents.items():
            try:
                results = agent.query(query_text)
                if results:
                    all_results.extend(results)
                    agents_used.append(agent_name)
            except Exception as e:
                print(
                    f"[{self.name}] Error querying {agent_name}: {e}"
                )

        return {
            "results": all_results,
            "agents_used": agents_used,
        }


# ---- Module-level convenience instance ----
coordinator = CoordinatorAgent()
