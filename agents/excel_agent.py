# ============================================================
# agents/excel_agent.py — Excel/CSV Processing Agent
# ============================================================
# Handles .xlsx and .csv files:
#   - Reads spreadsheets using pandas
#   - Converts each row to a natural-language string
#   - Embeds and stores rows in ChromaDB 'excel' collection
#   - Provides semantic search over indexed tabular data
# ============================================================

import os
import pandas as pd
from config import COLLECTION_EXCEL
from core.vector_store import vector_store


class ExcelAgent:
    """Agent responsible for indexing and querying Excel/CSV files."""

    def __init__(self):
        self.name = "ExcelAgent"
        self.collection = COLLECTION_EXCEL

    # ---- Indexing ----

    def index(self, file_path: str) -> str:
        """
        Index an Excel or CSV file into the vector store.

        Each row is converted to a natural-language sentence:
        'Row N: [column] is [value], [column] is [value], ...'

        Args:
            file_path: Absolute path to the file.

        Returns:
            A status message string.
        """
        try:
            ext = os.path.splitext(file_path)[1].lower()
            filename = os.path.basename(file_path)

            # Read the file into a DataFrame
            if ext == ".xlsx":
                df = pd.read_excel(file_path, engine="openpyxl")
            elif ext == ".csv":
                df = pd.read_csv(file_path)
            else:
                return f"[{self.name}] Unsupported file type: {ext}"

            if df.empty:
                return f"[{self.name}] No data found in {filename}"

            # Convert each row to a readable text string
            texts = []
            metadatas = []
            ids = []

            for idx, row in df.iterrows():
                # Build "Row N: col1 is val1, col2 is val2, ..."
                parts = [
                    f"{col} is {val}"
                    for col, val in row.items()
                    if pd.notna(val)
                ]
                row_text = f"Row {idx + 1}: {', '.join(parts)}"
                texts.append(row_text)
                metadatas.append(
                    {
                        "source": filename,
                        "row_index": int(idx),
                        "type": ext,
                    }
                )
                ids.append(f"excel_{filename}_{idx}")

            # Store in ChromaDB
            vector_store.add_documents(
                collection_name=self.collection,
                texts=texts,
                metadatas=metadatas,
                ids=ids,
            )

            return (
                f"[{self.name}] Indexed '{filename}': "
                f"{len(texts)} rows stored."
            )

        except Exception as e:
            return f"[{self.name}] Error indexing {file_path}: {e}"

    # ---- Querying ----

    def query(self, query_text: str) -> list[dict]:
        """
        Search the excel collection for relevant rows.

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


# ---- Module-level convenience instance ----
excel_agent = ExcelAgent()
