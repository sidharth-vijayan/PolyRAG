# ============================================================
# agents/image_agent.py — Image Processing Agent (Vision LLM + OCR)
# ============================================================
# Handles .png, .jpg, .jpeg image files:
#   1. Sends the image to a vision-capable LLM to get a rich
#      description (Groq vision → Gemini vision fallback)
#   2. Also runs Tesseract OCR to extract any printed text
#   3. Combines both into a single text for ChromaDB indexing
#   4. Provides semantic search over indexed image descriptions
# ============================================================

import os
import base64
import requests
import pytesseract
from PIL import Image
import google.generativeai as genai
from config import (
    TESSERACT_PATH,
    COLLECTION_IMAGES,
    GROQ_API_KEY,
    GROQ_VISION_MODEL,
    GROQ_BASE_URL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)
from core.vector_store import vector_store

# ---- Configure Tesseract path for Windows ----
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---- Configure Gemini for vision fallback ----
genai.configure(api_key=GEMINI_API_KEY)

# ---- MIME type mapping ----
MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class ImageAgent:
    """Agent responsible for indexing and querying images via Vision LLM + OCR."""

    def __init__(self):
        self.name = "ImageAgent"
        self.collection = COLLECTION_IMAGES

    # ---- Indexing ----

    def index(self, file_path: str) -> str:
        """
        Index an image file by describing it with a vision LLM
        and extracting any text via OCR.

        Args:
            file_path: Absolute path to the image file.

        Returns:
            A status message string.
        """
        try:
            ext = os.path.splitext(file_path)[1].lower()
            filename = os.path.basename(file_path)

            if ext not in (".png", ".jpg", ".jpeg"):
                return f"[{self.name}] Unsupported file type: {ext}"

            # ---- Step 1: Get vision LLM description ----
            vision_desc = self._describe_with_vision_llm(file_path, ext)

            # ---- Step 2: Get OCR text (printed text extraction) ----
            ocr_text = self._extract_ocr_text(file_path)

            # ---- Step 3: Combine both into final text ----
            parts = []
            if vision_desc:
                parts.append(f"Image Description: {vision_desc}")
            if ocr_text:
                parts.append(f"OCR Text: {ocr_text}")

            if not parts:
                # Absolute fallback — nothing worked
                combined_text = (
                    f"Image: {filename} — no description or text detected"
                )
            else:
                combined_text = f"[Image: {filename}]\n" + "\n".join(parts)

            # ---- Step 4: Store in ChromaDB ----
            texts = [combined_text]
            metadatas = [
                {
                    "source": filename,
                    "type": ext,
                    "has_vision_desc": bool(vision_desc),
                    "has_ocr_text": bool(ocr_text),
                }
            ]
            ids = [f"img_{filename}"]

            vector_store.add_documents(
                collection_name=self.collection,
                texts=texts,
                metadatas=metadatas,
                ids=ids,
            )

            method = []
            if vision_desc:
                method.append("Vision LLM")
            if ocr_text:
                method.append("OCR")
            method_str = " + ".join(method) if method else "fallback"

            return (
                f"[{self.name}] Indexed '{filename}' via {method_str}."
            )

        except Exception as e:
            return f"[{self.name}] Error indexing {file_path}: {e}"

    # ---- Querying ----

    def query(self, query_text: str) -> list[dict]:
        """
        Search the images collection for relevant descriptions.

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

    def _describe_with_vision_llm(self, file_path: str, ext: str) -> str:
        """
        Send the image to a vision-capable LLM for description.
        Tries: Groq vision → Gemini vision → Ollama text fallback.

        When offline, skips cloud APIs entirely and uses Ollama with
        OCR text to generate a description.

        Args:
            file_path: Path to the image file.
            ext:       File extension (e.g. '.png').

        Returns:
            A text description of the image, or empty string if all fail.
        """
        from core.llm_router import _is_online

        prompt_text = (
            "Describe this image in detail. Include all visible text, "
            "objects, people, diagrams, charts, tables, colors, layout, "
            "and any other relevant information. Be thorough."
        )

        online = _is_online()

        # ---- Try Groq Vision (skip if offline) ----
        if online:
            # Read and base64-encode the image
            with open(file_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
            mime_type = MIME_TYPES.get(ext, "image/png")

            try:
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": GROQ_VISION_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_b64}"
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 1024,
                }
                resp = requests.post(
                    GROQ_BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                answer = resp.json()["choices"][0]["message"]["content"]
                print(f"[{self.name}] Image described via Groq Vision.")
                return answer

            except Exception as e:
                print(f"[{self.name}] Groq vision failed: {e}")

            # ---- Try Gemini Vision ----
            try:
                model = genai.GenerativeModel(GEMINI_MODEL)
                img = Image.open(file_path)
                response = model.generate_content([prompt_text, img])
                print(f"[{self.name}] Image described via Gemini Vision.")
                return response.text

            except Exception as e:
                print(f"[{self.name}] Gemini vision failed: {e}")

        else:
            print(
                f"[{self.name}] Offline — skipping cloud vision APIs."
            )

        # ---- Ollama text fallback (local, works offline) ----
        # Use OCR text (if available) and ask Ollama to summarize
        try:
            ocr_text = self._extract_ocr_text(file_path)
            if ocr_text:
                ollama_prompt = (
                    "I have an image that contains the following text "
                    "(extracted via OCR). Based on this text, provide a "
                    "detailed description of what this image likely "
                    "contains and what information it conveys.\n\n"
                    f"OCR Text:\n{ocr_text}\n\n"
                    "Description:"
                )
            else:
                filename = os.path.basename(file_path)
                ollama_prompt = (
                    f"I have an image file named '{filename}' "
                    f"(type: {ext}). I cannot see the image, but "
                    "based on the filename, provide a brief placeholder "
                    "description noting that this image was indexed "
                    "offline without vision analysis."
                )

            from config import OLLAMA_BASE_URL, OLLAMA_MODEL
            ollama_url = f"{OLLAMA_BASE_URL}/api/generate"
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": ollama_prompt,
                "stream": False,
            }
            resp = requests.post(ollama_url, json=payload, timeout=300)
            resp.raise_for_status()
            result = resp.json()
            print(f"[{self.name}] Image described via Ollama (text fallback).")
            return result["response"]

        except Exception as e:
            print(f"[{self.name}] Ollama text fallback failed: {e}")

        # All vision methods failed
        print(f"[{self.name}] No vision LLM available, using OCR only.")
        return ""

    def _extract_ocr_text(self, file_path: str) -> str:
        """
        Extract printed text from an image using Tesseract OCR.

        Returns:
            Extracted text string, or empty string if none found.
        """
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img).strip()
            return text
        except Exception as e:
            print(f"[{self.name}] OCR failed: {e}")
            return ""


# ---- Module-level convenience instance ----
image_agent = ImageAgent()
