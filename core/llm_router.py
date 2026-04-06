# ============================================================
# core/llm_router.py — LLM Router: Groq → Gemini → Ollama
# ============================================================
# This is the SINGLE entry point for all LLM calls in the
# entire application. No other module should call any LLM
# directly.
#
# Priority chain (when online):
#   1. Groq (llama-3.3-70b-versatile) — fast cloud inference
#   2. Google Gemini 2.0 Flash        — free-tier cloud
#   3. Ollama (gpt-oss:20b)           — local offline fallback
#   4. If ALL three fail              → raise RuntimeError
#
# When offline (no internet detected):
#   → Skips Groq & Gemini, goes directly to Ollama
# ============================================================

import json
import socket
import requests
import google.generativeai as genai
from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_BASE_URL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

# ---- Configure the Gemini SDK with the API key ----
genai.configure(api_key=GEMINI_API_KEY)


def _is_online(timeout: float = 2.0) -> bool:
    """
    Quick check for internet connectivity.

    Tries to open a TCP connection to a reliable DNS server.
    Returns True if the connection succeeds within `timeout` seconds.
    """
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout).close()
        return True
    except OSError:
        return False


def generate(prompt: str) -> tuple[str, str]:
    """
    Generate a response from an LLM.

    Tries Groq first, then Gemini, then Ollama as final fallback.

    Args:
        prompt: The full prompt string to send to the LLM.

    Returns:
        A tuple of (response_text, source) where source is
        'Groq', 'Gemini', or 'Ollama fallback'.

    Raises:
        RuntimeError: If all three LLMs are unavailable.
    """

    # ---- Quick connectivity check ----
    online = _is_online()
    if not online:
        print(
            "[LLM Router] No internet detected. "
            "Skipping cloud LLMs, going straight to Ollama."
        )

    # ---- Step 1: Try Groq (Primary) — skip if offline ----
    if online:
      try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        resp = requests.post(
            GROQ_BASE_URL, headers=headers, json=payload, timeout=15
        )
        resp.raise_for_status()
        result = resp.json()
        answer = result["choices"][0]["message"]["content"]
        return answer, "Groq"

      except Exception as groq_error:
        print(
            f"[LLM Router] Groq failed: {groq_error}. "
            f"Trying Gemini next..."
        )

    # ---- Step 2: Try Gemini (Secondary) — skip if offline ----
    if online:
      try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text, "Gemini"

      except Exception as gemini_error:
        print(
            f"[LLM Router] Gemini failed: {gemini_error}. "
            f"Switching to Ollama fallback ({OLLAMA_MODEL})..."
        )

    # ---- Step 3: Try Ollama (Local, streaming) ----
    try:
        ollama_url = f"{OLLAMA_BASE_URL}/api/generate"
        print(f"[LLM Router] Calling Ollama at {ollama_url} "
              f"with model '{OLLAMA_MODEL}'...")
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
        }
        # timeout=(connect, read_between_bytes)
        # 10s to connect, 120s max between streamed chunks
        resp = requests.post(
            ollama_url, json=payload, timeout=(10, 120), stream=True
        )
        resp.raise_for_status()

        # Accumulate streamed tokens
        full_response = ""
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                full_response += chunk.get("response", "")
                if chunk.get("done", False):
                    break

        source = "Ollama (offline)" if not online else "Ollama fallback"
        print(f"[LLM Router] Ollama responded ({len(full_response)} chars).")
        return full_response, source

    except Exception as ollama_error:
        print(f"[LLM Router] Ollama also failed: "
              f"{type(ollama_error).__name__}: {ollama_error}")
        raise RuntimeError(
            f"All LLMs unavailable. Ollama error: {ollama_error}"
        )

def generate_stream(prompt: str):
    """
    Streaming version of generate().

    Tries Groq → Gemini → Ollama, same as generate(), but returns
    a generator that yields string tokens. Cloud providers yield
    their full response as a single chunk; Ollama yields token by
    token for real-time display.

    Returns:
        A tuple of (token_generator, source) where token_generator
        yields string chunks and source is the LLM provider name.

    Raises:
        RuntimeError: If all three LLMs are unavailable.
    """
    online = _is_online()
    if not online:
        print(
            "[LLM Router] No internet detected. "
            "Skipping cloud LLMs, going straight to Ollama."
        )

    # ---- Step 1: Try Groq — skip if offline ----
    if online:
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            }
            resp = requests.post(
                GROQ_BASE_URL, headers=headers, json=payload, timeout=15
            )
            resp.raise_for_status()
            result = resp.json()
            answer = result["choices"][0]["message"]["content"]

            def _groq_gen():
                yield answer

            return _groq_gen(), "Groq"

        except Exception as e:
            print(f"[LLM Router] Groq failed: {e}. Trying Gemini...")

    # ---- Step 2: Try Gemini — skip if offline ----
    if online:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)

            def _gemini_gen():
                yield response.text

            return _gemini_gen(), "Gemini"

        except Exception as e:
            print(
                f"[LLM Router] Gemini failed: {e}. "
                f"Switching to Ollama ({OLLAMA_MODEL})..."
            )

    # ---- Step 3: Ollama — stream token by token ----
    ollama_url = f"{OLLAMA_BASE_URL}/api/generate"
    print(
        f"[LLM Router] Streaming from Ollama at {ollama_url} "
        f"with model '{OLLAMA_MODEL}'..."
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
    }

    try:
        resp = requests.post(
            ollama_url, json=payload, timeout=(10, 120), stream=True
        )
        resp.raise_for_status()
    except Exception as e:
        print(
            f"[LLM Router] Ollama connection failed: "
            f"{type(e).__name__}: {e}"
        )
        raise RuntimeError(f"All LLMs unavailable. Ollama error: {e}")

    source = "Ollama (offline)" if not online else "Ollama fallback"

    def _ollama_stream():
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done", False):
                    break
        print("[LLM Router] Ollama stream complete.")

    return _ollama_stream(), source



def check_llm_status() -> dict:
    """
    Check the availability of all three LLM services.

    Returns:
        A dict like:
        {
            "is_online": True or False,
            "groq": "available" or "unavailable",
            "gemini": "available" or "unavailable",
            "ollama": "available" or "unavailable"
        }
    """
    online = _is_online()

    status = {
        "is_online": online,
        "groq": "unavailable",
        "gemini": "unavailable",
        "ollama": "unavailable",
    }

    # ---- Skip cloud checks if offline ----
    if online:
        # ---- Check Groq (fast: list models, 3s timeout) ----
        try:
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            resp = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers=headers,
                timeout=3,
            )
            if resp.status_code == 200:
                status["groq"] = "available"
        except Exception:
            pass  # Offline — silently mark unavailable

        # ---- Check Gemini (fast: list models, 3s timeout) ----
        try:
            resp = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models"
                f"?key={GEMINI_API_KEY}",
                timeout=3,
            )
            if resp.status_code == 200:
                status["gemini"] = "available"
        except Exception:
            pass

    # ---- Check Ollama (always — it's local) ----
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            status["ollama"] = "available"
    except Exception:
        pass

    return status
