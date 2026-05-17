<div align="center">

# 🧠 PolyRAG

### Multi-Agent Multimodal Retrieval-Augmented Generation System

**Query PDFs · Spreadsheets · Images — all in one intelligent pipeline**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=flat-square&logo=langchain)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B35?style=flat-square)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=flat-square)](https://groq.com/)
[![Gemini](https://img.shields.io/badge/Google-Gemini_2.0-4285F4?style=flat-square&logo=google)](https://ai.google.dev/)

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-polyrag.streamlit.app-FF4B4B?style=flat-square)](https://polyrag.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 📌 Overview

**PolyRAG** is a modular, multi-agent Retrieval-Augmented Generation (RAG) system that answers natural language queries by intelligently retrieving information from heterogeneous data sources — including PDF documents, Word files, Excel spreadsheets, CSVs, and images.

At its core, a **Coordinator Agent** receives the user query, determines which data modality is most relevant, and routes the query to the appropriate **specialized sub-agent**. Retrieved context is then passed to an **Aggregator** that synthesizes a grounded, context-aware response using either Groq (LLaMA) or Google Gemini.

The system includes a full **evaluation framework** with retrieval, generation, and system-level metrics — giving it the rigor of a research-grade implementation.

> 🔗 **Live Demo:** [polyrag.streamlit.app](https://polyrag.streamlit.app/)

---

## ✨ Features

### 🤖 Multi-Agent Architecture
- **Coordinator Agent** — classifies the query and routes it to the right sub-agent
- **Document Agent** — handles PDFs, `.txt`, and `.docx` files using PyMuPDF and python-docx
- **Excel Agent** — processes `.xlsx` and `.csv` files with pandas and openpyxl
- **Image Agent** — extracts text from images via Tesseract OCR and handles visual queries using Groq's vision model (LLaMA 4 Scout)
- **Aggregator** — synthesizes context from one or multiple agents into a final answer

### 🔍 Advanced Retrieval Pipeline
- Semantic chunking with configurable `CHUNK_SIZE` and `CHUNK_OVERLAP`
- Dense embeddings via `all-MiniLM-L6-v2` (Sentence Transformers)
- Vector storage and similarity search with **ChromaDB** — separate collections per modality
- Top-K context retrieval with distance scoring

### 🧠 Multi-LLM Support
- **Groq (LLaMA 3.3 70B)** — default fast inference for text queries
- **Groq Vision (LLaMA 4 Scout)** — multimodal understanding for image inputs
- **Google Gemini 2.0 Flash** — fallback and alternative LLM
- **Ollama** — local model support for offline/private deployments

### 💾 Conversation Memory
- Persistent conversational context across turns (up to 10 messages)
- Enables follow-up questions and coherent multi-turn dialogue

### 🧪 Evaluation Framework
Built-in benchmarking with `evaluate.py` — runs a test bench of Q&A pairs through the full pipeline and reports:

| Metric | Description |
|---|---|
| **Context Precision** | Fraction of retrieved chunks from the expected source |
| **Answer Similarity** | Cosine similarity between generated and expected answer embeddings |
| **Faithfulness** | Whether the answer is grounded in retrieved documents |
| **Routing Accuracy** | Whether the coordinator selected the correct agent |
| **Retrieval Latency** | Time to retrieve relevant chunks (ms) |
| **Generation Latency** | LLM response time (ms) |
| **Time to First Token (TTFT)** | Streaming responsiveness (ms) |
| **Tokens/sec** | Approximate generation throughput |
| **End-to-End Latency** | Total retrieval + generation time (ms) |

> Evaluated across 10 test cases (1 runs each) — **routing accuracy: 1.0 · answer similarity: 0.883 · avg E2E latency: 0.992s**
📊 **[View Full Evaluation Report →](results/eval_report.md)**
---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **LLM Orchestration** | LangChain 0.2 |
| **LLMs** | Groq (LLaMA 3.3 70B, LLaMA 4 Scout), Google Gemini 2.0 Flash, Ollama |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector Store** | ChromaDB |
| **Document Parsing** | PyMuPDF (PDF), python-docx (DOCX) |
| **Spreadsheet Parsing** | pandas, openpyxl |
| **Image / OCR** | Pillow, Tesseract OCR (pytesseract) |
| **Frontend / UI** | Streamlit |
| **Config & Secrets** | python-dotenv |

---

## 📁 Project Structure

```
PolyRAG/
├── agents/
│   ├── coordinator.py       # Query classification & agent routing
│   ├── document_agent.py    # PDF, TXT, DOCX ingestion & retrieval
│   ├── excel_agent.py       # XLSX, CSV ingestion & retrieval
│   ├── image_agent.py       # Image OCR & vision-based retrieval
│   └── aggregator.py        # Multi-source synthesis & LLM generation
├── core/
│   ├── vector_store.py      # ChromaDB client & collection management
│   ├── embeddings.py        # Sentence Transformer embedding wrapper
│   └── memory.py            # Conversational memory buffer
├── data/
│   └── eval_samples/        # Sample files for evaluation test cases
├── results/                 # Evaluation reports (JSON + Markdown)
├── app.py                   # Streamlit application entry point
├── config.py                # Centralized config (models, paths, chunking)
├── evaluate.py              # Evaluation benchmarking engine
├── test_bench.json          # Q&A test cases for evaluation
├── requirements.txt
└── packages.txt             # System-level dependencies (Tesseract)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Tesseract OCR installed on your system
- API keys for Groq and/or Google Gemini

**Install Tesseract:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows — download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### 1. Clone the Repository

```bash
git clone https://github.com/sidharth-vijayan/PolyRAG.git
cd PolyRAG
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

> On Streamlit Cloud, add these as **Secrets** in the dashboard instead.

### 4. Run the Application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Running Evaluations

Run the full evaluation bench against the provided test cases:

```bash
python evaluate.py --bench test_bench.json --output results/ --runs 3
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--bench` | `test_bench.json` | Path to the Q&A test bench JSON |
| `--output` | `results/` | Directory to save reports |
| `--runs` | `3` | Number of runs per test case (for mean ± std) |

Reports are saved as `results/eval_report.json` and `results/eval_report.md`.

---

## ⚙️ Configuration

All key settings are in `config.py`:

```python
# LLM Models
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GEMINI_MODEL = "gemini-2.0-flash"

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K_RESULTS = 4

# Memory
MEMORY_MAX_MESSAGES = 10
```

---

## 🗺️ How It Works

```
User Query
    │
    ▼
┌─────────────────────┐
│   Coordinator Agent │  ← Classifies query → selects agent(s)
└─────────────────────┘
    │         │         │
    ▼         ▼         ▼
Document    Excel     Image
 Agent      Agent     Agent
    │         │         │
    └────┬────┘─────────┘
         ▼
  ChromaDB Vector Store
  (Semantic similarity search)
         │
         ▼
  ┌─────────────┐
  │  Aggregator │  ← Synthesizes context → calls LLM
  └─────────────┘
         │
         ▼
  Final Answer (streamed)
```

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Sidharth Vijayan**  
B.Tech CSE (AI & DS) | MIT World Peace University  
- GitHub: [@sidharth-vijayan](https://github.com/sidharth-vijayan)
- Live Demo: [polyrag.streamlit.app](https://polyrag.streamlit.app/)

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
