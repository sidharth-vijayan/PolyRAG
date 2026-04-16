# Runs a test bench of Q&A pairs through the full RAG pipeline
# and computes retrieval, generation, and system-level metrics.

import os
import sys
import json
import time
import argparse
import statistics
from datetime import datetime

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.document_agent import document_agent
from agents.excel_agent import excel_agent
from agents.image_agent import image_agent
from agents.coordinator import coordinator
from agents.aggregator import aggregator
from core.vector_store import vector_store
from core.embeddings import embedder
from core.memory import ConversationMemory
from config import COLLECTION_DOCUMENTS, COLLECTION_EXCEL, COLLECTION_IMAGES


# ---- Agent dispatch table ----
AGENT_MAP = {
    ".pdf":  ("DocumentAgent", document_agent),
    ".txt":  ("DocumentAgent", document_agent),
    ".docx": ("DocumentAgent", document_agent),
    ".xlsx": ("ExcelAgent",    excel_agent),
    ".csv":  ("ExcelAgent",    excel_agent),
    ".png":  ("ImageAgent",    image_agent),
    ".jpg":  ("ImageAgent",    image_agent),
    ".jpeg": ("ImageAgent",    image_agent),
}

# Map agent names to their ChromaDB collection
AGENT_COLLECTION = {
    "DocumentAgent": COLLECTION_DOCUMENTS,
    "ExcelAgent":    COLLECTION_EXCEL,
    "ImageAgent":    COLLECTION_IMAGES,
}


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RAGEvaluator:
    """
    Evaluates the full RAG pipeline with retrieval, generation,
    and system-level metrics. Supports multi-run averaging.
    """

    def __init__(self, num_runs: int = 1):
        self.num_runs = num_runs

    def evaluate(self, test_cases: list[dict]) -> dict:
        """
        Run the full evaluation bench.

        Args:
            test_cases: List of test case dicts from test_bench.json.

        Returns:
            Full results dict with per-case and aggregate metrics.
        """
        print(f"\n{'='*60}")
        print(f"  RAG Evaluation Bench")
        print(f"  Test cases: {len(test_cases)}")
        print(f"  Runs per case: {self.num_runs}")
        print(f"{'='*60}\n")

        all_case_results = []

        for tc in test_cases:
            print(f"\n--- {tc['id']}: {tc['description']} ---")
            case_result = self._evaluate_single_case(tc)
            all_case_results.append(case_result)

        # Compute aggregate summary
        summary = self._compute_aggregate(all_case_results)

        report = {
            "timestamp": datetime.now().isoformat(),
            "num_runs": self.num_runs,
            "num_test_cases": len(test_cases),
            "summary": summary,
            "test_cases": all_case_results,
        }

        return report


    def _evaluate_single_case(self, tc: dict) -> dict:
        """Evaluate one test case across multiple runs."""

        run_metrics = []

        for run_idx in range(self.num_runs):
            print(f"  Run {run_idx + 1}/{self.num_runs}...")

            # Clean state for each run
            vector_store.clear_all()
            memory = ConversationMemory()

            metrics = self._run_once(tc, memory)
            run_metrics.append(metrics)

        # Aggregate across runs
        averaged = self._average_runs(run_metrics)

        return {
            "id": tc["id"],
            "description": tc["description"],
            "source_file": tc["source_file"],
            "question": tc["question"],
            "expected_answer": tc["expected_answer"],
            "sample_generated_answer": run_metrics[-1].get("generated_answer", ""),
            "llm_source": run_metrics[-1].get("llm_source", ""),
            "metrics": averaged,
        }

    def _run_once(self, tc: dict, memory) -> dict:
        """Execute a single run of a test case and collect raw metrics."""

        source_file = tc["source_file"]
        file_type = tc["file_type"]
        expected_agent = tc["expected_agent"]
        question = tc["question"]
        expected_answer = tc["expected_answer"]

        agent_name, agent = AGENT_MAP.get(file_type, (None, None))
        if agent is None:
            return {"error": f"No agent for file type {file_type}"}

        # ---- 1. Indexing ----
        t0 = time.time()
        index_result = agent.index(source_file)
        indexing_latency_ms = (time.time() - t0) * 1000
        print(f"    Indexed in {indexing_latency_ms:.1f}ms — {index_result}")

        # ---- 2. Retrieval with scores ----
        collection_name = AGENT_COLLECTION.get(expected_agent, COLLECTION_DOCUMENTS)

        t0 = time.time()
        coord_result = coordinator.query(
            question,
            last_upload_agent=expected_agent,
        )
        retrieval_latency_ms = (time.time() - t0) * 1000

        context_chunks = coord_result.get("results", [])
        agents_used = coord_result.get("agents_used", [])

        # Get scored results for similarity metric
        scored_results = vector_store.search_with_scores(
            collection_name=collection_name,
            query_text=question,
        )

        # Context Precision: what fraction of retrieved chunks are from the expected source?
        expected_source = os.path.basename(source_file)
        chunks_from_source = sum(
            1 for c in context_chunks
            if c.get("metadata", {}).get("source") == expected_source
        )
        context_precision = (
            chunks_from_source / len(context_chunks)
            if context_chunks else 0.0
        )

        # Average distance (lower = better)
        distances = [r["distance"] for r in scored_results if r.get("distance") is not None]
        avg_distance = statistics.mean(distances) if distances else None

        # Routing accuracy
        routing_correct = 1.0 if expected_agent in agents_used else 0.0

        # ---- 3. Generation (non-streaming for timing) ----
        history = memory.get_history()

        t0 = time.time()
        generated_answer, llm_source = aggregator.generate_answer(
            query=question,
            context_chunks=context_chunks,
            history=history,
        )
        generation_latency_ms = (time.time() - t0) * 1000

        ttft_ms = None
        try:
            t_stream_start = time.time()
            stream, _ = aggregator.generate_answer_stream(
                query=question,
                context_chunks=context_chunks,
                history=history,
            )
            first_token = next(stream, None)
            if first_token is not None:
                ttft_ms = (time.time() - t_stream_start) * 1000
            # Consume remaining tokens to avoid resource leaks
            for _ in stream:
                pass
        except Exception:
            ttft_ms = None

        # ---- 5. E2E latency (indexing excluded, retrieval + generation) ----
        e2e_latency_ms = retrieval_latency_ms + generation_latency_ms

        # ---- 6. Generation-side metrics ----
        # Faithfulness: does the answer claim to use the document?
        answer_lower = generated_answer.lower()
        if "found from document" in answer_lower:
            faithfulness = 1.0
        elif "based on general knowledge" in answer_lower:
            faithfulness = 0.0
        else:
            faithfulness = 0.5  # Ambiguous

        # Answer similarity: cosine similarity between generated and expected
        try:
            gen_emb = embedder.encode([generated_answer])[0]
            exp_emb = embedder.encode([expected_answer])[0]
            answer_similarity = cosine_similarity(gen_emb, exp_emb)
        except Exception:
            answer_similarity = 0.0

        # Tokens per second (approximate, using word count)
        word_count = len(generated_answer.split())
        tokens_per_sec = (
            (word_count / (generation_latency_ms / 1000))
            if generation_latency_ms > 0 else 0.0
        )

        return {
            "generated_answer": generated_answer,
            "llm_source": llm_source,
            # Retrieval metrics
            "retrieval_latency_ms": retrieval_latency_ms,
            "chunks_retrieved": len(context_chunks),
            "context_precision": context_precision,
            "avg_distance": avg_distance,
            "routing_correct": routing_correct,
            # Generation metrics
            "generation_latency_ms": generation_latency_ms,
            "ttft_ms": ttft_ms,
            "tokens_per_sec": tokens_per_sec,
            "faithfulness": faithfulness,
            "answer_similarity": answer_similarity,
            # System metrics
            "e2e_latency_ms": e2e_latency_ms,
            "indexing_latency_ms": indexing_latency_ms,
        }

    def _average_runs(self, run_metrics: list[dict]) -> dict:
        """
        Average metrics across multiple runs.
        Returns mean ± std for numeric metrics.
        """
        NUMERIC_KEYS = [
            "retrieval_latency_ms",
            "chunks_retrieved",
            "context_precision",
            "avg_distance",
            "routing_correct",
            "generation_latency_ms",
            "ttft_ms",
            "tokens_per_sec",
            "faithfulness",
            "answer_similarity",
            "e2e_latency_ms",
            "indexing_latency_ms",
        ]

        averaged = {}
        for key in NUMERIC_KEYS:
            values = [
                m[key] for m in run_metrics
                if m.get(key) is not None
            ]
            if values:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                averaged[key] = {
                    "mean": round(mean_val, 3),
                    "std": round(std_val, 3),
                }
            else:
                averaged[key] = {"mean": None, "std": None}

        return averaged

    def _compute_aggregate(self, all_case_results: list[dict]) -> dict:
        """Compute aggregate metrics across all test cases."""

        SUMMARY_KEYS = [
            ("avg_context_precision",   "context_precision"),
            ("avg_answer_similarity",   "answer_similarity"),
            ("avg_retrieval_latency_ms", "retrieval_latency_ms"),
            ("avg_generation_latency_ms", "generation_latency_ms"),
            ("avg_e2e_latency_ms",       "e2e_latency_ms"),
            ("avg_indexing_latency_ms",  "indexing_latency_ms"),
            ("avg_ttft_ms",              "ttft_ms"),
            ("avg_tokens_per_sec",       "tokens_per_sec"),
            ("routing_accuracy",         "routing_correct"),
            ("faithfulness_rate",        "faithfulness"),
        ]

        summary = {}
        for summary_key, metric_key in SUMMARY_KEYS:
            values = []
            for cr in all_case_results:
                m = cr["metrics"].get(metric_key, {})
                if isinstance(m, dict) and m.get("mean") is not None:
                    values.append(m["mean"])

            if values:
                summary[summary_key] = {
                    "mean": round(statistics.mean(values), 3),
                    "std": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
                }
            else:
                summary[summary_key] = {"mean": None, "std": None}

        return summary


def save_json_report(report: dict, output_dir: str):
    """Save the full report as JSON."""
    path = os.path.join(output_dir, "eval_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[Report] JSON saved to: {path}")


def save_markdown_report(report: dict, output_dir: str):
    """Generate a paper-ready markdown report."""

    lines = []
    lines.append("# RAG System Evaluation Report\n")
    lines.append(f"**Date:** {report['timestamp'][:10]}  ")
    lines.append(f"**Test Cases:** {report['num_test_cases']}  ")
    lines.append(f"**Runs per Case:** {report['num_runs']}\n")

    # ---- Aggregate Summary ----
    lines.append("## Aggregate Summary\n")
    lines.append("| Metric | Mean | Std |")
    lines.append("|--------|------|-----|")

    DISPLAY_NAMES = {
        "avg_context_precision":    "Context Precision",
        "avg_answer_similarity":    "Answer Similarity",
        "avg_retrieval_latency_ms": "Retrieval Latency (ms)",
        "avg_generation_latency_ms": "Generation Latency (ms)",
        "avg_e2e_latency_ms":       "End-to-End Latency (ms)",
        "avg_indexing_latency_ms":  "Indexing Latency (ms)",
        "avg_ttft_ms":              "Time to First Token (ms)",
        "avg_tokens_per_sec":       "Tokens/sec (approx.)",
        "routing_accuracy":         "Routing Accuracy",
        "faithfulness_rate":        "Faithfulness Rate",
    }

    summary = report["summary"]
    for key, display_name in DISPLAY_NAMES.items():
        entry = summary.get(key, {})
        mean_v = entry.get("mean")
        std_v = entry.get("std")
        mean_str = f"{mean_v:.3f}" if mean_v is not None else "N/A"
        std_str = f"±{std_v:.3f}" if std_v is not None else "N/A"
        lines.append(f"| {display_name} | {mean_str} | {std_str} |")

    # ---- Retrieval Metrics per Test Case ----
    lines.append("\n## Retrieval Metrics (per Test Case)\n")
    lines.append("| ID | Description | Chunks | Context Precision | Avg Distance | Retrieval Latency (ms) |")
    lines.append("|----|-------------|--------|-------------------|--------------|------------------------|")

    for tc in report["test_cases"]:
        m = tc["metrics"]
        lines.append(
            f"| {tc['id']} "
            f"| {tc['description'][:40]} "
            f"| {_fmt(m, 'chunks_retrieved')} "
            f"| {_fmt(m, 'context_precision')} "
            f"| {_fmt(m, 'avg_distance')} "
            f"| {_fmt(m, 'retrieval_latency_ms')} |"
        )

    # ---- Generation Metrics per Test Case ----
    lines.append("\n## Generation Metrics (per Test Case)\n")
    lines.append("| ID | Faithfulness | Answer Similarity | Gen Latency (ms) | TTFT (ms) | Tokens/sec |")
    lines.append("|----|-------------|-------------------|-------------------|-----------|------------|")

    for tc in report["test_cases"]:
        m = tc["metrics"]
        lines.append(
            f"| {tc['id']} "
            f"| {_fmt(m, 'faithfulness')} "
            f"| {_fmt(m, 'answer_similarity')} "
            f"| {_fmt(m, 'generation_latency_ms')} "
            f"| {_fmt(m, 'ttft_ms')} "
            f"| {_fmt(m, 'tokens_per_sec')} |"
        )

    # ---- System Metrics per Test Case ----
    lines.append("\n## System Metrics (per Test Case)\n")
    lines.append("| ID | Routing Correct | Indexing (ms) | E2E Latency (ms) | LLM Source |")
    lines.append("|----|-----------------|---------------|-------------------|------------|")

    for tc in report["test_cases"]:
        m = tc["metrics"]
        lines.append(
            f"| {tc['id']} "
            f"| {_fmt(m, 'routing_correct')} "
            f"| {_fmt(m, 'indexing_latency_ms')} "
            f"| {_fmt(m, 'e2e_latency_ms')} "
            f"| {tc.get('llm_source', 'N/A')} |"
        )

    # ---- Sample Answers ----
    lines.append("\n## Sample Generated Answers\n")
    for tc in report["test_cases"]:
        lines.append(f"### {tc['id']}: {tc['question']}\n")
        lines.append(f"**Expected:** {tc['expected_answer'][:200]}...\n")
        answer_preview = tc.get('sample_generated_answer', '')[:300]
        lines.append(f"**Generated:** {answer_preview}...\n")

    path = os.path.join(output_dir, "eval_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Report] Markdown saved to: {path}")


def _fmt(metrics: dict, key: str) -> str:
    """Format a metric for display: mean ± std."""
    entry = metrics.get(key, {})
    if not isinstance(entry, dict):
        return str(entry)
    mean_v = entry.get("mean")
    std_v = entry.get("std")
    if mean_v is None:
        return "N/A"
    if std_v is not None and std_v > 0:
        return f"{mean_v:.2f}±{std_v:.2f}"
    return f"{mean_v:.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Bench — compute retrieval, "
                    "generation, and system metrics."
    )
    parser.add_argument(
        "--bench",
        type=str,
        default="test_bench.json",
        help="Path to test bench JSON file (default: test_bench.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for reports (default: results/)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per test case for mean±std (default: 3)",
    )
    args = parser.parse_args()

    # Load test bench
    if not os.path.exists(args.bench):
        print(f"Error: Test bench file not found: {args.bench}")
        sys.exit(1)

    with open(args.bench, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases from {args.bench}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run evaluation
    evaluator = RAGEvaluator(num_runs=args.runs)
    report = evaluator.evaluate(test_cases)

    # Save reports
    save_json_report(report, args.output)
    save_markdown_report(report, args.output)

    # Print summary to console
    print(f"\n{'='*60}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*60}")
    summary = report["summary"]
    for key, entry in summary.items():
        if isinstance(entry, dict) and entry.get("mean") is not None:
            display = key.replace("_", " ").title()
            print(f"  {display}: {entry['mean']:.3f} ± {entry['std']:.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
