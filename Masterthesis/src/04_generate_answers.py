"""Generates baseline answers for queries using a local language model.

This module processes a global cache of queries and their associated web search
results. It formats these inputs into comprehensive prompts and requests
generated responses from a local large language model without any generative
engine optimization. Results are saved incrementally to persist data across
execution sessions.
"""

import json
import os
import requests
import time

CACHE_FILE = "global_cache.json"
OUTPUT_FILE = "responses_cache_no_optimization.json"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
MODEL_NAME = "qwen3:8b"

SYSTEM_PROMPT_TEMPLATE = """Write an accurate and concise answer for the given user question, using only the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Francais, Espamol, Deutsch, or should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be immediately followed by an in-line citation to the search result(s). The cited search result(s) should fully support all the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {query}

Search Results:
{source_text}"""


def main():
    """Execute the baseline answer generation workflow.

    Loads the global cache containing queries and search summaries. Retrieves
    or initializes an output cache to support resuming interrupted executions.
    Iterates through queries, formats prompts utilizing the designated template,
    and executes requests to the local language model API. Successfully generated
    answers and their processing durations are incrementally stored in the output
    cache file.
    """
    # 1. Load the global cache
    if not os.path.exists(CACHE_FILE):
        print(f"Error: {CACHE_FILE} not found.")
        return

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        global_cache = json.load(f)

    # 2. Load or initialize output cache
    output_cache = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            output_cache = json.load(f)

    total_prompts = len(global_cache)
    processed = 0

    print(f"Processing {total_prompts} prompts...")

    for query, data_list in global_cache.items():
        processed += 1
        if not data_list:
            continue

        # Skip if already processed
        if query in output_cache:
            print(f"[{processed}/{total_prompts}] Skipping (already processed).")
            continue

        print(f"\n[{processed}/{total_prompts}] Generating answer for: {query[:50]}...")

        sources = data_list[0].get("sources", [])

        # Build source text
        source_text_parts = []
        for idx, source in enumerate(sources):
            # The prompt asks for [1], [2], etc. so it is 1-indexed
            text = source.get("summary", "") or source.get("text", "")
            source_text_parts.append(f"[{idx + 1}] {text}")

        source_text = "\n\n".join(source_text_parts)

        prompt = SYSTEM_PROMPT_TEMPLATE.format(query=query, source_text=source_text)

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            # Additional options can be added here, e.g., temperature
            "options": {"temperature": 0.3, "num_ctx": 16384},
        }

        headers = {}
        if OLLAMA_API_KEY:
            headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

        try:
            start_time = time.time()
            res = requests.post(
                OLLAMA_API_GENERATE, json=payload, headers=headers, timeout=600
            )
            res.raise_for_status()
            res_data = res.json()
            answer = res_data.get("response", "")

            output_cache[query] = {
                "answer": answer,
                "duration_seconds": time.time() - start_time,
            }
            print(f"  -> Done in {time.time() - start_time:.1f}s")

            # Save incrementally
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(output_cache, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"  -> Error calling Ollama API: {e}")
            # break here to avoid hammering API if it's down
            break


if __name__ == "__main__":
    main()
