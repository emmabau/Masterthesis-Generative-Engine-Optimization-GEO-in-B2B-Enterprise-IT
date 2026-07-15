"""Re-generates baseline answers and computes visibility metrics remotely.

This script executes exclusively on a remote GPU server. It orchestrates the
regeneration of unoptimized baseline answers utilizing a specified language model
(qwen3:8b), caches the generated outputs, and sequentially computes ten predefined
visibility metrics to evaluate the results.
"""

import csv
import json
import os
import time
from typing import List

import requests

from utils import (
    absolute_wordcount,
    extract_citations_new,
    impression_bleu,
    impression_chat_cite,
    impression_length_ratio,
    impression_perplexity,
    impression_pos_count_simple,
    impression_rouge,
    impression_rvs,
    impression_sentence_cite,
    position_adjusted_wordcount,
)

query_prompt = """Write an accurate and concise answer for the given user question, using _only_ the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Français, Español, Deutsch, or 日本語 should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be _immediately followed_ by an in-line citation to the search result(s). The cited search result(s) should fully support _all_ the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {query}

Search Results:
{source_text}
"""


def generate_answer_remote(
    query: str, sources: List[str], model: str = "qwen3:8b", temperature: float = 0.5
) -> str:
    """Generates an answer using a remote language model API.

    Args:
        query: The user query to answer.
        sources: A list of source texts to draw information from.
        model: The model identifier to use for generation.
        temperature: The sampling temperature.

    Returns:
        The generated answer as a string.

    Raises:
        requests.exceptions.RequestException: If the API call fails continuously.
    """
    source_text = "\n\n".join(
        [
            "### Source " + str(idx + 1) + ":\n" + source + "\n\n\n"
            for idx, source in enumerate(sources)
        ]
    )
    prompt = query_prompt.format(query=query, source_text=source_text)

    # Wir greifen direkt lokal auf dem Server auf den Ollama-Port zu (11434 statt 11435)
    url = "http://localhost:11434/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
        "top_p": 1,
        "n": 1,
    }

    while True:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"] + "\n"
        except Exception as e:
            print("Error in calling local Ollama API", e)
            time.sleep(15)


def main() -> None:
    """Executes the baseline V2 simulation and metric calculation process.

    The workflow encompasses the following steps:
    1. Reads user queries from the designated dataset.
    2. Loads existing unoptimized sources from previous cache versions.
    3. Initiates generation of new baseline answers utilizing the remote model API.
    4. Caches newly generated responses.
    5. Computes fast visibility metrics (e.g., citation frequencies, word counts).
    6. Computes computationally intensive metrics (e.g., ROUGE, BLEU, perplexity).
    7. Progressively saves the aggregated results to an output CSV file.
    """
    print("Starting V2 Baseline Simulation on remote server...")

    csv_file = "masterarbeit_dataset_systematisch.csv"
    output_csv = "baseline_v2_full_results.csv"
    cache_file_v2 = "responses_cache_baseline_v2.json"

    # Read Prompts
    prompts = []
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # skip header
            for row in reader:
                if row:
                    prompts.append(row[0])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Read original unoptimized sources from V1 cache
    baseline_cache_file = "responses_cache_no_optimization.json"
    if os.path.exists(baseline_cache_file):
        old_cache = json.load(open(baseline_cache_file, "r", encoding="utf-8"))
    else:
        print("Error: Old Baseline Cache not found! Needed for sources.")
        return

    # Initialize or load V2 Cache
    if os.path.exists(cache_file_v2):
        v2_cache = json.load(open(cache_file_v2, "r", encoding="utf-8"))
    else:
        v2_cache = {}

    # Resume Logic for CSV
    processed_prompts = set()
    if os.path.exists(output_csv):
        try:
            with open(output_csv, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=";")
                for row in reader:
                    if row and row[0].isdigit():
                        processed_prompts.add(int(row[0]))
            print(
                f"Resuming: Found {len(processed_prompts)} already processed prompts in {output_csv}."
            )
        except Exception as e:
            print(f"Warning reading existing CSV: {e}")

    csv_rows = []
    ppl_cache = {}

    for i, query in enumerate(prompts):
        prompt_id = i + 1

        if prompt_id in processed_prompts:
            continue

        if query not in old_cache:
            print(f"Prompt {prompt_id} not found in old cache. Skipping...")
            continue

        # Get original sources
        ans_dict = old_cache[query][-1]
        summaries = [s["summary"] for s in ans_dict["sources"]]

        # 1. Generate NEW answer (if not already generated and cached in V2)
        if query in v2_cache:
            ans = v2_cache[query][-1]["responses"][-1][0]
        else:
            print(f"[{prompt_id}/{len(prompts)}] Generating new qwen3:8b answer...")
            ans = generate_answer_remote(query, summaries, model="qwen3:8b")

            # Save to V2 Cache
            new_entry = {
                "sources": [{"summary": s} for s in summaries],
                "responses": [[ans]],
            }
            if query not in v2_cache:
                v2_cache[query] = []
            v2_cache[query].append(new_entry)

            # Write cache to disk
            with open(cache_file_v2, "w", encoding="utf-8") as f:
                json.dump(v2_cache, f, indent=2)

        # Truncate summaries to prevent infinite hangs in NLP metrics
        for s_idx in range(len(summaries)):
            if len(summaries[s_idx]) > 50000:
                summaries[s_idx] = summaries[s_idx][:50000]

        citations = extract_citations_new(ans)

        # 2. Compute Fast Metrics
        method = "Baseline_V2"

        chat_cite_scores = impression_chat_cite(citations, 5)
        sent_cite_scores = impression_sentence_cite(citations, 5)
        abs_wc_scores = absolute_wordcount(citations, 5, normalize=False)
        pos_wc_scores = position_adjusted_wordcount(citations, 5, normalize=False)
        pos_simp_scores = impression_pos_count_simple(citations, 5, normalize=False)

        csv_rows.append(
            [prompt_id, "chat_cite", method] + [round(x, 4) for x in chat_cite_scores]
        )
        csv_rows.append(
            [prompt_id, "sentence_cite", method]
            + [round(x, 4) for x in sent_cite_scores]
        )
        csv_rows.append(
            [prompt_id, "absolute_wordcount", method]
            + [round(x, 4) for x in abs_wc_scores]
        )
        csv_rows.append(
            [prompt_id, "position_adjusted_wordcount", method]
            + [round(x, 4) for x in pos_wc_scores]
        )
        csv_rows.append(
            [prompt_id, "impression_pos_count_simple", method]
            + [round(x, 4) for x in pos_simp_scores]
        )

        # 3. Compute Heavy Metrics
        rvs_scores = impression_rvs(citations, 5)
        csv_rows.append(
            [prompt_id, "rvs_sentiment", method] + [round(x, 4) for x in rvs_scores]
        )

        rouge_scores, bleu_scores, lr_scores, ppl_scores = [], [], [], []
        for idx in range(5):
            rouge_scores.append(impression_rouge(ans, summaries[idx], 5, idx=idx)[idx])
            bleu_scores.append(impression_bleu(ans, summaries[idx], 5, idx=idx)[idx])
            lr_scores.append(
                impression_length_ratio(ans, summaries[idx], 5, idx=idx)[idx]
            )

            text_hash = hash(summaries[idx])
            if text_hash not in ppl_cache:
                if len(summaries[idx].strip()) > 0:
                    ppl_cache[text_hash] = impression_perplexity(
                        ans, summaries[idx], 5, idx=idx
                    )[idx]
                else:
                    ppl_cache[text_hash] = 0.0
            ppl_scores.append(ppl_cache[text_hash])

        csv_rows.append(
            [prompt_id, "rouge_l_recall", method] + [round(x, 4) for x in rouge_scores]
        )
        csv_rows.append(
            [prompt_id, "bleu", method] + [round(x, 4) for x in bleu_scores]
        )
        csv_rows.append(
            [prompt_id, "length_ratio", method] + [round(x, 4) for x in lr_scores]
        )
        csv_rows.append(
            [prompt_id, "perplexity_qwen", method] + [round(x, 4) for x in ppl_scores]
        )

        print(f"Prompt {prompt_id} / {len(prompts)} completed. (V2 Metrics saved)")

        if prompt_id % 20 == 0 or prompt_id == len(prompts):
            with open(output_csv, "a", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf, delimiter=";")
                writer.writerows(csv_rows)
            csv_rows = []

    if len(csv_rows) > 0:
        with open(output_csv, "a", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf, delimiter=";")
            writer.writerows(csv_rows)

    print(f"\nDone! V2 Baseline simulation complete. Results saved to {output_csv}.")


if __name__ == "__main__":
    main()
