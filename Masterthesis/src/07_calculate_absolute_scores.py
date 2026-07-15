"""Calculates the absolute visibility scores for deterministic metrics.

This module processes fast, deterministic metrics such as citation counts and word counts
across all generated responses. It parses cached Large Language Model (LLM) responses,
applies the relevant metric functions, and aggregates the results into a comprehensive
CSV format.
"""

import csv
import json
import os


from utils import (
    absolute_wordcount,
    extract_citations_new,
    impression_chat_cite,
    impression_sentence_cite,
    position_adjusted_wordcount,
)

# Fast metrics that do not require local LLMs or computationally heavy ROUGE evaluations
METRICS = {
    "chat_cite": lambda citations: impression_chat_cite(citations, 5),
    "sentence_cite": lambda citations: impression_sentence_cite(citations, 5),
    "position_adjusted_wordcount": lambda citations: position_adjusted_wordcount(
        citations, 5
    ),
    "absolute_wordcount": lambda citations: absolute_wordcount(citations, 5),
}

METHODS = [
    "fluent_gpt",
    "authoritative",
    "more_quotes",
    "citing_credible",
    "simple_language",
    "technical_terms",
    "stats_optimization",
    "llms_txt",
    "inverted_pyramid_mine",
    "autogeo_api_mine",
]


def main() -> None:
    """Executes the calculation of absolute visibility scores.

    The workflow encompasses the following steps:
    1. Reads the list of queries from the systematic dataset CSV.
    2. Loads the baseline cache and all optimization method caches into memory.
    3. Extracts generated citations and computes visibility scores for all fast
       metrics for each query.
    4. Writes the aggregated row-level data to a final CSV output file.

    Raises:
        FileNotFoundError: If the baseline cache or dataset files are unavailable.
        json.JSONDecodeError: If a cache file is invalid JSON.
    """
    print("Starting calculation of absolute visibility scores...")
    csv_file = r"C:\Users\finnb\Documents\emma\Masterthesis\masterarbeit_dataset_systematisch.csv"
    output_csv = (
        r"C:\Users\finnb\Documents\emma\Masterthesis\absolute_visibility_results.csv"
    )

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

    # Read Baseline Cache
    baseline_cache_file = "responses_cache_no_optimization.json"
    if os.path.exists(baseline_cache_file):
        baseline_cache = json.load(open(baseline_cache_file, "r", encoding="utf-8"))
    else:
        print("Error: Baseline Cache not found!")
        return

    # Read Method Caches
    method_caches = {}
    for m in METHODS:
        cache_file = f"responses_cache_{m}.json"
        if os.path.exists(cache_file):
            method_caches[m] = json.load(open(cache_file, "r", encoding="utf-8"))
        else:
            print(f"Warning: Cache for {m} not found!")
            method_caches[m] = {}

    csv_rows = [
        [
            "Prompt_ID",
            "Metric",
            "Method",
            "Source_0",
            "Source_1",
            "Source_2",
            "Source_3",
            "Source_4",
        ]
    ]

    for i, query in enumerate(prompts):
        prompt_id = i + 1

        # 1. Baseline Scores
        if query in baseline_cache:
            ans_dict = baseline_cache[query][-1]
            baseline_ans = ans_dict["responses"][-1][0]
            citations = extract_citations_new(baseline_ans)

            for m_name, m_fn in METRICS.items():
                baseline_scores = m_fn(citations)
                csv_rows.append(
                    [
                        prompt_id,
                        m_name,
                        "Baseline",
                        round(baseline_scores[0], 4),
                        round(baseline_scores[1], 4),
                        round(baseline_scores[2], 4),
                        round(baseline_scores[3], 4),
                        round(baseline_scores[4], 4),
                    ]
                )
        else:
            print(f"Prompt {prompt_id} not found in Baseline Cache. Skipping...")
            continue

        # 2. Method Scores
        for method in METHODS:
            if query in method_caches[method]:
                ans_dict = method_caches[method][query][-1]
                ans = ans_dict["responses"][-1][0]
                citations = extract_citations_new(ans)

                for m_name, m_fn in METRICS.items():
                    scores = m_fn(citations)
                    csv_rows.append(
                        [
                            prompt_id,
                            m_name,
                            method,
                            round(scores[0], 4),
                            round(scores[1], 4),
                            round(scores[2], 4),
                            round(scores[3], 4),
                            round(scores[4], 4),
                        ]
                    )

        if prompt_id % 50 == 0:
            print(f"{prompt_id} / {len(prompts)} processed...")

    with open(output_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf, delimiter=";")
        writer.writerows(csv_rows)

    print(f"\nDone! Absolute scores saved to {output_csv}.")


if __name__ == "__main__":
    main()
