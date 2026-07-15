"""Evaluates GEval visibility metrics on baseline responses via parallel execution.

This module is designed to run remotely, applying a suite of GEval metrics
(e.g., diversity, uniqueness, influence) against regenerated baseline texts.
It utilizes a multi-threading approach to accelerate the LLM-based evaluation
process, appending results securely to an output file.
"""

import concurrent.futures
import csv
import json
import os
import time
from threading import Lock
from typing import Dict, Set

import numpy as np

from utils import (
    impression_diversity_detailed,
    impression_follow_detailed,
    impression_influence_detailed,
    impression_relevance_detailed,
    impression_subjcount_detailed,
    impression_subjpos_detailed,
    impression_uniqueness_detailed,
)

GEVAL_METRICS = {
    "subjpos_detailed": impression_subjpos_detailed,
    "diversity_detailed": impression_diversity_detailed,
    "uniqueness_detailed": impression_uniqueness_detailed,
    "follow_detailed": impression_follow_detailed,
    "influence_detailed": impression_influence_detailed,
    "relevance_detailed": impression_relevance_detailed,
    "subjcount_detailed": impression_subjcount_detailed,
}

file_lock = Lock()


def process_prompt(
    i: int,
    query: str,
    total_prompts: int,
    v2_cache: Dict,
    completed_rows: Set[str],
    output_csv: str,
) -> None:
    """Processes a single prompt to calculate GEval metrics.

    Args:
        i: The zero-based index of the prompt.
        query: The user query string.
        total_prompts: The total number of prompts to process.
        v2_cache: The loaded cache dictionary containing generated responses.
        completed_rows: A set tracking metrics that have already been computed.
        output_csv: The filepath for the output CSV.
    """
    prompt_id = i + 1

    if query not in v2_cache:
        return

    ans_dict = v2_cache[query][-1]
    v2_answers = ans_dict["responses"][-1]

    method = "Baseline_V2"

    for metric_name, metric_fn in GEVAL_METRICS.items():
        key = f"{prompt_id}_{metric_name}_{method}"
        if key in completed_rows:
            continue

        scores = [0] * 5
        try:
            for src_i in range(5):
                res = np.array([metric_fn(x, query, 5, idx=src_i) for x in v2_answers])
                if len(res) > 0 and not np.all(res == 0):
                    scores[src_i] = round(float(np.mean(res, axis=0)[src_i]), 4)
        except Exception as e:
            print(f"Error evaluating metric {metric_name} for Prompt {prompt_id}: {e}")
            continue

        row = [prompt_id, metric_name, method, *scores]
        with file_lock:
            with open(output_csv, "a", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf, delimiter=";")
                writer.writerow(row)
            completed_rows.add(key)

    print(
        f"[{time.strftime('%H:%M:%S')}] Prompt {prompt_id} / {total_prompts} GEval completed."
    )


def main() -> None:
    """Executes the multi-threaded GEval baseline simulation.

    The workflow encompasses the following steps:
    1. Reads user queries from the systematic dataset CSV.
    2. Loads the cached baseline responses into memory.
    3. Retrieves previous execution progress from the output CSV.
    4. Initializes a thread pool to compute GEval metrics in parallel for each prompt.
    5. Aggregates and logs the metric scores to the CSV securely.
    """
    print("Starting GEval V2 Baseline Simulation on remote server...")
    csv_file = "masterarbeit_dataset_systematisch.csv"
    cache_file_v2 = "responses_cache_baseline_v2.json"
    output_csv = "geval_baseline_v2_results.csv"

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

    print("Loading V2 Cache into RAM...")
    if os.path.exists(cache_file_v2):
        with open(cache_file_v2, "r", encoding="utf-8") as f:
            v2_cache = json.load(f)
    else:
        print(
            f"Error: {cache_file_v2} not found! Cannot evaluate GEval without generated answers."
        )
        return

    completed_rows = set()
    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            for row in reader:
                if row and len(row) > 3:
                    completed_rows.add(f"{row[0]}_{row[1]}_{row[2]}")
        print(
            f"Resuming: Found {len(completed_rows) // 7} already fully processed prompts in {output_csv}."
        )
    else:
        header = [
            "Prompt_ID",
            "Metric",
            "Method",
            "Source_0",
            "Source_1",
            "Source_2",
            "Source_3",
            "Source_4",
        ]
        with open(output_csv, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf, delimiter=";")
            writer.writerow(header)

    total_prompts = len(prompts)
    max_threads = 10

    print(f"Starting multi-threading with {max_threads} parallel threads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for i, query in enumerate(prompts):
            futures.append(
                executor.submit(
                    process_prompt,
                    i,
                    query,
                    total_prompts,
                    v2_cache,
                    completed_rows,
                    output_csv,
                )
            )

        try:
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"A thread crashed: {e}")
        except KeyboardInterrupt:
            print("\n[INFO] Cancelled by user. Terminating...")
            os._exit(1)

    print(f"\nDone! GEval Baseline V2 complete. Results saved to {output_csv}.")


if __name__ == "__main__":
    main()
