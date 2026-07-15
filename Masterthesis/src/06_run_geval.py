"""Computes subjective evaluation metrics for Generative Engine Optimization methods.

This module processes generated responses across all evaluated Generative Engine
Optimization (GEO) methods and calculates GPT-Eval (GEval) subjective visibility
metrics. It implements a multi-threaded execution pool to accelerate the evaluation
process while utilizing thread locks to ensure safe, concurrent writes to the
output CSV file.
"""

import json
import csv
import os
import sys
import time
import numpy as np
import concurrent.futures
from threading import Lock

from utils import (
    impression_subjpos_detailed,
    impression_diversity_detailed,
    impression_uniqueness_detailed,
    impression_follow_detailed,
    impression_influence_detailed,
    impression_relevance_detailed,
    impression_subjcount_detailed,
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

GEO_METHODS = [
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

file_lock = Lock()


def load_cache(filename):
    """Load a JSON cache file from the current directory or a designated fallback path.

    Args:
        filename (str): The name of the cache file to retrieve.

    Returns:
        dict: The loaded data dictionary, or an empty dictionary if the file is not found.
    """
    logs_dir = r"C:\Users\finnb\Documents\emma\Masterthesis\logs and cache"
    alt_path = os.path.join(logs_dir, filename)

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    elif os.path.exists(alt_path):
        with open(alt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def process_prompt(
    i,
    query,
    total_prompts,
    base_scrape_dir,
    baseline_cache,
    method_caches,
    completed_rows,
    output_csv,
):
    """Evaluate a single user query against all subjective GEval metrics.

    Calculates metrics for both the baseline answer and all applied optimization
    methods. Results are concurrently appended to the output file using thread locks.

    Args:
        i (int): The zero-based index of the current prompt.
        query (str): The user query to evaluate.
        total_prompts (int): The total number of prompts in the dataset.
        base_scrape_dir (str): The directory containing original prompt subfolders.
        baseline_cache (dict): The loaded JSON cache containing baseline responses.
        method_caches (dict): A mapping of GEO method names to their respective JSON caches.
        completed_rows (set): A set of previously processed identifiers to avoid redundant calculation.
        output_csv (str): The absolute path to the output CSV file.
    """
    prompt_start_time = time.time()
    prompt_id = i + 1

    prompt_dir = os.path.join(base_scrape_dir, f"prompt_{prompt_id:03d}")
    if not os.path.exists(prompt_dir):
        return

    if query not in baseline_cache:
        print(f"No baseline data found for Prompt {prompt_id}.")
        return

    ans_dict = baseline_cache[query][-1]
    baseline_answers = ans_dict["responses"][-1]

    # Calculate and write Baseline
    for metric_name, metric_fn in GEVAL_METRICS.items():
        key = f"{prompt_id}_{metric_name}_Baseline"
        if key not in completed_rows:
            scores_baseline = [0] * 5
            try:
                for src_i in range(5):
                    res_baseline = np.array(
                        [metric_fn(x, query, 5, idx=src_i) for x in baseline_answers]
                    )
                    if len(res_baseline) > 0 and not np.all(res_baseline == 0):
                        scores_baseline[src_i] = round(
                            float(np.mean(res_baseline, axis=0)[src_i]), 4
                        )
            except Exception as e:
                print(
                    f"Error evaluating Baseline metric {metric_name} for Prompt {prompt_id}: {e}"
                )
                continue

            row = [prompt_id, metric_name, "Baseline", *scores_baseline]
            with file_lock:
                with open(output_csv, "a", newline="", encoding="utf-8") as cf:
                    writer = csv.writer(cf, delimiter=";")
                    writer.writerow(row)
                completed_rows.add(key)

    # Calculate and write for each optimization method
    for m_name in GEO_METHODS:
        m_cache = method_caches[m_name]
        if query not in m_cache:
            continue

        opt_ans_dict = m_cache[query][-1]
        opt_answers = opt_ans_dict["responses"][-1]

        for metric_name, metric_fn in GEVAL_METRICS.items():
            key = f"{prompt_id}_{metric_name}_{m_name}"
            if key in completed_rows:
                continue

            scores_opt = [0] * 5
            try:
                for src_i in range(5):
                    res_after = np.array(
                        [metric_fn(x, query, 5, idx=src_i) for x in opt_answers]
                    )
                    if len(res_after) > 0 and not np.all(res_after == 0):
                        scores_opt[src_i] = round(
                            float(np.mean(res_after, axis=0)[src_i]), 4
                        )
            except Exception as e:
                print(
                    f"Error evaluating metric {metric_name} for Prompt {prompt_id} ({m_name}): {e}"
                )
                continue

            row = [prompt_id, metric_name, m_name, *scores_opt]
            with file_lock:
                with open(output_csv, "a", newline="", encoding="utf-8") as cf:
                    writer = csv.writer(cf, delimiter=";")
                    writer.writerow(row)
                completed_rows.add(key)

    prompt_duration = time.time() - prompt_start_time
    mins, secs = divmod(int(prompt_duration), 60)
    log_msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Prompt {prompt_id} completed. Duration: {mins} Min {secs} Sec\n"

    with file_lock:
        with open("geval_runtime.log", "a", encoding="utf-8") as lf:
            lf.write(log_msg)
    print(f"Prompt {prompt_id} completed. Duration: {mins} Min {secs} Sec")


def main():
    """Execute the GEval subjective metrics workflow concurrently.

    Reads user queries from the generated dataset, loads associated answer caches,
    and initiates a thread pool to compute missing evaluation metrics. Prevents
    redundant calculations by inspecting existing output logs.
    """
    csv_file = r"C:\Users\finnb\Documents\emma\Masterthesis\01_Datapreparation\User Query creation\masterarbeit_dataset_systematisch.csv"
    base_scrape_dir = r"C:\Users\finnb\Documents\emma\Masterthesis\01_Datapreparation\Webcrawling of User Queries\scraped_data_full"
    output_csv = (
        r"C:\Users\finnb\Documents\emma\Masterthesis\geval_absolute_results.csv"
    )

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
        sys.exit(1)

    print("Loading caches into RAM... This might take a moment.")
    baseline_cache = load_cache("responses_cache_no_optimization.json")
    method_caches = {m: load_cache(f"responses_cache_{m}.json") for m in GEO_METHODS}
    print("Caches loaded successfully!")

    completed_rows = set()
    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            for row in reader:
                if row and len(row) > 3:
                    # Key: Prompt_ID_Metric_Method
                    completed_rows.add(f"{row[0]}_{row[1]}_{row[2]}")
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

    # 10 parallel threads are the sweet spot for a single 24GB GPU (prevents KV-Cache Thrashing)
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
                    base_scrape_dir,
                    baseline_cache,
                    method_caches,
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
            print(
                "\n[INFO] Cancelled by user (CTRL+C). Terminating program immediately..."
            )
            os._exit(1)


if __name__ == "__main__":
    main()
