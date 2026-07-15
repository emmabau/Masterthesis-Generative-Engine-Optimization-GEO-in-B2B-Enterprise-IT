"""Performs a quality review on generated texts and regenerates failed cases.

This module scans through optimized textual outputs to identify instances that
fall below a minimal word count threshold. For these identified problematic cases,
it clears corresponding cache entries and invokes optimization methods and language
models to regenerate the text and corresponding AI search engine responses.
"""

import csv
import glob
import json
import logging
import os
import sys

from geo_functions import (
    authoritative,
    autogeo_api_mine,
    citing_credible,
    fluent_optimization_gpt,
    inverted_pyramid_mine,
    llms_txt,
    more_quotes,
    simple_language,
    stats_optimization,
    technical_terms,
)
from utils import get_answer

GEO_METHODS = {
    "fluent_gpt": fluent_optimization_gpt,
    "authoritative": authoritative,
    "more_quotes": more_quotes,
    "citing_credible": citing_credible,
    "simple_language": simple_language,
    "technical_terms": technical_terms,
    "stats_optimization": stats_optimization,
    "llms_txt": llms_txt,
    "inverted_pyramid_mine": inverted_pyramid_mine,
    "autogeo_api_mine": autogeo_api_mine,
    "autogeo_api": autogeo_api_mine,
}

base_opt = r"C:\Users\finnb\Documents\emma\Masterthesis\02_Benchmark\AI Answers - with GEO optimization"
base_scrape_dir = r"C:\Users\finnb\Documents\emma\Masterthesis\01_Datapreparation\Webcrawling of User Queries\scraped_data_full"
csv_file = r"C:\Users\finnb\Documents\emma\Masterthesis\01_Datapreparation\User Query creation\masterarbeit_dataset_systematisch.csv"


def main() -> None:
    """Executes the quality review and regeneration process.

    The workflow encompasses the following steps:
    1. Initializes logging to track the regeneration progress.
    2. Reads the dataset to retrieve original prompts.
    3. Identifies poorly optimized responses based on a word count threshold.
    4. Iterates over problematic cases, clears relevant cache entries, and triggers
       regeneration via the respective optimization methods.
    5. Cleans corresponding evaluation results from the GEval CSV.
    """
    log_file = r"C:\Users\finnb\Documents\emma\Masterthesis\logs and cache\quality_review_progress.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Starting Quality Review Regeneration...")

    # 1. Read queries
    prompts = []
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)
            for row in reader:
                if row:
                    prompts.append(row[0])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. Find bad cases
    methods = [
        d.replace("optimized_data_", "")
        for d in os.listdir(base_opt)
        if os.path.isdir(os.path.join(base_opt, d)) and d != "optimized_data_Baseline"
    ]
    bad_cases = []

    for method in methods:
        method_dir = os.path.join(base_opt, f"optimized_data_{method}")
        prompt_dirs = [d for d in os.listdir(method_dir) if d.startswith("prompt_")]
        for prompt_dir in prompt_dirs:
            files = glob.glob(os.path.join(method_dir, prompt_dir, "source_*.txt"))
            if not files:
                continue
            opt_file = files[0]

            with open(opt_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            words = content.split()
            if len(words) <= 10:
                prompt_id = int(prompt_dir.replace("prompt_", ""))
                source_idx = (
                    int(
                        os.path.basename(opt_file)
                        .replace("source_", "")
                        .replace(".txt", "")
                    )
                    - 1
                )
                bad_cases.append(
                    {
                        "method": method,
                        "prompt_id": prompt_id,
                        "source_idx": source_idx,
                        "file": opt_file,
                    }
                )

    logging.info(f"Found {len(bad_cases)} bad cases to regenerate.")
    if len(bad_cases) == 0:
        return

    # 3. Regenerate!
    logs_dir = r"C:\Users\finnb\Documents\emma\Masterthesis\logs and cache"
    review_cache_file = os.path.join(logs_dir, "quality_review_cache.json")
    if os.path.exists(review_cache_file):
        with open(review_cache_file, "r", encoding="utf-8") as f:
            completed_cases = set(json.load(f))
    else:
        completed_cases = set()

    # Track which methods and prompts need geval deletion
    geval_deletions = set()  # tuples of (Prompt_ID, Method)

    for c in bad_cases:
        method = c["method"]
        prompt_id = c["prompt_id"]
        source_idx = c["source_idx"]
        query = prompts[prompt_id - 1]

        cache_id = f"{method}_prompt_{prompt_id}"
        if cache_id in completed_cases:
            logging.info(
                f"Skipping {method} for Prompt {prompt_id} (already regenerated)"
            )
            continue

        logging.info(
            f"Regenerating {method} for Prompt {prompt_id} (Source {source_idx + 1})"
        )
        geval_deletions.add((str(prompt_id), method))

        # Load original summaries
        summaries = []
        prompt_dir = os.path.join(base_scrape_dir, f"prompt_{prompt_id:03d}")
        for j in range(1, 6):
            source_file = os.path.join(prompt_dir, f"source_{j}.txt")
            if os.path.exists(source_file):
                with open(source_file, "r", encoding="utf-8") as sf:
                    summaries.append(sf.read())
            else:
                summaries.append("")

        # Delete GEO Cache Entry
        geo_cache_file = os.path.join(
            logs_dir, f"geo_optimizations_cache_{method}_qwen3-8b.json"
        )
        if os.path.exists(geo_cache_file):
            with open(geo_cache_file, "r", encoding="utf-8") as f:
                geo_cache = json.load(f)
            cache_key = f"Prompt_{prompt_id:03d}_Source_{source_idx + 1}"
            if cache_key in geo_cache:
                del geo_cache[cache_key]
                with open(geo_cache_file, "w", encoding="utf-8") as f:
                    json.dump(geo_cache, f, indent=2)

        # Call the GEO method to force regeneration
        logging.info("  -> Calling Ollama to rewrite text...")
        GEO_METHODS[method](
            summaries[source_idx], prompt_id=prompt_id, source_idx=source_idx
        )

        # Load newly generated text
        with open(c["file"], "r", encoding="utf-8") as f:
            optimized_text = f.read()

        # Update summaries array
        summaries[source_idx] = optimized_text

        # Delete Response Cache Entry
        resp_cache_file = os.path.join(logs_dir, f"responses_cache_{method}.json")
        if os.path.exists(resp_cache_file):
            with open(resp_cache_file, "r", encoding="utf-8") as f:
                resp_cache = json.load(f)
        else:
            resp_cache = {}

        if query in resp_cache:
            del resp_cache[query]
            with open(resp_cache_file, "w", encoding="utf-8") as f:
                json.dump(resp_cache, f, indent=2)

        # Call get_answer to regenerate the AI Search Engine response
        logging.info("  -> Regenerating AI Search Engine Answer...")
        get_answer(
            query,
            summaries=summaries,
            num_completions=1,
            n=5,
            loaded_cache=resp_cache,
            target_cache_file=resp_cache_file,
        )

        completed_cases.add(cache_id)
        with open(review_cache_file, "w", encoding="utf-8") as f:
            json.dump(list(completed_cases), f, indent=2)

        logging.info(f"  -> Done with Prompt {prompt_id} for {method}.\n")

    # 4. Clean GEval CSV
    geval_csv = r"C:\Users\finnb\Documents\emma\Masterthesis\geval_absolute_results.csv"
    if os.path.exists(geval_csv):
        logging.info(
            f"Cleaning GEval CSV for the {len(bad_cases)} regenerated prompts..."
        )
        rows_kept = []
        with open(geval_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader)
            rows_kept.append(header)
            for row in reader:
                if not row:
                    continue
                # if (Prompt_ID, Method) in geval_deletions, we drop it!
                if (row[0], row[2]) not in geval_deletions:
                    rows_kept.append(row)

        with open(geval_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerows(rows_kept)

    logging.info(
        "\nAll 80 prompts successfully regenerated! Now run 07_calculate_absolute_scores.py and 06_run_geval.py."
    )


if __name__ == "__main__":
    main()
