"""Calculates computationally intensive absolute visibility metrics on a remote server.

This module processes metrics that require significant computational resources,
such as ROUGE, BLEU, and perplexity scoring. It iterates over generated responses
and source texts, applying the metrics and progressively writing the results
to a CSV file. The execution is intended for a remote server environment.
"""

import csv
import json
import os


from utils import (
    extract_citations_new,
    impression_bleu,
    impression_length_ratio,
    impression_perplexity,
    impression_rouge,
    impression_rvs,
)

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
    """Executes the calculation of heavy absolute visibility scores.

    The workflow encompasses the following steps:
    1. Reads the list of queries from the systematic dataset CSV.
    2. Loads the baseline cache and method caches into memory.
    3. Iterates over prompts to calculate intensive metrics including ROUGE, BLEU,
       length ratio, and perplexity.
    4. Caches perplexity calculations to optimize performance.
    5. Appends the computed metrics to the output CSV file in batches.
    """
    print(
        "Starting calculation of HEAVY absolute visibility scores on remote server..."
    )

    csv_file = "masterarbeit_dataset_systematisch.csv"
    output_csv = "absolute_heavy_metrics_results.csv"

    # 1. Read Prompts
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

    # 2. Read Baseline Cache
    baseline_cache_file = "responses_cache_no_optimization.json"
    if os.path.exists(baseline_cache_file):
        baseline_cache = json.load(open(baseline_cache_file, "r", encoding="utf-8"))
    else:
        print("Error: Baseline Cache not found!")
        return

    # 3. Read Method Caches
    method_caches = {}
    for m in METHODS:
        cache_file = f"responses_cache_{m}.json"
        if os.path.exists(cache_file):
            method_caches[m] = json.load(open(cache_file, "r", encoding="utf-8"))
        else:
            print(f"Warning: Cache for {m} not found!")
            method_caches[m] = {}

    csv_rows = []
    ppl_cache = {}

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

    for i, query in enumerate(prompts):
        prompt_id = i + 1

        if prompt_id in processed_prompts:
            # print(f"Prompt {prompt_id} / {len(prompts)} already computed. Skipping...")
            continue

        # --- BASELINE ---
        if query in baseline_cache:
            ans_dict = baseline_cache[query][-1]
            ans = ans_dict["responses"][-1][0]
            summaries = [s["summary"] for s in ans_dict["sources"]]

            # Truncate summaries to prevent infinite hangs on massive texts
            for s_idx in range(len(summaries)):
                if len(summaries[s_idx]) > 50000:
                    summaries[s_idx] = summaries[s_idx][:50000]

            citations = extract_citations_new(ans)

            rvs_scores = impression_rvs(citations, 5)
            csv_rows.append(
                [prompt_id, "rvs_sentiment", "Baseline"]
                + [round(x, 4) for x in rvs_scores]
            )

            rouge_scores, bleu_scores, lr_scores, ppl_scores = [], [], [], []
            for idx in range(5):
                rouge_scores.append(
                    impression_rouge(ans, summaries[idx], 5, idx=idx)[idx]
                )
                bleu_scores.append(
                    impression_bleu(ans, summaries[idx], 5, idx=idx)[idx]
                )
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
                [prompt_id, "rouge_l_recall", "Baseline"]
                + [round(x, 4) for x in rouge_scores]
            )
            csv_rows.append(
                [prompt_id, "bleu", "Baseline"] + [round(x, 4) for x in bleu_scores]
            )
            csv_rows.append(
                [prompt_id, "length_ratio", "Baseline"]
                + [round(x, 4) for x in lr_scores]
            )
            csv_rows.append(
                [prompt_id, "perplexity_qwen", "Baseline"]
                + [round(x, 4) for x in ppl_scores]
            )
        else:
            print(f"Prompt {prompt_id} not found in Baseline Cache. Skipping...")
            continue

        # --- METHODS ---
        for method in METHODS:
            if query in method_caches[method]:
                ans_dict = method_caches[method][query][-1]
                ans = ans_dict["responses"][-1][0]
                summaries = [s["summary"] for s in ans_dict["sources"]]

                # Truncate summaries to prevent infinite hangs on massive texts
                for s_idx in range(len(summaries)):
                    if len(summaries[s_idx]) > 50000:
                        summaries[s_idx] = summaries[s_idx][:50000]

                citations = extract_citations_new(ans)

                rvs_scores = impression_rvs(citations, 5)
                csv_rows.append(
                    [prompt_id, "rvs_sentiment", method]
                    + [round(x, 4) for x in rvs_scores]
                )

                rouge_scores, bleu_scores, lr_scores, ppl_scores = [], [], [], []
                for idx in range(5):
                    rouge_scores.append(
                        impression_rouge(ans, summaries[idx], 5, idx=idx)[idx]
                    )
                    bleu_scores.append(
                        impression_bleu(ans, summaries[idx], 5, idx=idx)[idx]
                    )
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
                    [prompt_id, "rouge_l_recall", method]
                    + [round(x, 4) for x in rouge_scores]
                )
                csv_rows.append(
                    [prompt_id, "bleu", method] + [round(x, 4) for x in bleu_scores]
                )
                csv_rows.append(
                    [prompt_id, "length_ratio", method]
                    + [round(x, 4) for x in lr_scores]
                )
                csv_rows.append(
                    [prompt_id, "perplexity_qwen", method]
                    + [round(x, 4) for x in ppl_scores]
                )

        print(f"Prompt {prompt_id} / {len(prompts)} completed. (Heavy Metrics)")

        if prompt_id % 20 == 0:
            with open(output_csv, "a", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf, delimiter=";")
                writer.writerows(csv_rows)
            csv_rows = []

    if len(csv_rows) > 0:
        with open(output_csv, "a", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf, delimiter=";")
            writer.writerows(csv_rows)

    print(f"\nDone! Heavy absolute scores saved to {output_csv}.")


if __name__ == "__main__":
    main()
