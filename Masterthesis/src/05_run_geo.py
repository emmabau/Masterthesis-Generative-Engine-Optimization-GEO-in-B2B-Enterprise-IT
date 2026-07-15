"""Evaluates Generative Engine Optimization (GEO) methods on baseline answers.

This module computes visibility metrics (e.g., word counts, citations,
perplexity, and relative visibility score) before and after applying various
GEO optimizations. It measures the relative improvement (Delta) for each
method and stores the final aggregated results in a structured format.
"""

import csv
import json
import logging
import os
import random
import sys
from typing import List

import numpy as np

from geo_functions import (
    autogeo_api_mine,
    authoritative,
    citing_credible,
    fluent_optimization_gpt,
    inverted_pyramid_mine,
    llms_txt,
    more_quotes,
    simple_language,
    stats_optimization,
    technical_terms,
)
from utils import (
    absolute_wordcount,
    extract_citations_new,
    get_answer,
    impression_bleu,
    impression_chat_cite,
    impression_diversity_detailed,
    impression_follow_detailed,
    impression_influence_detailed,
    impression_length_ratio,
    impression_perplexity,
    impression_pos_count_simple,
    impression_relevance_detailed,
    impression_rouge,
    impression_rvs,
    impression_sentence_cite,
    impression_subjcount_detailed,
    impression_subjective_impression,
    impression_subjpos_detailed,
    impression_uniqueness_detailed,
    position_adjusted_wordcount,
)


def identity(summary, source=None):
    """Return the exact summary unchanged as a baseline optimization method.

    Args:
            summary (str): The original text summary.
            source (str, optional): The original source URL or text. Defaults to None.

    Returns:
            str: The unchanged summary text.
    """
    return summary


IMPRESSION_FNS = {
    "position_adjusted_wordcount": position_adjusted_wordcount,
    "absolute_wordcount": absolute_wordcount,
    "simple_pos": impression_pos_count_simple,
    "subjective_score": impression_subjective_impression,
    "subjpos_detailed": impression_subjpos_detailed,
    "diversity_detailed": impression_diversity_detailed,
    "uniqueness_detailed": impression_uniqueness_detailed,
    "follow_detailed": impression_follow_detailed,
    "influence_detailed": impression_influence_detailed,
    "relevance_detailed": impression_relevance_detailed,
    "subjcount_detailed": impression_subjcount_detailed,
    "rouge_l_recall": impression_rouge,
    "perplexity_qwen": impression_perplexity,
    "chat_cite": impression_chat_cite,
    "sentence_cite": impression_sentence_cite,
    "bleu": impression_bleu,
    "length_ratio": impression_length_ratio,
    "rvs_sentiment": impression_rvs,
}


GEO_METHODS = {
    "identity": identity,
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
}

loaded_cache = None


def improve(
    query: str,
    idx: int,
    sources: List[str] = None,
    summaries: List[str] = None,
    metrics: List[str] = None,
    prompt_id=None,
) -> dict:
    """Calculate performance deltas for applied generative engine optimizations.

    Computes the difference in visibility metrics between baseline responses
    and responses generated after altering specific source content via GEO
    optimization methods.

    Args:
            query (str): The search query string.
            idx (int): The index of the targeted source or summary to optimize.
            sources (list of str, optional): A list containing the raw source texts. Defaults to None.
            summaries (list of str, optional): A list containing source summaries. Defaults to None.
            metrics (list of str, optional): A list specifying the metric keys to evaluate. Defaults to None.
            prompt_id (int, optional): The identifier of the current prompt to assure deterministic processing. Defaults to None.

    Returns:
            dict: A mapping of each evaluated metric to a NumPy array containing the computed score improvements (deltas).
    """
    baseline_cache_file = "responses_cache_no_optimization.json"
    if os.path.exists(baseline_cache_file):
        baseline_cache = json.load(open(baseline_cache_file, "r", encoding="utf-8"))
    else:
        baseline_cache = {}

    if metrics is None:
        metrics = ["position_adjusted_wordcount", "rouge_l_recall", "perplexity_qwen"]

    print(f"query is: {query}")
    # Scientific adjustment: num_completions=1 saves 80% runtime!
    answers_dict = get_answer(
        query,
        summaries=summaries,
        num_completions=1,
        n=5,
        loaded_cache=baseline_cache,
        target_cache_file=baseline_cache_file,
    )
    if sources is None:
        sources = [x["source"] for x in answers_dict["sources"]]
    if summaries is None:
        summaries = [x["summary"] for x in answers_dict["sources"]]

    answers = answers_dict["responses"][-1]

    # 1. Calculate Baseline Scores for all metrics
    init_scores_dict = {}
    for m_name in metrics:
        imp_fn = IMPRESSION_FNS[m_name]
        if imp_fn in [
            impression_rouge,
            impression_perplexity,
            impression_bleu,
            impression_length_ratio,
        ]:
            scores = np.array([imp_fn(x, summaries[idx], 5, idx=idx) for x in answers])
        elif imp_fn in [
            impression_subjective_impression,
            impression_subjpos_detailed,
            impression_diversity_detailed,
            impression_uniqueness_detailed,
            impression_follow_detailed,
            impression_influence_detailed,
            impression_relevance_detailed,
            impression_subjcount_detailed,
        ]:
            scores = np.array([imp_fn(x, query, 5, idx=idx) for x in answers])
            scores = scores[~np.all(scores == 0, axis=1)]
        else:
            scores = np.array([imp_fn(extract_citations_new(x), 5) for x in answers])
        init_scores_dict[m_name] = scores.mean(axis=0)

    improvements_dict = {m: [] for m in metrics}

    # 2. Loop over all GEO methods (Generate answer and immediately evaluate all metrics)
    for meth_name in GEO_METHODS:
        method_cache_file = f"responses_cache_{meth_name}.json"
        if os.path.exists(method_cache_file):
            method_cache = json.load(open(method_cache_file, "r", encoding="utf-8"))
        else:
            method_cache = {}

        if query in method_cache:
            print("Cache Hit")
            ans_dict = method_cache[query][-1]
            summaries_copy = [src["summary"] for src in ans_dict["sources"]]
        else:
            if prompt_id is not None:
                optimized_text = GEO_METHODS[meth_name](
                    summaries[idx], prompt_id=prompt_id, source_idx=idx
                )
            else:
                optimized_text = GEO_METHODS[meth_name](summaries[idx])

            summaries_copy = summaries[:idx] + [optimized_text] + summaries[idx + 1 :]
            ans_dict = get_answer(
                query,
                summaries=summaries_copy,
                num_completions=1,
                n=5,
                loaded_cache=method_cache,
                target_cache_file=method_cache_file,
            )
        ans = ans_dict["responses"][-1]

        for m_name in metrics:
            imp_fn = IMPRESSION_FNS[m_name]
            if imp_fn in [
                impression_rouge,
                impression_perplexity,
                impression_bleu,
                impression_length_ratio,
            ]:
                # PPL, ROUGE, BLEU, LengthRatio need reference text
                scores = np.array(
                    [imp_fn(x, summaries_copy[idx], 5, idx=idx) for x in ans]
                )
            elif imp_fn in [
                impression_subjective_impression,
                impression_subjpos_detailed,
                impression_diversity_detailed,
                impression_uniqueness_detailed,
                impression_follow_detailed,
                impression_influence_detailed,
                impression_relevance_detailed,
                impression_subjcount_detailed,
            ]:
                scores = np.array([imp_fn(x, query, 5, idx=idx) for x in ans])
                scores = scores[~np.all(scores == 0, axis=1)]
            else:
                scores = [imp_fn(extract_citations_new(x), 5) for x in ans]

            final_scores = np.array(scores).mean(axis=0)
            improvements_dict[m_name].append(final_scores - init_scores_dict[m_name])

    # Convert lists to NumPy Arrays
    for m_name in metrics:
        improvements_dict[m_name] = np.vstack(improvements_dict[m_name])

    return improvements_dict


if __name__ == "__main__":
    log_file = r"c:\Users\finnb\Documents\emma\Masterthesis\benchmark_progress.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Set the cache to the user-requested baseline file
    os.environ["GLOBAL_CACHE_FILE"] = "responses_cache_no_optimization.json"

    method_names = list(GEO_METHODS.keys())
    metrics_to_run = [
        "chat_cite",
        "sentence_cite",
        "position_adjusted_wordcount",
        "absolute_wordcount",
        "rouge_l_recall",
        "bleu",
        "length_ratio",
        "rvs_sentiment",
    ]

    # Three-dimensional Dictionary for: Metric -> Method -> List of Deltas
    all_deltas = {metric: {m: [] for m in method_names} for metric in metrics_to_run}

    logging.info(f"Starting simultaneous evaluation for metrics: {metrics_to_run}")

    csv_file = r"C:\Users\finnb\Documents\emma\Masterthesis\masterarbeit_dataset_systematisch.csv"
    base_scrape_dir = r"C:\Users\finnb\Documents\emma\Masterthesis\scraped_data_full"

    # Read all prompts from the CSV
    prompts = []
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # skip header
            for row in reader:
                if row:
                    prompts.append(row[0])  # User Prompt is in the 1st column
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    for i, query in enumerate(prompts):
        prompt_dir = os.path.join(base_scrape_dir, f"prompt_{i + 1:03d}")
        if not os.path.exists(prompt_dir):
            logging.warning(f"Folder {prompt_dir} does not exist. Skipping...")
            continue

        # Read the 5 sources
        summaries = []
        for j in range(1, 6):
            source_file = os.path.join(prompt_dir, f"source_{j}.txt")
            if os.path.exists(source_file):
                try:
                    with open(source_file, "r", encoding="utf-8") as sf:
                        summaries.append(sf.read())
                except Exception:
                    summaries.append("")
            else:
                summaries.append("")

        # Random selection of a source (seed with Prompt-ID for reproducibility during interruptions)
        random.seed(i)
        valid_indices = [idx for idx, s in enumerate(summaries) if len(s.strip()) > 0]
        if not valid_indices:
            logging.warning(f"No valid sources for prompt {i + 1}. Skipping...")
            continue
        idx = random.choice(valid_indices)

        logging.info(
            f"Processing Prompt {i + 1}/{len(prompts)} (Optimizing random source at index {idx})..."
        )

        # IMPORTANT: We pass 'summaries' explicitly so no live web search is started!
        improvements_dict = improve(
            query,
            idx=idx,
            summaries=summaries,
            sources=summaries,
            metrics=metrics_to_run,
            prompt_id=i + 1,
        )

        for metric in metrics_to_run:
            deltas_for_this_prompt = improvements_dict[metric][:, idx]
            for m_idx, m_name in enumerate(method_names):
                all_deltas[metric][m_name].append(deltas_for_this_prompt[m_idx])

        # if i >= 4: break

    csv_output_file = (
        r"c:\Users\finnb\Documents\emma\Masterthesis\stability_results.csv"
    )
    csv_rows = [
        ["Visibility Metric", "Optimization Method", "Avg Delta", "WCP", "DR", "WTR"]
    ]

    # Print results per metric
    for metric in metrics_to_run:
        print("\n" + "=" * 50)
        print(f"RESULTS FOR METRIC: {metric.upper()}")
        print("=" * 50)

        for m_name in method_names:
            delta_v = np.array(all_deltas[metric][m_name])
            m = len(delta_v)
            if m == 0:
                continue

            wcp = np.min(delta_v)
            dr = np.sum(np.minimum(0, delta_v) ** 2) / m
            wtr = np.sum(delta_v >= 0) / m
            avg_delta = np.mean(delta_v)

            print(f"\nMethod: {m_name}")
            print(f"  Avg Delta: {avg_delta:.4f}")
            print(f"  WCP:       {wcp:.4f}")
            print(f"  DR:        {dr:.4f}")
            print(f"  WTR:       {wtr:.2%}")

            csv_rows.append(
                [
                    metric,
                    m_name,
                    round(avg_delta, 4),
                    round(wcp, 4),
                    round(dr, 4),
                    round(wtr, 4),
                ]
            )

    try:
        with open(csv_output_file, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf, delimiter=";")
            writer.writerows(csv_rows)
        logging.info(f"All results successfully saved to {csv_output_file}.")
    except Exception as e:
        logging.error(f"Failed to save stability_results.csv: {e}")
