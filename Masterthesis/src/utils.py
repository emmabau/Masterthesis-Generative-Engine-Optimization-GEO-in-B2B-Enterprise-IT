"""Module containing the core visibility metrics and evaluation utilities.

This module provides functions to calculate various metrics used in the
Generative Engine Optimization (GEO) project, including Word Count,
Citations, ROUGE, Perplexity, and Relative Visibility Score (RVS).
It also manages caching logic for the Large Language Model (LLM) evaluations.
"""

import itertools
import json
import logging
import math
import os
import re
import threading
import time
import warnings
from glob import glob

import nltk
import requests
import torch
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from generative_le import generate_answer

# Global variables to ensure models are loaded only once and cache is thread-safe
PPL_TOKENIZER = None
PPL_MODEL = None
VADER_ANALYZER = None
SUBJ_CACHE_FILE = None
CACHE_LOCK = threading.Lock()
CACHE_FILE = os.environ.get("GLOBAL_CACHE_FILE", "global_cache.json")


def get_num_words(line):
    """Calculate the number of meaningful words in a given list of tokens.

    Args:
        line (list): A list of word tokens.

    Returns:
        int: The count of words that have a length greater than 2 characters.
    """
    return len([x for x in line if len(x) > 2])


def extract_citations_new(text):
    """Parse a text to extract sentences and their corresponding inline citations.

    Args:
        text (str): The raw text response containing inline citations (e.g., [1][2]).

    Returns:
        list: A nested list structure where each paragraph contains a list of sentences,
            and each sentence is a tuple containing:
            - A list of word tokens.
            - The raw sentence string.
            - A list of integer citation indices found in that sentence.
    """

    def _extract_citation_numbers(sentence):
        citation_pattern = r"\[[^\w\s]*\d+[^\w\s]*\]"
        citations = re.findall(citation_pattern, sentence)
        return [int(re.findall(r"\d+", citation)[0]) for citation in citations]

    paras = re.split(r"\n\n", text)
    sentences = [nltk.sent_tokenize(para) for para in paras]

    words = [
        [(nltk.word_tokenize(s), s, _extract_citation_numbers(s)) for s in sentence]
        for sentence in sentences
    ]
    return words


def impression_rouge(answer, reference_text, n=5, idx=0):
    """Calculate the ROUGE-L recall score between the generated answer and the reference text.

    Args:
        answer (str): The generated response text.
        reference_text (str): The original source text.
        n (int): Total number of sources (default is 5).
        idx (int): The index of the source being evaluated.

    Returns:
        list: A list of length `n` containing the ROUGE score at the specified `idx`
            and 0 elsewhere.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference_text, answer)
    recall = scores["rougeL"].recall

    res = [0] * n
    res[idx] = recall
    return res


def impression_perplexity(answer, reference_text, n=5, idx=0):
    """Calculate the perplexity of the text using a causal language model.

    A lower perplexity indicates better fluency and predictability. The returned
    score is inverted (negative) so that a mathematically higher value represents
    a better perplexity score.

    Args:
        answer (str): Unused in this context. Kept for signature compatibility.
        reference_text (str): The source text to measure perplexity for.
        n (int): Total number of sources.
        idx (int): The index of the source.

    Returns:
        list: A list of length `n` containing the negative perplexity score at
            the specified `idx`.
    """
    global PPL_TOKENIZER, PPL_MODEL

    if PPL_TOKENIZER is None:
        model_id = "Qwen/Qwen-1_8B"
        try:
            PPL_TOKENIZER = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            PPL_MODEL = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", trust_remote_code=True
            )
        except Exception as e:
            logging.warning(
                "Failed to load model %s. Trying Qwen1.5-1.8B... Error: %s", model_id, e
            )
            model_id = "Qwen/Qwen1.5-1.8B"
            PPL_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
            PPL_MODEL = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto"
            )

    inputs = PPL_TOKENIZER(
        reference_text, return_tensors="pt", truncation=True, max_length=32768
    )
    device = PPL_MODEL.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = PPL_MODEL(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = math.exp(loss.item())

    res = [0] * n
    res[idx] = -perplexity
    return res


def impression_chat_cite(sentences, n=5, normalize=False):
    """Evaluate ChatCite (Document-level citation).

    Assign a score of 1 if the source is cited anywhere in the document, otherwise 0.

    Args:
        sentences (list): Nested list of sentences generated by extract_citations_new.
        n (int): Total number of sources.
        normalize (bool): Unused parameter, kept for signature consistency.

    Returns:
        list: A list of binary scores (1 or 0) for each source.
    """
    sentences_flat = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for sent in sentences_flat:
        for cit in sent[2]:
            try:
                scores[cit - 1] = 1
            except IndexError:
                pass
    return scores


def impression_sentence_cite(sentences, n=5, normalize=False):
    """Evaluate SentenceCite (Sentence-level citation).

    Count the exact number of sentences that cite a given source.

    Args:
        sentences (list): Nested list of sentences generated by extract_citations_new.
        n (int): Total number of sources.
        normalize (bool): Unused parameter, kept for signature consistency.

    Returns:
        list: A list of integer counts representing the number of citing sentences per source.
    """
    sentences_flat = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]
    for sent in sentences_flat:
        for cit in set(sent[2]):
            try:
                scores[cit - 1] += 1
            except IndexError:
                pass
    return scores


def impression_bleu(answer, reference_text, n=5, idx=0):
    """Calculate the Sentence BLEU score between the generated answer and the source text.

    Args:
        answer (str): The generated response text.
        reference_text (str): The original source text.
        n (int): Total number of sources.
        idx (int): The index of the source.

    Returns:
        list: A list of length `n` containing the BLEU score at the specified `idx`.
    """
    warnings.filterwarnings("ignore")

    ref_tokens = reference_text.split()
    gen_tokens = answer.split()

    bleu_score = sentence_bleu([ref_tokens], gen_tokens)

    res = [0] * n
    res[idx] = bleu_score
    return res


def impression_length_ratio(answer, reference_text, n=5, idx=0):
    """Calculate the word length ratio of the generated text compared to the source text.

    Args:
        answer (str): The generated response text.
        reference_text (str): The original source text.
        n (int): Total number of sources.
        idx (int): The index of the source.

    Returns:
        list: A list of length `n` containing the Length Ratio at the specified `idx`.
    """
    ref_tokens = reference_text.split()
    gen_tokens = answer.split()

    if ref_tokens:
        length_ratio = len(gen_tokens) / len(ref_tokens)
    else:
        length_ratio = 0.0

    res = [0] * n
    res[idx] = length_ratio
    return res


def impression_rvs(sentences, n=5, normalize=False):
    """Calculate the Relative Visibility Score (RVS) with Sentiment Analysis.

    Based on the formula: RVS = Word Count * Position Weight * Sentiment Score.

    Args:
        sentences (list): Nested list of sentences generated by extract_citations_new.
        n (int): Total number of sources.
        normalize (bool): Unused parameter.

    Returns:
        list: A list of computed RVS scores for each source.
    """
    global VADER_ANALYZER
    if VADER_ANALYZER is None:
        VADER_ANALYZER = SentimentIntensityAnalyzer()

    sentences_flat = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]

    for i, sent in enumerate(sentences_flat):
        sentiment_score = VADER_ANALYZER.polarity_scores(sent[1])["compound"]

        for cit in sent[2]:
            word_count = get_num_words(sent[0])
            position_weight = (
                math.exp(-1 * i / (len(sentences_flat) - 1))
                if len(sentences_flat) > 1
                else 1
            )
            rvs = word_count * position_weight * sentiment_score

            try:
                scores[cit - 1] += rvs
            except IndexError:
                pass

    return scores


def position_adjusted_wordcount(sentences, n=5, normalize=True):
    """Calculate a position-adjusted word count for each cited source.

    Words in earlier sentences receive a higher weight based on an exponential decay function.

    Args:
        sentences (list): Nested list of sentences generated by extract_citations_new.
        n (int): Total number of sources.
        normalize (bool): If True, normalize the scores to sum to 1.

    Returns:
        list: A list of position-adjusted visibility scores for each source.
    """
    sentences_flat = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]

    for i, sent in enumerate(sentences_flat):
        for cit in sent[2]:
            score = get_num_words(sent[0])
            score *= (
                math.exp(-1 * i / (len(sentences_flat) - 1))
                if len(sentences_flat) > 1
                else 1
            )
            score /= len(sent[2])

            try:
                scores[cit - 1] += score
            except IndexError:
                logging.warning("Citation Hallucinated: %s", cit)

    if normalize:
        total_score = sum(scores)
        if total_score != 0:
            return [x / total_score for x in scores]
        return [1 / n for _ in range(n)]
    return scores


def absolute_wordcount(sentences, n=5, normalize=True):
    """Calculate the absolute word count attributed to each source based on inline citations.

    Args:
        sentences (list): Nested list of sentences generated by extract_citations_new.
        n (int): Total number of sources.
        normalize (bool): If True, normalize the scores to sum to 1.

    Returns:
        list: A list of absolute word count visibility scores for each source.
    """
    sentences_flat = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]

    for sent in sentences_flat:
        for cit in sent[2]:
            score = get_num_words(sent[0])
            score /= len(sent[2])
            try:
                scores[cit - 1] += score
            except IndexError:
                logging.warning("Citation Hallucinated: %s", cit)

    if normalize:
        total_score = sum(scores)
        if total_score != 0:
            return [x / total_score for x in scores]
        return [1 / n for _ in range(n)]
    return scores


def impression_pos_count_simple(sentences, n=5, normalize=True):
    """Calculate a simplified position-adjusted score based solely on sentence counts.

    This function ignores the actual word counts in the sentences.

    Args:
        sentences (list): Nested list of sentences generated by extract_citations_new.
        n (int): Total number of sources.
        normalize (bool): If True, normalize the scores to sum to 1.

    Returns:
        list: A list of simplified position-adjusted visibility scores.
    """
    sentences_flat = list(itertools.chain(*sentences))
    scores = [0 for _ in range(n)]

    for i, sent in enumerate(sentences_flat):
        for cit in sent[2]:
            score = 1.0
            score *= (
                math.exp(-1 * i / (len(sentences_flat) - 1))
                if len(sentences_flat) > 1
                else 1
            )
            score /= len(sent[2])
            try:
                scores[cit - 1] += score
            except IndexError:
                logging.warning("Citation Hallucinated: %s", cit)

    if normalize:
        total_score = sum(scores)
        if total_score != 0:
            return [x / total_score for x in scores]
        return [1 / n for _ in range(n)]
    return scores


def impression_subjpos_detailed(sentences, query, n=5, normalize=True, idx=0):
    """Evaluate the Subjective Position metric (GEval).

    Args:
        sentences (list): The generated text sentences.
        query (str): The original user query.
        n (int): Total number of sources.
        normalize (bool): Whether to normalize the result.
        idx (int): Index of the target source.

    Returns:
        list: Computed subjective scores.
    """
    return impression_subjective_impression(
        sentences, query, n=n, normalize=normalize, idx=idx, metric="subjpos_detailed"
    )


def impression_diversity_detailed(sentences, query, n=5, normalize=True, idx=0):
    """Evaluate the Subjective Diversity metric (GEval).

    Args:
        sentences (list): The generated text sentences.
        query (str): The original user query.
        n (int): Total number of sources.
        normalize (bool): Whether to normalize the result.
        idx (int): Index of the target source.

    Returns:
        list: Computed subjective scores.
    """
    return impression_subjective_impression(
        sentences, query, n=n, normalize=normalize, idx=idx, metric="diversity_detailed"
    )


def impression_uniqueness_detailed(sentences, query, n=5, normalize=True, idx=0):
    """Evaluate the Subjective Uniqueness metric (GEval).

    Args:
        sentences (list): The generated text sentences.
        query (str): The original user query.
        n (int): Total number of sources.
        normalize (bool): Whether to normalize the result.
        idx (int): Index of the target source.

    Returns:
        list: Computed subjective scores.
    """
    return impression_subjective_impression(
        sentences,
        query,
        n=n,
        normalize=normalize,
        idx=idx,
        metric="uniqueness_detailed",
    )


def impression_follow_detailed(sentences, query, n=5, normalize=True, idx=0):
    """Evaluate the Subjective Follow/Fluency metric (GEval).

    Args:
        sentences (list): The generated text sentences.
        query (str): The original user query.
        n (int): Total number of sources.
        normalize (bool): Whether to normalize the result.
        idx (int): Index of the target source.

    Returns:
        list: Computed subjective scores.
    """
    return impression_subjective_impression(
        sentences, query, n=n, normalize=normalize, idx=idx, metric="follow_detailed"
    )


def impression_influence_detailed(sentences, query, n=5, normalize=True, idx=0):
    """Evaluate the Subjective Influence metric (GEval).

    Args:
        sentences (list): The generated text sentences.
        query (str): The original user query.
        n (int): Total number of sources.
        normalize (bool): Whether to normalize the result.
        idx (int): Index of the target source.

    Returns:
        list: Computed subjective scores.
    """
    return impression_subjective_impression(
        sentences, query, n=n, normalize=normalize, idx=idx, metric="influence_detailed"
    )


def impression_relevance_detailed(sentences, query, n=5, normalize=True, idx=0):
    """Evaluate the Subjective Relevance metric (GEval).

    Args:
        sentences (list): The generated text sentences.
        query (str): The original user query.
        n (int): Total number of sources.
        normalize (bool): Whether to normalize the result.
        idx (int): Index of the target source.

    Returns:
        list: Computed subjective scores.
    """
    return impression_subjective_impression(
        sentences, query, n=n, normalize=normalize, idx=idx, metric="relevance_detailed"
    )


def impression_subjcount_detailed(sentences, query, n=5, normalize=True, idx=0):
    """Evaluate the Subjective Citation Count metric (GEval).

    Args:
        sentences (list): The generated text sentences.
        query (str): The original user query.
        n (int): Total number of sources.
        normalize (bool): Whether to normalize the result.
        idx (int): Index of the target source.

    Returns:
        list: Computed subjective scores.
    """
    return impression_subjective_impression(
        sentences, query, n=n, normalize=normalize, idx=idx, metric="subjcount_detailed"
    )


def impression_subjective_impression(
    sentences, query, n=5, normalize=True, idx=0, metric="subjective_impression"
):
    """Evaluate GPT-Eval subjective metrics using a local language model.

    This function processes a set of GEval prompt templates, queries a local model
    (qwen2.5:7b) to score the text from 1 to 5, and parses the logprobs to compute
    an expected continuous score. Thread-safe caching is implemented to avoid
    re-computing results.

    Args:
        sentences (str): The text to evaluate.
        query (str): The original user query.
        n (int): Total number of sources.
        normalize (bool): Whether to normalize the result (unused internally).
        idx (int): The index of the target source.
        metric (str): The specific GEval metric to extract from the computed dictionary.

    Returns:
        list: A list of length `n` containing the GEval score at the specified `idx`.
    """

    def _returnable_score_from_scores(local_scores):
        avg_score = sum(local_scores.values()) / len(local_scores.values())
        if metric != "subjective_impression":
            avg_score = local_scores[metric]
        return [avg_score if _ == idx else 0 for _ in range(n)]

    global SUBJ_CACHE_FILE
    cache_file = "gpt-eval-scores-cache_new-new.json"

    with CACHE_LOCK:
        if os.environ.get("SUBJ_STATIC_CACHE", None) is not None:
            if SUBJ_CACHE_FILE is None:
                with open(cache_file, encoding="utf-8") as f:
                    SUBJ_CACHE_FILE = json.load(f)
        else:
            if os.path.exists(cache_file):
                with open(cache_file, encoding="utf-8") as f:
                    SUBJ_CACHE_FILE = json.load(f)
            else:
                SUBJ_CACHE_FILE = {}
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(SUBJ_CACHE_FILE, f, indent=2)

    cache = SUBJ_CACHE_FILE
    cache_key = str((sentences, query))

    if cache_key in cache:
        if str(idx) in cache[cache_key]:
            return _returnable_score_from_scores(cache[cache_key][str(idx)])

    scores = {}
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    geval_dir = os.path.join(
        base_dir, "03_Visibility Metrics and Results", "geval_prompts"
    )

    for prompt_file in glob(os.path.join(geval_dir, "*.txt")):
        with open(prompt_file, encoding="utf-8") as f:
            prompt = f.read()

        prompt = prompt.replace("[1]", f"[{idx + 1}]")
        safe_sentences = sentences[:3000] + ("..." if len(sentences) > 3000 else "")
        cur_prompt = prompt.format(query=query, answer=safe_sentences)

        while True:
            try:
                payload = {
                    "model": "qwen2.5:7b",
                    "messages": [{"role": "user", "content": cur_prompt}],
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "logprobs": True,
                    "top_logprobs": 5,
                }

                resp = requests.post(
                    "http://localhost:11435/v1/chat/completions",
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

                top_logprobs = data["choices"][0]["logprobs"]["content"][0][
                    "top_logprobs"
                ]
                logprobs_dict = {
                    item["token"]: item["logprob"] for item in top_logprobs
                }
                valid_logprobs = {
                    k: v
                    for k, v in logprobs_dict.items()
                    if k.strip().isdigit() and 1 <= int(k.strip()) <= 5
                }

                if not valid_logprobs:
                    avg_score = 1.0
                else:
                    total_sum = sum([math.exp(v) for v in valid_logprobs.values()])
                    avg_score = sum(
                        [
                            float(k.strip()) * math.exp(v) / total_sum
                            for k, v in valid_logprobs.items()
                        ]
                    )

                metric_name = os.path.split(prompt_file)[-1].split(".")[0]
                scores[metric_name] = avg_score
                break
            except Exception as e:
                logging.error("Error in GPT-Eval: %s", e)
                time.sleep(10)

    with CACHE_LOCK:
        with open(cache_file, encoding="utf-8") as f:
            cache = json.load(f)
        if cache_key not in cache:
            cache[cache_key] = {}
        cache[cache_key][str(idx)] = scores
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    return _returnable_score_from_scores(scores)


def check_summaries_exist(sources, summaries):
    """Check if a given set of summaries exists in the cache for a specific query.

    Args:
        sources (list): The list of cached source objects for a query.
        summaries (list): The new list of summaries to check against.

    Returns:
        dict or None: The matching source object if found, otherwise None.
    """
    for source in sources:
        cached_summaries = [x["summary"] for x in source["sources"]]
        if cached_summaries == summaries:
            return source
    return None


def get_answer(
    query,
    summaries=None,
    n=5,
    num_completions=1,
    cache_idx=0,
    regenerate_answer=False,
    write_to_cache=True,
    loaded_cache=None,
    target_cache_file=None,
):
    """Retrieve or generate an answer for a query based on a list of summaries.

    This function accesses a local JSON cache to prevent regenerating answers
    for identical prompts and summaries. If the combination is new or forced,
    it delegates generation to the internal generation module.

    Args:
        query (str): The user's query.
        summaries (list, optional): The list of text summaries. If None, expects them in cache.
        n (int): Number of sources (default is 5).
        num_completions (int): How many completions to generate (default is 1).
        cache_idx (int): Which cache entry to use if multiple exist.
        regenerate_answer (bool): Force generation even if found in cache.
        write_to_cache (bool): Whether to save the new answer to disk.
        loaded_cache (dict, optional): An in-memory cache dictionary.
        target_cache_file (str, optional): A specific file path to save the cache.

    Returns:
        dict: A dictionary containing the sources and the generated responses.

    Raises:
        Exception: If summaries are not provided and the query is missing from the cache.
    """
    if loaded_cache is None:
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = loaded_cache

    if summaries is None:
        if cache.get(query) is None:
            raise Exception("Live search via search_handler disabled.")
        search_results = cache[query][cache_idx]
        summaries = [x["summary"] for x in search_results["sources"]]

    cached_source = check_summaries_exist(cache.get(query, []), summaries)

    if not regenerate_answer and cached_source is not None:
        if cached_source["responses"]:
            return cached_source
        answers = generate_answer(query, summaries, num_completions=num_completions)
    else:
        answers = generate_answer(query, summaries, num_completions=num_completions)

    ret_value = None

    if loaded_cache is None:
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = loaded_cache

    if cache.get(query) is None:
        if summaries is None:
            cache[query] = [
                {"sources": search_results["sources"], "responses": [answers]}
            ]
        else:
            cache[query] = [
                {"sources": [{"summary": x} for x in summaries], "responses": [answers]}
            ]
    else:
        flag = False
        for source in cache[query]:
            cached_summaries = [x["summary"] for x in source["sources"]]
            if cached_summaries == summaries:
                source["responses"].append(answers)
                ret_value = source
                flag = True
                break
        if not flag:
            if summaries is None:
                cache[query].append(
                    {"sources": search_results["sources"], "responses": [answers]}
                )
            else:
                sources_list = [
                    {"summary": x, "source": y}
                    for x, y in zip(summaries, cache[query][0]["sources"])
                ]
                cache[query].append({"sources": sources_list, "responses": [answers]})

    if write_to_cache:
        save_file = target_cache_file if target_cache_file is not None else CACHE_FILE
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    return ret_value if ret_value is not None else cache[query][-1]
