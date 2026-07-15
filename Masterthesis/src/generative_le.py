"""Module for querying local language models to generate baseline answers.

This module provides the core interface for interacting with a local Ollama
Language Model via an OpenAI-compatible API endpoint. It generates the baseline
answers for user queries based on a provided set of web search summaries, enforcing
strict citation rules.
"""

import os
import pickle
import time
import uuid

import openai

QUERY_PROMPT = (
    "Write an accurate and concise answer for the given user question, using _only_ "
    "the provided summarized web search results. The answer should be correct, high-quality, "
    "and written by an expert using an unbiased and journalistic tone. The user's language "
    "of choice such as English, Français, Español, Deutsch, or 日本語 should be used. "
    "The answer should be informative, interesting, and engaging. The answer's logic and "
    "reasoning should be rigorous and defensible. Every sentence in the answer should be "
    "_immediately followed_ by an in-line citation to the search result(s). The cited search "
    "result(s) should fully support _all_ the information in the sentence. Search results "
    "need to be cited using [index]. When citing several search results, use [1][2][3] "
    "format rather than [1, 2, 3]. You can use multiple search results to respond "
    "comprehensively while avoiding irrelevant search results.\n\n"
    "Question: {query}\n\n"
    "Search Results:\n"
    "{source_text}\n"
)


def generate_answer(
    query, sources, num_completions, temperature=0.5, verbose=False, model="qwen3:8b"
):
    """Generate an answer to a query using the local language model.

    Format the input sources, construct the prompt, and query the local LLM endpoint
    to retrieve one or more completions based on the provided parameters.

    Args:
        query (str): The user's search query.
        sources (list): A list of text summaries representing search results.
        num_completions (int): The number of alternative answers to generate.
        temperature (float): The sampling temperature (default is 0.5).
        verbose (bool): Unused parameter, maintained for signature consistency.
        model (str): The specific LLM model to request from the endpoint.

    Returns:
        list: A list of generated answer strings.
    """
    openai.api_base = "http://localhost:11435/v1"
    openai.api_key = "ollama"

    source_text = "\n\n".join(
        [f"### Source {idx + 1}:\n{source}\n\n\n" for idx, source in enumerate(sources)]
    )
    prompt = QUERY_PROMPT.format(query=query, source_text=source_text)

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                temperature=temperature,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                top_p=1,
                n=num_completions,
            )
            break
        except Exception as e:
            # Handle potential connection or timeout issues by sleeping and retrying
            import logging

            logging.error("Error in calling OpenAI API: %s", e)
            time.sleep(15)
            continue

    os.makedirs("response_usages_16k", exist_ok=True)
    with open(f"response_usages_16k/{uuid.uuid4()}.pkl", "wb") as f:
        pickle.dump(response.usage, f)

    return [x.message.content + "\n" for x in response.choices]
