"""Module providing the core Generative Engine Optimization (GEO) functions.

This module defines methods to modify and optimize source texts before they
are processed by a language model. It includes the system prompts, logic to
call the local Ollama API for text rewriting, and specific GEO techniques
such as authoritative styling, fluency improvements, and structural changes.
"""

import json
import os
import pickle
import uuid

import openai

CACHE_FILE = os.environ.get("GEO_CACHE_FILE", "geo_optimizations_cache.json")

COMMON_SYSTEM_PROMPT = (
    "You are an expert ml researcher having previous background in SEO and search engines "
    "in general. You are working on novel research ideas for next generation of products. "
    "These products will have language models augmented with search engines, with the task "
    "of answering questions based on sources backed by the search engine. This new set of "
    "systems will be collectively called language engines (generative search engines). "
    "This will require websites to update their SEO techniques to rank higher in the llm "
    "generated answer. Specifically they will use GEO (Generative Engine Optimization) "
    "techniques to boost their visibility in the final text answer outputted by the Language Engine."
)

GLOBAL_CACHE = None


def call_gpt(
    user_prompt,
    system_prompt=COMMON_SYSTEM_PROMPT,
    model="qwen3:8b",
    temperature=0.0,
    num_completions=1,
    regenerate_answer=False,
    pre_msgs=None,
    method_name="default",
    prompt_id=None,
    source_idx=None,
):
    """Call the local Ollama language model to generate optimized text.

    Args:
        user_prompt (str): The specific rewrite instruction and source text.
        system_prompt (str): The system persona instruction.
        model (str): The name of the local Ollama model to use.
        temperature (float): The generation temperature.
        num_completions (int): Number of generations requested.
        regenerate_answer (bool): Force answer regeneration ignoring cache.
        pre_msgs (list, optional): Previous conversation context.
        method_name (str): The name of the GEO method being applied.
        prompt_id (int, optional): The ID of the original query prompt.
        source_idx (int, optional): The index of the source within the query.

    Returns:
        str: The generated optimized text from the model.
    """
    global GLOBAL_CACHE

    if method_name != "default":
        cache_file = f"geo_optimizations_cache_{method_name}_{model}.json"
    else:
        cache_file = CACHE_FILE.replace(".json", f"_{model}.json")

    cache_file = cache_file.replace(":", "-")

    if os.environ.get("STATIC_CACHE", None) == "True":
        if GLOBAL_CACHE is None:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    GLOBAL_CACHE = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                GLOBAL_CACHE = {}
        cache = GLOBAL_CACHE
    else:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            cache = {}

    if prompt_id is not None and source_idx is not None:
        cache_key = f"Prompt_{prompt_id:03d}_Source_{source_idx + 1}"
    else:
        cache_key = str((user_prompt, system_prompt))

    if cache_key in cache and not regenerate_answer:
        return cache[cache_key][-1]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if pre_msgs is not None:
        messages = [messages[0]] + pre_msgs + messages[1:]

    openai.api_base = "http://localhost:11435/v1"
    openai.api_key = "ollama"

    def _get_summary(tex):
        tex = tex.replace("```\n```", "```")
        b = tex.rfind("```")
        if b != -1:
            if tex.count("```") < 2:
                a = b + 3
                b = -1
            else:
                a = tex[:b].rfind("```") + 3
        else:
            a = -1

        if b != -1 and (b - a < 50):
            a = b if len(tex) - b > 200 else a
            b = -1

        if a <= 2:
            a = 0

        if b != -1:
            new_tex = tex[a:b].strip()
        else:
            new_tex = tex[a:].strip()

        if new_tex.lower().startswith("updated"):
            new_tex = "\n".join(new_tex.splitlines()[1:])

        if len(new_tex) == 0:
            return tex
        return new_tex

    attempt = 0
    current_temp = temperature
    while True:
        attempt += 1
        try:
            responses = openai.ChatCompletion.create(
                model=model,
                temperature=current_temp,
                max_tokens=3192,
                messages=messages,
                top_p=1,
                n=num_completions,
            )

            if num_completions == 1:
                extracted = _get_summary(responses.choices[0].message.content)
                if len(extracted.split()) <= 10:
                    current_temp = min(1.0, current_temp + 0.3)
                    if attempt > 4:
                        break
                    import time

                    time.sleep(2)
                    continue
            break
        except Exception as e:
            error_str = str(e)
            if "maximum context length" in error_str:
                try:
                    a = error_str.find("messages resulted in ") + len(
                        "messages resulted in "
                    )
                    b = error_str.find(" tokens", a)
                    num_tokens_excess = 2000 / int(error_str[a:b])
                except ValueError:
                    a = error_str.find("you requested ") + len("you requested ")
                    b = error_str.find(" tokens", a)
                    num_tokens_excess = 2000 / int(error_str[a:b])

                lup = len(messages[-1]["content"])
                messages[-1]["content"] = messages[-1]["content"][
                    : int(lup * num_tokens_excess)
                ]

            if attempt > 5:
                messages[0]["content"] = messages[0]["content"][:-200]
                messages[-1]["content"] = messages[-1]["content"][:-1000]

            import time

            time.sleep(15)
            continue

    os.makedirs("response_usages_16k", exist_ok=True)
    with open(f"response_usages_16k/{uuid.uuid4()}.pkl", "wb") as f:
        pickle.dump(responses.usage, f)

    if GLOBAL_CACHE is None:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            cache = {}

    if cache_key not in cache:
        cache[cache_key] = []

    cache[cache_key].extend(
        [_get_summary(x.message.content) for x in responses.choices]
    )
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    if os.environ.get("STATIC_CACHE", None) == "True":
        GLOBAL_CACHE = cache

    return cache[cache_key][-1]


def apply_geo_rule(
    summary, rule, method_name="default", prompt_id=None, source_idx=None
):
    """Apply a specific GEO rule to a given summary text.

    Construct a prompt with the optimization rule and call the language model.
    Save the generated optimized text to disk for future evaluation.

    Args:
        summary (str): The original text summary to be optimized.
        rule (str): The specific GEO optimization instruction.
        method_name (str): The identifier for the optimization method.
        prompt_id (int, optional): The prompt ID for file saving.
        source_idx (int, optional): The source index for file saving.

    Returns:
        str: The fully optimized and parsed text.
    """
    user_prompt = f"""Here is the source that you need to update:
```
{summary}
```

You are given a website document as a source. This source, along with other sources, will be used by a language model (LLM) to generate answers to user questions. Your task, as the owner of the source, is to **rewrite your document in a way that maximizes its visibility and impact in the LLM's final answer, ensuring your source is more likely to be quoted and cited**.

You must strictly adhere to the following Quality Guideline:

## Quality Guidelines to Follow:
- {rule}

Please output ONLY the final rewritten text inside triple backticks.

Updated Output:
```
<Output>
```""".strip()

    optimized_text = call_gpt(
        user_prompt, method_name=method_name, prompt_id=prompt_id, source_idx=source_idx
    )

    if prompt_id is not None and source_idx is not None and method_name != "default":
        method_folder = rf"C:\Users\finnb\Documents\emma\Masterthesis\02_Benchmark\AI Answers - with GEO optimization\optimized_data_{method_name}"
        prompt_folder = os.path.join(method_folder, f"prompt_{prompt_id:03d}")
        os.makedirs(prompt_folder, exist_ok=True)
        file_path = os.path.join(prompt_folder, f"source_{source_idx + 1}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(optimized_text)

    return optimized_text


def fluent_optimization_gpt(summary, prompt_id=None, source_idx=None):
    """Apply the 'Fluent' GEO method to improve clarity.

    Args:
        summary (str): Original text.
        prompt_id (int, optional): Prompt identifier.
        source_idx (int, optional): Source identifier.

    Returns:
        str: Fluent optimized text.
    """
    rule = "Rewrite the text to make it extremely fluent, clear, and easy to understand without altering the core content."
    return apply_geo_rule(
        summary,
        rule,
        method_name="fluent_gpt",
        prompt_id=prompt_id,
        source_idx=source_idx,
    )


def authoritative(summary, prompt_id=None, source_idx=None):
    """Apply the 'Authoritative' GEO method for expert styling.

    Args:
        summary (str): Original text.
        prompt_id (int, optional): Prompt identifier.
        source_idx (int, optional): Source identifier.

    Returns:
        str: Authoritative optimized text.
    """
    rule = "Transform the text into a highly authoritative, confident, and expert style that asserts this source as the most valuable and authentic information available."
    return apply_geo_rule(
        summary,
        rule,
        method_name="authoritative",
        prompt_id=prompt_id,
        source_idx=source_idx,
    )


def more_quotes(summary, prompt_id=None, source_idx=None):
    """Apply the 'Quotes' GEO method to include direct quotes.

    Args:
        summary (str): Original text.
        prompt_id (int, optional): Prompt identifier.
        source_idx (int, optional): Source identifier.

    Returns:
        str: Quote optimized text.
    """
    rule = "Add relevant direct quotes from authoritative figures to increase credibility and influence."
    return apply_geo_rule(
        summary,
        rule,
        method_name="more_quotes",
        prompt_id=prompt_id,
        source_idx=source_idx,
    )


def citing_credible(summary, prompt_id=None, source_idx=None):
    """Apply the 'Cite Credible Sources' GEO method.

    Args:
        summary (str): Original text.
        prompt_id (int, optional): Prompt identifier.
        source_idx (int, optional): Source identifier.

    Returns:
        str: Source-cited optimized text.
    """
    rule = "Naturally include citations from credible sources (e.g., 'According to...') to substantiate claims without altering the core content."
    return apply_geo_rule(
        summary,
        rule,
        method_name="citing_credible",
        prompt_id=prompt_id,
        source_idx=source_idx,
    )


def simple_language(summary, prompt_id=None, source_idx=None):
    """Apply the 'Simple Language' GEO method.

    Args:
        summary (str): Original text.
        prompt_id (int, optional): Prompt identifier.
        source_idx (int, optional): Source identifier.

    Returns:
        str: Simplified optimized text.
    """
    rule = "Simplify the language to make it easy to understand, while ensuring all key information is still conveyed."
    return apply_geo_rule(
        summary,
        rule,
        method_name="simple_language",
        prompt_id=prompt_id,
        source_idx=source_idx,
    )


def technical_terms(summary, prompt_id=None, source_idx=None):
    """Apply the 'Technical Terms' GEO method.

    Args:
        summary (str): Original text.
        prompt_id (int, optional): Prompt identifier.
        source_idx (int, optional): Source identifier.

    Returns:
        str: Technically enriched optimized text.
    """
    rule = "Make the text more technical by incorporating specific technical terms and facts where appropriate."
    return apply_geo_rule(
        summary,
        rule,
        method_name="technical_terms",
        prompt_id=prompt_id,
        source_idx=source_idx,
    )


def stats_optimization(summary, prompt_id=None, source_idx=None):
    """Apply the 'Statistics' GEO method to add facts inline.

    Args:
        summary (str): Original text.
        prompt_id (int, optional): Prompt identifier.
        source_idx (int, optional): Source identifier.

    Returns:
        str: Statistic enriched optimized text.
    """
    rule = "Substantiate claims by subtly adding positive, compelling statistics, objective facts, and exact numbers inline."
    return apply_geo_rule(
        summary,
        rule,
        method_name="stats_optimization",
        prompt_id=prompt_id,
        source_idx=source_idx,
    )


def llms_txt(summary, prompt_id=None, source_idx=None):
    """Apply the 'LLMs.txt' GEO method for structural markdown formatting.

    Args:
        summary (str): Original text.
        prompt_id (int, optional): Prompt identifier.
        source_idx (int, optional): Source identifier.

    Returns:
        str: Markdown structured optimized text.
    """
    rule = (
        "Create a llms.txt markdown file to provide LLM-friendly content. This file summarizes the main "
        "text and offers brief background information, guidance, and links (if available). Follow this template: \n"
        "# Title \n"
        "> Introduction paragraph \n"
        "Optional details go here \n"
        "## Section name \n"
        "More details"
    )
    return apply_geo_rule(
        summary,
        rule,
        method_name="llms_txt",
        prompt_id=prompt_id,
        source_idx=source_idx,
    )


def inverted_pyramid_mine(summary, prompt_id=None, source_idx=None):
    """Apply the 'Inverted Pyramid' GEO method for BLUF structuring.

    Args:
        summary (str): Original text.
        prompt_id (int, optional): Prompt identifier.
        source_idx (int, optional): Source identifier.

    Returns:
        str: Inverted pyramid optimized text.
    """
    rule = (
        "Rewrite the text using the 'Inverted Pyramid' principle (Bottom Line Up Front). "
        "Begin with a structured abstract (30-70 words) that directly answers the core question, "
        "followed by a Q&A structure using H2/H3 headers for the remaining content."
    )
    return apply_geo_rule(
        summary,
        rule,
        method_name="inverted_pyramid_mine",
        prompt_id=prompt_id,
        source_idx=source_idx,
    )


def autogeo_api_mine(summary, prompt_id=None, source_idx=None):
    """Apply the 'AutoGEO' (All-in-One) optimization method.

    Aggregate multiple GEO techniques (fluency, statistics, inverted pyramid, etc.)
    into a single comprehensive prompt to produce heavily optimized text.

    Args:
        summary (str): Original text.
        prompt_id (int, optional): Prompt identifier.
        source_idx (int, optional): Source identifier.

    Returns:
        str: Comprehensive optimized text.
    """
    custom_rules = [
        "Naturally include citations from credible sources (e.g., 'According to...') to substantiate claims without altering the core content.",
        "Transform the text into a highly authoritative, confident, and expert style that asserts this source as the most valuable and authentic information available.",
        "Rewrite the text to make it extremely fluent, clear, and easy to understand without altering the core content.",
        "Add relevant direct quotes from authoritative figures to increase credibility and influence.",
        "Substantiate claims by subtly adding positive, compelling statistics, objective facts, and exact numbers inline.",
        "Simplify the language to make it easy to understand, while ensuring all key information is still conveyed.",
        "Make the text more technical by incorporating specific technical terms and facts where appropriate.",
        "Structure the document to be highly LLM-friendly (acting like an llms.txt summary), providing clear background information, guidance, and logical flow.",
        "Rewrite the text using the 'Inverted Pyramid' principle (Bottom Line Up Front). Begin with a structured abstract (30-70 words) that directly answers the core question, followed by a Q&A structure using H2/H3 headers for the remaining content.",
    ]
    rules_string = "- " + "\n- ".join(custom_rules)
    user_prompt = f"""Here is the source that you need to update:
```
{summary}
```

You are given a website document as a source. This source, along with other sources, will be used by a language model (LLM) to generate answers to user questions. Your task, as the owner of the source, is to **rewrite your document in a way that maximizes its visibility and impact in the LLM's final answer, ensuring your source is more likely to be quoted and cited**.

You must strictly adhere to all of the following Quality Guidelines, which combine various optimization strategies into one comprehensive rewrite.

## Quality Guidelines to Follow:
{rules_string}

Please output ONLY the final rewritten text inside triple backticks.

Updated Output:
```
<Output>
```""".strip()

    optimized_text = call_gpt(
        user_prompt,
        method_name="autogeo_api_mine",
        prompt_id=prompt_id,
        source_idx=source_idx,
    )

    if prompt_id is not None and source_idx is not None:
        method_folder = r"C:\Users\finnb\Documents\emma\Masterthesis\02_Benchmark\AI Answers - with GEO optimization\optimized_data_autogeo_api"
        prompt_folder = os.path.join(method_folder, f"prompt_{prompt_id:03d}")
        os.makedirs(prompt_folder, exist_ok=True)
        file_path = os.path.join(prompt_folder, f"source_{source_idx + 1}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(optimized_text)

    return optimized_text
