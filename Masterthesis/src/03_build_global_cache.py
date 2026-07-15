"""Constructs a global cache and benchmark dataset from scraped web content.

This module aggregates the text content previously scraped for user queries.
It truncates and cleans the text to comply with typical language model
context window constraints, and generates a structured JSON cache alongside
a benchmark CSV file utilized in Generative Engine Optimization (GEO) experiments.
"""

import csv
import json
import os

INPUT_DIR = "scraped_data_full"
OUTPUT_CACHE_FILE = os.path.join("GEO", "global_cache.json")
OUTPUT_CSV_FILE = os.path.join("GEO", "my_geo_bench.csv")


def build_cache():
    """Aggregate scraped data into a global JSON cache and a benchmark CSV file.

    Iterates through the directory structure containing scraped documents and
    metadata, cleans and truncates the textual data, and maps them to their
    respective user queries. Writes the consolidated output to designated files.
    """
    global_cache = {}
    csv_data = []

    for i in range(1, 601):
        prompt_dir = os.path.join(INPUT_DIR, f"prompt_{i:03d}")
        metadata_file = os.path.join(prompt_dir, "metadata.json")

        if not os.path.exists(metadata_file):
            print(f"Skipping {prompt_dir} - no metadata.json found")
            continue

        with open(metadata_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error reading {metadata_file}: {e}")
                continue

        prompt_text = data.get("prompt", "")
        if not prompt_text:
            continue

        sources_list = []
        for source_meta in data.get("sources", []):
            filename = source_meta.get("filename")
            url = source_meta.get("url", "")
            title = source_meta.get("title", "")

            filepath = os.path.join(prompt_dir, filename)
            if not os.path.exists(filepath):
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()

            # Truncate text to 8000 characters to fit standard context windows
            clean_text = (
                raw_text.strip()
                .replace("\n\n\n", "\n\n")
                .replace("\n\n", " ")
                .replace("  ", " ")
                .replace("\t", "")
                .replace("\n", "")
            )
            truncated_text = clean_text[:8000]

            formatted_text = f"Title: {title}\nSummary:{truncated_text}"

            sources_list.append(
                {
                    "url": url,
                    "text": formatted_text,
                    "raw_source": raw_text,
                    "source": truncated_text,
                    "summary": truncated_text,
                }
            )

        if sources_list:
            global_cache[prompt_text] = [{"sources": sources_list, "responses": []}]

            csv_data.append(
                {
                    "id": i,
                    "query": prompt_text,
                    "sugg_idx": 0,  # Default target to optimize in the GEO experiment is the first source
                }
            )

    # Ensure GEO directory exists
    os.makedirs("GEO", exist_ok=True)

    # Write global_cache.json
    print("Writing global_cache.json...")
    with open(OUTPUT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(global_cache, f, indent=2)

    # Write CSV
    print("Writing my_geo_bench.csv...")
    with open(OUTPUT_CSV_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "query", "sugg_idx"])
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"Successfully processed {len(global_cache)} prompts.")
    print(f"Created {OUTPUT_CACHE_FILE} and {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
    build_cache()
