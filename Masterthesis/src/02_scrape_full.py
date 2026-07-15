"""Retrieves and processes search engine results for generated user queries.

This module automates the execution of user queries using the Serper API,
extracts URLs from the search results, and scrapes the text content of those
URLs to build a local repository of reference documents.
"""

import csv
import json
import os
import time

import requests
import trafilatura
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("SERPER_API_KEY")
if not API_KEY:
    raise ValueError("SERPER_API_KEY is missing. Please set it in the .env file.")

CSV_FILE = "masterarbeit_dataset_systematisch.csv"
OUTPUT_DIR = "scraped_data_full"


def search_serper(query, num_results=20):
    """Execute a search query using the Serper API.

    Args:
        query (str): The search string to query.
        num_results (int, optional): The maximum number of search results to return. Defaults to 20.

    Returns:
        dict or None: A dictionary containing the parsed JSON response from the API,
        or None if an exception occurs during the request.
    """
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": num_results})
    headers = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error calling Serper API for query '{query}': {e}")
        return None


def scrape_url(url):
    """Extract readable text content from a given URL.

    Args:
        url (str): The target webpage URL to scrape.

    Returns:
        str or None: The extracted text content, or None if the request times out,
        fails, or text extraction is unsuccessful.
    """
    try:
        # Use requests with a strict timeout first to prevent indefinite hangs
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            },
        )
        if response.status_code == 200:
            downloaded = response.text
            text = trafilatura.extract(downloaded)
            return text
        return None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def main():
    """Coordinate the search and scraping workflow for all dataset prompts.

    Reads prompts from the dataset CSV, retrieves search results, attempts
    to scrape web pages, and stores successfully extracted text locally
    with associated metadata. Continues from previous state if interrupted.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    prompts = []
    # Read all prompts
    with open(CSV_FILE, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for i, row in enumerate(reader):
            if len(row) >= 1:
                prompts.append(row[0].strip('"'))

    print(f"Starting full run for {len(prompts)} prompts...")

    for idx, prompt in enumerate(prompts):
        prompt_id = idx + 1
        prompt_dir = os.path.join(OUTPUT_DIR, f"prompt_{prompt_id:03d}")

        # Check if this prompt was already processed (resume capability)
        metadata_file = os.path.join(prompt_dir, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                try:
                    metadata = json.load(f)
                    if metadata.get("sources_collected") == 5:
                        print(
                            f"\n[{prompt_id}/{len(prompts)}] Skipping Prompt (Already completed): {prompt}"
                        )
                        continue
                except Exception:
                    pass

        if not os.path.exists(prompt_dir):
            os.makedirs(prompt_dir)

        print(f"\n[{prompt_id}/{len(prompts)}] Processing Prompt: {prompt}")

        search_results = search_serper(prompt, num_results=20)
        if not search_results or "organic" not in search_results:
            print("  No organic results found.")
            continue

        usable_sources = []
        metadata_log = []

        for result in search_results["organic"]:
            if len(usable_sources) >= 5:
                break

            url = result.get("link")
            title = result.get("title")
            print(f"  -> Trying URL: {url}")

            content = scrape_url(url)

            if content and len(content) >= 100:
                source_idx = len(usable_sources) + 1
                filename = f"source_{source_idx}.txt"
                filepath = os.path.join(prompt_dir, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

                usable_sources.append(url)
                metadata_log.append(
                    {
                        "source_id": source_idx,
                        "url": url,
                        "title": title,
                        "filename": filename,
                        "character_count": len(content),
                    }
                )
                print(f"     [SUCCESS] Saved as {filename} ({len(content)} chars)")
            else:
                reason = (
                    "Blocked/Failed"
                    if not content
                    else f"Too short ({len(content)} chars)"
                )
                print(f"     [SKIPPED] {reason}")

            time.sleep(1)  # Polite delay between scrapes

        # Save metadata for this prompt
        metadata = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "sources_collected": len(usable_sources),
            "sources": metadata_log,
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        if len(usable_sources) < 5:
            print(
                f"  [WARNING] Only found {len(usable_sources)} usable sources for this prompt."
            )

        time.sleep(1)  # Delay between prompts to respect API rate limits

    print("\nFull run completed. Check the 'scraped_data_full' directory.")


if __name__ == "__main__":
    main()
